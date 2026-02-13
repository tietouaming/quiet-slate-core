"""主耦合求解器模块（中文注释版）。

本模块是项目核心，负责在每个物理时间步内完成：
1. 力学平衡与晶体塑性更新；
2. 腐蚀相场与浓度扩散推进；
3. 孪晶序参数演化；
4. ML surrogate 与物理步之间的门控切换；
5. 历史量记录、快照与可视化输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math
import time
from pathlib import Path
from typing import Dict, List

import torch

from .config import SimulationConfig
from .couplers import build_couplers
from .crystal_plasticity import CrystalPlasticityModel
from .geometry import Grid2D, initial_fields, make_grid, pitting_mobility_field
from .io_utils import ensure_dir, render_fields, render_final_clouds, render_grid_figure, save_history_csv, save_snapshot_npz
from .mechanics import MechanicsModel
from .ml.scaling import build_mechanics_field_scales, build_surrogate_field_scales
from .ml.mech_warmstart import load_mechanics_warmstart
from .ml.surrogate import load_surrogate
from .operators import (
    divergence,
    double_well,
    double_well_prime,
    grad_xy,
    laplacian,
    smooth_heaviside,
    smooth_heaviside_prime,
)


def _torch_dtype(name: str) -> torch.dtype:
    """字符串 dtype 映射到 torch dtype。"""
    if name == "float64":
        return torch.float64
    return torch.float32


def _fmt_wall_time(seconds: float) -> str:
    """将秒数格式化为 `MM:SS.xx` 或 `HH:MM:SS.xx`。"""
    s = max(float(seconds), 0.0)
    h = int(s // 3600.0)
    m = int((s % 3600.0) // 60.0)
    sec = s - 3600.0 * h - 60.0 * m
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:05.2f}"
    return f"{m:02d}:{sec:05.2f}"


@dataclass
class StepDiagnostics:
    """单步诊断量。"""
    step: int
    time_s: float
    solid_fraction: float
    avg_eta: float
    max_sigma_h: float
    avg_epspeq: float
    used_surrogate_only: bool
    loss_phi_mae: float = 0.0
    loss_c_mae: float = 0.0
    loss_eta_mae: float = 0.0
    loss_epspeq_mae: float = 0.0
    free_energy: float = 0.0
    pde_res_phi: float = 0.0
    pde_res_c: float = 0.0
    pde_res_eta: float = 0.0
    pde_res_mech: float = 0.0
    rollout_every: int = 1


class CoupledSimulator:
    """镁合金腐蚀-孪晶-晶体塑性耦合求解器。"""
    def __init__(self, cfg: SimulationConfig):
        """初始化网格、场变量、子模型与可选 surrogate。"""
        self.cfg = cfg
        self.device = self._pick_device(cfg.runtime.device)
        self.dtype = _torch_dtype(cfg.numerics.dtype)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.manual_seed(cfg.numerics.seed)
        self.grid: Grid2D = make_grid(cfg, self.device, self.dtype)
        self.dx_um = self.grid.dx_min_um
        self.dy_um = self.grid.dy_min_um
        self.dx_m = self.dx_um * 1e-6
        self.dy_m = self.dy_um * 1e-6
        self.x_coords_um = self.grid.x_vec_um
        self.y_coords_um = self.grid.y_vec_um
        self.x_coords_m = self.x_coords_um * 1e-6
        self.y_coords_m = self.y_coords_um * 1e-6
        self.area_weights = self._build_area_weights(self.x_coords_um, self.y_coords_um)
        self.diffusion_dt_limit_s = self._estimate_diffusion_dt_limit()
        self._diffusion_stage_cache: Dict[float, List[float]] = {}

        ang_s = torch.deg2rad(torch.tensor(cfg.twinning.twin_shear_dir_angle_deg, device=self.device, dtype=self.dtype))
        ang_n = torch.deg2rad(torch.tensor(cfg.twinning.twin_plane_normal_angle_deg, device=self.device, dtype=self.dtype))
        self.twin_sx = torch.cos(ang_s)
        self.twin_sy = torch.sin(ang_s)
        self.twin_nx = torch.cos(ang_n)
        self.twin_ny = torch.sin(ang_n)

        self.state = initial_fields(cfg, self.grid)
        self.couplers = build_couplers(cfg)
        self.cp = CrystalPlasticityModel(cfg, self.device, self.dtype)
        self.cp_state = self.cp.init_state(cfg.domain.ny, cfg.domain.nx)
        self.mech = MechanicsModel(cfg)
        self.pitting_field = pitting_mobility_field(cfg, self.grid)
        self.surrogate_field_scales = build_surrogate_field_scales(cfg)
        self.mech_input_scales, self.mech_output_scales = build_mechanics_field_scales(cfg)

        z = torch.zeros_like(self.state["phi"])
        # 塑性张量内部表示与 state 通道保持同步，便于 surrogate 与物理步共享。
        self.epsp = {
            "exx": self.state.get("epsp_xx", z).clone(),
            "eyy": self.state.get("epsp_yy", z).clone(),
            "exy": self.state.get("epsp_xy", z).clone(),
        }
        self.state["epsp_xx"] = self.epsp["exx"].clone()
        self.state["epsp_yy"] = self.epsp["eyy"].clone()
        self.state["epsp_xy"] = self.epsp["exy"].clone()
        self.state["epspeq_dot"] = torch.zeros_like(self.state["phi"])
        self.last_mech = {
            "sigma_xx": z.clone(),
            "sigma_yy": z.clone(),
            "sigma_xy": z.clone(),
            "sigma_zz": z.clone(),
            "sigma_h": z.clone(),
        }

        self.surrogate = None
        self.mech_warmstart = None
        if cfg.ml.enabled:
            p = Path(cfg.ml.model_path)
            if p.exists():
                self.surrogate = load_surrogate(
                    p,
                    device=self.device,
                    use_torch_compile=cfg.numerics.use_torch_compile and self.device.type == "cuda",
                    fallback_model_arch=cfg.ml.model_arch,
                    fallback_arch_kwargs={
                        "hidden": cfg.ml.model_hidden,
                        "dw_hidden": cfg.ml.dw_hidden,
                        "dw_depth": cfg.ml.dw_depth,
                        "fno_width": cfg.ml.fno_width,
                        "fno_modes_x": cfg.ml.fno_modes_x,
                        "fno_modes_y": cfg.ml.fno_modes_y,
                        "fno_depth": cfg.ml.fno_depth,
                        "afno_width": cfg.ml.afno_width,
                        "afno_modes_x": cfg.ml.afno_modes_x,
                        "afno_modes_y": cfg.ml.afno_modes_y,
                        "afno_depth": cfg.ml.afno_depth,
                        "afno_expansion": cfg.ml.afno_expansion,
                        "add_coord_features": cfg.ml.surrogate_add_coord_features,
                    },
                    fallback_field_scales=self.surrogate_field_scales,
                )
                self.surrogate.enforce_displacement_constraints = bool(
                    getattr(cfg.ml, "surrogate_enforce_displacement_projection", True)
                )
                self.surrogate.loading_mode = str(cfg.mechanics.loading_mode)
                self.surrogate.dirichlet_right_ux = float(self.mech._dirichlet_right_ux())
                self.surrogate.enforce_uy_anchor = True
                self.surrogate.allow_plastic_outputs = bool(getattr(cfg.ml, "surrogate_update_plastic_fields", False))
            pm = Path(getattr(cfg.ml, "mechanics_warmstart_model_path", ""))
            if bool(getattr(cfg.ml, "mechanics_warmstart_enabled", False)) and pm.exists():
                self.mech_warmstart = load_mechanics_warmstart(
                    pm,
                    device=self.device,
                    fallback_hidden=int(getattr(cfg.ml, "mechanics_warmstart_hidden", 24)),
                    fallback_add_coord_features=bool(getattr(cfg.ml, "mechanics_warmstart_add_coord_features", True)),
                    use_torch_compile=cfg.numerics.use_torch_compile and self.device.type == "cuda",
                    channels_last=self.device.type == "cuda",
                    fallback_input_scales=self.mech_input_scales,
                    fallback_output_scales=self.mech_output_scales,
                )
        self._surrogate_reject_streak = 0
        self._surrogate_pause_until_step = 0
        self._current_rollout_every = max(1, int(cfg.ml.rollout_every))
        self._rollout_success_streak = 0
        self._rollout_reject_streak = 0
        self._last_gate_metrics: Dict[str, float] = {
            "free_energy": 0.0,
            "pde_phi": 0.0,
            "pde_c": 0.0,
            "pde_eta": 0.0,
            "pde_mech": 0.0,
            "uncertainty": 0.0,
            "mass_delta": 0.0,
        }

        self.history: List[Dict[str, float]] = []
        self.stats: Dict[str, int] = {
            "mech_predict_solve": 0,
            "mech_correct_solve": 0,
            "mech_correct_skipped_elastic": 0,
            "mech_cg_converged": 0,
            "mech_cg_failed": 0,
            "mech_cg_iters_sum": 0,
            "mech_ml_initial_guess_used": 0,
            "mech_ml_initial_guess_mech_warmstart": 0,
            "mech_ml_initial_guess_surrogate": 0,
            "ml_surrogate_attempt": 0,
            "ml_surrogate_accept": 0,
            "ml_surrogate_reject": 0,
            "ml_reject_phi_drop": 0,
            "ml_reject_eta_rise": 0,
            "ml_reject_phi_mean_delta": 0,
            "ml_reject_eta_mean_delta": 0,
            "ml_reject_c_mean_delta": 0,
            "ml_reject_eps_mean_delta": 0,
            "ml_reject_phi_field_delta": 0,
            "ml_reject_eta_field_delta": 0,
            "ml_reject_c_field_delta": 0,
            "ml_reject_mass": 0,
            "ml_reject_liquid_eta": 0,
            "ml_reject_liquid_eps": 0,
            "ml_reject_residual_gate": 0,
            "ml_reject_pde_phi": 0,
            "ml_reject_pde_c": 0,
            "ml_reject_pde_eta": 0,
            "ml_reject_pde_mech": 0,
            "ml_reject_energy": 0,
            "ml_energy_gate_skipped_nonvariational": 0,
            "ml_reject_uncertainty": 0,
            "ml_rollout_increase": 0,
            "ml_rollout_decrease": 0,
            "scalar_pcg_calls": 0,
            "scalar_pcg_converged": 0,
            "scalar_bicgstab_calls": 0,
            "scalar_bicgstab_converged": 0,
            "scalar_solver_fallback": 0,
            "mass_projection_applied": 0,
        }
        # 力学自适应触发参考态：记录上次力学求解对应的微结构。
        self._last_mech_phi = self.state["phi"].clone()
        self._last_mech_eta = self.state["eta"].clone()

    def _build_area_weights(self, x_coords_um: torch.Tensor, y_coords_um: torch.Tensor) -> torch.Tensor:
        """构造面积加权矩阵，用于非均匀网格上的加权平均。"""
        def nodal_weights(v: torch.Tensor) -> torch.Tensor:
            if v.numel() < 2:
                return torch.ones_like(v)
            dv = v[1:] - v[:-1]
            w = torch.zeros_like(v)
            w[0] = 0.5 * dv[0]
            w[-1] = 0.5 * dv[-1]
            if v.numel() > 2:
                w[1:-1] = 0.5 * (dv[1:] + dv[:-1])
            return torch.clamp(w, min=1e-12)

        wx = nodal_weights(x_coords_um)
        wy = nodal_weights(y_coords_um)
        w2 = wy[:, None] * wx[None, :]
        w2 = w2 / torch.clamp(torch.sum(w2), min=1e-12)
        return w2[None, None]

    def _mean_field(self, x: torch.Tensor) -> torch.Tensor:
        """计算全域面积加权平均。"""
        return torch.sum(x * self.area_weights)

    @staticmethod
    def _twin_fraction(eta: torch.Tensor) -> torch.Tensor:
        """孪晶体积分数：f_twin = h(eta)。"""
        return smooth_heaviside(torch.clamp(eta, 0.0, 1.0))

    @staticmethod
    def _solid_weight(phi: torch.Tensor) -> torch.Tensor:
        """固相连续权重：w_solid = h(phi)。"""
        return smooth_heaviside(torch.clamp(phi, 0.0, 1.0))

    def _project_concentration_mean(
        self,
        c: torch.Tensor,
        target_mean: float,
        *,
        cmin: float | None = None,
        cmax: float | None = None,
        n_iter: int = 2,
    ) -> torch.Tensor:
        """将浓度场均值投影到目标值（带上下界裁剪）。"""
        lo = self.cfg.corrosion.cMg_min if cmin is None else float(cmin)
        hi = self.cfg.corrosion.cMg_max if cmax is None else float(cmax)
        out = torch.clamp(torch.nan_to_num(c, nan=lo, posinf=hi, neginf=lo), min=lo, max=hi)
        tgt = float(target_mean)
        for _ in range(max(1, int(n_iter))):
            cur = float(self._mean_field(out).item())
            delta = tgt - cur
            if abs(delta) <= 1e-14:
                break
            out = torch.clamp(out + delta, min=lo, max=hi)
        return out

    @staticmethod
    def _soft_project_range(x: torch.Tensor, lo: float, hi: float, beta: float = 20.0) -> torch.Tensor:
        """可微范围投影：区间内近似恒等、区间外平滑回推。"""
        if hi <= lo:
            return torch.full_like(x, float(lo))
        y = (x - float(lo)) / float(hi - lo)
        y = y - torch.nn.functional.softplus(y - 1.0, beta=beta) / beta + torch.nn.functional.softplus(-y, beta=beta) / beta
        return float(lo) + (float(hi - lo)) * y

    def _masked_mean_abs(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        """计算带掩膜的面积加权绝对值均值。"""
        ax = torch.abs(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        if mask is None:
            return float(self._mean_field(ax).item())
        m = torch.clamp(torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
        num = self._mean_field(ax * m)
        den = torch.clamp(self._mean_field(m), min=1e-12)
        return float((num / den).item())

    def _epsp_from_state(self, state: Dict[str, torch.Tensor] | None = None) -> Dict[str, torch.Tensor]:
        """从 state 字典读取塑性张量；缺失时回退到内部缓存。"""
        st = self.state if state is None else state
        return {
            "exx": st.get("epsp_xx", self.epsp["exx"]),
            "eyy": st.get("epsp_yy", self.epsp["eyy"]),
            "exy": st.get("epsp_xy", self.epsp["exy"]),
        }

    def _sync_epsp_to_state(self) -> None:
        """将内部塑性张量同步到主 state。"""
        self.state["epsp_xx"] = self.epsp["exx"]
        self.state["epsp_yy"] = self.epsp["eyy"]
        self.state["epsp_xy"] = self.epsp["exy"]

    def _estimate_state_mechanics(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """根据给定位移场快速估计应变/应力（不做平衡迭代）。"""
        epsp_state = self._epsp_from_state(state)
        eps = self.mech._strain_from_displacement(
            state["ux"],
            state["uy"],
            state["eta"],
            epsp_state,
            self.dx_um,
            self.dy_um,
            x_coords_um=self.x_coords_um,
            y_coords_um=self.y_coords_um,
        )
        sig = self.mech.constitutive_stress(
            state["phi"],
            eps["eps_xx"],
            eps["eps_yy"],
            eps["eps_xy"],
            eta=state.get("eta"),
        )
        out = {
            "ux": torch.nan_to_num(state["ux"], nan=0.0, posinf=0.0, neginf=0.0),
            "uy": torch.nan_to_num(state["uy"], nan=0.0, posinf=0.0, neginf=0.0),
            "eps_xx": torch.nan_to_num(eps["eps_xx"], nan=0.0, posinf=0.0, neginf=0.0),
            "eps_yy": torch.nan_to_num(eps["eps_yy"], nan=0.0, posinf=0.0, neginf=0.0),
            "eps_xy": torch.nan_to_num(eps["eps_xy"], nan=0.0, posinf=0.0, neginf=0.0),
            "sigma_xx": torch.nan_to_num(sig["sigma_xx"], nan=0.0, posinf=0.0, neginf=0.0),
            "sigma_yy": torch.nan_to_num(sig["sigma_yy"], nan=0.0, posinf=0.0, neginf=0.0),
            "sigma_xy": torch.nan_to_num(sig["sigma_xy"], nan=0.0, posinf=0.0, neginf=0.0),
            "sigma_zz": torch.nan_to_num(sig["sigma_zz"], nan=0.0, posinf=0.0, neginf=0.0),
            "sigma_h": torch.nan_to_num(sig["sigma_h"], nan=0.0, posinf=0.0, neginf=0.0),
        }
        return out

    def _compute_mechanics_residual(self, mech_state: Dict[str, torch.Tensor], phi: torch.Tensor) -> float:
        """计算 `div(sigma)=0` 的离散残差均值（仅固相统计）。"""
        res_x, res_y = self.mech.stress_divergence(
            mech_state["sigma_xx"],
            mech_state["sigma_yy"],
            mech_state["sigma_xy"],
            self.dx_um,
            self.dy_um,
            x_coords_um=self.x_coords_um,
            y_coords_um=self.y_coords_um,
        )
        solid = self._solid_weight(phi)
        return 0.5 * (
            self._masked_mean_abs(res_x, solid)
            + self._masked_mean_abs(res_y, solid)
        )

    def _compute_free_energy(
        self,
        state: Dict[str, torch.Tensor],
        mech_state: Dict[str, torch.Tensor],
        *,
        for_gate: bool = False,
    ) -> float:
        """计算离散自由能（面积加权平均）。

        - `for_gate=True`：返回 E_gate（仅包含与当前变分驱动一致的核心项）；
        - `for_gate=False`：返回 E_diag（用于诊断展示，可含弹性能等扩展项）。
        """
        bc = self._scalar_bc()
        phi = torch.clamp(state["phi"], 0.0, 1.0)
        cbar = torch.clamp(state["c"], self.cfg.corrosion.cMg_min, self.cfg.corrosion.cMg_max)
        eta = torch.clamp(state["eta"], 0.0, 1.0)
        hphi = smooth_heaviside(phi)
        cl_eq = self.cfg.corrosion.c_l_eq_norm
        cs_eq = self.cfg.corrosion.c_s_eq_norm
        delta_eq = cs_eq - cl_eq
        A_mpa = self.cfg.corrosion.A_J_m3 / 1e6
        omega, kappa_phi = self._omega_kappa_phi()
        chem = 0.5 * A_mpa * (cbar - hphi * delta_eq - cl_eq) ** 2
        phi_dw = omega * double_well(phi)
        gx_phi, gy_phi = grad_xy(
            phi,
            self.dx_m,
            self.dy_m,
            x_coords=self.x_coords_m,
            y_coords=self.y_coords_m,
            bc=bc,
        )
        grad_phi = 0.5 * kappa_phi * (gx_phi * gx_phi + gy_phi * gy_phi)
        eta_barrier = hphi * self.cfg.twinning.W_barrier_MPa * eta * eta * (1.0 - eta) * (1.0 - eta)
        gx_eta, gy_eta = grad_xy(
            eta,
            self.dx_um,
            self.dy_um,
            x_coords=self.x_coords_um,
            y_coords=self.y_coords_um,
            bc=bc,
        )
        if self.cfg.twinning.scale_twin_gradient_by_hphi:
            grad_coef_eta = self.cfg.twinning.kappa_eta * hphi
        else:
            grad_coef_eta = torch.full_like(phi, float(self.cfg.twinning.kappa_eta))
        grad_eta = 0.5 * grad_coef_eta * (gx_eta * gx_eta + gy_eta * gy_eta)
        eps_xx = mech_state["eps_xx"]
        eps_yy = mech_state["eps_yy"]
        eps_xy = mech_state["eps_xy"]
        sigma_xx = mech_state["sigma_xx"]
        sigma_yy = mech_state["sigma_yy"]
        sigma_xy = mech_state["sigma_xy"]
        elastic = 0.5 * (sigma_xx * eps_xx + sigma_yy * eps_yy + 2.0 * sigma_xy * eps_xy)
        mech_phi_coupling = torch.zeros_like(phi)
        if bool(getattr(self.cfg.corrosion, "include_mech_term_in_free_energy", False)):
            # 诊断项：-h(phi)*e_mech。仅在配置开启时计入。
            mech_phi_coupling = -hphi * self._mech_phi_coupling_energy_density(mech_state)
        # E_gate：严格对应变分核心项；机械-phi 项仅在“方程+自由能”双开启时计入。
        gate_total = chem + phi_dw + grad_phi + eta_barrier + grad_eta
        if bool(self.cfg.corrosion.include_mech_term_in_phi_variation) and bool(
            getattr(self.cfg.corrosion, "include_mech_term_in_free_energy", False)
        ):
            gate_total = gate_total + mech_phi_coupling
        # E_diag：用于日志与可视化，不参与严格门控判据。
        diag_total = gate_total + elastic
        if bool(getattr(self.cfg.corrosion, "include_mech_term_in_free_energy", False)) and not bool(
            self.cfg.corrosion.include_mech_term_in_phi_variation
        ):
            # 当用户仅想诊断机械耦合项时，允许其仅出现在 E_diag。
            diag_total = diag_total + mech_phi_coupling
        total = gate_total if for_gate else diag_total
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        return float(self._mean_field(total).item())

    def _compute_pde_residuals(
        self,
        prev: Dict[str, torch.Tensor],
        nxt: Dict[str, torch.Tensor],
        mech_state: Dict[str, torch.Tensor],
        dt_s: float,
    ) -> Dict[str, float]:
        """计算 surrogate 候选状态的离散 PDE 残差。"""
        bc = self._scalar_bc()
        dt = max(float(dt_s), 1e-12)
        phi = torch.nan_to_num(nxt["phi"], nan=0.5, posinf=1.0, neginf=0.0)
        cbar = torch.nan_to_num(nxt["c"], nan=0.5, posinf=self.cfg.corrosion.cMg_max, neginf=self.cfg.corrosion.cMg_min)
        eta = torch.nan_to_num(nxt["eta"], nan=0.0, posinf=1.0, neginf=0.0)
        hphi = smooth_heaviside(phi, clamp_input=False)
        hphi_d = smooth_heaviside_prime(phi, clamp_input=False)
        cl_eq = self.cfg.corrosion.c_l_eq_norm
        cs_eq = self.cfg.corrosion.c_s_eq_norm
        delta_eq = cs_eq - cl_eq
        omega, kappa_phi = self._omega_kappa_phi()

        chem_drive = -self.cfg.corrosion.A_J_m3 / 1e6 * (cbar - hphi * delta_eq - cl_eq) * delta_eq * hphi_d
        phi_nonlin = chem_drive + omega * double_well_prime(phi, clamp_input=False)
        if self.cfg.corrosion.include_mech_term_in_phi_variation:
            e_mech = self._mech_phi_coupling_energy_density(mech_state)
            phi_nonlin = phi_nonlin - hphi_d * e_mech
        if self.cfg.corrosion.include_twin_grad_term_in_phi_variation and self.cfg.twinning.scale_twin_gradient_by_hphi:
            # 与 eta-TDGL 保持一致：kappa_eta 与 eta 梯度统一采用 um 尺度。
            gx_eta_um, gy_eta_um = grad_xy(
                eta,
                self.dx_um,
                self.dy_um,
                x_coords=self.x_coords_um,
                y_coords=self.y_coords_um,
                bc=bc,
            )
            twin_grad_e = 0.5 * self.cfg.twinning.kappa_eta * (gx_eta_um * gx_eta_um + gy_eta_um * gy_eta_um)
            # 变分一致性：F 中为 +0.5*kappa_eta*h(phi)|grad eta|^2，则 dF/dphi 为 +h'(phi)*twin_grad_e。
            phi_nonlin = phi_nonlin + hphi_d * twin_grad_e
        L_phi = torch.clamp(self._corrosion_mobility(state=nxt, mech=mech_state), min=0.0)
        phi_lap_kappa = self._div_diffusive_fv(
            phi,
            torch.full_like(phi, float(kappa_phi)),
            dx=self.dx_m,
            dy=self.dy_m,
            x_coords=self.x_coords_m,
            y_coords=self.y_coords_m,
            bc=bc,
        )
        r_phi = (phi - prev["phi"]) / dt + L_phi * (phi_nonlin - phi_lap_kappa)
        pde_phi = self._masked_mean_abs(r_phi)

        hphi_c = smooth_heaviside(phi, clamp_input=False)
        hphi_d_c = smooth_heaviside_prime(phi, clamp_input=False)
        D = self.cfg.corrosion.D_s_m2_s * hphi_c + (1.0 - hphi_c) * self.cfg.corrosion.D_l_m2_s
        gx_phi, gy_phi = grad_xy(
            phi,
            self.dx_m,
            self.dy_m,
            x_coords=self.x_coords_m,
            y_coords=self.y_coords_m,
            bc=bc,
        )
        corr = hphi_d_c * (cl_eq - cs_eq)
        rhs_c = self._diffusion_rhs(
            cbar,
            D,
            corr,
            gx_phi,
            gy_phi,
            phi,
            state_for_coupler=nxt,
            mech_for_coupler=mech_state,
        )
        r_c = (cbar - prev["c"]) / dt - rhs_c
        pde_c_raw = self._masked_mean_abs(r_c)

        heta_d = smooth_heaviside_prime(eta, clamp_input=False)
        if self.cfg.twinning.scale_twin_gradient_by_hphi:
            grad_coef_eta = self.cfg.twinning.kappa_eta * hphi
        else:
            grad_coef_eta = torch.full_like(hphi, float(self.cfg.twinning.kappa_eta))
        twin_dw = hphi * 2.0 * self.cfg.twinning.W_barrier_MPa * eta * (1.0 - eta) * (1.0 - 2.0 * eta)
        tau_tw = self._resolved_twin_shear(mech_state["sigma_xx"], mech_state["sigma_yy"], mech_state["sigma_xy"])
        tw_drive = hphi * tau_tw * self.cfg.twinning.gamma_twin * heta_d
        r2 = (self.grid.x_um - self.cfg.domain.notch_tip_x_um) ** 2 + (self.grid.y_um - self.cfg.domain.notch_center_y_um) ** 2
        nuc = torch.exp(-r2 / max(self.cfg.twinning.nucleation_center_radius_um ** 2, 1e-12))
        act = self._twin_nucleation_activation(tau_tw)
        nuc_source = hphi * self._twin_nucleation_source_amp() * nuc * act
        eta_source = twin_dw - tw_drive - nuc_source
        grad_term = self._div_diffusive_fv(
            eta,
            torch.clamp(grad_coef_eta, min=0.0),
            dx=self.dx_um,
            dy=self.dy_um,
            x_coords=self.x_coords_um,
            y_coords=self.y_coords_um,
            bc=bc,
        )
        r_eta = (eta - prev["eta"]) / dt + self.cfg.twinning.L_eta * (eta_source - grad_term)
        solid = self._solid_weight(phi)
        pde_eta_raw = self._masked_mean_abs(r_eta, solid)

        pde_mech_raw = self._compute_mechanics_residual(mech_state, phi)
        # 无量纲残差：便于统一门控阈值。
        phi_scale = max(1.0 / dt, 1e-12)
        eta_scale = max(1.0 / dt, 1e-12)
        c_scale = max((self.cfg.corrosion.cMg_max - self.cfg.corrosion.cMg_min) / dt, 1e-12)
        l_ref = max(min(float(self.dx_um), float(self.dy_um)), 1e-9)
        sigma_ref = max(float(self.cfg.mechanics.mu_GPa * 1e3), 1.0)
        mech_scale = max(sigma_ref / l_ref, 1e-12)
        pde_phi = float(pde_phi / phi_scale)
        pde_c = float(pde_c_raw / c_scale)
        pde_eta = float(pde_eta_raw / eta_scale)
        pde_mech = float(pde_mech_raw / mech_scale)
        return {
            "pde_phi": pde_phi,
            "pde_c": pde_c,
            "pde_eta": pde_eta,
            "pde_mech": pde_mech,
        }

    def _estimate_surrogate_uncertainty(self, prev: Dict[str, torch.Tensor]) -> float:
        """通过输入微扰多次推理估计 surrogate 局部不确定性。"""
        if self.surrogate is None or (not bool(self.cfg.ml.enable_uncertainty_gate)):
            return 0.0
        n = max(2, int(self.cfg.ml.uncertainty_samples))
        std = max(float(self.cfg.ml.uncertainty_jitter_std), 0.0)
        if std <= 0.0:
            return 0.0
        vals: List[float] = []
        cspan = max(self.cfg.corrosion.cMg_max - self.cfg.corrosion.cMg_min, 1e-12)
        for _ in range(n):
            pert: Dict[str, torch.Tensor] = {}
            for k, v in prev.items():
                noise = torch.randn_like(v) * std
                vv = v + noise
                if k == "phi":
                    vv = self._soft_project_range(vv, 0.0, 1.0)
                elif k == "c":
                    vv = self._soft_project_range(vv, self.cfg.corrosion.cMg_min, self.cfg.corrosion.cMg_max)
                elif k == "eta":
                    vv = self._soft_project_range(vv, 0.0, 1.0)
                elif k == "epspeq":
                    vv = torch.clamp(vv, 0.0, 1e6)
                elif k in {"epsp_xx", "epsp_yy", "epsp_xy"}:
                    vv = torch.clamp(vv, -1.0, 1.0)
                pert[k] = vv
            pred = self.surrogate.predict(pert)
            dphi = self._masked_mean_abs(pred["phi"] - prev["phi"])
            dc = self._masked_mean_abs(pred["c"] - prev["c"]) / cspan
            deta = self._masked_mean_abs(self._twin_fraction(pred["eta"]) - self._twin_fraction(prev["eta"]))
            vals.append(dphi + dc + deta)
        vt = torch.tensor(vals, device=self.device, dtype=self.dtype)
        return float(torch.std(vt, unbiased=False).item())

    def _on_surrogate_accept(self) -> None:
        """更新 surrogate 接受后的自适应 rollout 状态。"""
        self._rollout_reject_streak = 0
        self._rollout_success_streak += 1
        if not bool(self.cfg.ml.adaptive_rollout):
            return
        need = max(1, int(self.cfg.ml.rollout_success_streak_to_increase))
        if self._rollout_success_streak < need:
            return
        rmax = max(1, int(self.cfg.ml.rollout_max_every))
        if self._current_rollout_every < rmax:
            self._current_rollout_every += 1
            self.stats["ml_rollout_increase"] += 1
        self._rollout_success_streak = 0

    def _on_surrogate_reject(self) -> None:
        """更新 surrogate 拒绝后的自适应 rollout 状态。"""
        self._rollout_success_streak = 0
        self._rollout_reject_streak += 1
        if not bool(self.cfg.ml.adaptive_rollout):
            return
        need = max(1, int(self.cfg.ml.rollout_reject_streak_to_decrease))
        if self._rollout_reject_streak < need:
            return
        rmin = max(1, int(self.cfg.ml.rollout_min_every))
        if self._current_rollout_every > rmin:
            self._current_rollout_every -= 1
            self.stats["ml_rollout_decrease"] += 1
        self._rollout_reject_streak = 0

    def _sync_cp_state_on_surrogate_accept(self, dt_s: float, mech_for_cp: Dict[str, torch.Tensor] | None = None) -> None:
        """在 surrogate 接受后同步 CP 内变量，避免 epsp 与硬化历史脱节。"""
        if not bool(getattr(self.cfg.ml, "surrogate_update_plastic_fields", False)):
            return
        if not bool(getattr(self.cfg.ml, "surrogate_sync_cp_state_on_accept", True)):
            return
        mech_cp = mech_for_cp if mech_for_cp is not None else self._estimate_state_mechanics(self.state)
        cp_out = self.cp.update(
            cp_state=self.cp_state,
            sigma_xx=mech_cp["sigma_xx"],
            sigma_yy=mech_cp["sigma_yy"],
            sigma_xy=mech_cp["sigma_xy"],
            eta=self.state["eta"],
            dt_s=dt_s,
            material_overrides=self._material_overrides(),
        )
        self.cp_state["g"] = cp_out["g"]
        self.cp_state["gamma_accum"] = cp_out["gamma_accum"]
        self.state["epspeq_dot"] = torch.clamp(
            torch.nan_to_num(cp_out["epspeq_dot"], nan=0.0, posinf=0.0, neginf=0.0),
            min=0.0,
            max=1e6,
        )
        # surrogate 已经给出塑性场时，不在此重复积分 epsp_dot，仅同步内部缓存。
        self.epsp = self._epsp_from_state(self.state)
        self._sync_epsp_to_state()

    def _estimate_diffusion_dt_limit(self) -> float:
        """估计显式扩散稳定步长上限。"""
        dmax = max(float(self.cfg.corrosion.D_l_m2_s), float(self.cfg.corrosion.D_s_m2_s), 1e-20)
        dx = max(float(self.dx_m), 1e-20)
        dy = max(float(self.dy_m), 1e-20)
        return 1.0 / (2.0 * dmax * ((1.0 / (dx * dx)) + (1.0 / (dy * dy))))

    def _diffusion_substeps(self, dt_s: float) -> int:
        """根据稳定性约束估计扩散子步数。"""
        if not self.cfg.numerics.diffusion_auto_substeps:
            return 1
        safety = max(min(float(self.cfg.numerics.diffusion_dt_safety), 1.0), 1e-6)
        dt_safe = max(self.diffusion_dt_limit_s * safety, 1e-16)
        n = int(math.ceil(dt_s / dt_safe))
        return max(1, min(n, int(max(self.cfg.numerics.diffusion_substeps_cap, 1))))

    def _diffusion_stage_dts(self, dt_s: float) -> List[float]:
        """为扩散方程生成子步时间序列（普通子循环或 STS）。"""
        key = round(float(dt_s), 15)
        cached = self._diffusion_stage_cache.get(key)
        if cached is not None:
            return cached

        mode = str(self.cfg.numerics.diffusion_integrator).strip().lower()
        if mode != "sts":
            n_sub = self._diffusion_substeps(dt_s)
            out = [float(dt_s) / float(n_sub)] * n_sub
            self._diffusion_stage_cache[key] = out
            return out

        safety = max(min(float(self.cfg.numerics.diffusion_dt_safety), 1.0), 1e-6)
        dt_fe = max(self.diffusion_dt_limit_s * safety, 1e-16)
        if dt_s <= dt_fe:
            out = [float(dt_s)]
            self._diffusion_stage_cache[key] = out
            return out

        nu = min(max(float(self.cfg.numerics.diffusion_sts_nu), 1e-4), 0.5)
        m_max = max(2, int(self.cfg.numerics.diffusion_sts_max_stages))
        best: List[float] | None = None
        for m in range(2, m_max + 1):
            raw: List[float] = []
            total = 0.0
            for j in range(1, m + 1):
                theta = (2.0 * j - 1.0) * math.pi / (2.0 * m)
                denom = (1.0 + nu) + (nu - 1.0) * math.cos(theta)
                denom = max(denom, 1e-9)
                dt_j = dt_fe / denom
                raw.append(dt_j)
                total += dt_j
            if total >= dt_s:
                scale = float(dt_s) / max(total, 1e-16)
                best = [scale * v for v in raw]
                break

        if best is None:
            # Fallback to classic subcycling when requested super-step exceeds configured stage cap.
            n_sub = self._diffusion_substeps(dt_s)
            best = [float(dt_s) / float(n_sub)] * n_sub

        self._diffusion_stage_cache[key] = best
        return best

    def _scalar_bc(self) -> str:
        """返回标量场离散边界条件。"""
        return str(getattr(self.cfg.numerics, "scalar_bc", "neumann")).strip().lower()

    def _axis_metrics(
        self,
        n: int,
        d_default: float,
        coords: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构造面间距与控制体宽度。"""
        if n <= 1:
            face = torch.full((1,), max(float(d_default), 1e-12), device=device, dtype=dtype)
            cell = torch.full((1,), max(float(d_default), 1e-12), device=device, dtype=dtype)
            return face, cell
        if coords is None:
            d = max(float(d_default), 1e-12)
            face = torch.full((n - 1,), d, device=device, dtype=dtype)
            cell = torch.full((n,), d, device=device, dtype=dtype)
            return face, cell
        face = torch.clamp(coords[1:] - coords[:-1], min=1e-12).to(device=device, dtype=dtype)
        cell = torch.zeros((n,), device=device, dtype=dtype)
        cell[0] = 0.5 * face[0]
        cell[-1] = 0.5 * face[-1]
        if n > 2:
            cell[1:-1] = 0.5 * (face[1:] + face[:-1])
        return face, cell

    def _div_diffusive_fv(
        self,
        u: torch.Tensor,
        diff_coef: torch.Tensor,
        *,
        dx: float,
        dy: float,
        x_coords: torch.Tensor | None,
        y_coords: torch.Tensor | None,
        bc: str | None = None,
    ) -> torch.Tensor:
        """有限体积守恒离散：返回 `div(diff_coef * grad(u))`。"""
        b = self._scalar_bc() if bc is None else str(bc).strip().lower()
        eps = 1e-12
        B, C, H, W = u.shape
        if H < 2 or W < 2:
            return torch.zeros_like(u)

        # 周期边界在非均匀网格下未实现，避免误用。
        if b in {"periodic", "circular"}:
            if x_coords is not None and x_coords.numel() > 2:
                dxv = x_coords[1:] - x_coords[:-1]
                if float(torch.max(torch.abs(dxv - dxv[0])).item()) > 1e-12:
                    raise ValueError("periodic scalar_bc is only supported on uniform x grid.")
            if y_coords is not None and y_coords.numel() > 2:
                dyv = y_coords[1:] - y_coords[:-1]
                if float(torch.max(torch.abs(dyv - dyv[0])).item()) > 1e-12:
                    raise ValueError("periodic scalar_bc is only supported on uniform y grid.")

        fx, cx = self._axis_metrics(W, dx, x_coords, dtype=u.dtype, device=u.device)
        fy, cy = self._axis_metrics(H, dy, y_coords, dtype=u.dtype, device=u.device)
        fx = fx.view(1, 1, 1, -1)
        cx = cx.view(1, 1, 1, -1)
        fy = fy.view(1, 1, -1, 1)
        cy = cy.view(1, 1, -1, 1)

        d = torch.clamp(diff_coef, min=0.0)
        if b in {"periodic", "circular"}:
            d_xf = 2.0 * d * torch.roll(d, shifts=-1, dims=3) / torch.clamp(d + torch.roll(d, shifts=-1, dims=3), min=eps)
            d_yf = 2.0 * d * torch.roll(d, shifts=-1, dims=2) / torch.clamp(d + torch.roll(d, shifts=-1, dims=2), min=eps)
            flux_x_out = -d_xf * (torch.roll(u, shifts=-1, dims=3) - u) / max(float(dx), eps)
            flux_y_out = -d_yf * (torch.roll(u, shifts=-1, dims=2) - u) / max(float(dy), eps)
            div_x = (flux_x_out - torch.roll(flux_x_out, shifts=1, dims=3)) / cx
            div_y = (flux_y_out - torch.roll(flux_y_out, shifts=1, dims=2)) / cy
            return torch.nan_to_num(div_x + div_y, nan=0.0, posinf=0.0, neginf=0.0)

        # Neumann/Dirichlet0: 先组装内部面通量，再做控制体通量差。
        d_xf = 2.0 * d[:, :, :, 1:] * d[:, :, :, :-1] / torch.clamp(d[:, :, :, 1:] + d[:, :, :, :-1], min=eps)
        d_yf = 2.0 * d[:, :, 1:, :] * d[:, :, :-1, :] / torch.clamp(d[:, :, 1:, :] + d[:, :, :-1, :], min=eps)
        flux_x = -d_xf * (u[:, :, :, 1:] - u[:, :, :, :-1]) / fx
        flux_y = -d_yf * (u[:, :, 1:, :] - u[:, :, :-1, :]) / fy

        div_x = torch.zeros_like(u)
        div_y = torch.zeros_like(u)
        if b in {"dirichlet0", "dirichlet", "fixed0"}:
            # 边界面值固定为 0 时，边界通量按单边差商计算。
            d_l = d[:, :, :, 0]
            d_r = d[:, :, :, -1]
            fx0 = cx[:, :, :, 0]
            fxn = cx[:, :, :, -1]
            flux_left = -d_l * (u[:, :, :, 0] - 0.0) / torch.clamp(fx0, min=eps)
            flux_right = -d_r * (0.0 - u[:, :, :, -1]) / torch.clamp(fxn, min=eps)
            div_x[:, :, :, 0] = (flux_x[:, :, :, 0] - flux_left) / torch.clamp(cx[:, :, :, 0], min=eps)
            div_x[:, :, :, -1] = (flux_right - flux_x[:, :, :, -1]) / torch.clamp(cx[:, :, :, -1], min=eps)
        else:
            # Neumann 零通量：边界外侧面通量为 0。
            div_x[:, :, :, 0] = flux_x[:, :, :, 0] / torch.clamp(cx[:, :, :, 0], min=eps)
            div_x[:, :, :, -1] = -flux_x[:, :, :, -1] / torch.clamp(cx[:, :, :, -1], min=eps)
        if W > 2:
            div_x[:, :, :, 1:-1] = (flux_x[:, :, :, 1:] - flux_x[:, :, :, :-1]) / torch.clamp(cx[:, :, :, 1:-1], min=eps)

        if b in {"dirichlet0", "dirichlet", "fixed0"}:
            d_b = d[:, :, 0, :]
            d_t = d[:, :, -1, :]
            fy0 = cy[:, :, 0, :]
            fyn = cy[:, :, -1, :]
            flux_bottom = -d_b * (u[:, :, 0, :] - 0.0) / torch.clamp(fy0, min=eps)
            flux_top = -d_t * (0.0 - u[:, :, -1, :]) / torch.clamp(fyn, min=eps)
            div_y[:, :, 0, :] = (flux_y[:, :, 0, :] - flux_bottom) / torch.clamp(cy[:, :, 0, :], min=eps)
            div_y[:, :, -1, :] = (flux_top - flux_y[:, :, -1, :]) / torch.clamp(cy[:, :, -1, :], min=eps)
        else:
            div_y[:, :, 0, :] = flux_y[:, :, 0, :] / torch.clamp(cy[:, :, 0, :], min=eps)
            div_y[:, :, -1, :] = -flux_y[:, :, -1, :] / torch.clamp(cy[:, :, -1, :], min=eps)
        if H > 2:
            div_y[:, :, 1:-1, :] = (flux_y[:, :, 1:, :] - flux_y[:, :, :-1, :]) / torch.clamp(cy[:, :, 1:-1, :], min=eps)

        return torch.nan_to_num(div_x + div_y, nan=0.0, posinf=0.0, neginf=0.0)

    def _diffusion_rhs(
        self,
        c_field: torch.Tensor,
        D: torch.Tensor,
        corr: torch.Tensor,
        gx_phi: torch.Tensor,
        gy_phi: torch.Tensor,
        phi_for_state: torch.Tensor,
        state_for_coupler: Dict[str, torch.Tensor] | None = None,
        mech_for_coupler: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """扩散方程右端项。"""
        bc = self._scalar_bc()
        use_mu_form = bool(getattr(self.cfg.corrosion, "concentration_use_mu_form", True))
        if use_mu_form:
            # 与化学自由能一致的写法：
            # mu = A*(c - h(phi)*(cs-cl) - cl),  c_t = div(M*grad(mu)), M = D/A。
            hphi = smooth_heaviside(phi_for_state)
            cl_eq = float(self.cfg.corrosion.c_l_eq_norm)
            cs_eq = float(self.cfg.corrosion.c_s_eq_norm)
            delta_eq = cs_eq - cl_eq
            A_mpa = max(float(self.cfg.corrosion.A_J_m3) / 1e6, 1e-12)
            mu = A_mpa * (c_field - hphi * delta_eq - cl_eq)
            M = torch.clamp(D / A_mpa, min=0.0)
            diff_term = self._div_diffusive_fv(
                mu,
                M,
                dx=self.dx_m,
                dy=self.dy_m,
                x_coords=self.x_coords_m,
                y_coords=self.y_coords_m,
                bc=bc,
            )
            jx = torch.zeros_like(c_field)
            jy = torch.zeros_like(c_field)
        else:
            # 旧形式（保留兼容）：div(D*grad(c)) - div(-D*corr*grad(phi))。
            diff_term = self._div_diffusive_fv(
                c_field,
                D,
                dx=self.dx_m,
                dy=self.dy_m,
                x_coords=self.x_coords_m,
                y_coords=self.y_coords_m,
                bc=bc,
            )
            jx = -D * (corr * gx_phi)
            jy = -D * (corr * gy_phi)
        if self.couplers:
            state_tmp = dict(self.state if state_for_coupler is None else state_for_coupler)
            state_tmp["phi"] = phi_for_state
            state_tmp["c"] = c_field
            aux = self.last_mech if mech_for_coupler is None else mech_for_coupler
            for cp in self.couplers:
                ex, ey = cp.concentration_drift_flux(state_tmp, aux)
                jx = jx + ex
                jy = jy + ey
        rhs = diff_term - divergence(
            jx,
            jy,
            self.dx_m,
            self.dy_m,
            x_coords=self.x_coords_m,
            y_coords=self.y_coords_m,
            bc=bc,
        )
        return torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

    def _pcg_scalar_with_info(
        self,
        apply_A,
        rhs: torch.Tensor,
        *,
        x0: torch.Tensor | None = None,
        diag_inv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool, int, float]:
        """通用标量场 PCG 迭代器（矩阵自由，含收敛信息）。"""
        if x0 is None:
            x = rhs.clone()
        else:
            x = x0.clone()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        rhs = torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
        r = rhs - apply_A(x)
        z = r if diag_inv is None else diag_inv * r
        p = z.clone()
        rz_old = torch.sum(r * z)
        r0 = torch.sqrt(torch.clamp(torch.sum(r * r), min=0.0))
        abs_tol = max(float(self.cfg.numerics.imex_solver_tol_abs), 0.0)
        rel_tol = max(float(self.cfg.numerics.imex_solver_tol_rel), 0.0)
        tol = max(abs_tol, rel_tol * float(r0.item()))
        relax = max(min(float(self.cfg.numerics.imex_relaxation), 1.0), 0.1)
        if float(r0.item()) <= tol:
            return x, True, 0, float(r0.item())

        n_iter = max(1, int(self.cfg.numerics.imex_solver_iters))
        conv = False
        it_used = 0
        r_norm = float(r0.item())
        for it in range(1, n_iter + 1):
            Ap = apply_A(p)
            denom = torch.sum(p * Ap)
            if torch.abs(denom).item() <= 1e-30:
                break
            alpha = rz_old / denom
            x = x + relax * alpha * p
            r = r - relax * alpha * Ap
            r_norm_t = torch.sqrt(torch.clamp(torch.sum(r * r), min=0.0))
            r_norm = float(r_norm_t.item())
            it_used = it
            if r_norm <= tol:
                conv = True
                break
            z = r if diag_inv is None else diag_inv * r
            rz_new = torch.sum(r * z)
            if torch.abs(rz_old).item() <= 1e-30:
                break
            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), conv, int(it_used), float(r_norm)

    def _pcg_scalar(
        self,
        apply_A,
        rhs: torch.Tensor,
        *,
        x0: torch.Tensor | None = None,
        diag_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """兼容旧调用接口：仅返回解。"""
        x, _, _, _ = self._pcg_scalar_with_info(apply_A, rhs, x0=x0, diag_inv=diag_inv)
        return x

    def _bicgstab_scalar_with_info(
        self,
        apply_A,
        rhs: torch.Tensor,
        *,
        x0: torch.Tensor | None = None,
        diag_inv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, bool, int, float]:
        """通用标量场 BiCGStab（矩阵自由，适配非SPD场景）。"""
        if x0 is None:
            x = rhs.clone()
        else:
            x = x0.clone()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        rhs = torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

        r = rhs - apply_A(x)
        r_hat = r.clone()
        r0 = torch.sqrt(torch.clamp(torch.sum(r * r), min=0.0))
        abs_tol = max(float(self.cfg.numerics.imex_solver_tol_abs), 0.0)
        rel_tol = max(float(self.cfg.numerics.imex_solver_tol_rel), 0.0)
        tol = max(abs_tol, rel_tol * float(r0.item()))
        if float(r0.item()) <= tol:
            return x, True, 0, float(r0.item())

        p = torch.zeros_like(r)
        v = torch.zeros_like(r)
        rho_old = torch.tensor(1.0, device=rhs.device, dtype=rhs.dtype)
        alpha = torch.tensor(1.0, device=rhs.device, dtype=rhs.dtype)
        omega = torch.tensor(1.0, device=rhs.device, dtype=rhs.dtype)
        n_iter = max(1, int(self.cfg.numerics.imex_solver_iters))
        conv = False
        it_used = 0
        r_norm = float(r0.item())

        for it in range(1, n_iter + 1):
            rho_new = torch.sum(r_hat * r)
            if torch.abs(rho_new).item() <= 1e-30:
                break
            beta = (rho_new / rho_old) * (alpha / torch.clamp(omega, min=1e-30))
            p = r + beta * (p - omega * v)
            p_hat = p if diag_inv is None else diag_inv * p
            v = apply_A(p_hat)
            den = torch.sum(r_hat * v)
            if torch.abs(den).item() <= 1e-30:
                break
            alpha = rho_new / den
            s = r - alpha * v
            s_norm_t = torch.sqrt(torch.clamp(torch.sum(s * s), min=0.0))
            s_norm = float(s_norm_t.item())
            if s_norm <= tol:
                x = x + alpha * p_hat
                conv = True
                it_used = it
                r_norm = s_norm
                break
            s_hat = s if diag_inv is None else diag_inv * s
            t = apply_A(s_hat)
            tt = torch.sum(t * t)
            if torch.abs(tt).item() <= 1e-30:
                break
            omega = torch.sum(t * s) / tt
            x = x + alpha * p_hat + omega * s_hat
            r = s - omega * t
            r_norm_t = torch.sqrt(torch.clamp(torch.sum(r * r), min=0.0))
            r_norm = float(r_norm_t.item())
            it_used = it
            if r_norm <= tol:
                conv = True
                break
            if torch.abs(omega).item() <= 1e-30:
                break
            rho_old = rho_new

        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), conv, int(it_used), float(r_norm)

    def _solve_scalar_linear(
        self,
        apply_A,
        rhs: torch.Tensor,
        *,
        x0: torch.Tensor | None = None,
        diag_inv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """统一标量线性求解入口，支持 auto/pcg/bicgstab。"""
        mode = str(getattr(self.cfg.numerics, "scalar_linear_solver", "auto")).strip().lower()
        if mode in {"pcg"}:
            self.stats["scalar_pcg_calls"] += 1
            x, conv, _, _ = self._pcg_scalar_with_info(apply_A, rhs, x0=x0, diag_inv=diag_inv)
            if conv:
                self.stats["scalar_pcg_converged"] += 1
            return x
        if mode in {"bicgstab", "bicg"}:
            self.stats["scalar_bicgstab_calls"] += 1
            x, conv, _, _ = self._bicgstab_scalar_with_info(apply_A, rhs, x0=x0, diag_inv=diag_inv)
            if conv:
                self.stats["scalar_bicgstab_converged"] += 1
            return x

        # auto: 优先 PCG，失败后回退 BiCGStab。
        self.stats["scalar_pcg_calls"] += 1
        x_pcg, conv_pcg, _, _ = self._pcg_scalar_with_info(apply_A, rhs, x0=x0, diag_inv=diag_inv)
        if conv_pcg:
            self.stats["scalar_pcg_converged"] += 1
            return x_pcg
        self.stats["scalar_solver_fallback"] += 1
        self.stats["scalar_bicgstab_calls"] += 1
        x_bi, conv_bi, _, _ = self._bicgstab_scalar_with_info(apply_A, rhs, x0=x_pcg, diag_inv=diag_inv)
        if conv_bi:
            self.stats["scalar_bicgstab_converged"] += 1
        return x_bi

    def _solve_helmholtz_imex(
        self,
        rhs: torch.Tensor,
        *,
        alpha: float,
        dx: float,
        dy: float,
        x_coords: torch.Tensor | None,
        y_coords: torch.Tensor | None,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """求解 `(I - alpha*Lap)u = rhs`。"""
        a = max(float(alpha), 0.0)
        if a <= 0.0:
            return rhs
        bc = self._scalar_bc()

        inv_diag = 1.0 / (1.0 + a * (2.0 / max(dx * dx, 1e-20) + 2.0 / max(dy * dy, 1e-20)))
        diag_inv = torch.full_like(rhs, fill_value=float(inv_diag))

        def apply_A(u: torch.Tensor) -> torch.Tensor:
            return u - a * laplacian(
                u,
                dx,
                dy,
                x_coords=x_coords,
                y_coords=y_coords,
                bc=bc,
            )

        return self._solve_scalar_linear(apply_A, rhs, x0=x0, diag_inv=diag_inv)

    def _solve_variable_diffusion_imex(
        self,
        rhs: torch.Tensor,
        *,
        coeff: float,
        diff_coef: torch.Tensor,
        dx: float,
        dy: float,
        x_coords: torch.Tensor | None,
        y_coords: torch.Tensor | None,
        x0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """求解 `(I - coeff*div(diff_coef*grad))u = rhs`。"""
        c = max(float(coeff), 0.0)
        if c <= 0.0:
            return rhs
        bc = self._scalar_bc()
        if x_coords is not None and x_coords.numel() > 1:
            dx_eff = float(torch.min(x_coords[1:] - x_coords[:-1]).item())
        else:
            dx_eff = float(dx)
        if y_coords is not None and y_coords.numel() > 1:
            dy_eff = float(torch.min(y_coords[1:] - y_coords[:-1]).item())
        else:
            dy_eff = float(dy)
        diag_lap = 2.0 / max(dx_eff * dx_eff, 1e-20) + 2.0 / max(dy_eff * dy_eff, 1e-20)
        diag_inv = 1.0 / (1.0 + c * torch.clamp(diff_coef, min=0.0) * diag_lap)
        diag_inv = torch.clamp(torch.nan_to_num(diag_inv, nan=1.0, posinf=1.0, neginf=1.0), min=1e-8, max=1e8)

        def apply_A(u: torch.Tensor) -> torch.Tensor:
            return u - c * self._div_diffusive_fv(
                u,
                diff_coef,
                dx=dx,
                dy=dy,
                x_coords=x_coords,
                y_coords=y_coords,
                bc=bc,
            )

        return self._solve_scalar_linear(apply_A, rhs, x0=x0, diag_inv=diag_inv)

    def _pick_device(self, mode: str) -> torch.device:
        """解析设备选择策略。"""
        if mode == "cpu":
            return torch.device("cpu")
        if mode == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _omega_kappa_phi(self) -> tuple[float, float]:
        """由界面能和厚度换算相场参数 omega 与 kappa。"""
        gamma = self.cfg.corrosion.gamma_J_m2
        ell_um = float(self.cfg.corrosion.interface_thickness_um)
        if ell_um <= 0.0:
            ell_um = float(self.cfg.domain.interface_width_um)
        ell_m = ell_um * 1e-6
        omega = 3.0 * gamma / (4.0 * ell_m) / 1e6  # MPa
        kappa = 1.5 * gamma * ell_m / 1e6  # MPa*m^2
        return omega, kappa

    def _material_overrides(self) -> Dict[str, torch.Tensor]:
        """汇总扩展耦合器对材料参数的覆盖项。"""
        out: Dict[str, torch.Tensor] = {}
        for c in self.couplers:
            upd = c.update_material_overrides(self.state, self.last_mech)
            for k, v in upd.items():
                out[k] = v if k not in out else out[k] * v
        return out

    def _corrosion_mobility(
        self,
        state: Dict[str, torch.Tensor] | None = None,
        mech: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """计算腐蚀迁移率（含力学与扩展耦合影响）。"""
        st = self.state if state is None else state
        mech_state = self.last_mech if mech is None else mech
        c = self.cfg.corrosion
        L0 = float(c.L0)
        l0_unit = str(getattr(c, "L0_unit", "1_over_MPa_s")).strip().lower()
        if l0_unit in {"1_over_pa_s", "1/pa/s", "pa"}:
            # 代码中化学势/能量以 MPa 为标度，若输入 SI(1/Pa/s) 需转为 1/(MPa*s)。
            L0 = L0 * 1e6
        sigma_h_pa = mech_state["sigma_h"] * 1e6
        # 机械放大项默认采用“塑性应变率”而非“累计塑性应变”，避免历史量永久放大腐蚀速度。
        if bool(getattr(c, "use_epspeq_rate_for_mobility", True)):
            epsdot = torch.clamp(
                torch.nan_to_num(st.get("epspeq_dot", torch.zeros_like(st["phi"])), nan=0.0, posinf=0.0, neginf=0.0),
                min=0.0,
            )
            epsdot_ref = max(float(getattr(c, "epspeq_dot_ref_s_inv", 1e-3)), 1e-12)
            k_epsdot = max(float(getattr(c, "k_epsdot", 1.0)), 0.0)
            strain_fac = 1.0 + k_epsdot * torch.tanh(epsdot / epsdot_ref)
        else:
            strain_fac = st["epspeq"] / max(c.yield_strain_for_mech, 1e-12) + 1.0
        stress_fac = torch.exp(
            torch.clamp(sigma_h_pa * c.molar_volume_m3_mol / (c.gas_constant_J_mol_K * c.temperature_K), -12.0, 12.0)
        )
        mech_fac = strain_fac * stress_fac
        out = L0 * self.pitting_field * mech_fac
        for cp in self.couplers:
            out = out * cp.corrosion_mobility_multiplier(st, mech_state)
        return out

    def _mech_phi_coupling_energy_density(self, mech_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """与自由能一致的弹性能密度 e_mech = 0.5*sigma:epsilon。"""
        return 0.5 * (
            mech_state["sigma_xx"] * mech_state["eps_xx"]
            + mech_state["sigma_yy"] * mech_state["eps_yy"]
            + 2.0 * mech_state["sigma_xy"] * mech_state["eps_xy"]
        )

    def _select_phi_imex_l0(self, L_phi_pos: torch.Tensor) -> float:
        """选择 phi-IMEX 的常数 L0（用于常系数隐式+显式补偿分裂）。"""
        strategy = str(getattr(self.cfg.numerics, "phi_imex_l0_strategy", "max")).strip().lower()
        safety = max(float(getattr(self.cfg.numerics, "phi_imex_l0_safety", 1.0)), 1e-8)
        if strategy in {"max", "upper"}:
            base = float(torch.max(L_phi_pos).item())
        elif strategy in {"p95", "q95", "quantile"}:
            q = float(getattr(self.cfg.numerics, "phi_imex_l0_quantile", 0.95))
            q = min(max(q, 0.0), 1.0)
            base = float(torch.quantile(L_phi_pos.reshape(-1), q).item())
        elif strategy in {"mean", "avg", "average"}:
            base = float(torch.mean(L_phi_pos).item())
        else:
            base = float(torch.max(L_phi_pos).item())
        return max(base * safety, 0.0)

    def _resolved_twin_shear(self, sigma_xx: torch.Tensor, sigma_yy: torch.Tensor, sigma_xy: torch.Tensor) -> torch.Tensor:
        """计算孪晶系分解剪应力。"""
        tau = self.twin_sx * (sigma_xx * self.twin_nx + sigma_xy * self.twin_ny) + self.twin_sy * (
            sigma_xy * self.twin_nx + sigma_yy * self.twin_ny
        )
        return tau

    def _twin_nucleation_activation(self, tau_tw: torch.Tensor) -> torch.Tensor:
        """孪晶形核激活：仅对有利剪切方向（tau_tw > twin_crss）开启。"""
        return torch.relu(tau_tw - float(self.cfg.twinning.twin_crss_MPa))

    def _twin_nucleation_source_amp(self) -> float:
        """获取形核源项幅值（兼容旧键名 langevin_nucleation_noise）。"""
        amp_new = float(getattr(self.cfg.twinning, "nucleation_source_amp", 0.0))
        if amp_new > 0.0:
            return amp_new
        return float(getattr(self.cfg.twinning, "langevin_nucleation_noise", 0.0))

    def _sanitize_state(self) -> None:
        """对状态做数值清洗与物理约束裁剪。"""
        cmin = self.cfg.corrosion.cMg_min
        cmax = self.cfg.corrosion.cMg_max
        self.state["phi"] = self._soft_project_range(
            torch.nan_to_num(self.state["phi"], nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        solid = self._solid_weight(self.state["phi"])
        self.state["c"] = self._soft_project_range(
            torch.nan_to_num(self.state["c"], nan=cmin, posinf=cmax, neginf=cmin),
            cmin,
            cmax,
        )
        self.state["eta"] = self._soft_project_range(
            torch.nan_to_num(self.state["eta"], nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        self.state["eta"] = self.state["eta"] * solid
        self.state["epspeq"] = torch.clamp(
            torch.nan_to_num(self.state["epspeq"], nan=0.0, posinf=1e6, neginf=0.0),
            0.0,
            1e6,
        )
        self.state["epspeq"] = self.state["epspeq"] * solid
        self.state["epspeq_dot"] = torch.clamp(
            torch.nan_to_num(self.state.get("epspeq_dot", torch.zeros_like(self.state["phi"])), nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            1e6,
        ) * solid
        self.state["epsp_xx"] = torch.clamp(
            torch.nan_to_num(self.state.get("epsp_xx", self.epsp["exx"]), nan=0.0, posinf=0.0, neginf=0.0),
            min=-1.0,
            max=1.0,
        ) * solid
        self.state["epsp_yy"] = torch.clamp(
            torch.nan_to_num(self.state.get("epsp_yy", self.epsp["eyy"]), nan=0.0, posinf=0.0, neginf=0.0),
            min=-1.0,
            max=1.0,
        ) * solid
        self.state["epsp_xy"] = torch.clamp(
            torch.nan_to_num(self.state.get("epsp_xy", self.epsp["exy"]), nan=0.0, posinf=0.0, neginf=0.0),
            min=-1.0,
            max=1.0,
        ) * solid
        self.epsp["exx"] = self.state["epsp_xx"]
        self.epsp["eyy"] = self.state["epsp_yy"]
        self.epsp["exy"] = self.state["epsp_xy"]
        self.last_mech = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in self.last_mech.items()}
        for k in ("sigma_xx", "sigma_yy", "sigma_xy", "sigma_zz", "sigma_h"):
            if k in self.last_mech:
                self.last_mech[k] = self.last_mech[k] * solid

    def _surrogate_update_is_valid(
        self,
        prev: Dict[str, torch.Tensor],
        nxt: Dict[str, torch.Tensor],
        *,
        candidate_mech: Dict[str, torch.Tensor] | None = None,
    ) -> bool:
        """判断 surrogate 提议步是否通过门控。"""
        def reject(reason: str) -> bool:
            key = f"ml_reject_{reason}"
            self.stats[key] = self.stats.get(key, 0) + 1
            return False

        self._last_gate_metrics["pde_phi"] = 0.0
        self._last_gate_metrics["pde_c"] = 0.0
        self._last_gate_metrics["pde_eta"] = 0.0
        self._last_gate_metrics["pde_mech"] = 0.0
        self._last_gate_metrics["uncertainty"] = 0.0
        self._last_gate_metrics["mass_delta"] = 0.0

        max_phi_drop = self.cfg.ml.max_mean_phi_drop
        if max_phi_drop > 0.0:
            phi_drop = float(self._mean_field(prev["phi"] - nxt["phi"]).item())
            if phi_drop > max_phi_drop:
                return reject("phi_drop")
        max_eta_rise = self.cfg.ml.max_mean_eta_rise
        if max_eta_rise > 0.0:
            eta_rise = float(self._mean_field(self._twin_fraction(nxt["eta"]) - self._twin_fraction(prev["eta"])).item())
            if eta_rise > max_eta_rise:
                return reject("eta_rise")
        phi_abs_delta = self.cfg.ml.max_mean_phi_abs_delta
        if phi_abs_delta > 0.0:
            if abs(float(self._mean_field(nxt["phi"] - prev["phi"]).item())) > phi_abs_delta:
                return reject("phi_mean_delta")
        eta_abs_delta = self.cfg.ml.max_mean_eta_abs_delta
        if eta_abs_delta > 0.0:
            if abs(float(self._mean_field(self._twin_fraction(nxt["eta"]) - self._twin_fraction(prev["eta"])).item())) > eta_abs_delta:
                return reject("eta_mean_delta")
        c_abs_delta = self.cfg.ml.max_mean_c_abs_delta
        if c_abs_delta > 0.0:
            if abs(float(self._mean_field(nxt["c"] - prev["c"]).item())) > c_abs_delta:
                return reject("c_mean_delta")
        if bool(getattr(self.cfg.ml, "enable_mass_gate", False)):
            m_prev = float(self._mean_field(prev["c"]).item())
            m_nxt = float(self._mean_field(nxt["c"]).item())
            m_abs = abs(m_nxt - m_prev)
            self._last_gate_metrics["mass_delta"] = float(m_abs)
            abs_tol = max(float(getattr(self.cfg.ml, "mass_abs_delta_max", 0.0)), 0.0)
            rel_tol = max(float(getattr(self.cfg.ml, "mass_rel_delta_max", 0.0)), 0.0)
            allow = abs_tol + rel_tol * max(abs(m_prev), 1e-12)
            if m_abs > allow:
                return reject("mass")
        eps_abs_delta = self.cfg.ml.max_mean_epspeq_abs_delta
        if eps_abs_delta > 0.0:
            if abs(float(self._mean_field(nxt["epspeq"] - prev["epspeq"]).item())) > eps_abs_delta:
                return reject("eps_mean_delta")
            z_eps = torch.zeros_like(prev["epspeq"])
            prev_epsp_xx = prev.get("epsp_xx", z_eps)
            prev_epsp_yy = prev.get("epsp_yy", z_eps)
            prev_epsp_xy = prev.get("epsp_xy", z_eps)
            nxt_epsp_xx = nxt.get("epsp_xx", prev_epsp_xx)
            nxt_epsp_yy = nxt.get("epsp_yy", prev_epsp_yy)
            nxt_epsp_xy = nxt.get("epsp_xy", prev_epsp_xy)
            if abs(float(self._mean_field(nxt_epsp_xx - prev_epsp_xx).item())) > eps_abs_delta:
                return reject("eps_mean_delta")
            if abs(float(self._mean_field(nxt_epsp_yy - prev_epsp_yy).item())) > eps_abs_delta:
                return reject("eps_mean_delta")
            if abs(float(self._mean_field(nxt_epsp_xy - prev_epsp_xy).item())) > eps_abs_delta:
                return reject("eps_mean_delta")

        if self.cfg.ml.max_field_delta > 0.0:
            gate = self.cfg.ml.max_field_delta
            tol = 1e-6
            if torch.max(torch.abs(nxt["phi"] - prev["phi"])).item() > gate + tol:
                return reject("phi_field_delta")
            if torch.max(torch.abs(self._twin_fraction(nxt["eta"]) - self._twin_fraction(prev["eta"]))).item() > gate + tol:
                return reject("eta_field_delta")
            if torch.max(torch.abs(nxt["c"] - prev["c"])).item() > gate * 1.5 + tol:
                return reject("c_field_delta")
        dphi = float(self._mean_field(torch.abs(nxt["phi"] - prev["phi"])).item())
        cspan = max(self.cfg.corrosion.cMg_max - self.cfg.corrosion.cMg_min, 1e-12)
        dc = float(self._mean_field(torch.abs(nxt["c"] - prev["c"])).item()) / cspan
        deta = float(self._mean_field(torch.abs(self._twin_fraction(nxt["eta"]) - self._twin_fraction(prev["eta"]))).item())
        solid_nxt = self._solid_weight(nxt["phi"])
        liquid = torch.clamp(1.0 - solid_nxt, min=0.0, max=1.0)
        liquid_eta = float(self._mean_field(torch.abs(self._twin_fraction(nxt["eta"]) * liquid)).item())
        if liquid_eta > max(self.cfg.ml.max_mean_eta_abs_delta, 1e-6):
            return reject("liquid_eta")
        liquid_eps = float(self._mean_field(torch.abs(nxt["epspeq"] * liquid)).item())
        if liquid_eps > max(self.cfg.ml.max_mean_epspeq_abs_delta, 1e-6):
            return reject("liquid_eps")
        liquid_eps_xx = float(self._mean_field(torch.abs(nxt.get("epsp_xx", torch.zeros_like(liquid)) * liquid)).item())
        liquid_eps_yy = float(self._mean_field(torch.abs(nxt.get("epsp_yy", torch.zeros_like(liquid)) * liquid)).item())
        liquid_eps_xy = float(self._mean_field(torch.abs(nxt.get("epsp_xy", torch.zeros_like(liquid)) * liquid)).item())
        if max(liquid_eps_xx, liquid_eps_yy, liquid_eps_xy) > max(self.cfg.ml.max_mean_epspeq_abs_delta, 1e-6):
            return reject("liquid_eps")
        rel = dphi + dc + deta
        if rel > self.cfg.ml.residual_gate:
            return reject("residual_gate")

        if bool(self.cfg.ml.enable_uncertainty_gate):
            unc = self._estimate_surrogate_uncertainty(prev)
            self._last_gate_metrics["uncertainty"] = float(unc)
            if unc > max(float(self.cfg.ml.uncertainty_gate), 0.0):
                return reject("uncertainty")

        mech_prev = self._estimate_state_mechanics(prev)
        mech_nxt = candidate_mech if candidate_mech is not None else self._estimate_state_mechanics(nxt)

        if bool(self.cfg.ml.enable_pde_residual_gate):
            residuals = self._compute_pde_residuals(prev, nxt, mech_nxt, self.cfg.numerics.dt_s)
            self._last_gate_metrics["pde_phi"] = residuals["pde_phi"]
            self._last_gate_metrics["pde_c"] = residuals["pde_c"]
            self._last_gate_metrics["pde_eta"] = residuals["pde_eta"]
            self._last_gate_metrics["pde_mech"] = residuals["pde_mech"]
            if (
                self.cfg.ml.pde_residual_phi_abs_max > 0.0
                and residuals["pde_phi"] > float(self.cfg.ml.pde_residual_phi_abs_max)
            ):
                return reject("pde_phi")
            if (
                self.cfg.ml.pde_residual_c_abs_max > 0.0
                and residuals["pde_c"] > float(self.cfg.ml.pde_residual_c_abs_max)
            ):
                return reject("pde_c")
            if (
                self.cfg.ml.pde_residual_eta_abs_max > 0.0
                and residuals["pde_eta"] > float(self.cfg.ml.pde_residual_eta_abs_max)
            ):
                return reject("pde_eta")
            if (
                self.cfg.ml.pde_residual_mech_abs_max > 0.0
                and residuals["pde_mech"] > float(self.cfg.ml.pde_residual_mech_abs_max)
            ):
                return reject("pde_mech")

        energy_gate_enabled = bool(self.cfg.ml.enable_energy_gate)
        use_variational_gate_energy = bool(getattr(self.cfg.ml, "energy_gate_use_variational_core", True))
        if (
            energy_gate_enabled
            and (not use_variational_gate_energy)
            and bool(self.cfg.corrosion.include_mech_term_in_phi_variation)
            and (not bool(getattr(self.cfg.corrosion, "include_mech_term_in_free_energy", False)))
        ):
            # 仅当用户要求以 E_diag 做门控且其与方程不一致时，自动跳过。
            energy_gate_enabled = False
            self.stats["ml_energy_gate_skipped_nonvariational"] += 1

        if energy_gate_enabled:
            e_prev = self._compute_free_energy(prev, mech_prev, for_gate=use_variational_gate_energy)
            e_nxt = self._compute_free_energy(nxt, mech_nxt, for_gate=use_variational_gate_energy)
            self._last_gate_metrics["free_energy"] = float(e_nxt)
            dE = float(e_nxt - e_prev)
            dE_abs_max = max(float(self.cfg.ml.energy_abs_increase_max), 0.0)
            dE_rel_max = max(float(self.cfg.ml.energy_rel_increase_max), 0.0)
            dE_allow = dE_abs_max + dE_rel_max * max(abs(e_prev), 1.0)
            if dE > dE_allow:
                return reject("energy")
        else:
            self._last_gate_metrics["free_energy"] = float(
                self._compute_free_energy(nxt, mech_nxt, for_gate=use_variational_gate_energy)
            )
        return True

    def _limit_surrogate_update(self, prev: Dict[str, torch.Tensor], nxt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """限制 surrogate 增量幅度，抑制单步跃迁。"""
        gate = max(self.cfg.ml.max_field_delta, 1e-6)
        relax = max(min(float(self.cfg.ml.surrogate_delta_scale), 1.0), 0.0)
        out = {k: v.clone() for k, v in prev.items()}

        def limit_field(name: str, scale: float) -> None:
            d_raw = (nxt[name] - prev[name]) * relax
            d = torch.clamp(d_raw, min=-gate * scale, max=gate * scale)
            out[name] = prev[name] + d

        limit_field("phi", 1.0)
        limit_field("eta", 1.0)
        limit_field("c", 1.5)
        update_mech = bool(getattr(self.cfg.ml, "surrogate_update_mechanics_fields", True))
        update_plastic = bool(getattr(self.cfg.ml, "surrogate_update_plastic_fields", False))
        if update_mech:
            limit_field("ux", 0.5)
            limit_field("uy", 0.5)
            if update_plastic:
                limit_field("epspeq", 0.2)
                if "epsp_xx" in prev and "epsp_xx" in nxt:
                    limit_field("epsp_xx", 0.2)
                if "epsp_yy" in prev and "epsp_yy" in nxt:
                    limit_field("epsp_yy", 0.2)
                if "epsp_xy" in prev and "epsp_xy" in nxt:
                    limit_field("epsp_xy", 0.2)
            else:
                out["epspeq"] = prev["epspeq"]
                if "epsp_xx" in prev:
                    out["epsp_xx"] = prev["epsp_xx"]
                if "epsp_yy" in prev:
                    out["epsp_yy"] = prev["epsp_yy"]
                if "epsp_xy" in prev:
                    out["epsp_xy"] = prev["epsp_xy"]
        else:
            out["ux"] = prev["ux"]
            out["uy"] = prev["uy"]
            out["epspeq"] = prev["epspeq"]
            if "epsp_xx" in prev:
                out["epsp_xx"] = prev["epsp_xx"]
            if "epsp_yy" in prev:
                out["epsp_yy"] = prev["epsp_yy"]
            if "epsp_xy" in prev:
                out["epsp_xy"] = prev["epsp_xy"]
        out["phi"] = self._soft_project_range(torch.nan_to_num(out["phi"], nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        out["c"] = self._soft_project_range(
            torch.nan_to_num(out["c"], nan=self.cfg.corrosion.cMg_min, posinf=self.cfg.corrosion.cMg_max, neginf=self.cfg.corrosion.cMg_min),
            self.cfg.corrosion.cMg_min,
            self.cfg.corrosion.cMg_max,
        )
        out["eta"] = self._soft_project_range(torch.nan_to_num(out["eta"], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        out["epspeq"] = torch.clamp(out["epspeq"], 0.0, 1e6)
        if "epsp_xx" in out:
            out["epsp_xx"] = torch.clamp(out["epsp_xx"], -1.0, 1.0)
        if "epsp_yy" in out:
            out["epsp_yy"] = torch.clamp(out["epsp_yy"], -1.0, 1.0)
        if "epsp_xy" in out:
            out["epsp_xy"] = torch.clamp(out["epsp_xy"], -1.0, 1.0)
        return out

    def _physics_step(self, dt_s: float, step_idx: int) -> None:
        """执行纯物理时间步推进。"""
        # 1) 力学 + 晶体塑性（可多速率更新）
        mech_every = max(1, int(self.cfg.numerics.mechanics_update_every))
        periodic_due = step_idx <= 1 or (step_idx % mech_every == 0)
        # 自适应触发：当微结构变化超阈值时，提前刷新力学场，避免滞后应力驱动误判。
        phi_trig = max(float(getattr(self.cfg.numerics, "mechanics_trigger_phi_max_delta", 0.0)), 0.0)
        eta_trig = max(float(getattr(self.cfg.numerics, "mechanics_trigger_eta_max_delta", 0.0)), 0.0)
        adaptive_due = False
        if not periodic_due:
            if phi_trig > 0.0:
                dphi = float(torch.max(torch.abs(self.state["phi"] - self._last_mech_phi)).item())
                adaptive_due = adaptive_due or (dphi > phi_trig)
            if eta_trig > 0.0:
                deta = float(torch.max(torch.abs(self.state["eta"] - self._last_mech_eta)).item())
                adaptive_due = adaptive_due or (deta > eta_trig)
        # 仅在指定频率点或触发阈值满足时更新力学。
        do_mech_update = periodic_due or adaptive_due
        if do_mech_update:
            if bool(self.cfg.numerics.mechanics_use_ml_initial_guess):
                # 优先使用“力学专用初值器”；若不存在则回退到通用 surrogate。
                if self.mech_warmstart is not None:
                    ux0, uy0 = self.mech_warmstart.predict(
                        self.state,
                        loading_mode=self.cfg.mechanics.loading_mode,
                        right_disp_um=float(self.mech._dirichlet_right_ux()),
                        enforce_anchor=True,
                    )
                    self.state["ux"] = ux0
                    self.state["uy"] = uy0
                    self.stats["mech_ml_initial_guess_used"] += 1
                    self.stats["mech_ml_initial_guess_mech_warmstart"] += 1
                elif self.surrogate is not None:
                    guess = self.surrogate.predict(self.state)
                    self.state["ux"] = torch.nan_to_num(guess["ux"], nan=0.0, posinf=0.0, neginf=0.0)
                    self.state["uy"] = torch.nan_to_num(guess["uy"], nan=0.0, posinf=0.0, neginf=0.0)
                    self.stats["mech_ml_initial_guess_used"] += 1
                    self.stats["mech_ml_initial_guess_surrogate"] += 1
            # 1.1 预测力学平衡：基于当前位移/本征应变求解应力。
            mech_predict = self.mech.solve_quasi_static(
                self.state,
                self.epsp,
                self.dx_um,
                self.dy_um,
                x_coords_um=self.x_coords_um,
                y_coords_um=self.y_coords_um,
            )
            cg_attempted = bool(float(mech_predict.get("mech_cg_attempted", torch.tensor(0.0)).item()) > 0.5)
            if cg_attempted:
                cg_ok = bool(float(mech_predict.get("mech_cg_converged", torch.tensor(0.0)).item()) > 0.5)
                cg_it = int(round(float(mech_predict.get("mech_cg_iters", torch.tensor(0.0)).item())))
                if cg_ok:
                    self.stats["mech_cg_converged"] += 1
                else:
                    self.stats["mech_cg_failed"] += 1
                self.stats["mech_cg_iters_sum"] += max(cg_it, 0)
            self.stats["mech_predict_solve"] += 1
            self.last_mech = mech_predict
            # 1.2 用预测应力驱动晶体塑性更新（滑移、硬化、等效塑性应变）。
            cp_out = self.cp.update(
                cp_state=self.cp_state,
                sigma_xx=self.last_mech["sigma_xx"],
                sigma_yy=self.last_mech["sigma_yy"],
                sigma_xy=self.last_mech["sigma_xy"],
                eta=self.state["eta"],
                dt_s=dt_s,
                material_overrides=self._material_overrides(),
            )
            self.cp_state["g"] = cp_out["g"]
            self.cp_state["gamma_accum"] = cp_out["gamma_accum"]
            # 1.3 将“应变率”积分到“应变增量”（显式欧拉）。
            self.epsp["exx"] = self.epsp["exx"] + dt_s * cp_out["epsp_dot_xx"]
            self.epsp["eyy"] = self.epsp["eyy"] + dt_s * cp_out["epsp_dot_yy"]
            self.epsp["exy"] = self.epsp["exy"] + dt_s * cp_out["epsp_dot_xy"]
            self.state["epspeq"] = self.state["epspeq"] + dt_s * cp_out["epspeq_dot"]
            self.state["epspeq_dot"] = torch.clamp(
                torch.nan_to_num(cp_out["epspeq_dot"], nan=0.0, posinf=0.0, neginf=0.0),
                min=0.0,
                max=1e6,
            )
            self.epsp["exx"] = torch.clamp(torch.nan_to_num(self.epsp["exx"], nan=0.0, posinf=0.0, neginf=0.0), min=-1.0, max=1.0)
            self.epsp["eyy"] = torch.clamp(torch.nan_to_num(self.epsp["eyy"], nan=0.0, posinf=0.0, neginf=0.0), min=-1.0, max=1.0)
            self.epsp["exy"] = torch.clamp(torch.nan_to_num(self.epsp["exy"], nan=0.0, posinf=0.0, neginf=0.0), min=-1.0, max=1.0)
            self.state["epspeq"] = torch.clamp(
                torch.nan_to_num(self.state["epspeq"], nan=0.0, posinf=1e6, neginf=0.0),
                min=0.0,
                max=1e6,
            )

            if bool(cp_out.get("plastic_active", True)):
                # 1.4 若塑性激活，执行力学校正步，减小“先塑性后应力”的不一致。
                self.last_mech = self.mech.solve_quasi_static(
                    self.state,
                    self.epsp,
                    self.dx_um,
                    self.dy_um,
                    x_coords_um=self.x_coords_um,
                    y_coords_um=self.y_coords_um,
                )
                cg_attempted = bool(float(self.last_mech.get("mech_cg_attempted", torch.tensor(0.0)).item()) > 0.5)
                if cg_attempted:
                    cg_ok = bool(float(self.last_mech.get("mech_cg_converged", torch.tensor(0.0)).item()) > 0.5)
                    cg_it = int(round(float(self.last_mech.get("mech_cg_iters", torch.tensor(0.0)).item())))
                    if cg_ok:
                        self.stats["mech_cg_converged"] += 1
                    else:
                        self.stats["mech_cg_failed"] += 1
                    self.stats["mech_cg_iters_sum"] += max(cg_it, 0)
                self.stats["mech_correct_solve"] += 1
            else:
                # 若本步纯弹性，直接复用预测力学结果。
                self.last_mech = mech_predict
                self.stats["mech_correct_skipped_elastic"] += 1
            self._last_mech_phi = self.state["phi"].clone()
            self._last_mech_eta = self.state["eta"].clone()
        else:
            # 非力学更新步视为准静态保持，塑性应变率设为 0，避免历史量持续驱动腐蚀。
            self.state["epspeq_dot"] = torch.zeros_like(self.state["phi"])

        # 无论力学本步是否更新，都保持 state 与内部塑性张量同步。
        self._sync_epsp_to_state()

        # 2) 腐蚀 + 扩散 + 孪晶演化
        phi = self.state["phi"]
        cbar = self.state["c"]
        eta = self.state["eta"]
        bc = self._scalar_bc()
        hphi = smooth_heaviside(phi)
        hphi_d = smooth_heaviside_prime(phi)

        cl_eq = self.cfg.corrosion.c_l_eq_norm
        cs_eq = self.cfg.corrosion.c_s_eq_norm
        delta_eq = cs_eq - cl_eq

        omega, kappa_phi = self._omega_kappa_phi()
        # 相场驱动力中的非线性部分：化学势差项 + 双稳势项。
        chem_drive = -self.cfg.corrosion.A_J_m3 / 1e6 * (cbar - hphi * delta_eq - cl_eq) * delta_eq * hphi_d
        phi_nonlin = chem_drive + omega * double_well_prime(phi)

        if self.cfg.corrosion.include_mech_term_in_phi_variation:
            # Optional "full variational" route.
            e_mech = self._mech_phi_coupling_energy_density(self.last_mech)
            phi_nonlin = phi_nonlin - hphi_d * e_mech

        if self.cfg.corrosion.include_twin_grad_term_in_phi_variation and self.cfg.twinning.scale_twin_gradient_by_hphi:
            # 与 eta 方程使用相同的长度单位，避免 kappa_eta 量纲失配。
            gx_eta_um, gy_eta_um = grad_xy(
                eta,
                self.dx_um,
                self.dy_um,
                x_coords=self.x_coords_um,
                y_coords=self.y_coords_um,
                bc=bc,
            )
            twin_grad_e = 0.5 * self.cfg.twinning.kappa_eta * (gx_eta_um * gx_eta_um + gy_eta_um * gy_eta_um)
            phi_nonlin = phi_nonlin + hphi_d * twin_grad_e

        L_phi = self._corrosion_mobility()
        phi_nonlin = torch.nan_to_num(phi_nonlin, nan=0.0, posinf=0.0, neginf=0.0)
        L_phi_pos = torch.clamp(L_phi, min=0.0)
        kappa_field = torch.full_like(phi, float(kappa_phi))
        # Allen-Cahn 梯度项：L(phi)*kappa*laplacian(phi)，不可写成 div(L*kappa*grad(phi))。
        phi_lap_kappa_prev = self._div_diffusive_fv(
            phi,
            kappa_field,
            dx=self.dx_m,
            dy=self.dy_m,
            x_coords=self.x_coords_m,
            y_coords=self.y_coords_m,
            bc=bc,
        )
        phi_mode = str(self.cfg.numerics.phi_integrator).strip().lower()
        if phi_mode.startswith("imex"):
            # IMEX-L0 分裂：隐式部分使用常数 L0，显式补偿 (L-L0) 项，避免引入 grad(L) 假项。
            L0 = self._select_phi_imex_l0(L_phi_pos)
            if L0 > 1e-18:
                rhs_phi = phi - dt_s * L_phi_pos * phi_nonlin + dt_s * (L_phi_pos - L0) * phi_lap_kappa_prev
                phi_new = self._solve_variable_diffusion_imex(
                    rhs_phi,
                    coeff=dt_s * L0,
                    diff_coef=kappa_field,
                    dx=self.dx_m,
                    dy=self.dy_m,
                    x_coords=self.x_coords_m,
                    y_coords=self.y_coords_m,
                    x0=phi,
                )
            else:
                phi_rhs = -L_phi_pos * phi_nonlin + L_phi_pos * phi_lap_kappa_prev
                phi_rhs = torch.nan_to_num(phi_rhs, nan=0.0, posinf=0.0, neginf=0.0)
                phi_new = phi + dt_s * phi_rhs
        else:
            phi_rhs = -L_phi_pos * phi_nonlin + L_phi_pos * phi_lap_kappa_prev
            phi_rhs = torch.nan_to_num(phi_rhs, nan=0.0, posinf=0.0, neginf=0.0)
            # 显式更新 phi（旧路径保留，便于 A/B 对照）。
            phi_new = phi + dt_s * phi_rhs
        phi_new = torch.clamp(phi_new, 0.0, 1.0)
        phi_new = torch.nan_to_num(phi_new, nan=0.0, posinf=1.0, neginf=0.0)

        # 扩散方程（Kovacevic 形式），支持可配置积分策略。
        phi_c = phi_new
        hphi_c = smooth_heaviside(phi_c)
        hphi_d_c = smooth_heaviside_prime(phi_c)
        D = self.cfg.corrosion.D_s_m2_s * hphi_c + (1.0 - hphi_c) * self.cfg.corrosion.D_l_m2_s
        # 耦合修正项：由相场梯度导致的平衡浓度梯度贡献。
        gx_phi, gy_phi = grad_xy(
            phi_c,
            self.dx_m,
            self.dy_m,
            x_coords=self.x_coords_m,
            y_coords=self.y_coords_m,
            bc=bc,
        )
        corr = hphi_d_c * (cl_eq - cs_eq)
        stage_dts = self._diffusion_stage_dts(dt_s)
        c_tmp = cbar
        for dt_stage in stage_dts:
            # STS / 子循环阶段推进：阶段内不做硬截断，减少对动力学的投影篡改。
            rhs_c = self._diffusion_rhs(c_tmp, D, corr, gx_phi, gy_phi, phi_c)
            c_tmp = c_tmp + float(dt_stage) * rhs_c
            c_tmp = torch.nan_to_num(
                c_tmp,
                nan=cl_eq,
                posinf=self.cfg.corrosion.cMg_max,
                neginf=self.cfg.corrosion.cMg_min,
            )
        c_new = torch.clamp(c_tmp, min=self.cfg.corrosion.cMg_min, max=self.cfg.corrosion.cMg_max)
        if bool(getattr(self.cfg.numerics, "concentration_mass_projection", False)):
            c_ref = float(self._mean_field(cbar).item())
            c_new = self._project_concentration_mean(
                c_new,
                c_ref,
                n_iter=int(getattr(self.cfg.numerics, "concentration_mass_projection_iters", 2)),
            )
            self.stats["mass_projection_applied"] += 1

        # 孪晶 TDGL 方程推进。
        heta_d = smooth_heaviside_prime(eta)
        # TDGL 与固液界面耦合应使用最新 phi，避免“先错算再投影”。
        hphi_eta = smooth_heaviside(phi_new)
        k_eta = self.cfg.twinning.kappa_eta
        grad_coef_eta = k_eta * hphi_eta
        w_t = self.cfg.twinning.W_barrier_MPa
        twin_dw = hphi_eta * 2.0 * w_t * eta * (1.0 - eta) * (1.0 - 2.0 * eta)
        tau_tw = self._resolved_twin_shear(self.last_mech["sigma_xx"], self.last_mech["sigma_yy"], self.last_mech["sigma_xy"])
        # 正向驱动：分解剪应力 * 孪晶剪切量 * 势函数导数。
        tw_drive = hphi_eta * tau_tw * self.cfg.twinning.gamma_twin * heta_d

        # 在缺口附近且分解剪应力超过阈值时引入成核扰动。
        r2 = (self.grid.x_um - self.cfg.domain.notch_tip_x_um) ** 2 + (self.grid.y_um - self.cfg.domain.notch_center_y_um) ** 2
        nuc = torch.exp(-r2 / max(self.cfg.twinning.nucleation_center_radius_um ** 2, 1e-12))
        act = self._twin_nucleation_activation(tau_tw)
        nuc_source = hphi_eta * self._twin_nucleation_source_amp() * nuc * act

        eta_source = twin_dw - tw_drive - nuc_source
        eta_source = torch.nan_to_num(eta_source, nan=0.0, posinf=0.0, neginf=0.0)
        eta_mode = str(self.cfg.numerics.eta_integrator).strip().lower()
        if eta_mode.startswith("imex"):
            # IMEX: 梯度正则项隐式，驱动项显式。
            rhs_eta = eta - dt_s * self.cfg.twinning.L_eta * eta_source
            eta_new = self._solve_variable_diffusion_imex(
                rhs_eta,
                coeff=dt_s * self.cfg.twinning.L_eta,
                diff_coef=grad_coef_eta,
                dx=self.dx_um,
                dy=self.dy_um,
                x_coords=self.x_coords_um,
                y_coords=self.y_coords_um,
                x0=eta,
            )
        else:
            grad_term = self._div_diffusive_fv(
                eta,
                torch.clamp(grad_coef_eta, min=0.0),
                dx=self.dx_um,
                dy=self.dy_um,
                x_coords=self.x_coords_um,
                y_coords=self.y_coords_um,
                bc=bc,
            )
            eta_rhs = eta_source - grad_term
            eta_rhs = torch.nan_to_num(eta_rhs, nan=0.0, posinf=0.0, neginf=0.0)
            eta_new = eta - dt_s * self.cfg.twinning.L_eta * eta_rhs
        eta_new = torch.clamp(eta_new, 0.0, 1.0)
        eta_new = torch.nan_to_num(eta_new, nan=0.0, posinf=1.0, neginf=0.0)
        eta_new = eta_new * self._solid_weight(phi_new)

        self.state["phi"] = phi_new
        self.state["c"] = c_new
        self.state["eta"] = eta_new
        self.state["ux"] = self.last_mech["ux"]
        self.state["uy"] = self.last_mech["uy"]

    def step(self, step_idx: int) -> StepDiagnostics:
        """执行一个总步（可能是 surrogate 步或物理步）。"""
        dt = self.cfg.numerics.dt_s
        used_surrogate_only = False
        prev_phi = self.state["phi"].clone()
        prev_c = self.state["c"].clone()
        prev_eta = self.state["eta"].clone()
        prev_epspeq = self.state["epspeq"].clone()
        self._last_gate_metrics["pde_phi"] = 0.0
        self._last_gate_metrics["pde_c"] = 0.0
        self._last_gate_metrics["pde_eta"] = 0.0
        self._last_gate_metrics["pde_mech"] = 0.0
        self._last_gate_metrics["uncertainty"] = 0.0
        self._last_gate_metrics["mass_delta"] = 0.0

        ctx = torch.inference_mode if self.cfg.numerics.inference_mode else nullcontext
        if self.device.type == "cuda":
            amp_enabled = self.cfg.numerics.mixed_precision
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)
        else:
            amp_ctx = nullcontext()
        with ctx(), amp_ctx:
            if self.surrogate is not None and self.cfg.ml.mode == "predictor_corrector":
                # predictor-corrector 模式：优先 surrogate，周期性强制物理锚定。
                anchor_every = max(0, int(self.cfg.ml.anchor_physics_every))
                if anchor_every > 0 and step_idx % anchor_every == 0:
                    # 锚定步：完整物理推进，重置拒绝计数。
                    self._physics_step(dt, step_idx)
                    self._surrogate_reject_streak = 0
                    self._surrogate_pause_until_step = step_idx
                else:
                    roll = max(1, int(self._current_rollout_every))
                    if step_idx < self._surrogate_pause_until_step:
                        self._physics_step(dt, step_idx)
                    elif step_idx % roll == 0:
                        # surrogate 候选步：预测 -> 限幅 -> 门控 -> 接受或回退。
                        self.stats["ml_surrogate_attempt"] += 1
                        pred = self.surrogate.predict(self.state)
                        pred = self._limit_surrogate_update(self.state, pred)
                        gate_mech: Dict[str, torch.Tensor] | None = None
                        if self.cfg.ml.enable_surrogate_mechanics_correction:
                            # 门控与执行必须使用一致的力学状态，避免“过门后再变系统”。
                            gate_mech = self.mech.solve_quasi_static(
                                pred,
                                self._epsp_from_state(pred),
                                self.dx_um,
                                self.dy_um,
                                x_coords_um=self.x_coords_um,
                                y_coords_um=self.y_coords_um,
                            )
                        if self._surrogate_update_is_valid(self.state, pred, candidate_mech=gate_mech):
                            self.stats["ml_surrogate_accept"] += 1
                            if bool(getattr(self.cfg.ml, "enforce_mass_projection_on_accept", False)):
                                c_ref = float(self._mean_field(self.state["c"]).item())
                                pred["c"] = self._project_concentration_mean(
                                    pred["c"],
                                    c_ref,
                                    n_iter=int(getattr(self.cfg.numerics, "concentration_mass_projection_iters", 2)),
                                )
                                self.stats["mass_projection_applied"] += 1
                            self.state = pred
                            self.epsp = self._epsp_from_state(self.state)
                            self._sync_epsp_to_state()
                            if not bool(getattr(self.cfg.ml, "surrogate_update_plastic_fields", False)):
                                self.state["epspeq_dot"] = torch.zeros_like(self.state["phi"])
                            if self.cfg.ml.enable_surrogate_mechanics_correction:
                                # 可选：对 surrogate 状态再做一次力学校正。
                                self.last_mech = gate_mech if gate_mech is not None else self.mech.solve_quasi_static(
                                    self.state,
                                    self.epsp,
                                    self.dx_um,
                                    self.dy_um,
                                    x_coords_um=self.x_coords_um,
                                    y_coords_um=self.y_coords_um,
                                )
                                self.state["ux"] = self.last_mech["ux"]
                                self.state["uy"] = self.last_mech["uy"]
                            else:
                                # 若不做校正，至少用当前位移场重建一致的应力诊断量。
                                self.last_mech = self._estimate_state_mechanics(self.state)
                            # 若 surrogate 直接更新了塑性场，则同步推进 CP 内变量，保持状态闭合。
                            self._sync_cp_state_on_surrogate_accept(dt_s=dt, mech_for_cp=self.last_mech)
                            self._surrogate_reject_streak = 0
                            self._surrogate_pause_until_step = step_idx
                            self._on_surrogate_accept()
                            used_surrogate_only = True
                        else:
                            self.stats["ml_surrogate_reject"] += 1
                            # 门控失败：记录拒绝并回退到物理步。
                            self._surrogate_reject_streak += 1
                            self._on_surrogate_reject()
                            if self._surrogate_reject_streak >= max(1, self.cfg.ml.max_consecutive_reject_before_pause):
                                if self.cfg.ml.disable_surrogate_after_reject_burst:
                                    self.surrogate = None
                                    self._surrogate_pause_until_step = self.cfg.numerics.n_steps + 1
                                else:
                                    # 仅暂停若干步后再尝试 surrogate。
                                    self._surrogate_pause_until_step = step_idx + max(1, self.cfg.ml.pause_steps_after_reject_burst)
                                self._surrogate_reject_streak = 0
                            self._physics_step(dt, step_idx)
                    else:
                        self._physics_step(dt, step_idx)
            else:
                self._physics_step(dt, step_idx)

        self._sanitize_state()

        # 诊断量均在清洗后的状态上计算，保证输出稳定且可比。
        solid_fraction = float(self._mean_field(self.state["phi"]).item())
        avg_eta = float(self._mean_field(self._twin_fraction(self.state["eta"])).item())
        solid = self._solid_weight(self.state["phi"])
        max_sigma_h = float(torch.max(self.last_mech["sigma_h"] * solid).item())
        avg_epspeq = float(self._mean_field(self.state["epspeq"]).item())
        loss_phi_mae = float(self._mean_field(torch.abs(self.state["phi"] - prev_phi)).item())
        loss_c_mae = float(self._mean_field(torch.abs(self.state["c"] - prev_c)).item())
        loss_eta_mae = float(self._mean_field(torch.abs(self.state["eta"] - prev_eta)).item())
        loss_epspeq_mae = float(self._mean_field(torch.abs(self.state["epspeq"] - prev_epspeq)).item())
        mech_diag = self._estimate_state_mechanics(self.state)
        free_energy = float(self._compute_free_energy(self.state, mech_diag))
        self._last_gate_metrics["free_energy"] = free_energy
        return StepDiagnostics(
            step=step_idx,
            time_s=step_idx * dt,
            solid_fraction=solid_fraction,
            avg_eta=avg_eta,
            max_sigma_h=max_sigma_h,
            avg_epspeq=avg_epspeq,
            used_surrogate_only=used_surrogate_only,
            loss_phi_mae=loss_phi_mae,
            loss_c_mae=loss_c_mae,
            loss_eta_mae=loss_eta_mae,
            loss_epspeq_mae=loss_epspeq_mae,
            free_energy=free_energy,
            pde_res_phi=float(self._last_gate_metrics.get("pde_phi", 0.0)),
            pde_res_c=float(self._last_gate_metrics.get("pde_c", 0.0)),
            pde_res_eta=float(self._last_gate_metrics.get("pde_eta", 0.0)),
            pde_res_mech=float(self._last_gate_metrics.get("pde_mech", 0.0)),
            rollout_every=int(self._current_rollout_every),
        )

    def run(
        self,
        progress: bool = False,
        progress_every: int = 50,
        progress_prefix: str = "sim",
    ) -> Dict[str, Path | float]:
        """执行完整算例并按配置输出快照/图像/历史表。"""
        t0_wall = time.perf_counter()
        out_dir = ensure_dir(Path(self.cfg.runtime.output_dir) / self.cfg.runtime.case_name)
        snap_dir = ensure_dir(out_dir / "snapshots")
        fig_dir = ensure_dir(out_dir / "figures")
        grid_dir = ensure_dir(out_dir / "grid")
        if self.cfg.runtime.clean_output:
            # 开启 clean_output 时先清理旧结果，再写入新结果。
            for p in snap_dir.glob("snapshot_*.npz"):
                p.unlink(missing_ok=True)
            for p in fig_dir.glob("fields_*.*"):
                p.unlink(missing_ok=True)
            for p in grid_dir.glob("*.*"):
                p.unlink(missing_ok=True)
            cloud_dir = out_dir / "final_clouds"
            if cloud_dir.exists():
                for p in cloud_dir.glob("*.*"):
                    p.unlink(missing_ok=True)
            tip_dir = out_dir / "final_clouds_tip_zoom"
            if tip_dir.exists():
                for p in tip_dir.glob("*.*"):
                    p.unlink(missing_ok=True)
            (out_dir / "history.csv").unlink(missing_ok=True)

        for i in range(1, self.cfg.numerics.n_steps + 1):
            # 主时间推进循环。
            diag = self.step(i)
            self.history.append(
                {
                    "step": diag.step,
                    "time_s": diag.time_s,
                    "solid_fraction": diag.solid_fraction,
                    "avg_eta": diag.avg_eta,
                    "max_sigma_h": diag.max_sigma_h,
                    "avg_epspeq": diag.avg_epspeq,
                    "used_surrogate_only": float(diag.used_surrogate_only),
                    "loss_phi_mae": diag.loss_phi_mae,
                    "loss_c_mae": diag.loss_c_mae,
                    "loss_eta_mae": diag.loss_eta_mae,
                    "loss_epspeq_mae": diag.loss_epspeq_mae,
                    "free_energy": diag.free_energy,
                    "pde_res_phi": diag.pde_res_phi,
                    "pde_res_c": diag.pde_res_c,
                    "pde_res_eta": diag.pde_res_eta,
                    "pde_res_mech": diag.pde_res_mech,
                    "rollout_every": float(diag.rollout_every),
                }
            )
            if progress and (i == 1 or i % max(1, progress_every) == 0 or i == self.cfg.numerics.n_steps):
                frac = i / max(self.cfg.numerics.n_steps, 1)
                bar_n = 28
                fill = int(bar_n * frac)
                bar = "#" * fill + "-" * (bar_n - fill)
                total_t = self.cfg.numerics.n_steps * self.cfg.numerics.dt_s
                wall_elapsed_s = time.perf_counter() - t0_wall
                wall_eta_s = wall_elapsed_s * (1.0 - frac) / max(frac, 1e-12)
                print(
                    f"[{progress_prefix}] [{bar}] {frac*100:6.2f}% "
                    f"t={diag.time_s:.6f}/{total_t:.6f} s dt={self.cfg.numerics.dt_s:.2e} s "
                    f"wall={_fmt_wall_time(wall_elapsed_s)} eta_wall={_fmt_wall_time(wall_eta_s)} "
                    f"phi={diag.solid_fraction:.6f} eta={diag.avg_eta:.6f} "
                    f"sigma_h_max={diag.max_sigma_h:.3f} epspeq={diag.avg_epspeq:.6e} "
                    f"loss_phi={diag.loss_phi_mae:.3e} loss_c={diag.loss_c_mae:.3e} "
                    f"loss_eta={diag.loss_eta_mae:.3e} loss_epspeq={diag.loss_epspeq_mae:.3e} "
                    f"F={diag.free_energy:.3e} "
                    f"Rphi={diag.pde_res_phi:.2e} Rc={diag.pde_res_c:.2e} "
                    f"Reta={diag.pde_res_eta:.2e} Rmech={diag.pde_res_mech:.2e} "
                    f"roll={diag.rollout_every} surrogate={int(diag.used_surrogate_only)}"
                )

            if i % self.cfg.numerics.save_every == 0 or i == 1:
                # 按保存节奏输出快照与中间图。
                save_snapshot_npz(
                    snap_dir,
                    i,
                    state=self.state,
                    extras={
                        "sigma_h": self.last_mech["sigma_h"],
                        "sigma_xx": self.last_mech["sigma_xx"],
                        "sigma_yy": self.last_mech["sigma_yy"],
                        "sigma_xy": self.last_mech["sigma_xy"],
                        "sigma_zz": self.last_mech["sigma_zz"],
                    },
                )
                if self.cfg.runtime.render_intermediate_fields:
                    render_fields(
                        fig_dir,
                        i,
                        self.state,
                        self.last_mech["sigma_h"],
                        self.state["epspeq"],
                        time_s=diag.time_s,
                        extent_um=(0.0, self.cfg.domain.lx_um, 0.0, self.cfg.domain.ly_um),
                        show_grid=False,
                        x_coords_um=self.x_coords_um,
                        y_coords_um=self.y_coords_um,
                    )

        hist = save_history_csv(out_dir, self.history)
        total_wall_s = time.perf_counter() - t0_wall
        if self.cfg.runtime.render_final_clouds:
            final_time_s = self.cfg.numerics.n_steps * self.cfg.numerics.dt_s
            if self.cfg.domain.initial_geometry.lower() == "half_space" and self.cfg.domain.add_half_space_triangular_notch:
                tip_center = (
                    self.cfg.domain.half_space_notch_tip_x_um,
                    self.cfg.domain.half_space_notch_center_y_um,
                )
            else:
                tip_center = (
                    self.cfg.domain.notch_tip_x_um,
                    self.cfg.domain.notch_center_y_um,
                )
            tip_zoom_half_w = 0.5 * max(self.cfg.domain.tip_zoom_width_um, 1e-6)
            tip_zoom_half_h = 0.5 * max(self.cfg.domain.tip_zoom_height_um, 1e-6)
            render_final_clouds(
                out_dir,
                state=self.state,
                sigma_xx=self.last_mech["sigma_xx"],
                sigma_yy=self.last_mech["sigma_yy"],
                sigma_xy=self.last_mech["sigma_xy"],
                time_s=final_time_s,
                extent_um=(0.0, self.cfg.domain.lx_um, 0.0, self.cfg.domain.ly_um),
                show_grid=False,
                x_coords_um=self.x_coords_um,
                y_coords_um=self.y_coords_um,
                save_svg=True,
                tip_zoom_center_um=tip_center,
                tip_zoom_half_width_um=tip_zoom_half_w,
                tip_zoom_half_height_um=tip_zoom_half_h,
            )
        if self.cfg.runtime.render_grid_figure:
            if self.cfg.domain.initial_geometry.lower() == "half_space" and self.cfg.domain.add_half_space_triangular_notch:
                notch_tip_x = self.cfg.domain.half_space_notch_tip_x_um
                notch_center_y = self.cfg.domain.half_space_notch_center_y_um
                notch_depth = self.cfg.domain.half_space_notch_depth_um
                notch_half_open = self.cfg.domain.half_space_notch_half_opening_um
            else:
                notch_tip_x = self.cfg.domain.notch_tip_x_um
                notch_center_y = self.cfg.domain.notch_center_y_um
                notch_depth = self.cfg.domain.notch_depth_um
                notch_half_open = self.cfg.domain.notch_half_opening_um
            render_grid_figure(
                out_dir,
                x_vec_um=self.x_coords_um,
                y_vec_um=self.y_coords_um,
                time_s=self.cfg.numerics.n_steps * self.cfg.numerics.dt_s,
                notch_tip_x_um=notch_tip_x,
                notch_center_y_um=notch_center_y,
                notch_depth_um=notch_depth,
                notch_half_opening_um=notch_half_open,
            )
        return {
            "output_dir": out_dir,
            "history_csv": hist,
            "snapshots_dir": snap_dir,
            "figures_dir": fig_dir,
            "grid_dir": grid_dir,
            "final_clouds_dir": out_dir / "final_clouds",
            "wall_time_s": total_wall_s,
        }
