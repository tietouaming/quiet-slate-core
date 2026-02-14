"""晶体塑性模块（中文注释版）。

该模块支持两类滑移系输入：
1. 旧版二维角度定义：`direction_angle_deg/normal_angle_deg`
2. 新版三维晶体学定义：`s_crystal/n_crystal`

并支持：
- Bunge ZXZ 欧拉角取向映射；
- 孪晶重取向（轴角）；
- 基于 `h(eta)` 的母相/孪晶 Schmid 与强度学平滑混合。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .config import SimulationConfig
from .crystallography import (
    axis_angle_matrix,
    bunge_euler_matrix,
    normalize_vec3,
    orthonormalize_pair,
    project_pair_to_xy,
    resolve_system_pair_crystal,
)
from .operators import smooth_heaviside


class CrystalPlasticityModel:
    """晶体塑性更新器。"""

    def __init__(self, cfg: SimulationConfig, device: torch.device, dtype: torch.dtype):
        """初始化滑移系方向、初始 CRSS 与潜硬化矩阵。"""
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.systems = cfg.slip_systems
        self.n_sys = len(self.systems)
        if self.n_sys == 0:
            raise ValueError("No slip systems were loaded.")

        euler = list(getattr(cfg.crystal_plasticity, "crystal_orientation_euler_deg", [0.0, 0.0, 0.0]))
        tw_axis = list(getattr(cfg.crystal_plasticity, "twin_reorientation_axis", [0.0, 0.0, 1.0]))
        tw_ang = float(getattr(cfg.crystal_plasticity, "twin_reorientation_angle_deg", 86.3))
        r_parent = bunge_euler_matrix(euler, device=device, dtype=dtype)
        q_tw = axis_angle_matrix(tw_axis, tw_ang, device=device, dtype=dtype)
        r_twin = q_tw @ r_parent

        sx_m: List[torch.Tensor] = []
        sy_m: List[torch.Tensor] = []
        nx_m: List[torch.Tensor] = []
        ny_m: List[torch.Tensor] = []
        sx_t: List[torch.Tensor] = []
        sy_t: List[torch.Tensor] = []
        nx_t: List[torch.Tensor] = []
        ny_t: List[torch.Tensor] = []
        c_over_a = float(getattr(cfg.crystal_plasticity, "hcp_c_over_a", 1.624))

        for s in self.systems:
            sc, nc = resolve_system_pair_crystal(
                s,
                c_over_a=c_over_a,
                device=device,
                dtype=dtype,
            )
            sc, nc = orthonormalize_pair(sc, nc)

            sp = r_parent @ sc
            np_ = r_parent @ nc
            st = r_twin @ sc
            nt = r_twin @ nc
            a, b, c, d = project_pair_to_xy(sp, np_, device=device, dtype=dtype)
            sx_m.append(a)
            sy_m.append(b)
            nx_m.append(c)
            ny_m.append(d)
            a, b, c, d = project_pair_to_xy(st, nt, device=device, dtype=dtype)
            sx_t.append(a)
            sy_t.append(b)
            nx_t.append(c)
            ny_t.append(d)

        self.sx_m = torch.stack(sx_m)
        self.sy_m = torch.stack(sy_m)
        self.nx_m = torch.stack(nx_m)
        self.ny_m = torch.stack(ny_m)
        self.sx_t = torch.stack(sx_t)
        self.sy_t = torch.stack(sy_t)
        self.nx_t = torch.stack(nx_t)
        self.ny_t = torch.stack(ny_t)

        # 2D 投影下的 Schmid 张量分量（矩阵相 / 孪晶相）。
        self.pxx_m = self.sx_m * self.nx_m
        self.pyy_m = self.sy_m * self.ny_m
        self.pxy_m = 0.5 * (self.sx_m * self.ny_m + self.sy_m * self.nx_m)
        self.pxx_t = self.sx_t * self.nx_t
        self.pyy_t = self.sy_t * self.ny_t
        self.pxy_t = 0.5 * (self.sx_t * self.ny_t + self.sy_t * self.nx_t)

        # 潜硬化矩阵：对角线=1（自硬化），非对角=q_latent（潜硬化）。
        q_latent = cfg.crystal_plasticity.q_latent
        q = torch.full((self.n_sys, self.n_sys), q_latent, device=device, dtype=dtype)
        q.fill_diagonal_(1.0)
        self.q_hardening = q

        # 根据滑移系族分配初始 CRSS。
        crss0 = []
        for s in self.systems:
            fam = s["family"]
            if fam == "basal":
                crss0.append(cfg.crystal_plasticity.basal_crss_MPa)
            elif fam == "prismatic":
                crss0.append(cfg.crystal_plasticity.prismatic_crss_MPa)
            else:
                crss0.append(cfg.crystal_plasticity.pyramidal_crss_MPa)
        self.crss0 = torch.tensor(crss0, device=device, dtype=dtype).view(1, self.n_sys, 1, 1)

    def init_state(self, ny: int, nx: int) -> Dict[str, torch.Tensor]:
        """初始化内部变量状态（g 与累计滑移）。"""
        g = self.crss0.expand(1, self.n_sys, ny, nx).clone()
        gamma = torch.zeros_like(g)
        return {
            # 兼容旧字段：g / gamma_accum 作为“混合视图”保留。
            "g": g.clone(),
            "gamma_accum": gamma.clone(),
            # 新字段：母晶/孪晶分相硬化历史。
            "g_parent": g.clone(),
            "g_twin": g.clone(),
            "gamma_accum_parent": gamma.clone(),
            "gamma_accum_twin": gamma.clone(),
        }

    def _effective_schmid_tensors(self, eta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """在基体/孪晶两组 Schmid 张量间做体积分数混合。"""
        w = smooth_heaviside(torch.clamp(eta, 0.0, 1.0))
        pxx = (1.0 - w) * self.pxx_m.view(1, self.n_sys, 1, 1) + w * self.pxx_t.view(1, self.n_sys, 1, 1)
        pyy = (1.0 - w) * self.pyy_m.view(1, self.n_sys, 1, 1) + w * self.pyy_t.view(1, self.n_sys, 1, 1)
        pxy = (1.0 - w) * self.pxy_m.view(1, self.n_sys, 1, 1) + w * self.pxy_t.view(1, self.n_sys, 1, 1)
        return {"p_xx": pxx, "p_yy": pyy, "p_xy": pxy}

    def update(
        self,
        cp_state: Dict[str, torch.Tensor],
        sigma_xx: torch.Tensor,
        sigma_yy: torch.Tensor,
        sigma_xy: torch.Tensor,
        eta: torch.Tensor,
        dt_s: float,
        material_overrides: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """执行一个时间步的晶体塑性更新。"""
        g_parent = cp_state.get("g_parent", cp_state["g"])
        g_twin = cp_state.get("g_twin", cp_state["g"])
        gamma_accum_parent = cp_state.get("gamma_accum_parent", cp_state["gamma_accum"])
        gamma_accum_twin = cp_state.get("gamma_accum_twin", cp_state["gamma_accum"])
        f_twin = smooth_heaviside(torch.clamp(eta, 0.0, 1.0))
        f_parent = 1.0 - f_twin

        # 分别计算母晶/孪晶的 Schmid 张量响应。
        p_xx_m = self.pxx_m.view(1, self.n_sys, 1, 1)
        p_yy_m = self.pyy_m.view(1, self.n_sys, 1, 1)
        p_xy_m = self.pxy_m.view(1, self.n_sys, 1, 1)
        p_xx_t = self.pxx_t.view(1, self.n_sys, 1, 1)
        p_yy_t = self.pyy_t.view(1, self.n_sys, 1, 1)
        p_xy_t = self.pxy_t.view(1, self.n_sys, 1, 1)

        # 分解剪应力 tau = P:σ，其中 P_xy 分量按工程应力需乘 2。
        tau_m = p_xx_m * sigma_xx + p_yy_m * sigma_yy + 2.0 * p_xy_m * sigma_xy
        tau_t = p_xx_t * sigma_xx + p_yy_t * sigma_yy + 2.0 * p_xy_t * sigma_xy

        gamma0 = self.cfg.crystal_plasticity.gamma0_s_inv
        m = self.cfg.crystal_plasticity.m_rate_sensitivity
        m_scale = 1.0
        t_scale = 1.0
        h_m_scale = 1.0
        h_t_scale = 1.0
        if bool(getattr(self.cfg.crystal_plasticity, "use_phase_dependent_strength", True)):
            m_scale = float(getattr(self.cfg.crystal_plasticity, "matrix_crss_scale", 1.0))
            t_scale = float(getattr(self.cfg.crystal_plasticity, "twin_crss_scale", 1.0))
            h_m_scale = float(getattr(self.cfg.crystal_plasticity, "matrix_hardening_scale", 1.0))
            h_t_scale = float(getattr(self.cfg.crystal_plasticity, "twin_hardening_scale", 1.0))

        yield_scale = 1.0
        if material_overrides and "yield_scale" in material_overrides:
            yield_scale = material_overrides["yield_scale"]
        g_eff_m = torch.clamp(g_parent * float(m_scale) * yield_scale, min=1e-6)
        g_eff_t = torch.clamp(g_twin * float(t_scale) * yield_scale, min=1e-6)

        ratio_m = torch.clamp(tau_m / g_eff_m, min=-8.0, max=8.0)
        ratio_t = torch.clamp(tau_t / g_eff_t, min=-8.0, max=8.0)
        plastic_active = True
        if self.cfg.crystal_plasticity.use_overstress:
            abs_over_m = torch.clamp(torch.abs(ratio_m) - 1.0, min=0.0)
            abs_over_t = torch.clamp(torch.abs(ratio_t) - 1.0, min=0.0)
            if self.cfg.crystal_plasticity.elastic_shortcut and float(torch.max(torch.maximum(abs_over_m, abs_over_t)).item()) <= 0.0:
                z = torch.zeros_like(sigma_xx)
                tau_mix = f_parent * tau_m + f_twin * tau_t
                g_mix = f_parent * g_parent + f_twin * g_twin
                ga_mix = f_parent * gamma_accum_parent + f_twin * gamma_accum_twin
                return {
                    "g": g_mix,
                    "gamma_accum": ga_mix,
                    "g_parent": g_parent,
                    "g_twin": g_twin,
                    "gamma_accum_parent": gamma_accum_parent,
                    "gamma_accum_twin": gamma_accum_twin,
                    "epsp_dot_xx": z,
                    "epsp_dot_yy": z,
                    "epsp_dot_xy": z,
                    "epspeq_dot": z,
                    "tau": tau_mix,
                    "plastic_active": False,
                }
            gamma_dot_m = gamma0 * torch.sign(ratio_m) * torch.pow(abs_over_m, m)
            gamma_dot_t = gamma0 * torch.sign(ratio_t) * torch.pow(abs_over_t, m)
        else:
            gamma_dot_m = gamma0 * torch.sign(ratio_m) * torch.pow(torch.abs(ratio_m), m)
            gamma_dot_t = gamma0 * torch.sign(ratio_t) * torch.pow(torch.abs(ratio_t), m)
        gamma_dot_m = torch.nan_to_num(gamma_dot_m, nan=0.0, posinf=0.0, neginf=0.0)
        gamma_dot_t = torch.nan_to_num(gamma_dot_t, nan=0.0, posinf=0.0, neginf=0.0)
        cap = max(self.cfg.crystal_plasticity.gamma_dot_max_s_inv, 1e-6)
        gamma_dot_m = torch.clamp(gamma_dot_m, min=-cap, max=cap)
        gamma_dot_t = torch.clamp(gamma_dot_t, min=-cap, max=cap)

        abs_gamma_m = torch.abs(gamma_dot_m)
        abs_gamma_t = torch.abs(gamma_dot_t)
        abs_perm_m = abs_gamma_m.permute(0, 2, 3, 1)
        abs_perm_t = abs_gamma_t.permute(0, 2, 3, 1)
        dg_perm_m = torch.matmul(abs_perm_m, self.q_hardening.T) * self.cfg.crystal_plasticity.h_MPa
        dg_perm_t = torch.matmul(abs_perm_t, self.q_hardening.T) * self.cfg.crystal_plasticity.h_MPa
        dg_m = dg_perm_m.permute(0, 3, 1, 2) * max(float(h_m_scale), 1e-3)
        dg_t = dg_perm_t.permute(0, 3, 1, 2) * max(float(h_t_scale), 1e-3)
        dg_cap = max(self.cfg.crystal_plasticity.hardening_rate_cap_MPa_s, 1e-6)
        dg_m = torch.clamp(dg_m, min=0.0, max=dg_cap)
        dg_t = torch.clamp(dg_t, min=0.0, max=dg_cap)
        g_parent_new = torch.clamp(g_parent + dt_s * dg_m, min=1e-3)
        g_twin_new = torch.clamp(g_twin + dt_s * dg_t, min=1e-3)

        # 母晶/孪晶分别求解塑性应变率，再按体积分数平滑混合。
        epsp_dot_xx_m = torch.sum(gamma_dot_m * p_xx_m, dim=1, keepdim=True)
        epsp_dot_yy_m = torch.sum(gamma_dot_m * p_yy_m, dim=1, keepdim=True)
        epsp_dot_xy_m = torch.sum(gamma_dot_m * p_xy_m, dim=1, keepdim=True)
        epsp_dot_xx_t = torch.sum(gamma_dot_t * p_xx_t, dim=1, keepdim=True)
        epsp_dot_yy_t = torch.sum(gamma_dot_t * p_yy_t, dim=1, keepdim=True)
        epsp_dot_xy_t = torch.sum(gamma_dot_t * p_xy_t, dim=1, keepdim=True)
        epsp_dot_xx = f_parent * epsp_dot_xx_m + f_twin * epsp_dot_xx_t
        epsp_dot_yy = f_parent * epsp_dot_yy_m + f_twin * epsp_dot_yy_t
        epsp_dot_xy = f_parent * epsp_dot_xy_m + f_twin * epsp_dot_xy_t

        # 塑性不可压近似：epsp_dot_zz = -(epsp_dot_xx + epsp_dot_yy)。
        epsp_dot_zz = -(epsp_dot_xx + epsp_dot_yy)
        j2 = 0.5 * (
            epsp_dot_xx * epsp_dot_xx
            + epsp_dot_yy * epsp_dot_yy
            + epsp_dot_zz * epsp_dot_zz
            + 2.0 * epsp_dot_xy * epsp_dot_xy
        )
        epspeq_dot = torch.sqrt(torch.clamp(2.0 * j2 / 3.0, min=0.0))
        epspeq_dot = torch.nan_to_num(epspeq_dot, nan=0.0, posinf=1e6, neginf=0.0)
        tau_mix = f_parent * tau_m + f_twin * tau_t
        gamma_accum_parent_new = gamma_accum_parent + dt_s * abs_gamma_m
        gamma_accum_twin_new = gamma_accum_twin + dt_s * abs_gamma_t
        g_mix = f_parent * g_parent_new + f_twin * g_twin_new
        ga_mix = f_parent * gamma_accum_parent_new + f_twin * gamma_accum_twin_new

        return {
            "g": g_mix,
            "gamma_accum": ga_mix,
            "g_parent": g_parent_new,
            "g_twin": g_twin_new,
            "gamma_accum_parent": gamma_accum_parent_new,
            "gamma_accum_twin": gamma_accum_twin_new,
            "epsp_dot_xx": epsp_dot_xx,
            "epsp_dot_yy": epsp_dot_yy,
            "epsp_dot_xy": epsp_dot_xy,
            "epspeq_dot": epspeq_dot,
            "tau": tau_mix,
            "plastic_active": plastic_active,
        }
