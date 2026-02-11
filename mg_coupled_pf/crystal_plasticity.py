"""晶体塑性模块（中文注释版）。

该模块实现一个面向二维投影的晶体塑性更新器：
- 根据滑移系方向计算分解剪应力；
- 采用速率相关流动法则更新滑移速率；
- 采用自硬化+潜硬化更新 CRSS；
- 输出塑性应变率与等效塑性应变率。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .config import SimulationConfig
from .operators import smooth_heaviside


def _angles_to_vectors(angle_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """角度（度）转换为单位方向向量。"""
    ang = torch.deg2rad(angle_deg)
    return torch.cos(ang), torch.sin(ang)


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

        # 1) 基体取向下的滑移方向/法向。
        direction_deg = torch.tensor([s["direction_angle_deg"] for s in self.systems], device=device, dtype=dtype)
        normal_deg = torch.tensor([s["normal_angle_deg"] for s in self.systems], device=device, dtype=dtype)
        self.sx_m, self.sy_m = _angles_to_vectors(direction_deg)
        self.nx_m, self.ny_m = _angles_to_vectors(normal_deg)

        # 2) 孪晶取向：以基体取向整体旋转约 86.3°（文档经验值）。
        theta_tw = torch.tensor(86.3, device=device, dtype=dtype)
        self.sx_t, self.sy_t = _angles_to_vectors(direction_deg + theta_tw)
        self.nx_t, self.ny_t = _angles_to_vectors(normal_deg + theta_tw)

        # 2D 投影下的 Schmid 张量分量（矩阵相 / 孪晶相）。
        self.pxx_m = self.sx_m * self.nx_m
        self.pyy_m = self.sy_m * self.ny_m
        self.pxy_m = 0.5 * (self.sx_m * self.ny_m + self.sy_m * self.nx_m)
        self.pxx_t = self.sx_t * self.nx_t
        self.pyy_t = self.sy_t * self.ny_t
        self.pxy_t = 0.5 * (self.sx_t * self.ny_t + self.sy_t * self.nx_t)

        # 3) 潜硬化矩阵：对角线=1（自硬化），非对角=q_latent（潜硬化）。
        q_latent = cfg.crystal_plasticity.q_latent
        q = torch.full((self.n_sys, self.n_sys), q_latent, device=device, dtype=dtype)
        q.fill_diagonal_(1.0)
        self.q_hardening = q

        # 4) 根据滑移系族分配初始 CRSS。
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
        return {"g": g, "gamma_accum": gamma}

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
        g = cp_state["g"]
        schmid = self._effective_schmid_tensors(eta)
        p_xx, p_yy, p_xy = schmid["p_xx"], schmid["p_yy"], schmid["p_xy"]

        # 分解剪应力 tau = P:σ，其中 P_xy 分量按工程应力需乘 2。
        tau = p_xx * sigma_xx + p_yy * sigma_yy + 2.0 * p_xy * sigma_xy

        gamma0 = self.cfg.crystal_plasticity.gamma0_s_inv
        m = self.cfg.crystal_plasticity.m_rate_sensitivity
        g_eff = torch.clamp(g, min=1e-6)

        yield_scale = 1.0
        if material_overrides and "yield_scale" in material_overrides:
            yield_scale = material_overrides["yield_scale"]
        g_eff = g_eff * yield_scale

        ratio = tau / g_eff
        # 限制过大比值，避免幂函数在极端值下数值爆炸。
        ratio = torch.clamp(ratio, min=-8.0, max=8.0)
        plastic_active = True
        if self.cfg.crystal_plasticity.use_overstress:
            # 过应力形式：仅当 |tau/g|-1 > 0 时产生塑性滑移。
            abs_over = torch.clamp(torch.abs(ratio) - 1.0, min=0.0)
            if self.cfg.crystal_plasticity.elastic_shortcut and float(torch.max(abs_over).item()) <= 0.0:
                z = torch.zeros_like(sigma_xx)
                return {
                    "g": g,
                    "gamma_accum": cp_state["gamma_accum"],
                    "epsp_dot_xx": z,
                    "epsp_dot_yy": z,
                    "epsp_dot_xy": z,
                    "epspeq_dot": z,
                    "tau": tau,
                    "plastic_active": False,
                }
            gamma_dot = gamma0 * torch.sign(ratio) * torch.pow(abs_over, m)
        else:
            gamma_dot = gamma0 * torch.sign(ratio) * torch.pow(torch.abs(ratio), m)
        gamma_dot = torch.nan_to_num(gamma_dot, nan=0.0, posinf=0.0, neginf=0.0)
        cap = max(self.cfg.crystal_plasticity.gamma_dot_max_s_inv, 1e-6)
        gamma_dot = torch.clamp(gamma_dot, min=-cap, max=cap)

        # 线性自硬化 + 潜硬化
        abs_gamma = torch.abs(gamma_dot)
        # [B, S, H, W] -> [B, H, W, S]
        abs_perm = abs_gamma.permute(0, 2, 3, 1)
        # 与潜硬化矩阵相乘，得到每个系的总硬化速率。
        dg_perm = torch.matmul(abs_perm, self.q_hardening.T) * self.cfg.crystal_plasticity.h_MPa
        dg = dg_perm.permute(0, 3, 1, 2)
        dg_cap = max(self.cfg.crystal_plasticity.hardening_rate_cap_MPa_s, 1e-6)
        dg = torch.clamp(dg, min=0.0, max=dg_cap)
        g_new = torch.clamp(g + dt_s * dg, min=1e-3)

        # 通过对称 Schmid 张量汇总塑性应变率张量。
        epsp_dot_xx = torch.sum(gamma_dot * p_xx, dim=1, keepdim=True)
        epsp_dot_yy = torch.sum(gamma_dot * p_yy, dim=1, keepdim=True)
        epsp_dot_xy = torch.sum(gamma_dot * p_xy, dim=1, keepdim=True)

        tr = epsp_dot_xx + epsp_dot_yy
        # J2 不变量 -> 等效塑性应变率。
        dev_xx = epsp_dot_xx - tr / 3.0
        dev_yy = epsp_dot_yy - tr / 3.0
        dev_zz = -tr / 3.0
        j2 = 0.5 * (dev_xx * dev_xx + dev_yy * dev_yy + dev_zz * dev_zz + 2.0 * epsp_dot_xy * epsp_dot_xy)
        epspeq_dot = torch.sqrt(torch.clamp(2.0 * j2 / 3.0, min=0.0))
        epspeq_dot = torch.nan_to_num(epspeq_dot, nan=0.0, posinf=1e6, neginf=0.0)

        return {
            "g": g_new,
            "gamma_accum": cp_state["gamma_accum"] + dt_s * abs_gamma,
            "epsp_dot_xx": epsp_dot_xx,
            "epsp_dot_yy": epsp_dot_yy,
            "epsp_dot_xy": epsp_dot_xy,
            "epspeq_dot": epspeq_dot,
            "tau": tau,
            "plastic_active": plastic_active,
        }
