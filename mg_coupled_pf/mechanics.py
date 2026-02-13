"""力学子模块（中文注释版）。

实现内容：
- 小变形假设下的二维准静态力学平衡；
- 支持两类求解器：矩阵自由 PCG 与阻尼松弛；
- 默认采用 hybrid（PCG 失败时回退松弛），兼顾速度与稳健性。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .config import SimulationConfig
from .operators import grad_xy, smooth_heaviside


def _normalize_vec3(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """三维向量归一化。"""
    n = torch.sqrt(torch.clamp(torch.sum(v * v), min=eps))
    return v / n


def _bunge_euler_matrix(euler_deg: List[float], *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Bunge ZXZ 欧拉角旋转矩阵（晶体系 -> 试样系）。"""
    if len(euler_deg) != 3:
        raise ValueError("crystal_orientation_euler_deg must have length 3.")
    a, b, c = [torch.deg2rad(torch.tensor(float(x), dtype=dtype)) for x in euler_deg]
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)
    z = torch.tensor(0.0, dtype=dtype)
    o = torch.tensor(1.0, dtype=dtype)
    rz1 = torch.stack(
        [
            torch.stack([ca, -sa, z]),
            torch.stack([sa, ca, z]),
            torch.stack([z, z, o]),
        ]
    )
    rx = torch.stack(
        [
            torch.stack([o, z, z]),
            torch.stack([z, cb, -sb]),
            torch.stack([z, sb, cb]),
        ]
    )
    rz2 = torch.stack(
        [
            torch.stack([cc, -sc, z]),
            torch.stack([sc, cc, z]),
            torch.stack([z, z, o]),
        ]
    )
    return rz1 @ rx @ rz2


def _axis_angle_matrix(axis: List[float], angle_deg: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """轴角旋转矩阵（Rodrigues）。"""
    if len(axis) != 3:
        raise ValueError("twin_reorientation_axis must have length 3.")
    ax = _normalize_vec3(torch.tensor([float(axis[0]), float(axis[1]), float(axis[2])], dtype=dtype))
    th = torch.deg2rad(torch.tensor(float(angle_deg), dtype=dtype))
    c = torch.cos(th)
    s = torch.sin(th)
    x, y, z = ax[0], ax[1], ax[2]
    k = torch.tensor(
        [
            [0.0, -z.item(), y.item()],
            [z.item(), 0.0, -x.item()],
            [-y.item(), x.item(), 0.0],
        ],
        dtype=dtype,
    )
    i = torch.eye(3, dtype=dtype)
    outer = ax.view(3, 1) @ ax.view(1, 3)
    return c * i + (1.0 - c) * outer + s * k


def _build_hcp_stiffness_tensor(
    c11: float,
    c12: float,
    c13: float,
    c33: float,
    c44: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """构造 HCP 晶体坐标系下的四阶弹性张量 C_ijkl（MPa）。"""
    c = torch.zeros((3, 3, 3, 3), dtype=dtype)
    c66 = 0.5 * (c11 - c12)

    def set_sym(i: int, j: int, k: int, l: int, val: float) -> None:
        idx = (
            (i, j, k, l),
            (j, i, k, l),
            (i, j, l, k),
            (j, i, l, k),
            (k, l, i, j),
            (l, k, i, j),
            (k, l, j, i),
            (l, k, j, i),
        )
        for a, b, d, e in idx:
            c[a, b, d, e] = float(val)

    set_sym(0, 0, 0, 0, c11)
    set_sym(1, 1, 1, 1, c11)
    set_sym(0, 0, 1, 1, c12)
    set_sym(0, 0, 2, 2, c13)
    set_sym(1, 1, 2, 2, c13)
    set_sym(2, 2, 2, 2, c33)
    set_sym(1, 2, 1, 2, c44)
    set_sym(0, 2, 0, 2, c44)
    set_sym(0, 1, 0, 1, c66)
    return c


def _rotate_stiffness(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """将四阶弹性张量从晶体系旋转到试样系。"""
    return torch.einsum("ai,bj,ck,dl,ijkl->abcd", r, r, r, r, c)


def _extract_plane_strain_coeffs(c: torch.Tensor) -> Dict[str, float]:
    """提取二维平面应变使用的应力-应变线性系数。"""
    out: Dict[str, float] = {}
    pairs = {
        "xx": (0, 0),
        "yy": (1, 1),
        "xy": (0, 1),
        "zz": (2, 2),
    }
    for name, (i, j) in pairs.items():
        out[f"{name}_exx"] = float(c[i, j, 0, 0].item())
        out[f"{name}_eyy"] = float(c[i, j, 1, 1].item())
        out[f"{name}_exy"] = float((c[i, j, 0, 1] + c[i, j, 1, 0]).item())
    return out


class MechanicsModel:
    """准静态力学求解器。"""

    def __init__(self, cfg: SimulationConfig):
        """读取材料常数并缓存孪晶方向向量。"""
        self.cfg = cfg
        self.lam = cfg.mechanics.lambda_GPa * 1e3  # MPa
        self.mu = cfg.mechanics.mu_GPa * 1e3  # MPa
        self.use_anisotropic_hcp = bool(getattr(cfg.mechanics, "use_anisotropic_hcp", False))
        self.anisotropic_blend_with_twin = bool(getattr(cfg.mechanics, "anisotropic_blend_with_twin", True))

        # 默认刚度尺度：各向同性模型的切线模量。
        self._stiffness_ref_mpa = float(self.lam + 2.0 * self.mu)
        self._c_parent_coeffs: Dict[str, float] | None = None
        self._c_twin_coeffs: Dict[str, float] | None = None
        if self.use_anisotropic_hcp:
            c11 = float(cfg.mechanics.C11_GPa) * 1e3
            c12 = float(cfg.mechanics.C12_GPa) * 1e3
            c13 = float(cfg.mechanics.C13_GPa) * 1e3
            c33 = float(cfg.mechanics.C33_GPa) * 1e3
            c44 = float(cfg.mechanics.C44_GPa) * 1e3
            c0 = _build_hcp_stiffness_tensor(c11, c12, c13, c33, c44)
            euler = list(getattr(cfg.mechanics, "crystal_orientation_euler_deg", [0.0, 0.0, 0.0]))
            tw_axis = list(getattr(cfg.mechanics, "twin_reorientation_axis", [0.0, 0.0, 1.0]))
            tw_ang = float(getattr(cfg.mechanics, "twin_reorientation_angle_deg", 86.3))
            r_parent = _bunge_euler_matrix(euler)
            r_twin = _axis_angle_matrix(tw_axis, tw_ang) @ r_parent
            c_parent = _rotate_stiffness(c0, r_parent)
            c_twin = _rotate_stiffness(c0, r_twin)
            self._c_parent_coeffs = _extract_plane_strain_coeffs(c_parent)
            self._c_twin_coeffs = _extract_plane_strain_coeffs(c_twin)
            # 线性求解器预条件的刚度参考值取主要系数上界，提升各向异性下的鲁棒性。
            cand: List[float] = []
            for coeffs in [self._c_parent_coeffs, self._c_twin_coeffs]:
                if coeffs is None:
                    continue
                for k in ("xx_exx", "xx_eyy", "xx_exy", "yy_exx", "yy_eyy", "yy_exy", "xy_exx", "xy_eyy", "xy_exy"):
                    cand.append(abs(float(coeffs.get(k, 0.0))))
            if cand:
                self._stiffness_ref_mpa = max(max(cand), 1e-6)

        ang_s = torch.deg2rad(torch.tensor(cfg.twinning.twin_shear_dir_angle_deg, dtype=torch.float64))
        ang_n = torch.deg2rad(torch.tensor(cfg.twinning.twin_plane_normal_angle_deg, dtype=torch.float64))
        self.sx = float(torch.cos(ang_s).item())
        self.sy = float(torch.sin(ang_s).item())
        self.nx = float(torch.cos(ang_n).item())
        self.ny = float(torch.sin(ang_n).item())
        self._krylov_fail_streak = 0
        self._krylov_pause_left = 0

    def twin_strain(self, eta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """根据孪晶序参量 eta 计算本征应变（体积分数使用 f_twin=h(eta)）。"""
        h = smooth_heaviside(eta)
        g = self.cfg.twinning.gamma_twin
        exx = h * g * self.sx * self.nx
        eyy = h * g * self.sy * self.ny
        exy = h * 0.5 * g * (self.sx * self.ny + self.sy * self.nx)
        return {"exx": exx, "eyy": eyy, "exy": exy}

    def _loading_mode(self) -> str:
        """返回规范化后的加载模式。"""
        return str(self.cfg.mechanics.loading_mode).strip().lower()

    def _mech_bc(self) -> str | Dict[str, str]:
        """返回力学离散边界条件（支持按轴独立设置）。"""
        bx = str(getattr(self.cfg.mechanics, "mechanics_bc_x", "")).strip().lower()
        by = str(getattr(self.cfg.mechanics, "mechanics_bc_y", "")).strip().lower()
        if bx or by:
            if not bx:
                bx = str(getattr(self.cfg.mechanics, "mechanics_bc", "neumann")).strip().lower()
            if not by:
                by = str(getattr(self.cfg.mechanics, "mechanics_bc", "neumann")).strip().lower()
            return {"x": bx, "y": by}
        return str(getattr(self.cfg.mechanics, "mechanics_bc", "neumann")).strip().lower()

    def _use_dirichlet_x(self) -> bool:
        """是否采用 x 向位移边界加载。"""
        return self._loading_mode() in {"dirichlet_x", "dirichlet", "ux_dirichlet"}

    def _mech_bc_axes(self) -> Tuple[str, str]:
        """返回按轴拆分后的力学边界条件 `(bc_x, bc_y)`。"""
        bc = self._mech_bc()
        if isinstance(bc, dict):
            bx = str(bc.get("x", "neumann")).strip().lower()
            by = str(bc.get("y", "neumann")).strip().lower()
            return bx, by
        b = str(bc).strip().lower()
        return b, b

    def _macro_external_strain_x(self) -> float:
        """返回宏观外加应变项（仅本征应变加载模式生效）。"""
        if self._use_dirichlet_x():
            return 0.0
        return float(self.cfg.mechanics.external_strain_x)

    def _dirichlet_right_ux(self) -> float:
        """计算右边界位移值（um）。"""
        u = float(self.cfg.mechanics.dirichlet_right_displacement_um)
        if u >= 0.0:
            return u
        return float(self.cfg.mechanics.external_strain_x) * float(self.cfg.domain.lx_um)

    def constitutive_stress(
        self,
        phi: torch.Tensor,
        eps_xx: torch.Tensor,
        eps_yy: torch.Tensor,
        eps_xy: torch.Tensor,
        *,
        eta: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """线弹性本构：由应变场计算 Cauchy 应力。"""
        hphi = smooth_heaviside(phi)
        if self.cfg.mechanics.strict_solid_stress_only:
            # 连续窄带门控，避免硬阈值引入的非物理跳变。
            thr = float(self.cfg.domain.solid_phase_threshold)
            band = 0.02
            gate = 0.5 * (1.0 + torch.tanh((phi - thr) / band))
            hphi = hphi * torch.clamp(gate, 0.0, 1.0)

        def compose_from_coeffs(coeffs: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            sxx = coeffs["xx_exx"] * eps_xx + coeffs["xx_eyy"] * eps_yy + coeffs["xx_exy"] * eps_xy
            syy = coeffs["yy_exx"] * eps_xx + coeffs["yy_eyy"] * eps_yy + coeffs["yy_exy"] * eps_xy
            sxy = coeffs["xy_exx"] * eps_xx + coeffs["xy_eyy"] * eps_yy + coeffs["xy_exy"] * eps_xy
            szz = coeffs["zz_exx"] * eps_xx + coeffs["zz_eyy"] * eps_yy + coeffs["zz_exy"] * eps_xy
            return sxx, syy, sxy, szz

        if self.use_anisotropic_hcp and self._c_parent_coeffs is not None:
            sxx_m, syy_m, sxy_m, szz_m = compose_from_coeffs(self._c_parent_coeffs)
            if self.anisotropic_blend_with_twin and (eta is not None) and (self._c_twin_coeffs is not None):
                sxx_t, syy_t, sxy_t, szz_t = compose_from_coeffs(self._c_twin_coeffs)
                wt = smooth_heaviside(torch.clamp(eta, 0.0, 1.0))
                sigma_xx = (1.0 - wt) * sxx_m + wt * sxx_t
                sigma_yy = (1.0 - wt) * syy_m + wt * syy_t
                sigma_xy = (1.0 - wt) * sxy_m + wt * sxy_t
                sigma_zz = (1.0 - wt) * szz_m + wt * szz_t
            else:
                sigma_xx, sigma_yy, sigma_xy, sigma_zz = sxx_m, syy_m, sxy_m, szz_m
            if not bool(self.cfg.mechanics.plane_strain):
                sigma_zz = torch.zeros_like(sigma_xx)
            sigma_xx = hphi * sigma_xx
            sigma_yy = hphi * sigma_yy
            sigma_xy = hphi * sigma_xy
            sigma_zz = hphi * sigma_zz
        else:
            tr = eps_xx + eps_yy
            sigma_xx = hphi * (self.lam * tr + 2.0 * self.mu * eps_xx)
            sigma_yy = hphi * (self.lam * tr + 2.0 * self.mu * eps_yy)
            sigma_xy = hphi * (2.0 * self.mu * eps_xy)
            if bool(self.cfg.mechanics.plane_strain):
                # 平面应变下 ezz=0，但 szz=lambda*(exx+eyy) 一般不为 0。
                sigma_zz = hphi * (self.lam * tr)
            else:
                sigma_zz = torch.zeros_like(sigma_xx)
        sigma_h = (sigma_xx + sigma_yy + sigma_zz) / 3.0
        return {
            "sigma_xx": sigma_xx,
            "sigma_yy": sigma_yy,
            "sigma_xy": sigma_xy,
            "sigma_zz": sigma_zz,
            "sigma_h": sigma_h,
        }

    def _apply_displacement_constraints(self, ux: torch.Tensor, uy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对位移场施加边界约束。"""
        ux = ux.clone()
        uy = uy.clone()
        ux[:, :, :, 0] = 0.0
        if self._use_dirichlet_x():
            ux[:, :, :, -1] = self._dirichlet_right_ux()
        uy[:, :, 0, 0] = 0.0
        return ux, uy

    def _apply_vector_constraints(self, vx: torch.Tensor, vy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对线性系统向量施加约束（约束自由度置零）。"""
        vx = vx.clone()
        vy = vy.clone()
        vx[:, :, :, 0] = 0.0
        if self._use_dirichlet_x():
            vx[:, :, :, -1] = 0.0
        vy[:, :, 0, 0] = 0.0
        return vx, vy

    def _axis_metrics(
        self,
        n: int,
        d_default: float,
        coords: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """构造控制体宽度。"""
        if n <= 1:
            return torch.full((1,), max(float(d_default), 1e-12), device=device, dtype=dtype)
        if coords is None:
            return torch.full((n,), max(float(d_default), 1e-12), device=device, dtype=dtype)
        dv = torch.clamp(coords[1:] - coords[:-1], min=1e-12).to(device=device, dtype=dtype)
        out = torch.zeros((n,), device=device, dtype=dtype)
        out[0] = 0.5 * dv[0]
        out[-1] = 0.5 * dv[-1]
        if n > 2:
            out[1:-1] = 0.5 * (dv[1:] + dv[:-1])
        return out

    def _face_grad_x(
        self,
        f: torch.Tensor,
        *,
        bc_x: str,
        dx_um: float,
        x_coords_um: torch.Tensor | None,
        left_face_value: float | None = None,
        right_face_value: float | None = None,
    ) -> torch.Tensor:
        """按 x 向面通量差分计算 `∂f/∂x`（FV 形式）。"""
        eps = 1e-12
        if bc_x in {"periodic", "circular"}:
            # 周期边界仅在均匀网格下使用，避免非均匀周期带来的面距歧义。
            dx = max(float(dx_um), eps)
            f_ip12 = 0.5 * (f + torch.roll(f, shifts=-1, dims=3))
            return (f_ip12 - torch.roll(f_ip12, shifts=1, dims=3)) / dx

        w = f.shape[-1]
        cw = self._axis_metrics(w, dx_um, x_coords_um, dtype=f.dtype, device=f.device).view(1, 1, 1, -1)
        left = torch.zeros_like(f[:, :, :, :1])
        right = torch.zeros_like(f[:, :, :, :1])
        if left_face_value is None:
            left = f[:, :, :, :1]
        else:
            left.fill_(float(left_face_value))
        if right_face_value is None:
            right = f[:, :, :, -1:]
        else:
            right.fill_(float(right_face_value))
        interior = 0.5 * (f[:, :, :, 1:] + f[:, :, :, :-1])
        faces = torch.cat([left, interior, right], dim=3)  # [B,C,H,W+1]
        return (faces[:, :, :, 1:] - faces[:, :, :, :-1]) / torch.clamp(cw, min=eps)

    def _face_grad_y(
        self,
        f: torch.Tensor,
        *,
        bc_y: str,
        dy_um: float,
        y_coords_um: torch.Tensor | None,
        bottom_face_value: float | None = None,
        top_face_value: float | None = None,
    ) -> torch.Tensor:
        """按 y 向面通量差分计算 `∂f/∂y`（FV 形式）。"""
        eps = 1e-12
        if bc_y in {"periodic", "circular"}:
            dy = max(float(dy_um), eps)
            f_jp12 = 0.5 * (f + torch.roll(f, shifts=-1, dims=2))
            return (f_jp12 - torch.roll(f_jp12, shifts=1, dims=2)) / dy

        h = f.shape[-2]
        ch = self._axis_metrics(h, dy_um, y_coords_um, dtype=f.dtype, device=f.device).view(1, 1, -1, 1)
        bottom = torch.zeros_like(f[:, :, :1, :])
        top = torch.zeros_like(f[:, :, :1, :])
        if bottom_face_value is None:
            bottom = f[:, :, :1, :]
        else:
            bottom.fill_(float(bottom_face_value))
        if top_face_value is None:
            top = f[:, :, -1:, :]
        else:
            top.fill_(float(top_face_value))
        interior = 0.5 * (f[:, :, 1:, :] + f[:, :, :-1, :])
        faces = torch.cat([bottom, interior, top], dim=2)  # [B,C,H+1,W]
        return (faces[:, :, 1:, :] - faces[:, :, :-1, :]) / torch.clamp(ch, min=eps)

    def _stress_divergence(
        self,
        sigma_xx: torch.Tensor,
        sigma_yy: torch.Tensor,
        sigma_xy: torch.Tensor,
        dx_um: float,
        dy_um: float,
        *,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 `div(sigma)`，在 Neumann 轴上按 traction-free 处理边界面应力。"""
        bx, by = self._mech_bc_axes()
        # x 动量方程：∂x(sxx) + ∂y(sxy) = 0
        # y 动量方程：∂x(sxy) + ∂y(syy) = 0
        tx_left = 0.0 if bx == "neumann" else None
        tx_right = 0.0 if bx == "neumann" else None
        ty_bottom = 0.0 if by == "neumann" else None
        ty_top = 0.0 if by == "neumann" else None

        d_sxx_dx = self._face_grad_x(
            sigma_xx,
            bc_x=bx,
            dx_um=dx_um,
            x_coords_um=x_coords_um,
            left_face_value=tx_left,
            right_face_value=tx_right,
        )
        d_sxy_dy = self._face_grad_y(
            sigma_xy,
            bc_y=by,
            dy_um=dy_um,
            y_coords_um=y_coords_um,
            bottom_face_value=ty_bottom,
            top_face_value=ty_top,
        )
        d_sxy_dx = self._face_grad_x(
            sigma_xy,
            bc_x=bx,
            dx_um=dx_um,
            x_coords_um=x_coords_um,
            left_face_value=tx_left,
            right_face_value=tx_right,
        )
        d_syy_dy = self._face_grad_y(
            sigma_yy,
            bc_y=by,
            dy_um=dy_um,
            y_coords_um=y_coords_um,
            bottom_face_value=ty_bottom,
            top_face_value=ty_top,
        )
        div_x = torch.nan_to_num(d_sxx_dx + d_sxy_dy, nan=0.0, posinf=0.0, neginf=0.0)
        div_y = torch.nan_to_num(d_sxy_dx + d_syy_dy, nan=0.0, posinf=0.0, neginf=0.0)
        return div_x, div_y

    def stress_divergence(
        self,
        sigma_xx: torch.Tensor,
        sigma_yy: torch.Tensor,
        sigma_xy: torch.Tensor,
        dx_um: float,
        dy_um: float,
        *,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对外暴露的力学散度接口（用于残差门控等诊断）。"""
        return self._stress_divergence(
            sigma_xx,
            sigma_yy,
            sigma_xy,
            dx_um,
            dy_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )

    def _disp_strain_only(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """仅由位移梯度构造线性应变。"""
        bc_base = self._mech_bc()
        bc_ux = bc_base
        ux_dirichlet = 0.0
        if self._use_dirichlet_x():
            # ux 在 x 两端为位移 Dirichlet，梯度离散需显式携带左右边界值（可非零）。
            if isinstance(bc_base, dict):
                bc_ux = {"x": "dirichlet0", "y": str(bc_base.get("y", "neumann")).strip().lower()}
            else:
                bc_ux = {"x": "dirichlet0", "y": str(bc_base).strip().lower()}
            ux_dirichlet = {"x_left": 0.0, "x_right": self._dirichlet_right_ux()}
        dux_dx, dux_dy = grad_xy(
            ux,
            dx_um,
            dy_um,
            x_coords=x_coords_um,
            y_coords=y_coords_um,
            bc=bc_ux,
            dirichlet_value=ux_dirichlet,
        )
        duy_dx, duy_dy = grad_xy(uy, dx_um, dy_um, x_coords=x_coords_um, y_coords=y_coords_um, bc=bc_base)
        ex = dux_dx
        ey = duy_dy
        exy = 0.5 * (dux_dy + duy_dx)
        cap = max(self.cfg.mechanics.max_abs_strain, 1e-6)
        ex = torch.clamp(torch.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0), min=-cap, max=cap)
        ey = torch.clamp(torch.nan_to_num(ey, nan=0.0, posinf=0.0, neginf=0.0), min=-cap, max=cap)
        exy = torch.clamp(torch.nan_to_num(exy, nan=0.0, posinf=0.0, neginf=0.0), min=-cap, max=cap)
        return {"eps_xx": ex, "eps_yy": ey, "eps_xy": exy}

    def _strain_from_displacement(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        eta: torch.Tensor,
        epsp: Dict[str, torch.Tensor],
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """位移梯度 -> 总应变，并扣除塑性/孪晶本征应变。"""
        eps_u = self._disp_strain_only(
            ux,
            uy,
            dx_um,
            dy_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )
        eps_tw = self.twin_strain(eta)
        ex = eps_u["eps_xx"] + self._macro_external_strain_x() - epsp["exx"] - eps_tw["exx"]
        ey = eps_u["eps_yy"] - epsp["eyy"] - eps_tw["eyy"]
        exy = eps_u["eps_xy"] - epsp["exy"] - eps_tw["exy"]
        cap = max(self.cfg.mechanics.max_abs_strain, 1e-6)
        ex = torch.clamp(torch.nan_to_num(ex, nan=0.0, posinf=0.0, neginf=0.0), min=-cap, max=cap)
        ey = torch.clamp(torch.nan_to_num(ey, nan=0.0, posinf=0.0, neginf=0.0), min=-cap, max=cap)
        exy = torch.clamp(torch.nan_to_num(exy, nan=0.0, posinf=0.0, neginf=0.0), min=-cap, max=cap)
        return {"eps_xx": ex, "eps_yy": ey, "eps_xy": exy}

    @staticmethod
    def _inner(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """向量场内积。"""
        return torch.sum(x1 * x2 + y1 * y2)

    @staticmethod
    def _pack_uv(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
        """将位移向量场打包为 2 通道张量。"""
        return torch.cat([ux, uy], dim=1)

    @staticmethod
    def _unpack_uv(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 2 通道张量拆为 ux, uy。"""
        return v[:, 0:1], v[:, 1:2]

    @staticmethod
    def _vec_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """打包向量场内积。"""
        return torch.sum(a * b)

    @staticmethod
    def _vec_norm(a: torch.Tensor) -> torch.Tensor:
        """打包向量场二范数。"""
        return torch.sqrt(torch.clamp(torch.sum(a * a), min=0.0))

    def _build_linear_operator_and_rhs(
        self,
        state: Dict[str, torch.Tensor],
        epsp: Dict[str, torch.Tensor],
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ):
        """构造矩阵自由线性算子 A(u)=b（固定 eta/epsp/phi 的单步问题）。"""
        eta = state["eta"]
        phi = state["phi"]
        eps_tw = self.twin_strain(eta)

        ex0 = self._macro_external_strain_x() - epsp["exx"] - eps_tw["exx"]
        ey0 = -epsp["eyy"] - eps_tw["eyy"]
        exy0 = -epsp["exy"] - eps_tw["exy"]
        sig0 = self.constitutive_stress(phi, ex0, ey0, exy0, eta=eta)
        div0_x, div0_y = self._stress_divergence(
            sig0["sigma_xx"],
            sig0["sigma_yy"],
            sig0["sigma_xy"],
            dx_um,
            dy_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )
        b_x = -div0_x
        b_y = -div0_y
        b_x = torch.nan_to_num(b_x, nan=0.0, posinf=0.0, neginf=0.0)
        b_y = torch.nan_to_num(b_y, nan=0.0, posinf=0.0, neginf=0.0)
        b_x, b_y = self._apply_vector_constraints(b_x, b_y)

        def apply_A(ux: torch.Tensor, uy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            eps_u = self._disp_strain_only(
                ux,
                uy,
                dx_um,
                dy_um,
                x_coords_um=x_coords_um,
                y_coords_um=y_coords_um,
            )
            sig_u = self.constitutive_stress(phi, eps_u["eps_xx"], eps_u["eps_yy"], eps_u["eps_xy"], eta=eta)
            ax, ay = self._stress_divergence(
                sig_u["sigma_xx"],
                sig_u["sigma_yy"],
                sig_u["sigma_xy"],
                dx_um,
                dy_um,
                x_coords_um=x_coords_um,
                y_coords_um=y_coords_um,
            )
            ax = torch.nan_to_num(ax, nan=0.0, posinf=0.0, neginf=0.0)
            ay = torch.nan_to_num(ay, nan=0.0, posinf=0.0, neginf=0.0)
            ax, ay = self._apply_vector_constraints(ax, ay)
            return ax, ay

        return apply_A, b_x, b_y

    def _prepare_krylov_linear_system(
        self,
        state: Dict[str, torch.Tensor],
        epsp: Dict[str, torch.Tensor],
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ):
        """准备 Krylov 求解所需的算子、右端和预条件对角。"""
        ux0 = state["ux"]
        uy0 = state["uy"]
        ux0, uy0 = self._apply_displacement_constraints(ux0, uy0)
        apply_A, b_x, b_y = self._build_linear_operator_and_rhs(
            state,
            epsp,
            dx_um,
            dy_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )
        hphi = smooth_heaviside(state["phi"])
        if self.cfg.mechanics.strict_solid_stress_only:
            thr = float(self.cfg.domain.solid_phase_threshold)
            band = 0.02
            gate = 0.5 * (1.0 + torch.tanh((state["phi"] - thr) / band))
            hphi = hphi * torch.clamp(gate, 0.0, 1.0)
        if x_coords_um is not None and x_coords_um.numel() > 1:
            dx_min = float(torch.min(x_coords_um[1:] - x_coords_um[:-1]).item())
        else:
            dx_min = float(dx_um)
        if y_coords_um is not None and y_coords_um.numel() > 1:
            dy_min = float(torch.min(y_coords_um[1:] - y_coords_um[:-1]).item())
        else:
            dy_min = float(dy_um)
        diag_lap = 2.0 / max(dx_min * dx_min, 1e-20) + 2.0 / max(dy_min * dy_min, 1e-20)
        shift = max(float(self.cfg.numerics.mechanics_cg_diag_shift), 1e-9)
        diag = hphi * max(float(self._stiffness_ref_mpa), 1e-6) * diag_lap + shift
        diag_inv = 1.0 / torch.clamp(diag, min=shift)
        n_iter = max(1, int(self.cfg.numerics.mechanics_cg_max_iters))
        abs_tol = max(float(self.cfg.numerics.mechanics_cg_tol_abs), 0.0)
        rel_tol = max(float(self.cfg.numerics.mechanics_cg_tol_rel), 0.0)
        return ux0, uy0, apply_A, b_x, b_y, diag_inv, n_iter, abs_tol, rel_tol

    def _solve_quasi_static_bicgstab(
        self,
        state: Dict[str, torch.Tensor],
        epsp: Dict[str, torch.Tensor],
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, int, float]:
        """矩阵自由 BiCGStab+Jacobi 求解准静态平衡。"""
        ux, uy, apply_A, b_x, b_y, diag_inv, n_iter, abs_tol, rel_tol = self._prepare_krylov_linear_system(
            state,
            epsp,
            dx_um,
            dy_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )

        ax, ay = apply_A(ux, uy)
        r_x = b_x - ax
        r_y = b_y - ay
        r_x, r_y = self._apply_vector_constraints(r_x, r_y)
        r_hat_x = r_x.clone()
        r_hat_y = r_y.clone()
        p_x = torch.zeros_like(r_x)
        p_y = torch.zeros_like(r_y)
        v_x = torch.zeros_like(r_x)
        v_y = torch.zeros_like(r_y)
        rho_old = torch.tensor(1.0, device=r_x.device, dtype=r_x.dtype)
        alpha = torch.tensor(1.0, device=r_x.device, dtype=r_x.dtype)
        omega = torch.tensor(1.0, device=r_x.device, dtype=r_x.dtype)
        r0 = torch.sqrt(torch.clamp(self._inner(r_x, r_y, r_x, r_y), min=0.0))
        tol = max(abs_tol, rel_tol * float(r0.item()))
        if float(r0.item()) <= tol:
            return ux, uy, True, 0, float(r0.item())
        converged = False
        last_res = float(r0.item())
        it = 0
        for it in range(1, n_iter + 1):
            rho = self._inner(r_hat_x, r_hat_y, r_x, r_y)
            if torch.abs(rho).item() <= 1e-30:
                break
            if it == 1:
                p_x = r_x.clone()
                p_y = r_y.clone()
            else:
                if torch.abs(omega).item() <= 1e-30:
                    break
                beta = (rho / rho_old) * (alpha / omega)
                p_x = r_x + beta * (p_x - omega * v_x)
                p_y = r_y + beta * (p_y - omega * v_y)
            p_x, p_y = self._apply_vector_constraints(p_x, p_y)
            ph_x = diag_inv * p_x
            ph_y = diag_inv * p_y
            v_x, v_y = apply_A(ph_x, ph_y)
            denom = self._inner(r_hat_x, r_hat_y, v_x, v_y)
            if torch.abs(denom).item() <= 1e-30:
                break
            alpha = rho / denom
            s_x = r_x - alpha * v_x
            s_y = r_y - alpha * v_y
            s_x, s_y = self._apply_vector_constraints(s_x, s_y)
            s_norm = torch.sqrt(torch.clamp(self._inner(s_x, s_y, s_x, s_y), min=0.0))
            if float(s_norm.item()) <= tol:
                ux = ux + alpha * ph_x
                uy = uy + alpha * ph_y
                ux, uy = self._apply_displacement_constraints(ux, uy)
                converged = True
                last_res = float(s_norm.item())
                break
            sh_x = diag_inv * s_x
            sh_y = diag_inv * s_y
            t_x, t_y = apply_A(sh_x, sh_y)
            tt = self._inner(t_x, t_y, t_x, t_y)
            if torch.abs(tt).item() <= 1e-30:
                break
            omega = self._inner(t_x, t_y, s_x, s_y) / tt
            ux = ux + alpha * ph_x + omega * sh_x
            uy = uy + alpha * ph_y + omega * sh_y
            ux, uy = self._apply_displacement_constraints(ux, uy)
            r_x = s_x - omega * t_x
            r_y = s_y - omega * t_y
            r_x, r_y = self._apply_vector_constraints(r_x, r_y)
            r_norm = torch.sqrt(torch.clamp(self._inner(r_x, r_y, r_x, r_y), min=0.0))
            last_res = float(r_norm.item())
            if last_res <= tol:
                converged = True
                break
            rho_old = rho
        return ux, uy, converged, it, last_res

    def _solve_quasi_static_gmres(
        self,
        state: Dict[str, torch.Tensor],
        epsp: Dict[str, torch.Tensor],
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, int, float]:
        """矩阵自由 GMRES(restart)+Jacobi 求解准静态平衡。"""
        ux, uy, apply_A, b_x, b_y, diag_inv, n_iter_total, abs_tol, rel_tol = self._prepare_krylov_linear_system(
            state,
            epsp,
            dx_um,
            dy_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )
        x = self._pack_uv(ux, uy)
        b = self._pack_uv(b_x, b_y)
        max_restarts = max(1, int(self.cfg.numerics.mechanics_gmres_max_restarts))
        restart = max(2, int(self.cfg.numerics.mechanics_gmres_restart))
        # 用总迭代上限裁剪 (restart * restarts)，保持与旧参数兼容。
        if restart * max_restarts > n_iter_total:
            max_restarts = max(1, n_iter_total // restart)
            restart = max(2, n_iter_total // max_restarts)
        iters_done = 0

        def m_inv(v: torch.Tensor) -> torch.Tensor:
            vx, vy = self._unpack_uv(v)
            return self._pack_uv(diag_inv * vx, diag_inv * vy)

        def apply_A_vec(v: torch.Tensor) -> torch.Tensor:
            vx, vy = self._unpack_uv(v)
            ax, ay = apply_A(vx, vy)
            return self._pack_uv(ax, ay)

        r = m_inv(b - apply_A_vec(x))
        r_norm0 = self._vec_norm(r)
        tol = max(abs_tol, rel_tol * float(r_norm0.item()))
        if float(r_norm0.item()) <= tol:
            ux, uy = self._unpack_uv(x)
            ux, uy = self._apply_displacement_constraints(ux, uy)
            return ux, uy, True, 0, float(r_norm0.item())

        converged = False
        last_res = float(r_norm0.item())
        for _ in range(max_restarts):
            r = m_inv(b - apply_A_vec(x))
            beta = self._vec_norm(r)
            beta_val = float(beta.item())
            last_res = beta_val
            if beta_val <= tol:
                converged = True
                break

            v0 = r / max(beta_val, 1e-30)
            V: List[torch.Tensor] = [v0]
            H = torch.zeros((restart + 1, restart), device=x.device, dtype=x.dtype)
            k = 0
            for j in range(restart):
                w = m_inv(apply_A_vec(V[j]))
                for i in range(j + 1):
                    hij = self._vec_inner(w, V[i])
                    H[i, j] = hij
                    w = w - hij * V[i]
                h_next = self._vec_norm(w)
                H[j + 1, j] = h_next
                iters_done += 1
                if float(h_next.item()) > 1e-30 and (j + 1) < restart:
                    V.append(w / h_next)
                k = j + 1
                if iters_done >= n_iter_total:
                    break
            # 最小二乘：min || beta*e1 - H_k y ||
            Hk = H[: k + 1, :k]
            e1 = torch.zeros((k + 1,), device=x.device, dtype=x.dtype)
            e1[0] = beta
            y = torch.linalg.lstsq(Hk, e1).solution
            dx_vec = torch.zeros_like(x)
            for j in range(k):
                dx_vec = dx_vec + y[j] * V[j]
            x = x + dx_vec

            r_post = m_inv(b - apply_A_vec(x))
            r_post_norm = self._vec_norm(r_post)
            last_res = float(r_post_norm.item())
            if last_res <= tol:
                converged = True
                break
            if iters_done >= n_iter_total:
                break

        ux, uy = self._unpack_uv(x)
        ux, uy = self._apply_displacement_constraints(ux, uy)
        return ux, uy, converged, iters_done, last_res

    def _solve_quasi_static_krylov(
        self,
        state: Dict[str, torch.Tensor],
        epsp: Dict[str, torch.Tensor],
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, int, float]:
        """按配置选择 Krylov 变体。"""
        method = str(self.cfg.numerics.mechanics_krylov_method).strip().lower()
        if method in {"bicgstab", "bicg"}:
            return self._solve_quasi_static_bicgstab(
                state,
                epsp,
                dx_um,
                dy_um,
                x_coords_um=x_coords_um,
                y_coords_um=y_coords_um,
            )
        return self._solve_quasi_static_gmres(
            state,
            epsp,
            dx_um,
            dy_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )

    def _solve_quasi_static_relax(
        self,
        state: Dict[str, torch.Tensor],
        epsp: Dict[str, torch.Tensor],
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
        ux0: torch.Tensor | None = None,
        uy0: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """阻尼残差松弛（旧求解器，作为稳健回退）。"""
        ux = state["ux"] if ux0 is None else ux0
        uy = state["uy"] if uy0 is None else uy0
        eta = state["eta"]
        phi = state["phi"]
        relax = self.cfg.numerics.mechanics_relaxation
        stiff = max(float(self._stiffness_ref_mpa), 1e-9)
        h = max(min(dx_um, dy_um), 1e-9)
        damping = max(self.cfg.mechanics.residual_damping_MPa_um, 1e-9)
        max_u = self.cfg.mechanics.max_abs_displacement_um
        if max_u <= 0.0:
            if self._use_dirichlet_x():
                max_u = max(1.0, 2.0 * abs(self._dirichlet_right_ux()))
            else:
                max_u = max(1.0, 2.0 * abs(self.cfg.mechanics.external_strain_x) * self.cfg.domain.lx_um)

        n_iter_max = max(0, int(self.cfg.numerics.mechanics_substeps))
        adaptive = bool(self.cfg.numerics.mechanics_adaptive_substeps)
        n_iter_min = min(n_iter_max, max(0, int(self.cfg.numerics.mechanics_min_substeps)))
        abs_tol = max(0.0, float(self.cfg.numerics.mechanics_residual_tol))
        rel_tol = max(0.0, float(self.cfg.numerics.mechanics_residual_rel_tol))
        res0 = None

        for it in range(n_iter_max):
            eps = self._strain_from_displacement(
                ux,
                uy,
                eta,
                epsp,
                dx_um,
                dy_um,
                x_coords_um=x_coords_um,
                y_coords_um=y_coords_um,
            )
            sig = self.constitutive_stress(phi, eps["eps_xx"], eps["eps_yy"], eps["eps_xy"], eta=eta)
            div_x, div_y = self._stress_divergence(
                sig["sigma_xx"],
                sig["sigma_yy"],
                sig["sigma_xy"],
                dx_um,
                dy_um,
                x_coords_um=x_coords_um,
                y_coords_um=y_coords_um,
            )
            div_x = torch.nan_to_num(div_x, nan=0.0, posinf=0.0, neginf=0.0)
            div_y = torch.nan_to_num(div_y, nan=0.0, posinf=0.0, neginf=0.0)
            div_x = torch.clamp(div_x, min=-5e4, max=5e4)
            div_y = torch.clamp(div_y, min=-5e4, max=5e4)

            res = 0.5 * (torch.mean(torch.abs(div_x)) + torch.mean(torch.abs(div_y)))
            res_val = float(res.item())
            if res0 is None:
                res0 = max(res_val, 1e-12)

            if adaptive and (it + 1) >= n_iter_min:
                abs_ok = (abs_tol > 0.0) and (res_val <= abs_tol)
                rel_ok = (rel_tol > 0.0) and ((res_val / res0) <= rel_tol)
                if abs_ok or rel_ok:
                    break

            relax_eff = relax / (1.0 + float(res.item()) / damping)
            step = relax_eff * h * h / stiff
            ux = ux + step * div_x
            uy = uy + step * div_y
            ux = torch.clamp(ux, min=-max_u, max=max_u)
            uy = torch.clamp(uy, min=-max_u, max=max_u)
            ux, uy = self._apply_displacement_constraints(ux, uy)

        return ux, uy

    def solve_quasi_static(
        self,
        state: Dict[str, torch.Tensor],
        epsp: Dict[str, torch.Tensor],
        dx_um: float,
        dy_um: float,
        x_coords_um: torch.Tensor | None = None,
        y_coords_um: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """求解力学平衡并返回位移与应力。"""
        mode = str(self.cfg.numerics.mechanics_solver).strip().lower()
        cg_ok = False
        cg_attempted = False
        cg_iters = 0
        cg_res = 0.0

        if mode in {"hybrid_cg_relax", "hybrid"} and self._krylov_pause_left > 0:
            self._krylov_pause_left -= 1

        should_try_cg = mode == "cg" or (
            mode in {"hybrid_cg_relax", "hybrid"} and self._krylov_pause_left <= 0
        )
        if should_try_cg:
            cg_attempted = True
            ux, uy, cg_ok, cg_iters, cg_res = self._solve_quasi_static_krylov(
                state,
                epsp,
                dx_um,
                dy_um,
                x_coords_um=x_coords_um,
                y_coords_um=y_coords_um,
            )
            if cg_ok:
                self._krylov_fail_streak = 0
                self._krylov_pause_left = 0
            elif mode in {"hybrid_cg_relax", "hybrid"}:
                self._krylov_fail_streak += 1
                if self._krylov_fail_streak >= max(1, int(self.cfg.numerics.mechanics_krylov_fail_streak_to_pause)):
                    self._krylov_pause_left = max(1, int(self.cfg.numerics.mechanics_krylov_pause_steps))
                    self._krylov_fail_streak = 0
                ux, uy = self._solve_quasi_static_relax(
                    state,
                    epsp,
                    dx_um,
                    dy_um,
                    x_coords_um=x_coords_um,
                    y_coords_um=y_coords_um,
                )
        else:
            ux, uy = self._solve_quasi_static_relax(
                state,
                epsp,
                dx_um,
                dy_um,
                x_coords_um=x_coords_um,
                y_coords_um=y_coords_um,
            )
        ux, uy = self._apply_displacement_constraints(ux, uy)

        eta = state["eta"]
        phi = state["phi"]
        eps = self._strain_from_displacement(
            ux,
            uy,
            eta,
            epsp,
            dx_um,
            dy_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )
        sig = self.constitutive_stress(phi, eps["eps_xx"], eps["eps_yy"], eps["eps_xy"], eta=eta)
        sig = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in sig.items()}

        return {
            "ux": ux,
            "uy": uy,
            "eps_xx": eps["eps_xx"],
            "eps_yy": eps["eps_yy"],
            "eps_xy": eps["eps_xy"],
            "mech_cg_attempted": torch.tensor(float(cg_attempted), device=ux.device, dtype=ux.dtype),
            "mech_cg_converged": torch.tensor(float(cg_ok), device=ux.device, dtype=ux.dtype),
            "mech_cg_iters": torch.tensor(float(cg_iters), device=ux.device, dtype=ux.dtype),
            "mech_cg_residual": torch.tensor(float(cg_res), device=ux.device, dtype=ux.dtype),
            **sig,
        }
