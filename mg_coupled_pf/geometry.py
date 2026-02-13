"""几何与初值构造模块（中文注释版）。

主要功能：
1. 生成二维均匀/局部加密网格；
2. 构造半空间、缺口等初始几何对应的相场初值；
3. 按文献形式生成点蚀迁移率空间分布场。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .config import SimulationConfig
from .operators import smooth_heaviside


@dataclass
class Grid2D:
    """二维网格信息容器。"""
    x_um: torch.Tensor
    y_um: torch.Tensor
    x_vec_um: torch.Tensor
    y_vec_um: torch.Tensor
    dx_um: float
    dy_um: float
    dx_min_um: float
    dy_min_um: float


def _enforce_strict_monotonic(axis: np.ndarray, length_um: float) -> np.ndarray:
    """修正坐标序列为严格递增，并缩放到目标长度。"""
    out = axis.copy()
    # 通过最小间距 eps 消除非单调/重复节点。
    eps = max(1e-9 * max(length_um, 1.0), 1e-12)
    out[0] = 0.0
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    scale = length_um / max(out[-1], eps)
    # 统一缩放到目标物理长度区间 [0, length_um]。
    out = out * scale
    out[0] = 0.0
    out[-1] = length_um
    return out


def _refined_axis(
    length_um: float,
    n: int,
    focus_um: float,
    sigma_um: float,
    strength: float,
    samples: int,
) -> np.ndarray:
    """按高斯权重在关注位置附近进行一维网格加密。"""
    if n <= 2:
        return np.linspace(0.0, length_um, max(n, 2), dtype=np.float64)
    if strength <= 0.0 or sigma_um <= 0.0 or samples < 32:
        return np.linspace(0.0, length_um, n, dtype=np.float64)
    focus = float(np.clip(focus_um, 0.0, length_um))
    # 在采样坐标上构造“密度函数”，focus 附近密度更高。
    x = np.linspace(0.0, length_um, int(samples), dtype=np.float64)
    rho = 1.0 + float(strength) * np.exp(-0.5 * ((x - focus) / max(float(sigma_um), 1e-9)) ** 2)
    # 将密度积分为累计分布，再做反插值得到非均匀节点。
    cum = np.zeros_like(x)
    cum[1:] = np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(x))
    if cum[-1] <= 0.0:
        return np.linspace(0.0, length_um, n, dtype=np.float64)
    cum /= cum[-1]
    target = np.linspace(0.0, 1.0, n, dtype=np.float64)
    axis = np.interp(target, cum, x)
    return _enforce_strict_monotonic(axis, length_um)


def _notch_focus(cfg: SimulationConfig) -> tuple[float, float]:
    """返回网格加密焦点（优先取半空间三角缺口尖端）。"""
    if cfg.domain.initial_geometry.lower() == "half_space" and cfg.domain.add_half_space_triangular_notch:
        return float(cfg.domain.half_space_notch_tip_x_um), float(cfg.domain.half_space_notch_center_y_um)
    return float(cfg.domain.notch_tip_x_um), float(cfg.domain.notch_center_y_um)


def _interface_width_um(cfg: SimulationConfig) -> float:
    """返回统一界面厚度（优先 corrosion.interface_thickness_um，其次 domain.interface_width_um）。"""
    ell = float(getattr(cfg.corrosion, "interface_thickness_um", -1.0))
    if ell > 0.0:
        return ell
    return float(cfg.domain.interface_width_um)


def make_grid(cfg: SimulationConfig, device: torch.device, dtype: torch.dtype) -> Grid2D:
    """根据配置创建网格坐标与步长。"""
    nx = cfg.domain.nx
    ny = cfg.domain.ny
    if cfg.domain.mesh_refine_near_notch:
        # 缺口附近加密：x/y 分别按高斯分布重排节点。
        fx, fy = _notch_focus(cfg)
        x_np = _refined_axis(
            cfg.domain.lx_um,
            nx,
            focus_um=fx,
            sigma_um=cfg.domain.mesh_refine_sigma_x_um,
            strength=cfg.domain.mesh_refine_strength_x,
            samples=cfg.domain.mesh_refine_samples,
        )
        y_np = _refined_axis(
            cfg.domain.ly_um,
            ny,
            focus_um=fy,
            sigma_um=cfg.domain.mesh_refine_sigma_y_um,
            strength=cfg.domain.mesh_refine_strength_y,
            samples=cfg.domain.mesh_refine_samples,
        )
        x = torch.from_numpy(x_np).to(device=device, dtype=dtype)
        y = torch.from_numpy(y_np).to(device=device, dtype=dtype)
    else:
        x = torch.linspace(0.0, cfg.domain.lx_um, nx, device=device, dtype=dtype)
        y = torch.linspace(0.0, cfg.domain.ly_um, ny, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    # dx,dy 为平均步长；dx_min,dy_min 用于稳定性估计（更保守）。
    dx = float((x[-1] - x[0]).item() / max(nx - 1, 1))
    dy = float((y[-1] - y[0]).item() / max(ny - 1, 1))
    dx_min = float(torch.min(x[1:] - x[:-1]).item()) if nx > 1 else dx
    dy_min = float(torch.min(y[1:] - y[:-1]).item()) if ny > 1 else dy
    return Grid2D(
        x_um=xx[None, None],
        y_um=yy[None, None],
        x_vec_um=x,
        y_vec_um=y,
        dx_um=dx,
        dy_um=dy,
        dx_min_um=dx_min,
        dy_min_um=dy_min,
    )


def smooth_notch_profile(
    x_um: torch.Tensor,
    y_um: torch.Tensor,
    tip_x: float,
    center_y: float,
    depth: float,
    half_opening: float,
    interface: float,
) -> torch.Tensor:
    """构造向左开口 V 形缺口的平滑掩模。"""
    # relx<0 表示位于缺口“向左张开”的范围内。
    relx = x_um - tip_x
    relx_clamped = torch.clamp(-relx / max(depth, 1e-9), min=0.0, max=1.0)
    local_half_width = half_opening * relx_clamped
    signed_dist = torch.abs(y_um - center_y) - local_half_width
    notch_inside = 0.5 * (1.0 - torch.tanh(signed_dist / max(interface, 1e-9)))
    notch_x_gate = 0.5 * (1.0 - torch.tanh(relx / max(interface, 1e-9)))
    # y 方向“在三角内”与 x 方向“在深度范围内”两者相乘。
    return notch_inside * notch_x_gate


def triangular_notch_profile(
    x_um: torch.Tensor,
    y_um: torch.Tensor,
    tip_x: float,
    center_y: float,
    depth: float,
    half_opening: float,
    interface: float,
    open_to_positive_x: bool = True,
) -> torch.Tensor:
    """构造三角形缺口的平滑掩模。"""
    # direction 控制开口方向：+1 向 +x 张开，-1 向 -x 张开。
    direction = 1.0 if open_to_positive_x else -1.0
    rel = direction * (x_um - tip_x)
    rel01 = torch.clamp(rel / max(depth, 1e-9), min=0.0, max=1.0)
    local_half = half_opening * rel01
    signed = torch.abs(y_um - center_y) - local_half
    inside_y = 0.5 * (1.0 - torch.tanh(signed / max(interface, 1e-9)))
    gate_lo = 0.5 * (1.0 + torch.tanh(rel / max(interface, 1e-9)))
    gate_hi = 0.5 * (1.0 - torch.tanh((rel - depth) / max(interface, 1e-9)))
    # 返回 [0,1] 掩模，1 代表缺口内部。
    return inside_y * gate_lo * gate_hi


def half_space_profile(cfg: SimulationConfig, grid: Grid2D) -> torch.Tensor:
    """构造“半固体-半液体”初始相场分布。"""
    interface = max(_interface_width_um(cfg), 1e-9)
    direction = cfg.domain.half_space_direction.lower()
    if direction == "y":
        y0 = cfg.domain.half_space_interface_y_um
        if y0 < 0.0:
            y0 = 0.5 * cfg.domain.ly_um
        signed = (grid.y_um - y0) / interface
    else:
        x0 = cfg.domain.half_space_interface_x_um
        if x0 < 0.0:
            x0 = 0.5 * cfg.domain.lx_um
        signed = (grid.x_um - x0) / interface
    if cfg.domain.half_space_solid_on_lower_side:
        # lower side: x<x0 or y<y0 is solid.
        return 0.5 * (1.0 - torch.tanh(signed))
    return 0.5 * (1.0 + torch.tanh(signed))


def initial_fields(cfg: SimulationConfig, grid: Grid2D) -> dict[str, torch.Tensor]:
    """生成所有主场变量初值。"""
    mode = cfg.domain.initial_geometry.lower()
    if mode == "half_space":
        # 半空间模式：固液由一条平滑界面分隔。
        phi = torch.clamp(half_space_profile(cfg, grid), 0.0, 1.0)
        if cfg.domain.add_half_space_triangular_notch:
            # 在半空间固相中切出一个三角尖锐缺口。
            notch = triangular_notch_profile(
                grid.x_um,
                grid.y_um,
                tip_x=cfg.domain.half_space_notch_tip_x_um,
                center_y=cfg.domain.half_space_notch_center_y_um,
                depth=cfg.domain.half_space_notch_depth_um,
                half_opening=cfg.domain.half_space_notch_half_opening_um,
                interface=cfg.domain.half_space_notch_sharpness_um,
                open_to_positive_x=cfg.domain.half_space_notch_open_to_positive_x,
            )
            solid = smooth_heaviside(phi)
            # 仅在固相侧应用缺口扣除，避免误改液相。
            phi = torch.clamp(phi * (1.0 - notch * solid), 0.0, 1.0)
    else:
        # 缺口模式：初始全固体，再切出缺口与可选点蚀种子。
        phi = torch.ones_like(grid.x_um)
        notch = smooth_notch_profile(
            grid.x_um,
            grid.y_um,
            tip_x=cfg.domain.notch_tip_x_um,
            center_y=cfg.domain.notch_center_y_um,
            depth=cfg.domain.notch_depth_um,
            half_opening=cfg.domain.notch_half_opening_um,
            interface=_interface_width_um(cfg),
        )
        phi = phi * (1.0 - notch)

        if cfg.domain.add_initial_pit_seed:
            r2 = (grid.x_um - cfg.domain.notch_tip_x_um) ** 2 + (grid.y_um - cfg.domain.notch_center_y_um) ** 2
            sigma = float(getattr(cfg.domain, "initial_pit_sigma_um", -1.0))
            if sigma <= 0.0:
                # 兼容旧参数：旧公式 exp(-r^2/R^2) 等价于新公式中 sigma=R/sqrt(2)。
                sigma = max(float(cfg.domain.initial_pit_radius_um) / math.sqrt(2.0), 1e-9)
            pit = torch.exp(-0.5 * r2 / max(sigma * sigma, 1e-12))
            phi = torch.clamp(phi - 0.35 * pit, 0.0, 1.0)

    hphi = smooth_heaviside(phi)
    # cMg 初值按固液插值：液相0、固相1（可由配置覆盖）。
    c = cfg.corrosion.cMg_init_liquid + (cfg.corrosion.cMg_init_solid - cfg.corrosion.cMg_init_liquid) * hphi
    c = torch.clamp(c, min=cfg.corrosion.cMg_min, max=cfg.corrosion.cMg_max)
    eta = torch.zeros_like(phi)
    ux = torch.zeros_like(phi)
    uy = torch.zeros_like(phi)
    epspeq = torch.zeros_like(phi)
    # 为 surrogate 与训练数据提供完整塑性张量通道（与 epspeq 并存）。
    epsp_xx = torch.zeros_like(phi)
    epsp_yy = torch.zeros_like(phi)
    epsp_xy = torch.zeros_like(phi)
    return {
        "phi": phi,
        "c": c,
        "eta": eta,
        "ux": ux,
        "uy": uy,
        "epspeq": epspeq,
        "epsp_xx": epsp_xx,
        "epsp_yy": epsp_yy,
        "epsp_xy": epsp_xy,
    }


def pitting_mobility_field(cfg: SimulationConfig, grid: Grid2D) -> torch.Tensor:
    """生成点蚀迁移率空间场（Kovacevic 风格傅里叶随机叠加）。"""
    rng = np.random.default_rng(cfg.numerics.seed)
    x = (grid.x_um[0, 0].detach().cpu().numpy() / cfg.domain.lx_um).astype(np.float64)
    y = (grid.y_um[0, 0].detach().cpu().numpy() / cfg.domain.ly_um).astype(np.float64)
    nmax = int(max(1, cfg.corrosion.pitting_N))
    beta = float(cfg.corrosion.pitting_beta)
    alpha = float(cfg.corrosion.pitting_alpha)

    field = np.zeros_like(x, dtype=np.float64)
    # 双重傅里叶模态叠加：幅值随机、相位随机，衰减指数由 beta 控制。
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            if m == 0 and n == 0:
                continue
            amp = rng.uniform(0.0, 1.0)
            phase = rng.uniform(-alpha * math.pi / 2.0, alpha * math.pi / 2.0)
            denom = (m * m + n * n) ** (beta / 2.0)
            field += amp / max(denom, 1e-8) * np.cos(2.0 * math.pi * (m * x + n * y) + phase)

    field -= field.min()
    if field.max() > 1e-12:
        field /= field.max()
    # 归一化后再线性映射到 [min_factor, max_factor]。
    lo, hi = cfg.corrosion.pitting_min_factor, cfg.corrosion.pitting_max_factor
    field = lo + (hi - lo) * field
    out = torch.from_numpy(field).to(grid.x_um.device, dtype=grid.x_um.dtype)[None, None]
    return out
