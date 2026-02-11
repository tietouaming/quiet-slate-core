"""二维数值算子模块（中文注释版）。

本模块提供相场-力学耦合求解中的基础离散算子：
- 梯度、散度、拉普拉斯
- 平滑 Heaviside 及其导数
- 双稳势函数及导数
- 固相指示函数

所有场变量采用 `[B, C, H, W]` 张量布局，并兼容均匀/非均匀网格。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pad_rep(x: torch.Tensor) -> torch.Tensor:
    """复制边界填充，便于中心差分在边缘点稳定计算。"""
    return F.pad(x, (1, 1, 1, 1), mode="replicate")


def _deriv_x(x: torch.Tensor, dx: float, x_coords: torch.Tensor | None = None) -> torch.Tensor:
    """计算 x 方向一阶导数；可选非均匀坐标。"""
    if x_coords is None:
        # 均匀网格：标准中心差分。
        xp = _pad_rep(x)
        return (xp[:, :, 1:-1, 2:] - xp[:, :, 1:-1, :-2]) / (2.0 * dx)

    # 非均匀网格：三点二阶格式（内点）+ 一侧二阶格式（边界）。
    eps = 1e-12
    w = x.shape[-1]
    out = torch.zeros_like(x)
    if w < 3:
        dx0 = torch.clamp(x_coords[min(1, w - 1)] - x_coords[0], min=eps)
        if w > 1:
            out[:, :, :, 0] = (x[:, :, :, 1] - x[:, :, :, 0]) / dx0
            out[:, :, :, -1] = out[:, :, :, 0]
        return out

    h1 = torch.clamp(x_coords[1:-1] - x_coords[:-2], min=eps).view(1, 1, 1, -1)
    h2 = torch.clamp(x_coords[2:] - x_coords[1:-1], min=eps).view(1, 1, 1, -1)
    c_im1 = -h2 / (h1 * (h1 + h2))
    c_i = (h2 - h1) / (h1 * h2)
    c_ip1 = h1 / (h2 * (h1 + h2))
    out[:, :, :, 1:-1] = (
        c_im1 * x[:, :, :, :-2]
        + c_i * x[:, :, :, 1:-1]
        + c_ip1 * x[:, :, :, 2:]
    )

    h10 = torch.clamp(x_coords[1] - x_coords[0], min=eps)
    h20 = torch.clamp(x_coords[2] - x_coords[1], min=eps)
    c0 = -(2.0 * h10 + h20) / (h10 * (h10 + h20))
    c1 = (h10 + h20) / (h10 * h20)
    c2 = -h10 / (h20 * (h10 + h20))
    out[:, :, :, 0] = c0 * x[:, :, :, 0] + c1 * x[:, :, :, 1] + c2 * x[:, :, :, 2]

    h1n = torch.clamp(x_coords[-2] - x_coords[-3], min=eps)
    h2n = torch.clamp(x_coords[-1] - x_coords[-2], min=eps)
    cm3 = h2n / (h1n * (h1n + h2n))
    cm2 = -(h1n + h2n) / (h1n * h2n)
    cm1 = (2.0 * h2n + h1n) / (h2n * (h1n + h2n))
    out[:, :, :, -1] = cm3 * x[:, :, :, -3] + cm2 * x[:, :, :, -2] + cm1 * x[:, :, :, -1]
    return out


def _deriv_y(x: torch.Tensor, dy: float, y_coords: torch.Tensor | None = None) -> torch.Tensor:
    """计算 y 方向一阶导数；可选非均匀坐标。"""
    if y_coords is None:
        # 均匀网格：标准中心差分。
        xp = _pad_rep(x)
        return (xp[:, :, 2:, 1:-1] - xp[:, :, :-2, 1:-1]) / (2.0 * dy)

    # 非均匀网格：三点二阶格式（内点）+ 一侧二阶格式（边界）。
    eps = 1e-12
    h = x.shape[-2]
    out = torch.zeros_like(x)
    if h < 3:
        dy0 = torch.clamp(y_coords[min(1, h - 1)] - y_coords[0], min=eps)
        if h > 1:
            out[:, :, 0, :] = (x[:, :, 1, :] - x[:, :, 0, :]) / dy0
            out[:, :, -1, :] = out[:, :, 0, :]
        return out

    h1 = torch.clamp(y_coords[1:-1] - y_coords[:-2], min=eps).view(1, 1, -1, 1)
    h2 = torch.clamp(y_coords[2:] - y_coords[1:-1], min=eps).view(1, 1, -1, 1)
    c_im1 = -h2 / (h1 * (h1 + h2))
    c_i = (h2 - h1) / (h1 * h2)
    c_ip1 = h1 / (h2 * (h1 + h2))
    out[:, :, 1:-1, :] = (
        c_im1 * x[:, :, :-2, :]
        + c_i * x[:, :, 1:-1, :]
        + c_ip1 * x[:, :, 2:, :]
    )

    h10 = torch.clamp(y_coords[1] - y_coords[0], min=eps)
    h20 = torch.clamp(y_coords[2] - y_coords[1], min=eps)
    c0 = -(2.0 * h10 + h20) / (h10 * (h10 + h20))
    c1 = (h10 + h20) / (h10 * h20)
    c2 = -h10 / (h20 * (h10 + h20))
    out[:, :, 0, :] = c0 * x[:, :, 0, :] + c1 * x[:, :, 1, :] + c2 * x[:, :, 2, :]

    h1n = torch.clamp(y_coords[-2] - y_coords[-3], min=eps)
    h2n = torch.clamp(y_coords[-1] - y_coords[-2], min=eps)
    cm3 = h2n / (h1n * (h1n + h2n))
    cm2 = -(h1n + h2n) / (h1n * h2n)
    cm1 = (2.0 * h2n + h1n) / (h2n * (h1n + h2n))
    out[:, :, -1, :] = cm3 * x[:, :, -3, :] + cm2 * x[:, :, -2, :] + cm1 * x[:, :, -1, :]
    return out


def _second_deriv_x(x: torch.Tensor, dx: float, x_coords: torch.Tensor | None = None) -> torch.Tensor:
    """计算 x 方向二阶导数；非均匀网格采用三点二阶公式。"""
    if x_coords is None:
        xp = _pad_rep(x)
        return (xp[:, :, 1:-1, 2:] - 2.0 * xp[:, :, 1:-1, 1:-1] + xp[:, :, 1:-1, :-2]) / (dx * dx)

    eps = 1e-12
    w = x.shape[-1]
    out = torch.zeros_like(x)
    if w < 3:
        return out
    h1 = torch.clamp(x_coords[1:-1] - x_coords[:-2], min=eps).view(1, 1, 1, -1)
    h2 = torch.clamp(x_coords[2:] - x_coords[1:-1], min=eps).view(1, 1, 1, -1)
    out[:, :, :, 1:-1] = 2.0 * (
        x[:, :, :, :-2] / (h1 * (h1 + h2))
        - x[:, :, :, 1:-1] / (h1 * h2)
        + x[:, :, :, 2:] / (h2 * (h1 + h2))
    )
    # 边界采用邻近内点外推，匹配零通量边界常见设定。
    out[:, :, :, 0] = out[:, :, :, 1]
    out[:, :, :, -1] = out[:, :, :, -2]
    return out


def _second_deriv_y(x: torch.Tensor, dy: float, y_coords: torch.Tensor | None = None) -> torch.Tensor:
    """计算 y 方向二阶导数；非均匀网格采用三点二阶公式。"""
    if y_coords is None:
        xp = _pad_rep(x)
        return (xp[:, :, 2:, 1:-1] - 2.0 * xp[:, :, 1:-1, 1:-1] + xp[:, :, :-2, 1:-1]) / (dy * dy)

    eps = 1e-12
    h = x.shape[-2]
    out = torch.zeros_like(x)
    if h < 3:
        return out
    h1 = torch.clamp(y_coords[1:-1] - y_coords[:-2], min=eps).view(1, 1, -1, 1)
    h2 = torch.clamp(y_coords[2:] - y_coords[1:-1], min=eps).view(1, 1, -1, 1)
    out[:, :, 1:-1, :] = 2.0 * (
        x[:, :, :-2, :] / (h1 * (h1 + h2))
        - x[:, :, 1:-1, :] / (h1 * h2)
        + x[:, :, 2:, :] / (h2 * (h1 + h2))
    )
    out[:, :, 0, :] = out[:, :, 1, :]
    out[:, :, -1, :] = out[:, :, -2, :]
    return out


def grad_xy(
    x: torch.Tensor,
    dx: float,
    dy: float,
    *,
    x_coords: torch.Tensor | None = None,
    y_coords: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回二维梯度 `(∂x, ∂y)`。"""
    return _deriv_x(x, dx, x_coords), _deriv_y(x, dy, y_coords)


def divergence(
    gx: torch.Tensor,
    gy: torch.Tensor,
    dx: float,
    dy: float,
    *,
    x_coords: torch.Tensor | None = None,
    y_coords: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回散度 `∂x(gx) + ∂y(gy)`。"""
    return _deriv_x(gx, dx, x_coords) + _deriv_y(gy, dy, y_coords)


def laplacian(
    x: torch.Tensor,
    dx: float,
    dy: float,
    *,
    x_coords: torch.Tensor | None = None,
    y_coords: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回拉普拉斯 `∇²x`。"""
    dxx = _second_deriv_x(x, dx, x_coords=x_coords)
    dyy = _second_deriv_y(x, dy, y_coords=y_coords)
    return dxx + dyy


def smooth_heaviside(phi: torch.Tensor) -> torch.Tensor:
    """平滑 Heaviside 插值函数（五次多项式）。"""
    # 6p^5 - 15p^4 + 10p^3，保证 0/1 两端一阶导连续为 0。
    p = torch.clamp(phi, 0.0, 1.0)
    return p * p * p * (10.0 - 15.0 * p + 6.0 * p * p)


def solid_indicator(phi: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """固相硬阈值指示函数。"""
    return (phi >= threshold).to(dtype=phi.dtype)


def smooth_heaviside_prime(phi: torch.Tensor) -> torch.Tensor:
    """平滑 Heaviside 的导数。"""
    # 对五次插值函数求导后的显式表达式。
    p = torch.clamp(phi, 0.0, 1.0)
    return 30.0 * p * p * (p - 1.0) * (p - 1.0)


def double_well(phi: torch.Tensor) -> torch.Tensor:
    """双稳势函数 `16*phi^2*(1-phi)^2`。"""
    p = torch.clamp(phi, 0.0, 1.0)
    return 16.0 * p * p * (1.0 - p) * (1.0 - p)


def double_well_prime(phi: torch.Tensor) -> torch.Tensor:
    """双稳势对 phi 的导数。"""
    p = torch.clamp(phi, 0.0, 1.0)
    return 32.0 * p * (1.0 - p) * (1.0 - 2.0 * p)
