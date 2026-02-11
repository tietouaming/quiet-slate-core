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


def _normalize_bc_name(name: str | None, fallback: str = "neumann") -> str:
    """归一化边界条件名称。"""
    if name is None:
        name = fallback
    b = str(name).strip().lower()
    aliases = {
        "zero_flux": "neumann",
        "zero-gradient": "neumann",
        "zero_gradient": "neumann",
        "circular": "periodic",
        "fixed0": "dirichlet0",
        "dirichlet": "dirichlet0",
    }
    return aliases.get(b, b)


def _split_bc(bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann") -> tuple[str, str]:
    """将边界配置拆分为 `(bc_x, bc_y)`。"""
    if isinstance(bc, dict):
        bx = _normalize_bc_name(bc.get("x", bc.get("bx", "neumann")))
        by = _normalize_bc_name(bc.get("y", bc.get("by", "neumann")))
        return bx, by
    if isinstance(bc, (tuple, list)):
        if len(bc) != 2:
            raise ValueError(f"Axis-wise bc expects length 2, got {len(bc)}.")
        return _normalize_bc_name(bc[0]), _normalize_bc_name(bc[1])
    b = str(bc).strip()
    if "," in b:
        parts = [p.strip() for p in b.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Cannot parse axis-wise bc string: {bc}")
        return _normalize_bc_name(parts[0]), _normalize_bc_name(parts[1])
    bn = _normalize_bc_name(b)
    return bn, bn


def _split_dirichlet_value(
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None,
    axis: str,
) -> tuple[float, float]:
    """解析按轴/按侧 Dirichlet 值。"""
    if dirichlet_value is None:
        return 0.0, 0.0
    if isinstance(dirichlet_value, dict):
        if axis == "x":
            left = dirichlet_value.get("x_left", dirichlet_value.get("left", dirichlet_value.get("xmin", 0.0)))
            right = dirichlet_value.get("x_right", dirichlet_value.get("right", dirichlet_value.get("xmax", left)))
            return float(left), float(right)
        bottom = dirichlet_value.get("y_bottom", dirichlet_value.get("bottom", dirichlet_value.get("ymin", 0.0)))
        top = dirichlet_value.get("y_top", dirichlet_value.get("top", dirichlet_value.get("ymax", bottom)))
        return float(bottom), float(top)
    if isinstance(dirichlet_value, (tuple, list)):
        if len(dirichlet_value) != 2:
            raise ValueError(f"dirichlet_value sequence expects length 2, got {len(dirichlet_value)}")
        return float(dirichlet_value[0]), float(dirichlet_value[1])
    v = float(dirichlet_value)
    return v, v


def _pad_x(
    x: torch.Tensor,
    bc_x: str,
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """沿 x 方向添加 1 层 ghost cell。"""
    b = _normalize_bc_name(bc_x)
    if b == "neumann":
        if x.shape[-1] < 2:
            left = x[:, :, :, :1]
            right = x[:, :, :, -1:]
        else:
            left = x[:, :, :, 1:2]
            right = x[:, :, :, -2:-1]
    elif b == "periodic":
        left = x[:, :, :, -1:]
        right = x[:, :, :, :1]
    elif b == "dirichlet0":
        left_v, right_v = _split_dirichlet_value(dirichlet_value, axis="x")
        left = torch.full_like(x[:, :, :, :1], float(left_v))
        right = torch.full_like(x[:, :, :, :1], float(right_v))
    else:
        raise ValueError(f"Unsupported x boundary condition: {bc_x}")
    return torch.cat([left, x, right], dim=3)


def _pad_y(
    x: torch.Tensor,
    bc_y: str,
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """沿 y 方向添加 1 层 ghost cell。"""
    b = _normalize_bc_name(bc_y)
    if b == "neumann":
        if x.shape[-2] < 2:
            bottom = x[:, :, :1, :]
            top = x[:, :, -1:, :]
        else:
            bottom = x[:, :, 1:2, :]
            top = x[:, :, -2:-1, :]
    elif b == "periodic":
        bottom = x[:, :, -1:, :]
        top = x[:, :, :1, :]
    elif b == "dirichlet0":
        bottom_v, top_v = _split_dirichlet_value(dirichlet_value, axis="y")
        bottom = torch.full_like(x[:, :, :1, :], float(bottom_v))
        top = torch.full_like(x[:, :, :1, :], float(top_v))
    else:
        raise ValueError(f"Unsupported y boundary condition: {bc_y}")
    return torch.cat([bottom, x, top], dim=2)


def _pad_with_bc(
    x: torch.Tensor,
    bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann",
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """按边界条件填充 ghost 区域（可按轴分别指定）。"""
    bx, by = _split_bc(bc)
    return _pad_x(_pad_y(x, by, dirichlet_value), bx, dirichlet_value)


def _deriv_x(
    x: torch.Tensor,
    dx: float,
    x_coords: torch.Tensor | None = None,
    *,
    bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann",
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """计算 x 方向一阶导数；可选非均匀坐标。"""
    bx, by = _split_bc(bc)
    if x_coords is None:
        # 均匀网格：标准中心差分。
        xp = _pad_with_bc(x, bc=(bx, by), dirichlet_value=dirichlet_value)
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
    # 非均匀网格边界若采用零通量，直接将边界法向梯度设为 0，避免伪通量泄漏。
    if bx == "neumann":
        out[:, :, :, 0] = 0.0
        out[:, :, :, -1] = 0.0
    return out


def _deriv_y(
    x: torch.Tensor,
    dy: float,
    y_coords: torch.Tensor | None = None,
    *,
    bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann",
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """计算 y 方向一阶导数；可选非均匀坐标。"""
    bx, by = _split_bc(bc)
    if y_coords is None:
        # 均匀网格：标准中心差分。
        xp = _pad_with_bc(x, bc=(bx, by), dirichlet_value=dirichlet_value)
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
    if by == "neumann":
        out[:, :, 0, :] = 0.0
        out[:, :, -1, :] = 0.0
    return out


def _second_deriv_x(
    x: torch.Tensor,
    dx: float,
    x_coords: torch.Tensor | None = None,
    *,
    bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann",
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """计算 x 方向二阶导数；非均匀网格采用三点二阶公式。"""
    if x_coords is None:
        xp = _pad_with_bc(x, bc=bc, dirichlet_value=dirichlet_value)
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


def _second_deriv_y(
    x: torch.Tensor,
    dy: float,
    y_coords: torch.Tensor | None = None,
    *,
    bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann",
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """计算 y 方向二阶导数；非均匀网格采用三点二阶公式。"""
    if y_coords is None:
        xp = _pad_with_bc(x, bc=bc, dirichlet_value=dirichlet_value)
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
    bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann",
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回二维梯度 `(∂x, ∂y)`。"""
    return (
        _deriv_x(x, dx, x_coords, bc=bc, dirichlet_value=dirichlet_value),
        _deriv_y(x, dy, y_coords, bc=bc, dirichlet_value=dirichlet_value),
    )


def divergence(
    gx: torch.Tensor,
    gy: torch.Tensor,
    dx: float,
    dy: float,
    *,
    x_coords: torch.Tensor | None = None,
    y_coords: torch.Tensor | None = None,
    bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann",
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """返回散度 `∂x(gx) + ∂y(gy)`。"""
    return _deriv_x(gx, dx, x_coords, bc=bc, dirichlet_value=dirichlet_value) + _deriv_y(
        gy, dy, y_coords, bc=bc, dirichlet_value=dirichlet_value
    )


def laplacian(
    x: torch.Tensor,
    dx: float,
    dy: float,
    *,
    x_coords: torch.Tensor | None = None,
    y_coords: torch.Tensor | None = None,
    bc: str | tuple[str, str] | list[str] | dict[str, str] = "neumann",
    dirichlet_value: float | tuple[float, float] | list[float] | dict[str, float] | None = 0.0,
) -> torch.Tensor:
    """返回拉普拉斯 `∇²x`。"""
    dxx = _second_deriv_x(x, dx, x_coords=x_coords, bc=bc, dirichlet_value=dirichlet_value)
    dyy = _second_deriv_y(x, dy, y_coords=y_coords, bc=bc, dirichlet_value=dirichlet_value)
    return dxx + dyy


def smooth_heaviside(phi: torch.Tensor, *, clamp_input: bool = True) -> torch.Tensor:
    """平滑 Heaviside 插值函数（五次多项式）。"""
    # 6p^5 - 15p^4 + 10p^3，保证 0/1 两端一阶导连续为 0。
    p = torch.clamp(phi, 0.0, 1.0) if clamp_input else phi
    return p * p * p * (10.0 - 15.0 * p + 6.0 * p * p)


def solid_indicator(phi: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """固相硬阈值指示函数。"""
    return (phi >= threshold).to(dtype=phi.dtype)


def smooth_heaviside_prime(phi: torch.Tensor, *, clamp_input: bool = True) -> torch.Tensor:
    """平滑 Heaviside 的导数。"""
    # 对五次插值函数求导后的显式表达式。
    p = torch.clamp(phi, 0.0, 1.0) if clamp_input else phi
    return 30.0 * p * p * (p - 1.0) * (p - 1.0)


def double_well(phi: torch.Tensor, *, clamp_input: bool = True) -> torch.Tensor:
    """双稳势函数 `16*phi^2*(1-phi)^2`。"""
    p = torch.clamp(phi, 0.0, 1.0) if clamp_input else phi
    return 16.0 * p * p * (1.0 - p) * (1.0 - p)


def double_well_prime(phi: torch.Tensor, *, clamp_input: bool = True) -> torch.Tensor:
    """双稳势对 phi 的导数。"""
    p = torch.clamp(phi, 0.0, 1.0) if clamp_input else phi
    return 32.0 * p * (1.0 - p) * (1.0 - 2.0 * p)
