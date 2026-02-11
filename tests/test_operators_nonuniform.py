"""非均匀网格离散算子精度测试（中文注释版）。"""

from __future__ import annotations

import math

import torch

from mg_coupled_pf.operators import grad_xy, laplacian


def _nonuniform_axis(n: int, power: float) -> torch.Tensor:
    """生成 [0,1] 上的严格递增非均匀网格坐标。"""
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    return torch.pow(t, power)


def _grad_error(n: int) -> float:
    """计算非均匀网格上一阶导误差（L2 均方根）。"""
    x = _nonuniform_axis(n, power=1.8)
    y = _nonuniform_axis(max(6, n // 8), power=1.2)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    # f(x,y)=sin(2πx)+x^2+y，解析导数 df/dx=2πcos(2πx)+2x。
    f = torch.sin(2.0 * math.pi * xx) + xx * xx + yy
    fx_true = 2.0 * math.pi * torch.cos(2.0 * math.pi * xx) + 2.0 * xx
    field = f[None, None]
    fx_num, _ = grad_xy(field, dx=1.0, dy=1.0, x_coords=x, y_coords=y)
    # 排除边界点，关注主区间收敛阶。
    err = fx_num[0, 0, 1:-1, 1:-1] - fx_true[1:-1, 1:-1]
    return float(torch.sqrt(torch.mean(err * err)).item())


def _lap_error(n: int) -> float:
    """计算非均匀网格上拉普拉斯误差（L2 均方根）。"""
    x = _nonuniform_axis(n, power=1.6)
    y = _nonuniform_axis(n, power=1.4)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    # f(x,y)=sin(πx)sin(πy)，解析 lap=-2π^2 sin(πx)sin(πy)。
    f = torch.sin(math.pi * xx) * torch.sin(math.pi * yy)
    lap_true = -2.0 * (math.pi**2) * f
    field = f[None, None]
    lap_num = laplacian(field, dx=1.0, dy=1.0, x_coords=x, y_coords=y)
    err = lap_num[0, 0, 2:-2, 2:-2] - lap_true[2:-2, 2:-2]
    return float(torch.sqrt(torch.mean(err * err)).item())


def test_nonuniform_grad_second_order_trend() -> None:
    """一阶导在非均匀网格上应表现出接近二阶收敛趋势。"""
    e1 = _grad_error(65)
    e2 = _grad_error(129)
    # 二阶理论比值约 4；考虑非均匀映射和边界截断，设置保守阈值。
    assert e1 > e2
    assert (e1 / max(e2, 1e-16)) > 2.4


def test_nonuniform_laplacian_second_order_trend() -> None:
    """拉普拉斯在非均匀网格上应表现出接近二阶收敛趋势。"""
    e1 = _lap_error(65)
    e2 = _lap_error(129)
    assert e1 > e2
    assert (e1 / max(e2, 1e-16)) > 2.2


def test_uniform_neumann_boundary_zero_normal_gradient() -> None:
    """均匀网格 Neumann 边界下，边界法向梯度应接近 0。"""
    torch.manual_seed(0)
    u = torch.rand((1, 1, 33, 49), dtype=torch.float64)
    gx, gy = grad_xy(u, dx=1.0, dy=1.0, bc="neumann")
    left = torch.max(torch.abs(gx[:, :, :, 0])).item()
    right = torch.max(torch.abs(gx[:, :, :, -1])).item()
    bottom = torch.max(torch.abs(gy[:, :, 0, :])).item()
    top = torch.max(torch.abs(gy[:, :, -1, :])).item()
    assert left < 1e-12
    assert right < 1e-12
    assert bottom < 1e-12
    assert top < 1e-12


def test_axis_wise_bc_tuple_and_dict_are_consistent() -> None:
    """按轴边界条件的 tuple 与 dict 写法应等价。"""
    torch.manual_seed(3)
    u = torch.rand((1, 1, 25, 31), dtype=torch.float64)
    gx1, gy1 = grad_xy(u, dx=1.0, dy=1.0, bc=("periodic", "neumann"))
    gx2, gy2 = grad_xy(u, dx=1.0, dy=1.0, bc={"x": "periodic", "y": "neumann"})
    assert torch.allclose(gx1, gx2, atol=1e-12, rtol=1e-12)
    assert torch.allclose(gy1, gy2, atol=1e-12, rtol=1e-12)


def test_dirichlet_side_values_are_respected_in_uniform_derivative() -> None:
    """均匀网格下 x 向左右 Dirichlet 值应分别进入导数离散。"""
    u = torch.zeros((1, 1, 6, 8), dtype=torch.float64)
    # 内部为线性场，边界分别固定到不同值。
    line = torch.linspace(0.0, 1.4, 8, dtype=torch.float64).view(1, 1, 1, -1)
    u[:] = line
    gx, _ = grad_xy(
        u,
        dx=1.0,
        dy=1.0,
        bc={"x": "dirichlet0", "y": "neumann"},
        dirichlet_value={"x_left": 0.0, "x_right": 1.4},
    )
    # 若右侧被错误当成 0，最右端导数会翻成负值；正确应保持非负。
    assert float(torch.mean(gx[:, :, :, -1]).item()) > 0.0
