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

