"""晶体学系统一致性测试。"""

from __future__ import annotations

import torch

from mg_coupled_pf import load_config
from mg_coupled_pf.mechanics import MechanicsModel


def _dot3(a, b) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def test_loaded_slip_and_twin_systems_are_orthonormal() -> None:
    """载入后的 slip/twin 系统应满足 |s|=|n|=1 且 s·n≈0。"""
    cfg = load_config("configs/notch_case.yaml")
    for e in cfg.slip_systems + cfg.twin_systems:
        s = e["s_crystal"]
        n = e["n_crystal"]
        ns = sum(v * v for v in s) ** 0.5
        nn = sum(v * v for v in n) ** 0.5
        dot = _dot3(s, n)
        assert abs(ns - 1.0) < 1e-7
        assert abs(nn - 1.0) < 1e-7
        assert abs(dot) < 1e-7


def test_twin_system_rotates_with_crystal_orientation() -> None:
    """孪晶系应随晶体取向变化，而不是固定实验室角度。"""
    cfg0 = load_config("configs/notch_case.yaml")
    cfg1 = load_config("configs/notch_case.yaml")
    cfg0.mechanics.crystal_orientation_euler_deg = [0.0, 0.0, 0.0]
    cfg1.mechanics.crystal_orientation_euler_deg = [35.0, 0.0, 0.0]
    m0 = MechanicsModel(cfg0)
    m1 = MechanicsModel(cfg1)
    v0 = torch.tensor([m0.sx, m0.sy], dtype=torch.float64)
    v1 = torch.tensor([m1.sx, m1.sy], dtype=torch.float64)
    # 旋转后方向必须发生可测变化。
    assert float(torch.linalg.norm(v0 - v1).item()) > 1e-3

