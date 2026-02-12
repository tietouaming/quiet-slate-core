"""力学初值器测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from mg_coupled_pf.ml.mech_warmstart import (
    MECH_INPUT_ORDER,
    build_mechanics_warmstart,
    load_mechanics_warmstart,
    save_mechanics_warmstart,
)


def _state(dev: torch.device, h: int = 24, w: int = 32):
    out = {}
    for k in MECH_INPUT_ORDER:
        out[k] = torch.rand((1, 1, h, w), device=dev, dtype=torch.float32)
    return out


def test_mechanics_warmstart_build_predict_constraints() -> None:
    """预测结果应满足位移边界约束。"""
    dev = torch.device("cpu")
    p = build_mechanics_warmstart(
        device=dev,
        hidden=12,
        add_coord_features=True,
        use_torch_compile=False,
        channels_last=False,
    )
    x = _state(dev, h=20, w=28)
    ux, uy = p.predict(
        x,
        loading_mode="dirichlet_x",
        right_disp_um=0.25,
        enforce_anchor=True,
    )
    assert ux.shape == (1, 1, 20, 28)
    assert uy.shape == (1, 1, 20, 28)
    assert float(torch.max(torch.abs(ux[:, :, :, 0])).item()) < 1e-8
    assert float(torch.max(torch.abs(ux[:, :, :, -1] - 0.25)).item()) < 1e-8
    assert abs(float(uy[0, 0, 0, 0].item())) < 1e-8


def test_mechanics_warmstart_save_load_roundtrip(tmp_path: Path) -> None:
    """保存/加载后应可正常推理。"""
    dev = torch.device("cpu")
    p = build_mechanics_warmstart(
        device=dev,
        hidden=10,
        add_coord_features=False,
        use_torch_compile=False,
        channels_last=False,
    )
    path = tmp_path / "mech_warmstart.pt"
    save_mechanics_warmstart(p, path)
    p2 = load_mechanics_warmstart(
        path,
        device=dev,
        fallback_hidden=10,
        fallback_add_coord_features=False,
        use_torch_compile=False,
        channels_last=False,
    )
    x = _state(dev, h=16, w=18)
    ux, uy = p2.predict(x, loading_mode="eigenstrain", right_disp_um=0.0, enforce_anchor=True)
    assert torch.isfinite(ux).all()
    assert torch.isfinite(uy).all()
