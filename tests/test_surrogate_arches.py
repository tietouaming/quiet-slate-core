"""surrogate 架构测试（中文注释版）。"""

from __future__ import annotations

from pathlib import Path

import torch

from mg_coupled_pf.ml.surrogate import FIELD_ORDER, build_surrogate, load_surrogate, save_surrogate


def _state(dev: torch.device, h: int = 32, w: int = 48):
    """构造随机状态输入。"""
    out = {}
    for k in FIELD_ORDER:
        out[k] = torch.rand((1, 1, h, w), device=dev, dtype=torch.float32)
    return out


def test_build_predict_save_load_all_arches(tmp_path: Path) -> None:
    """验证所有代理架构可构建、推理、保存与加载。"""
    dev = torch.device("cpu")
    cases = [
        ("tiny_unet", {"hidden": 16}),
        ("dw_unet", {"dw_hidden": 16, "dw_depth": 2}),
        ("fno2d", {"fno_width": 16, "fno_modes_x": 8, "fno_modes_y": 6, "fno_depth": 2}),
        ("afno2d", {"afno_width": 16, "afno_modes_x": 8, "afno_modes_y": 6, "afno_depth": 2, "afno_expansion": 1.5}),
    ]
    x = _state(dev)

    for arch, kwargs in cases:
        p = build_surrogate(device=dev, use_torch_compile=False, channels_last=False, model_arch=arch, **kwargs)
        y = p.predict(x)
        assert set(y.keys()) == set(FIELD_ORDER)
        for k in FIELD_ORDER:
            assert y[k].shape == x[k].shape
            assert torch.isfinite(y[k]).all()

        save_path = tmp_path / f"{arch}.pt"
        save_surrogate(p, save_path)
        p2 = load_surrogate(save_path, dev, use_torch_compile=False, fallback_model_arch=arch, fallback_arch_kwargs=kwargs)
        y2 = p2.predict(x)
        for k in FIELD_ORDER:
            assert y2[k].shape == x[k].shape
            assert torch.isfinite(y2[k]).all()
