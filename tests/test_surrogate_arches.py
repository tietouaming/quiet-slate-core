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


def test_predict_compatible_with_legacy_state_without_epsp_tensor_channels() -> None:
    """验证 surrogate 对旧版 state（无 epsp_xx/yy/xy）仍可推理。"""
    dev = torch.device("cpu")
    p = build_surrogate(device=dev, use_torch_compile=False, channels_last=False, model_arch="tiny_unet", hidden=12)
    x = _state(dev, h=16, w=20)
    x.pop("epsp_xx")
    x.pop("epsp_yy")
    x.pop("epsp_xy")
    y = p.predict(x)
    for k in FIELD_ORDER:
        assert k in y
        assert y[k].shape == (1, 1, 16, 20)


def test_predict_applies_mechanics_boundary_projection() -> None:
    """验证 surrogate 可将位移输出硬投影回力学边界约束。"""
    dev = torch.device("cpu")
    p = build_surrogate(
        device=dev,
        use_torch_compile=False,
        channels_last=False,
        model_arch="tiny_unet",
        hidden=12,
        add_coord_features=True,
    )
    p.enforce_displacement_constraints = True
    p.loading_mode = "dirichlet_x"
    p.dirichlet_right_ux = 0.123
    p.enforce_uy_anchor = True
    x = _state(dev, h=18, w=22)
    y = p.predict(x)
    assert float(torch.max(torch.abs(y["ux"][:, :, :, 0])).item()) < 1e-8
    assert float(torch.max(torch.abs(y["ux"][:, :, :, -1] - 0.123)).item()) < 1e-8
    assert abs(float(y["uy"][0, 0, 0, 0].item())) < 1e-8


def test_surrogate_output_mode_delta_vs_absolute() -> None:
    """surrogate 输出模式应支持 delta 与 absolute 两种语义。"""
    dev = torch.device("cpu")
    p = build_surrogate(
        device=dev,
        use_torch_compile=False,
        channels_last=False,
        model_arch="tiny_unet",
        hidden=12,
    )
    x = _state(dev, h=16, w=20)
    p.output_mode = "delta"
    y_delta = p.predict(x)
    p.output_mode = "absolute"
    y_abs = p.predict(x)
    # 同一模型下，两种输出语义应产生可测差异。
    diff = float(torch.mean(torch.abs(y_delta["phi"] - y_abs["phi"])).item())
    assert diff > 1e-5
