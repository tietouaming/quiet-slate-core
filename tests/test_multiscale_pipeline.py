"""跨尺度微结构->宏观预测与反向设计流程测试。"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch

from mg_coupled_pf import load_config
from mg_coupled_pf.multiscale.dataset import (
    MacroTargetSpec,
    MultiScaleDatasetConfig,
    build_multiscale_dataset_from_cases,
    save_multiscale_dataset_npz,
    load_multiscale_dataset_npz,
)
from mg_coupled_pf.multiscale.features import extract_micro_descriptors, extract_micro_tensor
from mg_coupled_pf.multiscale.models import MultiScaleModelConfig, save_multiscale_model
from mg_coupled_pf.multiscale.train import TrainingConfig, train_multiscale_model
from mg_coupled_pf.multiscale.inverse_design import DesignVariableSpec, InverseDesignConfig, run_inverse_design


def _write_case(case_dir: Path, n_steps: int = 6, h: int = 24, w: int = 28) -> None:
    snap_dir = case_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    hist_rows = []
    for i in range(1, n_steps + 1):
        # 构造可学习趋势：phi 随步数下降，eta/epspeq 增加。
        t = float(i)
        phi = np.clip(1.0 - 0.1 * t + 0.05 * np.random.RandomState(i).randn(h, w), 0.0, 1.0).astype(np.float32)
        c = np.clip(1.0 - phi + 0.02 * np.random.RandomState(10 + i).randn(h, w), 0.0, 1.0).astype(np.float32)
        eta = np.clip(0.1 * t + 0.03 * np.random.RandomState(20 + i).randn(h, w), 0.0, 1.0).astype(np.float32)
        epspeq = np.clip(0.02 * t + 0.01 * np.random.RandomState(30 + i).randn(h, w), 0.0, 1.0).astype(np.float32)
        sigma_h = (80.0 * (1.0 - phi) + 5.0 * np.random.RandomState(40 + i).randn(h, w)).astype(np.float32)
        np.savez_compressed(
            snap_dir / f"snapshot_{i:06d}.npz",
            phi=phi,
            c=c,
            eta=eta,
            epspeq=epspeq,
            sigma_h=sigma_h,
        )
        hist_rows.append(
            {
                "step": i,
                "time_s": 1e-4 * i,
                "avg_epspeq": float(np.mean(epspeq)),
                "max_sigma_h": float(np.max(np.abs(sigma_h))),
            }
        )
    with (case_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["step", "time_s", "avg_epspeq", "max_sigma_h"])
        wr.writeheader()
        wr.writerows(hist_rows)


def test_multiscale_features_extract() -> None:
    h, w = 20, 24
    snap = {
        "phi": np.ones((h, w), dtype=np.float32),
        "c": np.zeros((h, w), dtype=np.float32),
        "eta": np.zeros((h, w), dtype=np.float32),
        "epspeq": np.zeros((h, w), dtype=np.float32),
        "sigma_h": np.zeros((h, w), dtype=np.float32),
    }
    x = extract_micro_tensor(snap, channels=("phi", "c", "eta", "epspeq"), target_hw=(16, 16))
    d = extract_micro_descriptors(snap)
    assert tuple(x.shape) == (4, 16, 16)
    assert "solid_fraction" in d and "penetration_x_ratio" in d


def test_multiscale_dataset_build_and_reload(tmp_path: Path) -> None:
    c1 = tmp_path / "case_a"
    c2 = tmp_path / "case_b"
    _write_case(c1)
    _write_case(c2)
    cfg = MultiScaleDatasetConfig(
        target_hw=(24, 24),
        horizon_steps=1,
        frame_stride=1,
        macro_targets=MacroTargetSpec(
            names=["corrosion_loss", "twin_fraction", "epspeq_mean", "sigma_h_max_solid"]
        ),
    )
    ds = build_multiscale_dataset_from_cases([c1, c2], cfg)
    assert ds["x_field"].shape[0] > 0
    assert ds["x_field"].shape[1] == 4
    p = tmp_path / "ds.npz"
    save_multiscale_dataset_npz(p, ds)
    ds2 = load_multiscale_dataset_npz(p)
    assert ds2["x_field"].shape == ds["x_field"].shape


def test_multiscale_train_and_inverse_design(tmp_path: Path) -> None:
    case = tmp_path / "case_train"
    _write_case(case, n_steps=8, h=24, w=24)
    ds = build_multiscale_dataset_from_cases(
        [case],
        MultiScaleDatasetConfig(
            target_hw=(24, 24),
            horizon_steps=1,
            frame_stride=1,
            macro_targets=MacroTargetSpec(
                names=["corrosion_loss", "twin_fraction", "epspeq_mean", "sigma_h_max_solid"]
            ),
        ),
    )
    mcfg = MultiScaleModelConfig(
        in_channels=int(ds["x_field"].shape[1]),
        desc_dim=int(ds["x_desc"].shape[1]),
        target_dim=int(ds["y"].shape[1]),
        encoder_width=24,
        encoder_depth=2,
        fusion_hidden=64,
        fusion_depth=2,
        dropout=0.0,
        use_uncertainty_head=True,
    )
    tcfg = TrainingConfig(epochs=2, batch_size=4, lr=1e-3, train_split=0.8, seed=1)
    model, rep = train_multiscale_model(ds, mcfg, tcfg, device=torch.device("cpu"))
    assert rep["n_train"] > 0
    ckpt = tmp_path / "ms_model.pt"
    save_multiscale_model(
        ckpt,
        model,
        mcfg,
        field_channels=[str(v) for v in ds["field_channels"].tolist()],
        descriptor_names=[str(v) for v in ds["descriptor_names"].tolist()],
        target_names=[str(v) for v in ds["target_names"].tolist()],
        scalers=rep["scalers"],
    )

    base_cfg = load_config("configs/notch_case.yaml")
    base_cfg.runtime.device = "cpu"
    base_cfg.domain.nx = 32
    base_cfg.domain.ny = 24
    res = run_inverse_design(
        base_cfg=base_cfg,
        model=model,
        model_meta={
            "field_channels": [str(v) for v in ds["field_channels"].tolist()],
            "descriptor_names": [str(v) for v in ds["descriptor_names"].tolist()],
            "target_names": [str(v) for v in ds["target_names"].tolist()],
            "scalers": rep["scalers"],
        },
        variables=[
            DesignVariableSpec("domain.notch_depth_um", 20.0, 100.0, 60.0),
            DesignVariableSpec("domain.notch_half_opening_um", 15.0, 80.0, 35.0),
        ],
        opt_cfg=InverseDesignConfig(iterations=2, population=8, elite_frac=0.25, target_hw=(24, 24)),
        device=torch.device("cpu"),
    )
    assert isinstance(res.best_score, float)
    assert "domain.notch_depth_um" in res.best_params

