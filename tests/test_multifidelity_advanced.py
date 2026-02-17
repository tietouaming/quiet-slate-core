"""高级多保真跨尺度模块测试。"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch

from mg_coupled_pf.multiscale.advanced_models import (
    AdvancedModelConfig,
    load_advanced_model,
    save_advanced_model,
)
from mg_coupled_pf.multiscale.advanced_train import AdvancedTrainingConfig, train_advanced_multiscale_model
from mg_coupled_pf.multiscale.dataset import (
    MacroTargetSpec,
    MultiFidelityDatasetConfig,
    build_multifidelity_dataset_from_cases,
)
from mg_coupled_pf.multiscale.physics_metrics import PhysicsMetricConfig, evaluate_physics_metrics


def _write_case(case_dir: Path, n_steps: int = 7, h: int = 20, w: int = 24) -> None:
    snap_dir = case_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n_steps + 1):
        rs = np.random.RandomState(100 + i)
        t = float(i)
        phi = np.clip(1.0 - 0.08 * t + 0.03 * rs.randn(h, w), 0.0, 1.0).astype(np.float32)
        c = np.clip(1.0 - phi + 0.01 * rs.randn(h, w), 0.0, 1.0).astype(np.float32)
        eta = np.clip(0.05 * t + 0.02 * rs.randn(h, w), 0.0, 1.0).astype(np.float32)
        epspeq = np.clip(0.01 * t + 0.01 * rs.randn(h, w), 0.0, 1.0).astype(np.float32)
        sigma_h = (60.0 * (1.0 - phi) + 5.0 * rs.randn(h, w)).astype(np.float32)
        np.savez_compressed(
            snap_dir / f"snapshot_{i:06d}.npz",
            phi=phi,
            c=c,
            eta=eta,
            epspeq=epspeq,
            sigma_h=sigma_h,
        )
        rows.append({"step": i, "time_s": 1e-4 * i, "avg_epspeq": float(np.mean(epspeq)), "max_sigma_h": float(np.max(np.abs(sigma_h)))})
    with (case_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["step", "time_s", "avg_epspeq", "max_sigma_h"])
        wr.writeheader()
        wr.writerows(rows)


def test_build_multifidelity_dataset(tmp_path: Path) -> None:
    c1 = tmp_path / "case1"
    c2 = tmp_path / "case2"
    _write_case(c1)
    _write_case(c2)
    cfg = MultiFidelityDatasetConfig(
        target_hw=(24, 24),
        low_target_hw=(12, 12),
        horizon_steps=1,
        frame_stride=1,
        high_fidelity_keep_ratio=0.4,
        seed=11,
        macro_targets=MacroTargetSpec(names=["corrosion_loss", "twin_fraction", "epspeq_mean"]),
    )
    ds = build_multifidelity_dataset_from_cases([c1, c2], cfg)
    assert ds["x_field"].shape[0] > 0
    assert "y_low" in ds and "high_mask" in ds
    assert ds["y_low"].shape == ds["y"].shape
    assert 0.0 < float(np.mean(ds["high_mask"])) < 1.0


def test_train_advanced_multifidelity_and_metrics(tmp_path: Path) -> None:
    case = tmp_path / "case_train"
    _write_case(case, n_steps=8)
    ds = build_multifidelity_dataset_from_cases(
        [case],
        MultiFidelityDatasetConfig(
            target_hw=(24, 24),
            low_target_hw=(12, 12),
            horizon_steps=1,
            frame_stride=1,
            high_fidelity_keep_ratio=0.6,
            seed=4,
            macro_targets=MacroTargetSpec(
                names=["corrosion_loss", "twin_fraction", "epspeq_mean", "sigma_h_max_solid", "penetration_x_ratio"]
            ),
        ),
    )
    mcfg = AdvancedModelConfig(
        in_channels=int(ds["x_field"].shape[1]),
        desc_dim=int(ds["x_desc"].shape[1]),
        target_dim=int(ds["y"].shape[1]),
        width=24,
        depth=2,
        afno_blocks=1,
        modes_x=8,
        modes_y=8,
        dropout=0.0,
    )
    tcfg = AdvancedTrainingConfig(
        epochs=2,
        batch_size=4,
        lr=8e-4,
        train_split=0.8,
        model_name="multifidelity_uafno",
        seed=3,
    )
    model, rep = train_advanced_multiscale_model(ds, mcfg, tcfg, device=torch.device("cpu"))
    assert rep["n_train"] > 0
    assert "best_physics_metrics" in rep

    ckpt = tmp_path / "adv.pt"
    save_advanced_model(
        ckpt,
        model,
        mcfg,
        model_name="multifidelity_uafno",
        field_channels=[str(v) for v in ds["field_channels"].tolist()],
        descriptor_names=[str(v) for v in ds["descriptor_names"].tolist()],
        target_names=[str(v) for v in ds["target_names"].tolist()],
        scalers=rep["scalers"],
    )
    m2, meta = load_advanced_model(ckpt, device=torch.device("cpu"))
    xf = torch.from_numpy(ds["x_field"][:3].astype(np.float32))
    xd = torch.from_numpy(ds["x_desc"][:3].astype(np.float32))
    if "x_desc" in meta["scalers"]:
        mu = torch.from_numpy(np.asarray(meta["scalers"]["x_desc"]["mean"], dtype=np.float32))
        sd = torch.from_numpy(np.asarray(meta["scalers"]["x_desc"]["std"], dtype=np.float32))
        xd = (xd - mu) / sd
    with torch.no_grad():
        out = m2(xf, xd)
    assert "high_mean" in out

    # 物理一致性指标 smoke test
    y_true = ds["y"][:3].astype(np.float32)
    y_mean = y_true.copy()
    y_logv = np.zeros_like(y_true, dtype=np.float32)
    phy = evaluate_physics_metrics(
        y_true=y_true,
        y_mean=y_mean,
        y_logvar=y_logv,
        target_names=[str(v) for v in ds["target_names"].tolist()],
        case_ids=[str(v) for v in ds["case_ids"][:3].tolist()],
        t_inputs=ds["t_input"][:3].astype(np.float64),
        cfg=PhysicsMetricConfig(),
    )
    assert "physics_consistency_index" in phy

