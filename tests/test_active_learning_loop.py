"""主动学习闭环烟雾测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from mg_coupled_pf import load_config
from mg_coupled_pf.multiscale.active_learning import ActiveLearningConfig, run_active_learning_loop
from mg_coupled_pf.multiscale.advanced_models import AdvancedModelConfig
from mg_coupled_pf.multiscale.advanced_train import AdvancedTrainingConfig
from mg_coupled_pf.multiscale.data_generation import SweepVariable


def test_active_learning_smoke(tmp_path: Path) -> None:
    cfg = load_config("configs/notch_case.yaml")
    # 缩小算例规模，保证测试可快速完成。
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 28
    cfg.domain.ny = 20
    cfg.domain.lx_um = 120.0
    cfg.domain.ly_um = 80.0
    cfg.numerics.dt_s = 1e-4
    cfg.ml.enabled = False

    vars_spec = [
        SweepVariable("domain.notch_depth_um", 20.0, 40.0, False),
        SweepVariable("domain.notch_half_opening_um", 10.0, 25.0, False),
        SweepVariable("corrosion.pitting_alpha", 0.2, 0.8, False),
    ]
    model_cfg = AdvancedModelConfig(
        in_channels=4,
        desc_dim=8,
        target_dim=5,
        width=24,
        depth=2,
        afno_blocks=1,
        modes_x=8,
        modes_y=8,
    )
    train_cfg = AdvancedTrainingConfig(
        seed=1,
        epochs=2,
        batch_size=2,
        lr=5e-4,
        train_split=0.8,
        model_name="multifidelity_uafno",
    )
    al_cfg = ActiveLearningConfig(
        rounds=2,
        initial_cases=2,
        new_cases_per_round=1,
        candidate_pool=4,
        seed=3,
        target_hw=(24, 24),
        low_target_hw=(12, 12),
        sim_steps=16,
        sim_save_every=4,
        output_root=str(tmp_path / "active_learning"),
    )
    out = run_active_learning_loop(
        base_cfg=cfg,
        vars_spec=vars_spec,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        al_cfg=al_cfg,
        device=torch.device("cpu"),
    )
    assert "latest_model" in out and Path(out["latest_model"]).exists()
    assert "round_reports" in out and len(out["round_reports"]) >= 2

