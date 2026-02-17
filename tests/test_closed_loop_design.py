"""闭环反向设计烟雾测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from mg_coupled_pf import load_config
from mg_coupled_pf.multiscale.advanced_models import AdvancedModelConfig, build_advanced_model, save_advanced_model, load_advanced_model
from mg_coupled_pf.multiscale.closed_loop_design import ClosedLoopDesignConfig, run_closed_loop_design
from mg_coupled_pf.multiscale.data_generation import SweepVariable


def test_closed_loop_design_smoke(tmp_path: Path) -> None:
    cfg = load_config("configs/notch_case.yaml")
    cfg.runtime.device = "cpu"
    cfg.domain.nx = 28
    cfg.domain.ny = 20
    cfg.domain.lx_um = 120.0
    cfg.domain.ly_um = 80.0
    cfg.ml.enabled = False

    mcfg = AdvancedModelConfig(
        in_channels=4,
        desc_dim=8,
        target_dim=5,
        width=16,
        depth=2,
        afno_blocks=1,
        modes_x=8,
        modes_y=8,
    )
    m = build_advanced_model("multifidelity_uafno", mcfg)
    ckpt = tmp_path / "dummy_adv.pt"
    save_advanced_model(
        ckpt,
        m,
        mcfg,
        model_name="multifidelity_uafno",
        field_channels=["phi", "c", "eta", "epspeq"],
        descriptor_names=[
            "solid_fraction",
            "twin_fraction",
            "c_solid_mean",
            "c_liquid_mean",
            "epspeq_mean",
            "sigma_h_max_solid",
            "interface_density",
            "penetration_x_ratio",
        ],
        target_names=["corrosion_loss", "twin_fraction", "epspeq_mean", "sigma_h_max_solid", "penetration_x_ratio"],
        scalers={},
    )
    model, meta = load_advanced_model(ckpt, device=torch.device("cpu"))

    out = run_closed_loop_design(
        base_cfg=cfg,
        vars_spec=[
            SweepVariable("domain.notch_depth_um", 20.0, 40.0, False),
            SweepVariable("domain.notch_half_opening_um", 10.0, 25.0, False),
        ],
        model=model,
        model_meta=meta,
        cfg=ClosedLoopDesignConfig(
            seed=2,
            candidate_pool=4,
            surrogate_topk=2,
            physics_verify_topk=1,
            local_refine_trials=1,
            sim_steps=8,
            sim_save_every=4,
            output_root=str(tmp_path / "cld"),
        ),
        device=torch.device("cpu"),
        target_hw=(24, 24),
    )
    assert "best_final" in out

