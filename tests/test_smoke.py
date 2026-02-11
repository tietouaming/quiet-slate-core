"""基础烟测（中文注释版）。"""

from __future__ import annotations

from pathlib import Path

from mg_coupled_pf import CoupledSimulator, load_config


def test_smoke_run(tmp_path: Path) -> None:
    """验证最小步数算例可跑通且主场变量边界合理。"""
    cfg = load_config("configs/notch_case.yaml")
    cfg.numerics.n_steps = 8
    cfg.numerics.save_every = 4
    cfg.ml.enabled = False
    cfg.runtime.output_dir = str(tmp_path)
    cfg.runtime.case_name = "smoke_case"
    sim = CoupledSimulator(cfg)
    out = sim.run()
    assert Path(out["history_csv"]).exists()
    assert len(sim.history) == cfg.numerics.n_steps
    phi = sim.state["phi"]
    assert float(phi.min()) >= 0.0
    assert float(phi.max()) <= 1.0
    c = sim.state["c"]
    assert float(c.min()) >= 0.0
    assert float(c.max()) <= 1.0
