"""运行跨尺度主动学习闭环。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from mg_coupled_pf import load_config
from mg_coupled_pf.multiscale import (
    ActiveLearningConfig,
    AdvancedModelConfig,
    AdvancedTrainingConfig,
    plot_active_learning_progress,
    run_active_learning_loop,
)
from mg_coupled_pf.multiscale.data_generation import SweepVariable


def parse_var(s: str) -> SweepVariable:
    parts = [p.strip() for p in str(s).split(":")]
    if len(parts) < 3:
        raise ValueError(f"Invalid --var: {s}. expect path:lo:hi[:log]")
    return SweepVariable(
        path=parts[0],
        lower=float(parts[1]),
        upper=float(parts[2]),
        log_scale=(len(parts) >= 4 and parts[3].lower() in {"1", "true", "yes", "log"}),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run active-learning loop for multiscale macro surrogate.")
    p.add_argument("--config", type=str, default="configs/notch_case.yaml")
    p.add_argument("--var", type=str, action="append", required=True, help="path:lo:hi[:log]")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--initial-cases", type=int, default=12)
    p.add_argument("--new-cases-per-round", type=int, default=6)
    p.add_argument("--candidate-pool", type=int, default=48)
    p.add_argument("--sim-steps", type=int, default=220)
    p.add_argument("--sim-save-every", type=int, default=22)
    p.add_argument("--target-h", type=int, default=96)
    p.add_argument("--target-w", type=int, default=96)
    p.add_argument("--low-target-h", type=int, default=48)
    p.add_argument("--low-target-w", type=int, default=48)
    p.add_argument("--high-fidelity-ratio", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-root", type=str, default="artifacts/active_learning")
    p.add_argument("--output-json", type=str, default="")
    p.add_argument("--output-fig", type=str, default="")
    p.add_argument("--model-width", type=int, default=40)
    p.add_argument("--model-depth", type=int, default=2)
    p.add_argument("--afno-blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    n = str(name).strip().lower()
    if n == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(n)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    vars_spec: List[SweepVariable] = [parse_var(v) for v in args.var]
    dev = pick_device(args.device)

    model_cfg = AdvancedModelConfig(
        in_channels=4,
        desc_dim=8,
        target_dim=5,
        width=int(args.model_width),
        depth=int(args.model_depth),
        afno_blocks=int(args.afno_blocks),
        modes_x=16,
        modes_y=16,
    )
    train_cfg = AdvancedTrainingConfig(
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        train_split=0.85,
        model_name="multifidelity_uafno",
    )
    al_cfg = ActiveLearningConfig(
        rounds=int(args.rounds),
        initial_cases=int(args.initial_cases),
        new_cases_per_round=int(args.new_cases_per_round),
        candidate_pool=int(args.candidate_pool),
        seed=int(args.seed),
        target_hw=(int(args.target_h), int(args.target_w)),
        low_target_hw=(int(args.low_target_h), int(args.low_target_w)),
        high_fidelity_keep_ratio=float(args.high_fidelity_ratio),
        sim_steps=int(args.sim_steps),
        sim_save_every=int(args.sim_save_every),
        output_root=str(args.output_root),
    )
    out = run_active_learning_loop(
        base_cfg=cfg,
        vars_spec=vars_spec,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        al_cfg=al_cfg,
        device=dev,
    )

    output_json = Path(args.output_json) if args.output_json else Path(args.output_root) / "active_learning_report.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    rr = [r for r in out.get("round_reports", []) if isinstance(r, dict) and "train_report" in r]
    output_fig = Path(args.output_fig) if args.output_fig else Path(args.output_root) / "active_learning_progress.png"
    if rr:
        plot_active_learning_progress(rr, output_fig)

    print(
        json.dumps(
            {
                "report_json": str(output_json),
                "progress_fig": str(output_fig),
                "latest_model": out.get("latest_model", ""),
                "latest_dataset": out.get("latest_dataset", ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
