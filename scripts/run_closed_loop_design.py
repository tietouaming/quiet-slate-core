"""运行跨尺度闭环反向设计（surrogate + physics verify）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from mg_coupled_pf import load_config
from mg_coupled_pf.multiscale import ClosedLoopDesignConfig, load_advanced_model, run_closed_loop_design
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


def parse_weight(items: List[str]) -> dict:
    out = {}
    for s in items:
        k, v = s.split(":", 1)
        out[str(k).strip()] = float(v)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop inverse design with physics verification.")
    p.add_argument("--config", type=str, default="configs/notch_case.yaml")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--var", type=str, action="append", required=True)
    p.add_argument("--weight", type=str, action="append", default=[])
    p.add_argument("--candidate-pool", type=int, default=40)
    p.add_argument("--surrogate-topk", type=int, default=8)
    p.add_argument("--physics-verify-topk", type=int, default=4)
    p.add_argument("--local-refine-trials", type=int, default=6)
    p.add_argument("--local-refine-sigma-ratio", type=float, default=0.08)
    p.add_argument("--sim-steps", type=int, default=260)
    p.add_argument("--sim-save-every", type=int, default=26)
    p.add_argument("--target-h", type=int, default=128)
    p.add_argument("--target-w", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default="artifacts/closed_loop_design")
    p.add_argument("--output-json", type=str, default="")
    p.add_argument("--output-fig", type=str, default="")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    n = str(name).strip().lower()
    if n == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(n)


def plot_design_scores(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verified = payload.get("verified", [])
    refined = payload.get("refined", [])
    v_pred = [float(x.get("pred_score", 0.0)) for x in verified]
    v_true = [float(x.get("true_score", 0.0)) for x in verified]
    r_true = [float(x.get("true_score", 0.0)) for x in refined]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    if v_pred:
        ax.plot(np.arange(len(v_pred)), v_pred, "o--", label="verified pred score")
    if v_true:
        ax.plot(np.arange(len(v_true)), v_true, "o-", label="verified true score")
    if r_true:
        ax.plot(np.arange(len(r_true)), r_true, "s-", label="local refine true score")
    ax.set_xlabel("candidate index")
    ax.set_ylabel("objective score")
    ax.set_title("Closed-loop design score comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model, meta = load_advanced_model(args.model, device=pick_device(args.device))
    vars_spec = [parse_var(v) for v in args.var]
    weights = parse_weight(args.weight) if args.weight else ClosedLoopDesignConfig().objective_weights
    cl_cfg = ClosedLoopDesignConfig(
        seed=int(args.seed),
        candidate_pool=int(args.candidate_pool),
        surrogate_topk=int(args.surrogate_topk),
        physics_verify_topk=int(args.physics_verify_topk),
        local_refine_trials=int(args.local_refine_trials),
        local_refine_sigma_ratio=float(args.local_refine_sigma_ratio),
        sim_steps=int(args.sim_steps),
        sim_save_every=int(args.sim_save_every),
        objective_weights=weights,
        output_root=str(args.output_root),
    )
    out = run_closed_loop_design(
        base_cfg=cfg,
        vars_spec=vars_spec,
        model=model,
        model_meta=meta,
        cfg=cl_cfg,
        device=pick_device(args.device),
        target_hw=(int(args.target_h), int(args.target_w)),
    )
    out_json = Path(args.output_json) if args.output_json else Path(args.output_root) / "closed_loop_design_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    out_fig = Path(args.output_fig) if args.output_fig else Path(args.output_root) / "closed_loop_design_scores.png"
    plot_design_scores(out, out_fig)
    print(json.dumps({"report_json": str(out_json), "score_fig": str(out_fig), "best_final": out.get("best_final", {})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

