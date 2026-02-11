#!/usr/bin/env python
"""快速物理步长标定脚本（中文注释版）。

用途：在给定误差阈值下，从候选 dt 中自动筛选“最快且可接受”的时间步长。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf import CoupledSimulator, load_config


def parse_args() -> argparse.Namespace:
    """解析标定参数。"""
    p = argparse.ArgumentParser(
        description="Calibrate a fast physical time-step against a trusted baseline under accuracy constraints."
    )
    p.add_argument("--baseline-config", default="configs/half_half_highres_ml_pure.yaml")
    p.add_argument("--fast-config", default="configs/half_half_highres_fast_physics.yaml")
    p.add_argument("--baseline-dt-s", type=float, default=1e-3)
    p.add_argument("--dt-candidates-s", default="0.005,0.0075,0.01")
    p.add_argument("--total-time-s", type=float, default=0.5)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--c-mae-max", type=float, default=0.002)
    p.add_argument("--phi-mae-max", type=float, default=1e-4)
    p.add_argument("--epspeq-mae-max", type=float, default=1e-4)
    p.add_argument("--report-json", default="artifacts/reports/dt_calibration_report.json")
    return p.parse_args()


def _sync_cuda(dev: torch.device) -> None:
    """CUDA 同步，用于可靠计时。"""
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _apply_runtime(cfg, args: argparse.Namespace) -> None:
    """应用运行时覆盖（关闭 ML、禁用图像输出等）。"""
    cfg.ml.enabled = False
    if args.device != "auto":
        cfg.runtime.device = args.device
    if args.mixed_precision:
        cfg.numerics.mixed_precision = True
    cfg.runtime.clean_output = True
    cfg.runtime.render_intermediate_fields = False
    cfg.runtime.render_final_clouds = False
    cfg.runtime.render_grid_figure = False


def _run_case(cfg_path: str, dt_s: float, total_time_s: float, args: argparse.Namespace) -> tuple[CoupledSimulator, float]:
    """执行单个 dt 候选并返回仿真器与耗时。"""
    cfg = load_config(cfg_path)
    _apply_runtime(cfg, args)
    cfg.numerics.dt_s = dt_s
    # 用总物理时间反推步数，保证不同 dt 的“物理时长”一致。
    cfg.numerics.n_steps = max(1, int(math.ceil(total_time_s / max(dt_s, 1e-12))))
    cfg.numerics.save_every = cfg.numerics.n_steps
    sim = CoupledSimulator(cfg)
    _sync_cuda(sim.device)
    t0 = time.perf_counter()
    for i in range(1, cfg.numerics.n_steps + 1):
        sim.step(i)
    _sync_cuda(sim.device)
    return sim, time.perf_counter() - t0


def _field_mae(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个场的 MAE。"""
    return float(torch.mean(torch.abs(a - b)).item())


def main() -> None:
    """脚本主流程。"""
    args = parse_args()
    dt_candidates: List[float] = [float(x.strip()) for x in args.dt_candidates_s.split(",") if x.strip()]
    if not dt_candidates:
        raise RuntimeError("No valid dt candidates provided.")

    base_sim, base_wall = _run_case(args.baseline_config, args.baseline_dt_s, args.total_time_s, args)

    trials: List[Dict[str, float | bool]] = []
    best = None
    for dt in dt_candidates:
        # 逐个候选 dt 与基线终态直接做场误差比较。
        sim, wall = _run_case(args.fast_config, dt, args.total_time_s, args)
        c_mae = _field_mae(base_sim.state["c"], sim.state["c"])
        phi_mae = _field_mae(base_sim.state["phi"], sim.state["phi"])
        epspeq_mae = _field_mae(base_sim.state["epspeq"], sim.state["epspeq"])
        speedup = float(base_wall / max(wall, 1e-12))
        ok = (c_mae <= args.c_mae_max) and (phi_mae <= args.phi_mae_max) and (epspeq_mae <= args.epspeq_mae_max)
        row = {
            "dt_s": float(dt),
            "wall_time_s": float(wall),
            "speedup_vs_baseline": speedup,
            "c_mae": c_mae,
            "phi_mae": phi_mae,
            "epspeq_mae": epspeq_mae,
            "pass_accuracy": bool(ok),
            "diffusion_stage_count": int(len(sim._diffusion_stage_dts(dt))),
        }
        trials.append(row)
        if ok and (best is None or speedup > best["speedup_vs_baseline"]):
            # 在满足误差约束的候选中选速度最快者。
            best = row

    report = {
        "meta": {
            "baseline_config": args.baseline_config,
            "fast_config": args.fast_config,
            "baseline_dt_s": float(args.baseline_dt_s),
            "dt_candidates_s": dt_candidates,
            "total_time_s": float(args.total_time_s),
            "device": args.device,
            "mixed_precision": bool(args.mixed_precision),
            "accuracy_thresholds": {
                "c_mae_max": float(args.c_mae_max),
                "phi_mae_max": float(args.phi_mae_max),
                "epspeq_mae_max": float(args.epspeq_mae_max),
            },
        },
        "baseline": {
            "wall_time_s": float(base_wall),
        },
        "trials": trials,
        "recommended": best,
    }

    p = Path(args.report_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_json": str(p.resolve()), "recommended": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
