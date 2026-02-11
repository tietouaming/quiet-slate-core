#!/usr/bin/env python
"""力学加载模式对照基准脚本（中文注释版）。

对照两种加载模式：
1) eigenstrain（宏观应变项）
2) dirichlet_x（左右位移边界）

输出 wall-time、关键场统计、边界约束误差，用于验证改造正确性与代价。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf import CoupledSimulator, load_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="Benchmark mechanics loading modes (eigenstrain vs dirichlet_x).")
    p.add_argument("--config", default="configs/notch_case.yaml")
    p.add_argument("--dt-s", type=float, default=1e-4)
    p.add_argument("--total-time-s", type=float, default=1e-3)
    p.add_argument("--save-interval-s", type=float, default=1e-3)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--disable-torch-compile", action="store_true")
    p.add_argument("--dirichlet-right-ux-um", type=float, default=-1.0)
    p.add_argument("--report-json", default="artifacts/reports/loading_mode_benchmark.json")
    return p.parse_args()


def _run_once(args: argparse.Namespace, mode: str) -> dict:
    """执行单个加载模式并返回统计。"""
    cfg = load_config(args.config)
    cfg.runtime.device = args.device
    cfg.numerics.dt_s = float(args.dt_s)
    cfg.numerics.n_steps = max(1, int(math.ceil(float(args.total_time_s) / max(cfg.numerics.dt_s, 1e-12))))
    cfg.numerics.save_every = max(1, int(math.ceil(float(args.save_interval_s) / max(cfg.numerics.dt_s, 1e-12))))
    if args.disable_torch_compile:
        cfg.numerics.use_torch_compile = False
    cfg.ml.enabled = False
    cfg.runtime.render_intermediate_fields = False
    cfg.runtime.render_final_clouds = False
    cfg.runtime.render_grid_figure = False
    cfg.runtime.clean_output = True
    cfg.runtime.case_name = f"{cfg.runtime.case_name}_loading_{mode}"
    cfg.mechanics.loading_mode = mode
    if mode == "dirichlet_x" and float(args.dirichlet_right_ux_um) >= 0.0:
        cfg.mechanics.dirichlet_right_displacement_um = float(args.dirichlet_right_ux_um)

    sim = CoupledSimulator(cfg)
    t0 = time.perf_counter()
    for i in range(1, cfg.numerics.n_steps + 1):
        sim.step(i)
    wall = time.perf_counter() - t0

    ux = sim.state["ux"]
    left = float(torch.max(torch.abs(ux[:, :, :, 0])).item())
    right = float(torch.mean(ux[:, :, :, -1]).item())
    if mode == "dirichlet_x":
        if cfg.mechanics.dirichlet_right_displacement_um >= 0.0:
            target = float(cfg.mechanics.dirichlet_right_displacement_um)
        else:
            target = float(cfg.mechanics.external_strain_x) * float(cfg.domain.lx_um)
        right_err = abs(right - target)
    else:
        target = None
        right_err = None

    return {
        "mode": mode,
        "device": str(sim.device),
        "dt_s": float(cfg.numerics.dt_s),
        "total_time_s": float(cfg.numerics.dt_s * cfg.numerics.n_steps),
        "n_steps": int(cfg.numerics.n_steps),
        "wall_time_s": float(wall),
        "final_solid_fraction": float(sim.history[-1]["solid_fraction"]) if sim.history else float(torch.mean(sim.state["phi"]).item()),
        "final_avg_eta": float(sim.history[-1]["avg_eta"]) if sim.history else float(torch.mean(sim.state["eta"]).item()),
        "final_max_sigma_h": float(sim.history[-1]["max_sigma_h"]) if sim.history else float(torch.max(sim.last_mech["sigma_h"]).item()),
        "left_boundary_ux_abs_max": left,
        "right_boundary_ux_mean": right,
        "right_boundary_ux_target": target,
        "right_boundary_ux_abs_error": right_err,
        "solver_stats": dict(sim.stats),
    }


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    out = {}
    out["eigenstrain"] = _run_once(args, "eigenstrain")
    out["dirichlet_x"] = _run_once(args, "dirichlet_x")
    e = out["eigenstrain"]["wall_time_s"]
    d = out["dirichlet_x"]["wall_time_s"]
    out["summary"] = {
        "dirichlet_over_eigenstrain_wall_ratio": float(d / max(e, 1e-12)),
        "dirichlet_right_boundary_abs_error_um": out["dirichlet_x"]["right_boundary_ux_abs_error"],
    }

    p = Path(args.report_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report_json": str(p.resolve()), "summary": out["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

