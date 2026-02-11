#!/usr/bin/env python
"""求解器速度基准脚本（中文注释版）。

对比同一配置下：
- 纯物理求解
- 启用 ML surrogate

输出整体速度指标和加速比。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf import CoupledSimulator, load_config


def parse_args() -> argparse.Namespace:
    """解析基准参数。"""
    p = argparse.ArgumentParser(description="Benchmark coupled solver with/without ML acceleration.")
    p.add_argument("--config", default="configs/notch_case.yaml")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--disable-torch-compile", action="store_true")
    return p.parse_args()


def _sync(device: torch.device) -> None:
    """CUDA 同步，用于精确计时。"""
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_case(config_path: str, use_ml: bool, steps: int, warmup_steps: int, args: argparse.Namespace) -> dict:
    """运行一次基准用例并返回耗时统计。"""
    cfg = load_config(config_path)
    # 为纯计时场景关闭频繁输出，减少 IO 干扰。
    cfg.numerics.n_steps = steps
    cfg.numerics.save_every = max(steps + 1, 999999)
    cfg.ml.enabled = use_ml
    if args.device:
        cfg.runtime.device = args.device
    if args.mixed_precision:
        cfg.numerics.mixed_precision = True
    if args.disable_torch_compile:
        cfg.numerics.use_torch_compile = False
    cfg.runtime.case_name = f"{cfg.runtime.case_name}_{'ml' if use_ml else 'physics'}_bench"

    if warmup_steps > 0:
        # warmup 目的：触发 CUDA kernel 选择/缓存建立，避免首轮偏慢影响统计。
        warm_cfg = load_config(config_path)
        warm_cfg.numerics.n_steps = warmup_steps
        warm_cfg.numerics.save_every = max(warmup_steps + 1, 999999)
        warm_cfg.ml.enabled = use_ml
        if args.device:
            warm_cfg.runtime.device = args.device
        if args.mixed_precision:
            warm_cfg.numerics.mixed_precision = True
        if args.disable_torch_compile:
            warm_cfg.numerics.use_torch_compile = False
        warm_sim = CoupledSimulator(warm_cfg)
        for i in range(1, warmup_steps + 1):
            warm_sim.step(i)
        _sync(warm_sim.device)

    sim = CoupledSimulator(cfg)
    _sync(sim.device)
    # 正式计时：仅统计主循环 step 耗时。
    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        sim.step(i)
    _sync(sim.device)
    dt = time.perf_counter() - t0
    return {
        "use_ml": use_ml,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "wall_time_s": dt,
        "step_per_s": steps / max(dt, 1e-12),
        "device": str(sim.device),
    }


def main() -> None:
    """脚本主流程。"""
    args = parse_args()
    pure = run_case(args.config, use_ml=False, steps=args.steps, warmup_steps=args.warmup_steps, args=args)
    ml = run_case(args.config, use_ml=True, steps=args.steps, warmup_steps=args.warmup_steps, args=args)
    # 约定 speedup>1 表示“ML 版本更快”。
    speedup = pure["wall_time_s"] / max(ml["wall_time_s"], 1e-12)
    print(json.dumps({"physics_only": pure, "ml_enabled": ml, "speedup": speedup}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
