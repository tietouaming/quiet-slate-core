#!/usr/bin/env python
"""单算例入口脚本（中文注释版）。

用途：
- 按配置运行一次耦合仿真；
- 支持从命令行覆盖时间步、总物理时间、输出间隔与可视化开关；
- 运行结束后输出结构化 JSON 结果，便于自动化流程调用。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf import CoupledSimulator, load_config


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。

    参数分组：
    - 时间控制：`--dt-s`, `--total-time-s`, `--save-interval-s`
    - 运行设备：`--device`, `--dtype`, `--mixed-precision`
    - 编译/推理：`--disable-torch-compile`, `--disable-inference-mode`
    - 输出控制：`--no-clean-output` 与三类 render 开关
    - 算法开关：`--disable-ml`
    """
    p = argparse.ArgumentParser(description="Run coupled Mg notch corrosion-twinning-CP simulation.")
    p.add_argument("--config", default="configs/notch_case.yaml")
    p.add_argument("--override-steps", type=int, default=0)
    p.add_argument("--override-save-every", type=int, default=0)
    p.add_argument("--dt-s", type=float, default=0.0, help="Physical time step size (seconds).")
    p.add_argument("--total-time-s", type=float, default=0.0, help="Total physical simulation time (seconds).")
    p.add_argument(
        "--save-interval-s",
        type=float,
        default=0.0,
        help="Snapshot/output interval in physical time (seconds).",
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="")
    p.add_argument("--dtype", choices=["float32", "float64"], default="")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--disable-torch-compile", action="store_true")
    p.add_argument("--disable-inference-mode", action="store_true")
    p.add_argument("--no-clean-output", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--disable-ml", action="store_true")
    p.add_argument(
        "--render-intermediate-fields",
        dest="render_intermediate_fields",
        action="store_true",
        default=None,
        help="Force enable per-save field figures.",
    )
    p.add_argument(
        "--no-render-intermediate-fields",
        dest="render_intermediate_fields",
        action="store_false",
        default=None,
        help="Force disable per-save field figures.",
    )
    p.add_argument(
        "--render-final-clouds",
        dest="render_final_clouds",
        action="store_true",
        default=None,
        help="Force enable final cloud figures.",
    )
    p.add_argument(
        "--no-render-final-clouds",
        dest="render_final_clouds",
        action="store_false",
        default=None,
        help="Force disable final cloud figures.",
    )
    p.add_argument(
        "--render-grid-figure",
        dest="render_grid_figure",
        action="store_true",
        default=None,
        help="Force enable grid figures.",
    )
    p.add_argument(
        "--no-render-grid-figure",
        dest="render_grid_figure",
        action="store_false",
        default=None,
        help="Force disable grid figures.",
    )
    return p.parse_args()


def main() -> None:
    """脚本主流程：读取配置 -> 覆盖参数 -> 运行 -> 输出JSON。"""
    args = parse_args()

    # 1) 先从 YAML 读取基线配置。
    cfg = load_config(args.config)

    # 2) 依次应用命令行覆盖（CLI 优先级高于 YAML）。
    if args.dt_s > 0:
        cfg.numerics.dt_s = args.dt_s
    if args.total_time_s > 0:
        # 使用物理时间统一控制步数，避免“固定步数”口径歧义。
        cfg.numerics.n_steps = max(1, int(math.ceil(args.total_time_s / max(cfg.numerics.dt_s, 1e-12))))
    elif args.override_steps > 0:
        cfg.numerics.n_steps = args.override_steps
    if args.save_interval_s > 0:
        # 保存间隔同样按物理时间换算为步数。
        cfg.numerics.save_every = max(1, int(math.ceil(args.save_interval_s / max(cfg.numerics.dt_s, 1e-12))))
    elif args.override_save_every > 0:
        cfg.numerics.save_every = args.override_save_every
    if args.device:
        cfg.runtime.device = args.device
    if args.dtype:
        cfg.numerics.dtype = args.dtype
    if args.mixed_precision:
        cfg.numerics.mixed_precision = True
    if args.disable_torch_compile:
        cfg.numerics.use_torch_compile = False
    if args.disable_inference_mode:
        cfg.numerics.inference_mode = False
    if args.no_clean_output:
        cfg.runtime.clean_output = False
    if args.disable_ml:
        cfg.ml.enabled = False
    if args.render_intermediate_fields is not None:
        cfg.runtime.render_intermediate_fields = bool(args.render_intermediate_fields)
    if args.render_final_clouds is not None:
        cfg.runtime.render_final_clouds = bool(args.render_final_clouds)
    if args.render_grid_figure is not None:
        cfg.runtime.render_grid_figure = bool(args.render_grid_figure)

    # 3) 构建求解器并运行。
    sim = CoupledSimulator(cfg)
    outputs = sim.run(
        progress=args.progress,
        progress_every=max(1, args.progress_every),
        progress_prefix=cfg.runtime.case_name,
    )

    # 4) 统一输出关键元数据，方便后处理脚本直接消费。
    dt_s = float(cfg.numerics.dt_s)
    total_time_s = float(cfg.numerics.n_steps * cfg.numerics.dt_s)
    save_interval_s = float(cfg.numerics.save_every * cfg.numerics.dt_s)
    print(
        json.dumps(
            {
                "case": cfg.runtime.case_name,
                "dt_s": dt_s,
                "total_time_s": total_time_s,
                "save_interval_s": save_interval_s,
                "n_steps": int(cfg.numerics.n_steps),
                "diffusion_dt_limit_s": float(sim.diffusion_dt_limit_s),
                "ml_model_arch": str(cfg.ml.model_arch),
                "solver_stats": sim.stats,
                "output_dir": str(outputs["output_dir"]),
                "history_csv": str(outputs["history_csv"]),
                "snapshots_dir": str(outputs["snapshots_dir"]),
                "figures_dir": str(outputs["figures_dir"]),
                "grid_dir": str(outputs["grid_dir"]),
                "final_clouds_dir": str(outputs["final_clouds_dir"]),
                "render_intermediate_fields": bool(cfg.runtime.render_intermediate_fields),
                "render_final_clouds": bool(cfg.runtime.render_final_clouds),
                "render_grid_figure": bool(cfg.runtime.render_grid_figure),
                "figures_dir_exists": Path(outputs["figures_dir"]).exists(),
                "grid_dir_exists": Path(outputs["grid_dir"]).exists(),
                "final_clouds_dir_exists": Path(outputs["final_clouds_dir"]).exists(),
                "wall_time_s": float(outputs.get("wall_time_s", 0.0)),
                "device": str(sim.device),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
