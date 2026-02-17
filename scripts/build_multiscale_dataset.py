"""构建跨尺度训练数据集。

用法示例：
python scripts/build_multiscale_dataset.py ^
  --cases-root artifacts/sim_notch ^
  --pattern * ^
  --horizon-steps 4 ^
  --target-h 96 --target-w 96 ^
  --output artifacts/ml/multiscale_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mg_coupled_pf.multiscale.dataset import (
    MacroTargetSpec,
    MultiScaleDatasetConfig,
    build_multiscale_dataset_from_cases,
    discover_case_dirs,
    save_multiscale_dataset_npz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multiscale micro->macro dataset from simulation cases.")
    p.add_argument("--cases-root", type=str, required=True, help="包含多个 case 子目录的根目录。")
    p.add_argument("--pattern", type=str, default="*", help="case 目录匹配模式。")
    p.add_argument("--horizon-steps", type=int, default=1, help="输入到目标的快照步长间隔。")
    p.add_argument("--frame-stride", type=int, default=1, help="样本抽样步长。")
    p.add_argument("--max-samples-per-case", type=int, default=0, help="每个 case 最多采样数，0 为不限制。")
    p.add_argument("--target-h", type=int, default=96)
    p.add_argument("--target-w", type=int, default=96)
    p.add_argument(
        "--channels",
        type=str,
        default="phi,c,eta,epspeq",
        help="输入场通道，逗号分隔。",
    )
    p.add_argument(
        "--targets",
        type=str,
        default="corrosion_loss,twin_fraction,epspeq_mean,sigma_h_max_solid,penetration_x_ratio",
        help="目标宏观指标，逗号分隔。",
    )
    p.add_argument("--output", type=str, required=True, help="输出数据集 npz 路径。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.cases_root)
    cases = discover_case_dirs(root, pattern=str(args.pattern), require_history=False)
    if not cases:
        raise RuntimeError(f"No case dirs found under: {root}")
    cfg = MultiScaleDatasetConfig(
        field_channels=[s.strip() for s in str(args.channels).split(",") if s.strip()],
        target_hw=(int(args.target_h), int(args.target_w)),
        horizon_steps=max(1, int(args.horizon_steps)),
        frame_stride=max(1, int(args.frame_stride)),
        max_samples_per_case=max(0, int(args.max_samples_per_case)),
        macro_targets=MacroTargetSpec(names=[s.strip() for s in str(args.targets).split(",") if s.strip()]),
    )
    ds = build_multiscale_dataset_from_cases(cases, cfg)
    out = save_multiscale_dataset_npz(Path(args.output), ds)
    print(
        {
            "output": str(out),
            "n_samples": int(ds["x_field"].shape[0]),
            "field_shape": tuple(int(x) for x in ds["x_field"].shape[1:]),
            "n_targets": int(ds["y"].shape[1]),
        }
    )


if __name__ == "__main__":
    main()

