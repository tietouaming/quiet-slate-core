"""构建多保真跨尺度数据集。

示例：
python scripts/build_multifidelity_dataset.py ^
  --cases-root artifacts/sim_notch ^
  --pattern * ^
  --horizon-steps 4 ^
  --target-h 128 --target-w 128 ^
  --low-target-h 48 --low-target-w 48 ^
  --high-fidelity-ratio 0.25 ^
  --output artifacts/ml/multifidelity_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mg_coupled_pf.multiscale.dataset import (
    MacroTargetSpec,
    MultiFidelityDatasetConfig,
    build_multifidelity_dataset_from_cases,
    discover_case_dirs,
    save_multiscale_dataset_npz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build multi-fidelity micro->macro dataset from simulation cases.")
    p.add_argument("--cases-root", type=str, required=True)
    p.add_argument("--pattern", type=str, default="*")
    p.add_argument("--horizon-steps", type=int, default=1)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-samples-per-case", type=int, default=0)
    p.add_argument("--target-h", type=int, default=128)
    p.add_argument("--target-w", type=int, default=128)
    p.add_argument("--low-target-h", type=int, default=48)
    p.add_argument("--low-target-w", type=int, default=48)
    p.add_argument("--high-fidelity-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--channels", type=str, default="phi,c,eta,epspeq")
    p.add_argument(
        "--targets",
        type=str,
        default="corrosion_loss,twin_fraction,epspeq_mean,sigma_h_max_solid,penetration_x_ratio",
    )
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cases = discover_case_dirs(Path(args.cases_root), pattern=str(args.pattern), require_history=False)
    if not cases:
        raise RuntimeError(f"No cases found under {args.cases_root}")
    cfg = MultiFidelityDatasetConfig(
        field_channels=[s.strip() for s in str(args.channels).split(",") if s.strip()],
        target_hw=(int(args.target_h), int(args.target_w)),
        low_target_hw=(int(args.low_target_h), int(args.low_target_w)),
        horizon_steps=max(1, int(args.horizon_steps)),
        frame_stride=max(1, int(args.frame_stride)),
        max_samples_per_case=max(0, int(args.max_samples_per_case)),
        high_fidelity_keep_ratio=float(args.high_fidelity_ratio),
        seed=int(args.seed),
        macro_targets=MacroTargetSpec(names=[s.strip() for s in str(args.targets).split(",") if s.strip()]),
    )
    ds = build_multifidelity_dataset_from_cases(cases, cfg)
    out = save_multiscale_dataset_npz(Path(args.output), ds)
    print(
        {
            "output": str(out),
            "n_samples": int(ds["x_field"].shape[0]),
            "field_shape": tuple(int(v) for v in ds["x_field"].shape[1:]),
            "high_ratio_actual": float(ds["high_mask"].mean()),
            "n_targets": int(ds["y"].shape[1]),
        }
    )


if __name__ == "__main__":
    main()

