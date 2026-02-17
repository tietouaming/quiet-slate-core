"""批量生成用于跨尺度训练的物理仿真算例。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from mg_coupled_pf import load_config
from mg_coupled_pf.multiscale.data_generation import (
    PhysicsDataGenConfig,
    SweepVariable,
    generate_physics_cases,
)


def parse_var(s: str) -> SweepVariable:
    parts = [p.strip() for p in str(s).split(":")]
    if len(parts) < 3:
        raise ValueError(f"Invalid --var: {s}. Expect path:lo:hi[:log]")
    path = parts[0]
    lo = float(parts[1])
    hi = float(parts[2])
    log_scale = len(parts) >= 4 and parts[3].lower() in {"1", "true", "yes", "log"}
    return SweepVariable(path=path, lower=lo, upper=hi, log_scale=log_scale)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate physics simulation cases for multiscale dataset.")
    p.add_argument("--config", type=str, default="configs/notch_case.yaml")
    p.add_argument("--var", type=str, action="append", required=True, help="path:lo:hi[:log]")
    p.add_argument("--n-cases", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=300)
    p.add_argument("--save-every", type=int, default=30)
    p.add_argument("--output-root", type=str, default="artifacts/sim_multiscale_dataset")
    p.add_argument("--case-prefix", type=str, default="ms_case")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--summary-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    vars_spec: List[SweepVariable] = [parse_var(v) for v in args.var]
    gen_cfg = PhysicsDataGenConfig(
        n_cases=int(args.n_cases),
        seed=int(args.seed),
        output_root=str(args.output_root),
        case_prefix=str(args.case_prefix),
        n_steps=int(args.n_steps),
        save_every=int(args.save_every),
        progress=bool(args.progress),
    )
    summary = generate_physics_cases(base_cfg=cfg, vars_spec=vars_spec, gen_cfg=gen_cfg)
    out = Path(args.summary_json) if args.summary_json else Path(args.output_root) / "generation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"summary_json": str(out), "n_succeeded": summary["n_succeeded"], "n_failed": summary["n_failed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

