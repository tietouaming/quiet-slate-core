"""跨尺度 surrogate 反向设计脚本。

示例：
python scripts/inverse_design_multiscale.py ^
  --config configs/notch_case.yaml ^
  --model artifacts/ml/multiscale_macro.pt ^
  --var domain.notch_depth_um:30:90:60 ^
  --var domain.notch_half_opening_um:20:70:40 ^
  --var corrosion.pitting_alpha:0.1:0.8:0.4 ^
  --weight corrosion_loss:-1.0 ^
  --weight sigma_h_max_solid:-0.2 ^
  --weight twin_fraction:0.1 ^
  --iterations 25 --population 64 ^
  --output artifacts/reports/inverse_design_multiscale.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from mg_coupled_pf import load_config
from mg_coupled_pf.multiscale.inverse_design import (
    DesignVariableSpec,
    InverseDesignConfig,
    run_inverse_design,
)
from mg_coupled_pf.multiscale.models import load_multiscale_model


def parse_var(s: str) -> DesignVariableSpec:
    parts = [p.strip() for p in str(s).split(":")]
    if len(parts) < 3:
        raise ValueError(f"Invalid --var format: {s}")
    path = parts[0]
    lo = float(parts[1])
    hi = float(parts[2])
    init = float(parts[3]) if len(parts) >= 4 and parts[3] != "" else None
    return DesignVariableSpec(path=path, lower=lo, upper=hi, init=init)


def parse_weight(items: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in items:
        k, v = s.split(":", 1)
        out[str(k).strip()] = float(v)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inverse design with multiscale surrogate.")
    p.add_argument("--config", type=str, required=True, help="base simulation config yaml")
    p.add_argument("--model", type=str, required=True, help="trained multiscale checkpoint")
    p.add_argument("--var", type=str, action="append", required=True, help="design var: path:lo:hi[:init]")
    p.add_argument("--weight", type=str, action="append", default=[], help="objective weight: name:value")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--elite-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--target-h", type=int, default=96)
    p.add_argument("--target-w", type=int, default=96)
    p.add_argument("--output", type=str, default="", help="output json path")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    n = str(name).strip().lower()
    if n == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(n)


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    device = pick_device(args.device)
    model, meta = load_multiscale_model(args.model, device=device)
    vars_spec = [parse_var(s) for s in args.var]
    weights = parse_weight(args.weight) if args.weight else None
    opt_cfg = InverseDesignConfig(
        seed=int(args.seed),
        iterations=int(args.iterations),
        population=int(args.population),
        elite_frac=float(args.elite_frac),
        objective_weights=weights if weights is not None else InverseDesignConfig().objective_weights,
        target_hw=(int(args.target_h), int(args.target_w)),
    )
    res = run_inverse_design(
        base_cfg=base_cfg,
        model=model,
        model_meta=meta,
        variables=vars_spec,
        opt_cfg=opt_cfg,
        device=device,
    )
    payload = {
        "best_score": float(res.best_score),
        "best_params": {k: float(v) for k, v in res.best_params.items()},
        "best_targets": {k: float(v) for k, v in res.best_targets.items()},
        "history": res.history,
    }
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

