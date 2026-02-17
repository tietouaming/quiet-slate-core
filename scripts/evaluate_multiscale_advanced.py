"""评估高级跨尺度模型并输出物理一致性报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mg_coupled_pf.multiscale import (
    PhysicsMetricConfig,
    evaluate_physics_metrics,
    load_advanced_model,
    load_multiscale_dataset_npz,
)
from mg_coupled_pf.multiscale.advanced_models import unscale_advanced_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate advanced multiscale model on dataset.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    s = str(name).strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, meta = load_advanced_model(args.model, device=device)
    ds = load_multiscale_dataset_npz(args.dataset)
    scalers = meta.get("scalers", {})

    x_field = torch.from_numpy(ds["x_field"].astype(np.float32)).to(device=device)
    x_desc = torch.from_numpy(ds["x_desc"].astype(np.float32)).to(device=device)
    if "x_desc" in scalers and "mean" in scalers["x_desc"]:
        mu = torch.from_numpy(np.asarray(scalers["x_desc"]["mean"], dtype=np.float32)).to(device=device)
        sd = torch.from_numpy(np.asarray(scalers["x_desc"]["std"], dtype=np.float32)).to(device=device)
        x_desc = (x_desc - mu) / sd
    with torch.no_grad():
        out = model(x_field, x_desc)
    if "high_mean" in out:
        y_s = out["high_mean"].detach().cpu().numpy()
        lv_s = out["high_logvar"].detach().cpu().numpy()
    else:
        y_s = out["mean"].detach().cpu().numpy()
        lv_s = out["logvar"].detach().cpu().numpy()
    y_pred = unscale_advanced_targets(y_s, scalers, key="y_high")
    std = np.asarray(scalers["y_high"]["std"], dtype=np.float32).reshape(1, -1)
    y_logvar = np.log(np.maximum(np.exp(lv_s) * (std ** 2), 1e-18))
    y_true = ds["y"].astype(np.float32)

    metrics = evaluate_physics_metrics(
        y_true=y_true,
        y_mean=y_pred,
        y_logvar=y_logvar,
        target_names=[str(v) for v in ds["target_names"].tolist()],
        case_ids=[str(v) for v in ds["case_ids"].tolist()],
        t_inputs=ds["t_input"].astype(np.float64),
        cfg=PhysicsMetricConfig(),
    )
    reg = {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(1.0 - np.mean((y_true - y_pred) ** 2) / max(np.var(y_true), 1e-12)),
    }
    payload = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "device": str(device),
        "n_samples": int(y_true.shape[0]),
        "regression": reg,
        "physics_metrics": metrics,
    }
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
