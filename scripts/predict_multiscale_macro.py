"""使用跨尺度模型预测宏观指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mg_coupled_pf.multiscale.features import descriptor_vector, extract_micro_descriptors, extract_micro_tensor
from mg_coupled_pf.multiscale.models import load_multiscale_model, unscale_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict macro corrosion/mechanics metrics from micro snapshot.")
    p.add_argument("--model", type=str, required=True, help="trained multiscale model checkpoint")
    p.add_argument("--snapshot", type=str, required=True, help="input snapshot npz")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--target-h", type=int, default=96)
    p.add_argument("--target-w", type=int, default=96)
    p.add_argument("--output", type=str, default="", help="optional output json path")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    n = str(name).strip().lower()
    if n == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(n)


def main() -> None:
    args = parse_args()
    dev = pick_device(args.device)
    model, meta = load_multiscale_model(args.model, dev)
    snap = np.load(Path(args.snapshot), allow_pickle=False)
    data = {k: snap[k] for k in snap.files}
    field_channels = list(meta.get("field_channels", []))
    desc_names = list(meta.get("descriptor_names", []))
    target_names = list(meta.get("target_names", []))
    scalers = meta.get("scalers", {})

    x_field = extract_micro_tensor(
        data,
        channels=field_channels if field_channels else ("phi", "c", "eta", "epspeq"),
        target_hw=(int(args.target_h), int(args.target_w)),
    )[None]
    desc = extract_micro_descriptors(data)
    x_desc_np = descriptor_vector(desc, desc_names if desc_names else sorted(desc.keys()))[None]
    x_desc = torch.from_numpy(x_desc_np.astype(np.float32))

    with torch.no_grad():
        xf = x_field.to(device=dev, dtype=torch.float32)
        xd = x_desc.to(device=dev, dtype=torch.float32)
        if isinstance(scalers, dict) and "x_desc" in scalers and "mean" in scalers["x_desc"] and len(desc_names) > 0:
            mu = torch.tensor(np.asarray(scalers["x_desc"]["mean"], dtype=np.float32), device=dev)
            sd = torch.tensor(np.asarray(scalers["x_desc"]["std"], dtype=np.float32), device=dev)
            xd = (xd - mu) / sd
        out = model(xf, xd)
        y = out["mean"].detach().cpu().numpy()
        if isinstance(scalers, dict) and "y" in scalers and "mean" in scalers["y"]:
            y = unscale_targets(y, {"y": scalers["y"]})
    pred = {str(target_names[i] if i < len(target_names) else f"target_{i}"): float(y[0, i]) for i in range(y.shape[1])}
    result = {"snapshot": str(Path(args.snapshot)), "prediction": pred}
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

