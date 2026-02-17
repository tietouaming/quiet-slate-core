"""训练跨尺度微结构->宏观指标预测模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mg_coupled_pf.multiscale.dataset import load_multiscale_dataset_npz
from mg_coupled_pf.multiscale.models import MultiScaleModelConfig, save_multiscale_model
from mg_coupled_pf.multiscale.train import TrainingConfig, train_multiscale_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multiscale surrogate model (micro field -> macro targets).")
    p.add_argument("--dataset", type=str, required=True, help="multiscale dataset npz")
    p.add_argument("--output-model", type=str, required=True, help="checkpoint output path")
    p.add_argument("--output-report", type=str, default="", help="optional train report json path")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--encoder-width", type=int, default=48)
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--fusion-hidden", type=int, default=192)
    p.add_argument("--fusion-depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-uncertainty-head", action="store_true")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    n = str(name).strip().lower()
    if n == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(n)


def main() -> None:
    args = parse_args()
    ds = load_multiscale_dataset_npz(args.dataset)
    dev = pick_device(args.device)
    x_field = ds["x_field"]
    x_desc = ds["x_desc"]
    y = ds["y"]
    cfg_model = MultiScaleModelConfig(
        in_channels=int(x_field.shape[1]),
        desc_dim=int(x_desc.shape[1]),
        target_dim=int(y.shape[1]),
        encoder_width=int(args.encoder_width),
        encoder_depth=int(args.encoder_depth),
        fusion_hidden=int(args.fusion_hidden),
        fusion_depth=int(args.fusion_depth),
        dropout=float(args.dropout),
        use_uncertainty_head=not bool(args.no_uncertainty_head),
    )
    cfg_train = TrainingConfig(
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        train_split=float(args.train_split),
    )
    model, report = train_multiscale_model(ds, cfg_model, cfg_train, device=dev)
    out_model = Path(args.output_model)
    save_multiscale_model(
        out_model,
        model,
        cfg_model,
        field_channels=[str(v) for v in ds.get("field_channels", np.asarray([], dtype=object)).tolist()],
        descriptor_names=[str(v) for v in ds.get("descriptor_names", np.asarray([], dtype=object)).tolist()],
        target_names=[str(v) for v in ds.get("target_names", np.asarray([], dtype=object)).tolist()],
        scalers=report.get("scalers"),
    )
    out_report = Path(args.output_report) if args.output_report else out_model.with_suffix(".train_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(
        {
            "model": str(out_model),
            "report": str(out_report),
            "best_val_loss": float(report["best_val_loss"]),
            "n_train": int(report["n_train"]),
            "n_val": int(report["n_val"]),
        }
    )


if __name__ == "__main__":
    main()

