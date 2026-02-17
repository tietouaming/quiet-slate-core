"""跨尺度模型训练器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import (
    MacroCorrosionPredictor,
    MultiScaleModelConfig,
    apply_scalers,
    fit_scalers,
    gaussian_nll,
    regression_metrics,
)


@dataclass
class TrainingConfig:
    """训练超参数。"""

    seed: int = 42
    epochs: int = 60
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-5
    train_split: float = 0.9
    grad_clip: float = 1.0
    loss_l2_weight: float = 0.1
    use_amp: bool = False


def _split_indices(n: int, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(seed)
    idx = np.arange(n)
    rs.shuffle(idx)
    n_tr = int(max(1, min(n - 1, round(n * frac))))
    return idx[:n_tr], idx[n_tr:]


def train_multiscale_model(
    dataset: Dict[str, np.ndarray],
    model_cfg: MultiScaleModelConfig,
    train_cfg: TrainingConfig,
    *,
    device: torch.device,
) -> Tuple[MacroCorrosionPredictor, Dict[str, object]]:
    """训练跨尺度模型，返回模型与训练报告。"""
    x_field = dataset["x_field"].astype(np.float32)
    x_desc = dataset["x_desc"].astype(np.float32)
    y = dataset["y"].astype(np.float32)
    n = int(x_field.shape[0])
    if n < 4:
        raise RuntimeError(f"dataset too small: {n} samples")

    tr_idx, va_idx = _split_indices(n, train_cfg.train_split, train_cfg.seed)
    scalers = fit_scalers(x_desc[tr_idx], y[tr_idx])
    x_desc_s, y_s = apply_scalers(x_desc, y, scalers)

    xt_field = torch.from_numpy(x_field)
    xt_desc = torch.from_numpy(x_desc_s)
    yt = torch.from_numpy(y_s)

    tr_ds = TensorDataset(xt_field[tr_idx], xt_desc[tr_idx], yt[tr_idx])
    va_ds = TensorDataset(xt_field[va_idx], xt_desc[va_idx], yt[va_idx])
    tr_loader = DataLoader(tr_ds, batch_size=max(1, train_cfg.batch_size), shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=max(1, train_cfg.batch_size), shuffle=False, drop_last=False)

    torch.manual_seed(int(train_cfg.seed))
    model = MacroCorrosionPredictor(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.use_amp and device.type == "cuda"))

    best = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hist = []

    for ep in range(1, int(train_cfg.epochs) + 1):
        model.train()
        tr_loss = 0.0
        tr_count = 0
        for xb, db, yb in tr_loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            db = db.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(train_cfg.use_amp and device.type == "cuda")):
                out = model(xb, db)
                loss_main = gaussian_nll(out["mean"], out["logvar"], yb)
                loss_l2 = torch.mean((out["mean"] - yb) ** 2)
                loss = loss_main + float(train_cfg.loss_l2_weight) * loss_l2
            scaler.scale(loss).backward()
            if train_cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
            bs = int(xb.shape[0])
            tr_loss += float(loss.item()) * bs
            tr_count += bs
        tr_loss /= max(tr_count, 1)

        model.eval()
        va_loss = 0.0
        va_count = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, db, yb in va_loader:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
                db = db.to(device=device, dtype=torch.float32, non_blocking=True)
                yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
                out = model(xb, db)
                loss_main = gaussian_nll(out["mean"], out["logvar"], yb)
                loss_l2 = torch.mean((out["mean"] - yb) ** 2)
                loss = loss_main + float(train_cfg.loss_l2_weight) * loss_l2
                bs = int(xb.shape[0])
                va_loss += float(loss.item()) * bs
                va_count += bs
                y_true.append(yb.detach().cpu())
                y_pred.append(out["mean"].detach().cpu())
        va_loss /= max(va_count, 1)
        yt_cpu = torch.cat(y_true, dim=0)
        yp_cpu = torch.cat(y_pred, dim=0)
        met = regression_metrics(yt_cpu, yp_cpu)
        hist.append(
            {
                "epoch": ep,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_mae": met["mae"],
                "val_rmse": met["rmse"],
                "val_r2": met["r2"],
            }
        )
        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    report: Dict[str, object] = {
        "best_val_loss": float(best),
        "history": hist,
        "scalers": scalers,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
    }
    return model, report
