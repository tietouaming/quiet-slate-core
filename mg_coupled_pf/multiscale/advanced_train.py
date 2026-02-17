"""高级跨尺度模型训练器（多保真 + 物理一致性）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .advanced_models import (
    AdvancedModelConfig,
    apply_advanced_scalers,
    build_advanced_model,
    fit_advanced_scalers,
    gaussian_nll,
    regression_metrics,
    unscale_advanced_targets,
)
from .physics_metrics import PhysicsMetricConfig, evaluate_physics_metrics


@dataclass
class AdvancedTrainingConfig:
    """高级训练配置。"""

    seed: int = 42
    epochs: int = 120
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-5
    train_split: float = 0.9
    grad_clip: float = 1.0
    use_amp: bool = False
    model_name: str = "multifidelity_uafno"
    # 损失权重
    low_loss_weight: float = 0.6
    high_loss_weight: float = 1.0
    l2_weight: float = 0.05
    bound_penalty_weight: float = 0.02


def _split_indices(n: int, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(seed)
    idx = np.arange(n)
    rs.shuffle(idx)
    n_tr = int(max(1, min(n - 1, round(n * frac))))
    return idx[:n_tr], idx[n_tr:]


def _build_scaled_bounds(
    target_names: np.ndarray,
    y_scaler: Dict[str, np.ndarray],
    p_cfg: PhysicsMetricConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将目标边界转换到标准化空间，便于训练时快速计算约束损失。"""
    names = [str(x) for x in target_names.tolist()]
    mu = np.asarray(y_scaler["mean"], dtype=np.float32).reshape(-1)
    sd = np.asarray(y_scaler["std"], dtype=np.float32).reshape(-1)
    lo = np.full((len(names),), -np.inf, dtype=np.float32)
    hi = np.full((len(names),), np.inf, dtype=np.float32)
    for i, name in enumerate(names):
        bd = p_cfg.bounds.get(name)
        if bd is None:
            continue
        if bd.lower is not None:
            lo[i] = (float(bd.lower) - mu[i]) / max(sd[i], 1e-12)
        if bd.upper is not None:
            hi[i] = (float(bd.upper) - mu[i]) / max(sd[i], 1e-12)
    lo_t = torch.tensor(lo[None, :], device=device, dtype=torch.float32)
    hi_t = torch.tensor(hi[None, :], device=device, dtype=torch.float32)
    return lo_t, hi_t


def _bound_penalty(y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """边界软惩罚：超出边界时线性增长。"""
    p_lo = torch.relu(lo - y)
    p_hi = torch.relu(y - hi)
    return torch.mean(p_lo + p_hi)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=x.dtype)
    num = torch.sum(x * m)
    den = torch.clamp(torch.sum(m), min=1.0)
    return num / den


def train_advanced_multiscale_model(
    dataset: Dict[str, np.ndarray],
    model_cfg: AdvancedModelConfig,
    train_cfg: AdvancedTrainingConfig,
    *,
    device: torch.device,
    physics_cfg: PhysicsMetricConfig | None = None,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    """训练高级跨尺度模型并返回报告。"""
    p_cfg = physics_cfg if physics_cfg is not None else PhysicsMetricConfig()
    x_field = dataset["x_field"].astype(np.float32)
    x_desc = dataset["x_desc"].astype(np.float32)
    y_high = dataset["y"].astype(np.float32)
    y_low = dataset["y_low"].astype(np.float32) if "y_low" in dataset else y_high.copy()
    high_mask = dataset["high_mask"].astype(np.float32) if "high_mask" in dataset else np.ones((y_high.shape[0],), dtype=np.float32)
    if high_mask.ndim == 1:
        high_mask = high_mask[:, None]

    n = int(x_field.shape[0])
    if n < 6:
        raise RuntimeError(f"dataset too small: {n} samples")

    tr_idx, va_idx = _split_indices(n, train_cfg.train_split, train_cfg.seed)
    scalers = fit_advanced_scalers(x_desc[tr_idx], y_low[tr_idx], y_high[tr_idx])
    x_desc_s, y_low_s, y_high_s = apply_advanced_scalers(x_desc, y_low, y_high, scalers)

    xt_field = torch.from_numpy(x_field)
    xt_desc = torch.from_numpy(x_desc_s)
    y_low_t = torch.from_numpy(y_low_s)
    y_high_t = torch.from_numpy(y_high_s)
    high_mask_t = torch.from_numpy(high_mask.astype(np.float32))

    tr_ds = TensorDataset(
        xt_field[tr_idx],
        xt_desc[tr_idx],
        y_low_t[tr_idx],
        y_high_t[tr_idx],
        high_mask_t[tr_idx],
    )
    va_ds = TensorDataset(
        xt_field[va_idx],
        xt_desc[va_idx],
        y_low_t[va_idx],
        y_high_t[va_idx],
        high_mask_t[va_idx],
    )
    tr_loader = DataLoader(tr_ds, batch_size=max(1, train_cfg.batch_size), shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=max(1, train_cfg.batch_size), shuffle=False, drop_last=False)

    torch.manual_seed(int(train_cfg.seed))
    model = build_advanced_model(train_cfg.model_name, model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.use_amp and device.type == "cuda"))

    lo_s, hi_s = _build_scaled_bounds(dataset["target_names"], scalers["y_high"], p_cfg, device=device)

    best = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    hist = []
    best_phy: Dict[str, float] = {}

    for ep in range(1, int(train_cfg.epochs) + 1):
        model.train()
        tr_loss = 0.0
        tr_count = 0
        for xb, db, yl, yh, hm in tr_loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            db = db.to(device=device, dtype=torch.float32, non_blocking=True)
            yl = yl.to(device=device, dtype=torch.float32, non_blocking=True)
            yh = yh.to(device=device, dtype=torch.float32, non_blocking=True)
            hm = hm.to(device=device, dtype=torch.float32, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(train_cfg.use_amp and device.type == "cuda")):
                out = model(xb, db)
                if "high_mean" in out:
                    low_mean = out["low_mean"]
                    low_logvar = out["low_logvar"]
                    high_mean = out["high_mean"]
                    high_logvar = out["high_logvar"]
                else:
                    low_mean = out["mean"]
                    low_logvar = out["logvar"]
                    high_mean = out["mean"]
                    high_logvar = out["logvar"]
                low_nll = gaussian_nll(low_mean, low_logvar, yl)
                high_nll_all = 0.5 * (high_logvar + (yh - high_mean) * (yh - high_mean) * torch.exp(-high_logvar))
                high_nll = _masked_mean(high_nll_all, hm)
                l2 = torch.mean((high_mean - yh) ** 2)
                bpen = _bound_penalty(high_mean, lo_s, hi_s)
                loss = (
                    float(train_cfg.low_loss_weight) * low_nll
                    + float(train_cfg.high_loss_weight) * high_nll
                    + float(train_cfg.l2_weight) * l2
                    + float(train_cfg.bound_penalty_weight) * bpen
                )
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
        y_true_h = []
        y_pred_h = []
        y_logv_h = []
        with torch.no_grad():
            for xb, db, yl, yh, hm in va_loader:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
                db = db.to(device=device, dtype=torch.float32, non_blocking=True)
                yl = yl.to(device=device, dtype=torch.float32, non_blocking=True)
                yh = yh.to(device=device, dtype=torch.float32, non_blocking=True)
                hm = hm.to(device=device, dtype=torch.float32, non_blocking=True)
                out = model(xb, db)
                if "high_mean" in out:
                    low_mean = out["low_mean"]
                    low_logvar = out["low_logvar"]
                    high_mean = out["high_mean"]
                    high_logvar = out["high_logvar"]
                else:
                    low_mean = out["mean"]
                    low_logvar = out["logvar"]
                    high_mean = out["mean"]
                    high_logvar = out["logvar"]
                low_nll = gaussian_nll(low_mean, low_logvar, yl)
                high_nll_all = 0.5 * (high_logvar + (yh - high_mean) * (yh - high_mean) * torch.exp(-high_logvar))
                high_nll = _masked_mean(high_nll_all, hm)
                l2 = torch.mean((high_mean - yh) ** 2)
                bpen = _bound_penalty(high_mean, lo_s, hi_s)
                loss = (
                    float(train_cfg.low_loss_weight) * low_nll
                    + float(train_cfg.high_loss_weight) * high_nll
                    + float(train_cfg.l2_weight) * l2
                    + float(train_cfg.bound_penalty_weight) * bpen
                )
                bs = int(xb.shape[0])
                va_loss += float(loss.item()) * bs
                va_count += bs
                y_true_h.append(yh.detach().cpu())
                y_pred_h.append(high_mean.detach().cpu())
                y_logv_h.append(high_logvar.detach().cpu())
        va_loss /= max(va_count, 1)

        yt_s = torch.cat(y_true_h, dim=0).numpy()
        yp_s = torch.cat(y_pred_h, dim=0).numpy()
        yv_s = torch.cat(y_logv_h, dim=0).numpy()
        yt = unscale_advanced_targets(yt_s, scalers, key="y_high")
        yp = unscale_advanced_targets(yp_s, scalers, key="y_high")
        # 方差缩放：var_phys = var_scaled * std^2
        std_high = np.asarray(scalers["y_high"]["std"], dtype=np.float32).reshape(1, -1)
        yv = np.exp(yv_s) * (std_high ** 2)
        ylv = np.log(np.maximum(yv, 1e-18))
        met = regression_metrics(torch.from_numpy(yt), torch.from_numpy(yp))
        phy = evaluate_physics_metrics(
            y_true=yt,
            y_mean=yp,
            y_logvar=ylv,
            target_names=[str(v) for v in dataset["target_names"].tolist()],
            case_ids=[str(v) for v in dataset["case_ids"][va_idx].tolist()],
            t_inputs=dataset["t_input"][va_idx].astype(np.float64),
            cfg=p_cfg,
        )

        hist.append(
            {
                "epoch": ep,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_mae": met["mae"],
                "val_rmse": met["rmse"],
                "val_r2": met["r2"],
                "val_pci": float(phy.get("physics_consistency_index", 0.0)),
                "val_violation_rate": float(phy.get("overall_violation_rate", 0.0)),
                "val_monotonic_score": float(phy.get("overall_monotonic_score", 0.0)),
                "val_unc_coverage": float(phy.get("unc_coverage", 0.0)),
            }
        )
        if va_loss < best:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_phy = dict(phy)

    model.load_state_dict(best_state, strict=True)
    report: Dict[str, object] = {
        "best_val_loss": float(best),
        "history": hist,
        "scalers": scalers,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "best_physics_metrics": best_phy,
    }
    return model, report
