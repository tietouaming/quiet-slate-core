"""跨尺度模型：微结构场 -> 宏观指标。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiScaleModelConfig:
    """跨尺度模型结构配置。"""

    in_channels: int
    desc_dim: int
    target_dim: int
    encoder_width: int = 48
    encoder_depth: int = 4
    fusion_hidden: int = 192
    fusion_depth: int = 3
    dropout: float = 0.05
    use_uncertainty_head: bool = True


class _ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.n1 = nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)
        self.n2 = nn.GroupNorm(8 if ch % 8 == 0 else 1, ch)
        self.dp = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = F.gelu(self.n1(self.c1(x)))
        x = self.dp(x)
        x = self.n2(self.c2(x))
        return F.gelu(x + r)


class _MicroEncoder(nn.Module):
    """二维微结构场编码器。"""

    def __init__(self, in_ch: int, width: int, depth: int, dropout: float):
        super().__init__()
        w = max(16, int(width))
        d = max(1, int(depth))
        self.stem = nn.Conv2d(in_ch, w, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([_ResBlock(w, dropout=dropout) for _ in range(d)])
        self.down = nn.ModuleList()
        for _ in range(max(1, d // 2)):
            self.down.append(nn.Sequential(nn.Conv2d(w, w, kernel_size=3, stride=2, padding=1), nn.GELU()))
            self.down.append(_ResBlock(w, dropout=dropout))
        self.out_ch = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.stem(x))
        for blk in self.blocks:
            x = blk(x)
        for blk in self.down:
            x = blk(x)
        # 全局池化得到固定维度 latent
        return torch.mean(x, dim=(-2, -1))


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        h = max(16, int(hidden))
        d = max(1, int(depth))
        layers: List[nn.Module] = []
        c = int(in_dim)
        for _ in range(d - 1):
            layers += [nn.Linear(c, h), nn.GELU(), nn.Dropout(float(dropout))]
            c = h
        layers += [nn.Linear(c, int(out_dim))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MacroCorrosionPredictor(nn.Module):
    """跨尺度宏观指标预测器。

    输入：
    - `x_field`: `[B,C,H,W]`
    - `x_desc`:  `[B,D]`（可选）

    输出：
    - `mean`: `[B,T]` 目标均值
    - `logvar`: `[B,T]` 不确定性头（可选）
    """

    def __init__(self, cfg: MultiScaleModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = _MicroEncoder(
            in_ch=int(cfg.in_channels),
            width=int(cfg.encoder_width),
            depth=int(cfg.encoder_depth),
            dropout=float(cfg.dropout),
        )
        self.desc_proj = _MLP(
            in_dim=max(1, int(cfg.desc_dim)),
            out_dim=max(16, int(cfg.encoder_width)),
            hidden=max(16, int(cfg.encoder_width)),
            depth=2,
            dropout=float(cfg.dropout),
        )
        fuse_in = int(self.encoder.out_ch) + max(16, int(cfg.encoder_width))
        self.fusion = _MLP(
            in_dim=fuse_in,
            out_dim=max(32, int(cfg.fusion_hidden)),
            hidden=max(32, int(cfg.fusion_hidden)),
            depth=max(2, int(cfg.fusion_depth)),
            dropout=float(cfg.dropout),
        )
        h_out = max(32, int(cfg.fusion_hidden))
        self.head_mean = nn.Linear(h_out, int(cfg.target_dim))
        self.use_unc = bool(cfg.use_uncertainty_head)
        self.head_logvar = nn.Linear(h_out, int(cfg.target_dim)) if self.use_unc else None

    def forward(self, x_field: torch.Tensor, x_desc: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        zf = self.encoder(x_field)
        if x_desc is None:
            x_desc = torch.zeros((x_field.shape[0], max(1, int(self.cfg.desc_dim))), device=x_field.device, dtype=x_field.dtype)
        zd = self.desc_proj(x_desc)
        z = torch.cat([zf, zd], dim=1)
        h = self.fusion(z)
        mean = self.head_mean(h)
        if self.use_unc and self.head_logvar is not None:
            logvar = torch.clamp(self.head_logvar(h), min=-8.0, max=6.0)
        else:
            logvar = torch.zeros_like(mean)
        return {"mean": mean, "logvar": logvar}

    def predict_mean(self, x_field: torch.Tensor, x_desc: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x_field, x_desc)["mean"]


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """异方差高斯 NLL。"""
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (target - mean) * (target - mean) * inv_var)
    return torch.mean(nll)


def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """回归指标（MAE/RMSE/R2）。"""
    yt = y_true.detach()
    yp = y_pred.detach()
    mae = torch.mean(torch.abs(yp - yt))
    rmse = torch.sqrt(torch.mean((yp - yt) ** 2))
    var = torch.mean((yt - torch.mean(yt, dim=0, keepdim=True)) ** 2)
    mse = torch.mean((yp - yt) ** 2)
    r2 = 1.0 - mse / torch.clamp(var, min=1e-12)
    return {"mae": float(mae.item()), "rmse": float(rmse.item()), "r2": float(r2.item())}


def _scaler_fit(x: np.ndarray) -> Dict[str, np.ndarray]:
    mu = np.mean(x, axis=0, keepdims=True).astype(np.float32)
    std = np.std(x, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return {"mean": mu, "std": std}


def _scaler_apply(x: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - scaler["mean"]) / scaler["std"]


def _scaler_unapply(x: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return x * scaler["std"] + scaler["mean"]


def fit_scalers(
    x_desc: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """拟合描述符与目标的标准化器。"""
    return {"x_desc": _scaler_fit(x_desc), "y": _scaler_fit(y)}


def apply_scalers(
    x_desc: np.ndarray,
    y: np.ndarray,
    scalers: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """应用标准化。"""
    xd = _scaler_apply(x_desc, scalers["x_desc"])
    yy = _scaler_apply(y, scalers["y"])
    return xd.astype(np.float32), yy.astype(np.float32)


def unscale_targets(y_scaled: np.ndarray, scalers: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    """将标准化目标还原到物理尺度。"""
    return _scaler_unapply(y_scaled, scalers["y"]).astype(np.float32)


def save_multiscale_model(
    path: Path | str,
    model: MacroCorrosionPredictor,
    cfg: MultiScaleModelConfig,
    *,
    field_channels: Sequence[str],
    descriptor_names: Sequence[str],
    target_names: Sequence[str],
    scalers: Dict[str, Dict[str, np.ndarray]] | None = None,
) -> Path:
    """保存跨尺度模型权重与元数据。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "field_channels": list(field_channels),
        "descriptor_names": list(descriptor_names),
        "target_names": list(target_names),
        "scalers": scalers if scalers is not None else {},
    }
    torch.save(payload, p)
    return p


def load_multiscale_model(
    path: Path | str,
    device: torch.device,
) -> Tuple[MacroCorrosionPredictor, Dict[str, object]]:
    """加载跨尺度模型。"""
    p = Path(path)
    payload = torch.load(p, map_location=device)
    cfg = MultiScaleModelConfig(**payload["config"])
    model = MacroCorrosionPredictor(cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    meta = {
        "field_channels": list(payload.get("field_channels", [])),
        "descriptor_names": list(payload.get("descriptor_names", [])),
        "target_names": list(payload.get("target_names", [])),
        "scalers": payload.get("scalers", {}),
        "config": payload.get("config", {}),
    }
    return model, meta

