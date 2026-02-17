"""跨尺度高级模型（U-AFNO / 多保真）模块。

设计目标：
1. 参考 U-AFNO（局部 U-Net + 频域算子混合）思路，构建更强的微结构场编码器；
2. 提供多保真输出头：先预测低保真宏观量，再学习高保真校正；
3. 输出均值+方差，支持不确定性评估与风险感知反向设计。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdvancedModelConfig:
    """高级跨尺度模型结构参数。"""

    in_channels: int
    desc_dim: int
    target_dim: int
    width: int = 64
    depth: int = 3
    afno_blocks: int = 4
    modes_x: int = 16
    modes_y: int = 16
    dropout: float = 0.05
    use_uncertainty_head: bool = True


class SpectralConv2d(nn.Module):
    """FNO 风格的二维频域卷积（截断低频模态）。

    说明：
    - 输入/输出均为 `[B,C,H,W]`；
    - 仅学习低频模态的复权重，兼顾表达能力与稳定性；
    - 用两个频段权重（正低频与负低频）增强方向表达。
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = max(1, int(modes_x))
        self.modes_y = max(1, int(modes_y))
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        # 两组复权重：分别作用于 x 方向低频正/负半区。
        w1 = scale * torch.randn(self.in_channels, self.out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat)
        w2 = scale * torch.randn(self.in_channels, self.out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat)
        self.w_pos = nn.Parameter(w1)
        self.w_neg = nn.Parameter(w2)

    @staticmethod
    def _compl_mul2d(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # inp: [B, Cin, Hm, Wm], weight: [Cin, Cout, Hm, Wm] -> [B, Cout, Hm, Wm]
        return torch.einsum("bixy,ioxy->boxy", inp, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            (b, self.out_channels, h, w // 2 + 1),
            dtype=torch.cfloat,
            device=x.device,
        )
        mx = min(self.modes_x, h // 2 if h > 1 else 1)
        my = min(self.modes_y, w // 2 + 1)
        if mx > 0 and my > 0:
            out_ft[:, :, :mx, :my] = self._compl_mul2d(x_ft[:, :, :mx, :my], self.w_pos[:, :, :mx, :my])
            out_ft[:, :, -mx:, :my] = self._compl_mul2d(x_ft[:, :, -mx:, :my], self.w_neg[:, :, :mx, :my])
        y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return y


class AFNOResidualBlock(nn.Module):
    """带局部卷积分支的频域残差块。"""

    def __init__(self, width: int, modes_x: int, modes_y: int, dropout: float):
        super().__init__()
        w = int(width)
        self.norm1 = nn.GroupNorm(8 if w % 8 == 0 else 1, w)
        self.spec = SpectralConv2d(w, w, modes_x=modes_x, modes_y=modes_y)
        self.loc = nn.Sequential(
            nn.Conv2d(w, w, kernel_size=3, padding=1, groups=max(1, w // 8)),
            nn.GELU(),
            nn.Conv2d(w, w, kernel_size=1),
        )
        self.mix = nn.Conv2d(2 * w, w, kernel_size=1)
        self.dp = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.norm2 = nn.GroupNorm(8 if w % 8 == 0 else 1, w)
        self.ffn = nn.Sequential(
            nn.Conv2d(w, 2 * w, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * w, w, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        z = self.norm1(x)
        zf = self.spec(z)
        zl = self.loc(z)
        z = self.mix(torch.cat([zf, zl], dim=1))
        z = self.dp(z)
        x = r + z
        return x + self.dp(self.ffn(self.norm2(x)))


class DownBlock(nn.Module):
    """下采样块。"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """上采样块（双线性 + 卷积融合 skip）。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class HybridUAFNOBackbone(nn.Module):
    """U-Net + AFNO 混合主干。

    流程：
    - 编码路径提取局部几何与界面信息；
    - 瓶颈路径用 AFNO 残差块学习全局/长程耦合；
    - 解码路径聚合多尺度特征，得到宏观预测所需的统计表征。
    """

    def __init__(self, in_channels: int, width: int, depth: int, afno_blocks: int, modes_x: int, modes_y: int, dropout: float):
        super().__init__()
        w = max(32, int(width))
        d = max(2, int(depth))
        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), w, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(w, w, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down_blocks = nn.ModuleList()
        chs: List[int] = [w]
        cur = w
        for _ in range(d):
            nxt = min(cur * 2, 4 * w)
            self.down_blocks.append(DownBlock(cur, nxt, dropout=dropout))
            cur = nxt
            chs.append(cur)
        self.bottleneck = nn.Sequential(
            *[
                AFNOResidualBlock(
                    width=cur,
                    modes_x=max(4, modes_x // (2 ** d)),
                    modes_y=max(4, modes_y // (2 ** d)),
                    dropout=dropout,
                )
                for _ in range(max(1, int(afno_blocks)))
            ]
        )
        self.up_blocks = nn.ModuleList()
        for i in range(d - 1, -1, -1):
            skip = chs[i]
            out = max(skip, w)
            self.up_blocks.append(UpBlock(cur, skip, out, dropout=dropout))
            cur = out
        self.out_channels = cur
        self.head = nn.Sequential(
            nn.Conv2d(cur, cur, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cur, cur, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = [x]
        for blk in self.down_blocks:
            x = blk(x)
            skips.append(x)
        x = self.bottleneck(x)
        for i, blk in enumerate(self.up_blocks):
            skip = skips[-(i + 2)]
            x = blk(x, skip)
        x = self.head(x)
        return x


class _DescriptorFusionHead(nn.Module):
    """将场特征与描述符融合后输出回归头输入向量。"""

    def __init__(self, feat_dim: int, desc_dim: int, hidden: int, dropout: float):
        super().__init__()
        d = max(1, int(desc_dim))
        h = max(64, int(hidden))
        self.desc_proj = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Linear(int(feat_dim) + h, h),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.out_dim = h

    def forward(self, z_field: torch.Tensor, x_desc: torch.Tensor | None) -> torch.Tensor:
        if x_desc is None:
            x_desc = torch.zeros((z_field.shape[0], 1), device=z_field.device, dtype=z_field.dtype)
        z_desc = self.desc_proj(x_desc)
        return self.mix(torch.cat([z_field, z_desc], dim=1))


class HybridUAFNOMacroPredictor(nn.Module):
    """高级单保真模型：直接预测宏观指标。"""

    def __init__(self, cfg: AdvancedModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = HybridUAFNOBackbone(
            in_channels=int(cfg.in_channels),
            width=int(cfg.width),
            depth=int(cfg.depth),
            afno_blocks=int(cfg.afno_blocks),
            modes_x=int(cfg.modes_x),
            modes_y=int(cfg.modes_y),
            dropout=float(cfg.dropout),
        )
        self.fuser = _DescriptorFusionHead(
            feat_dim=int(self.backbone.out_channels),
            desc_dim=int(cfg.desc_dim),
            hidden=max(96, int(cfg.width) * 2),
            dropout=float(cfg.dropout),
        )
        self.head_mean = nn.Linear(self.fuser.out_dim, int(cfg.target_dim))
        self.use_unc = bool(cfg.use_uncertainty_head)
        self.head_logvar = nn.Linear(self.fuser.out_dim, int(cfg.target_dim)) if self.use_unc else None

    def forward(self, x_field: torch.Tensor, x_desc: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        zmap = self.backbone(x_field)
        z = torch.mean(zmap, dim=(-2, -1))
        h = self.fuser(z, x_desc)
        mean = self.head_mean(h)
        if self.use_unc and self.head_logvar is not None:
            logvar = torch.clamp(self.head_logvar(h), min=-8.0, max=6.0)
        else:
            logvar = torch.zeros_like(mean)
        return {"mean": mean, "logvar": logvar}


class MultiFidelityHybridPredictor(nn.Module):
    """多保真模型：低保真主头 + 高保真校正头。

    输出：
    - `low_mean/low_logvar`：低保真宏观预测；
    - `delta_mean/delta_logvar`：高保真相对低保真的校正；
    - `high_mean/high_logvar`：最终高保真预测（低保真 + 校正）。
    """

    def __init__(self, cfg: AdvancedModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = HybridUAFNOBackbone(
            in_channels=int(cfg.in_channels),
            width=int(cfg.width),
            depth=int(cfg.depth),
            afno_blocks=int(cfg.afno_blocks),
            modes_x=int(cfg.modes_x),
            modes_y=int(cfg.modes_y),
            dropout=float(cfg.dropout),
        )
        self.fuser = _DescriptorFusionHead(
            feat_dim=int(self.backbone.out_channels),
            desc_dim=int(cfg.desc_dim),
            hidden=max(128, int(cfg.width) * 2),
            dropout=float(cfg.dropout),
        )
        h = self.fuser.out_dim
        t = int(cfg.target_dim)
        self.low_mean = nn.Linear(h, t)
        self.low_logvar = nn.Linear(h, t) if bool(cfg.use_uncertainty_head) else None
        self.delta_mean = nn.Linear(h, t)
        self.delta_logvar = nn.Linear(h, t) if bool(cfg.use_uncertainty_head) else None
        self.use_unc = bool(cfg.use_uncertainty_head)

    def forward(self, x_field: torch.Tensor, x_desc: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        zmap = self.backbone(x_field)
        z = torch.mean(zmap, dim=(-2, -1))
        h = self.fuser(z, x_desc)
        low_mean = self.low_mean(h)
        delta_mean = self.delta_mean(h)
        high_mean = low_mean + delta_mean
        if self.use_unc and self.low_logvar is not None and self.delta_logvar is not None:
            low_logvar = torch.clamp(self.low_logvar(h), min=-8.0, max=6.0)
            delta_logvar = torch.clamp(self.delta_logvar(h), min=-8.0, max=6.0)
            # 独立高斯叠加近似：var_high = var_low + var_delta
            high_var = torch.exp(low_logvar) + torch.exp(delta_logvar)
            high_logvar = torch.log(torch.clamp(high_var, min=1e-12))
        else:
            low_logvar = torch.zeros_like(low_mean)
            delta_logvar = torch.zeros_like(delta_mean)
            high_logvar = torch.zeros_like(high_mean)
        return {
            "low_mean": low_mean,
            "low_logvar": low_logvar,
            "delta_mean": delta_mean,
            "delta_logvar": delta_logvar,
            "high_mean": high_mean,
            "high_logvar": high_logvar,
        }

    def predict_high(self, x_field: torch.Tensor, x_desc: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x_field, x_desc)["high_mean"]


def build_advanced_model(model_name: str, cfg: AdvancedModelConfig) -> nn.Module:
    """高级模型工厂。"""
    name = str(model_name).strip().lower()
    if name in {"uafno", "hybrid_uafno", "single"}:
        return HybridUAFNOMacroPredictor(cfg)
    if name in {"multifidelity_uafno", "mf_uafno", "multifidelity"}:
        return MultiFidelityHybridPredictor(cfg)
    raise ValueError(f"Unsupported advanced model name: {model_name}")


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """异方差高斯负对数似然。"""
    inv_var = torch.exp(-logvar)
    return torch.mean(0.5 * (logvar + (target - mean) * (target - mean) * inv_var))


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


def fit_advanced_scalers(x_desc: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    """拟合高级模型用标准化器。"""
    return {
        "x_desc": _scaler_fit(x_desc),
        "y_low": _scaler_fit(y_low),
        "y_high": _scaler_fit(y_high),
    }


def apply_advanced_scalers(
    x_desc: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
    scalers: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """应用高级模型标准化。"""
    xd = _scaler_apply(x_desc, scalers["x_desc"]).astype(np.float32)
    yl = _scaler_apply(y_low, scalers["y_low"]).astype(np.float32)
    yh = _scaler_apply(y_high, scalers["y_high"]).astype(np.float32)
    return xd, yl, yh


def unscale_advanced_targets(
    y_scaled: np.ndarray,
    scalers: Dict[str, Dict[str, np.ndarray]],
    key: str = "y_high",
) -> np.ndarray:
    """将标准化目标还原到物理尺度。"""
    return _scaler_unapply(y_scaled, scalers[key]).astype(np.float32)


def save_advanced_model(
    path: Path | str,
    model: nn.Module,
    cfg: AdvancedModelConfig,
    *,
    model_name: str,
    field_channels: Sequence[str],
    descriptor_names: Sequence[str],
    target_names: Sequence[str],
    scalers: Dict[str, Dict[str, np.ndarray]] | None = None,
    extra_meta: Dict[str, object] | None = None,
) -> Path:
    """保存高级模型权重与元数据。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "model_name": str(model_name),
        "field_channels": list(field_channels),
        "descriptor_names": list(descriptor_names),
        "target_names": list(target_names),
        "scalers": scalers if scalers is not None else {},
        "extra_meta": extra_meta if extra_meta is not None else {},
    }
    torch.save(payload, p)
    return p


def load_advanced_model(path: Path | str, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    """加载高级模型。"""
    p = Path(path)
    payload = torch.load(p, map_location=device)
    cfg = AdvancedModelConfig(**payload["config"])
    model_name = str(payload.get("model_name", "uafno"))
    model = build_advanced_model(model_name, cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    meta = {
        "model_name": model_name,
        "config": payload.get("config", {}),
        "field_channels": list(payload.get("field_channels", [])),
        "descriptor_names": list(payload.get("descriptor_names", [])),
        "target_names": list(payload.get("target_names", [])),
        "scalers": payload.get("scalers", {}),
        "extra_meta": payload.get("extra_meta", {}),
    }
    return model, meta
