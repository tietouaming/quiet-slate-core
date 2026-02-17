"""跨尺度训练与评估可视化工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_training_history(history: Sequence[Dict[str, float]], out_path: Path | str) -> Path:
    """绘制训练/验证损失与核心指标曲线。"""
    p = Path(out_path)
    _ensure_parent(p)
    if not history:
        raise ValueError("Empty history.")
    ep = [int(h.get("epoch", i + 1)) for i, h in enumerate(history)]
    tr = [float(h.get("train_loss", np.nan)) for h in history]
    va = [float(h.get("val_loss", np.nan)) for h in history]
    rmse = [float(h.get("val_rmse", np.nan)) for h in history]
    r2 = [float(h.get("val_r2", np.nan)) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    axes[0].plot(ep, tr, label="train_loss", lw=1.8)
    axes[0].plot(ep, va, label="val_loss", lw=1.8)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, rmse, label="val_rmse", lw=1.8)
    axes[1].plot(ep, r2, label="val_r2", lw=1.8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_parity_by_target(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str],
    out_dir: Path | str,
    prefix: str = "parity",
) -> List[Path]:
    """按目标分别绘制真值-预测散点图。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    t = min(yt.shape[1], yp.shape[1], len(target_names))
    paths: List[Path] = []
    for j in range(t):
        name = str(target_names[j])
        p = out / f"{prefix}_{j:02d}_{name}.png"
        fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=140)
        x = yt[:, j]
        y = yp[:, j]
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        pad = 0.02 * max(hi - lo, 1.0)
        lo -= pad
        hi += pad
        ax.scatter(x, y, s=10, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.set_title(f"Parity: {name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_uncertainty_calibration(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_logvar: np.ndarray,
    target_names: Sequence[str],
    out_path: Path | str,
) -> Path:
    """绘制 2-sigma 覆盖率柱状图与平均区间宽度。"""
    p = Path(out_path)
    _ensure_parent(p)
    yt = np.asarray(y_true, dtype=np.float64)
    ym = np.asarray(y_mean, dtype=np.float64)
    ys = np.sqrt(np.exp(np.asarray(y_logvar, dtype=np.float64)))
    t = min(yt.shape[1], ym.shape[1], ys.shape[1], len(target_names))
    cover: List[float] = []
    width: List[float] = []
    names: List[str] = []
    for j in range(t):
        lo = ym[:, j] - 2.0 * ys[:, j]
        hi = ym[:, j] + 2.0 * ys[:, j]
        c = np.mean(((yt[:, j] >= lo) & (yt[:, j] <= hi)).astype(np.float64))
        w = np.mean((hi - lo).astype(np.float64))
        cover.append(float(c))
        width.append(float(w))
        names.append(str(target_names[j]))

    x = np.arange(t)
    fig, ax1 = plt.subplots(figsize=(max(7.0, 1.1 * t), 4.2), dpi=140)
    ax1.bar(x - 0.15, cover, width=0.3, label="2σ coverage")
    ax1.axhline(0.9545, color="r", linestyle="--", linewidth=1.0, label="ideal 95.45%")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("coverage")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(x + 0.15, width, width=0.3, alpha=0.55, color="tab:orange", label="mean interval width")
    ax2.set_ylabel("interval width")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p

