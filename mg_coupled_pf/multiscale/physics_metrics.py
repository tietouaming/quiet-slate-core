"""跨尺度宏观模型的物理一致性指标。

背景：
- 当缺少大规模“高精度宏观标签”时，仅靠 MAE/RMSE 容易高估模型可信度；
- 本模块提供一组可解释指标，用于评估预测是否满足基础物理约束与时序规律。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass
class TargetBounds:
    """目标物理边界定义。

    `None` 表示该侧不约束。
    """

    lower: float | None = None
    upper: float | None = None


@dataclass
class PhysicsMetricConfig:
    """物理一致性指标配置。"""

    # 目标边界：用于统计越界率。
    bounds: Dict[str, TargetBounds] = field(
        default_factory=lambda: {
            "corrosion_loss": TargetBounds(0.0, 1.0),
            "twin_fraction": TargetBounds(0.0, 1.0),
            "penetration_x_ratio": TargetBounds(0.0, 1.0),
            "epspeq_mean": TargetBounds(0.0, None),
            "sigma_h_max_solid": TargetBounds(0.0, None),
        }
    )
    # 时序单调约束：+1 表示应随时间非减，-1 表示应随时间非增。
    monotonic_targets: Dict[str, int] = field(
        default_factory=lambda: {
            "corrosion_loss": +1,
            "penetration_x_ratio": +1,
        }
    )
    # 2-sigma 覆盖率目标（正态近似下约 95.45%）
    expected_coverage_2sigma: float = 0.9545
    # PCI 组合权重
    w_bound: float = 0.40
    w_monotonic: float = 0.35
    w_calibration: float = 0.25


def _safe_name_to_index(names: Sequence[str]) -> Dict[str, int]:
    return {str(n): i for i, n in enumerate(names)}


def bound_violation_rate(
    y_pred: np.ndarray,
    target_names: Sequence[str],
    bounds: Mapping[str, TargetBounds],
) -> Dict[str, float]:
    """统计每个目标越界率及总体越界率。"""
    yp = np.asarray(y_pred, dtype=np.float64)
    idx = _safe_name_to_index(target_names)
    out: Dict[str, float] = {}
    all_mask = np.zeros_like(yp, dtype=np.bool_)
    all_bad = np.zeros_like(yp, dtype=np.bool_)
    for name, bd in bounds.items():
        if name not in idx:
            continue
        j = idx[name]
        mask = np.ones((yp.shape[0],), dtype=np.bool_)
        bad = np.zeros((yp.shape[0],), dtype=np.bool_)
        if bd.lower is not None:
            bad |= yp[:, j] < float(bd.lower)
        if bd.upper is not None:
            bad |= yp[:, j] > float(bd.upper)
        out[f"{name}_violation_rate"] = float(np.mean(bad.astype(np.float64)))
        all_mask[:, j] = mask
        all_bad[:, j] = bad
    used = np.any(all_mask, axis=1)
    if np.any(used):
        out["overall_violation_rate"] = float(np.mean(np.any(all_bad[used], axis=1).astype(np.float64)))
    else:
        out["overall_violation_rate"] = 0.0
    return out


def monotonic_trend_score(
    y_pred: np.ndarray,
    target_names: Sequence[str],
    case_ids: Sequence[str],
    t_inputs: Sequence[float],
    monotonic_targets: Mapping[str, int],
) -> Dict[str, float]:
    """统计同一算例内的时序单调一致率。

    定义：
    - 对同一 case 的样本按时间排序；
    - 对每个相邻时刻比较 `y_{k+1}-y_k` 的符号与先验方向是否一致；
    - 一致率越高越好。
    """
    yp = np.asarray(y_pred, dtype=np.float64)
    ids = np.asarray(case_ids)
    tt = np.asarray(t_inputs, dtype=np.float64)
    idx = _safe_name_to_index(target_names)
    out: Dict[str, float] = {}
    uniq = np.unique(ids)
    if uniq.size == 0:
        for k in monotonic_targets.keys():
            out[f"{k}_monotonic_score"] = 0.0
        out["overall_monotonic_score"] = 0.0
        return out
    scores: List[float] = []
    for name, sign in monotonic_targets.items():
        if name not in idx:
            continue
        j = idx[name]
        good = 0
        total = 0
        sgn = 1 if int(sign) >= 0 else -1
        for cid in uniq:
            m = ids == cid
            if np.sum(m) < 2:
                continue
            order = np.argsort(tt[m])
            v = yp[m, j][order]
            dv = np.diff(v)
            total += int(dv.size)
            if sgn > 0:
                good += int(np.sum(dv >= -1e-12))
            else:
                good += int(np.sum(dv <= 1e-12))
        sc = float(good / max(total, 1))
        out[f"{name}_monotonic_score"] = sc
        scores.append(sc)
    out["overall_monotonic_score"] = float(np.mean(scores)) if scores else 0.0
    return out


def uncertainty_coverage_metrics(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_logvar: np.ndarray,
    *,
    z: float = 2.0,
) -> Dict[str, float]:
    """基于预测方差的区间覆盖率与尖锐度指标。"""
    yt = np.asarray(y_true, dtype=np.float64)
    ym = np.asarray(y_mean, dtype=np.float64)
    yv = np.exp(np.asarray(y_logvar, dtype=np.float64))
    ys = np.sqrt(np.maximum(yv, 1e-18))
    lo = ym - float(z) * ys
    hi = ym + float(z) * ys
    cover = (yt >= lo) & (yt <= hi)
    coverage = float(np.mean(cover.astype(np.float64)))
    sharpness = float(np.mean((hi - lo).astype(np.float64)))
    nll = float(np.mean(0.5 * (np.log(np.maximum(yv, 1e-18)) + (yt - ym) ** 2 / np.maximum(yv, 1e-18))))
    return {
        "coverage": coverage,
        "sharpness": sharpness,
        "nll": nll,
    }


def physics_consistency_index(
    *,
    bound_overall_violation_rate: float,
    monotonic_overall_score: float,
    coverage: float,
    cfg: PhysicsMetricConfig,
) -> float:
    """综合物理一致性指数 PCI（范围建议 [0,1]，越大越好）。

    构成：
    - 越界得分：`1 - violation_rate`
    - 单调得分：`monotonic_score`
    - 校准得分：`1 - |coverage - expected| / expected`
    """
    s_bound = 1.0 - float(np.clip(bound_overall_violation_rate, 0.0, 1.0))
    s_mono = float(np.clip(monotonic_overall_score, 0.0, 1.0))
    exp_cov = max(float(cfg.expected_coverage_2sigma), 1e-8)
    s_cal = 1.0 - min(abs(float(coverage) - exp_cov) / exp_cov, 1.0)
    wsum = float(cfg.w_bound + cfg.w_monotonic + cfg.w_calibration)
    if wsum <= 0.0:
        return float((s_bound + s_mono + s_cal) / 3.0)
    pci = (float(cfg.w_bound) * s_bound + float(cfg.w_monotonic) * s_mono + float(cfg.w_calibration) * s_cal) / wsum
    return float(np.clip(pci, 0.0, 1.0))


def evaluate_physics_metrics(
    *,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_logvar: np.ndarray,
    target_names: Sequence[str],
    case_ids: Sequence[str],
    t_inputs: Sequence[float],
    cfg: PhysicsMetricConfig | None = None,
) -> Dict[str, float]:
    """一站式计算宏观 surrogate 的物理一致性指标。"""
    pcfg = cfg if cfg is not None else PhysicsMetricConfig()
    b = bound_violation_rate(y_mean, target_names, pcfg.bounds)
    m = monotonic_trend_score(y_mean, target_names, case_ids, t_inputs, pcfg.monotonic_targets)
    u = uncertainty_coverage_metrics(y_true, y_mean, y_logvar, z=2.0)
    pci = physics_consistency_index(
        bound_overall_violation_rate=float(b.get("overall_violation_rate", 0.0)),
        monotonic_overall_score=float(m.get("overall_monotonic_score", 0.0)),
        coverage=float(u.get("coverage", 0.0)),
        cfg=pcfg,
    )
    out: Dict[str, float] = {}
    out.update(b)
    out.update(m)
    out.update({f"unc_{k}": float(v) for k, v in u.items()})
    out["physics_consistency_index"] = float(pci)
    return out

