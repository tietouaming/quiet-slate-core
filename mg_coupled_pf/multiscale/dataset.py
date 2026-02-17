"""跨尺度训练数据集构建。

从相场快照序列中构造监督样本：
- 输入：时刻 t 的微结构场张量 + 描述符；
- 输出：时刻 t+ΔT 的宏观指标向量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .features import DEFAULT_FIELD_CHANNELS, descriptor_vector, extract_micro_descriptors, extract_micro_tensor


@dataclass
class MacroTargetSpec:
    """宏观目标定义。"""

    names: List[str] = field(
        default_factory=lambda: [
            "corrosion_loss",
            "twin_fraction",
            "epspeq_mean",
            "sigma_h_max_solid",
            "penetration_x_ratio",
        ]
    )


@dataclass
class MultiScaleDatasetConfig:
    """数据集构建配置。"""

    field_channels: List[str] = field(default_factory=lambda: list(DEFAULT_FIELD_CHANNELS))
    descriptor_names: List[str] = field(
        default_factory=lambda: [
            "solid_fraction",
            "twin_fraction",
            "c_solid_mean",
            "c_liquid_mean",
            "epspeq_mean",
            "sigma_h_max_solid",
            "interface_density",
            "penetration_x_ratio",
        ]
    )
    target_hw: Tuple[int, int] = (96, 96)
    horizon_steps: int = 1
    frame_stride: int = 1
    max_samples_per_case: int = 0
    macro_targets: MacroTargetSpec = field(default_factory=MacroTargetSpec)


@dataclass
class MultiFidelityDatasetConfig(MultiScaleDatasetConfig):
    """多保真数据集构建配置。

    说明：
    - `target_hw` 对应高保真输入分辨率；
    - `low_target_hw` 用于构造低保真目标（从目标快照降采样后提取宏观指标）；
    - `high_fidelity_keep_ratio` 控制可用高保真标签比例（模拟“高精度宏观标签稀缺”场景）。
    """

    low_target_hw: Tuple[int, int] = (48, 48)
    high_fidelity_keep_ratio: float = 1.0
    seed: int = 42


def _parse_step_from_snapshot_name(path: Path) -> int:
    stem = path.stem
    # snapshot_000123
    try:
        return int(stem.split("_")[-1])
    except Exception:
        return -1


def _load_history_by_step(case_dir: Path) -> Dict[int, Dict[str, float]]:
    p = case_dir / "history.csv"
    if not p.exists():
        return {}
    out: Dict[int, Dict[str, float]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                st = int(float(row.get("step", "nan")))
            except Exception:
                continue
            rec: Dict[str, float] = {}
            for k, v in row.items():
                try:
                    rec[str(k)] = float(v)
                except Exception:
                    continue
            out[st] = rec
    return out


def _macro_target_value(name: str, desc: Dict[str, float], hist: Dict[str, float]) -> float:
    n = str(name).strip().lower()
    if n == "corrosion_loss":
        return 1.0 - float(desc.get("solid_fraction", 0.0))
    if n in {"twin_fraction", "epspeq_mean", "sigma_h_max_solid", "penetration_x_ratio"}:
        return float(desc.get(n, 0.0))
    # history csv 兼容键
    if n == "max_sigma_h":
        return float(hist.get("max_sigma_h", desc.get("sigma_h_max_solid", 0.0)))
    if n == "avg_epspeq":
        return float(hist.get("avg_epspeq", desc.get("epspeq_mean", 0.0)))
    return float(hist.get(name, desc.get(name, 0.0)))


def _load_snapshot_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _resize_snapshot_for_low_fidelity(snapshot: Dict[str, np.ndarray], target_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """将快照字段降采样到低保真网格，用于构造低保真目标。"""
    out: Dict[str, np.ndarray] = {}
    th, tw = int(target_hw[0]), int(target_hw[1])
    for k, v in snapshot.items():
        arr = np.asarray(v)
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr2 = arr[0, 0]
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr2 = arr[0]
        elif arr.ndim == 2:
            arr2 = arr
        else:
            continue
        t = torch.from_numpy(arr2.astype(np.float32, copy=False))[None, None]
        r = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
        out[str(k)] = r
    return out


def _sample_high_label_mask(n: int, ratio: float, seed: int) -> np.ndarray:
    """生成高保真标签可用掩码。"""
    r = float(np.clip(ratio, 0.0, 1.0))
    if r >= 1.0:
        return np.ones((n,), dtype=np.float32)
    if r <= 0.0:
        return np.zeros((n,), dtype=np.float32)
    rs = np.random.RandomState(int(seed))
    m = (rs.rand(n) < r).astype(np.float32)
    # 防止全部为 0，至少保留一个高保真样本用于监督锚定。
    if np.sum(m) < 1:
        m[int(rs.randint(0, n))] = 1.0
    return m


def build_multiscale_dataset_from_cases(
    case_dirs: Sequence[Path],
    cfg: MultiScaleDatasetConfig,
) -> Dict[str, np.ndarray]:
    """从多个算例目录构建跨尺度样本集。"""
    xs_field: List[np.ndarray] = []
    xs_desc: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    case_ids: List[str] = []
    t_inputs: List[float] = []
    t_targets: List[float] = []

    for case_dir in case_dirs:
        snap_dir = case_dir / "snapshots"
        if not snap_dir.exists():
            continue
        snaps = sorted(snap_dir.glob("snapshot_*.npz"))
        if len(snaps) <= cfg.horizon_steps:
            continue
        step_to_hist = _load_history_by_step(case_dir)
        n_keep = 0
        for i in range(0, len(snaps) - cfg.horizon_steps, max(cfg.frame_stride, 1)):
            j = i + cfg.horizon_steps
            p_in = snaps[i]
            p_out = snaps[j]
            step_in = _parse_step_from_snapshot_name(p_in)
            step_out = _parse_step_from_snapshot_name(p_out)
            if step_in < 0 or step_out < 0:
                continue
            s_in = _load_snapshot_npz(p_in)
            s_out = _load_snapshot_npz(p_out)
            d_in = extract_micro_descriptors(s_in)
            d_out = extract_micro_descriptors(s_out)
            x_field = extract_micro_tensor(
                s_in,
                channels=cfg.field_channels,
                target_hw=cfg.target_hw,
            ).numpy()
            x_desc = descriptor_vector(d_in, cfg.descriptor_names)
            h_out = step_to_hist.get(step_out, {})
            y = np.asarray(
                [_macro_target_value(k, d_out, h_out) for k in cfg.macro_targets.names],
                dtype=np.float32,
            )

            xs_field.append(x_field.astype(np.float32, copy=False))
            xs_desc.append(x_desc.astype(np.float32, copy=False))
            ys.append(y)
            case_ids.append(case_dir.name)
            t_inputs.append(float(step_to_hist.get(step_in, {}).get("time_s", float(step_in))))
            t_targets.append(float(step_to_hist.get(step_out, {}).get("time_s", float(step_out))))
            n_keep += 1
            if cfg.max_samples_per_case > 0 and n_keep >= int(cfg.max_samples_per_case):
                break

    if not xs_field:
        raise RuntimeError("No samples were built. Check case directory paths and horizon settings.")

    x_field_np = np.stack(xs_field, axis=0)  # [N,C,H,W]
    x_desc_np = np.stack(xs_desc, axis=0)  # [N,D]
    y_np = np.stack(ys, axis=0)  # [N,T]
    return {
        "x_field": x_field_np,
        "x_desc": x_desc_np,
        "y": y_np,
        "field_channels": np.asarray(list(cfg.field_channels), dtype=object),
        "descriptor_names": np.asarray(list(cfg.descriptor_names), dtype=object),
        "target_names": np.asarray(list(cfg.macro_targets.names), dtype=object),
        "case_ids": np.asarray(case_ids, dtype=object),
        "t_input": np.asarray(t_inputs, dtype=np.float32),
        "t_target": np.asarray(t_targets, dtype=np.float32),
    }


def build_multifidelity_dataset_from_cases(
    case_dirs: Sequence[Path],
    cfg: MultiFidelityDatasetConfig,
) -> Dict[str, np.ndarray]:
    """构建多保真数据集。

    输出字段：
    - `x_field`, `x_desc`：与单保真一致；
    - `y`：高保真目标；
    - `y_low`：低保真目标（降采样快照提取）；
    - `high_mask`：高保真标签可用掩码（1=有高保真标签，0=仅低保真）。
    """
    base = build_multiscale_dataset_from_cases(case_dirs, cfg)
    ys_low: List[np.ndarray] = []
    case_ids = [str(v) for v in base["case_ids"].tolist()]
    t_input = base["t_input"].astype(np.float32)
    # 为避免重复读取所有输入快照，这里重新遍历样本构建低保真目标。
    # 由于该阶段离线执行，开销可接受，且逻辑清晰可复现。
    for case_dir in case_dirs:
        snap_dir = case_dir / "snapshots"
        if not snap_dir.exists():
            continue
        snaps = sorted(snap_dir.glob("snapshot_*.npz"))
        if len(snaps) <= cfg.horizon_steps:
            continue
        step_to_hist = _load_history_by_step(case_dir)
        n_keep = 0
        for i in range(0, len(snaps) - cfg.horizon_steps, max(cfg.frame_stride, 1)):
            j = i + cfg.horizon_steps
            p_out = snaps[j]
            step_out = _parse_step_from_snapshot_name(p_out)
            if step_out < 0:
                continue
            s_out = _load_snapshot_npz(p_out)
            s_out_low = _resize_snapshot_for_low_fidelity(s_out, cfg.low_target_hw)
            d_out_low = extract_micro_descriptors(s_out_low)
            h_out = step_to_hist.get(step_out, {})
            y_low = np.asarray(
                [_macro_target_value(k, d_out_low, h_out) for k in cfg.macro_targets.names],
                dtype=np.float32,
            )
            ys_low.append(y_low)
            n_keep += 1
            if cfg.max_samples_per_case > 0 and n_keep >= int(cfg.max_samples_per_case):
                break
    if len(ys_low) != int(base["y"].shape[0]):
        raise RuntimeError(
            f"low fidelity sample count mismatch: low={len(ys_low)}, high={int(base['y'].shape[0])}."
        )
    y_low_np = np.stack(ys_low, axis=0).astype(np.float32)
    high_mask = _sample_high_label_mask(int(base["y"].shape[0]), cfg.high_fidelity_keep_ratio, cfg.seed)
    return {
        **base,
        "y_low": y_low_np,
        "high_mask": high_mask.astype(np.float32),
        "low_target_hw": np.asarray(list(cfg.low_target_hw), dtype=np.int32),
        "high_fidelity_keep_ratio": np.asarray([float(cfg.high_fidelity_keep_ratio)], dtype=np.float32),
        "seed": np.asarray([int(cfg.seed)], dtype=np.int32),
    }


def save_multiscale_dataset_npz(path: Path | str, payload: Dict[str, np.ndarray]) -> Path:
    """保存跨尺度数据集为 npz。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **payload)
    return p


def load_multiscale_dataset_npz(path: Path | str) -> Dict[str, np.ndarray]:
    """加载跨尺度数据集 npz。"""
    p = Path(path)
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def discover_case_dirs(
    root: Path | str,
    pattern: str = "*",
    *,
    require_history: bool = True,
) -> List[Path]:
    """自动发现包含 snapshots 的算例目录。"""
    r = Path(root)
    out: List[Path] = []
    for c in sorted(r.glob(pattern)):
        if not c.is_dir():
            continue
        if not (c / "snapshots").exists():
            continue
        if require_history and not (c / "history.csv").exists():
            continue
        out.append(c)
    return out
