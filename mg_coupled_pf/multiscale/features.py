"""跨尺度建模的微结构特征提取。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..operators import grad_xy, smooth_heaviside


DEFAULT_FIELD_CHANNELS: Tuple[str, ...] = ("phi", "c", "eta", "epspeq")


def _to_torch_4d(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """将输入标准化为 `[1,1,H,W]` 张量。"""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x
    if t.ndim == 2:
        return t[None, None]
    if t.ndim == 3:
        return t[None]
    if t.ndim == 4:
        return t
    raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")


def _safe_field(snapshot: Dict[str, np.ndarray], name: str, ref_shape: Tuple[int, int]) -> torch.Tensor:
    """从快照读取字段，不存在则返回零场。"""
    if name in snapshot:
        return _to_torch_4d(snapshot[name]).to(dtype=torch.float32)
    h, w = ref_shape
    return torch.zeros((1, 1, h, w), dtype=torch.float32)


def extract_micro_tensor(
    snapshot: Dict[str, np.ndarray],
    channels: Sequence[str] = DEFAULT_FIELD_CHANNELS,
    *,
    target_hw: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """将快照提取为微结构场张量 `[C,H,W]`。

    约定：
    - 输入快照字段可为 `H,W` 或 `1,1,H,W`；
    - 若 `target_hw` 提供，则统一双线性重采样到固定分辨率；
    - 缺失字段自动补零，保证通道长度稳定。
    """
    if not snapshot:
        raise ValueError("snapshot is empty.")
    # 以 phi 或任意首字段确定空间尺寸。
    if "phi" in snapshot:
        ref = _to_torch_4d(snapshot["phi"])
    else:
        k0 = next(iter(snapshot.keys()))
        ref = _to_torch_4d(snapshot[k0])
    h, w = int(ref.shape[-2]), int(ref.shape[-1])
    arr: List[torch.Tensor] = []
    for name in channels:
        t = _safe_field(snapshot, str(name), (h, w))
        if target_hw is not None and (t.shape[-2] != target_hw[0] or t.shape[-1] != target_hw[1]):
            t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
        arr.append(t)
    out = torch.cat(arr, dim=1)[0]  # [C,H,W]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _front_penetration_x_index(phi: torch.Tensor, threshold: float = 0.5) -> float:
    """估计腐蚀前沿穿透深度（基于 x 方向首个液化索引）。"""
    # phi: [1,1,H,W]
    p = torch.nan_to_num(phi, nan=0.0, posinf=1.0, neginf=0.0)[0, 0]
    h, w = p.shape
    liquid_mask = p < float(threshold)
    if not bool(liquid_mask.any().item()):
        return 0.0
    xs = torch.arange(w, dtype=torch.float32, device=p.device)[None, :].expand(h, w)
    # 每行若存在液相，取最小 x；否则记为 w（无穿透）。
    row_has = liquid_mask.any(dim=1)
    row_min = torch.where(
        row_has,
        torch.min(torch.where(liquid_mask, xs, torch.full_like(xs, float(w))), dim=1).values,
        torch.full((h,), float(w), device=p.device),
    )
    # 穿透定义为从右向左侵蚀深度占比（可按需要替换）。
    pen = torch.clamp((float(w) - row_min) / max(float(w), 1.0), min=0.0, max=1.0)
    return float(torch.mean(pen).item())


def extract_micro_descriptors(snapshot: Dict[str, np.ndarray]) -> Dict[str, float]:
    """提取可解释微结构描述符（用于跨尺度条件变量与反向设计）。"""
    phi = _to_torch_4d(snapshot.get("phi"))
    c = _to_torch_4d(snapshot.get("c", np.zeros_like(snapshot["phi"])))
    eta = _to_torch_4d(snapshot.get("eta", np.zeros_like(snapshot["phi"])))
    epspeq = _to_torch_4d(snapshot.get("epspeq", np.zeros_like(snapshot["phi"])))
    sigma_h = _to_torch_4d(snapshot.get("sigma_h", np.zeros_like(snapshot["phi"])))
    phi = torch.nan_to_num(phi, nan=0.0, posinf=1.0, neginf=0.0)
    c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    eta = torch.nan_to_num(eta, nan=0.0, posinf=1.0, neginf=0.0)
    epspeq = torch.nan_to_num(epspeq, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_h = torch.nan_to_num(sigma_h, nan=0.0, posinf=0.0, neginf=0.0)

    hphi = smooth_heaviside(torch.clamp(phi, 0.0, 1.0))
    heta = smooth_heaviside(torch.clamp(eta, 0.0, 1.0))
    solid = hphi
    liquid = 1.0 - hphi
    solid_frac = float(torch.mean(solid).item())
    liquid_frac = float(torch.mean(liquid).item())
    twin_frac = float(torch.mean(heta * solid).item())
    c_solid = float(torch.sum(c * solid).item() / max(float(torch.sum(solid).item()), 1e-12))
    c_liquid = float(torch.sum(c * liquid).item() / max(float(torch.sum(liquid).item()), 1e-12))
    epspeq_mean = float(torch.mean(epspeq * solid).item())
    sigma_h_max_solid = float(torch.max(torch.abs(sigma_h * solid)).item())
    pen_x = _front_penetration_x_index(phi, threshold=0.5)

    # 界面密度：|grad(phi)| 的平均值（索引尺度，训练时用于相对比较）。
    gx, gy = grad_xy(phi, dx=1.0, dy=1.0, bc="neumann")
    interface_density = float(torch.mean(torch.sqrt(torch.clamp(gx * gx + gy * gy, min=0.0))).item())

    return {
        "solid_fraction": solid_frac,
        "liquid_fraction": liquid_frac,
        "twin_fraction": twin_frac,
        "c_solid_mean": c_solid,
        "c_liquid_mean": c_liquid,
        "epspeq_mean": epspeq_mean,
        "sigma_h_max_solid": sigma_h_max_solid,
        "interface_density": interface_density,
        "penetration_x_ratio": pen_x,
    }


def descriptor_vector(desc: Dict[str, float], order: Iterable[str]) -> np.ndarray:
    """按给定顺序拼接描述符向量。"""
    return np.asarray([float(desc.get(k, 0.0)) for k in order], dtype=np.float32)

