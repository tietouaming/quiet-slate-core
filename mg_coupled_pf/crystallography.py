"""晶体学工具函数（HCP 方向/晶面转换与取向旋转）。

本模块用于统一：
1. Bunge 欧拉角与轴角旋转；
2. HCP 四指数 (Miller-Bravais) -> 笛卡尔向量转换；
3. 滑移/孪晶系统的 s,n 正交化与二维投影。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch


def normalize_vec3(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """三维向量归一化。"""
    n = torch.sqrt(torch.clamp(torch.sum(v * v), min=eps))
    return v / n


def orthonormalize_pair(s: torch.Tensor, n: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 (s,n) 做正交归一化，确保 s·n=0。"""
    s_u = normalize_vec3(s, eps=eps)
    n_p = n - torch.sum(n * s_u) * s_u
    nn = torch.sqrt(torch.clamp(torch.sum(n_p * n_p), min=0.0))
    if float(nn.item()) <= eps:
        # n 与 s 近平行时使用稳定回退基向量重建法向。
        ref = torch.tensor([0.0, 0.0, 1.0], device=s.device, dtype=s.dtype)
        if abs(float(torch.sum(ref * s_u).item())) > 0.9:
            ref = torch.tensor([1.0, 0.0, 0.0], device=s.device, dtype=s.dtype)
        n_p = ref - torch.sum(ref * s_u) * s_u
    n_u = normalize_vec3(n_p, eps=eps)
    return s_u, n_u


def bunge_euler_matrix(euler_deg: List[float], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Bunge ZXZ 欧拉角旋转矩阵（晶体系 -> 试样系）。"""
    if len(euler_deg) != 3:
        raise ValueError("Euler angles must have length 3.")
    a, b, c = [torch.deg2rad(torch.tensor(float(x), device=device, dtype=dtype)) for x in euler_deg]
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)
    z = torch.tensor(0.0, device=device, dtype=dtype)
    o = torch.tensor(1.0, device=device, dtype=dtype)
    rz1 = torch.stack([torch.stack([ca, -sa, z]), torch.stack([sa, ca, z]), torch.stack([z, z, o])])
    rx = torch.stack([torch.stack([o, z, z]), torch.stack([z, cb, -sb]), torch.stack([z, sb, cb])])
    rz2 = torch.stack([torch.stack([cc, -sc, z]), torch.stack([sc, cc, z]), torch.stack([z, z, o])])
    return rz1 @ rx @ rz2


def axis_angle_matrix(axis: List[float], angle_deg: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """轴角旋转矩阵（Rodrigues）。"""
    if len(axis) != 3:
        raise ValueError("Axis must have length 3.")
    ax = normalize_vec3(torch.tensor([float(axis[0]), float(axis[1]), float(axis[2])], device=device, dtype=dtype))
    th = torch.deg2rad(torch.tensor(float(angle_deg), device=device, dtype=dtype))
    c = torch.cos(th)
    s = torch.sin(th)
    x, y, z = ax[0], ax[1], ax[2]
    k = torch.tensor(
        [
            [0.0, -float(z.item()), float(y.item())],
            [float(z.item()), 0.0, -float(x.item())],
            [-float(y.item()), float(x.item()), 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    i = torch.eye(3, device=device, dtype=dtype)
    outer = ax.view(3, 1) @ ax.view(1, 3)
    return c * i + (1.0 - c) * outer + s * k


def _hcp_direct_basis(c_over_a: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """HCP 直接晶格基矢矩阵 A=[a1,a2,c]。"""
    rt3 = math.sqrt(3.0)
    a1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    a2 = torch.tensor([-0.5, 0.5 * rt3, 0.0], device=device, dtype=dtype)
    c = torch.tensor([0.0, 0.0, float(c_over_a)], device=device, dtype=dtype)
    return torch.stack([a1, a2, c], dim=1)


def hcp_direction_mb_to_cart(
    direction_mb: List[float],
    *,
    c_over_a: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """四指数方向 [u,v,t,w] -> 笛卡尔方向向量。"""
    if len(direction_mb) != 4:
        raise ValueError("direction_mb must have length 4.")
    u, v, t, w = [float(x) for x in direction_mb]
    # 若输入不满足四指数约束，则投影到 t=-(u+v)。
    if abs(u + v + t) > 1e-8:
        t = -(u + v)
    # 在独立基矢 (a1,a2,c) 下的坐标：r = (2u+v)a1 + (u+2v)a2 + w c
    u1 = 2.0 * u + v
    u2 = u + 2.0 * v
    a = _hcp_direct_basis(c_over_a, device=device, dtype=dtype)
    coeff = torch.tensor([u1, u2, w], device=device, dtype=dtype)
    return a @ coeff


def hcp_plane_mb_to_cart(
    plane_mb: List[float],
    *,
    c_over_a: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """四指数晶面 (h,k,i,l) -> 笛卡尔法向向量。"""
    if len(plane_mb) != 4:
        raise ValueError("plane_mb must have length 4.")
    h, k, i, l = [float(x) for x in plane_mb]
    # 若输入不满足四指数约束，则投影到 i=-(h+k)。
    if abs(h + k + i) > 1e-8:
        i = -(h + k)
    # 由独立基 (a1,a2,c) 的倒易基构造法向：g = h*b1 + k*b2 + l*b3
    a = _hcp_direct_basis(c_over_a, device=device, dtype=dtype)
    b = torch.linalg.inv(a).transpose(0, 1)
    coeff = torch.tensor([h, k, l], device=device, dtype=dtype)
    return b @ coeff


def resolve_system_pair_crystal(
    system: Dict[str, Any],
    *,
    c_over_a: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从系统定义解析晶体坐标下 (s,n)。"""
    if "s_crystal" in system and "n_crystal" in system:
        s = torch.tensor([float(v) for v in system["s_crystal"]], device=device, dtype=dtype)
        n = torch.tensor([float(v) for v in system["n_crystal"]], device=device, dtype=dtype)
    elif "direction_mb" in system and "plane_mb" in system:
        s = hcp_direction_mb_to_cart(
            [float(v) for v in system["direction_mb"]],
            c_over_a=float(c_over_a),
            device=device,
            dtype=dtype,
        )
        n = hcp_plane_mb_to_cart(
            [float(v) for v in system["plane_mb"]],
            c_over_a=float(c_over_a),
            device=device,
            dtype=dtype,
        )
    elif "direction_angle_deg" in system and "normal_angle_deg" in system:
        # 兼容旧二维角度输入：视作晶体坐标 x-y 平面方向。
        d = torch.deg2rad(torch.tensor(float(system["direction_angle_deg"]), device=device, dtype=dtype))
        a = torch.deg2rad(torch.tensor(float(system["normal_angle_deg"]), device=device, dtype=dtype))
        s = torch.tensor([torch.cos(d).item(), torch.sin(d).item(), 0.0], device=device, dtype=dtype)
        n = torch.tensor([torch.cos(a).item(), torch.sin(a).item(), 0.0], device=device, dtype=dtype)
    else:
        raise ValueError(
            f"System '{system.get('name', 'unknown')}' must provide "
            "s_crystal/n_crystal or direction_mb/plane_mb (or legacy angle fields)."
        )
    return orthonormalize_pair(s, n)


def project_pair_to_xy(
    s3: torch.Tensor,
    n3: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 3D (s,n) 投影到 2D x-y，并保持二维正交。"""
    s = s3[:2].clone()
    n = n3[:2].clone()
    ns = torch.sqrt(torch.clamp(torch.sum(s * s), min=0.0))
    if float(ns.item()) <= eps:
        s = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    else:
        s = s / ns
    n = n - torch.sum(n * s) * s
    nn = torch.sqrt(torch.clamp(torch.sum(n * n), min=0.0))
    if float(nn.item()) <= eps:
        n = torch.tensor([-float(s[1].item()), float(s[0].item())], device=device, dtype=dtype)
    else:
        n = n / nn
    return s[0], s[1], n[0], n[1]


def resolve_twin_pair_xy(
    twinning_cfg: Any,
    twin_systems: List[Dict[str, Any]],
    *,
    orientation_euler_deg: List[float],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, float, float, float]:
    """解析并旋转孪晶系，返回样品坐标下二维 `(sx, sy, nx, ny)`。"""
    c_over_a = float(getattr(twinning_cfg, "twin_hcp_c_over_a", 1.624))
    idx = int(getattr(twinning_cfg, "twin_variant_index", 0))
    if twin_systems:
        idx = max(0, min(idx, len(twin_systems) - 1))
        tw = twin_systems[idx]
    else:
        # 兼容旧配置：退回到角度定义并视作晶体坐标输入。
        tw = {
            "name": "legacy_angle_twin",
            "direction_angle_deg": float(getattr(twinning_cfg, "twin_shear_dir_angle_deg", 35.0)),
            "normal_angle_deg": float(getattr(twinning_cfg, "twin_plane_normal_angle_deg", 125.0)),
        }
    s_c, n_c = resolve_system_pair_crystal(
        tw,
        c_over_a=c_over_a,
        device=device,
        dtype=dtype,
    )
    r_parent = bunge_euler_matrix(list(orientation_euler_deg), device=device, dtype=dtype)
    s_s = r_parent @ s_c
    n_s = r_parent @ n_c
    s_s, n_s = orthonormalize_pair(s_s, n_s)
    sx, sy, nx, ny = project_pair_to_xy(s_s, n_s, device=device, dtype=dtype)
    return float(sx.item()), float(sy.item()), float(nx.item()), float(ny.item())

