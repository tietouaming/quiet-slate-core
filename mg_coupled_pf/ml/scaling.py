"""ML 通道尺度化工具。

本模块集中管理 surrogate / mechanics warmstart 的通道量纲归一化规则，
避免训练与推理阶段使用不同尺度导致的数值漂移。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple


def _safe_scale(v: float, default: float = 1.0) -> float:
    """将非法或过小尺度回退到默认值。"""
    try:
        x = float(v)
    except Exception:
        return float(default)
    if abs(x) < 1e-12:
        return float(default)
    return x


def sanitize_field_scales(order: Iterable[str], field_scales: Dict[str, float] | None = None) -> Dict[str, float]:
    """按通道顺序返回合法尺度字典（缺省=1）。"""
    fs = field_scales or {}
    out: Dict[str, float] = {}
    for k in order:
        out[k] = _safe_scale(fs.get(k, 1.0), default=1.0)
    return out


def build_surrogate_field_scales(cfg: Any) -> Dict[str, float]:
    """根据配置构建 surrogate 通道尺度。

    说明：
    - φ/c/η 属于 0~1 变量，默认尺度 1；
    - 位移以特征边界位移归一；
    - 塑性相关量以特征应变归一。
    """
    enable = bool(getattr(getattr(cfg, "ml", object()), "enable_field_scaling", True))
    if not enable:
        return {
            "phi": 1.0,
            "c": 1.0,
            "eta": 1.0,
            "ux": 1.0,
            "uy": 1.0,
            "epspeq": 1.0,
            "epsp_xx": 1.0,
            "epsp_yy": 1.0,
            "epsp_xy": 1.0,
        }

    lx_um = max(float(getattr(cfg.domain, "lx_um", 1.0)), 1e-6)
    ext_strain_x = abs(float(getattr(cfg.mechanics, "external_strain_x", 0.0)))
    right_u = float(getattr(cfg.mechanics, "dirichlet_right_displacement_um", -1.0))
    if right_u <= 0.0:
        right_u = ext_strain_x * lx_um
    if right_u <= 0.0:
        # 无明显外载时给一个保守尺度，避免除零与过度放大。
        right_u = max(1e-3, lx_um * 1e-3)
    disp_ref = max(abs(right_u), 1e-3)

    eps_ref = max(
        abs(float(getattr(cfg.corrosion, "yield_strain_for_mech", 0.002))),
        1e-5,
    )
    return {
        "phi": 1.0,
        "c": 1.0,
        "eta": 1.0,
        "ux": disp_ref,
        "uy": disp_ref,
        "epspeq": eps_ref,
        "epsp_xx": eps_ref,
        "epsp_yy": eps_ref,
        "epsp_xy": eps_ref,
    }


def build_mechanics_field_scales(cfg: Any) -> Tuple[Dict[str, float], Dict[str, float]]:
    """根据配置构建 warmstart 输入/输出尺度。"""
    s = build_surrogate_field_scales(cfg)
    in_scales = {
        "phi": s["phi"],
        "c": s["c"],
        "eta": s["eta"],
        "epspeq": s["epspeq"],
        "epsp_xx": s["epsp_xx"],
        "epsp_yy": s["epsp_yy"],
        "epsp_xy": s["epsp_xy"],
    }
    out_scales = {"ux": s["ux"], "uy": s["uy"]}
    return in_scales, out_scales

