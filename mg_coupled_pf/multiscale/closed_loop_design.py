"""跨尺度闭环反向设计（surrogate + physics verification）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..config import SimulationConfig
from ..geometry import initial_fields, make_grid
from ..simulator import CoupledSimulator
from .data_generation import SweepVariable
from .features import descriptor_vector, extract_micro_descriptors, extract_micro_tensor
from .advanced_models import unscale_advanced_targets


@dataclass
class ClosedLoopDesignConfig:
    """闭环设计配置。"""

    seed: int = 42
    candidate_pool: int = 80
    surrogate_topk: int = 10
    physics_verify_topk: int = 5
    local_refine_trials: int = 8
    local_refine_sigma_ratio: float = 0.08
    # 物理验证算例预算
    sim_steps: int = 360
    sim_save_every: int = 36
    # 目标函数（越大越好）
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "corrosion_loss": -1.0,
            "penetration_x_ratio": -1.0,
            "sigma_h_max_solid": -0.25,
            "epspeq_mean": -0.1,
            "twin_fraction": 0.05,
        }
    )
    output_root: str = "artifacts/closed_loop_design"
    case_prefix: str = "cld_case"


def _set_cfg_value(cfg: SimulationConfig, path: str, value: float) -> None:
    keys = str(path).split(".")
    obj = cfg
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], float(value))


def _sample_candidates(vars_spec: Sequence[SweepVariable], n: int, seed: int) -> List[Dict[str, float]]:
    rs = np.random.RandomState(int(seed))
    out: List[Dict[str, float]] = []
    for _ in range(int(n)):
        row: Dict[str, float] = {}
        for v in vars_spec:
            lo = float(v.lower)
            hi = float(v.upper)
            if v.log_scale:
                lo2 = max(lo, 1e-12)
                hi2 = max(hi, lo2 * 1.0001)
                x = rs.uniform(np.log(lo2), np.log(hi2))
                row[v.path] = float(np.exp(x))
            else:
                row[v.path] = float(rs.uniform(lo, hi))
        out.append(row)
    return out


def _objective_score(targets: Dict[str, float], weights: Dict[str, float]) -> float:
    s = 0.0
    for k, w in weights.items():
        if k in targets:
            s += float(w) * float(targets[k])
    return float(s)


def _predict_surrogate_targets(
    model: torch.nn.Module,
    model_meta: Dict[str, object],
    base_cfg: SimulationConfig,
    rows: Sequence[Dict[str, float]],
    *,
    target_hw: Tuple[int, int],
    device: torch.device,
) -> List[Dict[str, float]]:
    field_channels = list(model_meta.get("field_channels", ["phi", "c", "eta", "epspeq"]))
    desc_names = list(model_meta.get("descriptor_names", []))
    target_names = list(model_meta.get("target_names", []))
    scalers = model_meta.get("scalers", {})
    preds: List[Dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            cfg_i = deepcopy(base_cfg)
            for k, v in row.items():
                _set_cfg_value(cfg_i, k, float(v))
            g = make_grid(cfg_i, device=device, dtype=torch.float32)
            st = initial_fields(cfg_i, g)
            snap = {}
            for k, v in st.items():
                arr = v.detach().cpu().numpy()
                if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
                    arr = arr[0, 0]
                snap[k] = arr
            xf = extract_micro_tensor(snap, channels=field_channels, target_hw=target_hw)[None].to(device=device)
            xd_np = descriptor_vector(extract_micro_descriptors(snap), desc_names if desc_names else [])
            xd = torch.from_numpy(xd_np[None].astype(np.float32)).to(device=device)
            if isinstance(scalers, dict) and "x_desc" in scalers and "mean" in scalers["x_desc"] and len(desc_names) > 0:
                mu = torch.tensor(np.asarray(scalers["x_desc"]["mean"], dtype=np.float32), device=device)
                sd = torch.tensor(np.asarray(scalers["x_desc"]["std"], dtype=np.float32), device=device)
                xd = (xd - mu) / sd
            out = model(xf, xd)
            if "high_mean" in out:
                ym_s = out["high_mean"].detach().cpu().numpy()
                yv_s = out["high_logvar"].detach().cpu().numpy()
            else:
                ym_s = out["mean"].detach().cpu().numpy()
                yv_s = out["logvar"].detach().cpu().numpy()
            if isinstance(scalers, dict) and "y_high" in scalers and "mean" in scalers["y_high"]:
                ym = unscale_advanced_targets(ym_s, scalers, key="y_high")
                std = np.asarray(scalers["y_high"]["std"], dtype=np.float32).reshape(1, -1)
                yv = np.log(np.maximum(np.exp(yv_s) * (std ** 2), 1e-18))
            else:
                ym = ym_s
                yv = yv_s
            vals = {str(target_names[i] if i < len(target_names) else f"target_{i}"): float(ym[0, i]) for i in range(ym.shape[1])}
            vals["uncertainty_mean"] = float(np.mean(np.sqrt(np.exp(yv[0]))))
            preds.append(vals)
    return preds


def _run_physics_verify(
    base_cfg: SimulationConfig,
    row: Dict[str, float],
    cfg: ClosedLoopDesignConfig,
    *,
    case_name: str,
) -> Dict[str, float]:
    c = deepcopy(base_cfg)
    for k, v in row.items():
        _set_cfg_value(c, k, float(v))
    c.ml.enabled = False
    c.runtime.render_intermediate_fields = False
    c.runtime.render_final_clouds = False
    c.runtime.render_grid_figure = False
    c.runtime.clean_output = True
    c.numerics.n_steps = int(cfg.sim_steps)
    c.numerics.save_every = int(cfg.sim_save_every)
    c.runtime.case_name = case_name
    c.runtime.output_dir = str(Path(cfg.output_root) / "physics_verify")
    sim = CoupledSimulator(c)
    sim.run(progress=False, progress_every=100, progress_prefix=case_name)
    snap = {}
    for k, v in sim.state.items():
        arr = v.detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        snap[k] = arr
    # 用末态微结构描述符作为宏观评价（与训练目标同口径）。
    d = extract_micro_descriptors(snap)
    d["corrosion_loss"] = 1.0 - float(d.get("solid_fraction", 0.0))
    return {k: float(v) for k, v in d.items()}


def run_closed_loop_design(
    *,
    base_cfg: SimulationConfig,
    vars_spec: Sequence[SweepVariable],
    model: torch.nn.Module,
    model_meta: Dict[str, object],
    cfg: ClosedLoopDesignConfig,
    device: torch.device,
    target_hw: Tuple[int, int] = (128, 128),
) -> Dict[str, object]:
    """执行闭环反向设计。"""
    rs = np.random.RandomState(int(cfg.seed))
    rows = _sample_candidates(vars_spec, int(cfg.candidate_pool), seed=int(cfg.seed))
    preds = _predict_surrogate_targets(model, model_meta, base_cfg, rows, target_hw=target_hw, device=device)
    scores = np.asarray([_objective_score(p, cfg.objective_weights) for p in preds], dtype=np.float64)
    idx = np.argsort(scores)[-max(1, int(cfg.surrogate_topk)) :]
    top_rows = [rows[int(i)] for i in idx.tolist()]
    top_pred = [preds[int(i)] for i in idx.tolist()]

    # physics verify
    verify_count = min(int(cfg.physics_verify_topk), len(top_rows))
    verified: List[Dict[str, object]] = []
    for i in range(verify_count):
        row = top_rows[-(i + 1)]
        pred = top_pred[-(i + 1)]
        true_tar = _run_physics_verify(base_cfg, row, cfg, case_name=f"{cfg.case_prefix}_verify_{i:03d}")
        verified.append(
            {
                "params": row,
                "pred_targets": pred,
                "pred_score": _objective_score(pred, cfg.objective_weights),
                "true_targets": true_tar,
                "true_score": _objective_score(true_tar, cfg.objective_weights),
            }
        )

    # local refinement around best verified
    if verified:
        best = max(verified, key=lambda x: float(x["true_score"]))
        center = best["params"]
    else:
        center = top_rows[-1]
    refined: List[Dict[str, object]] = []
    for i in range(int(cfg.local_refine_trials)):
        row = {}
        for v in vars_spec:
            c0 = float(center[v.path])
            span = float(v.upper) - float(v.lower)
            sig = max(float(cfg.local_refine_sigma_ratio) * span, 1e-9)
            x = rs.normal(c0, sig)
            x = float(np.clip(x, float(v.lower), float(v.upper)))
            row[v.path] = x
        true_tar = _run_physics_verify(base_cfg, row, cfg, case_name=f"{cfg.case_prefix}_local_{i:03d}")
        refined.append(
            {
                "params": row,
                "true_targets": true_tar,
                "true_score": _objective_score(true_tar, cfg.objective_weights),
            }
        )

    best_verified = max(verified, key=lambda x: float(x["true_score"])) if verified else None
    best_refined = max(refined, key=lambda x: float(x["true_score"])) if refined else None
    best_final = best_verified
    if best_refined is not None and (best_verified is None or float(best_refined["true_score"]) > float(best_verified["true_score"])):
        best_final = best_refined

    return {
        "config": {
            "candidate_pool": int(cfg.candidate_pool),
            "surrogate_topk": int(cfg.surrogate_topk),
            "physics_verify_topk": int(cfg.physics_verify_topk),
            "local_refine_trials": int(cfg.local_refine_trials),
            "objective_weights": cfg.objective_weights,
        },
        "verified": verified,
        "refined": refined,
        "best_verified": best_verified,
        "best_refined": best_refined,
        "best_final": best_final,
    }

