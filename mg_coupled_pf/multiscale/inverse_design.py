"""基于跨尺度 surrogate 的反向设计优化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..config import SimulationConfig
from ..geometry import initial_fields, make_grid
from .features import descriptor_vector, extract_micro_descriptors, extract_micro_tensor
from .models import unscale_targets


@dataclass
class DesignVariableSpec:
    """反向设计变量定义。"""

    path: str
    lower: float
    upper: float
    init: float | None = None
    log_scale: bool = False


@dataclass
class InverseDesignConfig:
    """反向设计优化配置。"""

    seed: int = 7
    iterations: int = 20
    population: int = 48
    elite_frac: float = 0.2
    init_std_ratio: float = 0.25
    # 分数 = Σ weight[name] * predicted_target[name]
    # 权重为负：代表希望该指标越小越好（如 corrosion_loss）。
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "corrosion_loss": -1.0,
            "penetration_x_ratio": -1.0,
            "sigma_h_max_solid": -0.2,
            "epspeq_mean": -0.2,
            "twin_fraction": 0.1,
        }
    )
    target_hw: Tuple[int, int] = (96, 96)


@dataclass
class InverseDesignResult:
    """反向设计结果。"""

    best_score: float
    best_params: Dict[str, float]
    best_targets: Dict[str, float]
    history: List[Dict[str, float]]


def _set_cfg_value(cfg: SimulationConfig, path: str, value: float) -> None:
    keys = str(path).split(".")
    obj = cfg
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], float(value))


def _build_feature_inputs(
    cfg: SimulationConfig,
    *,
    field_channels: Sequence[str],
    descriptor_names: Sequence[str],
    target_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid = make_grid(cfg, device=device, dtype=torch.float32)
    state = initial_fields(cfg, grid)
    snap = {k: v.detach().cpu().numpy() for k, v in state.items()}
    x_field = extract_micro_tensor(snap, channels=field_channels, target_hw=target_hw)[None]  # [1,C,H,W]
    desc = extract_micro_descriptors(snap)
    x_desc = descriptor_vector(desc, descriptor_names)[None]  # [1,D]
    return x_field.to(dtype=torch.float32), torch.from_numpy(x_desc.astype(np.float32))


def _score_targets(targets: Dict[str, float], weights: Dict[str, float]) -> float:
    s = 0.0
    for k, w in weights.items():
        if k in targets:
            s += float(w) * float(targets[k])
    return float(s)


def run_inverse_design(
    *,
    base_cfg: SimulationConfig,
    model: torch.nn.Module,
    model_meta: Dict[str, object],
    variables: Sequence[DesignVariableSpec],
    opt_cfg: InverseDesignConfig,
    device: torch.device,
) -> InverseDesignResult:
    """执行 CEM 反向设计搜索。"""
    if not variables:
        raise ValueError("variables is empty.")
    rng = np.random.RandomState(int(opt_cfg.seed))
    var_names = [v.path for v in variables]
    lo = np.asarray([float(v.lower) for v in variables], dtype=np.float32)
    hi = np.asarray([float(v.upper) for v in variables], dtype=np.float32)
    x0 = np.asarray(
        [
            float(v.init if v.init is not None else 0.5 * (float(v.lower) + float(v.upper)))
            for v in variables
        ],
        dtype=np.float32,
    )
    std = np.maximum((hi - lo) * float(opt_cfg.init_std_ratio), 1e-6)
    mean = x0.copy()

    field_channels = list(model_meta.get("field_channels", ["phi", "c", "eta", "epspeq"]))
    desc_names = list(model_meta.get("descriptor_names", []))
    target_names = list(model_meta.get("target_names", []))
    scalers = model_meta.get("scalers", {})
    history: List[Dict[str, float]] = []

    best_score = -1e30
    best_vec = mean.copy()
    best_targets: Dict[str, float] = {}

    model.eval()
    for it in range(1, int(opt_cfg.iterations) + 1):
        pop = rng.normal(loc=mean[None, :], scale=std[None, :], size=(int(opt_cfg.population), len(variables))).astype(np.float32)
        pop = np.clip(pop, lo[None, :], hi[None, :])
        scores = np.zeros((pop.shape[0],), dtype=np.float32)
        targets_all: List[Dict[str, float]] = []
        with torch.no_grad():
            for i in range(pop.shape[0]):
                cfg_i = deepcopy(base_cfg)
                for j, spec in enumerate(variables):
                    _set_cfg_value(cfg_i, spec.path, float(pop[i, j]))
                xf, xd = _build_feature_inputs(
                    cfg_i,
                    field_channels=field_channels,
                    descriptor_names=desc_names,
                    target_hw=opt_cfg.target_hw,
                    device=device,
                )
                xf = xf.to(device=device, dtype=torch.float32)
                xd = xd.to(device=device, dtype=torch.float32)
                # 应用描述符标准化（若存在）。
                if isinstance(scalers, dict) and "x_desc" in scalers and "mean" in scalers["x_desc"] and desc_names:
                    mu = torch.tensor(np.asarray(scalers["x_desc"]["mean"], dtype=np.float32), device=device)
                    sd = torch.tensor(np.asarray(scalers["x_desc"]["std"], dtype=np.float32), device=device)
                    xd = (xd - mu) / sd
                out = model(xf, xd)
                y_hat = out["mean"].detach().cpu().numpy()
                if isinstance(scalers, dict) and "y" in scalers and "mean" in scalers["y"]:
                    y_hat = unscale_targets(y_hat, {"y": scalers["y"]})
                vals = {str(target_names[k]): float(y_hat[0, k]) for k in range(min(len(target_names), y_hat.shape[1]))}
                sc = _score_targets(vals, opt_cfg.objective_weights)
                scores[i] = float(sc)
                targets_all.append(vals)
                if sc > best_score:
                    best_score = float(sc)
                    best_vec = pop[i].copy()
                    best_targets = dict(vals)

        # CEM 更新：取精英集更新均值/方差。
        n_elite = max(2, int(round(pop.shape[0] * float(opt_cfg.elite_frac))))
        elite_idx = np.argsort(scores)[-n_elite:]
        elite = pop[elite_idx]
        mean = np.mean(elite, axis=0)
        std = np.maximum(np.std(elite, axis=0), 1e-6)
        history.append(
            {
                "iter": float(it),
                "best_score": float(np.max(scores)),
                "mean_score": float(np.mean(scores)),
                "global_best_score": float(best_score),
            }
        )

    best_params = {var_names[i]: float(best_vec[i]) for i in range(len(var_names))}
    return InverseDesignResult(
        best_score=float(best_score),
        best_params=best_params,
        best_targets=best_targets,
        history=history,
    )
