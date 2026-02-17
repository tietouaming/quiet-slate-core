"""跨尺度主动学习闭环。"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..config import SimulationConfig
from ..geometry import initial_fields, make_grid
from .advanced_models import AdvancedModelConfig, save_advanced_model, unscale_advanced_targets
from .advanced_train import AdvancedTrainingConfig, train_advanced_multiscale_model
from .data_generation import PhysicsDataGenConfig, SweepVariable, generate_physics_cases
from .dataset import (
    MultiFidelityDatasetConfig,
    build_multifidelity_dataset_from_cases,
    discover_case_dirs,
    save_multiscale_dataset_npz,
)
from .features import descriptor_vector, extract_micro_descriptors, extract_micro_tensor
from .physics_metrics import PhysicsMetricConfig


@dataclass
class ActiveLearningConfig:
    """主动学习配置。"""

    rounds: int = 3
    initial_cases: int = 16
    new_cases_per_round: int = 8
    candidate_pool: int = 64
    seed: int = 42
    # 采样评分 = alpha*uncertainty + beta*objective_priority
    alpha_uncertainty: float = 1.0
    beta_objective: float = 0.15
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "corrosion_loss": 1.0,
            "penetration_x_ratio": 0.8,
            "sigma_h_max_solid": 0.2,
        }
    )
    # 数据构建参数
    target_hw: Tuple[int, int] = (128, 128)
    low_target_hw: Tuple[int, int] = (64, 64)
    horizon_steps: int = 1
    frame_stride: int = 1
    high_fidelity_keep_ratio: float = 0.35
    # 单算例仿真预算
    sim_steps: int = 260
    sim_save_every: int = 26
    # 输出根目录
    output_root: str = "artifacts/active_learning"
    case_prefix: str = "al_case"


def _set_cfg_value(cfg: SimulationConfig, path: str, value: float) -> None:
    keys = str(path).split(".")
    obj = cfg
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], float(value))


def _sample_candidate_table(
    vars_spec: Sequence[SweepVariable],
    n: int,
    seed: int,
) -> List[Dict[str, float]]:
    rs = np.random.RandomState(int(seed))
    out: List[Dict[str, float]] = []
    for _ in range(int(n)):
        row: Dict[str, float] = {}
        for v in vars_spec:
            lo = float(v.lower)
            hi = float(v.upper)
            if bool(v.log_scale):
                lo2 = max(lo, 1e-12)
                hi2 = max(hi, lo2 * 1.0001)
                x = rs.uniform(np.log(lo2), np.log(hi2))
                row[v.path] = float(np.exp(x))
            else:
                row[v.path] = float(rs.uniform(lo, hi))
        out.append(row)
    return out


def _prepare_feature_for_row(
    base_cfg: SimulationConfig,
    row: Dict[str, float],
    *,
    field_channels: Sequence[str],
    descriptor_names: Sequence[str],
    target_hw: Tuple[int, int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg_i = deepcopy(base_cfg)
    for k, v in row.items():
        _set_cfg_value(cfg_i, k, float(v))
    g = make_grid(cfg_i, device=device, dtype=torch.float32)
    st = initial_fields(cfg_i, g)
    snap = {}
    for k, v in st.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
                arr = arr[0, 0]
            snap[k] = arr
    x_field = extract_micro_tensor(snap, channels=field_channels, target_hw=target_hw)[None]
    desc = extract_micro_descriptors(snap)
    x_desc = descriptor_vector(desc, descriptor_names)[None]
    return x_field.to(dtype=torch.float32), torch.from_numpy(x_desc.astype(np.float32))


def _score_predictions(
    y_mean: np.ndarray,
    y_logvar: np.ndarray,
    target_names: Sequence[str],
    *,
    alpha_uncertainty: float,
    beta_objective: float,
    objective_weights: Dict[str, float],
) -> np.ndarray:
    tmap = {str(n): i for i, n in enumerate(target_names)}
    unc = np.mean(np.sqrt(np.exp(y_logvar)), axis=1)
    obj = np.zeros((y_mean.shape[0],), dtype=np.float64)
    for k, w in objective_weights.items():
        j = tmap.get(str(k))
        if j is None:
            continue
        obj += float(w) * y_mean[:, j]
    return float(alpha_uncertainty) * unc + float(beta_objective) * obj


def _append_cases_via_generation(
    *,
    base_cfg: SimulationConfig,
    rows: Sequence[Dict[str, float]],
    round_id: int,
    cfg: ActiveLearningConfig,
) -> Dict[str, object]:
    # 复用现有批量生成器，保证输出目录结构一致。
    tmp_vars: List[SweepVariable] = [SweepVariable(path=k, lower=float(v), upper=float(v), log_scale=False) for k, v in rows[0].items()]
    gen_cfg = PhysicsDataGenConfig(
        n_cases=len(rows),
        seed=int(cfg.seed + round_id * 101),
        output_root=str(Path(cfg.output_root) / "cases"),
        case_prefix=f"{cfg.case_prefix}_r{round_id}",
        clean_output=True,
        n_steps=int(cfg.sim_steps),
        save_every=int(cfg.sim_save_every),
        disable_ml=True,
        render_intermediate_fields=False,
        render_final_clouds=False,
        render_grid_figure=False,
        progress=False,
    )

    # generate_physics_cases 内部按范围随机采样；这里将每个变量下界=上界锁定，实现“按指定候选”运行。
    # 因此依次执行每个候选，避免随机误差。
    results = []
    fails = []
    for i, row in enumerate(rows):
        vars_i = [SweepVariable(path=k, lower=float(v), upper=float(v), log_scale=False) for k, v in row.items()]
        one_cfg = deepcopy(gen_cfg)
        one_cfg.n_cases = 1
        one_cfg.case_prefix = f"{cfg.case_prefix}_r{round_id}_c{i:03d}"
        s = generate_physics_cases(base_cfg=base_cfg, vars_spec=vars_i, gen_cfg=one_cfg)
        if int(s.get("n_succeeded", 0)) > 0:
            results.extend(s.get("cases", []))
        if int(s.get("n_failed", 0)) > 0:
            fails.extend(s.get("failed", []))
    return {"cases": results, "failed": fails}


def run_active_learning_loop(
    *,
    base_cfg: SimulationConfig,
    vars_spec: Sequence[SweepVariable],
    model_cfg: AdvancedModelConfig,
    train_cfg: AdvancedTrainingConfig,
    al_cfg: ActiveLearningConfig,
    device: torch.device,
) -> Dict[str, object]:
    """执行主动学习循环。"""
    out_root = Path(al_cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cases_root = out_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    reports: List[Dict[str, object]] = []

    # Round 0: 初始随机样本
    init_gen = PhysicsDataGenConfig(
        n_cases=int(al_cfg.initial_cases),
        seed=int(al_cfg.seed),
        output_root=str(cases_root),
        case_prefix=f"{al_cfg.case_prefix}_r0",
        clean_output=True,
        n_steps=int(al_cfg.sim_steps),
        save_every=int(al_cfg.sim_save_every),
        disable_ml=True,
        render_intermediate_fields=False,
        render_final_clouds=False,
        render_grid_figure=False,
        progress=False,
    )
    init_summary = generate_physics_cases(base_cfg=base_cfg, vars_spec=vars_spec, gen_cfg=init_gen)
    reports.append({"round": 0, "generation": init_summary})

    latest_model_path = out_root / "advanced_model_latest.pt"
    latest_dataset_path = out_root / "multifidelity_dataset_latest.npz"

    for rd in range(int(al_cfg.rounds)):
        # 1) 重新构建全量数据集
        case_dirs = discover_case_dirs(cases_root, pattern=f"{al_cfg.case_prefix}_*", require_history=False)
        ds_cfg = MultiFidelityDatasetConfig(
            target_hw=tuple(int(v) for v in al_cfg.target_hw),
            low_target_hw=tuple(int(v) for v in al_cfg.low_target_hw),
            horizon_steps=max(1, int(al_cfg.horizon_steps)),
            frame_stride=max(1, int(al_cfg.frame_stride)),
            high_fidelity_keep_ratio=float(al_cfg.high_fidelity_keep_ratio),
            seed=int(al_cfg.seed + rd * 13),
        )
        ds = build_multifidelity_dataset_from_cases(case_dirs, ds_cfg)
        save_multiscale_dataset_npz(latest_dataset_path, ds)

        # 2) 训练高级模型
        model, train_rep = train_advanced_multiscale_model(
            ds,
            model_cfg,
            train_cfg,
            device=device,
            physics_cfg=PhysicsMetricConfig(),
        )
        save_advanced_model(
            latest_model_path,
            model,
            model_cfg,
            model_name=str(train_cfg.model_name),
            field_channels=[str(v) for v in ds["field_channels"].tolist()],
            descriptor_names=[str(v) for v in ds["descriptor_names"].tolist()],
            target_names=[str(v) for v in ds["target_names"].tolist()],
            scalers=train_rep["scalers"],
            extra_meta={"round": rd},
        )

        round_payload: Dict[str, object] = {
            "round": rd,
            "n_cases_total": len(case_dirs),
            "n_samples": int(ds["x_field"].shape[0]),
            "train_report": train_rep,
        }

        if rd == int(al_cfg.rounds) - 1:
            reports.append(round_payload)
            break

        # 3) 候选池不确定性选样
        candidates = _sample_candidate_table(vars_spec, int(al_cfg.candidate_pool), seed=int(al_cfg.seed + 1000 + rd))
        field_channels = [str(v) for v in ds["field_channels"].tolist()]
        desc_names = [str(v) for v in ds["descriptor_names"].tolist()]
        target_names = [str(v) for v in ds["target_names"].tolist()]
        scalers = train_rep["scalers"]

        y_mean_list: List[np.ndarray] = []
        y_logv_list: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for row in candidates:
                xf, xd = _prepare_feature_for_row(
                    base_cfg,
                    row,
                    field_channels=field_channels,
                    descriptor_names=desc_names,
                    target_hw=tuple(int(v) for v in al_cfg.target_hw),
                    device=device,
                )
                xf = xf.to(device=device)
                xd = xd.to(device=device)
                if "x_desc" in scalers and "mean" in scalers["x_desc"]:
                    mu = torch.from_numpy(np.asarray(scalers["x_desc"]["mean"], dtype=np.float32)).to(device=device)
                    sd = torch.from_numpy(np.asarray(scalers["x_desc"]["std"], dtype=np.float32)).to(device=device)
                    xd = (xd - mu) / sd
                out = model(xf, xd)
                if "high_mean" in out:
                    ym_s = out["high_mean"].detach().cpu().numpy()
                    yv_s = out["high_logvar"].detach().cpu().numpy()
                else:
                    ym_s = out["mean"].detach().cpu().numpy()
                    yv_s = out["logvar"].detach().cpu().numpy()
                ym = unscale_advanced_targets(ym_s, scalers, key="y_high")
                std = np.asarray(scalers["y_high"]["std"], dtype=np.float32).reshape(1, -1)
                yv = np.log(np.maximum(np.exp(yv_s) * (std ** 2), 1e-18))
                y_mean_list.append(ym[0])
                y_logv_list.append(yv[0])
        y_mean_np = np.asarray(y_mean_list, dtype=np.float64)
        y_logv_np = np.asarray(y_logv_list, dtype=np.float64)
        scores = _score_predictions(
            y_mean_np,
            y_logv_np,
            target_names,
            alpha_uncertainty=float(al_cfg.alpha_uncertainty),
            beta_objective=float(al_cfg.beta_objective),
            objective_weights=al_cfg.objective_weights,
        )
        idx = np.argsort(scores)[-max(1, int(al_cfg.new_cases_per_round)) :]
        selected = [candidates[int(i)] for i in idx.tolist()]
        append_summary = _append_cases_via_generation(
            base_cfg=base_cfg,
            rows=selected,
            round_id=rd + 1,
            cfg=al_cfg,
        )
        round_payload["selection"] = {
            "scores_top": [float(scores[int(i)]) for i in idx.tolist()],
            "selected_params": selected,
        }
        round_payload["append_generation"] = append_summary
        reports.append(round_payload)

    final = {
        "output_root": str(out_root),
        "latest_model": str(latest_model_path),
        "latest_dataset": str(latest_dataset_path),
        "round_reports": reports,
    }
    return final

