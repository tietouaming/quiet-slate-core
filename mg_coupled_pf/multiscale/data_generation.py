"""基于物理仿真的多尺度数据自动构造器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..config import SimulationConfig
from ..simulator import CoupledSimulator


@dataclass
class SweepVariable:
    """参数扫描变量定义。"""

    path: str
    lower: float
    upper: float
    log_scale: bool = False


@dataclass
class PhysicsDataGenConfig:
    """物理数据生成配置。"""

    n_cases: int = 24
    seed: int = 42
    output_root: str = "artifacts/sim_multiscale_dataset"
    case_prefix: str = "ms_case"
    clean_output: bool = True
    # 每个算例的仿真时长控制
    n_steps: int = 300
    save_every: int = 30
    # 运行性能参数（默认更偏数据生产）
    disable_ml: bool = True
    render_intermediate_fields: bool = False
    render_final_clouds: bool = False
    render_grid_figure: bool = False
    progress: bool = False
    progress_every: int = 100


def _set_cfg_value(cfg: SimulationConfig, path: str, value: float) -> None:
    keys = str(path).split(".")
    obj = cfg
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], float(value))


def _sample_variable(rs: np.random.RandomState, spec: SweepVariable) -> float:
    lo = float(spec.lower)
    hi = float(spec.upper)
    if spec.log_scale:
        lo2 = max(lo, 1e-12)
        hi2 = max(hi, lo2 * 1.0001)
        x = rs.uniform(np.log(lo2), np.log(hi2))
        return float(np.exp(x))
    return float(rs.uniform(lo, hi))


def sample_sweep_table(
    vars_spec: Sequence[SweepVariable],
    n_cases: int,
    seed: int,
) -> List[Dict[str, float]]:
    """按均匀随机采样生成参数表。"""
    rs = np.random.RandomState(int(seed))
    table: List[Dict[str, float]] = []
    for _ in range(int(n_cases)):
        row: Dict[str, float] = {}
        for v in vars_spec:
            row[v.path] = _sample_variable(rs, v)
        table.append(row)
    return table


def generate_physics_cases(
    *,
    base_cfg: SimulationConfig,
    vars_spec: Sequence[SweepVariable],
    gen_cfg: PhysicsDataGenConfig,
) -> Dict[str, object]:
    """批量生成物理算例并返回汇总信息。"""
    out_root = Path(gen_cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    table = sample_sweep_table(vars_spec, gen_cfg.n_cases, gen_cfg.seed)
    produced: List[str] = []
    failed: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []

    for i, row in enumerate(table):
        cfg_i = deepcopy(base_cfg)
        cfg_i.numerics.n_steps = int(gen_cfg.n_steps)
        cfg_i.numerics.save_every = int(gen_cfg.save_every)
        cfg_i.ml.enabled = not bool(gen_cfg.disable_ml)
        cfg_i.runtime.clean_output = bool(gen_cfg.clean_output)
        cfg_i.runtime.render_intermediate_fields = bool(gen_cfg.render_intermediate_fields)
        cfg_i.runtime.render_final_clouds = bool(gen_cfg.render_final_clouds)
        cfg_i.runtime.render_grid_figure = bool(gen_cfg.render_grid_figure)
        cfg_i.runtime.output_dir = str(out_root)
        cfg_i.runtime.case_name = f"{gen_cfg.case_prefix}_{i:04d}"
        for k, v in row.items():
            _set_cfg_value(cfg_i, k, float(v))
        try:
            sim = CoupledSimulator(cfg_i)
            out = sim.run(
                progress=bool(gen_cfg.progress),
                progress_every=max(1, int(gen_cfg.progress_every)),
                progress_prefix=cfg_i.runtime.case_name,
            )
            produced.append(str(Path(out["output_dir"])))
            rows.append(
                {
                    "case_name": cfg_i.runtime.case_name,
                    "output_dir": str(Path(out["output_dir"])),
                    "history_csv": str(Path(out["history_csv"])),
                    "snapshots_dir": str(Path(out["snapshots_dir"])),
                    "wall_time_s": float(out.get("wall_time_s", 0.0)),
                    "params": {k: float(v) for k, v in row.items()},
                }
            )
        except Exception as ex:
            failed.append({"case_name": cfg_i.runtime.case_name, "error": str(ex), "params": row})

    summary = {
        "output_root": str(out_root),
        "n_requested": int(gen_cfg.n_cases),
        "n_succeeded": int(len(produced)),
        "n_failed": int(len(failed)),
        "cases": rows,
        "failed": failed,
        "sweep_variables": [vars(v) for v in vars_spec],
    }
    return summary

