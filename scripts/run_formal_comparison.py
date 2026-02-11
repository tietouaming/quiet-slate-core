#!/usr/bin/env python
"""正式对照脚本（中文注释版）。

该脚本在同一时间设置下并行执行两条流程：
1. 纯物理求解（基准）
2. 启用 ML surrogate 的求解

随后输出：
- 性能对比（耗时、加速比、surrogate 占比）
- 误差对比（历史量与最终场）
- 可视化图与网格图
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf import CoupledSimulator, load_config
from mg_coupled_pf.io_utils import (
    ensure_dir,
    render_fields,
    render_final_clouds,
    render_grid_figure,
    save_history_csv,
    save_snapshot_npz,
)


SCALAR_KEYS = ["solid_fraction", "avg_eta", "max_sigma_h", "avg_epspeq"]
FIELD_KEYS = ["phi", "c", "eta", "epspeq"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="Run long formal comparison with online progress/loss reporting.")
    p.add_argument("--config", default="configs/notch_case.yaml")
    p.add_argument("--steps", type=int, default=1600, help="Legacy override for number of steps.")
    p.add_argument("--save-every", type=int, default=1600, help="Legacy override for save interval in steps.")
    p.add_argument("--dt-s", type=float, default=0.0, help="Physical time step size (seconds).")
    p.add_argument("--total-time-s", type=float, default=0.0, help="Total physical simulation time (seconds).")
    p.add_argument("--save-interval-s", type=float, default=0.0, help="Output interval in physical time (seconds).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--disable-torch-compile", action="store_true")
    p.add_argument("--case-tag", default="formal1600")
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--report-json", default="artifacts/reports/formal_1600_report.json")
    p.add_argument("--report-md", default="artifacts/reports/formal_1600_report.md")
    return p.parse_args()


def _resolve_time_controls(cfg, args: argparse.Namespace) -> tuple[int, int]:
    """根据 dt/总物理时长解析为步数与保存间隔。"""
    if args.dt_s > 0:
        # CLI 优先：显式指定的 dt 覆盖配置文件值。
        cfg.numerics.dt_s = args.dt_s
    if args.total_time_s > 0:
        # 推荐口径：由总物理时间反推步数，保证实验可复现。
        steps = max(1, int(math.ceil(args.total_time_s / max(cfg.numerics.dt_s, 1e-12))))
    else:
        steps = max(1, int(args.steps))
    if args.save_interval_s > 0:
        # 输出同样按物理时间间隔控制，而非固定步数。
        save_every = max(1, int(math.ceil(args.save_interval_s / max(cfg.numerics.dt_s, 1e-12))))
    else:
        save_every = max(1, int(args.save_every))
    return steps, save_every


def _sync_if_cuda(device: torch.device) -> None:
    """CUDA 场景下同步设备，保证计时准确。"""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _apply_runtime_flags(cfg, args: argparse.Namespace) -> None:
    """把命令行运行时参数覆盖到配置对象。"""
    if args.device != "auto":
        cfg.runtime.device = args.device
    if args.mixed_precision:
        cfg.numerics.mixed_precision = True
    if args.disable_torch_compile:
        cfg.numerics.use_torch_compile = False


def _history_row(diag) -> Dict[str, float]:
    """将单步诊断对象转换为可序列化行记录。"""
    return {
        "step": float(diag.step),
        "time_s": float(diag.time_s),
        "solid_fraction": float(diag.solid_fraction),
        "avg_eta": float(diag.avg_eta),
        "max_sigma_h": float(diag.max_sigma_h),
        "avg_epspeq": float(diag.avg_epspeq),
        "used_surrogate_only": float(diag.used_surrogate_only),
    }


def _series_error(a: List[float], b: List[float]) -> Dict[str, float]:
    """计算两条标量序列的 MAE/RMSE/最大绝对误差。"""
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    d = torch.abs(ta - tb)
    mse = torch.mean((ta - tb) ** 2)
    return {
        "mae": float(torch.mean(d).item()),
        "rmse": float(torch.sqrt(mse).item()),
        "max_abs": float(torch.max(d).item()),
    }


def _field_error(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """计算两个场张量的 MAE/RMSE/最大绝对误差。"""
    d = torch.abs(a - b)
    return {
        "mae": float(torch.mean(d).item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
        "max_abs": float(torch.max(d).item()),
    }


def _prepare_case_dirs(sim: CoupledSimulator) -> Dict[str, Path]:
    """准备并按需清理算例输出目录。"""
    out_dir = ensure_dir(Path(sim.cfg.runtime.output_dir) / sim.cfg.runtime.case_name)
    snap_dir = ensure_dir(out_dir / "snapshots")
    fig_dir = ensure_dir(out_dir / "figures")
    grid_dir = ensure_dir(out_dir / "grid")
    if sim.cfg.runtime.clean_output:
        # 按文件类型精确清理，避免历史结果混入本次实验。
        for p in snap_dir.glob("snapshot_*.npz"):
            p.unlink(missing_ok=True)
        for p in fig_dir.glob("fields_*.*"):
            p.unlink(missing_ok=True)
        for p in grid_dir.glob("*.*"):
            p.unlink(missing_ok=True)
        (out_dir / "history.csv").unlink(missing_ok=True)
        cloud_dir = out_dir / "final_clouds"
        if cloud_dir.exists():
            for p in cloud_dir.glob("*.*"):
                p.unlink(missing_ok=True)
        tip_dir = out_dir / "final_clouds_tip_zoom"
        if tip_dir.exists():
            for p in tip_dir.glob("*.*"):
                p.unlink(missing_ok=True)
    return {"out_dir": out_dir, "snap_dir": snap_dir, "fig_dir": fig_dir, "grid_dir": grid_dir}


def _save_step_outputs(sim: CoupledSimulator, dirs: Dict[str, Path], step: int) -> None:
    """保存单步快照与中间图。"""
    save_snapshot_npz(
        dirs["snap_dir"],
        step,
        state=sim.state,
        extras={
            "sigma_h": sim.last_mech["sigma_h"],
            "sigma_xx": sim.last_mech["sigma_xx"],
            "sigma_yy": sim.last_mech["sigma_yy"],
            "sigma_xy": sim.last_mech["sigma_xy"],
        },
    )
    render_fields(
        dirs["fig_dir"],
        step,
        sim.state,
        sim.last_mech["sigma_h"],
        sim.state["epspeq"],
        time_s=step * sim.cfg.numerics.dt_s,
        extent_um=(0.0, sim.cfg.domain.lx_um, 0.0, sim.cfg.domain.ly_um),
        show_grid=False,
        x_coords_um=sim.x_coords_um,
        y_coords_um=sim.y_coords_um,
    )


def _progress_bar(frac: float, width: int = 26) -> str:
    """生成文本进度条。"""
    frac = max(0.0, min(1.0, frac))
    fill = int(width * frac)
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def _fmt_wall_time(seconds: float) -> str:
    """格式化运行时间字符串。"""
    s = max(float(seconds), 0.0)
    h = int(s // 3600.0)
    m = int((s % 3600.0) // 60.0)
    sec = s - 3600.0 * h - 60.0 * m
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:05.2f}"
    return f"{m:02d}:{sec:05.2f}"


def build_report(
    args: argparse.Namespace,
    phy_sim: CoupledSimulator,
    ml_sim: CoupledSimulator,
    phy_hist: List[Dict[str, float]],
    ml_hist: List[Dict[str, float]],
    phy_time_s: float,
    ml_time_s: float,
    loss_log: Dict[str, List[float]],
    out_phy: Dict[str, Path],
    out_ml: Dict[str, Path],
) -> Dict[str, Any]:
    """汇总性能与精度对照报告。"""
    # 历史序列按较短长度对齐，避免越界并保证逐步配对。
    n = min(len(phy_hist), len(ml_hist))
    series_metrics: Dict[str, Dict[str, float]] = {}
    for k in SCALAR_KEYS:
        a = [phy_hist[i][k] for i in range(n)]
        b = [ml_hist[i][k] for i in range(n)]
        series_metrics[k] = _series_error(a, b)

    final_field_metrics: Dict[str, Dict[str, float]] = {}
    for k in FIELD_KEYS:
        # 最终场误差：直接比较结束时刻的场分布。
        a = phy_sim.state[k].detach().float().cpu()
        b = ml_sim.state[k].detach().float().cpu()
        final_field_metrics[k] = _field_error(a, b)

    surrogate_steps = int(sum(int(float(r["used_surrogate_only"])) for r in ml_hist))
    # speedup > 1 代表在同等物理时长下 ML 更快。
    speedup = phy_time_s / max(ml_time_s, 1e-12)
    dt_s = float(phy_sim.cfg.numerics.dt_s)
    steps = int(len(phy_hist))
    total_time_s = float(steps * dt_s)
    save_every = int(phy_sim.cfg.numerics.save_every)
    save_interval_s = float(save_every * dt_s)

    report = {
        "meta": {
            "config": args.config,
            "dt_s": dt_s,
            "total_time_s": total_time_s,
            "n_steps": steps,
            "save_every": save_every,
            "save_interval_s": save_interval_s,
            "device_request": args.device,
            "mixed_precision": bool(args.mixed_precision),
            "torch_compile_enabled": not bool(args.disable_torch_compile),
            "ml_model_arch": str(ml_sim.cfg.ml.model_arch),
        },
        "environment": {
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
        "performance": {
            "physics": {
                "use_ml": False,
                "device": str(phy_sim.device),
                "dt_s": dt_s,
                "total_time_s": total_time_s,
                "n_steps": steps,
                "save_every": save_every,
                "wall_time_s": float(phy_time_s),
                "simulated_physical_time_per_wall_s": float(total_time_s / max(phy_time_s, 1e-12)),
                "output_dir": str(out_phy["out_dir"]),
                "history_csv": str(out_phy["history_csv"]),
                "snapshots_dir": str(out_phy["snap_dir"]),
                "figures_dir": str(out_phy["fig_dir"]),
                "grid_dir": str(out_phy["grid_dir"]),
                "solver_stats": dict(phy_sim.stats),
            },
            "ml": {
                "use_ml": True,
                "device": str(ml_sim.device),
                "dt_s": dt_s,
                "total_time_s": total_time_s,
                "n_steps": steps,
                "save_every": save_every,
                "wall_time_s": float(ml_time_s),
                "simulated_physical_time_per_wall_s": float(total_time_s / max(ml_time_s, 1e-12)),
                "output_dir": str(out_ml["out_dir"]),
                "history_csv": str(out_ml["history_csv"]),
                "snapshots_dir": str(out_ml["snap_dir"]),
                "figures_dir": str(out_ml["fig_dir"]),
                "grid_dir": str(out_ml["grid_dir"]),
                "solver_stats": dict(ml_sim.stats),
            },
            "speedup_physics_over_ml": float(speedup),
            "ml_surrogate_steps": surrogate_steps,
            "ml_surrogate_ratio": float(surrogate_steps / max(len(ml_hist), 1)),
        },
        "accuracy_vs_physics": {
            "history_metrics": series_metrics,
            "final_field_metrics": final_field_metrics,
            "physics_final": phy_hist[-1] if phy_hist else {},
            "ml_final": ml_hist[-1] if ml_hist else {},
            "final_solid_fraction_abs_diff": abs(
                float((phy_hist[-1] if phy_hist else {}).get("solid_fraction", 0.0))
                - float((ml_hist[-1] if ml_hist else {}).get("solid_fraction", 0.0))
            ),
        },
        "online_losses": {
            k: {
                "mean": float(torch.tensor(v, dtype=torch.float64).mean().item()) if v else 0.0,
                "max": float(torch.tensor(v, dtype=torch.float64).max().item()) if v else 0.0,
                "last": float(v[-1]) if v else 0.0,
            }
            for k, v in loss_log.items()
        },
    }
    return report


def report_to_markdown(report: Dict[str, Any]) -> str:
    """把 JSON 报告渲染为 Markdown 文本。"""
    p = report["performance"]
    m = report["meta"]
    a = report["accuracy_vs_physics"]
    hm = a["history_metrics"]
    fm = a["final_field_metrics"]
    ol = report["online_losses"]
    lines = [
        "# 正式算例性能/精度对照报告（按物理时间）",
        "",
        "## 0) 时间设置",
        f"- 时间步长 dt: {m['dt_s']:.6e} s",
        f"- 总物理时长: {m['total_time_s']:.6e} s",
        f"- 等效步数: {m['n_steps']}",
        f"- 输出时间间隔: {m['save_interval_s']:.6e} s",
        "",
        "## 1) 性能",
        f"- 纯物理总耗时: {p['physics']['wall_time_s']:.3f} s",
        f"- ML总耗时: {p['ml']['wall_time_s']:.3f} s",
        f"- 加速比(Physics/ML): {p['speedup_physics_over_ml']:.4f}",
        f"- ML surrogate 步数: {p['ml_surrogate_steps']} ({p['ml_surrogate_ratio']:.2%})",
        "",
        "## 2) 历史量误差（相对纯物理）",
        f"- solid_fraction: MAE={hm['solid_fraction']['mae']:.6e}, RMSE={hm['solid_fraction']['rmse']:.6e}, MAX={hm['solid_fraction']['max_abs']:.6e}",
        f"- avg_eta: MAE={hm['avg_eta']['mae']:.6e}, RMSE={hm['avg_eta']['rmse']:.6e}, MAX={hm['avg_eta']['max_abs']:.6e}",
        f"- max_sigma_h: MAE={hm['max_sigma_h']['mae']:.6e}, RMSE={hm['max_sigma_h']['rmse']:.6e}, MAX={hm['max_sigma_h']['max_abs']:.6e}",
        f"- avg_epspeq: MAE={hm['avg_epspeq']['mae']:.6e}, RMSE={hm['avg_epspeq']['rmse']:.6e}, MAX={hm['avg_epspeq']['max_abs']:.6e}",
        "",
        "## 3) 最终场误差（相对纯物理）",
        f"- phi: MAE={fm['phi']['mae']:.6e}, RMSE={fm['phi']['rmse']:.6e}, MAX={fm['phi']['max_abs']:.6e}",
        f"- c: MAE={fm['c']['mae']:.6e}, RMSE={fm['c']['rmse']:.6e}, MAX={fm['c']['max_abs']:.6e}",
        f"- eta: MAE={fm['eta']['mae']:.6e}, RMSE={fm['eta']['rmse']:.6e}, MAX={fm['eta']['max_abs']:.6e}",
        f"- epspeq: MAE={fm['epspeq']['mae']:.6e}, RMSE={fm['epspeq']['rmse']:.6e}, MAX={fm['epspeq']['max_abs']:.6e}",
        "",
        "## 4) 在线损失统计（计算过程中）",
        f"- loss_solid: mean={ol['loss_solid']['mean']:.6e}, max={ol['loss_solid']['max']:.6e}, last={ol['loss_solid']['last']:.6e}",
        f"- loss_eta: mean={ol['loss_eta']['mean']:.6e}, max={ol['loss_eta']['max']:.6e}, last={ol['loss_eta']['last']:.6e}",
        f"- loss_sigma_h: mean={ol['loss_sigma_h']['mean']:.6e}, max={ol['loss_sigma_h']['max']:.6e}, last={ol['loss_sigma_h']['last']:.6e}",
        f"- loss_epspeq: mean={ol['loss_epspeq']['mean']:.6e}, max={ol['loss_epspeq']['max']:.6e}, last={ol['loss_epspeq']['last']:.6e}",
        f"- loss_phi_mae: mean={ol['loss_phi_mae']['mean']:.6e}, max={ol['loss_phi_mae']['max']:.6e}, last={ol['loss_phi_mae']['last']:.6e}",
        f"- loss_c_mae: mean={ol['loss_c_mae']['mean']:.6e}, max={ol['loss_c_mae']['max']:.6e}, last={ol['loss_c_mae']['last']:.6e}",
        f"- loss_eta_mae: mean={ol['loss_eta_mae']['mean']:.6e}, max={ol['loss_eta_mae']['max']:.6e}, last={ol['loss_eta_mae']['last']:.6e}",
        "",
        "## 5) 关键结论",
        f"- 最终固相分数绝对差: {a['final_solid_fraction_abs_diff']:.6e}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """脚本主流程。"""
    args = parse_args()

    # 1) 构建两条分支配置：纯物理分支与 ML 分支。
    phy_cfg = load_config(args.config)
    ml_cfg = load_config(args.config)
    phy_steps, phy_save_every = _resolve_time_controls(phy_cfg, args)
    ml_steps, ml_save_every = _resolve_time_controls(ml_cfg, args)
    # Keep both branches at same temporal resolution.
    if (phy_steps != ml_steps) or (phy_save_every != ml_save_every):
        # 强制统一时间网格，是误差公平比较的前提。
        ml_steps = phy_steps
        ml_save_every = phy_save_every
    for cfg in (phy_cfg, ml_cfg):
        cfg.numerics.n_steps = phy_steps
        cfg.numerics.save_every = phy_save_every
        cfg.runtime.clean_output = True
        _apply_runtime_flags(cfg, args)
    phy_cfg.ml.enabled = False
    ml_cfg.ml.enabled = True
    phy_cfg.runtime.case_name = f"{phy_cfg.runtime.case_name}_{args.case_tag}_physics"
    ml_cfg.runtime.case_name = f"{ml_cfg.runtime.case_name}_{args.case_tag}_ml"

    # 2) 初始化两个仿真器与输出目录。
    phy_sim = CoupledSimulator(phy_cfg)
    ml_sim = CoupledSimulator(ml_cfg)

    out_phy = _prepare_case_dirs(phy_sim)
    out_ml = _prepare_case_dirs(ml_sim)

    phy_history: List[Dict[str, float]] = []
    ml_history: List[Dict[str, float]] = []
    loss_log: Dict[str, List[float]] = {
        "loss_solid": [],
        "loss_eta": [],
        "loss_sigma_h": [],
        "loss_epspeq": [],
        "loss_phi_mae": [],
        "loss_c_mae": [],
        "loss_eta_mae": [],
    }

    phy_time = 0.0
    ml_time = 0.0
    _sync_if_cuda(phy_sim.device)

    total_steps = int(phy_cfg.numerics.n_steps)
    dt_s = float(phy_cfg.numerics.dt_s)
    total_time_s = total_steps * dt_s
    t_wall_start = time.perf_counter()
    for i in range(1, total_steps + 1):
        # A) 推进一步纯物理分支并计时。
        t0 = time.perf_counter()
        phy_diag = phy_sim.step(i)
        _sync_if_cuda(phy_sim.device)
        phy_time += time.perf_counter() - t0
        phy_row = _history_row(phy_diag)
        phy_history.append(phy_row)
        if i % phy_cfg.numerics.save_every == 0 or i == 1:
            _save_step_outputs(phy_sim, out_phy, i)

        # B) 推进一步 ML 分支并计时。
        t0 = time.perf_counter()
        ml_diag = ml_sim.step(i)
        _sync_if_cuda(ml_sim.device)
        ml_time += time.perf_counter() - t0
        ml_row = _history_row(ml_diag)
        ml_history.append(ml_row)
        if i % ml_cfg.numerics.save_every == 0 or i == 1:
            _save_step_outputs(ml_sim, out_ml, i)

        # C) 在线误差：同一步时刻 ML 与物理分支的横向差。
        loss_solid = abs(ml_diag.solid_fraction - phy_diag.solid_fraction)
        loss_eta = abs(ml_diag.avg_eta - phy_diag.avg_eta)
        loss_sigma = abs(ml_diag.max_sigma_h - phy_diag.max_sigma_h)
        loss_epspeq = abs(ml_diag.avg_epspeq - phy_diag.avg_epspeq)
        loss_phi_mae = float(torch.mean(torch.abs(ml_sim.state["phi"] - phy_sim.state["phi"])).item())
        loss_c_mae = float(torch.mean(torch.abs(ml_sim.state["c"] - phy_sim.state["c"])).item())
        loss_eta_mae = float(torch.mean(torch.abs(ml_sim.state["eta"] - phy_sim.state["eta"])).item())
        loss_log["loss_solid"].append(float(loss_solid))
        loss_log["loss_eta"].append(float(loss_eta))
        loss_log["loss_sigma_h"].append(float(loss_sigma))
        loss_log["loss_epspeq"].append(float(loss_epspeq))
        loss_log["loss_phi_mae"].append(float(loss_phi_mae))
        loss_log["loss_c_mae"].append(float(loss_c_mae))
        loss_log["loss_eta_mae"].append(float(loss_eta_mae))

        if i == 1 or i % max(1, args.progress_every) == 0 or i == total_steps:
            frac = i / max(total_steps, 1)
            bar = _progress_bar(frac)
            t_now = i * dt_s
            wall_elapsed_s = time.perf_counter() - t_wall_start
            wall_eta_s = wall_elapsed_s * (1.0 - frac) / max(frac, 1e-12)
            print(
                f"[formal] {bar} {frac*100:6.2f}% t={t_now:.6f}/{total_time_s:.6f} s dt={dt_s:.2e} s "
                f"wall={_fmt_wall_time(wall_elapsed_s)} eta_wall={_fmt_wall_time(wall_eta_s)} "
                f"loss_solid={loss_solid:.3e} loss_eta={loss_eta:.3e} "
                f"loss_sigma={loss_sigma:.3e} loss_epspeq={loss_epspeq:.3e} "
                f"loss_phi_mae={loss_phi_mae:.3e} loss_c_mae={loss_c_mae:.3e} "
                f"phy_t/wall={t_now/max(phy_time,1e-12):.3e} ml_t/wall={t_now/max(ml_time,1e-12):.3e}"
            )

    # 3) 保存两条分支历史。
    out_phy["history_csv"] = save_history_csv(out_phy["out_dir"], phy_history)
    out_ml["history_csv"] = save_history_csv(out_ml["out_dir"], ml_history)

    def _notch_meta(cfg):
        if cfg.domain.initial_geometry.lower() == "half_space" and cfg.domain.add_half_space_triangular_notch:
            return (
                cfg.domain.half_space_notch_tip_x_um,
                cfg.domain.half_space_notch_center_y_um,
                cfg.domain.half_space_notch_depth_um,
                cfg.domain.half_space_notch_half_opening_um,
            )
        return (
            cfg.domain.notch_tip_x_um,
            cfg.domain.notch_center_y_um,
            cfg.domain.notch_depth_um,
            cfg.domain.notch_half_opening_um,
        )

    ptx, pcy, pdepth, phalf = _notch_meta(phy_cfg)
    mtx, mcy, mdepth, mhalf = _notch_meta(ml_cfg)
    # 4) 导出终态云图与网格图，便于人工核查空间分布差异。
    render_final_clouds(
        out_phy["out_dir"],
        state=phy_sim.state,
        sigma_xx=phy_sim.last_mech["sigma_xx"],
        sigma_yy=phy_sim.last_mech["sigma_yy"],
        sigma_xy=phy_sim.last_mech["sigma_xy"],
        time_s=phy_cfg.numerics.n_steps * phy_cfg.numerics.dt_s,
        extent_um=(0.0, phy_cfg.domain.lx_um, 0.0, phy_cfg.domain.ly_um),
        show_grid=False,
        x_coords_um=phy_sim.x_coords_um,
        y_coords_um=phy_sim.y_coords_um,
        save_svg=True,
        tip_zoom_center_um=(ptx, pcy),
        tip_zoom_half_width_um=0.5 * max(phy_cfg.domain.tip_zoom_width_um, 1e-6),
        tip_zoom_half_height_um=0.5 * max(phy_cfg.domain.tip_zoom_height_um, 1e-6),
    )
    render_final_clouds(
        out_ml["out_dir"],
        state=ml_sim.state,
        sigma_xx=ml_sim.last_mech["sigma_xx"],
        sigma_yy=ml_sim.last_mech["sigma_yy"],
        sigma_xy=ml_sim.last_mech["sigma_xy"],
        time_s=ml_cfg.numerics.n_steps * ml_cfg.numerics.dt_s,
        extent_um=(0.0, ml_cfg.domain.lx_um, 0.0, ml_cfg.domain.ly_um),
        show_grid=False,
        x_coords_um=ml_sim.x_coords_um,
        y_coords_um=ml_sim.y_coords_um,
        save_svg=True,
        tip_zoom_center_um=(mtx, mcy),
        tip_zoom_half_width_um=0.5 * max(ml_cfg.domain.tip_zoom_width_um, 1e-6),
        tip_zoom_half_height_um=0.5 * max(ml_cfg.domain.tip_zoom_height_um, 1e-6),
    )
    render_grid_figure(
        out_phy["out_dir"],
        x_vec_um=phy_sim.x_coords_um,
        y_vec_um=phy_sim.y_coords_um,
        time_s=phy_cfg.numerics.n_steps * phy_cfg.numerics.dt_s,
        notch_tip_x_um=ptx,
        notch_center_y_um=pcy,
        notch_depth_um=pdepth,
        notch_half_opening_um=phalf,
    )
    render_grid_figure(
        out_ml["out_dir"],
        x_vec_um=ml_sim.x_coords_um,
        y_vec_um=ml_sim.y_coords_um,
        time_s=ml_cfg.numerics.n_steps * ml_cfg.numerics.dt_s,
        notch_tip_x_um=mtx,
        notch_center_y_um=mcy,
        notch_depth_um=mdepth,
        notch_half_opening_um=mhalf,
    )

    report = build_report(
        args=args,
        phy_sim=phy_sim,
        ml_sim=ml_sim,
        phy_hist=phy_history,
        ml_hist=ml_history,
        phy_time_s=phy_time,
        ml_time_s=ml_time,
        loss_log=loss_log,
        out_phy=out_phy,
        out_ml=out_ml,
    )

    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    # 5) 输出结构化报告（JSON + Markdown）。
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(report_to_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "report_json": str(report_json.resolve()),
                "report_md": str(report_md.resolve()),
                "speedup_physics_over_ml": report["performance"]["speedup_physics_over_ml"],
                "final_solid_fraction_abs_diff": report["accuracy_vs_physics"]["final_solid_fraction_abs_diff"],
                "final_clouds_physics": str((out_phy["out_dir"] / "final_clouds" / "final_clouds_2x2.png").resolve()),
                "final_clouds_ml": str((out_ml["out_dir"] / "final_clouds" / "final_clouds_2x2.png").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
