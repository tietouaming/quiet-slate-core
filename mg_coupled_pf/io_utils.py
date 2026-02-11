"""输入输出与可视化模块（中文注释版）。

提供能力：
- 快照保存（NPZ）
- 历史曲线 CSV 导出
- 中间场图与终态云图渲染
- 网格可视化渲染
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator

from .operators import solid_indicator

VALUE_CMAP = LinearSegmentedColormap.from_list("blue_to_red_value", ["#1f4aff", "#ff2d2d"])


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则递归创建。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    """将 torch 张量安全转为 numpy 数组。"""
    return x.detach().cpu().numpy()


def save_snapshot_npz(out_dir: Path, step: int, state: Dict[str, torch.Tensor], extras: Dict[str, torch.Tensor]) -> Path:
    """保存单步快照为压缩 NPZ。"""
    # 统一把 state 与 extras 都写入同一个 NPZ，便于后处理一次读取。
    payload = {}
    for k, v in state.items():
        payload[k] = tensor_to_np(v)
    for k, v in extras.items():
        payload[k] = tensor_to_np(v)
    p = out_dir / f"snapshot_{step:06d}.npz"
    np.savez_compressed(p, **payload)
    return p


def _time_title(title: str, time_s: float | None, wall_time_s: float | None = None) -> str:
    """拼接图标题中的物理时间/运行时间信息。"""
    lines = [title]
    if time_s is not None:
        lines.append(f"t = {time_s:.6f} s")
    if wall_time_s is not None:
        lines.append(f"runtime = {wall_time_s:.3f} s")
    return "\n".join(lines)


def _apply_axes_style(
    ax: plt.Axes,
    extent_um: tuple[float, float, float, float] | None,
    show_grid: bool,
) -> None:
    """统一坐标轴样式（单位、刻度、网格）。"""
    if extent_um is not None:
        xmin, xmax, ymin, ymax = extent_um
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_xticks(np.linspace(xmin, xmax, 9))
        ax.set_yticks(np.linspace(ymin, ymax, 6))
    else:
        ax.set_xlabel("x (index)")
        ax.set_ylabel("y (index)")
    if show_grid:
        ax.grid(True, which="major", color="white", alpha=0.35, linewidth=0.6)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", color="white", alpha=0.2, linewidth=0.35)


def _cell_edges(v: np.ndarray) -> np.ndarray:
    """由网格节点坐标推导网格单元边界坐标。"""
    if v.size < 2:
        return np.array([v[0] - 0.5, v[0] + 0.5], dtype=np.float64)
    e = np.empty(v.size + 1, dtype=np.float64)
    e[1:-1] = 0.5 * (v[1:] + v[:-1])
    # 两端采用外推半步，保证边界单元完整显示。
    e[0] = v[0] - 0.5 * (v[1] - v[0])
    e[-1] = v[-1] + 0.5 * (v[-1] - v[-2])
    return e


def _plot_field(
    ax: plt.Axes,
    arr: torch.Tensor,
    *,
    cmap: str | LinearSegmentedColormap,
    extent_um: tuple[float, float, float, float] | None,
    x_coords_um: torch.Tensor | None,
    y_coords_um: torch.Tensor | None,
):
    """根据是否非均匀网格选择 pcolormesh 或 imshow 绘图。"""
    a = arr.detach().cpu().numpy()
    if x_coords_um is not None and y_coords_um is not None:
        # 非均匀网格：优先使用 pcolormesh，避免均匀采样误导。
        xv = x_coords_um.detach().cpu().numpy().astype(np.float64)
        yv = y_coords_um.detach().cpu().numpy().astype(np.float64)
        xe = _cell_edges(xv)
        ye = _cell_edges(yv)
        return ax.pcolormesh(xe, ye, a, shading="auto", cmap=cmap, rasterized=True)
    return ax.imshow(a, origin="lower", cmap=cmap, extent=extent_um, aspect="auto")


def render_fields(
    out_dir: Path,
    step: int,
    state: Dict[str, torch.Tensor],
    sigma_h: torch.Tensor,
    epspeq: torch.Tensor,
    *,
    time_s: float | None = None,
    wall_time_s: float | None = None,
    extent_um: tuple[float, float, float, float] | None = None,
    show_grid: bool = False,
    x_coords_um: torch.Tensor | None = None,
    y_coords_um: torch.Tensor | None = None,
) -> Path:
    """渲染中间步骤多场图。"""
    solid = solid_indicator(state["phi"], 0.5)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=120)
    data = [
        ("phi", state["phi"][0, 0]),
        ("c", state["c"][0, 0]),
        ("eta", state["eta"][0, 0]),
        ("sigma_h (MPa)", (sigma_h * solid)[0, 0]),
        ("epspeq", epspeq[0, 0]),
        ("notch view", 1.0 - state["phi"][0, 0]),
    ]
    for ax, (title, arr) in zip(axes.ravel(), data):
        # 每个子图统一共享色图与坐标样式，保证横向可比。
        im = _plot_field(
            ax,
            arr,
            cmap=VALUE_CMAP,
            extent_um=extent_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )
        ax.set_title(_time_title(title, time_s, wall_time_s))
        _apply_axes_style(ax, extent_um, show_grid)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    p = out_dir / f"fields_{step:06d}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def von_mises_2d(sigma_xx: torch.Tensor, sigma_yy: torch.Tensor, sigma_xy: torch.Tensor) -> torch.Tensor:
    """计算二维后处理所用的 von-Mises 等效应力。"""
    return torch.sqrt(torch.clamp(sigma_xx * sigma_xx - sigma_xx * sigma_yy + sigma_yy * sigma_yy + 3.0 * sigma_xy * sigma_xy, min=0.0))


def _save_single_cloud(
    arr: torch.Tensor,
    title: str,
    out_path: Path,
    *,
    cmap: str | LinearSegmentedColormap = VALUE_CMAP,
    time_s: float | None = None,
    wall_time_s: float | None = None,
    extent_um: tuple[float, float, float, float] | None = None,
    show_grid: bool = True,
    x_coords_um: torch.Tensor | None = None,
    y_coords_um: torch.Tensor | None = None,
    save_svg: bool = True,
    xlim_um: tuple[float, float] | None = None,
    ylim_um: tuple[float, float] | None = None,
) -> None:
    """保存单张云图（支持局部放大窗口）。"""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    im = _plot_field(
        ax,
        arr,
        cmap=cmap,
        extent_um=extent_um,
        x_coords_um=x_coords_um,
        y_coords_um=y_coords_um,
    )
    ax.set_title(_time_title(title, time_s, wall_time_s))
    _apply_axes_style(ax, extent_um, show_grid)
    if xlim_um is not None:
        # 局部放大窗口用于裂纹尖端细节观察。
        ax.set_xlim(xlim_um[0], xlim_um[1])
    if ylim_um is not None:
        ax.set_ylim(ylim_um[0], ylim_um[1])
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    if save_svg:
        fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def render_final_clouds(
    out_dir: Path,
    state: Dict[str, torch.Tensor],
    sigma_xx: torch.Tensor,
    sigma_yy: torch.Tensor,
    sigma_xy: torch.Tensor,
    *,
    time_s: float | None = None,
    wall_time_s: float | None = None,
    extent_um: tuple[float, float, float, float] | None = None,
    show_grid: bool = False,
    x_coords_um: torch.Tensor | None = None,
    y_coords_um: torch.Tensor | None = None,
    save_svg: bool = True,
    tip_zoom_center_um: tuple[float, float] | None = None,
    tip_zoom_half_width_um: float = 30.0,
    tip_zoom_half_height_um: float = 20.0,
) -> Dict[str, Path]:
    """渲染并保存最终云图及可选尖端局部放大图。"""
    cloud_dir = ensure_dir(out_dir / "final_clouds")
    vm = von_mises_2d(sigma_xx, sigma_yy, sigma_xy)

    phi = state["phi"][0, 0]
    solid = solid_indicator(state["phi"], 0.5)[0, 0]
    cmg = state["c"][0, 0]
    # eta 和应力只在固相内有物理意义，图上显式掩蔽液相。
    eta = state["eta"][0, 0] * solid
    vm2 = vm[0, 0] * solid

    p_phi = cloud_dir / "phi_cloud.png"
    p_cmg = cloud_dir / "cMg_cloud.png"
    p_eta = cloud_dir / "eta_cloud.png"
    p_yeta = cloud_dir / "yeta_cloud.png"
    p_vm = cloud_dir / "von_mises_cloud.png"

    _save_single_cloud(
        phi,
        "phi",
        p_phi,
        time_s=time_s,
        wall_time_s=wall_time_s,
        extent_um=extent_um,
        show_grid=show_grid,
        x_coords_um=x_coords_um,
        y_coords_um=y_coords_um,
        save_svg=save_svg,
    )
    _save_single_cloud(
        cmg,
        "cMg",
        p_cmg,
        time_s=time_s,
        wall_time_s=wall_time_s,
        extent_um=extent_um,
        show_grid=show_grid,
        x_coords_um=x_coords_um,
        y_coords_um=y_coords_um,
        save_svg=False,
    )
    _save_single_cloud(
        eta,
        "eta",
        p_eta,
        time_s=time_s,
        wall_time_s=wall_time_s,
        extent_um=extent_um,
        show_grid=show_grid,
        x_coords_um=x_coords_um,
        y_coords_um=y_coords_um,
        save_svg=False,
    )
    _save_single_cloud(
        eta,
        "yeta",
        p_yeta,
        time_s=time_s,
        wall_time_s=wall_time_s,
        extent_um=extent_um,
        show_grid=show_grid,
        x_coords_um=x_coords_um,
        y_coords_um=y_coords_um,
        save_svg=False,
    )
    _save_single_cloud(
        vm2,
        "von-Mises Stress (MPa)",
        p_vm,
        time_s=time_s,
        wall_time_s=wall_time_s,
        extent_um=extent_um,
        show_grid=show_grid,
        x_coords_um=x_coords_um,
        y_coords_um=y_coords_um,
        save_svg=False,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=140)
    items = [
        ("von-Mises Stress (MPa)", vm2),
        ("eta", eta),
        ("cMg", cmg),
        ("phi", phi),
    ]
    for ax, (title, arr) in zip(axes.ravel(), items):
        im = _plot_field(
            ax,
            arr,
            cmap=VALUE_CMAP,
            extent_um=extent_um,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
        )
        ax.set_title(_time_title(title, time_s, wall_time_s))
        _apply_axes_style(ax, extent_um, show_grid)
        fig.colorbar(im, ax=ax, shrink=0.8)
    supt = []
    if time_s is not None:
        supt.append(f"t = {time_s:.6f} s")
    if wall_time_s is not None:
        supt.append(f"runtime = {wall_time_s:.3f} s")
    if supt:
        fig.suptitle(f"Final Clouds ({', '.join(supt)})")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    else:
        fig.tight_layout()
    p_grid = cloud_dir / "final_clouds_2x2.png"
    fig.savefig(p_grid)
    # 仅保留 phi 的 SVG，避免超大网格下矢量文件过重。
    plt.close(fig)

    tip_zoom_dir = ensure_dir(out_dir / "final_clouds_tip_zoom")
    if tip_zoom_center_um is not None:
        # 若给定尖端中心，则额外输出固定尺寸的局部放大图。
        cx, cy = float(tip_zoom_center_um[0]), float(tip_zoom_center_um[1])
        xlim = (cx - float(tip_zoom_half_width_um), cx + float(tip_zoom_half_width_um))
        ylim = (cy - float(tip_zoom_half_height_um), cy + float(tip_zoom_half_height_um))
        _save_single_cloud(
            phi,
            "phi (tip zoom)",
            tip_zoom_dir / "phi_tip_zoom.png",
            time_s=time_s,
            wall_time_s=wall_time_s,
            extent_um=extent_um,
            show_grid=show_grid,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
            save_svg=save_svg,
            xlim_um=xlim,
            ylim_um=ylim,
        )
        _save_single_cloud(
            cmg,
            "cMg (tip zoom)",
            tip_zoom_dir / "cMg_tip_zoom.png",
            time_s=time_s,
            wall_time_s=wall_time_s,
            extent_um=extent_um,
            show_grid=show_grid,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
            save_svg=False,
            xlim_um=xlim,
            ylim_um=ylim,
        )
        _save_single_cloud(
            eta,
            "yeta (tip zoom)",
            tip_zoom_dir / "yeta_tip_zoom.png",
            time_s=time_s,
            wall_time_s=wall_time_s,
            extent_um=extent_um,
            show_grid=show_grid,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
            save_svg=False,
            xlim_um=xlim,
            ylim_um=ylim,
        )
        _save_single_cloud(
            vm2,
            "von-Mises (tip zoom)",
            tip_zoom_dir / "von_mises_tip_zoom.png",
            time_s=time_s,
            wall_time_s=wall_time_s,
            extent_um=extent_um,
            show_grid=show_grid,
            x_coords_um=x_coords_um,
            y_coords_um=y_coords_um,
            save_svg=False,
            xlim_um=xlim,
            ylim_um=ylim,
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=180)
        for ax, (title, arr) in zip(axes.ravel(), items):
            im = _plot_field(
                ax,
                arr,
                cmap=VALUE_CMAP,
                extent_um=extent_um,
                x_coords_um=x_coords_um,
                y_coords_um=y_coords_um,
            )
            ax.set_title(_time_title(f"{title} (tip zoom)", time_s, wall_time_s))
            _apply_axes_style(ax, extent_um, show_grid)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        p_tip = tip_zoom_dir / "tip_zoom_2x2.png"
        fig.savefig(p_tip)
        # 局部图同样只保留必要 SVG，控制输出体积。
        plt.close(fig)

    return {
        "phi_cloud": p_phi,
        "cMg_cloud": p_cmg,
        "eta_cloud": p_eta,
        "yeta_cloud": p_yeta,
        "von_mises_cloud": p_vm,
        "grid_2x2": p_grid,
        "tip_zoom_dir": tip_zoom_dir,
    }


def render_grid_figure(
    out_dir: Path,
    x_vec_um: torch.Tensor,
    y_vec_um: torch.Tensor,
    *,
    time_s: float | None = None,
    wall_time_s: float | None = None,
    notch_tip_x_um: float | None = None,
    notch_center_y_um: float | None = None,
    notch_depth_um: float | None = None,
    notch_half_opening_um: float | None = None,
) -> Dict[str, Path]:
    """输出全域网格图与缺口附近局部网格图。"""
    grid_dir = ensure_dir(out_dir / "grid")
    x = x_vec_um.detach().cpu().numpy().astype(np.float64)
    y = y_vec_um.detach().cpu().numpy().astype(np.float64)
    if x.size < 2 or y.size < 2:
        raise ValueError("Grid vectors are too short for plotting.")

    xmin, xmax = float(x[0]), float(x[-1])
    ymin, ymax = float(y[0]), float(y[-1])
    ttl = "Computational Mesh"
    tt = []
    if time_s is not None:
        tt.append(f"t = {time_s:.6f} s")
    if wall_time_s is not None:
        tt.append(f"runtime = {wall_time_s:.3f} s")
    if tt:
        ttl = f"{ttl} ({', '.join(tt)})"

    def _draw_mesh(ax: plt.Axes) -> None:
        # 逐条绘制网格线，便于观察局部加密效果。
        for xv in x:
            ax.plot([xv, xv], [ymin, ymax], color="black", linewidth=0.18, alpha=0.22)
        for yv in y:
            ax.plot([xmin, xmax], [yv, yv], color="black", linewidth=0.18, alpha=0.22)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(ttl)

    p_full = grid_dir / "mesh_full.png"
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    _draw_mesh(ax)
    fig.tight_layout()
    fig.savefig(p_full)
    plt.close(fig)

    out = {"mesh_full": p_full}
    if (
        notch_tip_x_um is not None
        and notch_center_y_um is not None
        and notch_depth_um is not None
        and notch_half_opening_um is not None
    ):
        xw = max(1.2 * float(notch_depth_um), 10.0)
        yw = max(2.2 * float(notch_half_opening_um), 10.0)
        p_zoom = grid_dir / "mesh_notch_zoom.png"
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        _draw_mesh(ax)
        ax.set_xlim(float(notch_tip_x_um) - xw, float(notch_tip_x_um) + xw)
        ax.set_ylim(float(notch_center_y_um) - yw, float(notch_center_y_um) + yw)
        fig.tight_layout()
        fig.savefig(p_zoom)
        plt.close(fig)
        out["mesh_notch_zoom"] = p_zoom
    return out


def save_history_csv(out_dir: Path, history: List[Dict[str, float]]) -> Path:
    """将逐步历史量写入 CSV。"""
    p = out_dir / "history.csv"
    if not history:
        p.write_text("step,time_s,solid_fraction,avg_eta,max_sigma_h,avg_epspeq\n", encoding="utf-8")
        return p
    keys = list(history[0].keys())
    lines = [",".join(keys)]
    for row in history:
        lines.append(",".join(str(row[k]) for k in keys))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
