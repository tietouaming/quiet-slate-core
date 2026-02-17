"""训练高级跨尺度模型（U-AFNO / 多保真）。

功能：
1. 训练高级模型并保存 checkpoint；
2. 输出训练历史曲线、parity 图、不确定性校准图；
3. 生成 JSON + Markdown 报告（含物理一致性指标 PCI）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from mg_coupled_pf.multiscale import (
    AdvancedModelConfig,
    AdvancedTrainingConfig,
    PhysicsMetricConfig,
    evaluate_physics_metrics,
    load_multiscale_dataset_npz,
    plot_parity_by_target,
    plot_training_history,
    plot_uncertainty_calibration,
    save_advanced_model,
    train_advanced_multiscale_model,
)
from mg_coupled_pf.multiscale.advanced_models import unscale_advanced_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train advanced multi-fidelity macro surrogate.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output-model", type=str, required=True)
    p.add_argument("--output-report-json", type=str, default="")
    p.add_argument("--output-report-md", type=str, default="")
    p.add_argument("--fig-dir", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--model-name", type=str, default="multifidelity_uafno", choices=["multifidelity_uafno", "uafno"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--afno-blocks", type=int, default=4)
    p.add_argument("--modes-x", type=int, default=16)
    p.add_argument("--modes-y", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--high-loss-weight", type=float, default=1.0)
    p.add_argument("--low-loss-weight", type=float, default=0.6)
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    s = str(name).strip().lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _predict_all(
    model: torch.nn.Module,
    ds: Dict[str, np.ndarray],
    scalers: Dict[str, Dict[str, np.ndarray]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    x_field = torch.from_numpy(ds["x_field"].astype(np.float32)).to(device=device)
    x_desc = torch.from_numpy(ds["x_desc"].astype(np.float32)).to(device=device)
    if "x_desc" in scalers and "mean" in scalers["x_desc"]:
        mu = torch.from_numpy(np.asarray(scalers["x_desc"]["mean"], dtype=np.float32)).to(device=device)
        sd = torch.from_numpy(np.asarray(scalers["x_desc"]["std"], dtype=np.float32)).to(device=device)
        x_desc = (x_desc - mu) / sd
    with torch.no_grad():
        out = model(x_field, x_desc)
    if "high_mean" in out:
        y_mean_s = out["high_mean"].detach().cpu().numpy()
        y_logv_s = out["high_logvar"].detach().cpu().numpy()
    else:
        y_mean_s = out["mean"].detach().cpu().numpy()
        y_logv_s = out["logvar"].detach().cpu().numpy()
    y_mean = unscale_advanced_targets(y_mean_s, scalers, key="y_high")
    std_high = np.asarray(scalers["y_high"]["std"], dtype=np.float32).reshape(1, -1)
    y_var = np.exp(y_logv_s) * (std_high ** 2)
    y_logv = np.log(np.maximum(y_var, 1e-18))
    y_true = ds["y"].astype(np.float32)
    return y_true, y_mean, y_logv


def _write_md_report(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = payload
    phy = p.get("physics_metrics", {})
    lines = [
        "# 高级跨尺度模型训练报告",
        "",
        "## 模型与数据",
        f"- 模型: `{p.get('model_name', '')}`",
        f"- 样本数: {p.get('n_samples', 0)}",
        f"- 高保真标签占比: {p.get('high_label_ratio', 0.0):.4f}",
        f"- 设备: `{p.get('device', '')}`",
        "",
        "## 核心精度",
        f"- MAE: {p.get('mae', 0.0):.6f}",
        f"- RMSE: {p.get('rmse', 0.0):.6f}",
        f"- R2: {p.get('r2', 0.0):.6f}",
        "",
        "## 物理一致性指标（无高精度宏观全量数据场景）",
        f"- PCI（Physics Consistency Index）: {float(phy.get('physics_consistency_index', 0.0)):.6f}",
        f"- 越界率: {float(phy.get('overall_violation_rate', 0.0)):.6f}",
        f"- 时序单调分数: {float(phy.get('overall_monotonic_score', 0.0)):.6f}",
        f"- 2σ 覆盖率: {float(phy.get('unc_coverage', 0.0)):.6f}",
        "",
        "## 物理含义说明",
        "- PCI 越接近 1，表示模型越符合“物理边界 + 时序趋势 + 不确定性校准”三类约束。",
        "- 越界率衡量预测是否违反目标物理范围（如腐蚀损失、孪晶体积分数应在 [0,1]）。",
        "- 时序单调分数衡量同一算例内随时间的演化趋势是否保持物理方向。",
        "- 2σ 覆盖率衡量模型给出的不确定性区间是否与真实误差匹配。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ds = load_multiscale_dataset_npz(args.dataset)
    device = pick_device(args.device)

    model_cfg = AdvancedModelConfig(
        in_channels=int(ds["x_field"].shape[1]),
        desc_dim=int(ds["x_desc"].shape[1]),
        target_dim=int(ds["y"].shape[1]),
        width=int(args.width),
        depth=int(args.depth),
        afno_blocks=int(args.afno_blocks),
        modes_x=int(args.modes_x),
        modes_y=int(args.modes_y),
        dropout=float(args.dropout),
        use_uncertainty_head=True,
    )
    train_cfg = AdvancedTrainingConfig(
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        train_split=float(args.train_split),
        use_amp=bool(args.use_amp),
        model_name=str(args.model_name),
        high_loss_weight=float(args.high_loss_weight),
        low_loss_weight=float(args.low_loss_weight),
    )
    physics_cfg = PhysicsMetricConfig()
    model, rep = train_advanced_multiscale_model(ds, model_cfg, train_cfg, device=device, physics_cfg=physics_cfg)

    out_model = Path(args.output_model)
    save_advanced_model(
        out_model,
        model,
        model_cfg,
        model_name=str(args.model_name),
        field_channels=[str(v) for v in ds["field_channels"].tolist()],
        descriptor_names=[str(v) for v in ds["descriptor_names"].tolist()],
        target_names=[str(v) for v in ds["target_names"].tolist()],
        scalers=rep["scalers"],
        extra_meta={"training_config": vars(args)},
    )

    y_true, y_mean, y_logv = _predict_all(model, ds, rep["scalers"], device=device)
    reg = {
        "mae": float(np.mean(np.abs(y_mean - y_true))),
        "rmse": float(np.sqrt(np.mean((y_mean - y_true) ** 2))),
        "r2": float(1.0 - np.mean((y_mean - y_true) ** 2) / max(np.var(y_true), 1e-12)),
    }
    phy = evaluate_physics_metrics(
        y_true=y_true,
        y_mean=y_mean,
        y_logvar=y_logv,
        target_names=[str(v) for v in ds["target_names"].tolist()],
        case_ids=[str(v) for v in ds["case_ids"].tolist()],
        t_inputs=ds["t_input"].astype(np.float64),
        cfg=physics_cfg,
    )

    fig_dir = Path(args.fig_dir) if args.fig_dir else out_model.parent / "figures_advanced"
    fig_dir.mkdir(parents=True, exist_ok=True)
    p_hist = plot_training_history(rep["history"], fig_dir / "training_history.png")
    parity = plot_parity_by_target(y_true, y_mean, [str(v) for v in ds["target_names"].tolist()], fig_dir, prefix="parity")
    p_unc = plot_uncertainty_calibration(
        y_true,
        y_mean,
        y_logv,
        [str(v) for v in ds["target_names"].tolist()],
        fig_dir / "uncertainty_calibration.png",
    )

    payload: Dict[str, object] = {
        "model": str(out_model),
        "model_name": str(args.model_name),
        "device": str(device),
        "n_samples": int(ds["x_field"].shape[0]),
        "high_label_ratio": float(np.mean(ds["high_mask"])) if "high_mask" in ds else 1.0,
        "best_val_loss": float(rep["best_val_loss"]),
        "mae": reg["mae"],
        "rmse": reg["rmse"],
        "r2": reg["r2"],
        "physics_metrics": phy,
        "figures": {
            "history": str(p_hist),
            "parity": [str(p) for p in parity],
            "uncertainty": str(p_unc),
        },
        "training_report": rep,
    }

    out_json = Path(args.output_report_json) if args.output_report_json else out_model.with_suffix(".advanced_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")

    out_md = Path(args.output_report_md) if args.output_report_md else out_model.with_suffix(".advanced_report.md")
    _write_md_report(out_md, payload)

    print(
        {
            "model": str(out_model),
            "report_json": str(out_json),
            "report_md": str(out_md),
            "pci": float(phy.get("physics_consistency_index", 0.0)),
            "rmse": float(reg["rmse"]),
            "mae": float(reg["mae"]),
        }
    )


if __name__ == "__main__":
    main()
