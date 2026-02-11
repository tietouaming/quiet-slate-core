#!/usr/bin/env python
"""surrogate 训练脚本（中文注释版）。

流程：
1. 从仿真快照构建监督学习样本（x_t -> Δx）；
2. 训练代理网络并做验证；
3. 输出最佳权重到 artifacts/ml。
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf import load_config
from mg_coupled_pf.ml.surrogate import FIELD_ORDER, arch_requires_full_precision, build_surrogate, save_surrogate


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""
    p = argparse.ArgumentParser(description="Train ML surrogate for coupled phase-field solver.")
    p.add_argument("--config", default="configs/notch_case.yaml")
    p.add_argument("--snapshots-dir", default="")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--disable-torch-compile", action="store_true")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--model-arch", choices=["tiny_unet", "dw_unet", "fno2d", "afno2d"], default="")
    p.add_argument("--model-hidden", type=int, default=0, help="Hidden width for tiny_unet.")
    p.add_argument("--dw-hidden", type=int, default=0, help="Hidden width for dw_unet.")
    p.add_argument("--dw-depth", type=int, default=0)
    p.add_argument("--fno-width", type=int, default=0)
    p.add_argument("--fno-modes-x", type=int, default=0)
    p.add_argument("--fno-modes-y", type=int, default=0)
    p.add_argument("--fno-depth", type=int, default=0)
    p.add_argument("--afno-width", type=int, default=0)
    p.add_argument("--afno-modes-x", type=int, default=0)
    p.add_argument("--afno-modes-y", type=int, default=0)
    p.add_argument("--afno-depth", type=int, default=0)
    p.add_argument("--afno-expansion", type=float, default=0.0)
    p.add_argument("--streaming", action="store_true", help="Load snapshot pairs on-the-fly instead of all-in-memory.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=0, help="Optional cap of training pairs for quick experiments.")
    p.add_argument("--rollout-steps", type=int, default=0, help="Multi-step rollout steps for training (>=2 enables rollout loss).")
    p.add_argument("--rollout-weight", type=float, default=0.0, help="Weight for rollout loss.")
    p.add_argument("--loss-data-weight", type=float, default=1.0)
    p.add_argument("--loss-bounds-weight", type=float, default=0.02)
    p.add_argument("--loss-mass-weight", type=float, default=0.02)
    p.add_argument("--loss-mech-weight", type=float, default=0.01)
    p.add_argument("--loss-pde-weight", type=float, default=0.05, help="Weight for phi/c/eta discrete PDE residual loss.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clip; <=0 disables clip.")
    return p.parse_args()


def _load_snapshot_tensor(path: Path) -> torch.Tensor:
    """加载单个快照并拼接为模型通道顺序张量。"""
    # 约定：每个快照里每个字段都是 [1,1,H,W]。
    data = np.load(path)
    ref = None
    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            ref = arr
            break
    if ref is None:
        raise RuntimeError(f"snapshot {path} does not contain valid [1,1,H,W] fields.")
    chans = []
    for k in FIELD_ORDER:
        if k in data.files:
            chans.append(data[k])
        else:
            # 兼容旧快照：缺失新通道时以零场补齐。
            chans.append(np.zeros_like(ref))
    arr = np.concatenate(chans, axis=1)  # [1, C, H, W]
    return torch.from_numpy(arr.astype(np.float32))


def _snapshot_step(path: Path) -> int:
    """从 `snapshot_XXXXXX` 文件名中解析时间步编号。"""
    stem = path.stem  # snapshot_000100
    return int(stem.split("_")[-1])


def load_pairs(paths: List[Path]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将快照序列转换为监督学习样本对。"""
    xs: List[torch.Tensor] = []
    dys: List[torch.Tensor] = []
    dy2s: List[torch.Tensor] = []
    m2s: List[torch.Tensor] = []
    for i, (a, b) in enumerate(zip(paths[:-1], paths[1:])):
        xa = _load_snapshot_tensor(a)
        xb = _load_snapshot_tensor(b)
        da = _snapshot_step(a)
        db = _snapshot_step(b)
        # 若快照并非每步保存，按步距做归一化，使学习目标统一为“每步增量”。
        gap = max(1, db - da)
        xs.append(xa)
        dys.append((xb - xa) / float(gap))
        if i + 2 < len(paths):
            c = paths[i + 2]
            xc = _load_snapshot_tensor(c)
            dc = _snapshot_step(c)
            gap2 = max(1, dc - db)
            dy2s.append((xc - xb) / float(gap2))
            m2s.append(torch.ones((1, 1, 1, 1), dtype=torch.float32))
        else:
            dy2s.append(torch.zeros_like(xa))
            m2s.append(torch.zeros((1, 1, 1, 1), dtype=torch.float32))
    x = torch.cat(xs, dim=0)
    dy = torch.cat(dys, dim=0)
    dy2 = torch.cat(dy2s, dim=0)
    m2 = torch.cat(m2s, dim=0)
    return x, dy, dy2, m2


class SnapshotPairDataset(Dataset):
    """流式读取快照对的数据集，适合大数据量训练。"""
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return max(0, len(self.paths) - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 流式模式下逐对读取，减少内存峰值占用。
        a = self.paths[idx]
        b = self.paths[idx + 1]
        xa = _load_snapshot_tensor(a).squeeze(0)
        xb = _load_snapshot_tensor(b).squeeze(0)
        da = _snapshot_step(a)
        db = _snapshot_step(b)
        gap = max(1, db - da)
        dy = (xb - xa) / float(gap)
        if idx + 2 < len(self.paths):
            c = self.paths[idx + 2]
            xc = _load_snapshot_tensor(c).squeeze(0)
            dc = _snapshot_step(c)
            gap2 = max(1, dc - db)
            dy2 = (xc - xb) / float(gap2)
            m2 = torch.ones((1, 1, 1), dtype=torch.float32)
        else:
            dy2 = torch.zeros_like(dy)
            m2 = torch.zeros((1, 1, 1), dtype=torch.float32)
        return xa, dy, dy2, m2


def _smooth_heaviside(phi: torch.Tensor) -> torch.Tensor:
    """与主求解器一致的平滑 Heaviside。"""
    p = torch.clamp(phi, 0.0, 1.0)
    return p * p * p * (10.0 - 15.0 * p + 6.0 * p * p)


def _grad_xy_center(x: torch.Tensor, dx: float, dy: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """中心差分梯度（训练阶段物理残差近似）。"""
    # 与主求解器默认 Neumann 边界一致：reflect 对中心差分对应零法向梯度。
    if x.shape[-2] < 2 or x.shape[-1] < 2:
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    else:
        xp = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = (xp[:, :, 1:-1, 2:] - xp[:, :, 1:-1, :-2]) / (2.0 * dx)
    gy = (xp[:, :, 2:, 1:-1] - xp[:, :, :-2, 1:-1]) / (2.0 * dy)
    return gx, gy


def _div_center(gx: torch.Tensor, gy: torch.Tensor, dx: float, dy: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """中心差分散度（分别返回 x/y 分量散度对应项）。"""
    if gx.shape[-2] < 2 or gx.shape[-1] < 2:
        gxp = F.pad(gx, (1, 1, 1, 1), mode="replicate")
    else:
        gxp = F.pad(gx, (1, 1, 1, 1), mode="reflect")
    if gy.shape[-2] < 2 or gy.shape[-1] < 2:
        gyp = F.pad(gy, (1, 1, 1, 1), mode="replicate")
    else:
        gyp = F.pad(gy, (1, 1, 1, 1), mode="reflect")
    d_gx_dx = (gxp[:, :, 1:-1, 2:] - gxp[:, :, 1:-1, :-2]) / (2.0 * dx)
    d_gy_dy = (gyp[:, :, 2:, 1:-1] - gyp[:, :, :-2, 1:-1]) / (2.0 * dy)
    return d_gx_dx, d_gy_dy


def _physics_losses(
    x: torch.Tensor,
    pred_d: torch.Tensor,
    target_d: torch.Tensor,
    cfg,
    *,
    w_data: float,
    w_bounds: float,
    w_mass: float,
    w_mech: float,
    w_pde: float,
    rollout_pred_d: torch.Tensor | None = None,
    rollout_target_d: torch.Tensor | None = None,
    rollout_mask: torch.Tensor | None = None,
    rollout_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """构造监督+物理约束的组合损失。"""
    idx = {k: i for i, k in enumerate(FIELD_ORDER)}
    data_loss = F.mse_loss(pred_d, target_d)
    x_pred = x + pred_d
    x_true = x + target_d

    phi = x_pred[:, idx["phi"] : idx["phi"] + 1].float()
    c = x_pred[:, idx["c"] : idx["c"] + 1].float()
    eta = x_pred[:, idx["eta"] : idx["eta"] + 1].float()
    ux = x_pred[:, idx["ux"] : idx["ux"] + 1].float()
    uy = x_pred[:, idx["uy"] : idx["uy"] + 1].float()
    epspeq = x_pred[:, idx["epspeq"] : idx["epspeq"] + 1].float()
    epsp_xx = x_pred[:, idx["epsp_xx"] : idx["epsp_xx"] + 1].float()
    epsp_yy = x_pred[:, idx["epsp_yy"] : idx["epsp_yy"] + 1].float()
    epsp_xy = x_pred[:, idx["epsp_xy"] : idx["epsp_xy"] + 1].float()

    # 1) 有界性与“液相无孪晶/无塑性”约束（可微软化并参与反传）。
    b_phi = F.relu(-phi) + F.relu(phi - 1.0)
    b_c = F.relu(-c) + F.relu(c - 1.0)
    b_eta = F.relu(-eta) + F.relu(eta - 1.0)
    b_eps = F.relu(-epspeq)
    b_epsp_xx = F.relu(torch.abs(epsp_xx) - 1.0)
    b_epsp_yy = F.relu(torch.abs(epsp_yy) - 1.0)
    b_epsp_xy = F.relu(torch.abs(epsp_xy) - 1.0)
    liquid_soft = torch.sigmoid((0.5 - torch.clamp(phi, 0.0, 1.0)) * 24.0)
    liquid_penalty = (
        eta * liquid_soft
        + epspeq * liquid_soft
        + epsp_xx * liquid_soft
        + epsp_yy * liquid_soft
        + epsp_xy * liquid_soft
    )
    bounds_loss = torch.mean(
        b_phi * b_phi
        + b_c * b_c
        + b_eta * b_eta
        + b_eps * b_eps
        + b_epsp_xx * b_epsp_xx
        + b_epsp_yy * b_epsp_yy
        + b_epsp_xy * b_epsp_xy
        + liquid_penalty * liquid_penalty
    )

    # 2) 浓度守恒近似：预测下一步平均浓度应接近监督下一步平均浓度。
    phi_true = x_true[:, idx["phi"] : idx["phi"] + 1].float()
    c_true = x_true[:, idx["c"] : idx["c"] + 1].float()
    hphi = _smooth_heaviside(phi)
    hphi_true = _smooth_heaviside(phi_true)
    # 守恒量按固相插值场 h(phi)*c 统计，避免把液相本该无意义区域等权计入。
    mass_pred = torch.mean(hphi * c)
    mass_true = torch.mean(hphi_true * c_true)
    mass_loss = (mass_pred - mass_true) ** 2

    # 3) 力学平衡残差近似：对预测位移场计算 div(sigma)。
    dx_um = float(cfg.domain.lx_um) / max(int(cfg.domain.nx) - 1, 1)
    dy_um = float(cfg.domain.ly_um) / max(int(cfg.domain.ny) - 1, 1)
    dux_dx, dux_dy = _grad_xy_center(ux, dx_um, dy_um)
    duy_dx, duy_dy = _grad_xy_center(uy, dx_um, dy_um)
    # 与主求解器一致：扣除孪晶本征应变与塑性张量分量。
    ang_s = math.radians(float(cfg.twinning.twin_shear_dir_angle_deg))
    ang_n = math.radians(float(cfg.twinning.twin_plane_normal_angle_deg))
    sx, sy = math.cos(ang_s), math.sin(ang_s)
    nx, ny = math.cos(ang_n), math.sin(ang_n)
    h_eta = _smooth_heaviside(eta)
    gamma_tw = float(cfg.twinning.gamma_twin)
    eps_tw_xx = h_eta * gamma_tw * sx * nx
    eps_tw_yy = h_eta * gamma_tw * sy * ny
    eps_tw_xy = h_eta * 0.5 * gamma_tw * (sx * ny + sy * nx)
    loading_mode = str(getattr(cfg.mechanics, "loading_mode", "eigenstrain")).strip().lower()
    ext_x = float(cfg.mechanics.external_strain_x) if loading_mode not in {"dirichlet_x", "dirichlet", "ux_dirichlet"} else 0.0
    ex = dux_dx + ext_x - epsp_xx - eps_tw_xx
    ey = duy_dy - epsp_yy - eps_tw_yy
    exy = 0.5 * (dux_dy + duy_dx) - epsp_xy - eps_tw_xy
    lam = float(cfg.mechanics.lambda_GPa) * 1e3
    mu = float(cfg.mechanics.mu_GPa) * 1e3
    tr = ex + ey
    hphi_mech = hphi
    if bool(getattr(cfg.mechanics, "strict_solid_stress_only", False)):
        hphi_mech = hphi_mech * (phi >= float(cfg.domain.solid_phase_threshold)).to(dtype=phi.dtype)
    sig_xx = hphi_mech * (lam * tr + 2.0 * mu * ex)
    sig_yy = hphi_mech * (lam * tr + 2.0 * mu * ey)
    sig_xy = hphi_mech * (2.0 * mu * exy)
    dsxx_dx, _ = _div_center(sig_xx, sig_xx, dx_um, dy_um)
    _, dsxy_dy = _div_center(sig_xy, sig_xy, dx_um, dy_um)
    dsxy_dx, _ = _div_center(sig_xy, sig_xy, dx_um, dy_um)
    _, dsyy_dy = _div_center(sig_yy, sig_yy, dx_um, dy_um)
    res_x = dsxx_dx + dsxy_dy
    res_y = dsxy_dx + dsyy_dy
    mech_norm = max((lam + 2.0 * mu) ** 2, 1e-9)
    mech_loss = torch.mean((res_x * res_x + res_y * res_y) / mech_norm)

    # 4) 离散 PDE 残差：约束 surrogate 学到“物理可行”的时间推进。
    dt = max(float(cfg.numerics.dt_s), 1e-12)
    dx_m = dx_um * 1e-6
    dy_m = dy_um * 1e-6
    p = torch.clamp(phi, 0.0, 1.0)
    hphi_p = _smooth_heaviside(p)
    hphi_d = 30.0 * p * p * (p - 1.0) * (p - 1.0)
    c_l = float(cfg.corrosion.c_l_eq_norm)
    c_s = float(cfg.corrosion.c_s_eq_norm)
    delta_eq = c_s - c_l
    gamma = float(cfg.corrosion.gamma_J_m2)
    ell_m = max(float(cfg.corrosion.interface_thickness_um) * 1e-6, 1e-12)
    omega = 3.0 * gamma / (4.0 * ell_m) / 1e6
    kappa_phi = 1.5 * gamma * ell_m / 1e6
    phi_dw_prime = 32.0 * p * (1.0 - p) * (1.0 - 2.0 * p)
    chem_drive = -float(cfg.corrosion.A_J_m3) / 1e6 * (c - hphi_p * delta_eq - c_l) * delta_eq * hphi_d
    phi_nonlin = chem_drive + omega * phi_dw_prime
    L0 = float(cfg.corrosion.L0)
    gx_phi_m, gy_phi_m = _grad_xy_center(phi, dx_m, dy_m)
    dflux_phi_x_dx, dflux_phi_y_dy = _div_center(kappa_phi * L0 * gx_phi_m, kappa_phi * L0 * gy_phi_m, dx_m, dy_m)
    pde_phi = (pred_d[:, idx["phi"] : idx["phi"] + 1].float() / dt) + L0 * phi_nonlin - (dflux_phi_x_dx + dflux_phi_y_dy)

    D_l = float(cfg.corrosion.D_l_m2_s)
    D_s = float(cfg.corrosion.D_s_m2_s)
    D = D_s * hphi_p + (1.0 - hphi_p) * D_l
    gx_c_m, gy_c_m = _grad_xy_center(c, dx_m, dy_m)
    corr = hphi_d * (c_l - c_s)
    jx = -D * (gx_c_m + corr * gx_phi_m)
    jy = -D * (gy_c_m + corr * gy_phi_m)
    djx_dx, djy_dy = _div_center(jx, jy, dx_m, dy_m)
    rhs_c = -(djx_dx + djy_dy)
    pde_c = (pred_d[:, idx["c"] : idx["c"] + 1].float() / dt) - rhs_c

    eta_p = torch.clamp(eta, 0.0, 1.0)
    heta = _smooth_heaviside(eta_p)
    heta_d = 30.0 * eta_p * eta_p * (eta_p - 1.0) * (eta_p - 1.0)
    k_eta = float(cfg.twinning.kappa_eta)
    grad_coef_eta = k_eta * hphi_p if bool(cfg.twinning.scale_twin_gradient_by_hphi) else torch.full_like(hphi_p, k_eta)
    twin_dw = hphi_p * 2.0 * float(cfg.twinning.W_barrier_MPa) * eta_p * (1.0 - eta_p) * (1.0 - 2.0 * eta_p)
    tau_tw = sx * (sig_xx * nx + sig_xy * ny) + sy * (sig_xy * nx + sig_yy * ny)
    tw_drive = hphi_p * tau_tw * float(cfg.twinning.gamma_twin) * heta_d
    gx_eta_um, gy_eta_um = _grad_xy_center(eta_p, dx_um, dy_um)
    dgx_eta_dx, dgy_eta_dy = _div_center(grad_coef_eta * gx_eta_um, grad_coef_eta * gy_eta_um, dx_um, dy_um)
    grad_term_eta = dgx_eta_dx + dgy_eta_dy
    eta_rhs = twin_dw - tw_drive - grad_term_eta
    pde_eta = (pred_d[:, idx["eta"] : idx["eta"] + 1].float() / dt) + float(cfg.twinning.L_eta) * eta_rhs
    pde_loss = torch.mean(pde_phi * pde_phi + pde_c * pde_c + pde_eta * pde_eta)

    rollout_loss = torch.tensor(0.0, device=pred_d.device, dtype=pred_d.dtype)
    if (
        rollout_pred_d is not None
        and rollout_target_d is not None
        and rollout_mask is not None
        and rollout_weight > 0.0
    ):
        mask = rollout_mask.to(dtype=pred_d.dtype, device=pred_d.device).view(-1, 1, 1, 1)
        mse2 = torch.mean((rollout_pred_d - rollout_target_d) ** 2, dim=(1, 2, 3), keepdim=True)
        rollout_loss = torch.sum(mse2 * mask) / torch.clamp(torch.sum(mask), min=1.0)

    loss = (
        w_data * data_loss
        + w_bounds * bounds_loss
        + w_mass * mass_loss
        + w_mech * mech_loss
        + w_pde * pde_loss
        + rollout_weight * rollout_loss
    )
    scalars = {
        "data": float(data_loss.detach().item()),
        "bounds": float(bounds_loss.detach().item()),
        "mass": float(mass_loss.detach().item()),
        "mech": float(mech_loss.detach().item()),
        "pde": float(pde_loss.detach().item()),
        "rollout": float(rollout_loss.detach().item()),
    }
    return loss, scalars


def main() -> None:
    """训练入口：数据准备、模型训练、验证与保存。"""
    args = parse_args()
    cfg = load_config(args.config)
    # 1) 将 CLI 参数覆盖到训练配置（CLI 优先）。
    if args.epochs > 0:
        cfg.ml.train_epochs = args.epochs
    if args.batch_size > 0:
        cfg.ml.train_batch_size = args.batch_size
    if args.lr > 0:
        cfg.ml.train_lr = args.lr
    if args.device != "auto":
        cfg.runtime.device = args.device
    if args.mixed_precision:
        cfg.numerics.mixed_precision = True
    if args.model_arch:
        cfg.ml.model_arch = args.model_arch
    if args.model_hidden > 0:
        cfg.ml.model_hidden = int(args.model_hidden)
    if args.dw_hidden > 0:
        cfg.ml.dw_hidden = int(args.dw_hidden)
    if args.dw_depth > 0:
        cfg.ml.dw_depth = int(args.dw_depth)
    if args.fno_width > 0:
        cfg.ml.fno_width = int(args.fno_width)
    if args.fno_modes_x > 0:
        cfg.ml.fno_modes_x = int(args.fno_modes_x)
    if args.fno_modes_y > 0:
        cfg.ml.fno_modes_y = int(args.fno_modes_y)
    if args.fno_depth > 0:
        cfg.ml.fno_depth = int(args.fno_depth)
    if args.afno_width > 0:
        cfg.ml.afno_width = int(args.afno_width)
    if args.afno_modes_x > 0:
        cfg.ml.afno_modes_x = int(args.afno_modes_x)
    if args.afno_modes_y > 0:
        cfg.ml.afno_modes_y = int(args.afno_modes_y)
    if args.afno_depth > 0:
        cfg.ml.afno_depth = int(args.afno_depth)
    if args.afno_expansion > 0:
        cfg.ml.afno_expansion = float(args.afno_expansion)
    if args.disable_torch_compile:
        cfg.numerics.use_torch_compile = False

    # 2) 固定随机种子，确保训练可复现。
    torch.manual_seed(cfg.numerics.seed)
    np.random.seed(cfg.numerics.seed)

    if args.snapshots_dir:
        snap_dir = Path(args.snapshots_dir)
    else:
        snap_dir = Path(cfg.runtime.output_dir) / cfg.runtime.case_name / "snapshots"
    files = sorted(snap_dir.glob("snapshot_*.npz"))
    if len(files) < 3:
        raise RuntimeError(f"Need at least 3 snapshots, got {len(files)} in {snap_dir}")
    if args.max_pairs > 0:
        # 用于快速实验：截断训练数据规模。
        max_files = max(3, int(args.max_pairs) + 1)
        files = files[:max_files]

    n = max(0, len(files) - 1)
    n_train = max(1, int(cfg.ml.train_split * n))
    n_val = max(0, n - n_train)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.numerics.mixed_precision and device.type == "cuda"
    if arch_requires_full_precision(cfg.ml.model_arch):
        # Current CUDA complex FFT path used by FNO spectral conv does not support ComplexHalf reliably.
        use_amp = False

    # 3) 构建 surrogate 模型与优化器。
    predictor = build_surrogate(
        device=device,
        use_torch_compile=cfg.numerics.use_torch_compile and device.type == "cuda",
        model_arch=cfg.ml.model_arch,
        hidden=cfg.ml.model_hidden,
        dw_hidden=cfg.ml.dw_hidden,
        dw_depth=cfg.ml.dw_depth,
        fno_width=cfg.ml.fno_width,
        fno_modes_x=cfg.ml.fno_modes_x,
        fno_modes_y=cfg.ml.fno_modes_y,
        fno_depth=cfg.ml.fno_depth,
        afno_width=cfg.ml.afno_width,
        afno_modes_x=cfg.ml.afno_modes_x,
        afno_modes_y=cfg.ml.afno_modes_y,
        afno_depth=cfg.ml.afno_depth,
        afno_expansion=cfg.ml.afno_expansion,
    )
    model = predictor.model
    use_channels_last = args.channels_last and device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ml.train_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch = cfg.ml.train_batch_size
    rollout_steps = max(1, int(args.rollout_steps))
    rollout_weight = float(args.rollout_weight if args.rollout_weight > 0.0 else 0.0)
    w_data = float(max(args.loss_data_weight, 0.0))
    w_bounds = float(max(args.loss_bounds_weight, 0.0))
    w_mass = float(max(args.loss_mass_weight, 0.0))
    w_mech = float(max(args.loss_mech_weight, 0.0))
    w_pde = float(max(args.loss_pde_weight, 0.0))
    grad_clip = float(args.grad_clip)
    if args.streaming:
        # 4A) 流式模式：边读边训，适合长时大数据快照。
        train_ds = SnapshotPairDataset(files[: n_train + 1])
        val_ds = SnapshotPairDataset(files[n_train:]) if n_val > 0 else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch,
            shuffle=True,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        val_loader = (
            DataLoader(
                val_ds,
                batch_size=batch,
                shuffle=False,
                num_workers=max(0, int(args.num_workers)),
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            )
            if val_ds is not None and len(val_ds) > 0
            else None
        )
    else:
        # 4B) 内存模式：一次性载入，适合中小规模训练集。
        x, dy, dy2, m2 = load_pairs(files)
        x_train, dy_train, dy2_train, m2_train = x[:n_train], dy[:n_train], dy2[:n_train], m2[:n_train]
        x_val, dy_val, dy2_val, m2_val = x[n_train:], dy[n_train:], dy2[n_train:], m2[n_train:]

    best_val = float("inf")
    best_state = None
    # 5) 标准训练循环：训练 -> 验证 -> 保存最佳权重。
    for epoch in range(1, cfg.ml.train_epochs + 1):
        model.train()
        train_loss = 0.0
        train_comp = {"data": 0.0, "bounds": 0.0, "mass": 0.0, "mech": 0.0, "pde": 0.0, "rollout": 0.0}
        train_count = 0
        if args.streaming:
            for xb, db, db2, m2 in train_loader:
                xb = xb.to(device, non_blocking=True)
                db = db.to(device, non_blocking=True)
                db2 = db2.to(device, non_blocking=True)
                m2 = m2.to(device, non_blocking=True)
                if use_channels_last:
                    xb = xb.contiguous(memory_format=torch.channels_last)
                    db = db.contiguous(memory_format=torch.channels_last)
                    db2 = db2.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                # autocast 仅在 CUDA + 允许 AMP 时启用。
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred_d = model(xb)
                    roll_pred = None
                    if rollout_steps >= 2 and rollout_weight > 0.0:
                        roll_pred = model(xb + pred_d)
                    loss, comp = _physics_losses(
                        xb,
                        pred_d,
                        db,
                        cfg,
                        w_data=w_data,
                        w_bounds=w_bounds,
                        w_mass=w_mass,
                        w_mech=w_mech,
                        w_pde=w_pde,
                        rollout_pred_d=roll_pred,
                        rollout_target_d=db2,
                        rollout_mask=m2,
                        rollout_weight=rollout_weight,
                    )
                scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                bs = int(xb.shape[0])
                train_loss += float(loss.item()) * bs
                for k in train_comp:
                    train_comp[k] += float(comp[k]) * bs
                train_count += bs
        else:
            perm = torch.randperm(x_train.shape[0])
            for i in range(0, x_train.shape[0], batch):
                idx = perm[i : i + batch]
                xb = x_train[idx].to(device, non_blocking=True)
                db = dy_train[idx].to(device, non_blocking=True)
                db2 = dy2_train[idx].to(device, non_blocking=True)
                m2 = m2_train[idx].to(device, non_blocking=True)
                if use_channels_last:
                    xb = xb.contiguous(memory_format=torch.channels_last)
                    db = db.contiguous(memory_format=torch.channels_last)
                    db2 = db2.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred_d = model(xb)
                    roll_pred = None
                    if rollout_steps >= 2 and rollout_weight > 0.0:
                        roll_pred = model(xb + pred_d)
                    loss, comp = _physics_losses(
                        xb,
                        pred_d,
                        db,
                        cfg,
                        w_data=w_data,
                        w_bounds=w_bounds,
                        w_mass=w_mass,
                        w_mech=w_mech,
                        w_pde=w_pde,
                        rollout_pred_d=roll_pred,
                        rollout_target_d=db2,
                        rollout_mask=m2,
                        rollout_weight=rollout_weight,
                    )
                scaler.scale(loss).backward()
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                bs = int(xb.shape[0])
                train_loss += float(loss.item()) * bs
                for k in train_comp:
                    train_comp[k] += float(comp[k]) * bs
                train_count += bs
        train_loss /= max(train_count, 1)
        for k in train_comp:
            train_comp[k] /= max(train_count, 1)

        model.eval()
        if n_val > 0:
            with torch.no_grad():
                if args.streaming and val_loader is not None:
                    vloss = 0.0
                    vcomp = {"data": 0.0, "bounds": 0.0, "mass": 0.0, "mech": 0.0, "pde": 0.0, "rollout": 0.0}
                    vcount = 0
                    for xb, db, db2, m2 in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        db = db.to(device, non_blocking=True)
                        db2 = db2.to(device, non_blocking=True)
                        m2 = m2.to(device, non_blocking=True)
                        if use_channels_last:
                            xb = xb.contiguous(memory_format=torch.channels_last)
                            db = db.contiguous(memory_format=torch.channels_last)
                            db2 = db2.contiguous(memory_format=torch.channels_last)
                        val_pred_d = model(xb)
                        roll_pred = None
                        if rollout_steps >= 2 and rollout_weight > 0.0:
                            roll_pred = model(xb + val_pred_d)
                        l_t, comp = _physics_losses(
                            xb,
                            val_pred_d,
                            db,
                            cfg,
                            w_data=w_data,
                            w_bounds=w_bounds,
                            w_mass=w_mass,
                            w_mech=w_mech,
                            w_pde=w_pde,
                            rollout_pred_d=roll_pred,
                            rollout_target_d=db2,
                            rollout_mask=m2,
                            rollout_weight=rollout_weight,
                        )
                        l = float(l_t.item())
                        bs = int(xb.shape[0])
                        vloss += l * bs
                        for k in vcomp:
                            vcomp[k] += float(comp[k]) * bs
                        vcount += bs
                    val_loss = vloss / max(vcount, 1)
                    for k in vcomp:
                        vcomp[k] /= max(vcount, 1)
                else:
                    xb = x_val.to(device, non_blocking=True)
                    db = dy_val.to(device, non_blocking=True)
                    db2 = dy2_val.to(device, non_blocking=True)
                    m2 = m2_val.to(device, non_blocking=True)
                    if use_channels_last:
                        xb = xb.contiguous(memory_format=torch.channels_last)
                        db = db.contiguous(memory_format=torch.channels_last)
                        db2 = db2.contiguous(memory_format=torch.channels_last)
                    val_pred_d = model(xb)
                    roll_pred = None
                    if rollout_steps >= 2 and rollout_weight > 0.0:
                        roll_pred = model(xb + val_pred_d)
                    v_t, vcomp = _physics_losses(
                        xb,
                        val_pred_d,
                        db,
                        cfg,
                        w_data=w_data,
                        w_bounds=w_bounds,
                        w_mass=w_mass,
                        w_mech=w_mech,
                        w_pde=w_pde,
                        rollout_pred_d=roll_pred,
                        rollout_target_d=db2,
                        rollout_mask=m2,
                        rollout_weight=rollout_weight,
                    )
                    val_loss = float(v_t.item())
        else:
            val_loss = train_loss
            vcomp = dict(train_comp)

        if val_loss < best_val:
            # 只保留验证最优权重，避免最后一轮退化。
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {epoch:03d} train={train_loss:.6e} val={val_loss:.6e} "
            f"| train[data={train_comp['data']:.2e},bounds={train_comp['bounds']:.2e},mass={train_comp['mass']:.2e},"
            f"mech={train_comp['mech']:.2e},pde={train_comp['pde']:.2e},roll={train_comp['rollout']:.2e}] "
            f"| val[data={vcomp['data']:.2e},bounds={vcomp['bounds']:.2e},mass={vcomp['mass']:.2e},"
            f"mech={vcomp['mech']:.2e},pde={vcomp['pde']:.2e},roll={vcomp['rollout']:.2e}]"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    # 6) 训练结束：写出 surrogate 模型文件。
    save_surrogate(predictor, cfg.ml.model_path)

    print(
        json.dumps(
            {
                "snapshots_dir": str(snap_dir.resolve()),
                "n_pairs": int(n),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "best_val": best_val,
                "model_path": str(Path(cfg.ml.model_path).resolve()),
                "model_arch": cfg.ml.model_arch,
                "loss_weights": {
                    "data": w_data,
                    "bounds": w_bounds,
                    "mass": w_mass,
                    "mech": w_mech,
                    "pde": w_pde,
                    "rollout": rollout_weight,
                },
                "rollout_steps": rollout_steps,
                "model_kwargs": {
                    "hidden": int(cfg.ml.model_hidden),
                    "dw_hidden": int(cfg.ml.dw_hidden),
                    "dw_depth": int(cfg.ml.dw_depth),
                    "fno_width": int(cfg.ml.fno_width),
                    "fno_modes_x": int(cfg.ml.fno_modes_x),
                    "fno_modes_y": int(cfg.ml.fno_modes_y),
                    "fno_depth": int(cfg.ml.fno_depth),
                    "afno_width": int(cfg.ml.afno_width),
                    "afno_modes_x": int(cfg.ml.afno_modes_x),
                    "afno_modes_y": int(cfg.ml.afno_modes_y),
                    "afno_depth": int(cfg.ml.afno_depth),
                    "afno_expansion": float(cfg.ml.afno_expansion),
                },
                "device": str(device),
                "streaming": bool(args.streaming),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
