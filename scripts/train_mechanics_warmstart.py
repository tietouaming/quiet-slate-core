#!/usr/bin/env python
"""力学初值器训练脚本。

目标：
- 学习 (phi,c,eta,epsp*) -> (ux,uy) 的映射；
- 作为力学 Krylov 初值器，降低迭代步数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf import load_config
from mg_coupled_pf.ml.mech_warmstart import (
    MECH_INPUT_ORDER,
    build_mechanics_warmstart,
    save_mechanics_warmstart,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mechanics warmstart model.")
    p.add_argument("--config", default="configs/notch_case.yaml")
    p.add_argument("--snapshots-dir", default="")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--disable-torch-compile", action="store_true")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--bc-penalty", type=float, default=0.05)
    return p.parse_args()


def _pick_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_npz(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    ref = None
    for k in data.files:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            ref = arr
            break
    if ref is None:
        raise RuntimeError(f"{path} missing [1,1,H,W] fields")
    xs: List[np.ndarray] = []
    for k in MECH_INPUT_ORDER:
        if k in data.files:
            xs.append(data[k].astype(np.float32))
        else:
            xs.append(np.zeros_like(ref, dtype=np.float32))
    if "ux" not in data.files or "uy" not in data.files:
        raise RuntimeError(f"{path} missing ux/uy fields")
    x = torch.from_numpy(np.concatenate(xs, axis=1).astype(np.float32)).squeeze(0)  # [C,H,W]
    y = torch.from_numpy(np.concatenate([data["ux"], data["uy"]], axis=1).astype(np.float32)).squeeze(0)  # [2,H,W]
    return x, y


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    snap_dir = Path(args.snapshots_dir) if args.snapshots_dir else (Path(cfg.runtime.output_dir) / cfg.runtime.case_name / "snapshots")
    files = sorted(snap_dir.glob("snapshot_*.npz"))
    if len(files) < 2:
        raise RuntimeError(f"Need >=2 snapshots in {snap_dir}")
    if args.max_samples > 0:
        files = files[: int(args.max_samples)]

    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for p in files:
        x, y = _load_npz(p)
        xs.append(x)
        ys.append(y)
    x_all = torch.stack(xs, dim=0)  # [N,C,H,W]
    y_all = torch.stack(ys, dim=0)  # [N,2,H,W]

    n = x_all.shape[0]
    n_train = max(1, int(0.9 * n))
    x_train, y_train = x_all[:n_train], y_all[:n_train]
    if n_train < n:
        x_val, y_val = x_all[n_train:], y_all[n_train:]
    else:
        x_val, y_val = x_all[:0], y_all[:0]

    device = _pick_device(args.device)
    predictor = build_mechanics_warmstart(
        device=device,
        hidden=int(getattr(cfg.ml, "mechanics_warmstart_hidden", 24)),
        add_coord_features=bool(getattr(cfg.ml, "mechanics_warmstart_add_coord_features", True)),
        use_torch_compile=(not args.disable_torch_compile) and device.type == "cuda",
        channels_last=args.channels_last and device.type == "cuda",
    )
    model = predictor.model
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    bs = max(1, int(args.batch_size))
    bc_pen = max(float(args.bc_penalty), 0.0)
    loading_mode = str(cfg.mechanics.loading_mode).strip().lower()
    use_dirichlet = loading_mode in {"dirichlet_x", "dirichlet", "ux_dirichlet"}
    right_u = float(cfg.mechanics.dirichlet_right_displacement_um)
    if right_u < 0.0:
        right_u = float(cfg.mechanics.external_strain_x) * float(cfg.domain.lx_um)

    best_val = float("inf")
    best_state = None

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        perm = torch.randperm(x_train.shape[0])
        train_loss = 0.0
        train_cnt = 0
        for i in range(0, x_train.shape[0], bs):
            idx = perm[i : i + bs]
            xb = x_train[idx].to(device, non_blocking=True)
            yb = y_train[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss_data = F.mse_loss(pred, yb)
            # 边界约束正则：确保网络学到“可用初值”而非任意位移场。
            ux = pred[:, 0:1]
            uy = pred[:, 1:2]
            bc_loss = torch.mean(ux[:, :, :, 0] * ux[:, :, :, 0])
            if use_dirichlet:
                bc_loss = bc_loss + torch.mean((ux[:, :, :, -1] - right_u) * (ux[:, :, :, -1] - right_u))
            bc_loss = bc_loss + torch.mean(uy[:, :, 0, 0] * uy[:, :, 0, 0])
            loss = loss_data + bc_pen * bc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            b = int(xb.shape[0])
            train_loss += float(loss.item()) * b
            train_cnt += b
        train_loss /= max(train_cnt, 1)

        model.eval()
        with torch.no_grad():
            if x_val.shape[0] > 0:
                pv = model(x_val.to(device))
                val_loss = float(F.mse_loss(pv, y_val.to(device)).item())
            else:
                val_loss = train_loss
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:03d} train={train_loss:.6e} val={val_loss:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(getattr(cfg.ml, "mechanics_warmstart_model_path", "artifacts/ml/mech_warmstart_latest.pt"))
    save_mechanics_warmstart(predictor, out)

    print(
        json.dumps(
            {
                "snapshots_dir": str(snap_dir.resolve()),
                "n_samples": int(n),
                "n_train": int(n_train),
                "n_val": int(max(0, n - n_train)),
                "best_val": float(best_val),
                "model_path": str(out.resolve()),
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
