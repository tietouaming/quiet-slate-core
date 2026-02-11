#!/usr/bin/env python
"""surrogate 架构推理基准脚本（中文注释版）。

用于在相同输入尺寸和设备下比较不同代理网络的单步推理延迟，
输出包含延迟、吞吐与参数量的 JSON 报告。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf.ml.surrogate import FIELD_ORDER, build_surrogate


def parse_args() -> argparse.Namespace:
    """解析基准测试参数。

    参数设计说明：
    - `--batch-size/--height/--width` 决定单次推理输入规模；
    - `--warmup` 用于预热，避免首次调用的初始化开销污染统计；
    - `--iters` 为正式统计轮数；
    - `--channels-last` 仅在 CUDA 下生效，可改善卷积访存；
    - `--disable-torch-compile` 用于关闭编译优化，便于公平对比；
    - `--architectures` 支持一次评测多个架构并给出横向结论。
    """
    p = argparse.ArgumentParser(description="Benchmark surrogate inference latency for different model architectures.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--disable-torch-compile", action="store_true")
    p.add_argument(
        "--architectures",
        default="tiny_unet,dw_unet,fno2d,afno2d",
        help="Comma-separated list from: tiny_unet,dw_unet,fno2d,afno2d",
    )
    p.add_argument("--tiny-hidden", type=int, default=32)
    p.add_argument("--dw-hidden", type=int, default=24)
    p.add_argument("--dw-depth", type=int, default=2)
    p.add_argument("--fno-width", type=int, default=32)
    p.add_argument("--fno-modes-x", type=int, default=24)
    p.add_argument("--fno-modes-y", type=int, default=16)
    p.add_argument("--fno-depth", type=int, default=4)
    p.add_argument("--afno-width", type=int, default=24)
    p.add_argument("--afno-modes-x", type=int, default=20)
    p.add_argument("--afno-modes-y", type=int, default=12)
    p.add_argument("--afno-depth", type=int, default=4)
    p.add_argument("--afno-expansion", type=float, default=2.0)
    return p.parse_args()


def _pick_device(mode: str) -> torch.device:
    """选择运行设备。

    规则：
    - 显式 `cpu`：强制 CPU；
    - 显式 `cuda`：若不可用则自动回退 CPU；
    - `auto`：优先 CUDA。
    """
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync(dev: torch.device) -> None:
    """CUDA 计时同步。

    原因：
    - CUDA 算子默认异步提交，如果不 `synchronize`，
      `perf_counter` 只统计了“提交时间”，会严重低估真实耗时。
    """
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _param_count(model: torch.nn.Module) -> int:
    """统计模型参数总量（用于比较模型复杂度）。"""
    return int(sum(p.numel() for p in model.parameters()))


def _make_state(dev: torch.device, batch: int, h: int, w: int) -> dict[str, torch.Tensor]:
    """生成随机输入状态。

    张量布局约定：
    - 每个物理场通道形状为 `[B, 1, H, W]`；
    - 字典键顺序由 `FIELD_ORDER` 统一定义，保证和 surrogate 输入一致。
    """
    state = {}
    for k in FIELD_ORDER:
        state[k] = torch.rand((batch, 1, h, w), device=dev, dtype=torch.float32)
    return state


def _bench_one(
    *,
    arch: str,
    dev: torch.device,
    args: argparse.Namespace,
    state: dict[str, torch.Tensor],
) -> dict[str, float | int | str]:
    """对单个架构执行 warmup 与正式计时。

    统计指标：
    - `avg_ms_per_step`：平均每步推理毫秒数；
    - `throughput_mpx_per_s`：每秒处理像素百万数（MPx/s）；
    - `params`：参数总数，帮助判断“速度-容量”折中。
    """
    # 1) 按同一套超参数入口构建目标架构，保证对比口径统一。
    predictor = build_surrogate(
        device=dev,
        use_torch_compile=(not args.disable_torch_compile) and dev.type == "cuda",
        channels_last=args.channels_last and dev.type == "cuda",
        model_arch=arch,
        hidden=args.tiny_hidden,
        dw_hidden=args.dw_hidden,
        dw_depth=args.dw_depth,
        fno_width=args.fno_width,
        fno_modes_x=args.fno_modes_x,
        fno_modes_y=args.fno_modes_y,
        fno_depth=args.fno_depth,
        afno_width=args.afno_width,
        afno_modes_x=args.afno_modes_x,
        afno_modes_y=args.afno_modes_y,
        afno_depth=args.afno_depth,
        afno_expansion=args.afno_expansion,
    )
    # 2) 参数量统计，用于解释“更快是否只是更小模型”。
    model = predictor.model
    params = _param_count(model)

    # 3) 预热阶段：触发内核选择、缓存建立、可选编译路径稳定。
    for _ in range(max(1, args.warmup)):
        _ = predictor.predict(state)
    _sync(dev)

    # 4) 正式计时阶段：仅统计稳定后的推理环节。
    t0 = time.perf_counter()
    for _ in range(max(1, args.iters)):
        _ = predictor.predict(state)
    _sync(dev)

    # 5) 由总耗时换算平均单步时延与吞吐。
    elapsed = time.perf_counter() - t0
    avg_ms = 1000.0 * elapsed / max(args.iters, 1)
    mpx = (args.batch_size * args.height * args.width) / 1e6
    mpx_per_s = (mpx * max(args.iters, 1)) / max(elapsed, 1e-12)
    return {
        "arch": arch,
        "device": str(dev),
        "params": params,
        "avg_ms_per_step": avg_ms,
        "throughput_mpx_per_s": mpx_per_s,
    }


def main() -> None:
    """脚本主流程。

    流程：
    1. 解析参数并确定设备；
    2. 生成一次随机状态作为全部架构共用输入；
    3. 逐架构执行基准；
    4. 计算最快架构与相对 tiny_unet 的速度比；
    5. 输出结构化 JSON 报告。
    """
    args = parse_args()
    dev = _pick_device(args.device)
    # 基准公平性关键点：所有架构使用同一份输入尺寸与分布。
    state = _make_state(dev, args.batch_size, args.height, args.width)
    arch_list = [a.strip() for a in str(args.architectures).split(",") if a.strip()]
    if not arch_list:
        raise RuntimeError("No architectures specified.")
    res = {}
    for arch in arch_list:
        # 逐架构独立测量，避免不同模型间共享状态导致干扰。
        res[arch] = _bench_one(arch=arch, dev=dev, args=args, state=state)

    # 选出平均时延最小的架构作为当前硬件/尺寸下的“最快模型”。
    fastest = min(res.items(), key=lambda kv: float(kv[1]["avg_ms_per_step"]))
    baseline_tiny = res.get("tiny_unet")
    speedups_vs_tiny = {}
    if baseline_tiny is not None:
        # 相对 tiny_unet 的速度倍率：>1 表示更快。
        tiny_ms = float(baseline_tiny["avg_ms_per_step"])
        for k, v in res.items():
            speedups_vs_tiny[k] = float(tiny_ms / max(float(v["avg_ms_per_step"]), 1e-12))

    report = {
        "meta": {
            "batch_size": int(args.batch_size),
            "height": int(args.height),
            "width": int(args.width),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "channels_last": bool(args.channels_last),
            "torch_compile": bool((not args.disable_torch_compile) and dev.type == "cuda"),
            "architectures": arch_list,
            "tiny_hidden": int(args.tiny_hidden),
            "dw_hidden": int(args.dw_hidden),
            "dw_depth": int(args.dw_depth),
            "fno_width": int(args.fno_width),
            "fno_modes_x": int(args.fno_modes_x),
            "fno_modes_y": int(args.fno_modes_y),
            "fno_depth": int(args.fno_depth),
            "afno_width": int(args.afno_width),
            "afno_modes_x": int(args.afno_modes_x),
            "afno_modes_y": int(args.afno_modes_y),
            "afno_depth": int(args.afno_depth),
            "afno_expansion": float(args.afno_expansion),
        },
        "results": res,
        "fastest_arch": fastest[0],
        "fastest_avg_ms": float(fastest[1]["avg_ms_per_step"]),
        "speedup_vs_tiny": speedups_vs_tiny,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
