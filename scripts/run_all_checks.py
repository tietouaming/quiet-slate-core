#!/usr/bin/env python
"""一键检查脚本（中文注释版）。

集成执行：
1. 单元测试
2. 训练烟测
3. 物理/ML 稳定性与速度检查
并生成统一 JSON 报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析检查流程参数。"""
    p = argparse.ArgumentParser(description="Run environment checks, tests, and optimization validation.")
    p.add_argument("--config", default="configs/notch_case.yaml")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--disable-torch-compile", action="store_true")
    p.add_argument("--physics-steps", type=int, default=80)
    p.add_argument("--ml-steps", type=int, default=80)
    p.add_argument("--bench-steps", type=int, default=80)
    p.add_argument("--train-epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--max-sigma-mpa", type=float, default=5000.0)
    p.add_argument("--max-epspeq", type=float, default=5.0)
    p.add_argument("--max-solid-drift", type=float, default=0.1)
    p.add_argument("--min-speedup", type=float, default=0.0)
    p.add_argument("--report", default="artifacts/validation/run_all_checks_report.json")
    return p.parse_args()


def run_cmd(args: List[str], timeout_s: int = 600) -> Dict[str, Any]:
    """运行子命令并返回耗时与输出。"""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        # 失败时连同 stdout/stderr 抛出，便于直接定位问题。
        raise RuntimeError(
            f"Command failed: {args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return {"args": args, "wall_time_s": dt, "stdout": proc.stdout, "stderr": proc.stderr}


def parse_last_json(stdout: str) -> Dict[str, Any]:
    """从脚本 stdout 中提取最后一个 JSON 载荷。"""
    idx = [i for i, ch in enumerate(stdout) if ch == "{"]
    for i in reversed(idx):
        chunk = stdout[i:].strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON payload found in output:\n{stdout}")


def read_history_metrics(path: Path) -> Dict[str, float]:
    """读取 history.csv 并提取关键统计量。"""
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"History file has no rows: {path}")
    max_sigma = max(float(r["max_sigma_h"]) for r in rows)
    max_epspeq = max(float(r["avg_epspeq"]) for r in rows)
    final = rows[-1]
    return {
        "n_rows": float(len(rows)),
        "max_sigma_h_MPa": max_sigma,
        "max_avg_epspeq": max_epspeq,
        "final_solid_fraction": float(final["solid_fraction"]),
        "final_avg_eta": float(final["avg_eta"]),
    }


def main() -> None:
    """执行全流程检查并输出报告。"""
    args = parse_args()
    report: Dict[str, Any] = {"python": sys.executable}
    # 将常用运行参数拼接成共享尾参数，减少重复代码。
    common_run_args: List[str] = []
    if args.device != "auto":
        common_run_args += ["--device", args.device]
    if args.mixed_precision:
        common_run_args += ["--mixed-precision"]
    if args.disable_torch_compile:
        common_run_args += ["--disable-torch-compile"]

    # 0) 配置审计（先于测试与仿真，提前发现配置层问题）。
    audit = run_cmd(
        [
            "scripts/audit_config.py",
            "--config",
            args.config,
            "--output",
            "artifacts/validation/config_audit_run_all_checks.json",
            "--strict",
        ],
        timeout_s=120,
    )
    report["config_audit"] = parse_last_json(audit["stdout"])

    # 1) 单测。
    pytest_res = run_cmd(["-m", "pytest", "-q"], timeout_s=600)
    report["pytest"] = {"wall_time_s": pytest_res["wall_time_s"], "stdout": pytest_res["stdout"]}

    # 2) 纯物理短跑，并提取 history 指标。
    physics = run_cmd(
        [
            "scripts/run_notched_case.py",
            "--config",
            args.config,
            "--override-steps",
            str(args.physics_steps),
            "--override-save-every",
            "1",
            "--disable-ml",
            *common_run_args,
        ]
    )
    physics_json = parse_last_json(physics["stdout"])
    physics_metrics = read_history_metrics(Path(physics_json["history_csv"]))
    report["physics_run"] = {
        "wall_time_s": physics["wall_time_s"],
        "output": physics_json,
        "metrics": physics_metrics,
    }

    # 3) 用纯物理快照做 surrogate 短程训练烟测。
    train = run_cmd(
        [
            "scripts/train_surrogate.py",
            "--config",
            args.config,
            "--snapshots-dir",
            physics_json["snapshots_dir"],
            "--epochs",
            str(args.train_epochs),
            "--batch-size",
            str(args.train_batch_size),
            *common_run_args,
        ]
    )
    report["train_surrogate"] = {
        "wall_time_s": train["wall_time_s"],
        "output": parse_last_json(train["stdout"]),
    }

    # 4) 启用 ML 跑同类短程，并提取 history 指标。
    ml = run_cmd(
        [
            "scripts/run_notched_case.py",
            "--config",
            args.config,
            "--override-steps",
            str(args.ml_steps),
            "--override-save-every",
            str(max(1, args.ml_steps // 4)),
            *common_run_args,
        ]
    )
    ml_json = parse_last_json(ml["stdout"])
    ml_metrics = read_history_metrics(Path(ml_json["history_csv"]))
    report["ml_run"] = {
        "wall_time_s": ml["wall_time_s"],
        "output": ml_json,
        "metrics": ml_metrics,
    }

    # 5) 独立跑一次 benchmark，提取速度指标。
    bench = run_cmd(
        [
            "scripts/benchmark_solver.py",
            "--config",
            args.config,
            "--steps",
            str(args.bench_steps),
            "--warmup-steps",
            "20",
            *common_run_args,
        ]
    )
    report["benchmark"] = parse_last_json(bench["stdout"])
    solid_drift = abs(ml_metrics["final_solid_fraction"] - physics_metrics["final_solid_fraction"])
    report["drift"] = {"solid_fraction_abs_diff": solid_drift}

    checks = {
        "physics_sigma_ok": physics_metrics["max_sigma_h_MPa"] <= args.max_sigma_mpa,
        "physics_epspeq_ok": physics_metrics["max_avg_epspeq"] <= args.max_epspeq,
        "ml_sigma_ok": ml_metrics["max_sigma_h_MPa"] <= args.max_sigma_mpa,
        "ml_epspeq_ok": ml_metrics["max_avg_epspeq"] <= args.max_epspeq,
        "solid_drift_ok": solid_drift <= args.max_solid_drift,
        "speedup_ok": float(report["benchmark"]["speedup"]) >= args.min_speedup,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"report": str(report_path.resolve()), "passed": report["passed"], "checks": checks}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        # 非全通过时返回非零码，方便 CI/批处理识别失败。
        raise SystemExit(2)


if __name__ == "__main__":
    main()
