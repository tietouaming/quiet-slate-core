#!/usr/bin/env python
"""配置与物理一致性审计脚本（中文注释版）。

用途：
1. 在仿真前快速检查配置是否存在明显物理/数值风险；
2. 输出结构化 JSON，供自动化流水线直接消费；
3. 可选 strict 模式：发现 error 即返回非零退出码。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mg_coupled_pf import audit_config, load_config, save_audit_report, summarize_audit_report


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="Audit simulation config consistency before running.")
    p.add_argument("--config", default="configs/notch_case.yaml", help="YAML config path.")
    p.add_argument(
        "--output",
        default="artifacts/validation/config_audit.json",
        help="Audit report output path (JSON).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit code=2) when audit contains errors.",
    )
    return p.parse_args()


def main() -> None:
    """脚本主流程。"""
    args = parse_args()
    cfg = load_config(args.config)
    report = audit_config(cfg, config_path=args.config)
    out_path = save_audit_report(report, args.output)
    summary = summarize_audit_report(report)
    print(
        json.dumps(
            {
                "summary": summary,
                "passed": report.passed,
                "errors": report.error_count,
                "warnings": report.warning_count,
                "infos": report.info_count,
                "report": str(out_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.strict and not report.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

