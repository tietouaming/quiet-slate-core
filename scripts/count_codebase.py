"""统计仓库代码规模（按 git 跟踪文件）。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


def git_ls_files() -> List[str]:
    out = subprocess.check_output(["git", "ls-files"], text=True, encoding="utf-8")
    return [s.strip() for s in out.splitlines() if s.strip()]


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def sum_lines(files: Iterable[str]) -> int:
    s = 0
    for f in files:
        p = Path(f)
        if p.exists():
            s += count_lines(p)
    return int(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="Count codebase lines by tracked files.")
    ap.add_argument("--output", type=str, default="", help="optional output json path")
    args = ap.parse_args()

    files = git_ls_files()
    py = [f for f in files if f.endswith(".py")]
    md = [f for f in files if f.endswith(".md")]
    yaml = [f for f in files if f.endswith(".yaml") or f.endswith(".yml")]
    jsn = [f for f in files if f.endswith(".json")]
    ps1 = [f for f in files if f.endswith(".ps1")]
    core_py = [f for f in py if f.startswith("mg_coupled_pf/")]
    tests_py = [f for f in py if f.startswith("tests/")]
    scripts_py = [f for f in py if f.startswith("scripts/")]
    multiscale_py = [f for f in py if f.startswith("mg_coupled_pf/multiscale/")]

    result: Dict[str, int] = {
        "total_tracked_files": len(files),
        "py_files": len(py),
        "py_lines_total": sum_lines(py),
        "py_lines_core": sum_lines(core_py),
        "py_lines_multiscale": sum_lines(multiscale_py),
        "py_lines_tests": sum_lines(tests_py),
        "py_lines_scripts": sum_lines(scripts_py),
        "md_files": len(md),
        "md_lines_total": sum_lines(md),
        "yaml_files": len(yaml),
        "yaml_lines_total": sum_lines(yaml),
        "json_files": len(jsn),
        "json_lines_total": sum_lines(jsn),
        "ps1_files": len(ps1),
        "ps1_lines_total": sum_lines(ps1),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

