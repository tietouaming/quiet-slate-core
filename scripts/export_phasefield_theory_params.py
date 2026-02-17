"""导出相场理论与全局参数总表（自动同步当前配置）。"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, List, Tuple

from mg_coupled_pf import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export phase-field theory and global parameter registry document.")
    p.add_argument("--config", type=str, default="configs/notch_case.yaml")
    p.add_argument("--output", type=str, default="docs/相场理论与参数总表.md")
    return p.parse_args()


def _collect_dataclass_fields(obj: Any, prefix: str = "") -> List[Tuple[str, str, Any]]:
    rows: List[Tuple[str, str, Any]] = []
    if not is_dataclass(obj):
        return rows
    for f in fields(obj):
        name = f.name
        val = getattr(obj, name)
        path = f"{prefix}.{name}" if prefix else name
        if is_dataclass(val):
            rows.extend(_collect_dataclass_fields(val, path))
        else:
            rows.append((path, f.type.__name__ if hasattr(f.type, "__name__") else str(f.type), val))
    return rows


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.8g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    return str(v)


def build_doc(config_path: str) -> str:
    cfg = load_config(config_path)
    rows = _collect_dataclass_fields(cfg)
    lines: List[str] = []
    lines.append("# 相场理论与全局参数总表")
    lines.append("")
    lines.append("> 本文档由脚本自动生成：`scripts/export_phasefield_theory_params.py`。")
    lines.append("> 修改全局参数后重新执行该脚本，即可同步更新所有参数表。")
    lines.append("")
    lines.append("## 一、理论方程（与代码实现一致）")
    lines.append("")
    lines.append("1. 腐蚀相场（Allen-Cahn 非守恒形式）")
    lines.append("")
    lines.append("```text")
    lines.append("∂phi/∂t = -L_phi * ( d(f_chem+f_dw)/dphi - kappa_phi * ∇²phi + coupling_terms )")
    lines.append("```")
    lines.append("")
    lines.append("2. 镁离子归一化浓度扩散（守恒形式）")
    lines.append("")
    lines.append("```text")
    lines.append("mu_c = A * (cMg - h(phi)*(c_s_eq-c_l_eq) - c_l_eq)")
    lines.append("∂cMg/∂t = ∇·( M(phi) ∇mu_c )")
    lines.append("M = D / A")
    lines.append("```")
    lines.append("")
    lines.append("3. 孪晶序参量（TDGL / Allen-Cahn 形式）")
    lines.append("")
    lines.append("```text")
    lines.append("∂eta/∂t = -L_eta * ( d(f_eta)/deta - kappa_eta * ∇²eta - tau_tw*gamma_twin*dh(eta)/deta )")
    lines.append("```")
    lines.append("")
    lines.append("4. 力学平衡（准静态）")
    lines.append("")
    lines.append("```text")
    lines.append("∇·sigma = 0")
    lines.append("sigma = C : (eps(u) - eps_p - eps_tw)")
    lines.append("```")
    lines.append("")
    lines.append("5. 晶体塑性（速率型）")
    lines.append("")
    lines.append("```text")
    lines.append("gamma_dot = gamma0 * <|tau/g| - 1>^m * sign(tau)")
    lines.append("eps_p_dot = Σ gamma_dot * sym(s ⊗ n)")
    lines.append("```")
    lines.append("")
    lines.append("## 二、参数统一修改方式")
    lines.append("")
    lines.append("- 推荐统一在 `configs/notch_case.yaml` 修改。")
    lines.append("- 运行脚本同步文档：")
    lines.append("")
    lines.append("```powershell")
    lines.append("python scripts/export_phasefield_theory_params.py --config configs/notch_case.yaml --output docs/相场理论与参数总表.md")
    lines.append("```")
    lines.append("")
    lines.append("## 三、全局参数总表")
    lines.append("")
    lines.append("| 参数路径 | 类型 | 当前值 |")
    lines.append("|---|---|---|")
    for path, typ, val in rows:
        lines.append(f"| `{path}` | `{typ}` | `{_fmt_value(val)}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    text = build_doc(args.config)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print({"output": str(out), "chars": len(text)})


if __name__ == "__main__":
    main()

