#!/usr/bin/env python
"""公式 OCR 脚本（中文注释版）。

读取前序导出的公式图片清单，调用 pix2tex 批量识别 LaTeX，
并输出 JSONL 与 Markdown 预览结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from PIL import Image
import numpy as np
from pix2tex.cli import LatexOCR


def parse_args() -> argparse.Namespace:
    """解析 OCR 参数。"""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default="artifacts/formula_extract/formula_images_manifest.jsonl",
    )
    p.add_argument(
        "--out-jsonl",
        default="artifacts/formula_extract/formulas_ocr.jsonl",
    )
    p.add_argument(
        "--out-md",
        default="artifacts/formula_extract/formulas_ocr_preview.md",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of pending formulas to process in this run (0 = all).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore previous OCR output and recompute all rows.",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[Dict]:
    """读取 JSONL 文件为字典列表。"""
    if not path.exists():
        return []
    rows: list[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[Dict]) -> None:
    """将记录列表写出为 JSONL。"""
    with path.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def image_is_blank(img: Image.Image) -> bool:
    """检测图片是否接近空白占位图。"""
    arr = np.asarray(img.convert("L"))
    if arr.size == 0:
        return True
    # Treat near-uniform bright images as blank equation placeholders.
    if float(arr.std()) < 1e-3 and float(arr.mean()) > 250:
        return True
    return False


def main() -> None:
    """脚本主流程。"""
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_jsonl = Path(args.out_jsonl)
    out_md = Path(args.out_md)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读取公式图片清单（manifest）。
    rows = load_jsonl(manifest_path)
    if not rows:
        raise RuntimeError(f"manifest is empty or missing: {manifest_path}")

    # 2) 增量模式：默认跳过已完成 OCR 的记录。
    existing = [] if args.overwrite else load_jsonl(out_jsonl)
    existing_by_id = {rec["id"]: rec for rec in existing}
    pending = [r for r in rows if r["id"] not in existing_by_id]
    if args.limit > 0:
        # 调试模式：限制本轮处理数量，便于快速验证。
        pending = pending[: args.limit]

    if pending:
        # 仅在有待处理样本时加载模型，减少冷启动开销。
        model = LatexOCR()

    for idx, row in enumerate(pending, start=1):
        img_path = Path(row["image_path"])
        rec = dict(row)
        try:
            with Image.open(img_path) as im:
                if image_is_blank(im):
                    # 空白占位图不做 OCR，直接标记 blank。
                    rec["latex_ocr"] = ""
                    rec["ocr_status"] = "blank_image"
                else:
                    # 正常样本：调用 pix2tex 推理 LaTeX。
                    pred = model(im)
                    rec["latex_ocr"] = pred
                    rec["ocr_status"] = "ok"
        except Exception as e:
            # 对单条错误容错，不中断全量流程。
            rec["latex_ocr"] = ""
            rec["ocr_status"] = f"error:{type(e).__name__}"
            rec["ocr_error"] = str(e)
        existing_by_id[rec["id"]] = rec
        if idx % 20 == 0:
            print(f"processed {idx}/{len(pending)} in current run")
            # Periodic checkpoint.
            ordered_tmp = [existing_by_id[r["id"]] for r in rows if r["id"] in existing_by_id]
            dump_jsonl(out_jsonl, ordered_tmp)

    # 3) 按 manifest 顺序回排，确保下游比对稳定。
    ordered = [existing_by_id[r["id"]] for r in rows if r["id"] in existing_by_id]
    dump_jsonl(out_jsonl, ordered)

    # 4) 输出预览 markdown，便于人工快速抽查。
    preview_lines = ["# OCR Preview", ""]
    for rec in ordered[:120]:
        preview_lines.append(f"- `{rec['id']}`: `{rec.get('latex_ocr', '')}`")
    out_md.write_text("\n".join(preview_lines), encoding="utf-8")

    # 5) 汇总统计信息并输出 JSON。
    ok = sum(1 for r in ordered if r["ocr_status"] == "ok")
    blank = sum(1 for r in ordered if r["ocr_status"] == "blank_image")
    errors = len(ordered) - ok - blank
    print(
        json.dumps(
            {
                "total": len(ordered),
                "ok": ok,
                "blank": blank,
                "failed": errors,
                "pending_after_run": len(rows) - len(ordered),
                "processed_this_run": len(pending),
                "out_jsonl": str(out_jsonl.resolve()),
                "out_md": str(out_md.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
