#!/usr/bin/env python
"""公式图片导出脚本（中文注释版）。

功能：
- 从 Word/PPT 中提取 MathType 公式对象对应的位图；
- 构建统一 manifest，供后续 OCR 与核对流程使用。

说明：
- 兼容以 OLE 形式嵌入的 MathType 对象（如 `Equation.DSMT4/7`）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from PIL import Image
import win32com.client


def natural_key(path: Path) -> tuple[int, str]:
    """按文件名中的数字自然排序。"""
    m = re.search(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else -1, path.name)


def strip_html(text: str) -> str:
    """粗粒度移除 HTML 标签并压缩空白。"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser()
    p.add_argument("--docx", default="data/raw_documents/镁单晶塑性变形行为的晶体塑性-孪晶耦合相场模拟.docx")
    p.add_argument("--pptx", default="data/raw_documents/260113.pptx")
    p.add_argument("--out-dir", default="artifacts/formula_extract")
    return p.parse_args()


def export_word_images(docx_path: Path, out_dir: Path) -> list[dict]:
    """导出 Word 中的公式图片并收集上下文信息。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "word_export.html"

    # 1) 通过 COM 打开 Word，并导出过滤后 HTML（会生成 image*.png 资源）。
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = word.Documents.Open(str(docx_path.resolve()), ReadOnly=True)
    # wdFormatFilteredHTML = 10
    doc.SaveAs2(str(html_path.resolve()), FileFormat=10)
    doc.Close(False)
    word.Quit()

    images_dir = out_dir / "word_export.files"
    # 2) 收集导出的图片文件，并按自然顺序排列。
    imgs = sorted(images_dir.glob("image*.png"), key=natural_key)
    html = html_path.read_text(encoding="gb2312", errors="ignore")
    html_l = html.lower()

    entries: list[dict] = []
    for idx, img in enumerate(imgs, start=1):
        # 3) 在 HTML 中定位图片引用点，提取附近文本作为上下文。
        token = str(img.relative_to(out_dir)).replace("\\", "/").lower()
        pos = html_l.find(token)
        context = ""
        if pos >= 0:
            lo = max(0, pos - 220)
            hi = min(len(html), pos + 220)
            context = strip_html(html[lo:hi])
        with Image.open(img) as im:
            w, h = im.size
        entries.append(
            {
                "id": f"word_{idx:03d}",
                "source": "word",
                "docx_path": str(docx_path.resolve()),
                "image_path": str(img.resolve()),
                "width_px": w,
                "height_px": h,
                "context": context,
            }
        )
    return entries


def collect_slide_text(slide) -> str:
    """收集单页 PPT 的文本上下文。"""
    texts: list[str] = []
    for i in range(1, slide.Shapes.Count + 1):
        sh = slide.Shapes(i)
        try:
            if sh.HasTextFrame and sh.TextFrame.HasText:
                txt = sh.TextFrame.TextRange.Text
                if txt:
                    texts.append(str(txt))
        except Exception:
            continue
    return re.sub(r"\s+", " ", " ".join(texts)).strip()


def iter_equation_shapes(slide) -> Iterable[tuple[int, object]]:
    """遍历当前幻灯片中的公式 OLE 形状。"""
    for i in range(1, slide.Shapes.Count + 1):
        sh = slide.Shapes(i)
        try:
            prog = str(sh.OLEFormat.ProgID)
        except Exception:
            continue
        if "Equation" in prog:
            yield i, sh


def export_ppt_crops(pptx_path: Path, out_dir: Path) -> list[dict]:
    """导出 PPT 中公式区域的裁剪图。"""
    slides_dir = out_dir / "ppt_slides"
    crops_dir = out_dir / "ppt_crops"
    slides_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # 1) 先将整份 PPT 导出为逐页 PNG，后续按公式框坐标裁剪。
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    ppt.Visible = True
    pres = ppt.Presentations.Open(str(pptx_path.resolve()), WithWindow=False)
    # ppSaveAsPNG = 18
    pres.SaveAs(str(slides_dir.resolve()), 18)

    slide_w = float(pres.PageSetup.SlideWidth)
    slide_h = float(pres.PageSetup.SlideHeight)

    slide_imgs = {}
    for p in slides_dir.glob("*.PNG"):
        m = re.search(r"(\d+)", p.stem)
        if m:
            slide_imgs[int(m.group(1))] = p

    entries: list[dict] = []
    for sidx in range(1, pres.Slides.Count + 1):
        slide = pres.Slides(sidx)
        slide_img_path = slide_imgs.get(sidx)
        if not slide_img_path:
            continue
        with Image.open(slide_img_path) as simg:
            # 2) 将 PPT 点单位坐标映射到导出位图像素坐标。
            sx = simg.width / slide_w
            sy = simg.height / slide_h
            slide_text = collect_slide_text(slide)
            eq_count = 0
            for shape_idx, sh in iter_equation_shapes(slide):
                eq_count += 1
                left = int(round(float(sh.Left) * sx))
                top = int(round(float(sh.Top) * sy))
                width = int(round(float(sh.Width) * sx))
                height = int(round(float(sh.Height) * sy))
                pad = 4
                x0 = max(0, left - pad)
                y0 = max(0, top - pad)
                x1 = min(simg.width, left + width + pad)
                y1 = min(simg.height, top + height + pad)
                crop = simg.crop((x0, y0, x1, y1))

                out_name = f"slide{sidx:02d}_shape{shape_idx:03d}.png"
                out_path = crops_dir / out_name
                crop.save(out_path)
                entries.append(
                    {
                        "id": f"ppt_s{sidx:02d}_sh{shape_idx:03d}",
                        "source": "ppt",
                        "pptx_path": str(pptx_path.resolve()),
                        "slide": sidx,
                        "shape_index": shape_idx,
                        "image_path": str(out_path.resolve()),
                        "width_px": crop.width,
                        "height_px": crop.height,
                        "context": slide_text,
                    }
                )
            if eq_count == 0:
                continue

    pres.Close()
    ppt.Quit()
    return entries


def main() -> None:
    """脚本主流程。"""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 分别汇总 Word 与 PPT 公式图记录。
    all_entries: list[dict] = []
    all_entries.extend(export_word_images(Path(args.docx), out_dir))
    all_entries.extend(export_ppt_crops(Path(args.pptx), out_dir))

    # 2) 输出统一 manifest，作为后续 OCR 的唯一输入清单。
    manifest = out_dir / "formula_images_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in all_entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3) 输出摘要统计，便于快速确认导出量是否符合预期。
    summary = {
        "manifest": str(manifest.resolve()),
        "total_images": len(all_entries),
        "word_images": sum(1 for x in all_entries if x["source"] == "word"),
        "ppt_images": sum(1 for x in all_entries if x["source"] == "ppt"),
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
