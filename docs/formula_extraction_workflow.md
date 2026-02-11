# MathType 公式提取流程

这套流程针对 `docx/pptx` 中的 `Equation.DSMT4/DSMT7`（MathType OLE）对象。

## 1) 导出公式图像

```powershell
python scripts/export_mathtype_images.py
```

产物：

- `artifacts/formula_extract/formula_images_manifest.jsonl`
- `artifacts/formula_extract/word_export.files/*.png`
- `artifacts/formula_extract/ppt_crops/*.png`

## 2) 公式 OCR（pix2tex）

```powershell
.\.venv_formula\Scripts\python scripts/ocr_formulas_pix2tex.py
```

支持参数：

- `--limit N`：分批跑，最多处理 `N` 条待处理公式
- `--overwrite`：忽略已有结果，重算全部

产物：

- `artifacts/formula_extract/formulas_ocr.jsonl`
- `artifacts/formula_extract/formulas_ocr_preview.md`

## 当前状态

- 总公式图：`166`
- OCR 成功：`165`
- 空图：`1`（`id = ppt_s09_sh002`，来自 PPT 第 9 页一个空白 OLE 预览）
- 真实异常：`0`

说明：`blank_image` 是图像本身为空白（纯白），不是 OCR 崩溃。  
