"""Nemotron Parse 2.0 raw outputs (run/kaggle_eval.py's <arm>/<page>.json) -> the page JSONs
ocria-pack/run/grade.py reads, mapped as Omni Parse Bench's own Nemotron 3.5 Parse adapter maps its client's
document (omni_parse_bench/harness/providers/nemotron_parse.py, 1.8.0):

- blocks in model order, class -> benchmark label by that adapter's LABELS, boxes from the model's
  normalized 1664x2048 canvas back to page pixels (the model card's transform_bbox_to_original);
- markdown: each block's content in order; Table content (LaTeX) -> HTML with the model repo's
  latex2html.latex_table_to_html; Title as "# ", Section-header as "## "; Picture and Chart content as an
  image's alt text, as the benchmark does for every system.

Runs inside scripts/jail.sh: it imports the model repo's postprocessing.py and latex2html.py (NVIDIA code).

    python3 np_pages.py --vendor DIR_WITH_postprocessing.py --run kaggle_out/ARM --out runs/np_ARM
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABELS = {
    "Text": "text", "TOC": "text", "Bibliography": "text",
    "Title": "heading", "Section-header": "heading",
    "Page-header": "page_furniture", "Page-footer": "page_furniture",
    "List-item": "list", "Caption": "caption", "Footnote": "footnote", "Formula": "equation",
    "Code": "code", "Table": "table",
    "Picture": "figure", "Chart": "figure",
}
PICTURES = {"Picture", "Chart"}


def convert(rec: dict) -> dict:
    from postprocessing import extract_classes_bboxes, remove_nemotron_formatting, transform_bbox_to_original
    from latex2html import latex_table_to_html

    W, H = rec["width"], rec["height"]
    classes, boxes, texts = extract_classes_bboxes(rec["raw"])
    parts, blocks = [], []
    for cls, bb, txt in zip(classes, boxes, texts):
        content = remove_nemotron_formatting(txt).strip()
        if cls == "Table" and content:
            try:
                content = latex_table_to_html(content)
            except Exception:  # noqa: BLE001 - a table the converter can't read stays as the model wrote it
                pass
        x0, y0, x1, y1 = transform_bbox_to_original(bb, W, H)
        label = LABELS.get(cls)
        if label and x1 > x0 and y1 > y0:
            blocks.append({"kind": label, "bbox": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
                           "text": content})
        if not content:
            continue
        if cls in PICTURES:
            from omni_parse_bench import markdown
            parts.append(markdown.alt(markdown.to_html(content)))
        elif cls == "Title" and not content.startswith("#"):
            parts.append(f"# {content}")
        elif cls == "Section-header" and not content.startswith("#"):
            parts.append(f"## {content}")
        else:
            parts.append(content)
    return {"page_id": rec["page_id"], "markdown": "\n\n".join(parts), "width": W, "height": H, "blocks": blocks,
            "tokens": len(rec["tokens"]), "hit_limit": rec.get("hit_limit", False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.vendor)
    (a.out / "pages").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(a.run.glob("*.json")):
        rec = json.loads(f.read_text())
        (a.out / "pages" / f.name).write_text(json.dumps(convert(rec)))
        n += 1
    print(f"{n} pages -> {a.out}")


if __name__ == "__main__":
    main()
