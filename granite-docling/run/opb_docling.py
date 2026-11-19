"""Score granite-docling's DocTags on OmniParseBench with the benchmark's own scorer.

Runs INSIDE scripts/jail.sh: it imports docling_core and omni_parse_bench, code nobody here wrote.
OPB 1.6.0 has no docling adapter, so the outputs are made the way its indra-ocr adapter makes them:

  html    DocTags -> docling-core DoclingDocument -> export_to_markdown() -> the benchmark's
          markdown.to_html (the indra-ocr adapter's own `parse`). --html docling instead takes
          docling-core's export_to_html() (tables keep their row and column spans).
  blocks  every DoclingDocument item with a box, in reading order: its label through LABELS below
          (docling's item labels onto the benchmark's views.LABELS), its box divided by the page size
          (layout.block), its text.

The page image is the grayscale prerender Indra read (indra-browser/run/prerender.py), so a page's
size here is the size docling saw. One JSON per page in OUT/pages, in opb_eval.py's format (the
`lines` field feeds unit_tests.summarize and indra-browser/run/analyze.py); OUT/summary.json.

    sh scripts/jail.sh -C /tmp/jail/opb/eval -e PYTHONPATH=/tmp/jail/opb/.deps12 -- python3.12 \
        run/opb_docling.py --gen gen/vllm-float32 --gray gray --ds /tmp/opb-data/ds --out out/docling
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

# docling-core DocItemLabel values -> omni_parse_bench.views.LABELS.
LABELS = {
    "title": "heading", "section_header": "heading",
    "text": "text", "paragraph": "text", "reference": "text", "handwritten_text": "text",
    "document_index": "text", "checkbox_selected": "text", "checkbox_unselected": "text",
    "empty_value": "text", "grading_scale": "text", "marker": "text",
    "field_heading": "heading", "field_item": "form", "field_key": "form", "field_value": "form",
    "field_hint": "form", "field_region": "form", "form": "form", "key_value_region": "form",
    "list_item": "list", "caption": "caption", "footnote": "footnote",
    "page_header": "page_furniture", "page_footer": "page_furniture",
    "table": "table", "picture": "figure", "chart": "figure",
    "formula": "equation", "code": "code",
}
END = "<|end_of_text|>"


def convert(doctags: str, gray_path: Path):
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    from PIL import Image

    img = Image.open(gray_path).convert("RGB")
    text = doctags.replace(END, "").strip()
    dt = DocTagsDocument.from_doctags_and_image_pairs([text], [img])
    doc = DoclingDocument.load_from_doctags(dt, document_name="page")
    return doc, img.size


def blocks_of(doc, size):
    from omni_parse_bench.harness import layout

    w, h = size
    out = []
    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        label = str(getattr(item.label, "value", item.label))
        if label not in LABELS:
            raise ValueError(f"docling label {label!r} has no entry in LABELS")
        p = prov[0]
        ph = doc.pages[p.page_no].size.height if p.page_no in doc.pages else h
        bb = p.bbox.to_top_left_origin(page_height=ph)
        if label == "table":
            txt = " ".join(" ".join(c.text for c in row) for row in item.data.grid)
        else:
            txt = getattr(item, "text", "") or ""
        out.append(layout.block(LABELS[label], (bb.l, bb.t, bb.r, bb.b), (w, h), " ".join(txt.split())))
    return out


def page(row, gen_path: Path, gray_path: Path, gt_path: Path, html_from: str, kind: str = "doctags"):
    from omni_parse_bench import data, markdown
    from omni_parse_bench.scorers import unit_tests
    from omni_parse_bench.scorers.unit_tests import tests as T

    pg = data.Page(row["page_id"], row["document"], row["suite"], Path(row["doc_path"]), Path(gt_path),
                   tuple(row["outputs"]))
    tests = T.read(Path(gt_path))
    rec = {"page_id": row["page_id"], "suite": row["suite"]}
    t0 = time.perf_counter()
    try:
        g = json.loads(gen_path.read_text())
        rec["gen"] = {k: g.get(k) for k in ("gen_tokens", "prompt_tokens", "finish", "batch_seconds", "md5")}
        outputs = {}
        if kind == "markdown":  # a model that answers in Markdown (granite-vision): html only, no blocks
            md = g["doctags"].replace(END, "").strip()
            if "html" in pg.outputs:
                outputs["html"] = markdown.to_html(md)
        else:
            doc, size = convert(g["doctags"], gray_path)
            md = doc.export_to_markdown()
            if "html" in pg.outputs:
                outputs["html"] = markdown.to_html(md) if html_from == "md" else doc.export_to_html()
            if "blocks" in pg.outputs:
                outputs["blocks"] = blocks_of(doc, size)
        rec["markdown"] = md
        verdicts = unit_tests.score(outputs, tests)
    except Exception as e:  # noqa: BLE001 - a failed page fails all its tests
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        verdicts = [unit_tests.Verdict(t.id, "fail", "page failed") for t in tests]
    rec["lines"] = unit_tests.lines(verdicts, tests, pg)
    rec["ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return rec


def _job(args):
    row, gen, gray, gt, out, html_from, kind = args
    rec = page(row, gen, gray, gt, html_from, kind)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    tmp.rename(out)
    return "error" not in rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True, help="folder of <page_id>.json from kaggle_gen.py")
    ap.add_argument("--gray", type=Path, required=True)
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=Path, help="file of page ids; default every page in --gen")
    ap.add_argument("--html", choices=("md", "docling"), default="md")
    ap.add_argument("--kind", choices=("doctags", "markdown"), default="doctags",
                    help="what the model wrote: DocTags (docling) or Markdown (granite-vision)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    rows = pq.read_table(a.ds / "manifest.parquet").to_pylist()
    want = set(a.pages.read_text().split()) if a.pages else {p.stem for p in a.gen.glob("*.json")}
    rows = [r for r in rows if r["page_id"] in want]
    (a.out / "pages").mkdir(parents=True, exist_ok=True)
    jobs = [(r, a.gen / f"{r['page_id']}.json", a.gray / f"{r['page_id']}.png", a.ds / r["gt_path"],
             a.out / "pages" / f"{r['page_id']}.json", a.html, a.kind) for r in rows]
    todo = [j for j in jobs if not j[4].exists()]
    print(f"{len(jobs)} pages, {len(todo)} to score", flush=True)
    if todo:
        from multiprocessing import get_context

        with get_context("spawn").Pool(a.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_job, todo, chunksize=4), 1):
                if i % 200 == 0:
                    print(f"{i}/{len(todo)}", flush=True)
    from omni_parse_bench.scorers import unit_tests

    lines, failed, missing = [], 0, 0
    for j in jobs:
        rec = json.loads(j[4].read_text())
        lines += rec["lines"]
        failed += "error" in rec
        missing += "gen" not in rec
    s = unit_tests.summarize(lines)
    keep = {k: s[k] for k in ("score", "families", "per_suite") if k in s}
    keep |= {"pages": len(jobs), "failed_pages": failed, "pages_without_generation": missing,
             "gen": str(a.gen), "html_from": a.html, "kind": a.kind}
    (a.out / "summary.json").write_text(json.dumps(keep, indent=1))
    print(json.dumps({k: keep[k] for k in ("score", "pages", "failed_pages")}), flush=True)
    print({f: round(v["score"], 4) for f, v in keep["families"].items()})


if __name__ == "__main__":
    sys.exit(main())
