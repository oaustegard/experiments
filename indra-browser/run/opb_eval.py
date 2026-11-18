"""Indra OCR on OmniParseBench pages, scored by the benchmark's own scorer.

Runs INSIDE scripts/jail.sh: it imports indra_ocr and omni_parse_bench, code
nobody here wrote. Each page is the prerendered grayscale image
(prerender.py), read by the package's OrtOCR as Converter mode="ocr" reads a
page: turned upright, straightened, detected, recognized, ordered, written as
markdown. Settings follow the benchmark's indra-ocr adapter (headers and
footers kept) except math, which is off unless --math.

The page's markdown and blocks go through the adapter's own `parse` (markdown
-> html, block labels and boxes), then `unit_tests.score`. One JSON per page
in OUT/pages (a checkpoint: a page already there is skipped), and
OUT/summary.json from `unit_tests.summarize` over every page's test lines.

    sh scripts/jail.sh -C /tmp/indra-eval -e PYTHONPATH=... -- \
        python3 run/opb_eval.py --models arms/fp32 --gray gray --ds ds --out out/fp32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

_OCR = None
_BEAMS = [0]


def _init(models: str, math: bool, furniture: bool):
    global _OCR
    from indra_ocr import lm
    from indra_ocr.pipeline import OrtOCR

    _OCR = OrtOCR(models, threads=1, furniture=furniture, math=math)
    # Count the lines the n-gram beam search reads (recognize imports it at call time):
    # a page with none reads the same under every n-gram arm.
    inner = lm.beam_search

    def counted(*args, **kw):
        _BEAMS[0] += 1
        return inner(*args, **kw)

    lm.beam_search = counted


def _page(job):
    row, gray_path, out_path, gt_path = job
    import cv2
    from omni_parse_bench import data
    from omni_parse_bench.harness.providers import indra_ocr as P
    from omni_parse_bench.scorers import unit_tests
    from omni_parse_bench.scorers.unit_tests import tests as T

    page = data.Page(row["page_id"], row["document"], row["suite"], Path(row["doc_path"]),
                     Path(gt_path), tuple(row["outputs"]))
    tests = T.read(Path(gt_path))
    t0 = time.perf_counter()
    rec = {"page_id": row["page_id"], "suite": row["suite"]}
    _BEAMS[0] = 0
    try:
        gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        pg = _OCR(gray, True)
        raw = {"markdown": pg.markdown, "text": pg.text, "blocks": pg.blocks,
               "rotation": pg.rotation, "skew": pg.skew,
               "image_size": [int(gray.shape[1]), int(gray.shape[0])]}
        rec["raw"] = raw | {"lines": [[ln.box, ln.text, ln.conf] for ln in pg.lines],
                            "timings": pg.timings, "vertical": pg.vertical}
        outputs, errs = P.parse(raw, SimpleNamespace(outputs=frozenset(page.outputs)))
        pred = {k: outputs[k] for k in page.outputs if k in outputs}
        verdicts = unit_tests.score(pred, tests)
        rec["parse_errors"] = {k: str(v) for k, v in errs.items()}
    except Exception as e:  # noqa: BLE001 - a failed page fails all its tests
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        verdicts = [unit_tests.Verdict(t.id, "fail", "page failed") for t in tests]
    rec["beams"] = _BEAMS[0]
    rec["lines"] = unit_tests.lines(verdicts, tests, page)
    rec["ms"] = round(1000 * (time.perf_counter() - t0), 1)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    tmp.rename(out_path)
    return rec.get("error") is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--gray", type=Path, required=True)
    ap.add_argument("--ds", type=Path, required=True, help="dataset folder: manifest.parquet, gold/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=0, help="a fixed random sample of N pages (seed 0)")
    ap.add_argument("--pages", type=Path, help="file of page ids, one per line")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--math", action="store_true")
    ap.add_argument("--no-furniture", action="store_true")
    ap.add_argument("--base", type=Path, help="run folder whose page results stand in for pages not "
                    "listed in --pages (an arm that only changes some pages)")
    a = ap.parse_args()
    import pyarrow.parquet as pq

    rows = pq.read_table(a.ds / "manifest.parquet").to_pylist()
    if a.pages:
        want = set(a.pages.read_text().split())
        rows = [r for r in rows if r["page_id"] in want]
    elif a.sample:
        rows = random.Random(0).sample(rows, a.sample)
    (a.out / "pages").mkdir(parents=True, exist_ok=True)
    jobs = [(r, a.gray / f"{r['page_id']}.png", a.out / "pages" / f"{r['page_id']}.json",
             a.ds / r["gt_path"]) for r in rows]
    todo = [j for j in jobs if not j[2].exists()]
    print(f"{len(jobs)} pages, {len(todo)} to run", flush=True)
    t0 = time.time()
    if todo:
        from multiprocessing import get_context

        with get_context("spawn").Pool(a.workers, _init, (a.models, a.math, not a.no_furniture)) as pool:
            for i, ok in enumerate(pool.imap_unordered(_page, todo, chunksize=2), 1):
                if i % 50 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    from omni_parse_bench.scorers import unit_tests

    lines, failed, ms = [], 0, []
    if a.base:
        ran = {j[0]["page_id"] for j in jobs}
        allrows = pq.read_table(a.ds / "manifest.parquet").to_pylist()
        jobs += [(r, None, a.base / "pages" / f"{r['page_id']}.json", None)
                 for r in allrows if r["page_id"] not in ran]
    for j in jobs:
        rec = json.loads(j[2].read_text())
        lines += rec["lines"]
        failed += "error" in rec
        ms.append(rec["ms"])
    s = unit_tests.summarize(lines)
    keep = {k: s[k] for k in ("score", "families", "per_suite") if k in s}
    keep |= {"pages": len(jobs), "failed_pages": failed, "models": a.models,
             "math": a.math, "furniture": not a.no_furniture,
             "ms_median": sorted(ms)[len(ms) // 2] if ms else None}
    (a.out / "summary.json").write_text(json.dumps(keep, indent=1))
    print(json.dumps({k: keep[k] for k in ("score", "pages", "failed_pages")}), flush=True)
    print({f: round(v["score"], 4) for f, v in keep["families"].items()})


if __name__ == "__main__":
    sys.exit(main())
