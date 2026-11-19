"""indra-ocr's own page outputs, without the benchmark's scorer: the reference that
run/compare_pages.py checks the browser against, and that arms are compared on.

Runs INSIDE scripts/jail.sh (it imports indra_ocr). Each page is the prerendered
grayscale image (prerender.py), read as opb_eval.py reads it (OrtOCR, headers and
footers kept, math off unless --math). One JSON per page in OUT/pages with the same
"raw" record as opb_eval.py, plus, with --math, every line the classifier flagged:
its index, the math reading and its mean token probability (used or not).

    sh scripts/jail.sh -C /tmp/indra-eval -e PYTHONPATH=/tmp/jail/indra/.deps -- \
        python3 run/ref_pages.py --models arms/mix8m16 --gray gray --pages pages.txt --out out/ref-m16 --math
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_OCR = None
_MATH: list = []


def _init(models: str, math: bool):
    global _OCR
    from indra_ocr.pipeline import OrtOCR

    _OCR = OrtOCR(models, threads=1, furniture=True, math=math)
    if _OCR.math is not None:
        m = _OCR.math
        flags, read = m.flags, m.read

        def flags_rec(crops, *a, **k):
            f = flags(crops, *a, **k)
            _MATH.append({"flags": [j for j, v in enumerate(f) if v]})
            return f

        def read_rec(gray, boxes, *a, **k):
            r = read(gray, boxes, *a, **k)
            _MATH[-1]["reads"] = [[t, c] for t, c in r]
            return r

        m.flags, m.read = flags_rec, read_rec


def _page(job):
    pid, gray_path, out_path = job
    import cv2

    rec = {"page_id": pid}
    _MATH.clear()
    t0 = time.perf_counter()
    try:
        gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        pg = _OCR(gray, True)
        rec["raw"] = {"markdown": pg.markdown, "text": pg.text, "blocks": pg.blocks,
                      "rotation": pg.rotation, "skew": pg.skew,
                      "image_size": [int(gray.shape[1]), int(gray.shape[0])],
                      "lines": [[ln.box, ln.text, ln.conf] for ln in pg.lines],
                      "timings": pg.timings, "vertical": pg.vertical}
        rec["math"] = list(_MATH)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    rec["ms"] = round(1000 * (time.perf_counter() - t0), 1)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    tmp.rename(out_path)
    return "error" not in rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--gray", type=Path, required=True)
    ap.add_argument("--pages", type=Path, required=True, help="file of page ids, one per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--math", action="store_true")
    a = ap.parse_args()
    ids = a.pages.read_text().split()
    (a.out / "pages").mkdir(parents=True, exist_ok=True)
    jobs = [(p, a.gray / f"{p}.png", a.out / "pages" / f"{p}.json") for p in ids]
    todo = [j for j in jobs if not j[2].exists()]
    print(f"{len(jobs)} pages, {len(todo)} to run", flush=True)
    t0 = time.time()
    if todo:
        from multiprocessing import get_context

        with get_context("spawn").Pool(a.workers, _init, (a.models, a.math)) as pool:
            for i, ok in enumerate(pool.imap_unordered(_page, todo, chunksize=1), 1):
                if i % 10 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    recs = [json.loads(j[2].read_text()) for j in jobs]
    summ = {"pages": len(recs), "failed": sum("error" in r for r in recs), "models": a.models, "math": a.math,
            "ms_median": sorted(r["ms"] for r in recs)[len(recs) // 2] if recs else None}
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    sys.exit(main())
