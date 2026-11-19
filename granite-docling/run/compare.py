"""Systems on the same OmniParseBench pages: headline and family scores, each against a base with a
paired 95% bootstrap interval (documents resampled, 1,000 draws, seed 0), as indra-browser/run/analyze.py
computes them, but restricted to the pages every run has (no filling in from the base).

    python3 run/compare.py OUT BASE RUN [RUN ...] [--pages ids.txt] [--json results/x.json]

Each RUN is a folder OUT/RUN/pages/*.json with a `lines` list (opb_eval.py or opb_docling.py).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILY = {"present": "text", "order": "text", "repeat": "text", "page_near_empty": "text",
          "table_cell": "tables", "layout_kind": "layout"}


def load(run: Path):
    pages = {}
    for f in (run / "pages").glob("*.json"):
        r = json.loads(f.read_text())
        pages[r["page_id"]] = r
    return pages


def tests_of(pages, keep):
    t = {}
    for p in keep:
        for ln in pages[p]["lines"]:
            t[ln["test_id"]] = (ln["document"], FAMILY[ln["test_type"]], ln["result"] == "pass")
    return t


def headline(ids, tests, w=None):
    fam = defaultdict(lambda: [0.0, 0.0])
    for i in ids:
        d, f, ok = tests[i]
        x = 1.0 if w is None else w[d]
        fam[f][0] += x * ok
        fam[f][1] += x
    sc = {f: a / b for f, (a, b) in fam.items() if b}
    return float(np.mean(list(sc.values()))), sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("base")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pages", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    loaded = {r: load(a.out / r) for r in [a.base, *a.runs]}
    keep = set.intersection(*(set(p) for p in loaded.values()))
    if a.pages:
        keep &= set(a.pages.read_text().split())
    keep = sorted(keep)
    bt = tests_of(loaded[a.base], keep)
    ids = sorted(bt)
    docs = sorted({bt[i][0] for i in ids})
    rng = np.random.default_rng(0)
    weights = []
    for _ in range(a.boot):
        c = np.bincount(rng.choice(len(docs), len(docs)), minlength=len(docs))
        weights.append({docs[i]: float(c[i]) for i in range(len(docs))})
    bh, bf = headline(ids, bt)
    bboot = [headline(ids, bt, w)[0] for w in weights]
    res = {"pages": len(keep), "documents": len(docs), "tests": len(ids), "base": a.base,
           "runs": {a.base: {"headline": bh, "families": bf}}}
    print(f"{len(keep)} pages, {len(docs)} documents, {len(ids)} tests")
    print(f"{a.base}: {100 * bh:.2f} " + " ".join(f"{k} {100 * v:.2f}" for k, v in sorted(bf.items())))
    for r in a.runs:
        t = tests_of(loaded[r], keep)
        if set(t) != set(bt):
            raise SystemExit(f"{r}: its tests differ from {a.base}'s on the same pages")
        h, f = headline(ids, t)
        d = np.array([headline(ids, t, w)[0] - b for w, b in zip(weights, bboot)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        failed = sum("error" in loaded[r][p] for p in keep)
        res["runs"][r] = {"headline": h, "families": f, "delta": h - bh, "ci95": [float(lo), float(hi)],
                          "failed_pages": failed}
        print(f"{r}: {100 * h:.2f} ({100 * (h - bh):+.2f}, 95% CI {100 * lo:+.2f} to {100 * hi:+.2f}) "
              + " ".join(f"{k} {100 * v:.2f}" for k, v in sorted(f.items())) + f"; {failed} failed pages")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
