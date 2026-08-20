#!/usr/bin/env python3
"""Paired comparison of two arms from two `eval_dense.py` result files.

`eval_dense.py` tests every arm against the BM25 baseline *in its own run*, so a
page-level arm is scored against page-level BM25. The number issue #48 asks about
is different: the shipped chunk-level BM25 at 0.262 sources. This pairs any two
arms across files by query text, so the comparison stays per-query even when the
corpus granularity differs.

    python3 compare.py results_dense.json bm25 results_dense_page.json \\
        wsum0.5:bm25+bekko-a8m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dense import mcnemar  # noqa: E402

METRICS = ("gold_at_top1", "gold_in_topk", "gold_in_sources")


def pick(path: Path, arm: str) -> dict:
    data = json.loads(path.read_text())
    for entry in data["arms"]:
        if entry["arm"] == arm:
            return entry
    raise SystemExit(f"{path.name}: no arm {arm!r}; have "
                     + ", ".join(e["arm"] for e in data["arms"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a", type=Path)
    ap.add_argument("arm_a")
    ap.add_argument("file_b", type=Path)
    ap.add_argument("arm_b")
    a = ap.parse_args()

    A, B = pick(a.file_a, a.arm_a), pick(a.file_b, a.arm_b)
    rows_a = {r["nl"]: r for r in A["rows"] if not r["names_utility"]}
    rows_b = {r["nl"]: r for r in B["rows"] if not r["names_utility"]}
    shared = [q for q in rows_a if q in rows_b]
    print(f"{len(shared)} shared leak-free queries "
          f"({len(rows_a)} vs {len(rows_b)} available)\n")
    print(f"{'metric':<16}{a.arm_a[:22]:>24}{a.arm_b[:22]:>24}"
          f"{'wins A':>8}{'wins B':>8}{'p':>8}")
    for m in METRICS:
        va = [rows_a[q][m] for q in shared]
        vb = [rows_b[q][m] for q in shared]
        t = mcnemar(va, vb)
        print(f"{m:<16}{sum(va) / len(va):>24.3f}{sum(vb) / len(vb):>24.3f}"
              f"{t['wins_a']:>8}{t['wins_b']:>8}{t['p']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
