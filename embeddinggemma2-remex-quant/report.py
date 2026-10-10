#!/usr/bin/env python3
"""Pareto frontier and matched-byte comparisons from results_<corpus>.json.

For each byte budget where both exist: the best arm that keeps all 768 dims vs the
best arm that truncates (MRL d<768), paired bootstrap on per-query nDCG@10. And each
vendor fp32 MRL point vs the best quantized arm at <= its bytes.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "neomme-remex-quant"))
from neomme_quant import paired_bootstrap  # noqa: E402

EXCLUDE = {"seed", "nested"}       # diagnostics, not candidate codecs


def main(corpus):
    res = json.load(open(HERE / f"results_{corpus}.json")); pq = np.load(HERE / f"results_{corpus}_perquery.npz")
    rows = [r for r in res["rows"] if r["family"] not in EXCLUDE]
    print(f"\n## {corpus}  (fp32 d=768 nDCG@10 {rows[0]['ndcg10']:.4f})")
    front, best = [], -1
    for r in sorted(rows, key=lambda r: (r["bytes_per_doc"], -r["ndcg10"])):
        if r["ndcg10"] > best:
            front.append(r); best = r["ndcg10"]
    print("\nPareto frontier (bytes ascending):")
    for r in front:
        print(f"  {r['bytes_per_doc']:>7.0f} B  {r['ndcg10']:.4f}  {r['arm']}")

    def cmp(a, b):
        m, lo, hi, w, l = paired_bootstrap(pq[a["arm"]] - pq[b["arm"]])
        return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]"

    print("\nMatched bytes: best full-width (d=768) vs best truncated (d<768):")
    for B in sorted({r["bytes_per_doc"] for r in rows}):
        full = [r for r in rows if r["bytes_per_doc"] == B and r.get("dim") == 768]
        tr = [r for r in rows if r["bytes_per_doc"] == B and r.get("dim", 768) < 768]
        if full and tr:
            f = max(full, key=lambda r: r["ndcg10"]); t = max(tr, key=lambda r: r["ndcg10"])
            print(f"  {B:>6.0f} B  {f['arm']:34s} {f['ndcg10']:.4f}  vs  {t['arm']:30s} {t['ndcg10']:.4f}   Δ {cmp(f, t)}")

    print("\nVendor MRL fp32 points vs best quantized arm at <= the same bytes:")
    for r in rows:
        if r["family"] == "mrl":
            cand = [c for c in rows if c["family"] not in ("mrl",) and c["bytes_per_doc"] <= r["bytes_per_doc"]]
            c = max(cand, key=lambda c: c["ndcg10"])
            print(f"  {r['arm']:12s} {r['bytes_per_doc']:>6.0f} B {r['ndcg10']:.4f}  vs  {c['arm']:34s} {c['bytes_per_doc']:>5.0f} B {c['ndcg10']:.4f}   Δ {cmp(c, r)}")


if __name__ == "__main__":
    for c in sys.argv[1:] or ["scifact", "nfcorpus"]:
        main(c)
