#!/usr/bin/env python3
"""Render results*.json + results*_perquery.npz into the markdown tables RESULTS.md
embeds (between <!-- tables:start --> and <!-- tables:end --> for SciFact;
<!-- tables:docvqa:start --> etc. for the ViDoRe corpora; pass the corpus name), with paired
bootstrap CIs against each head's fp32 full-width reference. recheck.py
regenerates the same block and diffs it, so prose and data cannot drift."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from neomme_quant import paired_bootstrap  # noqa: E402

REF = {"dense": "dense fp32 d=1024", "late": "late fp32", "pipeline": "pipeline fp32 d=1024 -> late fp32 @100"}


def fmt_bytes(b):
    return f"{b/1024:.1f} KB" if b >= 1024 else f"{b:.0f} B"


def render(results: dict, perquery: dict) -> str:
    out = []
    for group, title in (("dense", "Dense head over 5,183 documents"), ("late", "Late-interaction head scored with MeanMaxSim"), ("pipeline", "Dense top-100 candidates reranked by late interaction")):
        rows = [(k, v) for k, v in results.items() if v["group"] == group]
        if not rows:
            continue
        ref = perquery[REF[group]]
        out.append(f"### {title}\n")
        out.append("| arm | bytes/doc | nDCG@10 | Δ vs fp32 [95% CI] | wins/losses | R@10 | R@100 | top-10 overlap w/ fp32 |")
        out.append("|---|---:|---:|---|---:|---:|---:|---:|")
        for k, v in sorted(rows, key=lambda kv: -kv[1]["bytes_per_doc"]):
            d = perquery[k] - ref
            m, lo, hi, w, l = paired_bootstrap(d)
            delta = "—" if k == REF[group] else f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            sig = "" if k == REF[group] or (lo <= 0 <= hi) else " *"
            fid = "—" if v.get("fid10") is None else f"{v['fid10']:.3f}"
            name = k.replace("pipeline ", "").replace(" @100", "")
            out.append(f"| {name} | {fmt_bytes(v['bytes_per_doc'])} | {v['ndcg10']:.4f}{sig} | {delta} | {w}/{l} | {v['r10']:.3f} | {v['r100']:.3f} | {fid} |")
        out.append("")
    out.append("`*` = 95% paired-bootstrap CI on ΔnDCG@10 excludes zero (300 queries, 5,000 resamples). Wins/losses count queries whose nDCG@10 moved vs the reference; ties omitted.")
    return "\n".join(out)


STEMS = {"scifact": ("results", "tables"), "docvqa": ("results_docvqa", "tables:docvqa"), "shift": ("results_shift", "tables:shift")}


def load(corpus):
    stem, marker = STEMS[corpus]
    return json.loads((HERE / f"{stem}.json").read_text()), dict(np.load(HERE / f"{stem}_perquery.npz")), marker


def main():
    corpus = next((a for a in sys.argv[1:] if a in STEMS), "scifact")
    results, perquery, marker = load(corpus)
    n_q = len(next(iter(perquery.values())))
    block = render(results, perquery).replace("(300 queries, 5,000 resamples)", f"({n_q:,} queries, 5,000 resamples)")
    if "--stdout" in sys.argv:
        print(block); return
    p = HERE / "RESULTS.md"; s = p.read_text()
    a, b = s.index(f"<!-- {marker}:start -->"), s.index(f"<!-- {marker}:end -->")
    p.write_text(s[:a] + f"<!-- {marker}:start -->\n" + block + "\n" + s[b:])
    print(f"RESULTS.md {marker} refreshed")


if __name__ == "__main__":
    main()
