#!/usr/bin/env python3
"""Render results.json + results_perquery.npz into the markdown tables RESULTS.md
embeds (between <!-- tables:start --> and <!-- tables:end -->), with paired
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


def main():
    results = json.loads((HERE / "results.json").read_text()); perquery = dict(np.load(HERE / "results_perquery.npz"))
    block = render(results, perquery)
    if "--stdout" in sys.argv:
        print(block); return
    p = HERE / "RESULTS.md"; s = p.read_text()
    a, b = s.index("<!-- tables:start -->"), s.index("<!-- tables:end -->")
    p.write_text(s[:a] + "<!-- tables:start -->\n" + block + "\n" + s[b:])
    print("RESULTS.md tables refreshed")


if __name__ == "__main__":
    main()
