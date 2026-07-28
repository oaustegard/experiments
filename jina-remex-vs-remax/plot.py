#!/usr/bin/env python3
"""Fidelity-vs-fp32 Pareto: remex vs remax as compressed-Jina codecs.

Reads assets/.fidelity.json (score_fidelity.py). Two panels on the NFCorpus
run (n=120, the credible one): bytes/row (log x) vs R@10-vs-fp32-kNN, and
vs Spearman rho. remex and remax families on the same axes.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parent / "assets"
C = {"remex": "#1f77b4", "remax": "#d62728"}
M = {"remex": "o", "remax": "s"}


def fam(lab):
    return "remex" if lab.startswith("remex") else "remax"


def panel(ax, rows, yfn, ylabel, title):
    for f in ("remex", "remax"):
        pts = [(by, yfn(rec, rho), lab) for lab, b, dim, by, rec, rho, recon in rows if fam(lab) == f]
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, M[f] + "-", color=C[f], label=f, ms=8, lw=1.5, zorder=3)
        for by, y, lab in pts:
            ax.annotate(lab.replace("remex ", "").replace("remax ", ""),
                        (by, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log", base=2); ax.set_xlabel("bytes / row (log2)")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(loc="lower right")


def main():
    data = json.loads((A / ".fidelity.json").read_text())
    rows = data.get("nfcorpus") or next(iter(data.values()))
    # rec is a dict keyed by str(k) after json round-trip
    norm = [(lab, b, dim, by, {int(k): v for k, v in rec.items()}, rho, recon)
            for lab, b, dim, by, rec, rho, recon in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panel(axes[0], norm, lambda rec, rho: rec[10], "Recall@10 vs fp32-kNN",
          "Retrieval agreement with fp32")
    panel(axes[1], norm, lambda rec, rho: rho, "Spearman rho (scores vs fp32)",
          "Rank correlation with fp32")
    fig.suptitle("remex (Lloyd-Max scalar) vs remax (1-bit SimHash) — fidelity to fp32 Jina, NFCorpus n=120",
                 fontsize=12)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fidelity.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
