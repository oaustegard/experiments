#!/usr/bin/env python3
"""'One bit beats two' is embedder-specific — chart it.

Parses reconcile.log (Jina) + reconcile_specter2.log (SPECTER2), plots
R@10-vs-fp32 across bit-width for independent codebooks. SPECTER2 reverses
(1>2), Jina is monotone. Same harness both — the difference is the embedder.
"""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BITS = [1, 2, 3, 4, 8]


def parse(logfile, label_match):
    """Return dict {bits: r10} from an 'indep' row in the log."""
    for line in (HERE / logfile).read_text().splitlines():
        if line.strip().startswith(label_match):
            nums = re.findall(r"\d\.\d{3}", line)
            if len(nums) >= 5:
                return dict(zip(BITS, [float(x) for x in nums[:5]]))
    raise ValueError(f"no '{label_match}' row in {logfile}")


def main():
    # reconcile.log has two 'indep' rows (muninn, nfcorpus); take nfcorpus (last)
    rows = [l for l in (HERE / "reconcile.log").read_text().splitlines() if l.strip().startswith("indep")]
    jina = dict(zip(BITS, [float(x) for x in re.findall(r"\d\.\d{3}", rows[-1])[:5]]))
    spec = parse("reconcile_specter2.log", "indep")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(BITS, [spec[b] for b in BITS], "s-", color="#d62728", ms=9, lw=2,
            label="SPECTER2 (specialized) — reverses: 1>2")
    ax.plot(BITS, [jina[b] for b in BITS], "o-", color="#1f77b4", ms=9, lw=2,
            label="Jina v5-nano (general) — monotone")
    for b in BITS:
        ax.annotate(f"{spec[b]:.2f}", (b, spec[b]), fontsize=8, xytext=(0, -14), textcoords="offset points", ha="center", color="#d62728")
        ax.annotate(f"{jina[b]:.2f}", (b, jina[b]), fontsize=8, xytext=(0, 7), textcoords="offset points", ha="center", color="#1f77b4")
    ax.set_xticks(BITS); ax.set_xlabel("bits per coordinate (remex Lloyd-Max, d=768)")
    ax.set_ylabel("Recall@10 vs fp32-kNN")
    ax.set_title("'One bit beats two' is embedder-specific\n(identical harness; only the embedder differs)")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right"); ax.set_ylim(0.2, 1.02)
    fig.tight_layout()
    out = HERE / "reversal.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}\n  SPECTER2 {spec}\n  Jina     {jina}")


if __name__ == "__main__":
    raise SystemExit(main())
