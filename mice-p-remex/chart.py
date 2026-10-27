"""Pareto chart: bytes per stored token against pooled nDCG@10 change.

    python3 chart.py   -> results/pareto.png (1200x628, the link-card shape)

Reads results/summary.json (pooled over every query the arms share).
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "results" / "summary.json").read_text())["pooled"]
N = S["fp32"]["n"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # reference palette slots 1-3
BYTES = {"plaid-2": 36, "plaid-4": 68, "plaid-n-2": 36, "plaid-rx-1": 22, "plaid-rx-2": 38,
         "remex-c-2": 34, "fp32": 512}
SERIES = [  # (arms, colour, legend label)
    (["plaid-2", "plaid-4"], BLUE, "FastPLAID as MICE-P ships it"),
    (["plaid-n-2"], ORANGE, "FastPLAID, vectors normalised first"),
    (["plaid-rx-1", "plaid-rx-2"], AQUA, "PLAID centroid + remex residual"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, ax = plt.subplots(figsize=(12, 6.28), dpi=100, facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.axhline(0, color=GRID, lw=1, zorder=1)
ax.plot([512], [0], "s", ms=9, color=INK2, mec=SURFACE, mew=2, zorder=3)
ax.annotate("uncompressed fp32", (512, 0), xytext=(6, 12), textcoords="offset points", ha="right",
            color=INK, fontsize=11.5)
ax.axvline(36, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(36, 0.0085, "LateOn's 2-bit PLAID: 36 B/token", ha="center", va="bottom",
        color=INK2, fontsize=11)


NUDGE = {"plaid-2": 0.97, "plaid-n-2": 1.03}  # both are 36 B/token; nudged apart to stay legible


def point(arm):
    r = S[arm]
    return BYTES[arm] * NUDGE.get(arm, 1.0), r["delta"], r["delta"] - r["ci"][0], r["ci"][1] - r["delta"]


for arms, col, label in SERIES:
    xs, ys, lo, hi = zip(*(point(a) for a in arms))
    ax.plot(xs, ys, color=col, lw=2, zorder=2)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=9, color=col, ecolor=col, elinewidth=1.5,
                capsize=0, mec=SURFACE, mew=2, zorder=3, label=label)

x, y, lo, hi = point("remex-c-2")
ax.errorbar([x], [y], yerr=[[lo], [hi]], fmt="D", ms=7, color=MUTED, ecolor=MUTED, elinewidth=1.2,
            mec=SURFACE, mew=2, zorder=3, label="remex centred, no centroids")

labels = {"plaid-2": ("2-bit", (-10, -4), "right"), "plaid-4": ("4-bit (shipped)", (10, -4), "left"),
          "plaid-n-2": ("2-bit", (10, -4), "left"), "plaid-rx-1": ("1-bit", (-10, 6), "right"),
          "plaid-rx-2": ("2-bit", (8, 8), "left")}
for arm, (txt, off, ha) in labels.items():
    x, y, *_ = point(arm)
    ax.annotate(f"{txt}  {y:+.3f}", (x, y), xytext=off, textcoords="offset points", ha=ha,
                va="center", color=INK, fontsize=11.5)

ax.set_xscale("log")
ax.set_xlim(16, 600)
ax.set_xticks([16, 22, 36, 68, 128, 256, 512])
ax.set_xticklabels(["16", "22", "36", "68", "128", "256", "512"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(-0.048, 0.012)
ax.set_xlabel("bytes stored per document token (log scale)", color=INK2)
ax.set_ylabel(f"change in nDCG@10 vs uncompressed\n(pooled, {N} queries, 95% CI)", color=INK2)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=INK2)
ax.text(0.012, 0.035, "better: up and left", transform=ax.transAxes, ha="left", va="bottom",
        color=MUTED, fontsize=11)
ax.legend(loc="lower right", frameon=False, fontsize=11.5, labelcolor=INK)
fig.suptitle("MICE-P document-vector storage: bytes against ranking quality", x=0.065, ha="left",
             y=0.965, fontsize=16, color=INK, weight="bold")
fig.text(0.065, 0.885, "Mice-P-130m reranking SPLADE-v3 top-100 on decontaminated SciFact, NFCorpus, "
         "FiQA, SciDocs", ha="left", color=INK2, fontsize=11.5)
fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.12)
fig.savefig(HERE / "results" / "pareto.png", facecolor=SURFACE)
print("wrote results/pareto.png")
