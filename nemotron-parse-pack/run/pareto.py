"""Pareto chart: first-run download size against Omni Parse Bench text score for the six OCR models on
anything-to-text, all scored on the same 250-page sample (run/sample250.txt) with omni-parse-bench 1.8.0.

    python3 run/pareto.py results/pareto.json results/pareto.png

results/pareto.json: {"models": [{"name", "mb", "text", "headline", "note"}]}. Download sizes are what the page
fetches on first use for an English page (compressed bytes on the wire).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

BLUE, ORANGE = "#257de0", "#f47230"   # dataviz reference palette, slots 1 and 2
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"


def frontier(ms):
    """Models no other model beats on both size (smaller) and score (higher)."""
    out = []
    for m in ms:
        if not any(o is not m and o["mb"] <= m["mb"] and o["text"] >= m["text"] and
                   (o["mb"] < m["mb"] or o["text"] > m["text"]) for o in ms):
            out.append(m)
    return sorted(out, key=lambda m: m["mb"])


def main(src, dst):
    ms = json.load(open(src))["models"]
    fr = frontier(ms)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.set_xscale("log")
    ax.step([m["mb"] for m in fr], [m["text"] for m in fr], where="post", color=BLUE, lw=2, alpha=.35, zorder=1)
    for m in ms:
        on = m in fr
        ax.scatter(m["mb"], m["text"], s=90, color=BLUE if on else "#9bbfe9", edgecolor="white", linewidth=2, zorder=3)
        dx, ha = (1.12, "left") if m["mb"] < 300 else (0.9, "right")
        ax.annotate(f'{m["name"]}\n{m["mb"]:.1f} MB · {m["text"]:.1f}', (m["mb"], m["text"]),
                    xytext=(m["mb"] * dx, m["text"] + 3 + m.get("dy", 0)), ha=ha, va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("First-run download (MB, log scale)", color=INK2)
    ax.set_ylabel("Omni Parse Bench text score (% of tests passed)", color=INK2)
    ax.set_ylim(0, 100)
    ax.set_xlim(1, 2000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, which="major", color=GRID, lw=.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.set_title("OCR models on anything-to-text: download size against accuracy", loc="left", color=INK, fontsize=12)
    fig.text(0.01, 0.01, "250 Omni Parse Bench pages, same pages for every model. Line: the models nothing smaller beats.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
