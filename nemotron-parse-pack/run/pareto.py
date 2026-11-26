"""Pareto chart: first-run download size against Omni Parse Bench accuracy for the OCR models on anything-to-text,
all scored on the same 250-page sample (run/sample250.txt) with omni-parse-bench 1.8.0.

    python3 run/pareto.py results/pareto.json results/pareto.png            # side by side
    python3 run/pareto.py results/pareto.json results/pareto_stacked.png --stack   # for a narrow blog column

results/pareto.json: {"models": [{"name", "mb", "text", "headline", "shipped"}]}. Download sizes are what the
page fetches on first use for an English page (compressed bytes on the wire). Models with shipped false are packs
that were measured but are not what the page loads; they are drawn hollow and left off the frontier.
Left panel: the text family (is the page's text there, in order, not repeated). Right panel: the headline (mean
of text, tables and layout), which plain-text OCR models cannot score on beyond text.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

BLUE, PALE = "#257de0", "#9bbfe9"
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
# label offsets in points, per model and panel, so labels clear the points and the frontier line
OFF = {
    "PP-OCRv6 tiny": {"text": (6, -14), "headline": (4, 22)},
    "Indra": {"text": (-6, 8), "headline": (-6, 8)},
    "Tesseract (fine-tuned)": {"text": (6, -12), "headline": (60, -2)},
    "PP-OCRv6 small": {"text": (8, 4), "headline": (8, 18)},
    "OCRia-nano-1": {"text": (8, -14), "headline": (8, -14)},
    "Nemotron 3.5 Parse": {"text": (-8, 8), "headline": (-8, 8)},
    "Nemotron 3.5, 4-bit encoder": {"text": (-8, -14), "headline": (-8, 6)},
    "Nemotron Parse 2.0": {"text": (-8, -14), "headline": (-8, -16)},
}


def frontier(ms, key):
    out = [m for m in ms if not any(o is not m and o["mb"] <= m["mb"] and o[key] >= m[key] and
                                    (o["mb"] < m["mb"] or o[key] > m[key]) for o in ms)]
    return sorted(out, key=lambda m: m["mb"])


def panel(ax, ms, key, title):
    shipped = [m for m in ms if m.get("shipped", True)]
    fr = frontier(shipped, key)
    ax.set_xscale("log")
    ax.step([m["mb"] for m in fr], [m[key] for m in fr], where="post", color=BLUE, lw=2, alpha=.35, zorder=1)
    for m in ms:
        on = m in fr
        if m.get("shipped", True):
            ax.scatter(m["mb"], m[key], s=80, color=BLUE if on else PALE, edgecolor="white", linewidth=2, zorder=3)
        else:
            ax.scatter(m["mb"], m[key], s=70, facecolor="white", edgecolor="#999999", linewidth=1.5, zorder=3)
        dx, dy = OFF.get(m["name"], {}).get(key, (8, 4))
        ax.annotate(f'{m["name"]}\n{m["mb"]:g} MB · {m[key]:.1f}', (m["mb"], m[key]), xytext=(dx, dy),
                    textcoords="offset points", ha="left" if dx >= 0 else "right", va="center", fontsize=8.5,
                    color=INK if m.get("shipped", True) else INK2)
    ax.set_xlim(1, 2500)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, which="major", color=GRID, lw=.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel("First-run download (MB, log scale)", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=10)


def main(src, dst, stack=False):
    """stack: the two panels one above the other, sized for a narrow blog column."""
    ms = json.load(open(src))["models"]
    if stack:
        plt.rcParams.update({"font.size": 11})
    fig, axes = (plt.subplots(2, 1, figsize=(7, 10), dpi=200) if stack
                 else plt.subplots(1, 2, figsize=(12, 5.2), dpi=200, sharey=True))
    panel(axes[0], ms, "text", "Text: is the page's text there, in order")
    panel(axes[1], ms, "headline", "Overall: text, tables and layout")
    for ax in (axes if stack else axes[:1]):
        ax.set_ylabel("Omni Parse Bench score (% of tests passed)", color=INK2, fontsize=9)
    fig.suptitle("OCR models on anything-to-text: download size against accuracy", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.text(0.01, 0.01, ("250 Omni Parse Bench pages, the same pages for every model.\nLine: models nothing smaller "
             "beats. Hollow: packs measured but not shipped.") if stack else ("250 Omni Parse Bench pages, the same pages "
             "for every model. Line: models nothing smaller beats. Hollow: packs measured but not shipped."),
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], stack="--stack" in sys.argv)
