"""Pareto chart of round 3: bytes per document against NanoBEIR mean NDCG@10.

Usage: python3 -I run/nanobeir_pareto.py     (reads results/nanobeir.json, writes results/nanobeir_pareto.svg/.png)

For scale, the float32 Matryoshka truncations to 256 and 128 dims that the tomaarsen/embeddinggemma-2-512d
card reports (its numbers, not re-measured here) are drawn as squares: they show what a large drop
looks like. Colour is the vector family (EG2 768, its 512 prefix, tomaarsen's SVD 512d model), filled markers are
float32 vectors and open markers remex codes. The line joins the arms no other arm beats at the same
or fewer bytes. Bars on the two seeded arms span remex seeds 0 to 4 and 42.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
r = json.loads((HERE / "results" / "nanobeir.json").read_text())
A = r["arms"]
FAM = {"eg2_768": ("EG2 768", "#4C72B0", 768), "prefix512": ("EG2 first 512", "#DD8452", 512),
       "svd512": ("SVD 512d (tomaarsen)", "#55A868", 512)}
pts = []
for fam, (label, color, d) in FAM.items():
    pts.append((d * 4, A[fam]["mean_of_sets"], fam, "float32", color))
    for bits in (2, 3, 4):
        key = f"{fam}_b{bits}"
        if key in A:
            pts.append((d * bits // 8, A[key]["mean_of_sets"], key, f"{bits}-bit", color))

# tomaarsen/embeddinggemma-2-512d card, "NanoBEIR at different Matryoshka dimensions" (mean of 13 sets)
CARD_TRUNC = [("eg2_768", 256, 61.246), ("eg2_768", 128, 56.297), ("svd512", 256, 57.767), ("svd512", 128, 48.821)]
for fam, dims, score in CARD_TRUNC:
    pts.append((dims * 4, score, f"{fam}_t{dims}", f"{dims}d", FAM[fam][1]))

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
front, best = [], -1
for b, s, *_ in sorted(pts, key=lambda p: (p[0], -p[1])):
    if s > best:
        front.append((b, s))
        best = s
ax.plot(*zip(*front), color="#999999", lw=1, zorder=1, label="Pareto frontier")
for key, (fam, bits, nudge) in {"svd512_b3": ("svd512", 3, 1.02), "eg2_768_b2": ("eg2_768", 2, 0.98)}.items():
    vals = r["seeds"][key]["mean_of_sets_per_seed"] + [A[key]["mean_of_sets"]]
    x = FAM[fam][2] * bits // 8 * nudge          # both sit at 192 B: nudge them apart
    ax.vlines(x, min(vals), max(vals), color=FAM[fam][1], lw=1, alpha=0.6, zorder=2)
# the remex points crowd the top of the chart at this scale: label the ones the text refers to
CALLOUT = {"svd512_b2": ("2-bit, 128 B: 62.13", 10, -38),
           "svd512_b3": ("3-bit on SVD 512d, 192 B: 62.86", 22, -62),
           "eg2_768_b2": ("2-bit on EG2 768, 192 B: 62.31", 30, -92),
           "svd512": ("SVD 512d float32: 63.21", -40, -40),
           "eg2_768": ("EG2 768 float32: 63.24", -60, -70)}
seen = set()
for b, s, key, kind, color in pts:
    fam = key.split("_b")[0].split("_t")[0]
    lab = FAM[fam][0] if fam not in seen else None
    seen.add(fam)
    trunc = "_t" in key
    ax.scatter(b, s, s=40 if trunc else 46, marker="s" if trunc else "o", zorder=3,
               color=color if kind == "float32" or trunc else "white", edgecolor=color, lw=1.6, label=lab)
    if key in CALLOUT:
        text, dx, dy = CALLOUT[key]
        ax.annotate(text, (b, s), textcoords="offset points", xytext=(dx, dy), fontsize=8, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7) if abs(dy) > 12 else None)
    elif trunc:
        ax.annotate(kind, (b, s), textcoords="offset points", xytext=(7, -3), fontsize=8, color="#444444")
ax.set_xscale("log", base=2)
ax.set_xticks([128, 192, 256, 512, 1024, 2048, 3072])
ax.set_ylim(45, 65)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
ax.set_xlabel("bytes per document (log scale)")
ax.set_ylabel("NanoBEIR NDCG@10, mean of 13 sets")
ax.set_title("EmbeddingGemma 2 on NanoBEIR: float32 vs remex codes", fontsize=11)
ax.text(0.99, 0.02, "open = remex codes, filled = float32; squares = float32 truncated to 256 / 128 dims (model card)", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7.5, color="#666666")
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1, 0.06))
fig.tight_layout()
for ext in ("svg", "png"):
    fig.savefig(HERE / "results" / f"nanobeir_pareto.{ext}")
print(front)
