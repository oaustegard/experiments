"""Chart for the blog post: download size per Whisper size, before and after (results/headline.json)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = json.loads((Path(__file__).resolve().parent.parent / "results" / "headline.json").read_text())
INDIGO, CORAL, SAGE, PAPER, INK, MUTED, RULE = "#2b3a67", "#cf6853", "#7a9e7e", "#f5f0e8", "#1e2a4a", "#66604f", "#d5cfc3"
rows = [("large-v3-turbo", h["page_turbo_mb"], h["turbo_all_gzip_mb"], h["turbo_en_gzip_mb"]),
        ("small", h["page_small_mb"], h["small_all_gzip_mb"], h["small_en_gzip_mb"]),
        ("base", h["page_base_mb"], h["base_all_gzip_mb"], h["base_en_gzip_mb"]),
        ("tiny", h["page_tiny_mb"], h["tiny_all_gzip_mb"], h["tiny_en_gzip_mb"])]
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
bh = 0.24
for i, (name, before, alls, en) in enumerate(rows):
    y = len(rows) - 1 - i
    for k, (v, c, lab) in enumerate([(before, INDIGO, "before: ONNX files"), (alls, SAGE, "packed, all languages"), (en, CORAL, "packed, English")]):
        yy = y + (1 - k) * (bh + 0.03)
        ax.barh(yy, v, height=bh, color=c, label=lab if i == 0 else None)
        ax.text(v + 8, yy, f"{v:.0f} MB", va="center", ha="left", fontsize=17, color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=20, color=INK)
ax.set_xlim(0, 860); ax.set_xlabel("download, MB", fontsize=17, color=MUTED)
ax.tick_params(axis="x", labelsize=15, colors=MUTED); ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(RULE); ax.xaxis.grid(True, color=RULE, linewidth=1); ax.set_axisbelow(True)
ax.set_title("Whisper in the browser: what each model size downloads", fontsize=24, color=INDIGO, loc="left", pad=18)
ax.legend(loc="lower right", fontsize=17, frameon=False, labelcolor=INK)
fig.tight_layout()
fig.savefig(sys.argv[1], facecolor=PAPER)
