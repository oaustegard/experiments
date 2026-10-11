"""Render a region's constellation to PNG (for the vision critic and for eyeballing).

  python3 render.py work/out1 work/png1 [rid ...]
"""
import json, sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

regions = json.load(open("work/regions.json"))

def render(r, o, path, title=True):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    fig.patch.set_facecolor("#0b1020"); ax.set_facecolor("#0b1020")
    pos = {s["id"]: (s["x"], s["y"]) for s in r["stars"]}
    for pl in o["figure"]:
        if len(pl) >= 2:
            xs, ys = zip(*pl); ax.plot(xs, ys, color="#7fa7d9", lw=1.2, alpha=0.8)
    for a, b in o["lines"]:
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#e8c46a", lw=1.6, alpha=0.9)
    for s in r["stars"]:
        ax.scatter(s["x"], s["y"], s=max(4, (6.2 - s["mag"]) ** 2.3 * 3), color="white", zorder=3)
    anat = {a["star"]: a["part"] for a in o["anatomy"]}
    for sid, part in anat.items():
        if sid in pos:
            ax.annotate(part[:28], pos[sid], xytext=(6, 6), textcoords="offset points", color="#c9d3e6", fontsize=6.5)
    if title:
        ax.set_title(f"{o['name']} — {o['gloss']}", color="white", fontsize=11)
    ax.set_xlim(-60, 1060); ax.set_ylim(1060, -60); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
    want = {int(x) for x in sys.argv[3:]}
    for r in regions:
        p = src / f"r{r['rid']:03d}.json"
        if p.exists() and (not want or r["rid"] in want):
            render(r, json.load(open(p))["output"], dst / f"r{r['rid']:03d}.png")
