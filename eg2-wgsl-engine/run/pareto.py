"""Speed/accuracy chart of the hand-written WGSL encoders: Mac query latency (GPU bridge job
gpu-bridge/jobs/wgsl-pareto) against Bluesky hit@10 from each model's own writeup.

    python3 run/pareto.py <bridge result.json> [<more result.json> ...]

Latency is the median over the given runs of each model's `ownMs`. Writes
results/pareto_mac.json and results/pareto_mac.png.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]

# download MB (weights + tokenizer) from the writeups cited below
MB = {"moedern-1.5M": 5.9, "moedern-27M": 108.5, "leaf-ir": 16.1, "ettin-17m": 10.2, "eg2-cut": 22.7, "eg2-full": 180}
# hit@10 on eg2-native-small's Bluesky task (1,498 synthetic queries, 61,686 posts)
ACC = {
    "moedern-1.5M": ("MoEdernColBERT 1.5M", 0.820, "moederncolbert-bench/RESULTS.md, round 4 (exact MaxSim, fp16 index)"),
    "moedern-27M": ("MoEdernColBERT 27M", 0.879, "moederncolbert-bench/RESULTS.md, round 4 (exact MaxSim, fp16 index)"),
    "leaf-ir": ("leaf-ir, tuned", 0.942, "ettin-native/RESULTS.md (int4 table, 2-bit index)"),
    "ettin-17m": ("Ettin-17m, tuned", 0.918, "ettin-native/RESULTS.md (int4 table, 2-bit index)"),
    "eg2-cut": ("EG2 4-layer cut", 0.907, "ettin-native/RESULTS.md (queries EG2's own index)"),
    "eg2-full": ("EG2 full (q4)", 0.918, "ettin-native/RESULTS.md (EG2's own index)"),
}


def main(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    for r in runs:
        assert r["device"] == "Mac", r["device"]
    rows = []
    for key, (label, hit, src) in ACC.items():
        ms = [r["outputs"]["models"][key]["ownMs"] for r in runs]
        rows.append({"model": key, "label": label, "hit10": hit, "mb": MB[key], "ms": statistics.median(ms), "ms_runs": ms,
                     "tokens": runs[0]["outputs"]["models"][key]["ownTokensMedian"], "accuracy_source": src})
    best = -1
    for row in sorted(rows, key=lambda x: x["ms"]):
        row["pareto"] = row["hit10"] > best
        best = max(best, row["hit10"])
    out = {"device": "Mac", "adapter": runs[0]["adapter"], "ua": runs[0]["ua"], "jobs": [r["id"] for r in runs], "rows": rows}
    (HERE / "results/pareto_mac.json").write_text(json.dumps(out, indent=1) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    pts = sorted(rows, key=lambda x: x["ms"])
    fr = [p for p in pts if p["pareto"]]
    ax.step([p["ms"] for p in fr], [p["hit10"] for p in fr], where="post", color="#9aa5b1", lw=1, zorder=1)
    for p in pts:
        ax.scatter(p["ms"], p["hit10"], s=46, zorder=3, color="#2463eb" if p["pareto"] else "#9aa5b1")
        ax.annotate(f'{p["label"]} ({p["mb"]:g} MB)', (p["ms"], p["hit10"]), textcoords="offset points", xytext=(7, -3), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 4])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.3, 9)
    ax.set_xlabel("ms per query, Mac (Chrome 155, Metal 3), WGSL engine, readback included")
    ax.set_ylabel("Bluesky hit@10")
    ax.set_title("Query encoders: speed against accuracy (blue = on the frontier)", fontsize=10)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "results/pareto_mac.png")
    for p in pts:
        print(f'{p["label"]:24s} {p["ms"]:6.3f} ms  {p["hit10"]:.3f}  {"*" if p["pareto"] else ""}')


if __name__ == "__main__":
    main(sys.argv[1:])
