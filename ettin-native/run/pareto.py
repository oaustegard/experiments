"""Pareto plot: total browser download (log MB) vs synth_test hit@10, from results/*.json.

Usage: .venv/bin/python -I run/pareto.py      -> results/pareto.png, results/pareto.json

Download = model + tokenizer at brotli-11 + 3.07 MB onnxruntime-web wasm runtime (ORT route).
SYM arms (own 2-bit index) are filled; EG2 layer cuts are ASYM (EG2-indexed docs, from
eg2-layer-student/results/final.json and its 3-bit-table commit) and drawn hollow, a different
metric family. Vocab-pruned arms are drawn hollow too: their out-of-domain query cosines
collapse (min 0.33-0.51), so they are not general-purpose. Palette: dataviz reference slots
1-4, validated (contrast WARN -> every point is direct-labelled and listed in RESULTS.md).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ORT = 3.07
C = {"ettin": "#2a78d6", "leaf": "#eb6834", "eg2": "#1baf7a", "cut": "#eda100"}
INK, MUTED, GRID, SURF = "#1a1a19", "#6b6a64", "#e6e5e0", "#fcfcfb"


def onnx_points(name, fam, label):
    r = json.loads((ROOT / "results" / f"onnx_{name}.json").read_text())
    pts = []
    for v, x in r.items():
        if v == "fp32" or "L63" not in v and v not in ("t8-w8",):
            continue
        pts.append({"family": fam, "arm": f"{label} {v}", "mb": round(x["br_mb"] + ORT, 2), "hit10": round(x["hit@10"][0], 3),
                    "pruned": v.startswith("k"), "sym": True,
                    "real_cos_min": round(x.get("cos_to_fp32_real", {}).get("min", 1.0), 3)})
    return pts


def main():
    pts = onnx_points("e17c", "ettin", "Ettin-17m") + onnx_points("leafirc", "leaf", "leaf-ir")
    pts += [
        {"family": "leaf", "arm": "leaf-ir stock int8 (MongoDB ONNX, no in-domain tuning)", "mb": round(15.85 + 0.25 + ORT, 2), "hit10": 0.889, "pruned": False, "sym": True},
        {"family": "eg2", "arm": "EG2 q4 (WebGPU only)", "mb": 179.9, "hit10": 0.918, "pruned": False, "sym": True},
        {"family": "cut", "arm": "EG2 cut k=2 q4, 32K vocab (ASYM)", "mb": 21.8, "hit10": 0.889, "pruned": True, "sym": False},
        {"family": "cut", "arm": "EG2 cut k=4 q4, 32K vocab (ASYM)", "mb": 29.0, "hit10": 0.901, "pruned": True, "sym": False},
        {"family": "cut", "arm": "EG2 cut k=4, 3-bit table (ASYM)", "mb": 25.4, "hit10": 0.907, "pruned": True, "sym": False},
    ]
    (ROOT / "results" / "pareto.json").write_text(json.dumps(pts, indent=1))

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160, facecolor=SURF)
    ax.set_facecolor(SURF)
    shape = {"ettin": "o", "leaf": "s", "eg2": "D", "cut": "^"}
    names = {"ettin": "Ettin-17m, in-domain tuned", "leaf": "leaf-ir, in-domain tuned", "eg2": "EmbeddingGemma 2", "cut": "EG2 layer cuts (ASYM metric)"}
    for fam in C:
        for hollow in (False, True):
            P = [p for p in pts if p["family"] == fam and (p["pruned"] or not p["sym"]) == hollow]
            if not P:
                continue
            ax.scatter([p["mb"] for p in P], [p["hit10"] for p in P], s=64, marker=shape[fam], zorder=3,
                       facecolors="none" if hollow else C[fam], edgecolors=C[fam], linewidths=2,
                       label=names[fam] if not hollow or fam == "cut" else None)
    # frontier over general-purpose SYM arms
    gp = sorted([p for p in pts if p["sym"] and not p["pruned"]], key=lambda p: p["mb"])
    front, best = [], -1
    for p in gp:
        if p["hit10"] > best:
            front.append(p)
            best = p["hit10"]
    ax.step([p["mb"] for p in front], [p["hit10"] for p in front], where="post", color=MUTED, lw=1, ls="--", zorder=2)
    short = {"k16000-t4-w8L63": "16K vocab", "k32000-t4-w8": "32K vocab", "t8-w8": "int8 table", "t4-w8L63": "int4 table",
             "t3-w8L63": "3-bit table", "t2-w8L63": "2-bit table"}
    for p in pts:
        lab = p["arm"]
        for k, v in short.items():
            if lab.endswith(k):
                lab = lab.replace(k, v)
                break
        lab = lab.replace(" (MongoDB ONNX, no in-domain tuning)", ", untuned").replace(" (WebGPU only)", "").replace(" (ASYM)", "")
        off = {"leaf-ir int8 table": (7, -12), "leaf-ir int4 table": (-20, 9), "leaf-ir 16K vocab": (-60, -14),
               "Ettin-17m int4 table": (-40, 9), "Ettin-17m int8 table": (7, -13), "EG2 cut k=4 q4, 32K vocab": (7, -12),
               "leaf-ir stock int8, untuned": (-75, 8), "EG2 cut k=2 q4, 32K vocab": (7, -12), "Ettin-17m 16K vocab": (7, -12)}.get(lab, (7, 6))
        ax.annotate(lab, (p["mb"], p["hit10"]), textcoords="offset points", xytext=off, fontsize=7.5, color=INK)
    ax.axhline(0.918, color=C["eg2"], lw=1, alpha=0.5, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_xlim(8, 260)
    ax.set_ylim(0.86, 0.95)
    ax.set_xlabel("Total browser download, MB (brotli; includes 3.07 MB ORT wasm runtime)", color=INK)
    ax.set_ylabel("hit@10, 1,498 synthetic Bluesky queries", color=INK)
    ax.set_title("Query embedder: download vs retrieval, 61,686 AI-list Bluesky posts", color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.text(0.01, 0.01, "Filled: own 2-bit index (SYM). Hollow: vocab-pruned (fragile on out-of-domain queries) or ASYM (EG2-indexed docs). "
             "Dashed: frontier of general-purpose SYM arms.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / "results" / "pareto.png", facecolor=SURF)
    for p in front:
        print("frontier", p["arm"], p["mb"], p["hit10"])


if __name__ == "__main__":
    main()
