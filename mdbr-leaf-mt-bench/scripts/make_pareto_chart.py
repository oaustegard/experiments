"""Byte-budget Pareto chart from results_headtohead_leaf.json -> pareto.png.

One panel per distribution. X = payload bytes/vector (log2), Y = R@10.
Every measured (dim, param) composition is a faint dot; the line per arm is
that arm's upper envelope (best R@10 at each byte budget); global
Pareto-frontier points get a dark ring. Colors are the dataviz reference
categorical palette in fixed slot order, validated (light mode) with the
contrast WARN relieved by direct labels.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]

ARMS = [  # fixed palette slot order
    ("matryoshka-fp32", "#2a78d6", "MRL fp32"),
    ("vendor-binary", "#eb6834", "binary (card)"),
    ("binary-asym", "#1baf7a", "binary asym"),
    ("vendor-int8", "#eda100", "int8 (card)"),
    ("remex", "#e87ba4", "remex"),
    ("remax", "#008300", "remax"),
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"


def envelope(rows: list[dict]) -> tuple[list[float], list[float]]:
    best: dict[float, float] = {}
    for r in rows:
        b = float(r["bytes"])
        best[b] = max(best.get(b, 0.0), r["r@10"])
    xs = sorted(best)
    return xs, [best[x] for x in xs]


def pareto(rows: list[dict]) -> set[tuple[float, float]]:
    pts = sorted({(float(r["bytes"]), r["r@10"]) for r in rows})
    out, ymax = set(), -1.0
    for b, y in pts:
        pass  # placeholder to keep structure clear
    # a point is on the frontier if no point has <= bytes and > r@10
    for b, y in pts:
        if not any(b2 <= b and y2 > y for b2, y2 in pts):
            out.add((b, y))
    return out


def main() -> None:
    data = json.load(open(HERE / "results_headtohead_leaf.json"))
    rows = data["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor=SURFACE)

    for ax, dist in zip(axes, ("blog", "code")):
        sub = [r for r in rows if r["dist"] == dist]
        by_arm = defaultdict(list)
        for r in sub:
            by_arm[r["arm"]].append(r)

        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color="#e6e5e1", linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c9c8c2")

        front = pareto(sub)
        for arm, color, label in ARMS:
            arows = by_arm[arm]
            ax.scatter([r["bytes"] for r in arows], [r["r@10"] for r in arows],
                       s=18, color=color, alpha=0.30, linewidths=0, zorder=2)
            xs, ys = envelope(arows)
            ax.plot(xs, ys, color=color, linewidth=2, zorder=3,
                    label=label if dist == "blog" else None)
        fx = sorted(front)
        ax.scatter([b for b, _ in fx], [y for _, y in fx], s=52,
                   facecolors="none", edgecolors=INK, linewidths=1.4, zorder=4)

        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
        ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512",
                            "1K", "2K", "4K"], color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_xlabel("payload bytes / vector (log scale)", color=INK2, fontsize=10)
        ax.set_ylabel("R@10" if dist == "blog" else "", color=INK2, fontsize=10)
        ax.set_title(f"{dist} distribution (n=179)", color=INK, fontsize=11,
                     loc="left", fontweight="bold")

        # direct labels; they also relieve the contrast WARN on aqua/yellow/magenta
        if dist == "blog":
            ax.annotate("binary asym", (128, 0.559), xytext=(150, 0.575),
                        color="#177a58", fontsize=9)
            ax.annotate("MRL fp32", (256, 0.464), xytext=(300, 0.446),
                        color="#2a78d6", fontsize=9)
            ax.annotate("remax", (32, 0.453), xytext=(24, 0.427),
                        color="#008300", fontsize=9)
            ax.annotate("remex", (12, 0.369), xytext=(9.5, 0.384),
                        color="#c2557e", fontsize=9)
            ax.annotate("int8", (64, 0.464), xytext=(70, 0.442),
                        color="#a87200", fontsize=9)
        else:
            ax.annotate("binary asym", (128, 0.832), xytext=(140, 0.858),
                        color="#177a58", fontsize=9)
            ax.annotate("MRL fp32", (256, 0.737), xytext=(300, 0.715),
                        color="#2a78d6", fontsize=9)
            ax.annotate("remax", (16, 0.665), xytext=(17, 0.630),
                        color="#008300", fontsize=9)
            ax.annotate("remex", (20, 0.670), xytext=(10, 0.690),
                        color="#c2557e", fontsize=9)
            ax.annotate("int8", (128, 0.754), xytext=(140, 0.732),
                        color="#a87200", fontsize=9)

    axes[0].legend(loc="lower right", frameon=False, fontsize=9,
                   labelcolor=INK2, handlelength=1.6)
    fig.suptitle("mdbr-leaf-mt (int8 export): retrieval per byte — "
                 "quantization arms vs MRL truncation; rings = Pareto frontier",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "pareto.png", dpi=160, facecolor=SURFACE)
    print("wrote pareto.png")


if __name__ == "__main__":
    main()
