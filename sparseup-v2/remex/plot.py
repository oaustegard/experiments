"""Pareto plots of delta nDCG@10 vs bytes/doc. Written by gemini-4-hydrogen (DELEGATION.md job 2)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATASETS = ["scifact", "nfcorpus", "fiqa", "arguana"]
K_VALS = [16, 32, 64, 128, 256, None]
TAB10 = plt.get_cmap("tab10").colors
K_COLOR = {k: TAB10[i] for i, k in enumerate(K_VALS)}
CB_MARKER = {
    "fp32": "o", "fp16": "s", "uniform": "^", "lloyd": "v",
    "loglloyd": "D", "binary": "x", "anserini8": "*", "table8": "P",
}


def pareto(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    mask = np.ones(len(xs), dtype=bool)
    for i in range(len(xs)):
        if np.any((xs <= xs[i]) & (ys >= ys[i]) & ((xs < xs[i]) | (ys > ys[i]))):
            mask[i] = False
    return mask


def arm_label(k, codebook, bits, scale):
    b = str(bits) if codebook not in ("fp32", "fp16", "binary", "anserini8", "table8") else ""
    s = scale if scale != "none" else ""
    return f"k{k or 'full'} {codebook}{b} {s}".strip()


def load_data():
    loaded = []
    for ds in DATASETS:
        p = HERE / "results" / f"arms_{ds}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        arms = [r for r in d["arms"] if "query" not in r]
        loaded.append((ds, d["base_ndcg10"], arms))
    return loaded


def plot_panel(ax, points, title, xkey, print_table=False):
    xs = np.array([p[xkey] for p in points], dtype=float)
    ys = np.array([p["d_ndcg10"] for p in points], dtype=float)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for p in points:
        m = CB_MARKER.get(p["codebook"], "o")
        c = K_COLOR.get(p["k"], "gray")
        ax.scatter(p[xkey], p["d_ndcg10"], c=[c], marker=m, s=18, alpha=0.6)
    mask = pareto(xs, ys)
    f_pts = sorted([(xs[i], ys[i], points[i]) for i in np.where(mask)[0]], key=lambda t: t[0])
    if f_pts:
        fx, fy = [t[0] for t in f_pts], [t[1] for t in f_pts]
        ax.step(fx, fy, where="post", color="k", linewidth=1.0)
        for x, y, p in f_pts:
            m = CB_MARKER.get(p["codebook"], "o")
            c = K_COLOR.get(p["k"], "gray")
            kw = {"edgecolors": "k", "linewidths": 0.8} if m != "x" else {}
            ax.scatter(x, y, c=[c], marker=m, s=18, alpha=0.9, **kw)
            lbl = arm_label(p["k"], p["codebook"], p["bits"], p["scale"])
            ax.annotate(lbl, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
            if print_table:
                print(f"{x:10.2f}  {y:+8.4f}  {lbl}")
    ax.set_xscale("log")
    ax.set_ylim(max(float(ys.min()), -0.12), float(ys.max()) + 0.005)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("bytes/doc")
    ax.set_ylabel("delta nDCG@10")


def make(xkey, outname, label):
    loaded = load_data()
    if not loaded:
        return
    n_panels = len(loaded) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(25, 5.2))
    for ax, (ds, base, arms) in zip(axes[:-1], loaded):
        plot_panel(ax, arms, f"{ds} (fp32 full {base:.3f})", xkey)
    by_ds = [{(r["k"], r["codebook"], r["bits"], r["scale"]): r for r in arms} for _, _, arms in loaded]
    common = set.intersection(*(set(m.keys()) for m in by_ds))
    macro_pts = []
    for key in common:
        recs = [m[key] for m in by_ds]
        macro_pts.append({
            "k": key[0], "codebook": key[1], "bits": key[2], "scale": key[3],
            xkey: float(np.mean([r[xkey] for r in recs])),
            "d_ndcg10": float(np.mean([r["d_ndcg10"] for r in recs])),
        })
    print(f"\n--- Macro frontier ({label}) ---")
    print(f"{'bytes':>10}  {'delta':>8}  label")
    plot_panel(axes[-1], macro_pts, f"macro ({len(loaded)} datasets)", xkey, print_table=True)
    k_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=K_COLOR[k], markersize=6,
                        label="full" if k is None else f"k={k}") for k in K_VALS]
    cb_handles = [Line2D([0], [0], marker=m, color="k", linestyle="None", markersize=6, label=cb)
                  for cb, m in CB_MARKER.items()]
    leg1 = axes[-1].legend(handles=k_handles, loc="lower right", fontsize=7, frameon=True)
    axes[-1].add_artist(leg1)
    axes[-1].legend(handles=cb_handles, loc="lower right", bbox_to_anchor=(0.76, 0.0), fontsize=7, frameon=True)
    fig.suptitle(f"SPARSEUP-v2 doc compression: delta nDCG@10 vs bytes/doc ({label})")
    fig.tight_layout()
    out = HERE / "results" / outname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    make("bytes_fixed", "frontier.png", "16-bit ids")
    make("bytes_varint", "frontier_varint.png", "delta-varint ids")
