#!/usr/bin/env python3
"""Reduce results/cluster_runs.jsonl, residual.json and perquery.npz to summary.json
and three figures. Every number quoted in RESULTS.md is a key of summary.json."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
runs = [json.loads(line) for line in (RES / "cluster_runs.jsonl").read_text().splitlines()]
resid = json.loads((RES / "residual.json").read_text())
pq = np.load(RES / "perquery.npz")
rng = np.random.default_rng(0)
out = {"cluster": {}, "residual": {}, "ivf": {}, "control": resid["control"]}


def mean(xs):
    return float(np.mean(xs))


# ---------------------------------------------------------------- clustering
by = defaultdict(dict)
for r in runs:
    by[(r["k"], r["seed"])][(r["arm"], r["budget"])] = r
ks = sorted({k for k, _ in by})
for k in ks:
    seeds = sorted(s for kk, s in by if kk == k and ("lloyd", 25) in by[(kk, s)])
    c = {"seeds": len(seeds)}
    ref = [by[(k, s)][("lloyd", 25)] for s in seeds]
    c["lloyd25_test_mse"] = mean([r["test_mse"] for r in ref])
    c["lloyd25_seconds"] = mean([r["seconds"] for r in ref])
    c["lloyd25_small"] = mean([r["small"] for r in ref])
    c["lloyd25_imbalance"] = mean([r["imbalance"] for r in ref])
    direct = [by[(k, s)].get(("lloyd_direct", 25)) for s in seeds]
    direct = [r for r in direct if r]
    c["lloyd25_direct_seconds"] = mean([r["seconds"] for r in direct])
    c["lloyd_iter_seconds"] = c["lloyd25_direct_seconds"] / 25
    c["init_test_ratio"] = mean([by[(k, s)][("init", 0)]["test_mse"] / by[(k, s)][("lloyd", 25)]["test_mse"] for s in seeds])
    for it in range(1, 26):
        rs = [by[(k, s)][("lloyd", it)] for s in seeds]
        c[f"lloyd{it}_test_ratio"] = mean([r["test_mse"] / by[(k, s)][("lloyd", 25)]["test_mse"] for r, s in zip(rs, seeds)])
        c[f"lloyd{it}_train_ratio"] = mean([r["train_mse"] / by[(k, s)][("lloyd", 25)]["train_mse"] for r, s in zip(rs, seeds)])
    budgets = sorted({b for s in seeds for (a, b) in by[(k, s)] if a == "emeans"})
    match = None
    for T in budgets:
        rs = [by[(k, s)][("emeans", T)] for s in seeds]
        ratios = [r["test_mse"] / by[(k, s)][("lloyd", 25)]["test_mse"] for r, s in zip(rs, seeds)]
        t = f"{T:g}"
        c[f"emeans{t}_test_ratio"] = mean(ratios)
        c[f"emeans{t}_test_ratio_min"] = float(min(ratios))
        c[f"emeans{t}_test_ratio_max"] = float(max(ratios))
        c[f"emeans{t}_train_ratio"] = mean([r["train_mse"] / by[(k, s)][("lloyd", 25)]["train_mse"] for r, s in zip(rs, seeds)])
        c[f"emeans{t}_seconds"] = mean([r["seconds"] for r in rs])
        c[f"emeans{t}_batch"] = rs[0]["batch_size"]
        c[f"emeans{t}_updates"] = rs[0]["updates"]
        c[f"emeans{t}_small"] = mean([r["small"] for r in rs])
        c[f"emeans{t}_imbalance"] = mean([r["imbalance"] for r in rs])
        c[f"emeans{t}_wins"] = int(sum(x < 1.0 for x in ratios))
        if match is None and mean(ratios) <= 1.0:
            match = T
    c["emeans_epoch_seconds"] = mean([c[f"emeans{T:g}_seconds"] / T for T in budgets])
    c["epoch_cost_ratio"] = c["emeans_epoch_seconds"] / c["lloyd_iter_seconds"]
    c["epochs_to_match_lloyd25"] = match
    if match is not None:
        sec = c[f"emeans{match:g}_seconds"]
        c["seconds_to_match_lloyd25"] = sec
        c["wallclock_speedup"] = c["lloyd25_direct_seconds"] / sec
        # what Lloyd reaches in the same wall-clock
        iters = max(1, min(25, int(sec / c["lloyd_iter_seconds"])))
        c["lloyd_iters_in_same_time"] = iters
        c["lloyd_same_time_test_ratio"] = c[f"lloyd{iters}_test_ratio"]
    best = min(budgets, key=lambda T: c[f"emeans{T:g}_test_ratio"])
    c["emeans_best_budget"] = best
    c["emeans_best_test_ratio"] = c[f"emeans{best:g}_test_ratio"]
    c["emeans_best_gain_pct"] = 100 * (1 - c["emeans_best_test_ratio"])
    out["cluster"][str(k)] = c

# ---------------------------------------------------------------- residual codes
R = resid["residual"]
n, d = resid["n"], resid["d"]
groups = defaultdict(list)
for r in R:
    groups[(r["source"], r["K"], r["bits"])].append(r)


def pq_mean(source, K, bits):
    """Per-query recall averaged over the k-means seeds of one arm."""
    arrs = [pq[f"{source}|K{K}|s{r['seed']}|b{bits}"] for r in groups[(source, K, bits)]]
    return np.mean(arrs, axis=0)


def paired(a, b, reps=10000):
    diff = a - b
    idx = rng.integers(0, len(diff), size=(reps, len(diff)))
    boot = diff[idx].mean(axis=1)
    return {"delta": float(diff.mean()), "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975))}


for (source, K, bits), rs in sorted(groups.items()):
    key = f"{source}|K{K}|b{bits}"
    e = {"recall10": mean([r["recall10"] for r in rs]),
         "recall10_sd": float(np.std([r["recall10"] for r in rs])),
         "seeds": len(rs), "corpus_mse": mean([r["corpus_mse"] for r in rs]),
         "row_bytes": rs[0]["row_bytes"], "table_bytes": rs[0]["table_bytes"],
         "total_bytes": rs[0]["total_bytes"]}
    if K > 1:
        e["vs_mean"] = paired(pq_mean(source, K, bits), pq_mean("mean", 1, bits))
        if source != "lloyd25" and ("lloyd25", K, bits) in groups:
            e["vs_lloyd25"] = paired(pq_mean(source, K, bits), pq_mean("lloyd25", K, bits))
        if source == "emeans5-fp16":
            a = pq[f"emeans5-fp16|K{K}|s{rs[0]['seed']}|b{bits}"]
            b = pq[f"emeans5|K{K}|s{rs[0]['seed']}|b{bits}"]
            e["vs_fp32_table"] = paired(a, b)
    out["residual"][key] = e

# equal-bytes reading: recall of the K=1 code at the next bit width up, and the
# bytes a residual arm would have to stay under to be the better buy
for bits in (1, 2):
    lo, hi = out["residual"][f"mean|K1|b{bits}"], out["residual"][f"mean|K1|b{bits * 2}"]
    out["residual"][f"mean_slope_b{bits}"] = {
        "recall_per_byte": (hi["recall10"] - lo["recall10"]) / (hi["total_bytes"] - lo["total_bytes"])}
for (source, K, bits), rs in sorted(groups.items()):
    if K == 1 or bits == 4:
        continue
    e = out["residual"][f"{source}|K{K}|b{bits}"]
    base = out["residual"][f"mean|K1|b{bits}"]
    slope = out["residual"][f"mean_slope_b{bits}"]["recall_per_byte"]
    extra = e["total_bytes"] - base["total_bytes"]
    e["extra_bytes"] = extra
    e["extra_row_bytes"] = e["row_bytes"] - base["row_bytes"]
    # recall the same extra bytes would buy if spent on code bits instead
    e["bits_instead"] = base["recall10"] + slope * extra
    e["bits_instead_row_only"] = base["recall10"] + slope * e["extra_row_bytes"]
    # corpus size at which the table costs no more than the recall gain is worth in bits
    gain = e["recall10"] - base["recall10"] - slope * e["extra_row_bytes"]
    e["breakeven_n"] = float(K * d * 4 / (gain / slope)) if gain > 0 else None

# ---------------------------------------------------------------- IVF
ivf = defaultdict(list)
for r in resid["ivf"]:
    ivf[(r["source"], r["K"])].append(r["curve"])


def scanned_at(curve, target):
    """Share of corpus scanned to reach a recall ceiling, log-interpolated over nprobe."""
    rec = [p["recall"] for p in curve]
    sc = [p["scanned"] for p in curve]
    if rec[0] >= target:
        return sc[0]
    for i in range(1, len(rec)):
        if rec[i] >= target:
            f = (target - rec[i - 1]) / (rec[i] - rec[i - 1])
            return float(np.exp(np.log(sc[i - 1]) + f * (np.log(sc[i]) - np.log(sc[i - 1]))))
    return None


for (source, K), curves in sorted(ivf.items()):
    avg = [{"nprobe": curves[0][i]["nprobe"],
            "recall": mean([c[i]["recall"] for c in curves]),
            "scanned": mean([c[i]["scanned"] for c in curves])}
           for i in range(len(curves[0]))]
    e = {"curve": avg, "seeds": len(curves)}
    for p in avg:
        e[f"recall_np{p['nprobe']}"] = p["recall"]
        e[f"scanned_np{p['nprobe']}"] = p["scanned"]
    for target in (0.9, 0.95, 0.99):
        vals = [scanned_at(c, target) for c in curves]
        ok = not any(v is None for v in vals)
        e[f"scanned_at_{int(target * 100)}"] = mean(vals) if ok else None
        e[f"scanned_at_{int(target * 100)}_pct"] = 100 * mean(vals) if ok else None
        e[f"scanned_at_{int(target * 100)}_min_pct"] = 100 * min(vals) if ok else None
        e[f"scanned_at_{int(target * 100)}_max_pct"] = 100 * max(vals) if ok else None
    out["ivf"][f"{source}|K{K}"] = e

(RES / "summary.json").write_text(json.dumps(out, indent=1))

# ---------------------------------------------------------------- figures
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "legend.frameon": False, "savefig.facecolor": SURF})

# 1. test MSE against passes over the data, one panel per k
fig, axes = plt.subplots(1, len(ks), figsize=(3.1 * len(ks), 3.3), sharey=True)
for ax, k in zip(axes, ks):
    c = out["cluster"][str(k)]
    its = list(range(1, 26))
    ax.plot(its, [100 * (c[f"lloyd{i}_test_ratio"] - 1) for i in its], color=BLUE, lw=2,
            label="Lloyd (faiss), after each iteration")
    budgets = sorted(float(key[6:-11]) for key in c if key.startswith("emeans") and key.endswith("_test_ratio") and key[6] != "_")
    ax.plot(budgets, [100 * (c[f"emeans{b:g}_test_ratio"] - 1) for b in budgets], color=ORANGE,
            lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label="e-means, one run per pass budget")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 25])
    ax.set_xticklabels(["1", "2", "5", "10", "25"])
    ax.set_ylim(-3, 6)
    ax.set_title(f"k = {k:,}")
    ax.set_xlabel("passes over the 90,000 training vectors")
axes[0].set_ylabel("held-out MSE, % above Lloyd at 25")
axes[0].legend(loc="upper right", fontsize=8.5)
fig.suptitle("e-means reaches Lloyd's 25-iteration held-out error in 1 to 3 passes at every k",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "fig_mse.png", dpi=160)
plt.close(fig)

# 2. recall against bytes per vector, and against K at a fixed bit width
from matplotlib.ticker import FuncFormatter  # noqa: E402

plain = FuncFormatter(lambda v, _: f"{v:g}")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
Ks = sorted({K for (_, K, _) in groups if K > 1})
ax = axes[0]
xs = [out["residual"][f"mean|K1|b{b}"]["total_bytes"] for b in (1, 2, 4)]
ys = [out["residual"][f"mean|K1|b{b}"]["recall10"] for b in (1, 2, 4)]
ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=7, mec=SURF, mew=1.5,
        label="one corpus mean (remex centred), 1 / 2 / 4 bits")
for bits, col in ((1, ORANGE), (2, AQUA), (4, VIOLET)):
    es = [out["residual"][f"emeans5|K{K}|b{bits}"] for K in Ks]
    ax.plot([e["total_bytes"] for e in es], [e["recall10"] for e in es], color=col, lw=2,
            marker="s", ms=6, mec=SURF, mew=1.5,
            label=f"{bits}-bit residual, K = 16 to 4,096 centroids")
ax.set_xscale("log")
ax.set_xticks([100, 150, 200, 300, 400, 500])
ax.xaxis.set_major_formatter(plain)
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_xlabel("bytes per vector, centroid table spread over the 100,000 vectors")
ax.set_ylabel("recall@10 against exact fp32 search")
ax.set_title("Counting the centroid table")
ax = axes[1]
for bits, col in ((1, ORANGE), (2, AQUA), (4, VIOLET)):
    for source, ls, mk in (("emeans5", "-", "s"), ("lloyd25", "--", "o"), ("random", ":", "^")):
        ys = [out["residual"][f"{source}|K{K}|b{bits}"]["recall10"] for K in Ks]
        ax.plot(Ks, ys, color=col, lw=2, ls=ls, marker=mk, ms=5, mec=SURF, mew=1.2)
    ax.axhline(out["residual"][f"mean|K1|b{bits}"]["recall10"], color=col, lw=1, alpha=0.5)
    ax.annotate(f"{bits}-bit", (Ks[-1], out["residual"][f"emeans5|K{Ks[-1]}|b{bits}"]["recall10"]),
                textcoords="offset points", xytext=(8, -3), fontsize=9, color=MUTED)
ax.set_xscale("log", base=2)
ax.set_xticks(Ks)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_xlim(Ks[0] * 0.8, Ks[-1] * 1.9)
ax.set_xlabel("K, number of centroids")
ax.set_title("By centroid source (thin line: one corpus mean)")
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], color=MUTED, lw=2, ls="-", marker="s", ms=5, label="e-means, 5 passes"),
                   Line2D([], [], color=MUTED, lw=2, ls="--", marker="o", ms=5, label="Lloyd, 25 iterations"),
                   Line2D([], [], color=MUTED, lw=2, ls=":", marker="^", ms=5, label="K random corpus points, untrained")],
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5)
axes[0].legend(loc="lower right", fontsize=8.5)
fig.suptitle("Residual codes raise recall at every bit width; at 100,000 vectors a large centroid table costs more than a second bit",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "fig_recall_bytes.png", dpi=160)
plt.close(fig)

# 3. IVF: share of true neighbours reachable against share of corpus scanned
fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
for ax, K in zip(axes, (256, 4096)):
    for source, col, label in (("remex-lsh-centred", MAGENTA, "remex IVFCoarseIndex, lsh, centred"),
                               ("remex-rotated_prefix-centred", YELLOW, "remex IVFCoarseIndex, rotated_prefix, centred"),
                               ("random", VIOLET, "nearest of K random corpus points"),
                               ("lloyd25", BLUE, "Lloyd, 25 iterations"),
                               ("emeans2", ORANGE, "e-means, 2 passes")):
        e = out["ivf"].get(f"{source}|K{K}")
        if not e:
            continue
        ax.plot([100 * p["scanned"] for p in e["curve"]], [p["recall"] for p in e["curve"]],
                color=col, lw=2, marker="o", ms=5, mec=SURF, mew=1.2, label=label)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plain)
    ax.set_xlabel("% of the corpus scanned (nprobe = 1, 2, 4, ...)")
    ax.set_title(f"{K:,} cells")
axes[0].set_ylabel("share of true top-10 inside the probed cells")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.5)
fig.suptitle("k-means cells reach the true neighbours after scanning a small fraction of what hash cells need",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout(rect=(0, 0.1, 1, 0.94))
fig.savefig(HERE / "fig_ivf.png", dpi=160)
plt.close(fig)
print("wrote summary.json and 3 figures")
