#!/usr/bin/env python3
"""Summaries, paired bootstrap CIs and figures for emeans-pooled-residual.

Per (corpus, pf, K, source, bits) the per-query nDCG@10 is averaged over centroid seeds first, then
compared per query against a reference arm with a paired bootstrap (10,000 resamples of queries).
Writes summary.json, fig_frontier.png, fig_fit.png (and fig_grid.png if results_grid.jsonl exists).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)
B = 10_000


def load(corpus):
    arms = defaultdict(list)
    for line in open(HERE / f"results_{corpus}.jsonl"):
        r = json.loads(line)
        arms[(r["pf"], r["K"], r["source"], r["bits"])].append(r)
    out = {}
    for key, rows in arms.items():
        pq = np.mean([r["perquery"] for r in rows], axis=0)
        r0 = rows[0]
        out[key] = {"perquery": pq, "ndcg": float(pq.mean()), "seeds": len(rows),
                    "seed_ndcgs": [r["ndcg10"] for r in rows],
                    "bytes": r0["row_bytes_per_doc"] + r0["table_bytes_per_doc"],
                    "row_bytes": r0["row_bytes_per_doc"], "table_bytes": r0["table_bytes_per_doc"],
                    "mse": float(np.mean([r["corpus_mse"] for r in rows])),
                    "fit_s": float(np.mean([r.get("fit_seconds", 0.0) for r in rows])),
                    "tiny": float(np.mean([r.get("tiny_cells", 0) for r in rows]))}
    return out


def paired(a, b):
    d = a - b
    idx = RNG.integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(axis=1)
    return [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def pareto(points):
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best = [], -1
    for p in pts:
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front


summary = {}
for corpus in ("scifact", "nfcorpus"):
    if not (HERE / f"results_{corpus}.jsonl").exists():
        continue
    A = load(corpus)
    s = {"arms": {f"pf{k[0]} K{k[1]} {k[2]} {k[3]}b": {kk: (round(v, 4) if isinstance(v, float) else v)
                                                       for kk, v in a.items() if kk != "perquery"}
                  for k, a in sorted(A.items())}}
    # 1. residual vs single mean, same pf and bits
    s["residual_vs_mean"] = {}
    for (pf, K, src, b), a in sorted(A.items()):
        if K == 1:
            continue
        ref = A.get((pf, 1, "mean", b))
        if ref is not None:
            s["residual_vs_mean"][f"pf{pf} K{K} {src} {b}b"] = paired(a["perquery"], ref["perquery"])
    # 2. trainer vs Lloyd-25, same pf, K, bits
    s["trainer_vs_lloyd25"] = {}
    for (pf, K, src, b), a in sorted(A.items()):
        if src in ("lloyd25", "mean"):
            continue
        ref = A.get((pf, K, "lloyd25", b))
        if ref is not None:
            s["trainer_vs_lloyd25"][f"pf{pf} K{K} {src} {b}b"] = paired(a["perquery"], ref["perquery"])
    # 3. fit time and error relative to Lloyd-25
    s["fit"] = {}
    for (pf, K, src, b), a in sorted(A.items()):
        if b != 1 or src == "mean":
            continue
        ref = A[(pf, K, "lloyd25", 1)]
        s["fit"][f"pf{pf} K{K} {src}"] = {"fit_s": round(a["fit_s"], 1),
                                         "speedup_vs_lloyd25": round(ref["fit_s"] / max(a["fit_s"], 1e-9), 2),
                                         "mse_over_lloyd25": round(a["mse"] / ref["mse"], 4),
                                         "tiny_cells": a["tiny"]}
    # 4. byte frontier (table included / excluded), and named comparisons at matched bytes
    pts = [(a["bytes"], a["ndcg"], f"pf{k[0]} K{k[1]} {k[2]} {k[3]}b") for k, a in A.items() if k[2] in ("mean", "emeans2")]
    pts_row = [(a["row_bytes"], a["ndcg"], f"pf{k[0]} K{k[1]} {k[2]} {k[3]}b") for k, a in A.items() if k[2] in ("mean", "emeans2")]
    s["frontier_table_included"] = [[round(p[0]), round(p[1], 4), p[2]] for p in pareto(pts)]
    s["frontier_table_excluded"] = [[round(p[0]), round(p[1], 4), p[2]] for p in pareto(pts_row)]
    named = {}
    for name, x, y in [
        ("pf4 K16384 e-means2 1b vs pf4 mean 2b (~2.3 KB)", (4, 16384, "emeans2", 1), (4, 1, "mean", 2)),
        ("pf4 K4096 e-means2 2b vs pf2 mean 2b (2.6 vs 4.6 KB)", (4, 4096, "emeans2", 2), (2, 1, "mean", 2)),
        ("pf4 K4096 e-means2 1b vs pf2 mean 1b (1.7 vs 2.8 KB)", (4, 4096, "emeans2", 1), (2, 1, "mean", 1)),
        ("pf2 K16384 e-means2 1b vs pf2 mean 2b (3.8 vs 4.6 KB)", (2, 16384, "emeans2", 1), (2, 1, "mean", 2)),
    ]:
        if x in A and y in A:
            named[name] = {"x_bytes": round(A[x]["bytes"]), "y_bytes": round(A[y]["bytes"]),
                           "x_ndcg": round(A[x]["ndcg"], 4), "y_ndcg": round(A[y]["ndcg"], 4),
                           "delta_ci": paired(A[x]["perquery"], A[y]["perquery"])}
    s["matched_bytes"] = named
    summary[corpus] = s

grid = HERE / "results_grid.jsonl"
if grid.exists():
    rows = [json.loads(l) for l in open(grid)]
    summary["grid"] = [{k: r[k] for k in r if k not in ("lloyd", "emeans_np", "emeans_torch")} for r in rows]

json.dump(summary, open(HERE / "summary.json", "w"), indent=1)

# ---------- figures ----------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, corpus in zip(axes, ("scifact", "nfcorpus")):
    if corpus not in summary:
        continue
    A = load(corpus)
    for (pf, K, src, b), a in A.items():
        if src not in ("mean", "emeans2"):
            continue
        mk = {1: "o", 1024: "s", 4096: "^", 16384: "D"}[K]
        col = {4: "#a8452f", 2: "#2b3a67"}[pf]
        ax.scatter(a["bytes"] / 1000, a["ndcg"], marker=mk, s=46 if b == 1 else 70,
                   facecolors="none" if b == 1 else col, edgecolors=col)
    ax.set_title(corpus)
    ax.set_xlabel("KB per document (codes + side info + fp16 centroid table)")
    ax.set_ylabel("nDCG@10")
axes[0].scatter([], [], marker="o", c="k", label="K=1 (one mean)")
for K, mk in ((1024, "s"), (4096, "^"), (16384, "D")):
    axes[0].scatter([], [], marker=mk, c="k", label=f"K={K} e-means-2")
axes[0].scatter([], [], c="#a8452f", label="pool factor 4")
axes[0].scatter([], [], c="#2b3a67", label="pool factor 2")
axes[0].legend(fontsize=8, title="hollow = 1 bit, filled = 2 bits", title_fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "fig_frontier.png", dpi=150)

if "grid" in summary:
    g = [r for r in summary["grid"] if not r.get("skipped")]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for r in g:
        t = r.get("np_time_to_lloyd25")
        sp = r["lloyd25_s"] / t if t else 0.0
        kd = r["k"] * r["d"]
        ax.scatter(kd, max(sp, 0.05), c={3: "#a8452f", 32: "#66604f", 128: "#2b3a67", 768: "#4a6f4e"}[r["d"]],
                   marker="o" if r["data"] == "gmm" else "*", s=30 + 10 * np.log10(r["n"]))
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k × d")
    ax.set_ylabel("Lloyd-25 time / e-means time to the same error (numpy)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_grid.png", dpi=150)
print(json.dumps({c: summary[c].get("matched_bytes") for c in ("scifact", "nfcorpus") if c in summary}, indent=1))


# ---------- grid: matched-error speedup for the pre-chosen e-means-2 budget ----------
def matched(r, name):
    """Lloyd's time to reach e-means-2's error over e-means-2's time; lower bound if Lloyd never does."""
    ep, t, m = next(p for p in r[f"emeans_{name}"] if p[0] == 2)
    hit = next((c[0] for c in r["lloyd"] if c[1] <= m), None)
    return {"speedup": round((hit if hit else r["lloyd"][-1][0]) / t, 2), "lower_bound": hit is None,
            "err_vs_lloyd25": round(m / r["lloyd"][-1][1], 4)}


if grid.exists():
    rows = [json.loads(l) for l in open(grid)]
    gm = []
    for r in rows:
        if r.get("skipped"):
            continue
        gm.append({"cell": r["cell"], "data": r["data"], "n": r["n"], "k": r["k"], "d": r["d"],
                   "n_per_k": round(r["n"] / r["k"], 1), "np": matched(r, "np"), "torch": matched(r, "torch")})
    summary["grid_matched_emeans2"] = gm
    json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    cols = {3: "#a8452f", 32: "#66604f", 128: "#2b3a67", 768: "#4a6f4e"}
    for ax, name in zip(axes, ("np", "torch")):
        for g in gm:
            ax.scatter(g["n_per_k"], g[name]["speedup"], c=cols[g["d"]], s=18 + 12 * np.log10(g["n"]),
                       marker="^" if g[name]["lower_bound"] else ("o" if g["data"] == "gmm" else "*"))
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("points per cluster (n / k)")
        ax.set_title({"np": "NumpyEmeans", "torch": "torch Emeans (CPU)"}[name])
    axes[0].set_ylabel("Lloyd time to e-means-2's error / e-means-2 time")
    for d, c in cols.items():
        axes[0].scatter([], [], c=c, label=f"d={d}")
    axes[0].scatter([], [], marker="*", c="k", label="real data (pixels, LateOn tokens)")
    axes[0].scatter([], [], marker="^", c="k", label="Lloyd-25 never matched (lower bound)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_grid.png", dpi=150)


# ---------- headline numbers cited by RESULTS.md (each one traceable to a key) ----------
def rng_(vals):
    vals = [float(v) for v in vals]
    return [round(min(vals), 4), round(max(vals), 4)]


H = {}
for corpus in ("scifact", "nfcorpus"):
    s = summary[corpus]
    rv = {k: v for k, v in s["residual_vs_mean"].items() if " emeans2 " in k}
    H[corpus] = {
        "resid_1b_gain": rng_(v[0] for k, v in rv.items() if k.endswith("1b")),
        "resid_1b_ci_clear": sum(v[1] > 0 for k, v in rv.items() if k.endswith("1b")),
        "resid_2b_gain": rng_(v[0] for k, v in rv.items() if k.endswith("2b")),
        "drop_2b_to_1b_pf4": round(s["arms"]["pf4 K1 mean 2b"]["ndcg"] - s["arms"]["pf4 K1 mean 1b"]["ndcg"], 4),
        "emeans2_fit_speedup": rng_(v["speedup_vs_lloyd25"] for k, v in s["fit"].items() if k.endswith("emeans2")),
        "emeans2_mse_ratio": rng_(v["mse_over_lloyd25"] for k, v in s["fit"].items() if k.endswith("emeans2")),
        "lloyd2_mse_ratio": rng_(v["mse_over_lloyd25"] for k, v in s["fit"].items() if k.endswith("lloyd2")),
        "trainer_within_0005": {src: sum(abs(v[0]) <= 0.005 for k, v in s["trainer_vs_lloyd25"].items() if f" {src} " in k)
                                for src in ("emeans2", "emeans5", "lloyd2")},
        "trainer_ci_clear": sum((v[1] > 0 or v[2] < 0) for v in s["trainer_vs_lloyd25"].values()),
        "trainer_ci_clear_favour_lloyd25": sum(v[2] < 0 for v in s["trainer_vs_lloyd25"].values()),
        "row_bytes_pf4_K16384_emeans2_1b": round(s["arms"]["pf4 K16384 emeans2 1b"]["row_bytes"]),
        "row_bytes_pf4_mean_2b": round(s["arms"]["pf4 K1 mean 2b"]["row_bytes"]),
        "row_bytes_saving_pct": round(100 * (1 - s["arms"]["pf4 K16384 emeans2 1b"]["row_bytes"]
                                            / s["arms"]["pf4 K1 mean 2b"]["row_bytes"]), 1),
        "table_bytes_K16384": round(s["arms"]["pf4 K16384 emeans2 1b"]["table_bytes"]),
        "tiny_cells_K16384_emeans2": rng_(v["tiny_cells"] for k, v in s["fit"].items() if "K16384 emeans2" in k),
        "tiny_cells_K16384_lloyd25": rng_(v["tiny_cells"] for k, v in s["fit"].items() if "K16384 lloyd25" in k),
    }
if "grid_matched_emeans2" in summary:
    G = summary["grid_matched_emeans2"]
    hi = [g for g in G if g["d"] >= 32 and g["n_per_k"] >= 50]
    H["grid"] = {
        "np_speedup_d32plus_nk50": rng_(g["np"]["speedup"] for g in hi),
        "torch_speedup_d128plus_nk50": rng_(g["torch"]["speedup"] for g in hi if g["d"] >= 128),
        "torch_speedup_d32_nk50": rng_(g["torch"]["speedup"] for g in hi if g["d"] == 32),
        "gmm_err_k256_1024_d32plus_nk50": rng_(g["np"]["err_vs_lloyd25"] for g in hi
                                               if g["data"] == "gmm" and 256 <= g["k"] <= 1024),
        "small_nk_d128plus_err": rng_(g["np"]["err_vs_lloyd25"] for g in G
                                      if g["d"] >= 128 and g["n_per_k"] < 25),
        "small_nk_d128plus_cells": sum(1 for g in G if g["d"] >= 128 and g["n_per_k"] < 25),
        "d3_k64_or_less_speedup": rng_(g["np"]["speedup"] for g in G if g["d"] == 3 and g["n"] >= 100_000 and g["k"] <= 64),
        "d3_k256_speedup": rng_(g["np"]["speedup"] for g in G if g["d"] == 3 and g["n"] >= 100_000 and g["k"] == 256),
        "d3_k1024plus_speedup": rng_(g["np"]["speedup"] for g in G if g["d"] == 3 and g["n"] >= 100_000 and g["k"] >= 1024),
        "lateon_np_speedup": rng_(g["np"]["speedup"] for g in G if g["data"] == "lateon-pf4"),
        "lateon_torch_speedup": rng_(g["torch"]["speedup"] for g in G if g["data"] == "lateon-pf4"),
        "lateon_err": rng_(g["np"]["err_vs_lloyd25"] for g in G if g["data"] == "lateon-pf4"),
    }
summary["headline"] = H
json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
print(json.dumps(H, indent=1))


# ---------- extra headline numbers asked for by the fresh-reader review ----------
for corpus in ("scifact", "nfcorpus"):
    rv = {k: v for k, v in summary[corpus]["residual_vs_mean"].items() if " emeans2 " in k}
    H[corpus]["resid_2b_ci_clear"] = sum(v[1] > 0 for k, v in rv.items() if k.endswith("2b"))
if grid.exists():
    rows = [json.loads(l) for l in open(grid)]
    ratio = []
    for r in rows:
        if r.get("skipped") or r["d"] < 128 or r["n"] < 100_000:
            continue
        t_np = next(p[1] for p in r["emeans_np"] if p[0] == 2)
        t_to = next(p[1] for p in r["emeans_torch"] if p[0] == 2)
        ratio.append(t_np / t_to)
    H["grid"]["torch_vs_np_emeans2_time_d128plus_n1e5plus"] = rng_(ratio)
summary["headline"] = H
json.dump(summary, open(HERE / "summary.json", "w"), indent=1)


# ---------- lower bounds in the grid's d >= 32, n/k >= 50 band ----------
if grid.exists():
    G = summary["grid_matched_emeans2"]
    hi = [g for g in G if g["d"] >= 32 and g["n_per_k"] >= 50]
    H["grid"]["np_jobs_d32plus_nk50"] = len(hi)
    H["grid"]["np_lower_bounds_d32plus_nk50"] = sum(g["np"]["lower_bound"] for g in hi)
    H["grid"]["np_exact_speedup_d32plus_nk50"] = rng_(g["np"]["speedup"] for g in hi if not g["np"]["lower_bound"])
    summary["headline"] = H
    json.dump(summary, open(HERE / "summary.json", "w"), indent=1)


# ---------- regime counts and torch at d = 3 (second post review) ----------
if grid.exists():
    G = summary["grid_matched_emeans2"]
    H["grid"]["jobs_run"] = len(G)
    H["grid"]["jobs_d3"] = sum(g["d"] == 3 for g in G)
    H["grid"]["jobs_d32plus_nk_lt25"] = sum(g["d"] >= 32 and g["n_per_k"] < 25 for g in G)
    H["grid"]["jobs_d32plus_nk_25_50"] = sum(g["d"] >= 32 and 25 <= g["n_per_k"] < 50 for g in G)
    H["grid"]["err_d32plus_nk_lt25"] = rng_(g["np"]["err_vs_lloyd25"] for g in G if g["d"] >= 32 and g["n_per_k"] < 25)
    d3 = [g for g in G if g["d"] == 3 and g["n"] >= 100_000]
    H["grid"]["torch_d3_n1e5"] = rng_(g["torch"]["speedup"] for g in d3 if g["n"] == 100_000)
    H["grid"]["torch_d3_n600kplus_k1024"] = rng_(g["torch"]["speedup"] for g in d3 if g["n"] > 100_000 and g["k"] == 1024)
    H["grid"]["torch_d3_k4096plus"] = rng_(g["torch"]["speedup"] for g in d3 if g["k"] >= 4096)
for corpus in ("scifact",):
    rows = [json.loads(l) for l in open(HERE / f"results_{corpus}.jsonl")]
    r = next(r for r in rows if r["pf"] == 4 and r["K"] == 16384)
    H[corpus]["tokens_per_centroid_pf4_K16384"] = round(r["tokens_per_doc"] * 5183 / 16384, 1)  # 5,183 SciFact docs
    H[corpus]["emeans2_mse_ratio_pf4_K16384"] = summary[corpus]["fit"]["pf4 K16384 emeans2"]["mse_over_lloyd25"]
summary["headline"] = H
json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
