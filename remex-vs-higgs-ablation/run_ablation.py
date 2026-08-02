#!/usr/bin/env python3
"""The sweep: 8 factorial arms + 3 controls x 6 bit widths x 3 corpora
x 5 rotation seeds x 2 similarity metrics.

Scoring is against **our own fp32 exact search**, never human qrels — human
labels conflate "is the base method good" with "did my approximation damage
it" (METHODS.md principle 4).  The metric set is the one
`jina-remex-vs-remax/score_fidelity.py` established and
`kb-k-sweep/sweep.py` reuses: recall@k versus the fp32 top-k, plus per-query
Spearman rho over the whole corpus, plus reconstruction error as a secondary
diagnostic only.

Asymmetric setting: documents are compressed, queries stay fp32.  That is
what "retrieval-index compression" means in deployment, and it is the setting
both lineages target.

Checkpointed per (dataset, metric, bits) block into results.json — CCotw
reaps idle background jobs and a sweep that cannot resume is a sweep that
never finishes (METHODS.md principle 7).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import quantizers as qzm
from quantizers import CONTROLS, FACTORIAL, make_arm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.pipeline import save_json

ASSETS = HERE / "assets"
OUT = HERE / "results.json"

BITS = (1, 2, 3, 4, 6, 8)
SEEDS = (0, 1, 2, 3, 4)
CONTROL_SEEDS = (0, 1)
METRICS = ("cosine", "ip")
KS = (10, 100)
SPEARMAN_QUERIES = 200      # cap: rank correlation is O(m n log n) per arm

DATASETS = ("arxiv768", "glove100", "nfcorpus1024")


# --------------------------------------------------------------------------
# scoring


def _ordinal_ranks(S: np.ndarray) -> np.ndarray:
    """Row-wise ordinal ranks.  Ties are broken arbitrarily rather than
    averaged; with float32 similarity scores over thousands of documents
    exact ties are vanishingly rare, and the alternative (scipy rankdata per
    row) costs more than the whole rest of the evaluation."""
    order = np.argsort(-S, axis=1)
    ranks = np.empty_like(order)
    idx = np.broadcast_to(np.arange(S.shape[1]), S.shape)
    np.put_along_axis(ranks, order, idx, axis=1)
    return ranks


def _topk(S: np.ndarray, k: int) -> np.ndarray:
    """Top-k indices per row, ordered.  argpartition then sort the survivors —
    a full argsort of a 1000 x 20000 score matrix costs more than everything
    else in the evaluation put together."""
    if k >= S.shape[1]:
        return np.argsort(-S, axis=1)
    part = np.argpartition(-S, k, axis=1)[:, :k]
    vals = np.take_along_axis(S, part, axis=1)
    return np.take_along_axis(part, np.argsort(-vals, axis=1), axis=1)


def _spearman_rows(ranks_a: np.ndarray, ranks_b: np.ndarray) -> float:
    """Mean per-row Pearson correlation of two rank matrices = Spearman rho."""
    a = ranks_a.astype(np.float64)
    b = ranks_b.astype(np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    return float(np.mean(num / np.maximum(den, 1e-30)))


class Reference:
    """fp32 exact search for one (corpus, metric) — the ceiling and the
    ground truth everything else is scored against."""

    def __init__(self, D: np.ndarray, Q: np.ndarray, metric: str):
        if metric == "cosine":
            D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
            Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        self.D = np.ascontiguousarray(D, dtype=np.float32)
        self.Q = np.ascontiguousarray(Q, dtype=np.float32)
        self.metric = metric
        S = self.Q @ self.D.T
        self.top = _topk(S, max(KS))
        self.gt_sets = {k: [set(row[:k].tolist()) for row in self.top] for k in KS}
        self.sq = min(SPEARMAN_QUERIES, S.shape[0])
        self.gt_ranks = _ordinal_ranks(S[: self.sq])
        self.dnorm2 = float(np.sum(self.D.astype(np.float64) ** 2))

    def score(self, Dhat: np.ndarray) -> dict:
        S = self.Q @ Dhat.T
        top = _topk(S, max(KS))
        out = {}
        for k in KS:
            hits = sum(len(self.gt_sets[k][i] & set(top[i, :k].tolist()))
                       for i in range(top.shape[0]))
            out[f"recall@{k}"] = hits / (top.shape[0] * k)
        out["spearman"] = _spearman_rows(_ordinal_ranks(S[: self.sq]), self.gt_ranks)
        err = Dhat.astype(np.float64) - self.D.astype(np.float64)
        out["rel_mse"] = float(np.sum(err ** 2) / self.dnorm2)
        return out


# --------------------------------------------------------------------------
# sweep


def arm_specs():
    specs = [(f"{s['rotation']}+{s['norm']}+{s['codebook']}", s) for s in FACTORIAL]
    specs.append(("control:uniform-norot", CONTROLS[0]))
    specs.append(("control:lm+qjl", CONTROLS[1]))
    return specs


def run(datasets=DATASETS, resume=True):
    results = json.loads(OUT.read_text()) if (resume and OUT.exists()) else {}
    for name in datasets:
        path = ASSETS / f"{name}.npz"
        if not path.exists():
            print(f"[{name}] no cache — skipping")
            continue
        z = np.load(path)
        D, Q = z["docs"].astype(np.float32), z["queries"].astype(np.float32)
        d = D.shape[1]
        print(f"\n### {name}: {D.shape[0]} docs x d={d}, {Q.shape[0]} queries",
              flush=True)
        for metric in METRICS:
            ref = Reference(D, Q, metric)
            key0 = f"{name}|{metric}"
            results.setdefault(key0, {})
            results[key0]["fp32"] = {
                "recall@10": 1.0, "recall@100": 1.0, "spearman": 1.0,
                "rel_mse": 0.0, "bytes": {"payload": d * 4.0, "side": 0.0,
                                          "total": d * 4.0},
            }
            for bits in BITS:
                bkey = str(bits)
                if bkey in results[key0] and resume:
                    print(f"  [{metric} {bits}b] cached", flush=True)
                    continue
                block, t0 = {}, time.time()
                for label, spec in arm_specs():
                    if spec.get("control") == "qjl" and bits < 2:
                        # `prod` is Lloyd-Max at bits-1 plus a 1-bit QJL
                        # residual.  At a 1-bit total budget that would be
                        # 1 bit of LM plus 1 bit of QJL charged as one — an
                        # unmatched budget, so the cell is simply absent.
                        continue
                    seeds = (CONTROL_SEEDS if spec.get("rotation") == "none"
                             else SEEDS)
                    runs = []
                    for seed in seeds:
                        arm = make_arm(spec, bits, d, seed)
                        Dhat = arm.encode_decode(ref.D)
                        m = ref.score(Dhat)
                        m["bytes"] = arm.bytes_per_vector()
                        m["shared_bytes"] = arm.shared_bytes()
                        m["codebook_m"] = arm.cb.m
                        runs.append(m)
                    block[label] = _summarize(runs)
                    r = block[label]
                    print(f"  [{metric} {bits}b] {label:<28} "
                          f"R@10={r['recall@10']['mean']:.3f}"
                          f"(min {r['recall@10']['min']:.3f}) "
                          f"rho={r['spearman']['mean']:.3f} "
                          f"B/vec={r['bytes']['total']:.1f}", flush=True)
                results[key0][bkey] = block
                save_json(OUT, results)
                print(f"  -- {metric} {bits}b done in {time.time() - t0:.0f}s",
                      flush=True)
    return results


def _summarize(runs):
    out = {}
    for k in ("recall@10", "recall@100", "spearman", "rel_mse"):
        v = np.array([r[k] for r in runs], dtype=np.float64)
        out[k] = {"mean": float(v.mean()), "min": float(v.min()),
                  "max": float(v.max()), "std": float(v.std()),
                  "n": int(v.size)}
    out["bytes"] = runs[0]["bytes"]
    out["shared_bytes"] = runs[0]["shared_bytes"]
    out["codebook_m"] = runs[0]["codebook_m"]
    return out


# --------------------------------------------------------------------------
# axis A wall-clock — measured on its own, not inferred from sweep noise


def timing(reps=5):
    """Wall-clock for axis A.

    Includes d well past the corpora so the O(d^2) vs O(d log d) crossover is
    measured rather than asserted -- at the experiment's own dimensions the
    dense rotation is one BLAS call and wins outright.
    """
    out = {}
    rng = np.random.default_rng(0)
    for d in (100, 768, 1024, 4096, 8192):
        nvec = 4096 if d <= 1024 else 512
        X = rng.standard_normal((nvec, d)).astype(np.float32)
        row = {}
        for kind in ("haar", "rht"):
            R = qzm.ROTATIONS[kind](d, 0)
            R.apply(X[:64])  # warm
            t = time.perf_counter()
            for _ in range(reps):
                R.apply(X)
            row[kind] = (time.perf_counter() - t) / reps
            row["nvec"] = X.shape[0]
        bt = time.perf_counter()
        qzm.HaarRotation(d, 0)
        row["haar_build_s"] = time.perf_counter() - bt
        bt = time.perf_counter()
        qzm.RHTRotation(d, 0)
        row["rht_build_s"] = time.perf_counter() - bt
        # Deliberately NOT called "speedup".  The RHT is genuinely O(d log d)
        # against Haar's O(d^2), but this measures numpy, and numpy runs the
        # dense rotation as a single BLAS sgemm while the FWHT is a Python
        # loop over strided slices.  Reporting the ratio as a speedup would be
        # a claim about the algorithm that the measurement does not support.
        row["haar_over_rht"] = row["haar"] / row["rht"]
        row["rounds"] = len(qzm.RHTRotation(d, 0).perms)
        out[str(d)] = row
        faster = "rht" if row["haar_over_rht"] > 1 else "haar"
        print(f"  d={d:>4}: haar {row['haar'] * 1e3:7.1f}ms  "
              f"rht {row['rht'] * 1e3:7.1f}ms  ratio haar/rht "
              f"{row['haar_over_rht']:.2f}x ({faster} faster)")
    return out


def main():
    args = sys.argv[1:]
    if args and args[0] == "timing":
        print("\n### axis A wall-clock (rotation apply, 4096 vectors)")
        t = timing()
        res = json.loads(OUT.read_text()) if OUT.exists() else {}
        res["_timing"] = t
        save_json(OUT, res)
        return 0
    run(datasets=tuple(args) if args else DATASETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
