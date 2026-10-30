"""Step 3: remex codes at 768, on the rotated 512 span, and on the 512 Matryoshka prefix.

Usage: python3 -I run/eval.py

Corpus: the 61,674 live posts of the Bluesky AI-list index (out/docs.npy from run/embed_docs.py,
the 12 deleted posts dropped). Queries: the two held-out test sets of
bsky-ai-list/embeddinggemma-2-q4/query-targets-v1, real_test (1,000 MS MARCO tech/AI queries)
and synth_test (1,498 Haiku queries written from held-out posts). Ground truth: exact
inner-product top 10 on the fp32 768-d vectors.

Arms, for bits in 1..4, each remex Quantizer(d, bits, seed=42, mean=corpus mean in that space):
  full768    the vectors as the live index encodes them (bits=2 is the live index)
  rot512     x @ B.T, B = out/basis_proj.npy (from the projection weight, no data)
  prefix512  x[:512], renormalized (Matryoshka truncation, as eval_512.py did)
plus rot512 at 2 bits on out/basis_heldout.npy (fitted on msmarco_train), and the fp32 controls
fp32_rot512 and fp32_prefix512. Codes are scored as search_adc scores them, as one matmul over
decoded directions (eg2-query-student/run/common.py Index); every arm checks that on 20
queries against search_adc itself. Metric: recall@10 against the ground truth, per query set,
with a paired bootstrap 95% CI for each arm minus full768 at 2 bits. Writes results/recall.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import remex

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
K = 10
SETS = ("real_test", "synth_test")


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def topk(score_fn, Q, k=K, chunk=250):
    out = np.empty((len(Q), k), np.int64)
    for s in range(0, len(Q), chunk):
        sc = score_fn(Q[s:s + chunk])
        part = np.argpartition(-sc, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sc, part, 1), axis=1)
        out[s:s + chunk] = np.take_along_axis(part, order, 1)
    return out


def recall(a, b):
    return np.array([len(set(x) & set(y)) / K for x, y in zip(a, b)])


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def code_arm(X, Qs, bits):
    d = X.shape[1]
    qz = remex.Quantizer(d, bits=bits, seed=42, mean=X.mean(0).astype(np.float32))
    t0 = time.perf_counter()
    cv = qz.encode(X)
    enc_s = time.perf_counter() - t0
    dirs = qz.centroids.astype(np.float32)[cv.indices]
    scale = qz._score_norms(cv).astype(np.float32)
    bias = np.asarray(qz._row_bias(cv), np.float32)

    def score(Q):
        return (qz._rotate_rows(np.asarray(Q, np.float32)).astype(np.float32) @ dirs.T) * scale + bias

    tops = {k: topk(score, Q) for k, Q in Qs.items()}
    q0 = Qs[SETS[0]][:20]
    adc = np.stack([qz.search_adc(cv, q, k=K)[0] for q in q0])
    agree = float(recall(adc, tops[SETS[0]][:20]).mean())
    return tops, {"code_bytes": d * bits // 8, "encode_s": round(enc_s, 2), "adc_check_top10_agree": agree}


def main() -> int:
    X = np.load(OUT / "docs.npy")
    live = np.linalg.norm(X, axis=1) > 0
    X = X[live]
    Qs = {k: np.load(OUT / "qt" / f"{k}.eg2.npy") for k in SETS}
    Bp = np.load(OUT / "basis_proj.npy")
    Bh = np.load(OUT / "basis_heldout.npy")
    truth = {k: topk(lambda q: q @ X.T, Q) for k, Q in Qs.items()}
    spaces = {
        "full768": (X, Qs),
        "rot512": (X @ Bp.T, {k: Q @ Bp.T for k, Q in Qs.items()}),
        "prefix512": (unit(X[:, :512]), {k: unit(Q[:, :512]) for k, Q in Qs.items()}),
        "rot512_heldout": (X @ Bh.T, {k: Q @ Bh.T for k, Q in Qs.items()}),
    }
    per_q, meta = {}, {}
    for name in ("rot512", "prefix512"):
        Xs, Qss = spaces[name]
        per_q[f"fp32_{name}"] = {k: recall(topk(lambda q: q @ Xs.T, Qss[k]), truth[k]) for k in SETS}
    for name, bitset in (("full768", (1, 2, 3, 4)), ("rot512", (1, 2, 3, 4)),
                         ("prefix512", (1, 2, 3, 4)), ("rot512_heldout", (2,))):
        Xs, Qss = spaces[name]
        for b in bitset:
            arm = f"{name}_b{b}"
            tops, meta[arm] = code_arm(Xs, Qss, b)
            per_q[arm] = {k: recall(tops[k], truth[k]) for k in SETS}
            print(arm, meta[arm], {k: round(float(v.mean()), 4) for k, v in per_q[arm].items()}, flush=True)
    ref = per_q["full768_b2"]
    res = {"n_docs": int(live.sum()), "n_missing": int((~live).sum()), "k": K,
           "n_queries": {k: len(Q) for k, Q in Qs.items()},
           "aux_bytes_per_vector": 8, "arms": {}}
    for arm, r in per_q.items():
        row = {"recall_at_10": {k: float(r[k].mean()) for k in SETS}, **meta.get(arm, {})}
        if arm != "full768_b2":
            row["minus_full768_b2"] = {k: {"mean": float((r[k] - ref[k]).mean()),
                                           "ci95": boot_ci(r[k] - ref[k])} for k in SETS}
        res["arms"][arm] = row
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "recall.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
