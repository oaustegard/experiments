#!/usr/bin/env python3
"""Follow-ups to bench.py, run after it:
  1. seed floor for the centered arms (haar s0/s1 at 2 and 1 bit) — uncentered 1-bit
     swung 0.11 nDCG@10 between two haar seeds on SciFact, so a single-seed centered
     number needs its own floor before it is compared with anything;
  2. remax on centered tokens (sign codes of x - mean; the q.mean term is a constant per
     query token, and MaxSim's max over document tokens passes it through unchanged);
  3. the geometry behind the centering predictor: cosine between random token pairs;
  4. paired bootstrap between the arms the writeup compares head to head.
Writes results_extra_<corpus>.json.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
from bench import late_rank, ndcg10, D_TOK, signs_pm1, paired_bootstrap  # noqa: E402
from remex import Quantizer, corpus_mean  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

PAIRS = [
    ("late remex 2-bit centered", "late plaid 2-bit"),
    ("late remex 1-bit centered", "late plaid 1-bit"),
    ("late remex 4-bit centered", "late plaid 4-bit"),
    ("late remex 1-bit centered", "late remex 1-bit"),
    ("late remex 2-bit centered", "dense bge-small fp32"),
    ("late remex 2-bit centered", "dense bge-small remex 4-bit"),
    ("late plaid 1-bit", "dense bge-small fp32"),
    ("late remex 1-bit centered", "dense bge-small fp32"),
    ("late fp32", "dense bge-small fp32"),
]


def main(corpus):
    W = bench.WORK / corpus
    T = np.load(W / "mv_tokens.npy"); off = np.load(W / "mv_offsets.npy")
    QT = np.load(W / "q_mv_tokens.npy"); qoff = np.load(W / "q_mv_offsets.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    pq = dict(np.load(HERE / f"results_{corpus}_perquery.npz"))
    ref = pq["late fp32"]
    mu = corpus_mean(T)
    rows = []

    def record(name, top):
        nd = np.array([ndcg10([dids[j] for j in r], qrels[q]) for r, q in zip(top, qids)])
        m, lo, hi, w, l = paired_bootstrap(nd - ref)
        pq[name] = nd
        rows.append(dict(arm=name, ndcg10=round(float(nd.mean()), 4), delta=round(m, 4), ci=[round(lo, 4), round(hi, 4)]))
        print(f"  {name:40s} nDCG@10 {nd.mean():.4f}  Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)

    print(f"[{corpus}]", flush=True)
    for bits in (2, 1):
        for seed in (0, 1):
            q = Quantizer(D_TOK, bits=bits, seed=seed, rotation="haar", mean=mu)
            record(f"late remex {bits}-bit centered haar s{seed}", late_rank(QT, qoff, q.decode(q.encode(T)), off))
    for k in (1, 2):
        rq = StackedSignBitQuantizer(D_TOK, k, seed=0, rotation="rht")
        S = signs_pm1(rq.encode(T - mu), k * D_TOK)
        R = rq._rotation_matrix
        record(f"late remax k={k} (asym) centered", late_rank(QT, qoff, S, off, lambda x, R=R: x @ R))

    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(T), 200_000), rng.integers(0, len(T), 200_000)
    cos = (T[a] * T[b]).sum(1)
    Rz = T - mu
    cos_c = (Rz[a] * Rz[b]).sum(1) / (np.linalg.norm(Rz[a], axis=1) * np.linalg.norm(Rz[b], axis=1))
    geom = dict(mean_norm_of_mean=round(float(np.linalg.norm(mu)), 3),
                random_pair_cos_mean=round(float(cos.mean()), 3), random_pair_cos_p05=round(float(np.percentile(cos, 5)), 3),
                centered_pair_cos_mean=round(float(cos_c.mean()), 3),
                residual_norm_mean=round(float(np.linalg.norm(Rz, axis=1).mean()), 3))
    print("  geometry:", geom, flush=True)

    comps = []
    for x, y in PAIRS:
        m, lo, hi, w, l = paired_bootstrap(pq[x] - pq[y])
        comps.append(dict(a=x, b=y, delta=round(m, 4), ci=[round(lo, 4), round(hi, 4)], wins=w, losses=l))
        print(f"  {x:28s} vs {y:30s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]  {w}/{l}", flush=True)
    json.dump(dict(rows=rows, geometry=geom, paired=comps), open(HERE / f"results_extra_{corpus}.json", "w"), indent=1)


if __name__ == "__main__":
    for c in sys.argv[1:] or ["scifact", "nfcorpus"]:
        main(c)
