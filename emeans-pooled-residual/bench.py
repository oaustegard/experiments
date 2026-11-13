#!/usr/bin/env python3
"""Residual remex codes for Ward-pooled LateOn tokens, against centroids from Lloyd and e-means.

Follows lateon-pool-remex (round 3: centered remex through RemexIndex) and emeans-residual
(remex 2's centred estimator applied per k-means cell). Centered remex codes every token as an
offset from one corpus mean (K = 1). Here each pooled token x in cell j is coded as the remex
direction of r = x - c_j plus fp32 ||r|| and c_j . r, and scored per (query token, doc token) as

    q . c_j  +  c_j . r  +  ||r|| * (q - c_j)_rot . v / (|v| * kappa)

then MaxSim (max over a document's tokens, summed over the query's tokens) and nDCG@10.

Centroid sources, all from the same random init (K rows of the pooled tokens, per seed):
  lloyd2, lloyd25   faiss.Kmeans stepped one iteration at a time (max_points_per_centroid huge)
  emeans2, emeans5  faiss.contrib.e_means.NumpyEmeans with nepochs 2 / 5
Fit wall-clock is recorded per source on 4 threads.

Control: K = 1 with c = the pooled corpus mean must score like remex.Quantizer(mean=mu).

Units checkpoint to .work/<corpus>/ (pooled tokens, centroids) and results_<corpus>.jsonl.
Usage: python3 bench.py <corpus> [--pfs 4,2] [--ks 1024,4096,16384] [--seeds 0,1] [--bits 1,2]
Prints 'BENCH DONE <corpus> pf<f>' per pooled set.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("corpus", choices=("scifact", "nfcorpus"))
p.add_argument("--pfs", default="4,2")
p.add_argument("--ks", default="1024,4096,16384")
p.add_argument("--seeds", default="0,1")
p.add_argument("--bits", default="1,2")
p.add_argument("--sources", default="lloyd2,lloyd25,emeans2,emeans5")
p.add_argument("--rot-seed", type=int, default=42)
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(args.threads)

import numpy as np  # noqa: E402
import faiss  # noqa: E402
import torch  # noqa: E402
import remex  # noqa: E402
from remex import Quantizer  # noqa: E402
from remex.codebook import assign_codes  # noqa: E402
from faiss.contrib.e_means import NumpyEmeans  # noqa: E402

warnings.filterwarnings("ignore", category=remex.AnisotropyWarning)
faiss.omp_set_num_threads(args.threads)
torch.set_num_threads(args.threads)

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "lateon-pool-remex" / ".work" / args.corpus
WORK = HERE / ".work" / args.corpus
WORK.mkdir(parents=True, exist_ok=True)
RES = HERE / f"results_{args.corpus}.jsonl"
D = 128
BITS = [int(b) for b in args.bits.split(",")]
QCHUNK = 96  # query-token rows per scoring block

done = set()
if RES.exists():
    for line in open(RES):
        r = json.loads(line)
        done.add((r["pf"], r["K"], r["seed"], r["source"], r["bits"]))

QT = np.load(SRC / "q_tokens.npy").astype(np.float32)
qoff = np.load(SRC / "q_offsets.npy")
dids = json.load(open(SRC / "doc_ids.json"))
qids = json.load(open(SRC / "query_ids.json"))
qrels = json.load(open(SRC / "qrels.json"))
NQ = len(qids)
q_of_row = np.repeat(np.arange(NQ), np.diff(qoff))
QUANT = {b: Quantizer(D, bits=b, seed=args.rot_seed) for b in BITS}
QT_ROT = QUANT[BITS[0]]._rotate_rows(QT)


def pooled(pf):
    f = WORK / f"pf{pf}.npz"
    if f.exists():
        z = np.load(f)
        return z["X"], z["off"]
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling
    T = np.load(SRC / "tokens.npy")
    off = np.load(SRC / "offsets.npy")
    mats = [T[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    pool = HierarchicalTokenPooling(pool_factor=pf)
    out = [t.numpy() for s in range(0, len(mats), 256)
           for t in pool.pool([torch.from_numpy(x) for x in mats[s:s + 256]])]
    X = np.concatenate(out).astype(np.float32)
    po = np.concatenate([[0], np.cumsum([len(o) for o in out])]).astype(np.int64)
    np.savez(f, X=X, off=po)
    return X, po


def ndcg10(scores):
    out = []
    for i, q in enumerate(qids):
        top = np.argsort(-scores[i])[:10]
        rel = qrels[q]
        dcg = sum(rel.get(dids[h], 0) / np.log2(r + 2) for r, h in enumerate(top))
        ideal = sorted(rel.values(), reverse=True)[:10]
        out.append(dcg / sum(g / np.log2(r + 2) for r, g in enumerate(ideal)))
    return np.array(out)


def maxsim(token_score_fn, doff, n):
    """Document scores (NQ, ndocs) from a function giving (rows, n) token scores for query-token rows."""
    ndocs = len(doff) - 1
    S = np.zeros((NQ, ndocs), dtype=np.float32)
    starts = doff[:-1]
    for a in range(0, len(QT), QCHUNK):
        rows = np.arange(a, min(a + QCHUNK, len(QT)))
        T = token_score_fn(rows)
        M = np.maximum.reduceat(T, starts, axis=1)
        np.add.at(S, q_of_row[rows], M)
    return S


def residual_parts(X, C, assign):
    """Per-bits (Y, bias) such that token score = QT_rot @ Y.T + bias + (QT @ C.T)[:, assign]."""
    Ca = C[assign].astype(np.float64)
    r = X.astype(np.float64) - Ca
    lengths = np.sqrt((r ** 2).sum(axis=1))
    dots = (r * Ca).sum(axis=1)
    u = (r / np.maximum(lengths, 1e-8)[:, None]).astype(np.float32)
    del r
    q0 = QUANT[BITS[0]]
    u_rot = q0._rotate_rows(u)
    ca_rot = q0._rotate_rows(np.ascontiguousarray(C, dtype=np.float32))[assign]
    for b in BITS:
        qz = QUANT[b]
        v = qz.centroids[assign_codes(u_rot, qz.boundaries, b)].astype(np.float32)
        vlen = np.sqrt((v.astype(np.float64) ** 2).sum(axis=1))
        scale = lengths / (np.maximum(vlen, 1e-12) * qz._cosine_factor())
        bias = (dots - scale * (v * ca_rot).sum(axis=1, dtype=np.float64)).astype(np.float32)
        yield b, v * scale[:, None].astype(np.float32), bias


def score_residual(X, doff, C, assign):
    QC = QT @ C.T
    for b, Y, bias in residual_parts(X, C, assign):
        fn = lambda rows: QT_ROT[rows] @ Y.T + bias[None, :] + QC[rows][:, assign]  # noqa: E731
        yield b, maxsim(fn, doff, len(X))


def bytes_per_doc(b, K, n, ndocs):
    cell = int(np.ceil(np.log2(K))) / 8 if K > 1 else 0.0
    row = D * b / 8 + 8 + cell  # codes, fp32 ||r|| and c.r, cell id
    return {"row_bytes_per_doc": row * n / ndocs, "table_bytes_per_doc": K * D * 2 / ndocs,
            "tokens_per_doc": n / ndocs}


def write(rec):
    with open(RES, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "perquery"}), flush=True)


def fit(X, K, seed, source):
    f = WORK / f"C_pf{pf}_K{K}_s{seed}_{source}.npy"
    meta = f.with_suffix(".json")
    if f.exists():
        return np.load(f), json.load(open(meta))
    rng = np.random.default_rng(seed)
    init = X[np.sort(rng.choice(len(X), K, replace=False))].copy()
    t0 = time.time()
    if source.startswith("lloyd"):
        iters = int(source[5:])
        C = init
        for _ in range(iters):
            km = faiss.Kmeans(D, K, niter=1, seed=seed + 1, max_points_per_centroid=10 ** 9,
                              verbose=False)
            km.train(X, init_centroids=C)
            C = km.centroids.copy()
    else:
        ep = int(source[6:])
        C, _ = NumpyEmeans(D, K, nepochs=ep, seed=seed).train(X, init_centroids=init)
        C = np.ascontiguousarray(C, dtype=np.float32)
    m = {"fit_seconds": round(time.time() - t0, 1)}
    np.save(f, C)
    json.dump(m, open(meta, "w"))
    return C, m


print(f"[{args.corpus}] remex {remex.__version__}, faiss {faiss.__version__}, {NQ} queries", flush=True)
for pf in [int(x) for x in args.pfs.split(",")]:
    X, doff = pooled(pf)
    n, ndocs = len(X), len(doff) - 1
    print(f"[{args.corpus} pf{pf}] {n} pooled tokens, {ndocs} docs, {n / ndocs:.1f}/doc", flush=True)

    # --- control and baseline: K = 1, the pooled corpus mean ------------------------------
    mu = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    zero = np.zeros(n, dtype=np.int64)
    if any((pf, 1, 0, "mean", b) not in done for b in BITS):
        parts = {b: (Y, bias) for b, Y, bias in residual_parts(X, mu[None, :], zero)}
        for b, S in score_residual(X, doff, mu[None, :], zero):
            qz = Quantizer(D, bits=b, seed=args.rot_seed, mean=mu)
            Yr, br = qz._score_vectors(qz.encode(X))
            ref = QT[:64] @ Yr.T + br[None, :]
            Y, bias = parts[b]
            mine = QT_ROT[:64] @ Y.T + bias[None, :] + (QT[:64] @ mu)[:, None]
            diff = np.abs(mine - ref)
            Sref = maxsim(lambda rows: QT[rows] @ Yr.T + br[None, :], doff, n)
            nd, ndref = ndcg10(S), ndcg10(Sref)
            write({"pf": pf, "K": 1, "seed": 0, "source": "mean", "bits": b,
                   "ndcg10": round(float(nd.mean()), 4), "ndcg10_remex_quantizer": round(float(ndref.mean()), 4),
                   "control_median_abs_diff": float(np.median(diff)),
                   "control_share_rows_differing": float((diff.max(axis=0) > 1e-4).mean()),
                   "corpus_mse": float(((X - mu) ** 2).sum(axis=1).mean()),
                   **bytes_per_doc(b, 1, n, ndocs), "perquery": [round(float(x), 5) for x in nd]})

    for K in [int(k) for k in args.ks.split(",")]:
        for seed in [int(s) for s in args.seeds.split(",")]:
            for source in args.sources.split(","):
                if all((pf, K, seed, source, b) in done for b in BITS):
                    continue
                C, m = fit(X, K, seed, source)
                _, I = faiss.knn(X, C, 1)
                assign = I[:, 0].astype(np.int64)
                mse = float(((X - C[assign]) ** 2).sum(axis=1).mean())
                sizes = np.bincount(assign, minlength=K)
                for b, S in score_residual(X, doff, C, assign):
                    if (pf, K, seed, source, b) in done:
                        continue
                    nd = ndcg10(S)
                    write({"pf": pf, "K": K, "seed": seed, "source": source, "bits": b,
                           "ndcg10": round(float(nd.mean()), 4), "corpus_mse": mse,
                           "empty_cells": int((sizes == 0).sum()),
                           "tiny_cells": int((sizes < 0.1 * n / K).sum()),
                           **m, **bytes_per_doc(b, K, n, ndocs),
                           "perquery": [round(float(x), 5) for x in nd]})
    print(f"BENCH DONE {args.corpus} pf{pf}", flush=True)
