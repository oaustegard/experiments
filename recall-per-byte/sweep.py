#!/usr/bin/env python3
"""Recall-per-byte bake-off for remax-style embedding compaction.

Generative-thinking move (random stimulus: "river") reframed the goal away from
"fewer model-weight bits" toward "more information per *stored* bit, fewer bits
where the corpus carries none." This tests two of the directions that fired,
against the remax StackedSignBit baseline, at MATCHED bytes/doc — reusing the
cached NFCorpus fp32 Jina embeddings (600 docs, 120 qrel queries), no re-embed.

Encoders compared at byte budgets B in {32, 64, 128, 256}:
  * remax    — StackedSignBitQuantizer (the shipped lib): dim=256, k=B/32 stacks
  * simhash  — centered random-hyperplane sign bits, 8B bits  (random rotation)
  * itq      — centered, PCA->8B dims, ITQ learned rotation, sign  (direction A)
  * pq       — Product Quantization, M=B subquantizers x 8-bit codebooks (dir B)

Metric: recall@10/@100 vs the topical gold (same as bench_nfcorpus.py). The clean
comparison is itq-vs-simhash at equal bytes (isolates the rotation). PQ trains a
codebook on the 600 docs — on a small corpus that favors PQ (centroids ~memorize
docs); flagged as an optimistic bound. Queries are held out of all training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment, spoke

JKB = experiment("jina-int8-remax_kb")
sys.path.insert(0, str(spoke("remax_kb")))
sys.path.insert(0, str(spoke("remax") / "src"))
sys.path.insert(0, str(JKB))

from remax_kb._hamming import hamming_scan, top_k          # noqa: E402
from remax import StackedSignBitQuantizer                  # noqa: E402
from bench_nfcorpus import load_nfcorpus                    # noqa: E402

BUDGETS = (16, 32, 48, 64)   # bytes per doc — the compactness regime. 1-bit-per-
# dim sign methods (itq) cap near min(N,D) bits, so the low end is the fair arena.
KS = (10, 100)
RNG = np.random.default_rng(0)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """(N, nbits) {0,1} -> (N, nbits/8) uint8, contiguous."""
    return np.packbits(bits.astype(np.uint8), axis=1)


def recall(ranked_ids, gold, ks):
    pos = {d: i + 1 for i, d in enumerate(ranked_ids)}
    return {k: len(set(ranked_ids[:k]) & gold) / len(gold) for k in ks}


# ---------------- encoders: return (doc_codes, encode_query_fn, scorer) ------- #

def enc_simhash(X, Q, nbits):
    mean = X.mean(0)
    W = RNG.standard_normal((X.shape[1], nbits)).astype(np.float32)
    dcodes = pack_bits((X - mean) @ W > 0)
    qcodes = pack_bits((Q - mean) @ W > 0)
    return dcodes, qcodes, "hamming"


def enc_itq(X, Q, nbits, iters=50):
    mean = X.mean(0)
    Xc = X - mean
    nbits = min(nbits, Xc.shape[1], Xc.shape[0] - 1)   # PCA cap: <= min(D, N-1)
    # PCA to nbits dims
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:nbits].T                      # (D, nbits)
    V = Xc @ P                            # (N, nbits)
    # ITQ: learn rotation R minimizing || sign(VR) - VR ||
    R, _ = np.linalg.qr(RNG.standard_normal((nbits, nbits)))
    for _ in range(iters):
        B = np.sign(V @ R); B[B == 0] = 1
        Uu, _, Vtt = np.linalg.svd(B.T @ V)
        R = (Vtt.T @ Uu.T)
    dcodes = pack_bits((V @ R) > 0)
    Vq = (Q - mean) @ P
    qcodes = pack_bits((Vq @ R) > 0)
    return dcodes, qcodes, "hamming"


def enc_remax(X, Q, nbits):
    # byte-matched: dim = nbits (k=1) so bytes = dim/8 = B; uses the lib's rotation.
    dim, k = min(nbits, X.shape[1]), 1
    mean = X.mean(0).astype(np.float32)
    q = StackedSignBitQuantizer(d=dim, k=k, seed=0)
    dcodes = q.encode(np.ascontiguousarray((X - mean)[:, :dim]))
    qcodes = q.encode(np.ascontiguousarray((Q - mean)[:, :dim]))
    return dcodes, qcodes, "hamming"


def enc_pq(X, Q, nbytes):
    from sklearn.cluster import KMeans
    D = X.shape[1]
    M = nbytes
    sub = np.array_split(np.arange(D), M)          # M subspaces, 8-bit each
    centroids, dcode = [], np.empty((X.shape[0], M), dtype=np.int32)
    for m, idx in enumerate(sub):
        km = KMeans(n_clusters=256, n_init=2, max_iter=50, random_state=0).fit(X[:, idx])
        centroids.append(km.cluster_centers_.astype(np.float32))
        dcode[:, m] = km.labels_
    return (dcode, centroids, sub), (Q, centroids, sub), "pq"


def score_hamming(dcodes, qcodes, doc_ids, gold, ks):
    dist = hamming_scan(dcodes, qcodes)
    idx = top_k(dist, max(ks))
    return recall([doc_ids[i] for i in idx], gold, ks)


def score_pq(dpack, qpack, qj, doc_ids, gold, ks):
    dcode, centroids, sub = dpack
    Q, _, _ = qpack
    q = Q[qj]
    # asymmetric IP: per subspace, table[c] = q_sub . centroid_c ; score = sum tables
    scores = np.zeros(dcode.shape[0], dtype=np.float32)
    for m, idx in enumerate(sub):
        tbl = centroids[m] @ q[idx]            # (256,)
        scores += tbl[dcode[:, m]]
    order = np.argsort(-scores)[:max(ks)]
    return recall([doc_ids[i] for i in order], gold, ks)


def main():
    docs, doc_ids, queries, qrels = load_nfcorpus()
    X = np.load(JKB / ".nf_doc_fp32.npz")["m"].astype(np.float32)
    Q = np.load(JKB / ".nf_qry_fp32.npz")["m"].astype(np.float32)
    assert X.shape[0] == len(doc_ids), f"{X.shape} vs {len(doc_ids)} — cache/subsample mismatch"
    print(f"docs {X.shape}, queries {Q.shape}, {len(queries)} eval queries\n", flush=True)

    methods = ["remax", "simhash", "itq", "pq"]
    print(f"{'bytes':>6} " + "".join(f"{m+'@10':>12}{m+'@100':>9}" for m in methods))
    for B in BUDGETS:
        nbits = B * 8
        enc = {
            "remax":   enc_remax(X, Q, nbits),
            "simhash": enc_simhash(X, Q, nbits),
            "itq":     enc_itq(X, Q, nbits),
            "pq":      enc_pq(X, Q, B),
        }
        agg = {m: {k: 0.0 for k in KS} for m in methods}
        for j, (qid, _) in enumerate(queries):
            gold = qrels[qid]
            for m in methods:
                d, qd, kind = enc[m]
                r = score_pq(d, qd, j, doc_ids, gold, KS) if kind == "pq" \
                    else score_hamming(d, qd[j], doc_ids, gold, KS)
                for k in KS:
                    agg[m][k] += r[k]
        n = len(queries)
        cells = "".join(f"{agg[m][10]/n:>12.3f}{agg[m][100]/n:>9.3f}" for m in methods)
        print(f"{B:>6} {cells}", flush=True)
    print("\nclean signal: itq vs simhash at equal bytes (learned vs random rotation).")
    print("pq trains codebook on the 600 docs — optimistic on a small corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
