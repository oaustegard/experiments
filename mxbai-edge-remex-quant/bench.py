#!/usr/bin/env python3
"""Compression curve for mxbai-edge-colbert-v0-32m token vectors (64-d), with
bge-small-en-v1.5 single vectors as the storage comparator.

Arms (late interaction, MaxSim over every document, no candidate stage):
  fp32, fp16, int8 (global symmetric scale)
  remex 8/4/3/2/1-bit (RHT, renorm on; seed 0, plus seed 1 at 2 and 1 bit for the noise floor)
  remex centered 4/2/1-bit (Quantizer(mean=corpus mean))
  remax k=1 asym, k=2 asym, k=1 sym
  plaid-style residual: ColBERTv2 compression (k-means centroid id + per-dimension
    residual buckets, nbits 1/2/4, plus centroid-only), the codec PyLate/PLAID ships
  pool2 (pylate hierarchical, factor 2) fp32 / remex 2-bit / remex 1-bit
Dense arms: bge-small fp32 / fp16 / int8 / remex 4,2,1-bit / remax k=1 asym.

Bytes are per document as stored. Unit-norm token codes exclude remex's 4 B/vector
norm (every norm is 1.0); pooled tokens are not unit-norm, so their remex arms
include it. plaid rows report bytes with and without the centroid table amortised
over this corpus (the table is corpus-size-dependent: it shrinks per doc as n grows).

Writes results_<corpus>.json and results_<corpus>_perquery.npz.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "neomme-remex-quant"))
from neomme_quant import seg_max_mean, paired_bootstrap, overlap_at_k, signs_pm1, l2n  # noqa: E402

from remex import Quantizer, corpus_mean  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

WORK = HERE / ".work"
D_TOK = 64


# ----------------------------------------------------------------------------- metrics
def ndcg10(ranked, rel: dict) -> float:
    dcg = sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall100(ranked, rel: dict) -> float:
    return len(set(ranked[:100]) & set(rel)) / len(rel)


# ----------------------------------------------------------------------------- codecs
def plaid_codec(T, nbits, seed=0, sample=262_144, niter=20):
    """ColBERTv2 residual compression (Santhanam et al. 2022, colbert/indexing/codecs).

    Centroids: k-means on a token sample, K = 2**floor(log2(16*sqrt(N))), L2-normalised,
    assignment by max dot product. Residual = x - c, quantised per dimension into 2**nbits
    buckets whose cutoffs/weights are quantiles of the flattened residual sample. Decode =
    normalise(c + weights[bucket]). nbits=0 is centroid-only.
    """
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(seed)
    N = len(T)
    K = int(2 ** np.floor(np.log2(16 * np.sqrt(N))))
    samp = T[rng.choice(N, size=min(sample, N), replace=False)]
    km = MiniBatchKMeans(n_clusters=K, batch_size=8192, max_iter=niter, n_init=1, random_state=seed).fit(samp)
    C = l2n(km.cluster_centers_.astype(np.float32))
    codes = np.empty(N, dtype=np.int64)
    for s in range(0, N, 65_536):
        codes[s:s + 65_536] = np.argmax(T[s:s + 65_536] @ C.T, axis=1)
    if nbits == 0:
        return l2n(C[codes]), K, codes
    held = samp[: min(len(samp), 65_536)]
    res_h = held - C[np.argmax(held @ C.T, axis=1)]
    nb = 2 ** nbits
    cut = np.quantile(res_h.ravel(), np.arange(1, nb) / nb)
    wts = np.quantile(res_h.ravel(), (np.arange(nb) + 0.5) / nb).astype(np.float32)
    R = T - C[codes]
    buckets = np.searchsorted(cut, R)                       # (N, d) ints in [0, nb)
    return l2n(C[codes] + wts[buckets]), K, codes


def pool2(T, off):
    import torch
    from pylate.models.colbert import ColBERT
    docs = [torch.from_numpy(np.ascontiguousarray(T[off[i]:off[i + 1]])) for i in range(len(off) - 1)]
    pooled = ColBERT.pool_embeddings_hierarchical(None, docs, pool_factor=2, protected_tokens=1)
    lens = [len(p) for p in pooled]
    return np.concatenate([p.numpy() for p in pooled]).astype(np.float32), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


# ----------------------------------------------------------------------------- scoring
def topk(s, k):
    """Indices of the k highest scores, best first."""
    top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
    return top[np.argsort(-s[top], kind="stable")]


def late_rank(QT, qoff, Tdec, off, rot=None):
    """Full MaxSim scan. Tdec is the scan representation (decoded, or +/-1 signs in the
    rotated space with rot = the query-side transform). Returns (n_q, 100) top doc indices."""
    K_ = min(100, len(off) - 1); out = np.empty((len(qoff) - 1, K_), dtype=np.int64)
    for i in range(len(qoff) - 1):
        q = QT[qoff[i]:qoff[i + 1]]
        if rot is not None:
            q = rot(q)
        s = seg_max_mean(q @ Tdec.T, off)
        out[i] = topk(s, K_)
    return out


def dense_rank(Qd, Xdec, rot=None):
    Q = rot(Qd) if rot is not None else Qd
    S = Q @ Xdec.T
    return np.stack([topk(s, min(100, S.shape[1])) for s in S])


def main(corpus):
    W = WORK / corpus
    T = np.load(W / "mv_tokens.npy"); off = np.load(W / "mv_offsets.npy")
    QT = np.load(W / "q_mv_tokens.npy"); qoff = np.load(W / "q_mv_offsets.npy")
    Dd = np.load(W / "dense.npy"); Qd = np.load(W / "q_dense.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    n_docs, N = len(off) - 1, len(T)
    tpd = N / n_docs
    print(f"[{corpus}] docs={n_docs} tokens={N} ({tpd:.1f}/doc) queries={len(qids)}", flush=True)
    mu = corpus_mean(T)
    predictor = float(np.linalg.norm(mu) / np.linalg.norm(T, axis=1).mean())
    print(f"  centering predictor ||mean||/mean||x|| = {predictor:.3f}", flush=True)

    rows, perq, ranks = [], {}, {}

    def record(name, family, top, bytes_doc, ref, extra=None):
        ranked = [[dids[j] for j in r] for r in top]
        nd = np.array([ndcg10(r, qrels[q]) for r, q in zip(ranked, qids)])
        r100 = float(np.mean([recall100(r, qrels[q]) for r, q in zip(ranked, qids)]))
        perq[name] = nd; ranks[name] = top
        row = dict(arm=name, family=family, bytes_per_doc=round(bytes_doc, 1), ndcg10=round(float(nd.mean()), 4),
                   recall100=round(r100, 4))
        if ref in perq and ref != name:
            m, lo, hi, w, l = paired_bootstrap(nd - perq[ref])
            row.update(delta=round(m, 4), ci=[round(lo, 4), round(hi, 4)], wins=w, losses=l,
                       overlap10=round(float(np.mean([overlap_at_k(a, b) for a, b in zip(top, ranks[ref])])), 3))
        if extra:
            row.update(extra)
        rows.append(row)
        tail = ""
        if "delta" in row:
            tail = f"  Δ {row['delta']:+.4f} [{row['ci'][0]:+.4f},{row['ci'][1]:+.4f}] ov {row['overlap10']:.2f}"
        print(f"  {name:34s} {row['bytes_per_doc']:>10.1f} B/doc  nDCG@10 {row['ndcg10']:.4f}{tail}", flush=True)

    t0 = time.time()
    REF = "late fp32"
    record(REF, "late", late_rank(QT, qoff, T, off), tpd * D_TOK * 4, REF)
    record("late fp16", "late", late_rank(QT, qoff, T.astype(np.float16).astype(np.float32), off), tpd * D_TOK * 2, REF)
    s8 = 127.0 / np.abs(T).max()
    record("late int8", "late", late_rank(QT, qoff, np.round(T * s8).astype(np.int8).astype(np.float32) / s8, off), tpd * D_TOK, REF)

    # rht at power-of-two d is seed-invariant (one round: the seed's permutation and signs
    # land on the output side of a fixed WHT, and a symmetric per-coordinate codebook is
    # equivariant to signed permutations), so the seed floor uses haar at seeds 0 and 1.
    for bits, seed, rot in [(8, 0, "rht"), (4, 0, "rht"), (3, 0, "rht"), (2, 0, "rht"), (1, 0, "rht"),
                            (2, 0, "haar"), (2, 1, "haar"), (1, 0, "haar"), (1, 1, "haar")]:
        q = Quantizer(D_TOK, bits=bits, seed=seed, rotation=rot)
        name = f"late remex {bits}-bit" + ("" if rot == "rht" else f" haar s{seed}")
        record(name, "late", late_rank(QT, qoff, q.decode(q.encode(T)), off), tpd * D_TOK * bits / 8, REF)
    for bits in (4, 2, 1):
        q = Quantizer(D_TOK, bits=bits, seed=0, rotation="rht", mean=mu)
        record(f"late remex {bits}-bit centered", "late", late_rank(QT, qoff, q.decode(q.encode(T)), off),
               tpd * D_TOK * bits / 8, REF, dict(note="+ one corpus mean (256 B total)"))
    for k, mode in [(1, "asym"), (2, "asym"), (1, "sym")]:
        rq = StackedSignBitQuantizer(D_TOK, k, seed=0, rotation="rht")
        S = signs_pm1(rq.encode(T), k * D_TOK)
        R = rq._rotation_matrix
        rot = (lambda x, R=R: x @ R) if mode == "asym" else (lambda x, R=R: np.where(x @ R > 0, 1.0, -1.0).astype(np.float32))
        record(f"late remax k={k} ({mode})", "late", late_rank(QT, qoff, S, off, rot), tpd * D_TOK * k / 8, REF)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    for nbits in (4, 2, 1, 0):
        Tdec, K, _ = plaid_codec(T, nbits)
        per_tok = 2 + D_TOK * nbits / 8                     # uint16 centroid id + residual bits
        table = K * D_TOK * 2 / n_docs                      # fp16 centroid table, amortised
        record(f"late plaid {nbits}-bit" if nbits else "late plaid centroid-only", "late",
               late_rank(QT, qoff, Tdec, off), tpd * per_tok, REF,
               dict(K=K, bytes_per_doc_with_table=round(tpd * per_tok + table, 1)))
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    Tp, offp = pool2(T, off)
    tpd_p = len(Tp) / n_docs
    record("late fp32 pool2", "late", late_rank(QT, qoff, Tp, offp), tpd_p * D_TOK * 4, REF)
    for bits in (2, 1):
        q = Quantizer(D_TOK, bits=bits, seed=0, rotation="rht")
        record(f"late remex {bits}-bit pool2", "late", late_rank(QT, qoff, q.decode(q.encode(Tp)), offp),
               tpd_p * (D_TOK * bits / 8 + 4), REF, dict(note="includes 4 B/token norm (pooled means are not unit-norm)"))
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    dd = Dd.shape[1]
    record("dense bge-small fp32", "dense", dense_rank(Qd, Dd), dd * 4, REF)
    record("dense bge-small fp16", "dense", dense_rank(Qd, Dd.astype(np.float16).astype(np.float32)), dd * 2, REF)
    s8 = 127.0 / np.abs(Dd).max()
    record("dense bge-small int8", "dense", dense_rank(Qd, np.round(Dd * s8).astype(np.int8).astype(np.float32) / s8), dd, REF)
    for bits in (4, 2, 1):
        q = Quantizer(dd, bits=bits, seed=0, rotation="rht")
        record(f"dense bge-small remex {bits}-bit", "dense", dense_rank(Qd, q.decode(q.encode(Dd))), dd * bits / 8, REF)
    rq = StackedSignBitQuantizer(dd, 1, seed=0, rotation="rht")
    R = rq._rotation_matrix
    record("dense bge-small remax k=1 (asym)", "dense", dense_rank(Qd, signs_pm1(rq.encode(Dd), dd), lambda x: x @ R), dd / 8, REF)

    meta = dict(corpus=corpus, n_docs=n_docs, n_tokens=N, tokens_per_doc=round(tpd, 2), pooled_tokens_per_doc=round(tpd_p, 2),
                n_queries=len(qids), centering_predictor=round(predictor, 3), seconds=round(time.time() - t0, 1))
    json.dump(dict(meta=meta, rows=rows), open(HERE / f"results_{corpus}.json", "w"), indent=1)
    np.savez_compressed(HERE / f"results_{corpus}_perquery.npz", qids=np.array(qids), **{r: v for r, v in perq.items()})
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:] or ["scifact", "nfcorpus"]:
        main(c)
