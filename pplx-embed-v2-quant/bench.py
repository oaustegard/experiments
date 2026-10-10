#!/usr/bin/env python3
"""Compression curve for pplx-embed-v2-late-0.6b token vectors (128-d, unit-norm).

Harness adapted from mxbai-edge-remex-quant/bench.py (same metrics, bootstrap, PLAID
codec and MaxSim scan). Changes: d=128, no dense comparator, PLAID k-means fit once and
shared across residual widths, and two arm groups that run did not have:
  - Matryoshka check: the model card and blog make no MRL claim, so prefix truncation
    to 64/32 dims (re-normalised, both sides) tests whether the space happens to be
    nested, and remex at matched bytes (128d x 2-bit = 64d x 4-bit = 32d x 8-bit =
    32 B/token) tests quantize-vs-truncate.
  - centered remax: StackedSignBitQuantizer has no mean, so codes are taken of
    (T - mu); the asym query is rotated uncentered (q.mu is constant across a query
    token's candidates, so MaxSim's per-doc max is unchanged by it).

Usage: bench.py [corpus] [--work DIR]
Writes results_<corpus>.json and results_<corpus>_perquery.npz.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "neomme-remex-quant"))
sys.path.insert(0, str(EXP / "mxbai-edge-remex-quant"))
from neomme_quant import seg_max_mean, paired_bootstrap, overlap_at_k, signs_pm1, l2n  # noqa: E402
from bench import ndcg10, recall100, topk  # noqa: E402  (mxbai-edge-remex-quant/bench.py)

from remex import Quantizer, corpus_mean  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

D_TOK = 128


def late_rank(QT, qoff, Tdec, off, qmap=None):
    """Full MaxSim scan (mean over query tokens of the per-doc max; same ranking as the
    sum). qmap transforms the query tokens into the scan space. Returns (n_q, 100)."""
    K_ = min(100, len(off) - 1); out = np.empty((len(qoff) - 1, K_), dtype=np.int64)
    for i in range(len(qoff) - 1):
        q = QT[qoff[i]:qoff[i + 1]]
        if qmap is not None:
            q = qmap(q)
        out[i] = topk(seg_max_mean(q @ Tdec.T, off), K_)
    return out


def plaid_fit(T, seed=0, sample=262_144, niter=20):
    """ColBERTv2 centroids (see mxbai-edge-remex-quant/bench.py::plaid_codec), fit once."""
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
    held = samp[: min(len(samp), 65_536)]
    res_h = held - C[np.argmax(held @ C.T, axis=1)]
    return C, K, codes, res_h


def plaid_decode(T, C, codes, res_h, nbits):
    if nbits == 0:
        return l2n(C[codes])
    nb = 2 ** nbits
    cut = np.quantile(res_h.ravel(), np.arange(1, nb) / nb)
    wts = np.quantile(res_h.ravel(), (np.arange(nb) + 0.5) / nb).astype(np.float32)
    out = np.empty_like(T)
    for s in range(0, len(T), 262_144):
        sl = slice(s, s + 262_144)
        out[sl] = l2n(C[codes[sl]] + wts[np.searchsorted(cut, T[sl] - C[codes[sl]])])
    return out


def trunc(x, dim):
    return l2n(np.ascontiguousarray(x[:, :dim]))


def main(corpus, W):
    T = np.load(W / "mv_tokens.npy"); off = np.load(W / "mv_offsets.npy")
    QT = np.load(W / "q_mv_tokens.npy"); qoff = np.load(W / "q_mv_offsets.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    n_docs, N = len(off) - 1, len(T)
    assert len(dids) == n_docs and len(qoff) - 1 == len(qids), (len(dids), n_docs, len(qoff), len(qids))
    tpd = N / n_docs
    print(f"[{corpus}] docs={n_docs} tokens={N} ({tpd:.1f}/doc) queries={len(qids)} "
          f"q_tokens={len(QT) / len(qids):.1f}/query", flush=True)
    mu = corpus_mean(T)
    predictor = float(np.linalg.norm(mu) / np.linalg.norm(T, axis=1).mean())
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, N, 20_000), rng.integers(0, N, 20_000)
    pair_cos = float(np.mean(np.sum(T[a] * T[b], axis=1)))
    print(f"  centering predictor ||mean||/mean||x|| = {predictor:.3f}; random-pair cosine {pair_cos:.3f}", flush=True)

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
        print(f"  {name:36s} {row['bytes_per_doc']:>10.1f} B/doc  nDCG@10 {row['ndcg10']:.4f} "
              f"R@100 {row['recall100']:.4f}{tail}", flush=True)

    t0 = time.time()
    REF = "fp32"
    record(REF, "late", late_rank(QT, qoff, T, off), tpd * D_TOK * 4, REF)
    record("fp16", "late", late_rank(QT, qoff, T.astype(np.float16).astype(np.float32), off), tpd * D_TOK * 2, REF)
    s8 = 127.0 / np.abs(T).max()
    record("int8", "late", late_rank(QT, qoff, np.round(T * s8).astype(np.int8).astype(np.float32) / s8, off), tpd * D_TOK, REF)

    # remex: encode once at 8 bits, search lower precisions by the Matryoshka bit nesting
    # (top k bits of an 8-bit code are a valid k-bit code), plus independent encodes at
    # 2/1 bit with a second seed for the noise floor. Unit-norm tokens: no 4 B norm counted.
    q8 = Quantizer(D_TOK, bits=8, seed=0, rotation="rht")
    c8 = q8.encode(T)
    for bits in (8, 4, 3, 2, 1):
        Td = q8.decode(c8, precision=bits) if bits < 8 else q8.decode(c8)
        record(f"remex {bits}-bit", "remex", late_rank(QT, qoff, Td, off), tpd * D_TOK * bits / 8, REF,
               dict(note="nested: decoded from the 8-bit code" if bits < 8 else None))
    del c8
    for bits in (2, 1):
        for seed in (0, 1):
            q = Quantizer(D_TOK, bits=bits, seed=seed, rotation="rht")
            record(f"remex {bits}-bit direct s{seed}", "remex", late_rank(QT, qoff, q.decode(q.encode(T)), off),
                   tpd * D_TOK * bits / 8, REF)
    for bits in (4, 2, 1):
        q = Quantizer(D_TOK, bits=bits, seed=0, rotation="rht", mean=mu)
        record(f"remex {bits}-bit centered", "remex", late_rank(QT, qoff, q.decode(q.encode(T)), off),
               tpd * D_TOK * bits / 8, REF, dict(note="+ one corpus mean (512 B total)"))
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    for k, mode, center in [(1, "asym", False), (2, "asym", False), (4, "asym", False), (1, "sym", False),
                            (1, "asym", True), (2, "asym", True), (1, "sym", True)]:
        rq = StackedSignBitQuantizer(D_TOK, k, seed=0, rotation="rht")
        S = signs_pm1(rq.encode(T - mu if center else T), k * D_TOK)
        R = rq._rotation_matrix
        if mode == "asym":
            qmap = lambda x, R=R: x @ R
        else:
            qmap = lambda x, R=R, c=center: np.where((x - mu if c else x) @ R > 0, 1.0, -1.0).astype(np.float32)
        name = f"remax k={k} ({mode})" + (" centered" if center else "")
        record(name, "remax", late_rank(QT, qoff, S, off, qmap), tpd * D_TOK * k / 8, REF)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    # Matryoshka: is the 128-d token space nested? And quantize-vs-truncate at 32 B/token.
    for dim in (64, 32):
        record(f"trunc {dim}d fp32", "mrl", late_rank(trunc(QT, dim), qoff, trunc(T, dim), off), tpd * dim * 4, REF)
    for dim, bits in ((64, 4), (32, 8), (64, 2), (32, 4)):
        Tt = trunc(T, dim)
        q = Quantizer(dim, bits=bits, seed=0, rotation="rht")
        record(f"trunc {dim}d + remex {bits}-bit", "mrl", late_rank(trunc(QT, dim), qoff, q.decode(q.encode(Tt)), off),
               tpd * dim * bits / 8, REF)
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    C, K, codes, res_h = plaid_fit(T)
    table = K * D_TOK * 2 / n_docs
    print(f"  plaid K={K} fit [{time.time() - t0:.0f}s]", flush=True)
    for nbits in (4, 2, 1, 0):
        per_tok = 2 + D_TOK * nbits / 8
        record(f"plaid {nbits}-bit" if nbits else "plaid centroid-only", "plaid",
               late_rank(QT, qoff, plaid_decode(T, C, codes, res_h, nbits), off), tpd * per_tok, REF,
               dict(K=K, bytes_per_doc_with_table=round(tpd * per_tok + table, 1)))
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    meta = dict(corpus=corpus, n_docs=n_docs, n_tokens=N, tokens_per_doc=round(tpd, 2), n_queries=len(qids),
                query_tokens_per_query=round(len(QT) / len(qids), 2), centering_predictor=round(predictor, 3),
                random_pair_cosine=round(pair_cos, 3), seconds=round(time.time() - t0, 1))
    json.dump(dict(meta=meta, rows=rows), open(HERE / f"results_{corpus}.json", "w"), indent=1)
    np.savez_compressed(HERE / f"results_{corpus}_perquery.npz", qids=np.array(qids), **{r: v for r, v in perq.items()})
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    work = None
    if "--work" in args:
        i = args.index("--work"); work = Path(args[i + 1]); del args[i:i + 2]
    corpus = args[0] if args else "scifact"
    main(corpus, work or HERE / ".work" / corpus)
