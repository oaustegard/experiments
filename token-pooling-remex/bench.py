#!/usr/bin/env python3
"""Token pooling x centered remex on the pooling-aware checkpoint and its baseline.

For each (model, corpus) with finished encodings in .work/:
  pooling: none, pool_factor 2/3/4, max_tokens 32/16 (sentence-transformers
           HierarchicalTokenPooling: Ward on cosine, first token protected, outputs
           L2-normalised — the post's defaults)
  codec:   fp32, remex centered 4/2/1-bit (Quantizer(mean=corpus_mean of the
           pooled tokens)); haar seed 1 on the iso-byte arms for a seed floor
Scoring is brute-force MaxSim over every document (no candidate stage).

Bytes per document = tokens/doc x 128 x bits / 8. Pooled tokens are re-normalised, so
no arm stores a per-vector norm. remex's 128-float mean (512 B per index) is left out.

Writes results_<model>_<corpus>.json and results_<model>_<corpus>_perquery.npz,
one pair per (model, corpus), so a rerun skips finished pairs.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
MODELS = ("aware", "baseline")
CORPORA = ("scifact", "nfcorpus")
POOLINGS = {
    "none": None,
    "pf2": dict(pool_factor=2),
    "pf3": dict(pool_factor=3),
    "pf4": dict(pool_factor=4),
    "mt32": dict(max_tokens=32),
    "mt16": dict(max_tokens=16),
}
BITS = (4, 2, 1)
# iso-byte triples get a second rotation seed (haar s1) for the noise floor
SEED_ARMS = {("none", 1), ("pf2", 2), ("pf4", 4), ("none", 2), ("pf2", 4)}


# ----------------------------------------------------------------------------- metrics
def ndcg10(ranked, rel: dict) -> float:
    dcg = sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def paired_bootstrap(delta: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return (float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)),
            int((delta > 0).sum()), int((delta < 0).sum()))


def maxsim_top10(QT, qoff, T, off):
    """Sum-of-MaxSim over every document; returns (n_q, 10) top doc indices."""
    starts = off[:-1]
    out = np.empty((len(qoff) - 1, 10), dtype=np.int64)
    for i in range(len(qoff) - 1):
        S = QT[qoff[i]:qoff[i + 1]] @ T.T
        s = np.maximum.reduceat(S, starts, axis=1).sum(0)
        top = np.argpartition(-s, 10)[:10]
        out[i] = top[np.argsort(-s[top])]
    return out


# ----------------------------------------------------------------------------- pooling
def pool(T, off, kwargs):
    if kwargs is None:
        return T, off
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling
    p = HierarchicalTokenPooling(**kwargs)
    docs = [torch.from_numpy(T[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
    out = []
    for s in range(0, len(docs), 256):
        out.extend(p.pool(docs[s:s + 256]))
    lens = [len(d) for d in out]
    return (torch.cat(out).numpy().astype(np.float32),
            np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))


def run(mkey, corpus):
    W = WORK / mkey / corpus
    res_json = HERE / f"results_{mkey}_{corpus}.json"
    if res_json.exists():
        print(f"[{mkey}/{corpus}] done, skipping", flush=True)
        return
    if not (W / "meta.json").exists():
        print(f"[{mkey}/{corpus}] no encodings yet, skipping", flush=True)
        return
    from remex import Quantizer, corpus_mean, anisotropy
    T0 = np.load(W / "tokens.npy"); off0 = np.load(W / "offsets.npy")
    QT = np.load(W / "q_tokens.npy"); qoff = np.load(W / "q_offsets.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    n_docs = len(off0) - 1
    rows, pq = [], {}
    t_all = time.time()

    def record(name, pooling, bits, tpd, top, extra=None):
        nd = np.array([ndcg10([dids[j] for j in r], qrels[q]) for r, q in zip(top, qids)])
        pq[name] = nd
        bpt = 128 * 4 if bits is None else 128 * bits / 8
        row = dict(arm=name, pooling=pooling, bits=bits, tokens_per_doc=round(tpd, 2),
                   bytes_per_doc=round(tpd * bpt), ndcg10=round(float(nd.mean()), 4))
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"  {name:22s} tok/doc {tpd:7.2f}  B/doc {row['bytes_per_doc']:7d}  nDCG@10 {nd.mean():.4f}", flush=True)

    print(f"[{mkey}/{corpus}] {n_docs} docs, {len(T0)} tokens, {len(qids)} queries", flush=True)
    geom = {}
    for pname, kw in POOLINGS.items():
        t0 = time.time()
        T, off = pool(T0, off0, kw)
        tpd = len(T) / n_docs
        t_pool = time.time() - t0
        mu = corpus_mean(T)
        geom[pname] = dict(norm_of_mean=round(float(np.linalg.norm(mu)), 3),
                           anisotropy=round(float(anisotropy(T)), 3), pool_seconds=round(t_pool, 1))
        record(f"{pname} fp32", pname, None, tpd, maxsim_top10(QT, qoff, T, off))
        for bits in BITS:
            q = Quantizer(128, bits=bits, mean=mu)
            record(f"{pname} {bits}-bit", pname, bits, tpd, maxsim_top10(QT, qoff, q.decode(q.encode(T)), off))
            if (pname, bits) in SEED_ARMS:
                q = Quantizer(128, bits=bits, seed=1, rotation="haar", mean=mu)
                record(f"{pname} {bits}-bit haar s1", pname, bits, tpd,
                       maxsim_top10(QT, qoff, q.decode(q.encode(T)), off), dict(seed_arm=True))
        del T
    ref = pq["none fp32"]
    for r in rows:
        m, lo, hi, w, l = paired_bootstrap(pq[r["arm"]] - ref)
        r.update(delta=round(m, 4), ci=[round(lo, 4), round(hi, 4)], retained=round(r["ndcg10"] / float(ref.mean()), 4))
    np.savez(HERE / f"results_{mkey}_{corpus}_perquery.npz", **pq)
    json.dump(dict(model=mkey, corpus=corpus, n_docs=n_docs, n_tokens=int(len(T0)), rows=rows, geometry=geom,
                   seconds=round(time.time() - t_all)), open(res_json, "w"), indent=1)
    print(f"[{mkey}/{corpus}] geometry {geom}\n[{mkey}/{corpus}] {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    args = sys.argv[1:]
    for c in [a for a in args if a in CORPORA] or CORPORA:
        for m in [a for a in args if a in MODELS] or MODELS:
            run(m, c)
