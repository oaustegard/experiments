"""Step 2: post-hoc compression of cached SPARSEUP-v2 doc vectors, scored exactly.

Arms: per-doc top-k x value codebook x bits x per-doc scale, plus reference points.
Every arm is scored by exact dot product against fp32 queries (or 8-bit queries in
the query-side arm), and per-query nDCG@10 / R@100 are kept for a paired bootstrap
against the fp32 full-k baseline.

Usage: python3 compress.py scifact nfcorpus arguana fiqa [--quick]
Writes results/arms_<dataset>.json (one record per arm with per-query metrics
dropped; the bootstrap CI is computed here) and results/perquery_<dataset>.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RES = HERE / "results"
KS = [16, 32, 64, 128, 256, None]  # None = full
BITS = [8, 4, 3, 2]
CODEBOOKS = ["uniform", "lloyd", "loglloyd"]
SCALES = ["none", "docmax", "l2"]
RNG_SEED = 0
N_BOOT = 1000


# ---------------------------------------------------------------- data
def load(name, variant=None):
    tag = name if variant is None else f"{name}_{variant}"
    D = sp.load_npz(CACHE / f"{tag}_docs.npz").tocsr()
    Q = sp.load_npz(CACHE / f"{tag}_queries.npz").tocsr()
    meta = json.loads((CACHE / f"{tag}_meta.json").read_text())
    D.sort_indices()
    Q.sort_indices()
    return D, Q, meta


# ---------------------------------------------------------------- top-k
def topk_rows(D, k):
    """Keep the k largest values of each CSR row (ties broken by column order)."""
    if k is None:
        return D.copy()
    nnz = np.diff(D.indptr)
    if nnz.max() <= k:
        return D.copy()
    row = np.repeat(np.arange(D.shape[0]), nnz)
    order = np.lexsort((-D.data, row))  # sort by row, then value desc
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order)) - np.repeat(D.indptr[:-1], nnz)
    keep = rank < k
    out = sp.csr_matrix((D.data[keep], D.indices[keep], np.r_[0, np.cumsum(np.minimum(nnz, k))]),
                        shape=D.shape)
    out.sort_indices()
    return out


# ---------------------------------------------------------------- codebooks
def lloyd_1d(x, n_levels, iters=100):
    """Lloyd-Max levels for a 1-D empirical sample, quantile init; returns (levels, boundaries)."""
    x = np.sort(x.astype(np.float64))
    lv = np.quantile(x, (np.arange(n_levels) + 0.5) / n_levels)
    lv = np.unique(lv)
    for _ in range(iters):
        b = (lv[1:] + lv[:-1]) / 2
        idx = np.searchsorted(b, x)
        sums = np.bincount(idx, weights=x, minlength=len(lv))
        cnt = np.bincount(idx, minlength=len(lv))
        new = np.where(cnt > 0, sums / np.maximum(cnt, 1), lv)
        if np.allclose(new, lv, rtol=0, atol=1e-10):
            lv = new
            break
        lv = new
    return lv, (lv[1:] + lv[:-1]) / 2


class Codebook:
    """Scalar codebook over positive values. kind in uniform|lloyd|loglloyd."""

    def __init__(self, kind, bits, sample):
        self.kind, self.bits = kind, bits
        n = 2 ** bits
        if kind == "uniform":
            self.hi = float(sample.max())
            self.levels = (np.arange(n) + 0.5) * self.hi / n
            self.bounds = (np.arange(1, n)) * self.hi / n
        elif kind == "lloyd":
            self.levels, self.bounds = lloyd_1d(sample, n)
        elif kind == "loglloyd":
            lv, b = lloyd_1d(np.log(sample), n)
            self.levels, self.bounds = np.exp(lv), np.exp(b)
        else:
            raise ValueError(kind)

    def __call__(self, v):
        return self.levels[np.searchsorted(self.bounds, v)].astype(np.float32)


def quantize_docs(Dk, kind, bits, scale, sample_rows):
    """Return the decoded matrix. Codebook fitted on values of `sample_rows` (a row subset)."""
    data = Dk.data.astype(np.float64)
    nnz = np.diff(Dk.indptr)
    row = np.repeat(np.arange(Dk.shape[0]), nnz)
    if scale == "none":
        s = np.ones(Dk.shape[0])
    elif scale == "docmax":
        s = np.maximum.reduceat(data, Dk.indptr[:-1]) if len(data) else np.ones(Dk.shape[0])
        s = s.astype(np.float16).astype(np.float64)  # stored as fp16
    elif scale == "l2":
        s = np.sqrt(np.add.reduceat(data ** 2, Dk.indptr[:-1]))
    else:
        raise ValueError(scale)
    u = data / s[row]
    in_sample = np.isin(row, sample_rows)
    cb = Codebook(kind, bits, u[in_sample])
    uh = cb(u).astype(np.float64)
    if scale == "l2":  # remex renorm: rescale the decoded direction to the exact (fp16) norm
        nh = np.sqrt(np.add.reduceat(uh ** 2, Dk.indptr[:-1]))
        s = s.astype(np.float16).astype(np.float64) / np.maximum(nh, 1e-12)
    out = sp.csr_matrix(((uh * s[row]).astype(np.float32), Dk.indices, Dk.indptr), shape=Dk.shape)
    return out, cb


def binary_docs(Dk, scale):
    data = Dk.data.astype(np.float64)
    nnz = np.diff(Dk.indptr)
    row = np.repeat(np.arange(Dk.shape[0]), nnz)
    if scale == "none":
        v = np.ones_like(data)
    elif scale == "docmax":
        v = np.maximum.reduceat(data, Dk.indptr[:-1]).astype(np.float16).astype(np.float64)[row]
    elif scale == "l2":
        n2 = np.sqrt(np.add.reduceat(data ** 2, Dk.indptr[:-1])).astype(np.float16).astype(np.float64)
        v = (n2 / np.sqrt(np.maximum(nnz, 1)))[row]
    return sp.csr_matrix((v.astype(np.float32), Dk.indices, Dk.indptr), shape=Dk.shape)


def anserini_8bit(Dk):
    """Lucene/Anserini-style impact quantization: round(w / max * 255) on a corpus max, zeros dropped."""
    hi = float(Dk.data.max())
    q = np.round(Dk.data / hi * 255)
    keep = q > 0
    row = np.repeat(np.arange(Dk.shape[0]), np.diff(Dk.indptr))
    out = sp.csr_matrix(((q[keep] * hi / 255).astype(np.float32), (row[keep], Dk.indices[keep])),
                        shape=Dk.shape)
    out.sort_indices()
    return out


def query_8bit(Q):
    """Uniform 8-bit per query on [0, query max], round to nearest, zeros dropped."""
    Q = Q.copy()
    m = np.maximum.reduceat(Q.data, Q.indptr[:-1])
    row = np.repeat(np.arange(Q.shape[0]), np.diff(Q.indptr))
    Q.data = (np.round(Q.data / m[row] * 255) * m[row] / 255).astype(np.float32)
    Q.eliminate_zeros()
    return Q


# ---------------------------------------------------------------- bytes
def varint_len(g):
    g = np.asarray(g, dtype=np.int64)
    return 1 + (g >= 128) + (g >= 16384)  # vocab < 2**21 so at most 3 bytes


def index_bytes(Dk, value_bits, scale_bytes):
    """Mean bytes/doc: (fixed) 16-bit ids + packed values; (varint) delta-varint ids + packed values."""
    nnz = np.diff(Dk.indptr).astype(np.int64)
    vbytes = np.ceil(nnz * value_bits / 8)
    fixed = np.ceil(nnz * 16 / 8) + vbytes + scale_bytes
    gaps = np.diff(np.r_[0, Dk.indices.astype(np.int64)])
    first = Dk.indptr[:-1][nnz > 0]
    gaps[first] = Dk.indices[first]  # first id of each row coded from 0
    vl = np.add.reduceat(varint_len(gaps), Dk.indptr[:-1]) * (nnz > 0) if len(gaps) else np.zeros_like(nnz)
    varint = vl + vbytes + scale_bytes
    return float(fixed.mean()), float(varint.mean()), float(nnz.mean())


# ---------------------------------------------------------------- scoring
def score_topn(Q, D, n=100, exclude=None, tiekey=None):
    """Top-n doc indices per query by exact dot product. Score ties are broken as
    trec_eval does (doc id descending) when `tiekey` (rank of each doc id string) is given."""
    S = (Q @ D.T).toarray()
    if exclude is not None:
        for qi, di in exclude:
            S[qi, di] = -np.inf
    if tiekey is not None:  # widen the candidate set so ties at the boundary are resolved too
        top = np.argpartition(-S, n + 50, axis=1)[:, : n + 50]
    else:
        top = np.argpartition(-S, n, axis=1)[:, :n]
    ts = np.take_along_axis(S, top, 1)
    if tiekey is None:
        o = np.argsort(-ts, axis=1, kind="stable")
    else:
        o = np.lexsort((-tiekey[top], -ts), axis=1)
    return np.take_along_axis(top, o, 1)[:, :n]


def tie_key(dids):
    order = np.argsort(np.array(dids, dtype=object))
    key = np.empty(len(dids), dtype=np.int64)
    key[order] = np.arange(len(dids))
    return key


def metrics(top, qrels_idx, ideal10, npos=None):
    """Per-query nDCG@10 (linear gain, as trec_eval) and R@100 (rel > 0)."""
    nq = top.shape[0]
    if npos is None:
        npos = [sum(1 for v in r.values() if v > 0) for r in qrels_idx]
    nd = np.zeros(nq)
    rc = np.zeros(nq)
    disc = 1 / np.log2(np.arange(2, 12))
    for i in range(nq):
        rel = qrels_idx[i]
        g = np.array([rel.get(int(d), 0) for d in top[i, :10]], dtype=float)
        g = np.maximum(g, 0)
        nd[i] = (g * disc[: len(g)]).sum() / ideal10[i] if ideal10[i] > 0 else 0
        rc[i] = sum(1 for d in top[i] if rel.get(int(d), 0) > 0) / npos[i]
    return nd, rc


def prep_qrels(meta):
    did_ix = {d: i for i, d in enumerate(meta["dids"])}
    # Ideal DCG and the recall denominator use every judged doc, as trec_eval does,
    # including judged docs absent from the corpus (they exist in ArguAna and NFCorpus).
    qrels_idx, ideal, npos = [], [], []
    disc = 1 / np.log2(np.arange(2, 12))
    for q in meta["qids"]:
        full = meta["qrels"][q]
        qrels_idx.append({did_ix[d]: s for d, s in full.items() if d in did_ix})
        gains = sorted([s for s in full.values() if s > 0], reverse=True)[:10]
        ideal.append(float((np.array(gains) * disc[: len(gains)]).sum()))
        npos.append(sum(1 for s in full.values() if s > 0))
    exclude = [(qi, did_ix[q]) for qi, q in enumerate(meta["qids"]) if q in did_ix]  # BEIR ignore_identical_ids
    return qrels_idx, np.array(ideal), exclude, np.array(npos)


def boot_ci(delta, n=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    m = delta[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------- driver
def arms(quick=False):
    ks = [64, None] if quick else KS
    bits = [8, 2] if quick else BITS
    for k in ks:
        yield {"k": k, "codebook": "fp32", "bits": 32, "scale": "none"}
        yield {"k": k, "codebook": "fp16", "bits": 16, "scale": "none"}
        for cb in CODEBOOKS:
            for b in bits:
                for s in SCALES:
                    yield {"k": k, "codebook": cb, "bits": b, "scale": s}
        for s in SCALES:
            yield {"k": k, "codebook": "binary", "bits": 1, "scale": s}
        yield {"k": k, "codebook": "anserini8", "bits": 8, "scale": "none"}
        yield {"k": k, "codebook": "table8", "bits": 8, "scale": "none"}


def build(Dk, arm, sample_rows):
    cb, b, s = arm["codebook"], arm["bits"], arm["scale"]
    if cb == "fp32":
        return Dk, 32, 0
    if cb == "fp16":
        return sp.csr_matrix((Dk.data.astype(np.float16).astype(np.float32), Dk.indices, Dk.indptr),
                             shape=Dk.shape), 16, 0
    if cb == "binary":
        return binary_docs(Dk, s), 0, (0 if s == "none" else 2)
    if cb == "anserini8":
        return anserini_8bit(Dk), 8, 0
    if cb == "table8":  # exact: an 8-bit index into the table of distinct bf16-derived values
        n_distinct = len(np.unique(Dk.data))
        assert n_distinct <= 256, n_distinct
        return Dk, 8, 0
    Dq, _ = quantize_docs(Dk, cb, b, s, sample_rows)
    return Dq, b, (0 if s == "none" else 2)


def run_dataset(name, quick=False, variant=None, extra_arms=None):
    D, Q, meta = load(name, variant)
    qrels_idx, ideal, exclude, npos = prep_qrels(meta)
    rng = np.random.default_rng(RNG_SEED)
    sample_rows = rng.choice(D.shape[0], size=min(D.shape[0], 5000), replace=False)
    tk = tie_key(meta["dids"])
    base_top = score_topn(Q, D, exclude=exclude, tiekey=tk)
    base_nd, base_rc = metrics(base_top, qrels_idx, ideal, npos)
    records, perq = [], {"base_ndcg": base_nd, "base_r100": base_rc}
    Q8 = query_8bit(Q)
    t0 = time.time()
    arm_list = list(arms(quick)) + (extra_arms or [])
    cache_k = {}
    for i, arm in enumerate(arm_list):
        k = arm["k"]
        if k not in cache_k:
            cache_k[k] = topk_rows(D, k)
        Dk = cache_k[k]
        Dq, vbits, sbytes = build(Dk, arm, sample_rows)
        Qa = Q8 if arm.get("query") == "8bit" else Q
        top = score_topn(Qa, Dq, exclude=exclude, tiekey=tk)
        nd, rc = metrics(top, qrels_idx, ideal, npos)
        fixed, varint, nnz = index_bytes(Dq, vbits, sbytes)
        dn, dr = nd - base_nd, rc - base_rc
        rec = dict(arm, dataset=name, ndcg10=float(nd.mean()), r100=float(rc.mean()),
                   d_ndcg10=float(dn.mean()), d_ndcg10_ci=boot_ci(dn),
                   d_r100=float(dr.mean()), d_r100_ci=boot_ci(dr),
                   bytes_fixed=fixed, bytes_varint=varint, nnz=nnz, n_queries=len(nd))
        records.append(rec)
        key = f"{arm['k']}|{arm['codebook']}|{arm['bits']}|{arm['scale']}|{arm.get('query', 'fp32')}"
        perq[key] = nd
        print(f"[{name}] {i+1}/{len(arm_list)} {key:40s} nDCG {rec['ndcg10']:.4f} "
              f"d {rec['d_ndcg10']:+.4f} B {fixed:7.1f}/{varint:7.1f} {time.time()-t0:.0f}s", flush=True)
    out = {"dataset": name, "variant": variant, "n_docs": D.shape[0], "n_queries": len(base_nd),
               "base_ndcg10": float(base_nd.mean()), "base_r100": float(base_rc.mean()),
               "encode_meta": meta["meta"], "arms": records}
    RES.mkdir(exist_ok=True)
    suffix = "" if variant is None else f"_{variant}"
    (RES / f"arms_{name}{suffix}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(CACHE / f"perquery_{name}{suffix}.npz", keys=np.array(list(perq)),
                        vals=np.stack([perq[k] for k in perq]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q_arms = [{"k": None, "codebook": "fp32", "bits": 32, "scale": "none", "query": "8bit"},
              {"k": None, "codebook": "uniform", "bits": 8, "scale": "none", "query": "8bit"},
              {"k": 128, "codebook": "lloyd", "bits": 4, "scale": "docmax", "query": "8bit"},
              {"k": 64, "codebook": "lloyd", "bits": 4, "scale": "docmax", "query": "8bit"}]
    for n in a.datasets:
        run_dataset(n, quick=a.quick, extra_arms=None if a.quick else q_arms)


if __name__ == "__main__":
    main()
