"""Token-table codecs. Each returns (reconstructed float32 table, bytes on the wire).

int_g32   asymmetric min/max in groups of 32 along the row, fp16 scale + fp16 min per group
          (the layout of the shipped t3w/t2w tables): n * (512*b/8 + 64) bytes.
pq        faiss product quantizer, M sub-vectors of 512/M dims, 256 centroids each (one byte a
          sub-vector): n*M bytes + 256*512 fp16 codebook. Trained on rows drawn with probability
          proportional to a weight (token frequency), so frequent rows get the codebook's attention.
rpq       residual stages of PQ: stage s codes only the rows in tier s (top-N by weight), each stage
          coding the residual the earlier ones left. Stage 0 covers every row.
"""
import faiss  # noqa: I001
import numpy as np

D = 512


def int_g32(X, bits):
    X = np.asarray(X, np.float32)
    n = len(X)
    G = X.reshape(n, D // 32, 32)
    lo, hi = G.min(2, keepdims=True), G.max(2, keepdims=True)
    lo16 = lo.astype(np.float16).astype(np.float32)
    L = 2 ** bits - 1
    sc = ((hi - lo16) / L).astype(np.float16).astype(np.float32)
    sc[sc == 0] = 1e-8
    q = np.clip(np.rint((G - lo16) / sc), 0, L)
    R = (q * sc + lo16).reshape(n, D)
    return R, n * (D * bits // 8 + (D // 32) * 4)


def _sample(w, n_train, seed):
    rng = np.random.default_rng(seed)
    p = np.asarray(w, np.float64)
    p = p / p.sum()
    return rng.choice(len(p), size=n_train, replace=True, p=p)


def pq_fit(X, M, w=None, n_train=200_000, seed=0, nbits=8):
    X = np.ascontiguousarray(X, np.float32)
    idx = _sample(np.ones(len(X)) if w is None else w, n_train, seed)
    pq = faiss.ProductQuantizer(X.shape[1], M, nbits)
    pq.cp.seed = seed
    pq.cp.niter = 25
    pq.train(np.ascontiguousarray(X[idx]))
    return pq


def pq_apply(pq, X):
    X = np.ascontiguousarray(X, np.float32)
    codes = pq.compute_codes(X)
    return pq.decode(codes), codes


def pq_bytes(n, M, nbits=8):
    return n * M * nbits // 8 + (2 ** nbits) * D * 2


def rpq(X, stages, w, seed=0):
    """stages: list of (M, n_rows) — stage s codes the residual of the n_rows highest-weight rows
    (n_rows=None: all). Returns (R, bytes, order)."""
    X = np.asarray(X, np.float32)
    order = np.argsort(-np.asarray(w), kind="stable")
    R = np.zeros_like(X)
    total = 0
    for s, (M, nr) in enumerate(stages):
        sel = order if nr is None else order[:nr]
        res = X[sel] - R[sel]
        pq = pq_fit(res, M, w=np.asarray(w)[sel], seed=seed + s)
        dec, _ = pq_apply(pq, res)
        R[sel] += dec
        total += pq_bytes(len(sel), M)
    return R, total


def tiered(X, w, tiers):
    """tiers: list of (n_rows or None, codec) applied to rows in weight order; codec is
    ('int', bits) or ('pq', M) or ('rpq', [(M, n)...]) or ('zero',). Each tier coded independently."""
    X = np.asarray(X, np.float32)
    order = np.argsort(-np.asarray(w), kind="stable")
    R = np.zeros_like(X)
    total, start = 0, 0
    for nr, codec in tiers:
        sel = order[start:] if nr is None else order[start:start + nr]
        start += len(sel)
        if codec[0] == "int":
            r, b = int_g32(X[sel], codec[1])
        elif codec[0] == "pq":
            pq = pq_fit(X[sel], codec[1], w=np.asarray(w)[sel])
            r, _ = pq_apply(pq, X[sel])
            b = pq_bytes(len(sel), codec[1])
        elif codec[0] == "intres":  # PQ base then int-g32 on the residual
            pq = pq_fit(X[sel], codec[1], w=np.asarray(w)[sel])
            base, _ = pq_apply(pq, X[sel])
            rr, b2 = int_g32(X[sel] - base, codec[2])
            r, b = base + rr, pq_bytes(len(sel), codec[1]) + b2
        elif codec[0] == "zero":
            r, b = np.zeros((len(sel), D), np.float32), 0
        R[sel] = r
        total += b
    return R, total
