"""Codecs for MICE-P's stored 128-d document token vectors.

MICE's interaction layers read a stored vector x only through
LayerNorm(W_up x) (doc_decompress has no bias, doc_cross_attn_norm has no
bias), so the score is invariant to a positive rescaling of x: only its
direction is information. Every codec here returns a decoded (N, 128) array
plus the bytes it stores per token.

  plaid      FastPLAID's residual codec, re-implemented: k-means centroids
             (L2-normalised, K = 2**floor(log2(16 sqrt(N)))), code = argmax x.c,
             residual x - c bucketed per dimension by global quantiles,
             decode = normalise(c + w). Inputs are NOT normalised first,
             matching what FastPLAID does with MICE's unnormalised vectors.
  plaid-n    the same on L2-normalised inputs (free under scale invariance).
  remex      rotation + Lloyd-Max codes of the direction; the norm is dropped
             (scale invariance), so a token costs d*bits/8 bytes.
  remex-c    remex centred on the corpus mean; keeps one per-token length
             (fp16, 2 B) because mu + m*u_hat needs m.
  remex-w    remex-c in a metric-aware basis: y = A(x - mu) with A from the
             SVD of the map the interaction layers apply to x, so squared
             error in y is squared error where the model reads it.
"""
import numpy as np
import remex

CODE_BYTES = 4  # centroid id; FastPLAID writes int64 (8 B), int32 is the fair floor


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


_KMEANS = {}


def _centroids(X, normalize_input, seed, sample_tokens, niter):
    """faiss k-means (as ColBERT's indexer), cached so nbits arms share one codebook."""
    import faiss
    key = (X.shape, float(X[:64].sum()), normalize_input, seed)
    if key not in _KMEANS:
        rng = np.random.default_rng(seed)
        N, d = X.shape
        K = int(2 ** np.floor(np.log2(16 * np.sqrt(N))))
        samp = np.ascontiguousarray(X[rng.choice(N, size=min(sample_tokens, N), replace=False)])
        km = faiss.Kmeans(d, K, niter=niter, seed=seed, max_points_per_centroid=256, verbose=False)
        km.train(samp)
        C = l2n(km.centroids.astype(np.float32))
        codes = np.concatenate([np.argmax(X[s:s + 65_536] @ C.T, axis=1) for s in range(0, N, 65_536)])
        _KMEANS[key] = (C, codes, samp)
    return _KMEANS[key]


def plaid(X, nbits, normalize_input=False, seed=0, sample_tokens=262_144, niter=4):  # FastPLAID: kmeans_niters=4
    X = l2n(X) if normalize_input else X
    d = X.shape[1]
    C, codes, samp = _centroids(X, normalize_input, seed, sample_tokens, niter)
    K = len(C)
    if nbits == 0:
        return l2n(C[codes]), CODE_BYTES, {"K": K}
    held = samp[: min(len(samp), 65_536)]
    res_h = held - C[np.argmax(held @ C.T, axis=1)]
    nb = 2 ** nbits
    cut = np.quantile(res_h.ravel(), np.arange(1, nb) / nb)
    wts = np.quantile(res_h.ravel(), (np.arange(nb) + 0.5) / nb).astype(np.float32)
    buckets = np.searchsorted(cut, X - C[codes])
    return l2n(C[codes] + wts[buckets]), d * nbits / 8 + CODE_BYTES, {"K": K}


def plaid_remex(X, bits, seed=0, sample_tokens=262_144, niter=4):
    """PLAID's centroid on the normalised vector, remex on the residual (RSLM-style).

    Stores centroid id + remex code of the residual direction + fp16 residual length."""
    Xn = l2n(X)
    C, codes, _ = _centroids(Xn, True, seed, sample_tokens, niter)
    R = Xn - C[codes]
    r_norm = np.linalg.norm(R, axis=1, keepdims=True).astype(np.float16).astype(np.float32)
    q = remex.Quantizer(X.shape[1], bits=bits, seed=42)
    u_hat = l2n(q.decode(q.encode(l2n(R))))
    return l2n(C[codes] + r_norm * u_hat), X.shape[1] * bits / 8 + CODE_BYTES + 2, {"K": len(C)}


def remex_plain(X, bits, seed=42):
    q = remex.Quantizer(X.shape[1], bits=bits, seed=seed)
    return l2n(q.decode(q.encode(l2n(X)))), X.shape[1] * bits / 8, {}


def remex_centered(X, bits, seed=42, A=None, mu=None):
    """Centred remex, optionally in a transformed basis y = A x (A invertible).

    The per-token length is rounded through fp16 to charge it 2 bytes honestly."""
    Y = X @ A.T if A is not None else X
    mu_y = Y.mean(axis=0) if mu is None else mu
    R = Y - mu_y
    r_norm = np.linalg.norm(R, axis=1, keepdims=True)
    q = remex.Quantizer(Y.shape[1], bits=bits, seed=seed)
    u_hat = q.decode(q.encode(l2n(R)))
    u_hat = l2n(u_hat)
    Y_hat = mu_y + r_norm.astype(np.float16).astype(np.float32) * u_hat
    X_hat = np.linalg.solve(A, Y_hat.T).T if A is not None else Y_hat
    return X_hat.astype(np.float32), X.shape[1] * bits / 8 + 2, {
        "anisotropy": float(np.linalg.norm(Y.mean(0)) / np.linalg.norm(Y, axis=1).mean())}


def metric_from_weights(st_file):
    """A with ||A e||^2 = the energy the interaction layers see from an error e in x.

    Two choices, both 128x128:
      'up'  : P W_up, where P removes the per-token mean LayerNorm subtracts.
      'kv'  : stack over the top layers of W_kv diag(gamma_l) P W_up (K and V rows
              of cross_attn.Wqkv applied after doc_cross_attn_norm).
    Returns {name: A} with A = Sigma V^T from the SVD of each map."""
    from safetensors.numpy import load_file
    W = load_file(st_file)
    W_up = W["doc_decompress.weight"]                      # (H, 128)
    H = W_up.shape[0]
    P = np.eye(H, dtype=np.float32) - 1.0 / H
    PW = P @ W_up
    maps = {"up": PW}
    blocks, l = [], 0
    while f"top_layers.{l}.cross_attn.Wqkv.weight" in W:
        g = W[f"top_layers.{l}.doc_cross_attn_norm.weight"]
        Wkv = W[f"top_layers.{l}.cross_attn.Wqkv.weight"][H:]    # K and V rows
        blocks.append(Wkv @ (g[:, None] * PW))
        l += 1
    maps["kv"] = np.concatenate(blocks)
    out = {}
    for k, M in maps.items():
        _, s, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
        out[k] = (s[:, None] * Vt).astype(np.float32)
        out[k + "_cond"] = float(s[0] / s[-1])
    return out


def score_err(X, X_hat, A):
    """Relative error of the direction as the model reads it, after mapping by A."""
    Y, Yh = l2n(X @ A.T), l2n(X_hat @ A.T)
    return float(np.mean(np.sum((Y - Yh) ** 2, axis=1)))
