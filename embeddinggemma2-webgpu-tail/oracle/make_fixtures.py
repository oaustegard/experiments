"""Build fixtures for the WebGPU tail from remex 2.1.0.

Synthetic EG2-shaped data: 768-d unit vectors in a cone around one mean
direction, ||mean|| / mean ||x|| ~ 0.85 and random-pair cosine ~ 0.72 (EG2 on
SciFact measured 0.845 and 0.72), plus 64 clusters so top-k is meaningful.
The kernel's correctness does not depend on the model, only on the numbers.

Usage: python3 -I oracle/make_fixtures.py OUTDIR [--n 4096] [--q 32]
"""
import argparse
import json
import os

import numpy as np
import remex
from remex.packing import pack

D, BITS, SEED, K = 768, 2, 42, 10


def unit(a):
    a = np.asarray(a, np.float64)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def make_vectors(rng, n, centers, m, c=0.85, spread=0.55):
    """Unit vectors: c*m + sqrt(1-c^2)*(cluster direction perturbed)."""
    cid = rng.integers(0, len(centers), n)
    g = unit(centers[cid] + spread * unit(rng.standard_normal((n, D))))
    # remove the component along m so c controls the cone angle exactly
    g = unit(g - (g @ m)[:, None] * m)
    return unit(c * m + np.sqrt(1 - c * c) * g)


def write(path, arr):
    np.ascontiguousarray(arr).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--q", type=int, default=32)
    ap.add_argument("--pool-b", type=int, default=6)
    ap.add_argument("--pool-s", type=int, default=37)
    a = ap.parse_args()
    rng = np.random.default_rng(20261018)
    os.makedirs(os.path.join(a.out, "expect"), exist_ok=True)

    m = unit(rng.standard_normal(D))
    centers = unit(rng.standard_normal((64, D)))
    docs64 = make_vectors(rng, a.n, centers, m)
    # queries: perturbed copies of random docs, so each has real neighbours
    src = rng.integers(0, a.n, a.q)
    queries64 = unit(docs64[src] + 0.35 * unit(rng.standard_normal((a.q, D))))

    docs = docs64.astype(np.float32)          # what remex encodes (unit)
    queries = queries64.astype(np.float32)
    mu = remex.corpus_mean(docs).astype(np.float32)

    # GPU input: the same rows times a random positive factor, so the
    # kernel's normalize step is exercised. remex sees the unit rows.
    fac = rng.uniform(0.5, 2.0, a.n).astype(np.float32)
    docs_in = (docs * fac[:, None]).astype(np.float32)
    qfac = rng.uniform(0.5, 2.0, a.q).astype(np.float32)
    queries_in = (queries * qfac[:, None]).astype(np.float32)

    qz = remex.Quantizer(D, bits=BITS, seed=SEED, mean=mu)
    R = np.ascontiguousarray(qz.R, np.float32)
    Rmu = (R.astype(np.float64) @ mu.astype(np.float64)).astype(np.float32)
    cv = qz.encode(docs)

    codes = cv.indices.astype(np.uint8)
    packed = pack(codes, BITS).reshape(a.n, D * BITS // 8)
    scale = qz._score_norms(cv).astype(np.float32)
    bias = qz._row_bias(cv).astype(np.float32)

    scores = np.empty((a.q, a.n), np.float32)
    topk = np.empty((a.q, K), np.int32)
    for qi in range(a.q):
        idx, sc = qz.search_adc(cv, queries[qi], k=a.n)
        scores[qi, idx] = sc
        topk[qi] = idx[:K]

    # float64 distance of every rotated unit residual to its nearest boundary
    r64 = docs.astype(np.float64) - mu.astype(np.float64)
    u64 = unit(r64) @ R.astype(np.float64).T
    b = qz.boundaries.astype(np.float64)
    margin = np.min(np.abs(u64[..., None] - b), axis=-1).astype(np.float32)

    # pool fixture: padded hidden states with a ragged mask
    B, S = a.pool_b, a.pool_s
    hidden = (rng.standard_normal((B, S, D)) * 0.7 + 0.1).astype(np.float32)
    lens = rng.integers(1, S + 1, B)
    lens[0] = S
    mask = (np.arange(S)[None, :] < lens[:, None]).astype(np.uint32)
    pooled = ((hidden.astype(np.float64) * mask[..., None]).sum(1)
              / mask.sum(1, keepdims=True)).astype(np.float32)

    E = os.path.join(a.out, "expect")
    for name, arr in {
        "R.f32": R, "Rmu.f32": Rmu, "mu.f32": mu,
        "docs_in.f32": docs_in, "queries_in.f32": queries_in,
        "docs_unit.f32": docs, "queries_unit.f32": queries,
        "hidden.f32": hidden, "mask.u32": mask,
    }.items():
        write(os.path.join(a.out, name), arr)
    for name, arr in {
        "codes.u8": codes, "packed.u8": packed,
        "len.f32": cv.norms.astype(np.float32),
        "mdot.f32": cv.mean_dots.astype(np.float32),
        "scale.f32": scale, "bias.f32": bias,
        "scores.f32": scores, "topk.i32": topk,
        "margin.f32": margin, "pooled.f32": pooled,
    }.items():
        write(os.path.join(E, name), arr)

    params = {
        "remex_version": __import__("importlib.metadata").metadata.version("remex"),
        "d": D, "bits": BITS, "seed": SEED, "rotation": qz.rotation,
        "words_per_vec": D * BITS // 32, "bytes_per_vec": D * BITS // 8,
        "kappa": float(qz._cosine_factor()),
        "boundaries": [float(v) for v in qz.boundaries],
        "centroids": [float(v) for v in qz.centroids],
        "n_docs": a.n, "n_queries": a.q, "k": K,
        "pool_b": B, "pool_s": S,
        "batches": [1000, 2000, a.n - 3000],  # encode in uneven batches
        "capacity": a.n + 904,                # > n, so offsets and padding matter
        "files": {
            "R": "R.f32", "Rmu": "Rmu.f32", "mu": "mu.f32",
            "docs_in": "docs_in.f32 [n_docs, d], NOT unit: encode with normalize=true",
            "queries_in": "queries_in.f32 [n_queries, d], NOT unit: search with normalize=true",
            "hidden": "hidden.f32 [pool_b, pool_s, d]", "mask": "mask.u32 [pool_b, pool_s]",
        },
        "stats": {
            "mean_norm_ratio": float(np.linalg.norm(docs64.mean(0))),
            "pair_cos": float(np.mean(np.sum(docs64[:500] * docs64[500:1000], 1))),
            "min_margin": float(margin.min()),
            "coords_margin_lt_1e-5": int((margin < 1e-5).sum()),
        },
    }
    with open(os.path.join(a.out, "params.json"), "w") as f:
        json.dump(params, f, indent=1)
    print(json.dumps(params["stats"]))


if __name__ == "__main__":
    main()
