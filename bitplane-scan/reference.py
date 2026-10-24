#!/usr/bin/env python3
"""
Bit-plane dispersion scan: numpy CPU reference.

Progressive-precision exact top-K inner-product search over int8 vectors,
using two's-complement bit planes and provable residual bounds.

Usage:
  python3 reference.py                 # run the experiment suite, print tables
  python3 reference.py --quick         # smaller N, fewer queries
  python3 reference.py --export DIR    # write a small test vector set for the JS mirror test

Everything that web/index.html does on the GPU is mirrored here, operation by
operation, including float32 rounding of the bounds (see bounds_f32).
"""
import argparse
import json
import os
import time

import numpy as np

# ----------------------------------------------------------------------------
# Quantization and packing
# ----------------------------------------------------------------------------

PLANE_W = np.array([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.int64)  # weight of bit b


def quantize(X):
    """Symmetric per-vector int8: x_q = round(x / s), s = max|x| / 127."""
    X = np.asarray(X, dtype=np.float32)
    m = np.max(np.abs(X), axis=1)
    m = np.maximum(m, np.float32(1e-12))
    s = (m / np.float32(127.0)).astype(np.float32)
    q = np.clip(np.rint(X / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def pack_planes(xq):
    """int8 [N, D] -> uint32 planes [8, N, D/32]; bit j%32 of word j//32 of plane b
    is bit b of the two's-complement byte of component j (little-endian bit order)."""
    N, D = xq.shape
    assert D % 32 == 0
    u = xq.view(np.uint8)
    W = D // 32
    planes = np.empty((8, N, W), dtype=np.uint32)
    shifts = np.arange(32, dtype=np.uint32)
    for b in range(8):
        bits = ((u >> b) & 1).astype(np.uint32).reshape(N, W, 32)
        planes[b] = np.bitwise_or.reduce(bits << shifts, axis=2)
    return planes


def residual_norms(xq):
    """R[r, i] = || e_r(x_i) - c_r ||_2 for r = 0..7 unseen low planes,
    e_r = low r bits of the byte (unsigned, in [0, 2^r - 1]), c_r = (2^r - 1)/2.
    Stored as float32 rounded *up* so the Cauchy-Schwarz bound stays safe."""
    u = xq.view(np.uint8).astype(np.int64)
    N = u.shape[0]
    R = np.zeros((8, N), dtype=np.float32)
    for r in range(1, 8):
        e2 = 2 * (u & ((1 << r) - 1)) - ((1 << r) - 1)  # 2*(e - c), integer
        ss = (e2 * e2).sum(axis=1).astype(np.float64) / 4.0
        v = np.sqrt(ss).astype(np.float32)
        R[r] = np.nextafter(v, np.float32(np.inf))
    return R


def query_planes(q):
    return pack_planes(q[None, :])[:, 0, :]  # [8, W]


# ----------------------------------------------------------------------------
# Popcount partial dot: q . bitplane_b(x) = sum_c w_c * popcount(Q_c & X_b)
# ----------------------------------------------------------------------------

def plane_dot(qp, Xb):
    """qp: [8, W] query planes; Xb: [M, W] candidate plane b. Returns int64 [M]."""
    acc = np.zeros(Xb.shape[0], dtype=np.int64)
    for c in range(8):
        acc += PLANE_W[c] * np.bitwise_count(Xb & qp[c]).sum(axis=1, dtype=np.int64)
    return acc


# ----------------------------------------------------------------------------
# Bounds (float32, mirrored 1:1 by the WGSL `score` kernel and the page's JS)
# ----------------------------------------------------------------------------

F = np.float32


def bounds_f32(qhi, s, Rr, r, sumq, l1q, l2q):
    """Return (lb, ub, mid) float32 arrays for candidates with partial dot qhi
    (exact integer q.hi_r), scale s, residual norm Rr, r unseen planes."""
    if r == 0:
        sc = (s * qhi.astype(F)).astype(F)
        return sc, sc, sc
    c = F(((1 << r) - 1) * 0.5)
    hc = F(c * F(sumq))                       # q . c_r
    center = (qhi.astype(F) + hc).astype(F)
    w = np.minimum(F(c * F(l1q)), (F(l2q) * Rr).astype(F)).astype(F)
    w = (w * F(1.00001) + np.abs(center) * F(1e-6) + F(4.0)).astype(F)
    lb = (s * (center - w).astype(F)).astype(F)
    ub = (s * (center + w).astype(F)).astype(F)
    mid = (s * center).astype(F)
    return lb, ub, mid


def f32_key(v):
    """Order-preserving float32 -> uint32 (same as WGSL fkey)."""
    b = np.asarray(v, dtype=np.float32).view(np.uint32)
    neg = (b >> 31).astype(bool)
    return np.where(neg, ~b, b | np.uint32(0x80000000)).astype(np.uint32)


def kth_largest(keys, K):
    """K-th largest key, or 0 (keeps everything) if fewer than K items."""
    if keys.size <= K:
        return np.uint32(0)
    return np.partition(keys, keys.size - K)[keys.size - K]


def topk_from_keys(idx, keys, K):
    """Exact top-K by (key desc, index asc) - same rule as the page."""
    order = np.lexsort((idx, -keys.astype(np.int64)))
    return idx[order[:K]]


# ----------------------------------------------------------------------------
# Brute force and the scan
# ----------------------------------------------------------------------------

def brute_force(xq, s, q, K):
    dot = xq.astype(np.int32) @ q.astype(np.int32)
    keys = f32_key((s * dot.astype(F)).astype(F))
    return topk_from_keys(np.arange(xq.shape[0]), keys, K), dot


def aux_read_bytes(Mi, p, r, last):
    """Non-corpus bytes READ by one pass of the GPU pipeline in web/index.html, per
    input candidate (writes not counted):
      score   : scale s (4) + residual norm R_r (4, if r>0) + carried idx, qhi (8, if p>0)
      select  : 4 radix-histogram rounds over the 32-bit lower-bound key (16)
      compact : upper-bound key (4) + partial dot qhi (4, unless final) + idx (4, if p>0)"""
    return Mi * (4 + (4 if r > 0 else 0) + (8 if p > 0 else 0) + 16 + 4
                 + (0 if last else 4) + (4 if p > 0 else 0))


def brute_read_bytes(N, D):
    """int8 corpus (D) + scale (4) + 4 histogram rounds (16) + compaction key (4)."""
    return N * (D + 4 + 16 + 4)


def bitplane_scan(planes, s, R, q, K, schedule=(3, 1, 1, 1, 1, 1), budget=8,
                  heuristic_sigma=None, collect=False):
    """Progressive scan. schedule: planes consumed per pass (sum 8).
    budget: total planes allowed; if < 8 the final ranking uses bound midpoints.
    heuristic_sigma: if set, replace the provable half-width by
      sigma * ||q|| * R / sqrt(D)  (NOT exact; used only to size what a
      statistical bound would buy).
    Returns (topk_idx, stats list per pass)."""
    _, N, W = planes.shape
    D = W * 32
    qp = query_planes(q)
    qi = q.astype(np.int64)
    sumq = int(qi.sum()); l1q = int(np.abs(qi).sum())
    l2q = F(np.sqrt(float((qi * qi).sum())))
    l2q = np.nextafter(l2q, F(np.inf))

    idx = np.arange(N)
    qhi = np.zeros(N, dtype=np.int64)
    seen = 0
    stats = []
    bytes_plane_row = D // 8
    for p, k in enumerate(schedule):
        if seen >= budget:
            break
        k = min(k, budget - seen)
        Mi = idx.size
        for b in range(7 - seen, 7 - seen - k, -1):
            qhi = qhi + PLANE_W[b] * plane_dot(qp, planes[b, idx])
        seen += k
        r = 8 - seen
        if heuristic_sigma is not None and r > 0:
            c = F(((1 << r) - 1) * 0.5)
            center = (qhi.astype(F) + F(c * F(sumq))).astype(F)
            w = (F(heuristic_sigma) * l2q * R[r, idx] / F(np.sqrt(D))).astype(F)
            lb = (s[idx] * (center - w)).astype(F); ub = (s[idx] * (center + w)).astype(F)
            mid = (s[idx] * center).astype(F)
        else:
            lb, ub, mid = bounds_f32(qhi, s[idx], R[r, idx], r, sumq, l1q, l2q)
        last = (seen >= budget) or (p == len(schedule) - 1)
        if last:
            # final ranking key: exact score when r == 0 (lb == ub == mid),
            # bound midpoint when stopping early at a plane budget.
            rk = f32_key(mid)
            theta = kth_largest(rk, K)
            keep = rk >= theta
        else:
            lbk, ubk = f32_key(lb), f32_key(ub)
            theta = kth_largest(lbk, K)    # K-th best lower bound (as a sortable key)
            keep = ubk >= theta            # upper bound can still reach the top K
        corpus_bytes = Mi * k * bytes_plane_row
        aux_bytes = aux_read_bytes(Mi, p, r, last)
        st = dict(pass_=p + 1, planes=k, seen=seen, r=r, n_in=int(Mi),
                  n_out=int(keep.sum()), corpus_bytes=int(corpus_bytes),
                  aux_bytes=int(aux_bytes))
        if collect and r > 0:
            st["halfwidth_med"] = float(np.median((ub - lb) / 2))
        stats.append(st)
        if last:
            return topk_from_keys(idx[keep], rk[keep], K), stats
        idx, qhi = idx[keep], qhi[keep]
    raise RuntimeError("schedule exhausted")


# ----------------------------------------------------------------------------
# Data sets
# ----------------------------------------------------------------------------

def gen_gaussian(rng, N, D):
    return rng.standard_normal((N, D), dtype=np.float32)


def gen_clustered(rng, N, D, C=1000, spread=0.35, mean_frac=0.4, alpha=1.0):
    """Mixture of anisotropic Gaussians on the unit sphere, loosely shaped like
    text embeddings: a shared mean direction (the 'cone' effect), cluster
    centres drawn with a power-law spectrum lambda_j ~ j^-alpha in a random
    orthogonal basis B, and within-cluster noise that is half low-rank
    anisotropic (top M_LOW directions of B, same spectrum) and half isotropic.
    web/index.html implements the same construction (its own RNG, so not the same
    numbers). Returns (X, sampler); sampler(n) draws new points (queries)."""
    M_LOW = 32
    spec = (np.arange(1, D + 1, dtype=np.float32) ** (-alpha / 2)).astype(np.float32)
    basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
    basis = basis.astype(np.float32)
    lowB = basis[:, :M_LOW]
    lowS = spec[:M_LOW] / np.sqrt((spec[:M_LOW] ** 2).sum())

    def rot(Z):
        return Z @ basis.T

    mu = rng.standard_normal(D).astype(np.float32); mu /= np.linalg.norm(mu)
    cen = rot(rng.standard_normal((C, D), dtype=np.float32) * spec)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    cen = (1 - mean_frac) * cen + mean_frac * mu
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)

    def sample(n):
        lab = rng.integers(0, C, n)
        low = (rng.standard_normal((n, M_LOW), dtype=np.float32) * lowS) @ lowB.T
        iso = rng.standard_normal((n, D), dtype=np.float32) / np.sqrt(D)
        noise = spread * np.sqrt(0.5) * (low + iso)
        X = cen[lab] + noise
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X.astype(np.float32)

    return sample(N), sample


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------

def run_dataset(name, X, Qf, K, schedules, out):
    xq, s = quantize(X)
    planes = pack_planes(xq)
    R = residual_norms(xq)
    qq, _ = quantize(Qf)
    N, D = xq.shape
    # sanity: popcount plane dot == direct dot
    q0 = qq[0].astype(np.int64)
    direct = np.zeros(N, dtype=np.int64)
    qp = query_planes(qq[0])
    for b in range(8):
        bits = ((xq.view(np.uint8) >> b) & 1).astype(np.int64)
        assert np.array_equal(bits[:2000] @ q0, plane_dot(qp, planes[b, :2000]))
        direct += PLANE_W[b] * (bits @ q0)
    assert np.array_equal(direct, xq.astype(np.int64) @ q0), "plane decomposition broken"

    # score distribution diagnostics (cosine of float vectors)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Qn = Qf / np.linalg.norm(Qf, axis=1, keepdims=True)
    cos = Qn[:4] @ Xn.T
    srt = -np.sort(-cos, axis=1)
    diag = dict(top1_cos=float(srt[:, 0].mean()), topK_cos=float(srt[:, K - 1].mean()),
                median_cos=float(np.median(cos)), std_cos=float(cos.std()))

    # bound tightness, first query: median half-width after (8-r) planes, relative
    # to the gap between the K-th best exact score and the median exact score.
    qv = qq[0]; qi = qv.astype(np.int64)
    exact = (xq.astype(np.int64) @ qi)
    srt_e = np.sort(exact)[::-1]
    gap = float(srt_e[K - 1] - np.median(exact))
    l2q = np.nextafter(F(np.sqrt(float((qi * qi).sum()))), F(np.inf))
    u = xq.view(np.uint8).astype(np.int64)
    hw = {}
    for r in range(1, 8):
        hi = (u >> r << r).astype(np.int64); hi = np.where(hi >= 128, hi - 256, hi)
        qhi = hi @ qi
        lb, ub, _ = bounds_f32(qhi, np.ones(N, F), R[r], r, int(qi.sum()), int(np.abs(qi).sum()), l2q)
        assert np.all(lb <= exact) and np.all(ub >= exact), "bound violated"
        hw[8 - r] = float(np.median((ub - lb) / 2)) / gap
    diag["halfwidth_over_gap"] = hw
    res = dict(dataset=name, N=N, D=D, K=K, nq=len(qq), diag=diag, schedules={})
    brute_bytes = brute_read_bytes(N, D)
    res["brute_bytes"] = brute_bytes
    for label, sched, budget, hsig in schedules:
        agg = None; exact_all = True; recall = []
        for qv in qq:
            ref, _ = brute_force(xq, s, qv, K)
            top, st = bitplane_scan(planes, s, R, qv, K, sched, budget, hsig)
            same = np.array_equal(top, ref)
            exact_all &= same
            recall.append(len(set(top.tolist()) & set(ref.tolist())) / K)
            if agg is None:
                agg = [dict(d, n_out=[d["n_out"]], n_in=[d["n_in"]],
                            corpus_bytes=[d["corpus_bytes"]], aux_bytes=[d["aux_bytes"]]) for d in st]
            else:
                for a, d in zip(agg, st):
                    for f in ("n_out", "n_in", "corpus_bytes", "aux_bytes"):
                        a[f].append(d[f])
        passes = []
        for a in agg:
            passes.append(dict(pass_=a["pass_"], planes=a["planes"], seen=a["seen"], r=a["r"],
                               n_in=float(np.mean(a["n_in"])), n_out=float(np.mean(a["n_out"])),
                               surv_frac=float(np.mean(a["n_out"]) / N),
                               corpus_bytes=float(np.mean(a["corpus_bytes"])),
                               aux_bytes=float(np.mean(a["aux_bytes"]))))
        tot = sum(p["corpus_bytes"] + p["aux_bytes"] for p in passes)
        res["schedules"][label] = dict(schedule=list(sched), budget=budget, heuristic_sigma=hsig,
                                       exact_all=bool(exact_all), recall_mean=float(np.mean(recall)),
                                       passes=passes, total_bytes=tot,
                                       bytes_vs_brute=tot / brute_bytes)
    out.append(res)
    return res


def print_res(res):
    d = res["diag"]
    print(f"\n=== {res['dataset']}  N={res['N']} D={res['D']} K={res['K']} queries={res['nq']}")
    print(f"    cosine: top1 {d['top1_cos']:.3f}  top{res['K']} {d['topK_cos']:.3f}  "
          f"median {d['median_cos']:.3f}  std {d['std_cos']:.3f}")
    print("    bound half-width / (topK - median) gap, by planes seen: " +
          "  ".join(f"{k}:{v:.2f}" for k, v in sorted(d["halfwidth_over_gap"].items())))
    for label, sc in res["schedules"].items():
        print(f"  [{label}] exact={sc['exact_all']} recall@K={sc['recall_mean']:.3f} "
              f"bytes/brute={sc['bytes_vs_brute']:.3f}")
        for p in sc["passes"]:
            print(f"     pass {p['pass_']}: +{p['planes']} planes (seen {p['seen']}, r={p['r']}) "
                  f"in {p['n_in']:.0f} -> out {p['n_out']:.1f} ({100*p['surv_frac']:.2f}% of N)  "
                  f"bytes {p['corpus_bytes']/1e6:.2f}+{p['aux_bytes']/1e6:.2f} MB")


def export_testvec(path, seed=7):
    """Small corpus + queries + expected per-pass survivors for the JS mirror test."""
    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(seed)
    N, D, K = 4000, 256, 10
    X, sampler = gen_clustered(rng, N, D, C=40, spread=0.3)
    Q = sampler(5)
    xq, s = quantize(X); qq, _ = quantize(Q)
    planes = pack_planes(xq); R = residual_norms(xq)
    cases = []
    for qv in qq:
        ref, _ = brute_force(xq, s, qv, K)
        top, st = bitplane_scan(planes, s, R, qv, K)
        assert np.array_equal(top, ref)
        cases.append(dict(topk=ref.tolist(), n_out=[d["n_out"] for d in st]))
    xq.tofile(os.path.join(path, "xq.i8")); s.tofile(os.path.join(path, "s.f32"))
    qq.tofile(os.path.join(path, "q.i8"))
    R.tofile(os.path.join(path, "R.f32")); planes.tofile(os.path.join(path, "planes.u32"))
    json.dump(dict(N=N, D=D, K=K, nq=len(qq), cases=cases), open(os.path.join(path, "meta.json"), "w"))
    print("exported", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--export")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.export:
        export_testvec(a.export); return
    N = 20000 if a.quick else 100000
    nq = 4 if a.quick else 16
    D = 768
    full = (3, 1, 1, 1, 1, 1)
    schedules = [
        ("exact 3+1+1+1+1+1", full, 8, None),
        ("exact 2+1+1+1+1+1+1", (2, 1, 1, 1, 1, 1, 1), 8, None),
        ("exact 4+4", (4, 4), 8, None),
        ("budget 3 planes", full, 3, None),
        ("budget 4 planes", full, 4, None),
        ("heuristic 6sigma (NOT exact)", full, 8, 6.0),
    ]
    out = []
    t0 = time.time()
    for K in (10, 100):
        rng = np.random.default_rng(1)
        X = gen_gaussian(rng, N, D); Q = gen_gaussian(rng, nq, D)
        print_res(run_dataset(f"gaussian iid", X, Q, K, schedules, out))
        for spread in (0.35, 0.6):
            rng = np.random.default_rng(2)
            X, sampler = gen_clustered(rng, N, D, C=1000, spread=spread)
            print_res(run_dataset(f"clustered spread={spread}", X, sampler(nq), K, schedules, out))
        rng = np.random.default_rng(3)
        X, sampler = gen_clustered(rng, N, D, C=1000, spread=0.35)
        Q = X[rng.integers(0, N, nq)] + rng.standard_normal((nq, D), dtype=np.float32) * (0.15 / np.sqrt(D))
        print_res(run_dataset(f"clustered spread=0.35, near-duplicate queries", X, Q, K, schedules, out))
    print(f"\n({time.time()-t0:.0f}s)")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
