#!/usr/bin/env python3
"""Codebooks for axis C: scalar Lloyd-Max vs Gaussian-optimal vector grids.

Both are **data-oblivious** — fitted to the standard normal, never to the
corpus.  That is deliberate and it is the only way the comparison is fair:
remex's scalar Lloyd-Max needs no calibration set, so giving the vector arm a
corpus-fitted codebook would confound axis C with a fit/transfer advantage,
which `recall-per-byte` and `rotation-decorrelation` in this repo have both
already shown reverses under an honest protocol (METHODS.md, principle 3).

Scalar side: exact Lloyd-Max via the continuous fixed point (centroid of an
interval under the normal density in closed form), not sampling — so it
reproduces Max (1960) to the printed digits and the calibration gate can be a
hard assertion rather than a tolerance-shrug.

Vector side: Lloyd/LBG on N(0, I_m) samples with a KD-tree for the assignment
step, which is the Pagès-Printems CLVQ construction that HIGGS's "Gaussian
MSE-optimal grids" come from.  KD-tree assignment is what makes K=65536
tractable: brute force would be 2M x 65536 distance evaluations per iteration.

Grids are cached under assets/grids/ — training is the expensive part and the
grid depends only on (m, K), never on the corpus or the rotation seed.
"""
from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
GRID_CACHE = HERE / "assets" / "grids"

#: Max (1960) table 1 — MSE of the optimal fixed-rate scalar quantizer for a
#: unit-variance Gaussian.  Used as a hard calibration anchor in calibrate.py.
MAX_1960_MSE = {1: 0.3634, 2: 0.1175, 3: 0.03454, 4: 0.009497, 5: 0.002499}

#: Largest codebook we will build.  KD-tree query stays comfortable here and
#: it covers the QuIP#-relevant regime (2 bits x 8 dims = 2^16).
K_MAX = 1 << 16

#: Sub-vector dimensions we are willing to use, largest first.  m must divide
#: the ambient dimension, so the usable set is per-dataset.
M_CANDIDATES = (8, 6, 5, 4, 3, 2)


# --------------------------------------------------------------------------
# scalar Lloyd-Max


@cache
def lloyd_max_1d(bits: int, iters: int = 20_000, tol: float = 1e-15):
    """Optimal fixed-rate scalar quantizer levels for N(0,1).

    Memoized: the fixed point depends only on the rate, and the sweep rebuilds
    a codebook for every (arm, seed) cell.  Uncached this costs 3.9s at 8 bits
    and dominated the sweep.  Callers must not mutate the returned levels.

    Lloyd's algorithm on the continuous density.  With boundaries b and the
    normal pdf phi / cdf Phi, the conditional mean of an interval is
    (phi(a) - phi(b)) / (Phi(b) - Phi(a)); the nearest-neighbour condition
    puts boundaries at level midpoints.  Both steps are exact, so this
    converges to the published table rather than near it.

    Returns (levels, mse) with levels sorted ascending.
    """
    k = 1 << bits
    # init at the k-quantiles' midpoints — well inside the basin of the
    # optimum for every rate we use
    q = norm.ppf((np.arange(k) + 0.5) / k)
    levels = q.astype(np.float64)
    prev = np.inf
    for _ in range(iters):
        bnd = 0.5 * (levels[:-1] + levels[1:])
        edges = np.concatenate([[-np.inf], bnd, [np.inf]])
        cdf = norm.cdf(edges)
        pdf = norm.pdf(edges)
        p = np.diff(cdf)
        # E[X | interval] * P(interval) = phi(lo) - phi(hi)
        num = pdf[:-1] - pdf[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            new = np.where(p > 1e-300, num / np.maximum(p, 1e-300), levels)
        levels = new
        mse = 1.0 - float(np.sum(p * levels**2))
        if abs(prev - mse) < tol:
            break
        prev = mse
    bnd = 0.5 * (levels[:-1] + levels[1:])
    edges = np.concatenate([[-np.inf], bnd, [np.inf]])
    p = np.diff(norm.cdf(edges))
    mse = 1.0 - float(np.sum(p * levels**2))
    return levels.astype(np.float32), mse


def uniform_levels(bits: int, clip: float | None = None):
    """Naive uniform scalar grid over [-clip, clip] — the floor control.

    Default clip is the load factor that minimises MSE for a Gaussian at this
    rate, found by a coarse 1-D search, so the floor is an honest uniform
    quantizer and not a straw one.
    """
    k = 1 << bits
    if clip is None:
        best, clip = np.inf, 1.0
        for c in np.linspace(0.5, 6.0, 111):
            lv = np.linspace(-c, c, k) if k > 1 else np.array([0.0])
            m = _scalar_mse_gaussian(lv)
            if m < best:
                best, clip = m, c
    return np.linspace(-clip, clip, k).astype(np.float32) if k > 1 else np.zeros(1, np.float32)


def _scalar_mse_gaussian(levels):
    """MSE of an arbitrary scalar level set against N(0,1), in closed form."""
    levels = np.sort(np.asarray(levels, dtype=np.float64))
    bnd = 0.5 * (levels[:-1] + levels[1:])
    edges = np.concatenate([[-np.inf], bnd, [np.inf]])
    cdf, pdf = norm.cdf(edges), norm.pdf(edges)
    p = np.diff(cdf)
    ex = pdf[:-1] - pdf[1:]          # E[X 1_interval]
    # E[(X - y)^2 1] = E[X^2 1] - 2 y E[X 1] + y^2 P
    exx = _trunc_second_moment(edges)
    return float(np.sum(exx - 2 * levels * ex + levels**2 * p))


def _trunc_second_moment(edges):
    """E[X^2 1{a < X < b}] for N(0,1) = Phi(b)-Phi(a) - (b phi(b) - a phi(a))."""
    a, b = edges[:-1], edges[1:]
    fa = np.zeros_like(a)
    fb = np.zeros_like(b)
    ma, mb = np.isfinite(a), np.isfinite(b)
    fa[ma] = a[ma] * norm.pdf(a[ma])
    fb[mb] = b[mb] * norm.pdf(b[mb])
    return (norm.cdf(b) - norm.cdf(a)) - (fb - fa)


class ScalarCodebook:
    """1-D codebook applied per coordinate.  bytes = d * bits / 8."""

    kind = "scalar"

    def __init__(self, bits: int, levels: np.ndarray):
        self.bits = bits
        self.m = 1
        self.levels = np.asarray(levels, dtype=np.float32)
        self.bnd = (0.5 * (self.levels[:-1] + self.levels[1:])).astype(np.float32)

    def encode_decode(self, Y: np.ndarray) -> np.ndarray:
        """Y is already scaled to unit variance.  Returns the reconstruction."""
        if self.levels.size == 1:
            return np.zeros_like(Y)
        idx = np.searchsorted(self.bnd, Y)
        return self.levels[idx]


class VectorCodebook:
    """m-dimensional Gaussian-optimal grid.  bytes = (d/m) * (bits*m) / 8,
    i.e. exactly the same d*bits/8 as the scalar codebook — the two arms are
    matched on payload bytes by construction, and differ only in how the bits
    are spent."""

    kind = "vector"

    def __init__(self, bits: int, m: int, centroids: np.ndarray, mse: float):
        self.bits = bits
        self.m = m
        self.C = np.ascontiguousarray(centroids, dtype=np.float32)
        self.mse = mse
        self._tree = cKDTree(self.C)

    def encode_decode(self, Y: np.ndarray) -> np.ndarray:
        n, d = Y.shape
        sub = Y.reshape(-1, self.m)
        _, idx = self._tree.query(sub, k=1, workers=-1)
        return self.C[idx].reshape(n, d)


# --------------------------------------------------------------------------
# vector grid training


def product_init(bits: int, m: int) -> np.ndarray:
    """The m-fold Cartesian product of the optimal scalar quantizer.

    This is the *scalar* arm's codebook viewed as a point set in m dimensions,
    and it has exactly (2^bits)^m points — which is exactly K for every
    (bits, m) this experiment uses.  Starting Lloyd here rather than from
    random samples is the fix for the failure documented in RESULTS.md:
    random-init Lloyd converges to a local optimum that is *worse* than the
    scalar product quantizer at 6 and 8 bits, which would have handed axis C a
    fake win for the scalar arm.  Because Lloyd is monotone non-increasing in
    training distortion, seeding it here makes the vector arm provably no worse
    than the scalar arm, so axis C measures real vector-quantization gain
    rather than an optimizer artifact.
    """
    lv, _ = lloyd_max_1d(bits)
    g = np.meshgrid(*([lv] * m), indexing="ij")
    return np.stack([x.ravel() for x in g], axis=-1).astype(np.float32)


def _lloyd(X, K, iters, rng, log=None, init=None):
    """LBG with KD-tree assignment and largest-cell splitting for empties."""
    n, m = X.shape
    if init is not None:
        C = np.ascontiguousarray(init, dtype=np.float32).copy()
    else:
        C = X[rng.choice(n, size=K, replace=False)].astype(np.float32).copy()
    prev = np.inf
    for it in range(iters):
        tree = cKDTree(C)
        dist, idx = tree.query(X, k=1, workers=-1)
        mse = float(np.mean(dist**2)) / m
        cnt = np.bincount(idx, minlength=K).astype(np.float64)
        sums = np.stack([np.bincount(idx, weights=X[:, j], minlength=K)
                         for j in range(m)], axis=1)
        nz = cnt > 0
        C_new = C.copy()
        C_new[nz] = (sums[nz] / cnt[nz, None]).astype(np.float32)
        empty = np.flatnonzero(~nz)
        if empty.size:
            # split the highest-population cells rather than leaving codepoints
            # stranded: an unused codepoint is wasted rate, and would make the
            # vector arm look worse than the construction actually is
            donors = np.argsort(-cnt)[: empty.size]
            jitter = rng.normal(scale=1e-2, size=(empty.size, m)).astype(np.float32)
            C_new[empty] = C_new[donors] + jitter
        C = C_new
        if log and (it % 5 == 0 or it == iters - 1):
            log(f"      lloyd it={it:>3} mse/dim={mse:.6f} empty={empty.size}")
        if abs(prev - mse) < 1e-7 * max(mse, 1e-12):
            break
        prev = mse
    tree = cKDTree(C)
    dist, _ = tree.query(X, k=1, workers=-1)
    return C, float(np.mean(dist**2)) / m


def held_out_mse(C: np.ndarray, n: int = 2_000_000, seed: int = 11) -> float:
    """Per-dimension MSE of a codebook on fresh N(0, I_m) samples.

    Always held out.  Training-set MSE of a K-codepoint grid fitted on a few
    hundred samples per codepoint is measurably optimistic, and reporting it
    would silently strengthen the vector arm.
    """
    m = C.shape[1]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)).astype(np.float32)
    dist, _ = cKDTree(np.ascontiguousarray(C, np.float32)).query(X, k=1, workers=-1)
    return float(np.mean(dist ** 2)) / m


def train_gaussian_grid(m: int, K: int, *, iters: int = 25, seed: int = 20260802,
                        log=None) -> tuple[np.ndarray, float]:
    """Gaussian-optimal m-dim grid with K codepoints, cached on disk.

    Two candidates are trained and the better on **held-out** MSE is kept:

      1. Lloyd seeded from `product_init` — the scalar quantizer lifted to m
         dimensions.  This is the candidate that matters; it makes the vector
         arm provably no worse than the scalar arm.
      2. Lloyd from a random sample of the source, the textbook LBG init.

    Keeping the better of the two is not hedging: candidate 2 wins at low rate
    (where the shaping gain is large and the product grid's rectangular
    boundary is a real handicap) and candidate 1 wins at high rate, and picking
    per-configuration is what gives the vector arm its best honest shot.

    Sample count is 300 per codepoint, floor 400k, cap 6M.  The earlier 40 per
    codepoint was too thin: at K=4096 it cost ~25% in held-out MSE.
    """
    GRID_CACHE.mkdir(parents=True, exist_ok=True)
    path = GRID_CACHE / f"grid_m{m}_K{K}.npz"
    if path.exists():
        z = np.load(path)
        return z["C"], float(z["mse"])
    bits = round(math.log2(K) / m)
    n = int(min(6_000_000, max(400_000, 300 * K)))
    # Lloyd cost is O(iters * n * log K) with a large constant in 8 dimensions.
    # Trim iterations rather than samples on the biggest grids: too few samples
    # per codepoint is a correctness problem (the grid overfits its own
    # sample), too few iterations only leaves a little convergence on the
    # table, and the held-out check catches either way.
    if n * K > 2e11:
        n, iters = min(n, 4_000_000), min(iters, 15)
    rng = np.random.default_rng(seed + 1009 * m + K)
    X = rng.standard_normal((n, m)).astype(np.float32)
    if log:
        log(f"    training grid m={m} K={K} ({bits}b/coord) on {n:,} samples")
    cands = []
    if (1 << (bits * m)) == K:
        C1, _ = _lloyd(X, K, iters, rng, log=log, init=product_init(bits, m))
        cands.append(("product-init", C1, held_out_mse(C1)))
    C2, _ = _lloyd(X, K, iters, rng, log=log)
    cands.append(("random-init", C2, held_out_mse(C2)))
    for tag, _, v in cands:
        if log:
            log(f"      {tag}: held-out mse/dim={v:.7f}")
    tag, C, mse_ho = min(cands, key=lambda c: c[2])
    if log:
        log(f"      -> keeping {tag}")
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, C=C, mse=mse_ho, init=tag)
    tmp.replace(path)
    return C, mse_ho


def pick_m(bits: int, d: int) -> int:
    """Largest sub-vector dimension that divides d and keeps 2^(bits*m) <= K_MAX."""
    for m in M_CANDIDATES:
        if d % m == 0 and bits * m <= int(math.log2(K_MAX)):
            return m
    return 1


def build_codebook(kind: str, bits: int, d: int, *, log=None):
    if kind == "scalar":
        levels, _ = lloyd_max_1d(bits)
        return ScalarCodebook(bits, levels)
    if kind == "uniform":
        return ScalarCodebook(bits, uniform_levels(bits))
    if kind == "vector":
        m = pick_m(bits, d)
        if m == 1:
            levels, _ = lloyd_max_1d(bits)
            return ScalarCodebook(bits, levels)
        C, mse = train_gaussian_grid(m, 1 << (bits * m), log=log)
        return VectorCodebook(bits, m, C, mse)
    raise ValueError(kind)


# --------------------------------------------------------------------------
# E8 lattice — published-number anchor for the vector machinery


def e8_quantize(X: np.ndarray) -> np.ndarray:
    """Nearest point of the E8 lattice, Conway & Sloane Algorithm 1/2.

    E8 = D8 union (D8 + 1/2).  Decode to the nearer of the two cosets.  Used
    only by the calibration gate: E8's normalised second moment is a published
    constant (0.0716821), so reproducing it proves the vector-quantization
    plumbing is right independently of any grid we trained ourselves.
    """
    def _d8(Y):
        r = np.rint(Y)
        # D8 = even-sum integers: if the sum is odd, flip the coordinate with
        # the largest rounding error to its second-nearest integer
        bad = (r.sum(axis=1) % 2 != 0)
        if bad.any():
            err = Y[bad] - r[bad]
            j = np.argmax(np.abs(err), axis=1)
            rows = np.arange(j.size)
            r[np.flatnonzero(bad), j] += np.sign(err[rows, j]) * 1.0
        return r

    a = _d8(X)
    b = _d8(X - 0.5) + 0.5
    da = np.sum((X - a) ** 2, axis=1)
    db = np.sum((X - b) ** 2, axis=1)
    return np.where((da <= db)[:, None], a, b)


def e8_nsm(n: int = 400_000, seed: int = 7) -> float:
    """Normalised second moment of E8, estimated over its Voronoi cell.

    Sampling uniformly in a large box and quantizing gives, by the lattice's
    translation invariance, samples of the error vector distributed uniformly
    over the Voronoi region.  NSM = E[||e||^2] / (m * V^(2/m)); det(E8) = 1 so
    the cell volume V = 1.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-8, 8, size=(n, 8))
    e = X - e8_quantize(X)
    return float(np.mean(np.sum(e**2, axis=1))) / 8.0
