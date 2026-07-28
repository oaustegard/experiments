"""Komlós-constant lower-bound engine (issue #166, Target A).

The Komlós discrepancy of a matrix V (columns = vectors, ||col||_2 <= 1) is

    disc(V) = min_{eps in {+-1}^n} || V eps ||_inf .

Any explicit normalized matrix with certified disc(V) = X is a witness that
K(n) >= X for the Komlós constant at that size.

Two exact arithmetic paths:

* Z[sqrt2] path — Kunisky's tree matrices A^{T_k} (arXiv:2111.02974 eq. 13)
  have normalized entries +-2^{-b/2}, so 2^k * V = A + sqrt(2) * B with A, B
  integer matrices.  All values |(V eps)_i| are of the form (u + w*sqrt2)/2^k
  with integer u, w, and comparisons are decided exactly in integers.

* Fraction path — arbitrary rational matrices (rationalized search records),
  compared with Fraction arithmetic, no floats anywhere.

Both paths enumerate all 2^{n-1} sign vectors (eps and -eps are equivalent),
so results are exhaustive certificates, not heuristics, for n <= ~24.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np

SQRT2 = 2 ** 0.5


# ---------------------------------------------------------------------------
# Exact arithmetic over Z[sqrt2]
# ---------------------------------------------------------------------------

def sign_p_q_sqrt2(p: int, q: int) -> int:
    """Exact sign of p + q*sqrt(2) for integers p, q."""
    if p >= 0 and q >= 0:
        return 0 if p == 0 and q == 0 else 1
    if p <= 0 and q <= 0:
        return 0 if p == 0 and q == 0 else -1
    # p, q have opposite signs: sign decided by p^2 vs 2q^2
    if p > 0:  # q < 0: p + q*sqrt2 > 0  iff  p^2 > 2 q^2
        return 1 if p * p > 2 * q * q else (-1 if p * p < 2 * q * q else 0)
    # p < 0, q > 0: positive iff 2 q^2 > p^2
    return 1 if 2 * q * q > p * p else (-1 if 2 * q * q < p * p else 0)


def abs_sq_sqrt2(u: int, w: int) -> tuple[int, int]:
    """(u + w*sqrt2)^2 = (u^2 + 2w^2) + (2uw)*sqrt2, exact and >= 0."""
    return u * u + 2 * w * w, 2 * u * w


def cmp_abs_sqrt2(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Compare |a_u + a_w*sqrt2| vs |b_u + b_w*sqrt2| exactly. -1/0/+1."""
    pa, qa = abs_sq_sqrt2(*a)
    pb, qb = abs_sq_sqrt2(*b)
    return sign_p_q_sqrt2(pa - pb, qa - qb)


def sqrt2_to_float(u: int, w: int) -> float:
    return u + w * SQRT2


# ---------------------------------------------------------------------------
# Kunisky tree matrices  (arXiv:2111.02974, Section 3.1)
# ---------------------------------------------------------------------------

def kunisky_tree_matrix(k: int) -> np.ndarray:
    """Unnormalized A^{T_k}: rows = leaves (length-k binary strings, lex
    order), columns = dummy column of ones followed by internal vertices
    (binary strings of length < k, lex order by (length, string)).

    Entry (s, t) for internal vertex t:  +1 if t is a strict prefix of s and
    the next character of s is 0, -1 if the next character is 1, else 0.
    Shape: 2^k x 2^k.
    """
    n = 2 ** k
    leaves = ["".join(bits) for bits in product("01", repeat=k)]
    internal = [""]
    for depth in range(1, k):
        internal += ["".join(bits) for bits in product("01", repeat=depth)]
    A = np.zeros((n, n), dtype=np.int64)
    A[:, 0] = 1  # dummy variable column
    for col, t in enumerate(internal, start=1):
        for row, s in enumerate(leaves):
            if s.startswith(t) and len(s) > len(t):
                A[row, col] = 1 if s[len(t)] == "0" else -1
    return A


def kunisky_normalized_int_pair(k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalized Kunisky matrix as (A, B, scale) with V = (A + sqrt2*B)/scale.

    Column of internal vertex at depth a has 2^{k-a} nonzeros, l2 norm
    2^{(k-a)/2}; the dummy column has norm 2^{k/2}.  Normalized entries are
    +-2^{-(k-a)/2}.  With scale = 2^k every entry times the scale is
    2^{k-(k-a)/2} = integer or integer*sqrt2.
    """
    raw = kunisky_tree_matrix(k)
    n = 2 ** k
    scale = 2 ** k
    A = np.zeros((n, n), dtype=np.int64)
    B = np.zeros((n, n), dtype=np.int64)
    # column depths: col 0 (dummy) behaves like depth 0
    depths = [0, 0] if k >= 1 else [0]
    for d in range(1, k):
        depths += [d] * (2 ** d)
    for j in range(n):
        b = k - depths[j]          # entry magnitude is 2^{-b/2}
        # scale * 2^{-b/2} = 2^{k - b/2}
        if b % 2 == 0:
            A[:, j] = raw[:, j] * (2 ** (k - b // 2))
        else:
            B[:, j] = raw[:, j] * (2 ** (k - (b + 1) // 2))
    return A, B, scale


def kunisky_delta_lower_bound(k: int) -> float:
    """delta(A-hat^{T_k}) = 2^{-k/2} + sum_{a=1..k} 2^{-a/2} (paper eq. 14)."""
    return 2 ** (-k / 2) + sum(2 ** (-a / 2) for a in range(1, k + 1))


# ---------------------------------------------------------------------------
# Exact discrepancy, Z[sqrt2] matrices
# ---------------------------------------------------------------------------

def all_sign_vectors(n: int) -> np.ndarray:
    """(2^{n-1}, n) matrix of sign vectors with first coordinate fixed +1."""
    count = 1 << (n - 1)
    idx = np.arange(count, dtype=np.uint64)
    bits = ((idx[:, None] >> np.arange(n - 1, dtype=np.uint64)) & 1).astype(np.int64)
    signs = 1 - 2 * bits  # bit 0 -> +1, bit 1 -> -1
    return np.hstack([np.ones((count, 1), dtype=np.int64), signs])


def exact_disc_sqrt2(A: np.ndarray, B: np.ndarray, scale: int) -> dict:
    """Exact disc of V = (A + sqrt2*B)/scale by exhaustive enumeration.

    Float arithmetic is used only to ORDER candidates; every comparison that
    decides the result is redone in exact integer arithmetic on (u, w) pairs.
    Returns the optimal value both exactly and as float, plus the argmin eps.
    """
    m, n = A.shape
    S = all_sign_vectors(n)
    U = S @ A.T          # integer, exact (magnitudes are tiny)
    W = S @ B.T
    F = np.abs(U + SQRT2 * W)
    maxima = F.max(axis=1)
    fmin = maxima.min()
    # exact re-check of every eps whose float max is near the float min;
    # 1e-6 dwarfs the ~1e-12 float error at these integer magnitudes
    margin = 1e-6
    cand = np.nonzero(maxima <= fmin + margin)[0]
    best = None  # (u, w) of |value|, plus row index
    for ci in cand:
        row_best = None
        for i in range(m):
            pair = (int(U[ci, i]), int(W[ci, i]))
            if sign_p_q_sqrt2(*pair) < 0:
                pair = (-pair[0], -pair[1])
            if row_best is None or cmp_abs_sqrt2(pair, row_best) > 0:
                row_best = pair
        if best is None or cmp_abs_sqrt2(row_best, best[0]) < 0:
            best = (row_best, int(ci))
    (u, w), argmin_idx = best
    value = sqrt2_to_float(u, w) / scale
    # certificate sanity: the float minimum and the exact minimum must agree
    # to within float error
    assert abs(value - fmin / scale) < 1e-9, (value, fmin / scale)
    return {
        "disc_float": value,
        "disc_exact_pair": (u, w),
        "scale": scale,
        "disc_exact_str": f"({u} + {w}*sqrt(2))/{scale}",
        "argmin_eps": S[argmin_idx].tolist(),
        "n_sign_vectors_checked": int(S.shape[0]),
    }


# ---------------------------------------------------------------------------
# Exact discrepancy, rational (Fraction) matrices
# ---------------------------------------------------------------------------

def exact_disc_fraction(V_frac: list[list[Fraction]]) -> dict:
    """Exact disc of a rational matrix (rows = coordinates, cols = vectors).

    Float arithmetic ORDERS the 2^{n-1} sign vectors; every candidate within
    a margin of the float minimum is re-evaluated in pure Fraction
    arithmetic, and the certified minimum is decided exactly.  The margin
    (1e-6) exceeds the accumulated float error (entries <= 1 in magnitude,
    n <= ~24 terms => error ~ 1e-14) by ~8 orders of magnitude.
    """
    m = len(V_frac)
    n = len(V_frac[0])
    Vf = np.array([[float(x) for x in row] for row in V_frac])
    S = all_sign_vectors(n)
    E = S @ Vf.T
    maxima = np.abs(E).max(axis=1)
    fmin = float(maxima.min())
    margin = 1e-6
    cand = np.nonzero(maxima <= fmin + margin)[0]
    best: Fraction | None = None
    best_idx = -1
    for ci in cand:
        eps = S[ci]
        mx = Fraction(0)
        for i in range(m):
            v = sum(V_frac[i][j] * int(eps[j]) for j in range(n))
            if v < 0:
                v = -v
            if v > mx:
                mx = v
        if best is None or mx < best:
            best, best_idx = mx, int(ci)
    assert best is not None and abs(float(best) - fmin) < margin
    return {
        "disc_float": float(best),
        "disc_exact": best,
        "disc_exact_str": f"{best.numerator}/{best.denominator}",
        "argmin_eps": S[best_idx].tolist(),
        "n_sign_vectors_checked": int(S.shape[0]),
        "n_exact_rechecked": int(len(cand)),
    }


def column_norms_ok(V_frac: list[list[Fraction]]) -> bool:
    """Exact check: every column l2 norm <= 1, i.e. sum of squares <= 1."""
    m = len(V_frac)
    n = len(V_frac[0])
    for j in range(n):
        if sum(V_frac[i][j] ** 2 for i in range(m)) > 1:
            return False
    return True


# ---------------------------------------------------------------------------
# Float screening + smoothed max-min local search (outer maximization)
# ---------------------------------------------------------------------------

def float_disc(V: np.ndarray, signs: np.ndarray | None = None) -> float:
    """disc(V) in float; V has shape (d, n) (columns = vectors)."""
    n = V.shape[1]
    if signs is None:
        signs = all_sign_vectors(n)
    E = signs @ V.T
    return float(np.abs(E).max(axis=1).min())


def project_columns(V: np.ndarray) -> np.ndarray:
    """Project every column onto the unit l2 ball."""
    norms = np.linalg.norm(V, axis=0)
    scale = np.maximum(norms, 1.0)
    return V / scale


def softmaxmin_ascent(
    V0: np.ndarray,
    iters: int = 400,
    lr: float = 0.02,
    beta_start: float = 8.0,
    beta_end: float = 60.0,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Maximize disc(V) by gradient ascent on a softmin/softmax surrogate.

    surrogate(V) = softmin_eps softmax_i |(V eps)_i|, annealing the sharpness
    beta.  Exact inner enumeration (all 2^{n-1} eps), so this is a true local
    ascent on the piecewise-linear objective as beta -> inf.  Returns the
    best iterate under the REAL (hard) disc, projected to the unit balls.
    """
    rng = np.random.default_rng(seed)
    V = project_columns(V0.astype(np.float64).copy())
    n = V.shape[1]
    S = all_sign_vectors(n).astype(np.float64)
    best_V, best_val = V.copy(), float_disc(V, S)
    for it in range(iters):
        beta = beta_start + (beta_end - beta_start) * it / max(iters - 1, 1)
        E = S @ V.T                      # (P, d)
        Aabs = np.abs(E)
        # softmax over coordinates within each eps
        row_w = np.exp(beta * (Aabs - Aabs.max(axis=1, keepdims=True)))
        row_w /= row_w.sum(axis=1, keepdims=True)
        row_val = (row_w * Aabs).sum(axis=1)
        # softmin over eps
        eps_w = np.exp(-beta * (row_val - row_val.min()))
        eps_w /= eps_w.sum()
        # gradient wrt V: sum_eps eps_w * sum_i row_w * sign(E) * (e_i eps^T)
        G = ((eps_w[:, None] * row_w) * np.sign(E)).T @ S   # (d, n)
        V = project_columns(V + lr * G + 1e-4 * rng.standard_normal(V.shape))
        val = float_disc(V, S)
        if val > best_val:
            best_val, best_V = val, V.copy()
    return best_V, best_val


def rationalize(V: np.ndarray, max_den: int = 64) -> list[list[Fraction]]:
    """Round to Fractions with bounded denominator, then shrink any column
    whose exact norm exceeds 1 (certified instances need exact feasibility)."""
    d, n = V.shape
    F = [[Fraction(V[i, j]).limit_denominator(max_den) for j in range(n)]
         for i in range(d)]
    for j in range(n):
        ss = sum(F[i][j] ** 2 for i in range(d))
        if ss > 1:
            # shrink by a small-denominator rational just below 1/sqrt(ss),
            # nudging down until exactly feasible (keeps denominators small
            # so the exact certification stays cheap)
            factor = Fraction(1 / float(ss) ** 0.5).limit_denominator(128)
            while sum((F[i][j] * factor) ** 2 for i in range(d)) > 1:
                factor *= Fraction(127, 128)
            for i in range(d):
                F[i][j] *= factor
    return F
