"""Independent oracles for the remex search-path agreement harness.

Nothing here calls `Quantizer._effective_norms`, `_direction_lengths`,
`_score_norms`, `_row_bias`, `_query_offset`, `_cosine_factor`, any `search*`
method, `RHTOperator`, or `PackedVectors.unpack_*`.  What it does read from
production:

  * the codebook tables (`pq.centroids`, `pq._nested`, `pq.boundaries`) --
    a shared input, deliberately: area A is about scoring, not codebooks;
  * the dense rotation matrix `pq.R`  (so an RHTOperator that disagrees with
    its own dense matrix shows up as a path/oracle disagreement);
  * the stored per-row floats of the ground-truth `CompressedVectors`
    (`norms`, `mean_dots`) and its `indices`.

Everything is float64.  Scores are a pure function of
(codes, norms, mean_dots, query) -- so a `PackedVectors` built from the same
container must score identically to the container itself.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate


def codes_at(indices: np.ndarray, bits: int, precision):
    if precision is None or precision == bits:
        return indices
    return indices >> (bits - precision)


def table_at(pq, precision) -> np.ndarray:
    if precision is None or precision == pq.bits:
        return pq.centroids.astype(np.float64)
    return pq._nested[precision].astype(np.float64)


def _cos_pow(p):
    return lambda t: math.cos(t) ** p


_KAPPA = {}


def kappa_oracle(pq, precision) -> float:
    """E[u.v] / sqrt(E||v||^2) for a uniform unit u, by quadrature."""
    prec = None if precision == pq.bits else precision
    key = (pq.d, pq.bits, prec, pq.boundaries.tobytes(), pq.centroids.tobytes())
    if key in _KAPPA:
        return _KAPPA[key]
    d, bits = pq.d, pq.bits
    if d == 1:
        _KAPPA[key] = 1.0
        return 1.0
    cent_full = pq.centroids.astype(np.float64)
    if prec is None:
        cent = cent_full
    else:
        cent = pq._nested[prec].astype(np.float64)[np.arange(2 ** bits) >> (bits - prec)]
    # A uniform unit vector's coordinate z has density ~ (1 - z^2)^((d-3)/2).
    # With z = sin(theta) that is cos^(d-2)(theta) dtheta on [-pi/2, pi/2].
    p = d - 2
    Z = integrate.quad(_cos_pow(p), -math.pi / 2, math.pi / 2)[0]
    edges = np.concatenate(([-1.0], np.clip(pq.boundaries.astype(np.float64), -1.0, 1.0), [1.0]))
    th = np.arcsin(edges)
    prob = np.empty(len(cent))
    mz = np.empty(len(cent))
    for c in range(len(cent)):
        prob[c] = integrate.quad(_cos_pow(p), th[c], th[c + 1],
                                 epsabs=1e-13, epsrel=1e-13)[0] / Z
        # int sin(t) cos^p(t) dt = -cos^(p+1)(t) / (p+1)
        mz[c] = (math.cos(th[c]) ** (p + 1) - math.cos(th[c + 1]) ** (p + 1)) \
            / ((p + 1) * Z)
    # E[u.v] = d * sum_c cent_c E[z 1{c}] ;  E||v||^2 = d * sum_c cent_c^2 P(c)
    val = float(d * np.sum(cent * mz) / math.sqrt(d * np.sum(prob * cent * cent)))
    _KAPPA[key] = val
    return val


def oracle_scores(pq, cv, query, precision=None) -> np.ndarray:
    """Brute-force scores of every row of the ground-truth container `cv`.

    Uncentred / scalar:  q . decode(x)  (decode rebuilt from first principles).
    Centred (mean=):     the documented RaBitQ estimator
        q.mu + mu.r + ||r|| * (q - mu).u_hat / kappa
    with kappa recomputed by quadrature (kappa_oracle).
    """
    v = table_at(pq, precision)[codes_at(cv.indices, pq.bits, precision)]  # (n,d) rotated
    R = pq.R.astype(np.float64)
    q = np.asarray(query, dtype=np.float32).astype(np.float64)
    direction = v @ R                      # rows are R^T v  (= unrotate)
    if cv.norms is None:                   # scalar mode: the coordinates are the code
        return direction @ q
    norms = cv.norms.astype(np.float64)
    length = np.sqrt((v * v).sum(axis=1))
    if pq.mean is None:
        eff = norms / np.maximum(length, 1e-12) if pq.renorm else norms
        return eff * (direction @ q)
    mu = pq.mean.astype(np.float64)
    kappa = kappa_oracle(pq, precision)
    u = direction / length[:, None]
    return (q @ mu + cv.mean_dots.astype(np.float64)
            + norms * (u @ (q - mu)) / kappa)


def oracle_decode(pq, cv, precision=None) -> np.ndarray:
    """q-independent reconstruction, used only for the decode cross-check."""
    v = table_at(pq, precision)[codes_at(cv.indices, pq.bits, precision)]
    direction = v @ pq.R.astype(np.float64)
    if cv.norms is None:
        return direction
    norms = cv.norms.astype(np.float64)
    length = np.sqrt((v * v).sum(axis=1))
    eff = norms / np.maximum(length, 1e-12) if pq.renorm else norms
    return direction * eff[:, None]


def score_tol(pq, cv, query) -> float:
    """Absolute tolerance for float32 paths vs the float64 oracle.

    All production paths accumulate d float32 terms of size ~|score scale|;
    the centred estimator additionally cancels `mean_dots - scale * along`.
    """
    q = np.asarray(query, dtype=np.float32).astype(np.float64)
    qn = float(np.linalg.norm(q))
    if cv.norms is None:
        # scalar mode: terms are (centroid_j * q_rot_j), centroid scale ~ pq.scale
        cmax = float(np.abs(pq.centroids).max())
        return 2e-5 * (1.0 + qn * cmax * math.sqrt(pq.d))
    nmax = float(np.abs(cv.norms).max()) if len(cv.norms) else 0.0
    if pq.mean is None:
        return 2e-5 * (1.0 + qn * nmax)
    mn = float(np.linalg.norm(pq.mean))
    kmin = max(kappa_oracle(pq, 1), 0.1)
    return 2e-5 * (1.0 + (qn + mn) * (nmax / kmin + mn))


def check_topk(ids, scores, oracle, k_expected, tol, *, label="", pool=None,
               must_include_above=None):
    """Return (ok, message, tie_at_boundary).

    `ids/scores` are what a path returned; `oracle` is the full per-row oracle
    score vector.  `pool` restricts the legal rows (IVF partial probes,
    two-stage candidate sets); None means the whole corpus.

    Checks: dtype/shape/range/uniqueness, exact length, scores descending as
    returned, scores within `tol` of the oracle for the returned ids, and
    top-k optimality *modulo ties*: nothing outside the returned set may beat
    the worst returned row by more than 2*tol.  A boundary tie (a pool row
    outside the set within 2*tol of the worst returned) is reported, not
    hidden.
    """
    ids = np.asarray(ids)
    scores = np.asarray(scores)
    n = len(oracle)
    if ids.ndim != 1 or scores.ndim != 1 or ids.shape != scores.shape:
        return False, f"{label}: bad shapes ids{ids.shape} scores{scores.shape}", False
    if not np.issubdtype(ids.dtype, np.integer):
        return False, f"{label}: ids dtype {ids.dtype}", False
    if len(ids) != k_expected:
        return False, f"{label}: returned {len(ids)} results, expected {k_expected}", False
    if len(ids) == 0:
        return True, "", False
    if ids.min() < 0 or ids.max() >= n:
        return False, f"{label}: id out of range {ids}", False
    if len(np.unique(ids)) != len(ids):
        return False, f"{label}: duplicate ids {ids.tolist()}", False
    if not np.all(np.isfinite(scores)):
        return False, f"{label}: non-finite scores {scores}", False
    if np.any(np.diff(scores.astype(np.float64)) > 0):
        return False, f"{label}: scores not descending {scores.tolist()}", False
    err = np.abs(scores.astype(np.float64) - oracle[ids])
    if err.max() > tol:
        j = int(err.argmax())
        return (False, f"{label}: score {scores[j]!r} for id {ids[j]} vs oracle "
                       f"{oracle[ids[j]]!r} (err {err[j]:.3e} > tol {tol:.3e})", False)
    legal = np.ones(n, bool) if pool is None else pool
    if not legal[ids].all():
        return False, f"{label}: returned ids outside the legal pool", False
    worst = oracle[ids].min()
    outside = legal.copy()
    outside[ids] = False
    if outside.any():
        best_out = oracle[outside].max()
        if best_out > worst + 2 * tol:
            j = int(np.flatnonzero(outside)[oracle[outside].argmax()])
            return (False, f"{label}: row {j} (oracle {oracle[j]!r}) beats returned "
                           f"worst {worst!r}; not a top-{k_expected}", False)
        tie = best_out >= worst - 2 * tol
    else:
        tie = False
    return True, "", bool(tie)
