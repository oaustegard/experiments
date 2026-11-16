"""Independent reference computations for the remex stateful harness.

Nothing here touches a container's caches (``_x_hat_rot``, ``_dir_len``) or
calls a search method.  Scores are rebuilt in float64 from a container's raw
arrays (``indices``, ``norms``, ``mean_dots``) and a *separate* read-only
Quantizer that only supplies the codebook tables, the dense rotation matrix and
the centred-mode kappa.  Shared with production, hence common-mode: the
Lloyd-Max tables, ``_cosine_factor`` and the dense ``R`` construction.
"""
from __future__ import annotations

import collections
import functools

import numpy as np

from remex import Quantizer, PackedVectors


def codes_of(c) -> np.ndarray:
    """Raw (n, d) uint8 codes of a CompressedVectors or PackedVectors."""
    if isinstance(c, PackedVectors):
        return c.unpack_rows(0, c.n)
    return c.indices


def make_quantizer(params: dict) -> Quantizer:
    return Quantizer(**params)


@functools.lru_cache(maxsize=64)
def _oracle_quantizer(key):
    params = dict(key)
    mean = params.pop("mean", None)
    if mean is not None:
        params["mean"] = np.array(mean, dtype=np.float32)
    return Quantizer(**params)


def freeze(params: dict):
    items = []
    for k, v in sorted(params.items()):
        if k == "mean" and v is not None:
            v = tuple(float(x) for x in np.asarray(v, np.float32))
        items.append((k, v))
    return tuple(items)


class Oracle:
    """Float64 brute-force scores for one quantizer configuration."""

    def __init__(self, params: dict):
        self.pq = _oracle_quantizer(freeze(params))
        self.bits = self.pq.bits
        self.d = self.pq.d
        self.R = np.asarray(self.pq.R, dtype=np.float64)

    def table(self, p=None) -> np.ndarray:
        p = self.bits if p is None else p
        return np.asarray(self.pq._nested[p], dtype=np.float64)

    def rotated_codes(self, c, p=None) -> np.ndarray:
        p = self.bits if p is None else p
        codes = codes_of(c) >> (self.bits - p)
        return self.table(p)[codes]

    def kappa(self, p=None) -> float:
        return self.pq._cosine_factor(None if p in (None, self.bits) else p)

    def scores(self, c, q, p=None, R=None) -> np.ndarray:
        """Search scores of query ``q`` against every row of ``c`` at precision ``p``."""
        R = self.R if R is None else R
        pq = self.pq
        v = self.rotated_codes(c, p)
        q32 = np.asarray(q, dtype=np.float32)
        qrot = R @ q32.astype(np.float64)
        dots = v @ qrot
        if c.norms is None:
            return dots
        norms = np.asarray(c.norms, dtype=np.float64)
        lens = np.sqrt((v * v).sum(axis=1))
        if pq.mean is None:
            scale = norms / np.maximum(lens, 1e-12) if pq.renorm else norms
            return scale * dots
        kappa = self.kappa(p)
        scale = norms / (np.maximum(lens, 1e-12) * kappa)
        mu = pq.mean.astype(np.float64)
        bias = np.asarray(c.mean_dots, np.float64) - scale * (v @ (R @ mu))
        return scale * dots + bias + float(q32 @ pq.mean)

    def matrix(self, c, Q, p=None, R=None) -> np.ndarray:
        return np.stack([self.scores(c, q, p, R) for q in np.atleast_2d(Q)])

    def decode(self, c, p=None, R=None) -> np.ndarray:
        """Uncentred / scalar decode: scale * unrotate(centroids[codes])."""
        R = self.R if R is None else R
        v = self.rotated_codes(c, p)
        X = v @ R
        if c.norms is None:
            return X
        lens = np.sqrt((v * v).sum(axis=1))
        norms = np.asarray(c.norms, dtype=np.float64)
        scale = norms / np.maximum(lens, 1e-12) if self.pq.renorm else norms
        return X * scale[:, None]

    def direction_lengths(self, c, p=None) -> np.ndarray:
        v = self.rotated_codes(c, p)
        return np.sqrt((v * v).sum(axis=1))


def tolerance(ref: np.ndarray, extra: float = 0.0) -> float:
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0, extra)
    return 2e-4 * scale


def check_topk(ref, ids, scores, k, tol, allowed=None, what="search"):
    """Assert (ids, scores) is a valid top-k of the reference scores ``ref``.

    Tolerance-aware: ties and float32-vs-float64 rounding may reorder items
    whose reference scores are within ``tol``.  Checks length, id range and
    uniqueness, descending order, returned scores against the reference for the
    same ids, and both soundness (nothing returned that the reference ranks
    clearly below the cut) and completeness (nothing clearly above the cut
    omitted).  ``allowed`` restricts the universe (IVF candidate cells).
    """
    ref = np.asarray(ref, dtype=np.float64)
    ids = np.asarray(ids)
    scores = np.asarray(scores, dtype=np.float64)
    universe = np.arange(len(ref)) if allowed is None else np.asarray(sorted(allowed), dtype=np.int64)
    exp_len = min(int(k), len(universe))
    assert ids.shape == (exp_len,), f"{what}: returned {ids.shape[0]} ids, expected {exp_len}"
    assert scores.shape == (exp_len,), f"{what}: returned {scores.shape[0]} scores, expected {exp_len}"
    if exp_len == 0:
        return
    assert len(set(ids.tolist())) == exp_len, f"{what}: duplicate ids {ids.tolist()}"
    assert set(ids.tolist()) <= set(universe.tolist()), f"{what}: ids outside the searchable set {ids.tolist()}"
    assert np.all(np.diff(scores) <= tol), f"{what}: scores not descending {scores.tolist()}"
    got_ref = ref[ids]
    assert np.allclose(scores, got_ref, rtol=0, atol=tol), (
        f"{what}: returned scores {scores.tolist()} != reference scores of the same ids {got_ref.tolist()} (tol={tol:.2e})")
    pool = ref[universe]
    kth = np.sort(pool)[::-1][exp_len - 1]
    assert np.all(got_ref >= kth - tol), (
        f"{what}: returned ids {ids.tolist()} include one the reference ranks below the cut {kth:.6g}")
    clearly_above = universe[pool > kth + tol]
    missing = set(clearly_above.tolist()) - set(ids.tolist())
    assert not missing, f"{what}: ids {sorted(missing)} score clearly above the cut {kth:.6g} but were not returned"


def check_twostage(ref_fine, ref_coarse, ids, scores, k, candidates, tol, allowed=None, what="twostage"):
    """Assert a two-stage answer is consistent with fine (full-precision) and
    coarse (coarse-precision) reference scores.

    With an exhaustive candidate set the answer must be the exact fine top-k.
    Otherwise: sound (every id is in the coarse top-``candidates`` up to ties,
    scores equal the fine reference) and complete with respect to the ids that are
    *certainly* candidates.
    """
    ref_fine = np.asarray(ref_fine, dtype=np.float64)
    ref_coarse = np.asarray(ref_coarse, dtype=np.float64)
    ids = np.asarray(ids)
    scores = np.asarray(scores, dtype=np.float64)
    universe = np.arange(len(ref_fine)) if allowed is None else np.asarray(sorted(allowed), dtype=np.int64)
    coarse_k = min(int(candidates), len(universe))
    exp_len = min(int(k), coarse_k)
    assert ids.shape == (exp_len,) and scores.shape == (exp_len,), (
        f"{what}: returned {ids.shape} ids / {scores.shape} scores, expected {exp_len}")
    if exp_len == 0:
        return
    if coarse_k >= len(universe):
        check_topk(ref_fine, ids, scores, k, tol, allowed=allowed, what=what)
        return
    assert len(set(ids.tolist())) == exp_len, f"{what}: duplicate ids"
    assert np.all(np.diff(scores) <= tol), f"{what}: scores not descending"
    assert np.allclose(scores, ref_fine[ids], rtol=0, atol=tol), (
        f"{what}: scores {scores.tolist()} != fine reference {ref_fine[ids].tolist()}")
    pool_c = ref_coarse[universe]
    kth_c = np.sort(pool_c)[::-1][coarse_k - 1]
    assert np.all(ref_coarse[ids] >= kth_c - tol), f"{what}: an id is not in the coarse top-{coarse_k}"
    sure = universe[pool_c > kth_c + tol]
    if exp_len == len(ids) and k <= coarse_k:
        cut = ref_fine[ids].min()
        better = [i for i in sure.tolist() if ref_fine[i] > cut + tol]
        missing = set(better) - set(ids.tolist())
        assert not missing, f"{what}: certain candidates {sorted(missing)} outscore the weakest result"
    else:
        missing = set(sure.tolist()) - set(ids.tolist())
        assert not missing, f"{what}: certain candidates {sorted(missing)} were dropped"


class Events:
    """Event counter that also forwards to Hypothesis' event()."""

    def __init__(self):
        self.counts = collections.Counter()

    def __call__(self, label: str):
        from hypothesis import event
        from hypothesis.errors import InvalidArgument
        self.counts[label] += 1
        try:
            event(label)
        except InvalidArgument:          # called from a plain pytest test, not inside @given
            pass


EV = Events()
