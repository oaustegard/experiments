"""Hypothesis harness for remex area D: encode, codebooks, rotations, scalar mode.

Targets and oracles
-------------------
A. Metamorphic encode (chunking, block size, thread count, row permutation,
   negation, power-of-two scaling, repeated/fresh instances, native vs NumPy).
B. Codebooks: symmetry, strict monotonicity, float32-midpoint boundaries,
   nesting, power-of-two scale equivariance, an independent Lloyd-Max
   re-derivation (scipy.special.ndtr, own pdf), assign_codes vs np.searchsorted.
C. Rotations: orthogonality, RHTOperator vs the dense matrix, an independent
   float64 construction of RHT from the documented recipe, Haar vs the unique
   positive-diagonal np.linalg.qr factor, rounds >= 2, odd/tiny d.
D. Scalar mode: codes vs searchsorted at boundaries, equal inputs, monotone,
   decode == centroids, mode/scale/seed guards, scale validation.
E. Input edge cases: n=0/1, dtypes, strides, read-only, NaN/inf, corpus_mean,
   anisotropy.

Tests that exercise a *confirmed finding* are marked ``xfail(strict=True)`` so a
green run means "no unexpected failures"; the deterministic reproduction of each
lives in ``test_encode_regressions.py``.

Run (about 3-4 minutes):
    python3 -m pytest test_encode_props.py -q -p no:cacheprovider \
        --hypothesis-show-statistics
Failure persistence: ``.hypothesis/examples`` next to this file.
"""
from __future__ import annotations

import contextlib
import copy
import functools
import math
import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase
from hypothesis.extra import numpy as npst
from scipy.linalg import hadamard
from scipy.special import ndtr

import remex.core as core
from remex import (AnisotropyWarning, Quantizer, _native, anisotropy,
                   corpus_mean)
from remex import rotation as rot
from remex.codebook import (assign_codes, coordinate_sigma, lloyd_max_codebook,
                            nested_codebooks)
from remex.rotation import (RHTOperator, haar_rotation, rht_plan, rht_rotation,
                            rht_rounds, _largest_pow2_divisor)

HERE = os.path.dirname(os.path.abspath(__file__))
DB = (None if os.environ.get("ENCODE_NO_DB")
      else DirectoryBasedExampleDatabase(os.path.join(HERE, ".hypothesis", "examples")))


# ENCODE_EXAMPLES_SCALE shrinks every budget (the mutation runs use 0.4); 1 = full.
EXAMPLES_SCALE = float(os.environ.get("ENCODE_EXAMPLES_SCALE", "1"))


def S(n):
    return settings(max_examples=max(8, int(n * EXAMPLES_SCALE)), deadline=None, database=DB,
                    print_blob=True, suppress_health_check=list(HealthCheck))


BITS = (1, 2, 3, 4, 8)
SEED_POOL = (0, 1, 2, 7, 42, 12345, 2**31 - 1, 2**32 + 5)
EVEN_DIMS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 32, 34, 48, 64, 66, 96,
             100, 128, 130]
ODD_DIMS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 33, 65, 127]


def have_native():
    return _native.kernel() is not None


# ---------------------------------------------------------------------------
# environment switches
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def encode_env(block=None, threads=None, parallel_min=None):
    """Block size, thread count and parallel threshold; all restored on exit."""
    old_block, old_pm = core.ENCODE_BLOCK_FLOATS, rot.PARALLEL_MIN_FLOATS
    try:
        if block is not None:
            core.ENCODE_BLOCK_FLOATS = block
        if parallel_min is not None:
            rot.PARALLEL_MIN_FLOATS = parallel_min
        if threads is not None:
            rot.set_num_threads(threads)
        yield
    finally:
        core.ENCODE_BLOCK_FLOATS, rot.PARALLEL_MIN_FLOATS = old_block, old_pm
        rot.set_num_threads(None)


@contextlib.contextmanager
def numpy_path(*quantizers):
    """Force the NumPy implementation of both compiled kernels."""
    orig = _native.kernel
    saved = []
    _native.kernel = lambda: None
    try:
        for q in quantizers:
            if getattr(q, "_op", None) is not None:
                saved.append((q._op, q._op._kernel))
                q._op._kernel = False
        yield
    finally:
        _native.kernel = orig
        for op, k in saved:
            op._kernel = k


env_strategy = st.fixed_dictionaries({
    "block": st.sampled_from([None, 1, 3, 17, 1000, 1 << 20]),
    "threads": st.sampled_from([None, 1, 2, 3, 4]),
    "parallel_min": st.sampled_from([None, 1]),
})


def note_env(env, n=None):
    if n is not None:
        parallel = env["parallel_min"] == 1 and env["threads"] != 1 and n >= 2
        event(f"parallel_path_executed={parallel}")
    event(f"env.threads={env['threads']}")
    event(f"env.parallel_min={env['parallel_min']}")
    event(f"env.block={'default' if env['block'] is None else 'tiny' if env['block'] <= 17 else 'big'}")


# ---------------------------------------------------------------------------
# quantizer cache and strategies
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=2048)
def Q(d, bits, seed, rotation, normalize=True, scale=None):
    return Quantizer(d, bits, seed=seed, rotation=rotation, normalize=normalize,
                     scale=scale)


def with_mean(q, mean):
    """Shallow copy of ``q`` declaring ``mean`` (what the constructor would do)."""
    q2 = copy.copy(q)
    q2.mean = np.asarray(mean, dtype=np.float32)
    return q2


dim_strategy = st.one_of(st.sampled_from(EVEN_DIMS + ODD_DIMS), st.integers(1, 100))


even_dim_strategy = st.one_of(st.sampled_from(EVEN_DIMS), st.integers(1, 50).map(lambda k: 2 * k))


@st.composite
def configs(draw, rotations=("rht", "haar", "none"), bits=BITS, dims=dim_strategy):
    """(d, bits, seed, rotation); rotation is drawn first so that rht, which
    needs an even d, is not starved by odd dimensions."""
    rotation = draw(st.sampled_from(rotations))
    d = draw(even_dim_strategy if rotation == "rht" else dims)
    b = draw(st.sampled_from(bits))
    seed = draw(st.sampled_from(SEED_POOL))
    event(f"cfg.rotation={rotation}")
    event(f"cfg.bits={b}")
    event(f"cfg.d={'1' if d == 1 else '2' if d == 2 else 'odd' if d % 2 else 'pow2' if d & (d - 1) == 0 else 'even'}")
    return d, b, seed, rotation


CORPUS_KINDS = ("gauss", "heavy", "onehot", "const", "zero_rows", "dups", "walsh",
                "grid", "raw", "lowrank", "hug", "zeros")


def hug_rows(rng, q, n):
    """Rows whose *rotated unit direction* has a random subset of coordinates
    exactly at cell boundaries (the rest random, total norm 1), mapped back
    through an independent float64 rotation. After encode's own rotation those
    coordinates land within float32 rounding of a boundary, so the code depends
    on the last bit: the region where chunking, BLAS shape and kernel choice
    could matter."""
    d = q.d
    M = ref_rotation_matrix(q.rotation, d, q.seed)
    b = q.boundaries.astype(np.float64)
    out = np.zeros((n, d))
    for i in range(n):
        u = np.zeros(d)
        order = rng.permutation(d)
        used, free = 0.0, []
        for j in order:
            cand = b[rng.integers(len(b))]
            if used + cand * cand <= 0.85 and len(free) < d - 1 and rng.random() < 0.6:
                u[j] = cand
                used += cand * cand
            else:
                free.append(j)
        g = rng.standard_normal(len(free))
        u[free] = g / np.linalg.norm(g) * math.sqrt(1.0 - used)
        out[i] = (u @ M.T) * 2.0 ** rng.integers(-4, 5)
    return out


@st.composite
def corpora(draw, d, n_min=0, n_max=20, kinds=CORPUS_KINDS, min_norm=None, q=None):
    """Finite float32 (n, d) arrays; ``min_norm`` drops rows that are neither
    zero nor at least that long (by zeroing them)."""
    n = draw(st.integers(n_min, n_max))
    kind = draw(st.sampled_from(kinds))
    event(f"corpus.kind={kind}")
    event(f"corpus.n={'0' if n == 0 else '1' if n == 1 else '2-8' if n <= 8 else '9+'}")
    if kind == "raw":
        elem = st.one_of(
            st.floats(-1e3, 1e3, width=32, allow_nan=False, allow_infinity=False),
            st.sampled_from([0.0, -0.0, 1.0, -1.0, 0.5, 1e-3]))
        X = draw(npst.arrays(np.float32, (n, d), elements=elem))
    else:
        rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
        if kind == "gauss":
            X = rng.standard_normal((n, d))
        elif kind == "heavy":
            X = rng.standard_normal((n, d)) * rng.lognormal(0, 3, (n, 1))
        elif kind == "onehot":
            X = np.zeros((n, d))
            if n:
                X[np.arange(n), rng.integers(0, d, n)] = rng.choice([-1.0, 1.0], n) * rng.lognormal(0, 1, n)
        elif kind == "const":
            X = np.outer(rng.lognormal(0, 2, n) * rng.choice([-1.0, 1.0], n), np.ones(d))
        elif kind == "zero_rows":
            X = rng.standard_normal((n, d))
            X[rng.random(n) < 0.5] = 0.0
        elif kind == "dups":
            base = rng.standard_normal((n // 3 + 1, d))
            X = base[rng.integers(0, len(base), n)]
        elif kind == "walsh":
            m = 1
            while m < d:
                m *= 2
            X = hadamard(m)[rng.integers(0, m, n)][:, :d].astype(np.float64) if n else np.zeros((0, d))
        elif kind == "grid":
            X = rng.integers(-3, 4, (n, d)).astype(np.float64)
        elif kind == "zeros":
            X = np.zeros((n, d))
        elif kind == "hug":
            X = hug_rows(rng, q, n) if (q is not None and d > 1) else rng.standard_normal((n, d))
        elif kind == "lowrank":
            u = rng.standard_normal(d)
            X = rng.standard_normal((n, 1)) * u + 1e-2 * rng.standard_normal((n, d))
            X += 3 * u   # shared direction: anisotropic corpus
        else:  # pragma: no cover
            raise AssertionError(kind)
        X = X.astype(np.float32)
    if min_norm is not None and n:
        norms = np.sqrt((X.astype(np.float64) ** 2).sum(axis=1))
        X[(norms > 0) & (norms < min_norm)] = 0.0
    return X


def bit_equal(a, b):
    """Bitwise equality of two float arrays (NaN == NaN)."""
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def assert_same_container(a, b, ctx=""):
    assert np.array_equal(a.indices, b.indices), f"indices differ {ctx}"
    assert bit_equal(a.norms, b.norms), f"norms differ {ctx}"
    assert bit_equal(a.mean_dots, b.mean_dots), f"mean_dots differ {ctx}"


# ---------------------------------------------------------------------------
# independent references
# ---------------------------------------------------------------------------
def _phi(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@functools.lru_cache(maxsize=None)
def ref_lloyd_unit(bits, iters=300):
    """Lloyd-Max for N(0,1), same recurrence as the spec but built on
    scipy.special.ndtr and a hand-written pdf rather than scipy.stats.norm."""
    c = np.linspace(-3.0, 3.0, 2 ** bits)
    for _ in range(iters):
        b = np.concatenate([[-np.inf], (c[:-1] + c[1:]) / 2, [np.inf]])
        prob = np.diff(ndtr(b))
        num = _phi(b[:-1]) - _phi(b[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(prob > 1e-15, num / prob, c)
    return c


def gaussian_mse_unit(c):
    """Exact MSE of N(0,1) quantized to centroids ``c`` with midpoint cells."""
    c = np.asarray(c, dtype=np.float64)
    b = np.concatenate([[-np.inf], (c[:-1] + c[1:]) / 2, [np.inf]])
    lo, hi = b[:-1], b[1:]

    def xphi(x):
        fin = np.isfinite(x)
        return np.where(fin, np.where(fin, x, 0.0) * _phi(np.where(fin, x, 0.0)), 0.0)

    i0 = ndtr(hi) - ndtr(lo)
    i1 = _phi(lo) - _phi(hi)
    i2 = i0 - (xphi(hi) - xphi(lo))
    return float(np.sum(i2 - 2 * c * i1 + c * c * i0))


def _pow2_div(d):
    b = 1
    while d % (2 * b) == 0:
        b *= 2
    return b


@functools.lru_cache(maxsize=4096)
def ref_rotation_matrix(rotation, d, seed):
    """float64 M with ``rotate_rows(X) == X @ M``, built independently.

    rht: the recipe in rotation.rht_rotation's docstring, with explicit
         block-Hadamard / diagonal-sign / permutation matrices.
    haar: the unique QR factor with positive diagonal R (Mezzadri).
    """
    if rotation == "none":
        return np.eye(d)
    if rotation == "haar":
        A = np.random.default_rng(seed).standard_normal((d, d))
        Qm, Rm = np.linalg.qr(A)
        s = np.sign(np.diag(Rm))
        s[s == 0] = 1.0
        return (Qm * s).T
    B = _pow2_div(d)
    rounds = max(2, math.ceil(math.log(d) / math.log(B)))
    rng = np.random.default_rng(seed)
    H = np.kron(np.eye(d // B), hadamard(B).astype(np.float64) / math.sqrt(B))
    M = np.eye(d)
    for _ in range(rounds):
        perm = rng.permutation(d)
        sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=d).astype(np.float64)
        P = np.zeros((d, d))
        P[np.arange(d), perm] = 1.0
        M = H @ np.diag(sign) @ P @ M
    return M


def semantic_values(q, X, mean=None):
    """The rotated coordinates ``encode`` should quantize, in float64."""
    X64 = np.asarray(X, dtype=np.float64)
    M = ref_rotation_matrix(q.rotation, q.d, q.seed)
    if not q.normalize:
        return X64 @ M
    if mean is not None:
        X64 = X64 - np.asarray(mean, dtype=np.float32).astype(np.float64)
    norms = np.sqrt((X64 ** 2).sum(axis=1, keepdims=True))
    unit = np.divide(X64, norms, out=np.zeros_like(X64), where=norms > 0)
    return unit @ M


def check_semantics(q, v, codes, tol):
    """Codes are the nearest-centroid cells, except within ``tol`` of a boundary."""
    c64 = q.centroids.astype(np.float64)
    b64 = q.boundaries.astype(np.float64)
    ref = np.abs(v[..., None] - c64).argmin(axis=-1)
    margin = np.abs(v[..., None] - b64).min(axis=-1)
    far = margin > tol
    event(f"semantic.near_boundary={'yes' if (~far).any() else 'no'}")
    bad = far & (codes != ref)
    assert not bad.any(), (
        f"{int(bad.sum())} coords not in nearest-centroid cell; first at "
        f"{np.argwhere(bad)[0].tolist()} v={v[bad][0]!r} code={codes[bad][0]} ref={ref[bad][0]}")
    assert np.all(np.abs(codes.astype(np.int64) - ref) <= 1), "near-boundary code is not adjacent"


# ===========================================================================
# A. metamorphic encode
# ===========================================================================
@S(220)
@given(st.data())
def test_chunked_encode_equals_whole(data):
    """encode(X) == concat(encode(chunks)) for any chunking, block size, thread
    count; bitwise for rht and none (exact), centred or not."""
    d, bits, seed, rotation = data.draw(configs(rotations=("rht", "none")))
    q = Q(d, bits, seed, rotation)
    X = data.draw(corpora(d, q=q))
    n = len(X)
    centered = data.draw(st.booleans()) and n > 0
    event(f"centered={centered}")
    if centered:
        q = with_mean(q, corpus_mean(X))
    cuts = sorted(data.draw(st.lists(st.integers(0, n), max_size=4)))
    event(f"chunks={len(cuts) + 1}")
    env = data.draw(env_strategy)
    note_env(env, n)
    whole = q.encode(X)
    with encode_env(**env):
        parts = [q.encode(p) for p in np.split(X, cuts)]
        whole_env = q.encode(X)
    assert_same_container(whole, whole_env, "whole vs whole-under-env")
    got = np.concatenate([p.indices for p in parts])
    assert np.array_equal(whole.indices, got)
    assert bit_equal(whole.norms, np.concatenate([p.norms for p in parts]))
    if centered:
        assert bit_equal(whole.mean_dots, np.concatenate([p.mean_dots for p in parts]))


@S(150)
@given(st.data())
def test_haar_chunking_only_differs_near_boundaries(data):
    """Haar goes through BLAS, whose rounding can depend on the batch shape
    (documented: CLAUDE.md 'haar still depends on the BLAS kernel'). Any
    difference must sit on a coordinate within float32 rounding of a boundary."""
    d, bits, seed, rotation = data.draw(configs(rotations=("haar",)))
    q = Q(d, bits, seed, rotation)
    X = data.draw(corpora(d, n_min=2, min_norm=1e-6, q=q))
    whole = q.encode(X)
    rows = np.concatenate([q.encode(X[i:i + 1]).indices for i in range(len(X))])
    diff = rows != whole.indices
    event(f"haar.chunk_mismatch={'yes' if diff.any() else 'no'}")
    if diff.any():
        v = semantic_values(q, X)
        margin = np.abs(v[..., None] - q.boundaries.astype(np.float64)).min(axis=-1)
        assert margin[diff].max() < 1e-5, "haar chunk difference away from a boundary"


@S(200)
@given(st.data())
def test_row_permutation_commutes_with_encode(data):
    d, bits, seed, rotation = data.draw(configs(rotations=("rht", "none")))
    q = Q(d, bits, seed, rotation)
    X = data.draw(corpora(d, n_min=1, q=q))
    perm = np.array(data.draw(st.permutations(range(len(X)))), dtype=np.intp)
    env = data.draw(env_strategy)
    note_env(env)
    base = q.encode(X)
    with encode_env(**env):
        shuffled = q.encode(X[perm])
    assert np.array_equal(shuffled.indices, base.indices[perm])
    assert bit_equal(shuffled.norms, base.norms[perm])


@S(120)
@given(st.data())
def test_encode_is_deterministic_across_calls_and_fresh_instances(data):
    d, bits, seed, rotation = data.draw(configs())
    X = data.draw(corpora(d))
    q1 = Quantizer(d, bits, seed=seed, rotation=rotation)
    q2 = Quantizer(d, bits, seed=seed, rotation=rotation)
    a, b, c = q1.encode(X), q1.encode(X), q2.encode(X)
    assert_same_container(a, b, "repeat call")
    assert_same_container(a, c, "fresh instance")
    assert np.array_equal(q1.boundaries, q2.boundaries) and np.array_equal(q1.centroids, q2.centroids)
    if seed < 2**31:   # seed spelled as a numpy integer must not matter
        q3 = Quantizer(d, bits, seed=np.int64(seed), rotation=rotation)
        assert_same_container(a, q3.encode(X), "np.int64 seed")


@S(160)
@given(st.data())
def test_native_and_numpy_paths_give_identical_bits(data):
    """Compiled kernels vs the NumPy fallback: rotation and code assignment,
    normalized and scalar mode, centred mode, float32/float64."""
    if not have_native():
        pytest.skip("no compiled kernel")
    d, bits, seed, rotation = data.draw(configs())
    mode = data.draw(st.sampled_from(["normalized", "centered", "scalar"]))
    event(f"mode={mode}")
    X = data.draw(corpora(d))
    env = data.draw(env_strategy)
    note_env(env)
    if mode == "scalar":
        q = Q(d, bits, seed, rotation, False, data.draw(st.sampled_from([None, 0.25, 1.0, 40.0])))
        X = X.astype(data.draw(st.sampled_from([np.float32, np.float64]))) * (q.scale if q.scale else 1.0)
    else:
        q = Q(d, bits, seed, rotation)
        if mode == "centered" and len(X):
            q = with_mean(q, corpus_mean(X))
    with encode_env(**env):
        compiled = q.encode(X)
        with numpy_path(q):
            fallback = q.encode(X)
    assert_same_container(compiled, fallback, "compiled vs numpy")
    if rotation == "rht":
        op = q._op
        for dt in (np.float32, np.float64):
            Z = np.asarray(X, dtype=dt)
            with encode_env(**env):
                a0, a1 = op.rotate_rows(Z), op.unrotate_rows(Z)
            with numpy_path(q):
                b0, b1 = op._apply_numpy(Z, 1), op._apply_numpy(Z, 0)
            assert bit_equal(a0, b0) and bit_equal(a1, b1)


@S(200)
@given(st.data())
def test_negation_mirrors_codes(data):
    """code(-x) == (2^b - 1) - code(x), minus one where a rotated coordinate sits
    exactly on a boundary (v > b is a strict comparison)."""
    d, bits, seed, rotation = data.draw(configs())
    scalar = data.draw(st.booleans())
    event(f"scalar={scalar}")
    X = data.draw(corpora(d, q=Q(d, bits, seed, rotation)))
    m = (1 << bits) - 1
    if scalar:
        q = Q(d, bits, seed, rotation, False, 1.0)
        X64 = X.astype(np.float64)
        rotated = q._rotate_rows(X64)
        a, b = q.encode(X64).indices, q.encode(-X64).indices
    else:
        q = Q(d, bits, seed, rotation)
        norms = np.sqrt((X.astype(np.float64) ** 2).sum(axis=1)).astype(np.float32)
        unit = X / np.maximum(norms, 1e-8)[:, None]
        rotated = q._rotate_rows(unit)
        a, b = q.encode(X).indices, q.encode(-X).indices
    tie = np.isin(rotated, q.boundaries)
    event(f"ties={'yes' if tie.any() else 'no'}")
    expected = m - a.astype(np.int64) - tie
    assert np.array_equal(b.astype(np.int64), expected)


@S(200)
@given(st.data())
def test_power_of_two_scaling_leaves_direction_codes_alone(data):
    """encode(2^k x) has the same codes as encode(x) and norms scaled exactly:
    the codes are of the unit direction. Restricted to norms in the range where
    no float32 or floor effect applies; the sub-floor family is
    test_direction_codes_below_the_1e8_floor."""
    d, bits, seed, rotation = data.draw(configs(rotations=("rht", "haar", "none")))
    X = data.draw(corpora(d, n_min=1, min_norm=1e-3))
    k = data.draw(st.integers(-30, 30))
    event(f"k={'neg' if k < 0 else 'pos' if k else 'zero'}")
    nz = np.sqrt((X.astype(np.float64) ** 2).sum(axis=1))
    nz = nz[nz > 0]
    assume(len(nz) == 0 or (nz.min() * 2.0 ** k >= 1e-6 and nz.max() * 2.0 ** k < 1e30))
    q = Q(d, bits, seed, rotation)
    f = np.float32(2.0) ** np.float32(k)
    a, b = q.encode(X), q.encode(X * f)
    assert np.array_equal(a.indices, b.indices)
    assert bit_equal(a.norms * f, b.norms)


@pytest.mark.xfail(strict=True, reason="finding F2: np.maximum(lengths, 1e-8) is an absolute floor, "
                                       "codes of vectors shorter than 1e-8 are not scale invariant")
@S(120)
@given(st.data())
def test_direction_codes_below_the_1e8_floor(data):
    d, bits, seed, rotation = data.draw(configs(rotations=("rht", "none")))
    X = data.draw(corpora(d, n_min=1, kinds=("gauss", "heavy", "grid")))
    X = X[(X != 0).any(axis=1)]
    assume(len(X))
    q = Q(d, bits, seed, rotation)
    k = data.draw(st.integers(-60, -28))
    f = np.float32(2.0) ** np.float32(k)
    small = (X * f).astype(np.float32)
    norms = np.sqrt((small.astype(np.float64) ** 2).sum(axis=1))
    assume(norms.max() < 1e-8 and norms.min() > 1e-30)
    assert np.array_equal(q.encode(small).indices, q.encode(X).indices)


@S(150)
@given(st.data())
def test_norms_are_float32_rounded_l2_norms(data):
    """Independent oracle for the norms column: exact (fsum) L2 norm, within one
    float32 ulp."""
    d, bits, seed, rotation = data.draw(configs(rotations=("none", "haar")))
    X = data.draw(corpora(d, n_min=1))
    q = Q(d, bits, seed, rotation)
    got = q.encode(X).norms.astype(np.float64)
    exact = np.array([math.sqrt(math.fsum(float(v) ** 2 for v in row)) for row in X])
    tol = np.spacing(exact.astype(np.float32)).astype(np.float64)
    assert np.all(np.abs(got - exact) <= tol + 1e-45)


@S(160)
@given(st.data())
def test_encode_matches_nearest_centroid_oracle(data):
    """End-to-end semantic oracle: independent float64 rotation, independent
    unit normalization, nearest-centroid assignment, tolerant at boundaries.
    Covers uncentred, centred and scalar modes for all rotations."""
    d, bits, seed, rotation = data.draw(configs())
    mode = data.draw(st.sampled_from(["normalized", "centered", "scalar"]))
    event(f"mode={mode}")
    X = data.draw(corpora(d, n_min=1, min_norm=1e-6, q=Q(d, bits, seed, rotation)))
    mean = None
    if mode == "scalar":
        scale = data.draw(st.sampled_from([0.5, 1.0, 8.0]))
        q = Q(d, bits, seed, rotation, False, scale)
        X = X * np.float32(scale)
        codes = q.encode(X).indices
        tol = 1e-5 * scale * max(1.0, math.sqrt(d))
    else:
        q = Q(d, bits, seed, rotation)
        if mode == "centered":
            mean = corpus_mean(X)
            q = with_mean(q, mean)
            res = np.asarray(X, np.float64) - mean.astype(np.float64)
            rn = np.sqrt((res ** 2).sum(axis=1))
            assume(not ((rn > 0) & (rn < 1e-6)).any())
        codes = q.encode(X).indices
        tol = 1e-5
    check_semantics(q, semantic_values(q, X, mean), codes, tol)


# ===========================================================================
# B. codebooks
# ===========================================================================
sigma_strategy = st.one_of(
    st.none(),
    st.sampled_from([1.0, 0.5, 2.0, 1e-6, 1e6]),
    st.floats(min_value=1e-20, max_value=1e20, allow_nan=False, allow_infinity=False),
)


@S(100)
@given(d=dim_strategy, bits=st.sampled_from(BITS), sigma=sigma_strategy)
def test_codebook_invariants(d, bits, sigma):
    event(f"sigma={'default' if sigma is None else 'explicit'}")
    b, c = lloyd_max_codebook(d, bits, sigma=sigma)
    assert b.dtype == np.float32 and c.dtype == np.float32
    assert len(c) == 2 ** bits and len(b) == 2 ** bits - 1
    assert np.all(np.isfinite(c)) and np.all(np.isfinite(b))
    assert np.all(np.diff(c) > 0) and np.all(np.diff(b) > 0)
    assert np.array_equal(c, -c[::-1]) and np.array_equal(b, -b[::-1])
    c64 = c.astype(np.float64)
    assert np.array_equal(b, ((c64[:-1] + c64[1:]) / 2.0).astype(np.float32))
    assert b[(len(b) - 1) // 2] == 0.0
    inside_lo = np.concatenate([[-np.inf], b.astype(np.float64)])
    inside_hi = np.concatenate([b.astype(np.float64), [np.inf]])
    assert np.all(c64 > inside_lo) and np.all(c64 < inside_hi), "centroid outside its cell"
    b2, c2 = lloyd_max_codebook(d, bits, sigma=sigma)
    assert np.array_equal(b, b2) and np.array_equal(c, c2), "codebook not deterministic"
    sig = coordinate_sigma(d, sigma)
    assert sig == (1.0 / math.sqrt(d) if sigma is None else float(sigma))


@S(60)
@given(bits=st.sampled_from(BITS), k=st.integers(-80, 80), d=dim_strategy)
def test_codebook_scale_equivariance_for_powers_of_two(bits, k, d):
    """sigma = 2^k * 1.0 scales every table entry exactly (IEEE scaling by a
    power of two commutes with every operation in the recurrence)."""
    b1, c1 = lloyd_max_codebook(d, bits, sigma=1.0)
    bs, cs = lloyd_max_codebook(d, bits, sigma=2.0 ** k)
    f = np.float32(2.0) ** np.float32(k) if abs(k) < 120 else None
    assert np.array_equal(cs, c1 * f) and np.array_equal(bs, b1 * f)


@S(60)
@given(bits=st.sampled_from(BITS), d=dim_strategy)
def test_codebook_matches_independent_lloyd_reference(bits, d):
    sig = 1.0 / math.sqrt(d)
    _, c = lloyd_max_codebook(d, bits)
    ref = ref_lloyd_unit(bits, 300) * sig
    assert np.max(np.abs(c.astype(np.float64) - ref)) < 2e-6 * sig


def test_reference_mse_matches_max_1960():
    """Anchor the independent MSE oracle to Max's published distortions."""
    published = {1: 0.3634, 2: 0.11748, 3: 0.03454, 4: 0.009497}
    for bits, want in published.items():
        assert abs(gaussian_mse_unit(ref_lloyd_unit(bits, 3000 if bits > 2 else 300)) - want) < 5e-5


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_shipped_codebook_is_lloyd_optimal_up_to_4_bits(bits):
    _, c = lloyd_max_codebook(1, bits, sigma=1.0)
    assert gaussian_mse_unit(c) <= gaussian_mse_unit(ref_lloyd_unit(bits, 20000)) * 1.001


@pytest.mark.xfail(strict=True, reason="finding F6: n_iter=300 Jacobi Lloyd is unconverged at 8 bits "
                                       "(MSE ~ +20% over the fixed point)")
def test_shipped_codebook_is_lloyd_optimal_at_8_bits():
    _, c = lloyd_max_codebook(1, 8, sigma=1.0)
    assert gaussian_mse_unit(c) <= 4.1185082867e-05 * 1.01   # 200k-iteration fixed point


@S(60)
@given(bits=st.sampled_from([2, 3, 4, 8]), d=dim_strategy, sigma=st.one_of(st.none(), st.sampled_from([0.5, 3.0, 1e-4])))
def test_nested_codebook_invariants(bits, d, sigma):
    nb = nested_codebooks(d, bits, sigma=sigma)
    _, top = lloyd_max_codebook(d, bits, sigma=sigma)
    assert sorted(nb) == list(range(1, bits + 1))
    assert np.array_equal(nb[bits], top)
    sig = coordinate_sigma(d, sigma)
    fine_b = ((top[:-1].astype(np.float64) + top[1:]) / 2)
    probs = np.diff(ndtr(np.concatenate([[-np.inf], fine_b / sig, [np.inf]])))
    for k in range(1, bits + 1):
        c = nb[k]
        assert len(c) == 2 ** k and c.dtype == np.float32
        assert np.all(np.isfinite(c)) and np.all(np.diff(c) > 0)
        assert np.array_equal(c, -c[::-1]), f"nested level {k} not antisymmetric"
        g = 2 ** (bits - k)
        for j in range(2 ** k):
            grp = top[j * g:(j + 1) * g].astype(np.float64)
            w = probs[j * g:(j + 1) * g]
            assert grp.min() - 1e-7 * sig <= c[j] <= grp.max() + 1e-7 * sig
            if w.sum() > 1e-15:
                assert abs(c[j] - np.average(grp, weights=w)) < 1e-6 * sig


@S(120)
@given(st.data())
def test_matryoshka_codes_nest(data):
    """Top k bits of an n-bit code index the k-bit table; the 1-bit level is
    bit-for-bit the standalone 1-bit code (the sign)."""
    d, bits, seed, rotation = data.draw(configs(bits=(2, 3, 4, 8)))
    X = data.draw(corpora(d, n_min=1))
    scalar = data.draw(st.booleans())
    event(f"scalar={scalar}")
    if scalar:
        q = Q(d, bits, seed, rotation, False, 1.0)
        q1 = Q(d, 1, seed, rotation, False, 1.0)
        X = X.astype(np.float64)
    else:
        q, q1 = Q(d, bits, seed, rotation), Q(d, 1, seed, rotation)
    cv = q.encode(X)
    assert np.array_equal(cv.indices >> (bits - 1), q1.encode(X).indices)
    for k in range(1, bits + 1):
        shifted = cv.indices >> (bits - k)
        assert shifted.max(initial=0) < 2 ** k
        if scalar and rotation == "none":
            got = q.decode(cv, precision=k)
            assert np.array_equal(got, q._nested[k][shifted])
            lo = np.concatenate([[-np.inf], q.boundaries.astype(np.float64)])[(shifted.astype(np.int64) << (bits - k))]
            hi = np.concatenate([q.boundaries.astype(np.float64), [np.inf]])[((shifted.astype(np.int64) + 1) << (bits - k)) - 1]
            assert np.all(got.astype(np.float64) > lo) and np.all(got.astype(np.float64) < hi)


# ---- assign_codes vs np.searchsorted -------------------------------------
_SPECIALS = [0.0, -0.0, 1e-45, -1e-45, 1.1754944e-38, 3.4028235e38, -3.4028235e38,
             np.inf, -np.inf, np.nan]


@st.composite
def boundary_arrays(draw, bits):
    """Sorted float32 boundary tables: real codebooks, or arbitrary sorted
    arrays (duplicates and +/-inf allowed)."""
    if draw(st.booleans()):
        d = draw(st.sampled_from([1, 4, 64, 768]))
        event("boundaries=codebook")
        return Q(d, bits, 0, "none").boundaries
    event("boundaries=arbitrary")
    m = 2 ** bits - 1
    vals = draw(st.lists(st.one_of(
        st.floats(-1e3, 1e3, width=32, allow_nan=False, allow_infinity=False),
        st.sampled_from([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, 3.4028235e38])),
        min_size=m, max_size=m))
    return np.sort(np.array(vals, dtype=np.float32))


@st.composite
def haystacks(draw, b, dtype):
    """Values around the boundaries: on them, +/-1 ulp, signed zeros, extremes."""
    shape = draw(st.sampled_from([(0,), (1,), (7,), (3, 5), (2, 3, 4), (4, 64)]))
    n = int(np.prod(shape))
    pool_b = b.astype(np.float64)
    pick = draw(st.lists(st.integers(0, 6), min_size=n, max_size=n))
    out = np.empty(n, dtype=np.float64)
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    for i, p in enumerate(pick):
        if p == 0:
            out[i] = pool_b[rng.integers(len(pool_b))] if len(pool_b) else 0.0
        elif p == 1:
            out[i] = np.nextafter(pool_b[rng.integers(len(pool_b))], np.inf)
        elif p == 2:
            out[i] = np.nextafter(pool_b[rng.integers(len(pool_b))], -np.inf)
        elif p == 3:
            out[i] = _SPECIALS[rng.integers(len(_SPECIALS))]
        elif p == 4:
            out[i] = rng.standard_normal() * 10.0 ** rng.integers(-3, 3)
        elif p == 5:
            out[i] = np.float32(pool_b[rng.integers(len(pool_b))])
        else:
            out[i] = rng.choice([-1.7e308, 1.7e308, 5e-324])
    with np.errstate(over="ignore"):
        arr = out.astype(dtype).reshape(shape)
    return arr


@S(300)
@given(st.data())
def test_assign_codes_equals_searchsorted(data):
    bits = data.draw(st.sampled_from(BITS))
    b = data.draw(boundary_arrays(bits))
    dtype = data.draw(st.sampled_from([np.float32, np.float64, np.float16]))
    event(f"hay.dtype={np.dtype(dtype).name}")
    X = data.draw(haystacks(b, dtype))
    layout = data.draw(st.sampled_from(["c", "reversed", "fortran", "readonly"]))
    event(f"layout={layout}")
    if layout == "reversed" and X.ndim:
        X = X[::-1]
    elif layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "readonly":
        X = X.copy()
        X.setflags(write=False)
    env = data.draw(env_strategy)
    note_env(env)
    expect = np.searchsorted(b, X).astype(np.uint8)
    event(f"hit_boundary={'yes' if np.isin(X, b).any() else 'no'}")
    event(f"has_nan={'yes' if np.isnan(X.astype(np.float64)).any() else 'no'}")
    with encode_env(**env):
        got = assign_codes(X, b, bits)
        with numpy_path():
            fallback = assign_codes(X, b, bits)
    assert got.dtype == np.uint8 and got.shape == X.shape
    assert np.array_equal(got, expect), "compiled assign_codes != searchsorted"
    assert np.array_equal(fallback, expect)


@pytest.mark.xfail(strict=True, reason="finding F4: assign_codes casts anything that is not native "
                                       "float32/float64 to float32 and casts float64 boundaries to float32")
@S(120)
@given(st.data())
def test_assign_codes_wider_domain_matches_searchsorted(data):
    bits = data.draw(st.sampled_from([1, 2, 4]))
    b = data.draw(boundary_arrays(bits)).astype(np.float64)
    b = np.sort(b + data.draw(st.sampled_from([0.0, 1e-9, -1e-9])))
    kind = data.draw(st.sampled_from(["f64_boundaries", ">f8", "int64"]))
    event(f"wide={kind}")
    if kind == "int64":
        b = np.sort(np.array([2.0 ** 24] * (2 ** bits - 1), dtype=np.float32))
        X = np.array([2 ** 24 + 1, 2 ** 24, 2 ** 24 + 2, -2 ** 40, 2 ** 40], dtype=np.int64)
    elif kind == ">f8":
        b = b.astype(np.float32)
        X = np.nextafter(b.astype(np.float64), np.inf).astype(">f8")
    else:
        X = np.nextafter(b, -np.inf).astype(np.float32)
    assume(len(b) == 2 ** bits - 1)
    assert np.array_equal(assign_codes(X, b, bits), np.searchsorted(b, X).astype(np.uint8))


# ===========================================================================
# C. rotations
# ===========================================================================
@S(150)
@given(d=st.integers(1, 40), seed=st.integers(0, 2**32 - 1), kind=st.sampled_from(["haar", "rht", "none"]))
def test_rotations_are_orthogonal_and_deterministic(d, seed, kind):
    if kind == "rht" and d % 2:
        with pytest.raises(ValueError, match="odd"):
            rht_rotation(d, seed)
        with pytest.raises(ValueError, match="odd"):
            RHTOperator(d, seed)
        event("rht.odd_d_rejected")
        return
    build = {"haar": haar_rotation, "rht": rht_rotation,
             "none": lambda d, s: np.eye(d, dtype=np.float32)}[kind]
    R = build(d, seed)
    assert R.dtype == np.float32 and R.shape == (d, d)
    R64 = R.astype(np.float64)
    tol = 3e-7 * (2 + math.log2(d + 1))
    assert np.abs(R64 @ R64.T - np.eye(d)).max() < tol
    assert np.abs(R64.T @ R64 - np.eye(d)).max() < tol
    assert np.array_equal(R, build(d, seed))


@S(200)
@given(st.data())
def test_rht_operator_matches_dense_and_independent_reference(data):
    d = data.draw(st.sampled_from([x for x in EVEN_DIMS if x <= 130] + [2 * k for k in range(1, 40)]))
    seed = data.draw(st.one_of(st.sampled_from(SEED_POOL), st.integers(0, 2**32 - 1)))
    dt = data.draw(st.sampled_from([np.float32, np.float64]))
    n = data.draw(st.integers(0, 12))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    X = (rng.standard_normal((n, d)) * rng.lognormal(0, 2, (n, 1))).astype(dt)
    op = RHTOperator(d, seed)
    R = rht_rotation(d, seed).astype(np.float64)
    M = ref_rotation_matrix("rht", d, seed)
    X64 = X.astype(np.float64)
    scale = max(1.0, float(np.abs(X64).max(initial=0.0)))
    tol = 5e-6 * scale
    rr, ur = op.rotate_rows(X), op.unrotate_rows(X)
    assert rr.dtype == dt and ur.dtype == dt and rr.shape == X.shape
    assert np.abs(rr - X64 @ R.T).max(initial=0.0) < tol
    assert np.abs(ur - X64 @ R).max(initial=0.0) < tol
    assert np.abs(R.T - M).max() < 5e-6, "dense rht_rotation vs independent construction"
    assert np.abs(rr - X64 @ M).max(initial=0.0) < tol, "operator vs independent construction"
    assert np.abs(op.materialize(np.float32) - R).max() < 5e-6
    if n:
        assert np.array_equal(op.rotate_query(X[0]), rr[0]), "rotate_query != rotate_rows[0]"
        back = op.unrotate_rows(rr)
        assert np.abs(back - X64).max() < 2e-5 * scale, "unrotate(rotate(x)) != x"
        nr = np.linalg.norm(rr.astype(np.float64), axis=1)
        nx = np.linalg.norm(X64, axis=1)
        assert np.all(np.abs(nr - nx) <= 1e-5 * (1 + nx)), "rotation changed a norm"


@S(80)
@given(d=st.integers(1, 60), seed=st.integers(0, 2**32 - 1))
def test_haar_matches_unique_qr_factor(d, seed):
    ref = ref_rotation_matrix("haar", d, seed).T
    assert np.abs(haar_rotation(d, seed).astype(np.float64) - ref).max() < 5e-5


def test_rht_rounds_floor_and_plan_shape_for_every_even_d():
    for d in range(2, 8194, 2):
        B = _largest_pow2_divisor(d)
        r = rht_rounds(d, B)
        assert r >= 2, d
        assert B ** r >= d, d
    for d, seed in [(2, 0), (6, 1), (10, 2), (96, 3), (130, 4), (1024, 5)]:
        B, rounds, perms, signs = rht_plan(d, seed)
        assert perms.shape == (rounds, d) and perms.dtype == np.int32
        assert all(sorted(p) == list(range(d)) for p in perms)
        assert all(set(np.unique(s)) <= {-1.0, 1.0} for s in signs)


@S(100)
@given(st.integers(2, 10**7).map(lambda x: 2 * x))
def test_rht_rounds_for_large_d_is_floating_point_safe(d):
    B = _largest_pow2_divisor(d)
    r = rht_rounds(d, B)
    assert r >= 2 and B ** r >= d and B ** (r - 1) < d or r == 2


@S(60)
@given(d=st.integers(1, 40), bits=st.sampled_from(BITS))
def test_default_rotation_selection_and_odd_rht_error(d, bits):
    default = Quantizer(d, bits)
    assert default.rotation == ("rht" if d % 2 == 0 else "haar")
    if d % 2:
        with pytest.raises(ValueError, match="odd"):
            Quantizer(d, bits, rotation="rht")


@pytest.mark.xfail(strict=True, reason="finding F1: at d=2 (and d=4) every seed gives the same decodes; "
                                       "rht_rounds=2 cannot mix 2 coordinates")
@S(40)
@given(seeds=st.lists(st.integers(0, 2**32 - 1), min_size=2, max_size=6, unique=True),
       d=st.sampled_from([2, 4]))
def test_rht_decode_depends_on_seed_at_tiny_d(seeds, d):
    X = np.random.default_rng(0).standard_normal((200, d)).astype(np.float32)
    decodes = []
    for s in seeds:
        q = Quantizer(d, 2, seed=s, rotation="rht")
        decodes.append(q.decode(q.encode(X)).tobytes())
    assert len(set(decodes)) > 1


@pytest.mark.parametrize("d", [16, 18, 24, 30, 32, 48, 64, 100, 128])
def test_rht_unmixed_basis_vectors_are_rare_from_d16(d):
    """Mixing is only probabilistic (see the xfail below), but from d=16 on at most
    ~0.5% of seeds leave a basis vector unmixed (measured over 1500 seeds: d=16
    7, d=20 1, d=28 1, d=34 1, d=50 2, the rest 0). Pin a 2% ceiling over 400
    fixed seeds so a construction change that makes it common is noticed."""
    I = np.eye(d, dtype=np.float32)
    unmixed = sum(np.abs(RHTOperator(d, s).unrotate_rows(I)).max() > 0.999 for s in range(400))
    assert unmixed <= 8, f"{unmixed}/400 seeds leave a basis vector unmixed at d={d}"


@pytest.mark.xfail(strict=True, reason="finding F1b: for d in {8,10,12,14,20,...} a sizeable fraction of seeds "
                                       "map some basis vector to a single coordinate (rotation does not mix)")
@S(150)
@given(d=st.sampled_from([4, 8, 10, 12, 14, 20]), seed=st.integers(0, 2**32 - 1))
def test_rht_leaves_no_basis_vector_unmixed_small_d(d, seed):
    assert np.abs(rht_rotation(d, seed)).max() < 0.999


# ===========================================================================
# D. scalar mode
# ===========================================================================
SCALES = (1.0, 0.5, 3.0, 1e-3, 1e5, 2.0 ** -20)


@st.composite
def scalar_values(draw, q, n_max=24):
    """float64 rows biased onto the cell boundaries of a scalar quantizer."""
    d = q.d
    n = draw(st.integers(0, n_max))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    b = q.boundaries.astype(np.float64)
    pool = np.concatenate([
        b, np.nextafter(b, np.inf), np.nextafter(b, -np.inf),
        b * (1 + 2.0 ** -23), b * (1 - 2.0 ** -23),
        [0.0, -0.0, 5e-324, -5e-324, 1e-300, 1.7976931348623157e308, -1.7976931348623157e308,
         np.inf, -np.inf, np.nan, 4 * q.scale, -4 * q.scale, 40 * q.scale],
        rng.standard_normal(16) * q.scale])
    X = rng.choice(pool, size=(n, d))
    event(f"scalar.has_nan={'yes' if np.isnan(X).any() else 'no'}")
    event(f"scalar.has_inf={'yes' if np.isinf(X).any() else 'no'}")
    return X


@S(220)
@given(st.data())
def test_scalar_none_codes_are_searchsorted_and_nearest_centroid(data):
    d, bits, seed, _ = data.draw(configs())
    scale = data.draw(st.sampled_from(SCALES))
    q = Q(d, bits, seed, "none", False, scale)
    X = data.draw(scalar_values(q))
    codes = q.encode(X).indices
    assert np.array_equal(codes, np.searchsorted(q.boundaries, X).astype(np.uint8))
    # independent boundaries: scale * (independent Lloyd-Max for sigma=1)
    ref_c = ref_lloyd_unit(bits, 300) * scale
    ref_b = (ref_c[:-1] + ref_c[1:]) / 2
    margin = np.abs(X[..., None] - ref_b).min(axis=-1)
    fin = np.isfinite(X)
    far = fin & (margin > 1e-5 * scale)
    assert np.array_equal(codes[far], np.searchsorted(ref_b, X[far]).astype(np.uint8))
    assert np.all(codes[np.isnan(X)] == (1 << bits) - 1)
    assert np.all(codes[X == np.inf] == (1 << bits) - 1)
    assert np.all(codes[X == -np.inf] == 0)
    f32 = X.astype(np.float32)   # exactly representable subset only
    exact = (f32.astype(np.float64) == X) | np.isnan(X)
    if exact.any():
        got32 = q.encode(np.where(exact, f32, 0.0).astype(np.float32)).indices
        assert np.array_equal(got32[exact], codes[exact]), "float32 input differs from float64"


@S(150)
@given(st.data())
def test_scalar_equal_inputs_equal_codes_and_monotone(data):
    d, bits, seed, rotation = data.draw(configs())
    scale = data.draw(st.sampled_from(SCALES))
    q = Q(d, bits, seed, rotation, False, scale)
    X = data.draw(scalar_values(q))
    assume(len(X) > 0)
    dup = np.concatenate([X, X, X.copy()])                       # duplicates
    flip0 = np.where(X == 0.0, -X, X)                            # +0.0 <-> -0.0
    assert np.array_equal(q.encode(X).indices, q.encode(flip0).indices), "-0.0 and 0.0 differ"
    c = q.encode(dup).indices
    n = len(X)
    assert np.array_equal(c[:n], c[n:2 * n]) and np.array_equal(c[:n], c[2 * n:])
    if rotation == "none":
        fin = np.isfinite(X)
        lo = np.where(fin, X, 0.0)
        hi = lo + np.abs(lo) * 0.25 + q.scale * 0.1
        assert np.all(q.encode(hi).indices >= q.encode(lo).indices), "codes not monotone in x"
        assert np.array_equal(q.encode(lo).indices >> (bits - 1), (lo > 0).astype(np.uint8)), "MSB is not sign"


@S(150)
@given(st.data())
def test_scalar_decode_returns_the_centroids(data):
    d, bits, seed, rotation = data.draw(configs())
    scale = data.draw(st.sampled_from(SCALES))
    q = Q(d, bits, seed, rotation, False, scale)
    X = data.draw(scalar_values(q))
    cv = q.encode(X)
    assert cv.norms is None and cv.scale == q.scale and cv.seed == seed
    for p in sorted({1, bits, max(1, bits // 2)}):
        got = q.decode(cv, precision=p)
        table = q.centroids if p == bits else q._nested[p]
        centroids = table[cv.indices >> (bits - p)]
        if rotation == "none":
            assert got.dtype == np.float32 and np.array_equal(got, centroids)
        else:
            M = ref_rotation_matrix(rotation, d, seed)
            ref = centroids.astype(np.float64) @ M.T      # unrotate: X_rot @ R
            assert np.abs(got - ref).max(initial=0.0) <= 3e-5 * float(np.abs(table).max())


@S(120)
@given(st.data())
def test_scalar_power_of_two_covariance(data):
    """codes(2^k x; scale=2^k s) == codes(x; scale=s), bitwise, for every rotation."""
    d, bits, seed, rotation = data.draw(configs(bits=(1, 2, 4)))
    s = data.draw(st.sampled_from([1.0, 0.5, 3.0, 0.1]))
    k = data.draw(st.integers(-20, 20))
    event(f"k={'neg' if k < 0 else 'pos' if k else 'zero'}")
    q1 = Q(d, bits, seed, rotation, False, s)
    q2 = Q(d, bits, seed, rotation, False, s * 2.0 ** k)
    X = data.draw(scalar_values(q1))
    X = np.where(np.isfinite(X) & (np.abs(X) < 1e30) & (np.abs(X) > 1e-30), X, 0.0)
    assert np.array_equal(q1.encode(X).indices, q2.encode(X * 2.0 ** k).indices)


@S(100)
@given(st.data())
def test_modes_cannot_be_mixed(data):
    d, bits, seed, _ = data.draw(configs(bits=(1, 2, 4)))
    qn = Q(d, bits, seed, "none")
    qs = Q(d, bits, seed, "none", False, 1.0)
    X = data.draw(corpora(d, n_min=1, min_norm=1e-3))
    cn, cs = qn.encode(X), qs.encode(X.astype(np.float64))
    assert cn.norms is not None and cs.norms is None
    query = X[0]
    for q, cv in ((qn, cs), (qs, cn)):
        for call in (lambda: q.decode(cv), lambda: q.search(cv, query, k=1),
                     lambda: q.search_adc(cv, query, k=1), lambda: q.search_batch(cv, query[None], k=1),
                     lambda: q.search_twostage(cv, query, k=1, candidates=2),
                     lambda: q.decode(cv, precision=1)):
            with pytest.raises(ValueError, match="mode mismatch"):
                call()
    other = Q(d, bits, seed, "none", False, 2.0)
    with pytest.raises(ValueError, match="scale mismatch"):
        other.decode(cs)
    if d != 1:
        with pytest.raises(ValueError, match="dimension mismatch"):
            Q(d + 1, bits, seed, "none", False, 1.0).decode(cs)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_scale_validation_and_normalize_guard(bad):
    with pytest.raises(ValueError):
        Quantizer(8, 2, normalize=False, scale=bad)
    with pytest.raises(ValueError):
        Quantizer(8, 2, normalize=True, scale=bad)
    with pytest.raises(ValueError):
        Quantizer(8, 2, normalize=True, scale=1.0)
    with pytest.raises(ValueError):
        Quantizer(8, 2, normalize=False, rotation="none", scale=1.0, mean=np.zeros(8, np.float32))


@pytest.mark.xfail(strict=True, reason="finding F3: scale outside ~[1e-43, 7e37] is accepted but the float32 "
                                       "tables collapse or overflow")
@S(80)
@given(scale=st.one_of(st.floats(1e-300, 1e-44), st.floats(1e38, 1e300)), bits=st.sampled_from(BITS))
def test_any_accepted_scale_yields_a_usable_codebook(scale, bits):
    try:
        q = Quantizer(8, bits, normalize=False, rotation="none", scale=scale)
    except ValueError:
        return                       # rejecting is fine
    b, c = q.boundaries, q.centroids
    assert np.all(np.isfinite(b)) and np.all(np.diff(b) > 0) and np.all(np.diff(c) > 0)


@pytest.mark.xfail(strict=True, reason="finding F7: overflow inside the rotation voids the scalar-mode "
                                       "saturation claim for |x| near float64 max")
@S(60)
@given(rotation=st.sampled_from(["rht", "haar"]), seed=st.sampled_from(SEED_POOL),
       mag=st.sampled_from([1e307, 1.7e308]))
def test_scalar_mode_saturates_huge_finite_values_under_rotation(rotation, seed, mag):
    d = 64
    q = Q(d, 2, seed, rotation, False, 1.0)
    X = np.random.default_rng(seed).standard_normal((8, d)) * mag
    X = np.clip(X, -1.7e308, 1.7e308)
    got = q.encode(X).indices
    ref = q.encode(X * 2.0 ** -200).indices       # same direction, no overflow
    assert np.array_equal(got, ref)


# ===========================================================================
# E. input edge cases
# ===========================================================================
@S(100)
@given(st.data())
def test_empty_and_singleton_batches(data):
    d, bits, seed, rotation = data.draw(configs())
    scalar = data.draw(st.booleans())
    q = Q(d, bits, seed, rotation, False, 1.0) if scalar else Q(d, bits, seed, rotation)
    empty = q.encode(np.zeros((0, d), np.float32))
    assert empty.n == 0 and empty.indices.shape == (0, d) and empty.indices.dtype == np.uint8
    assert (empty.norms is None) == scalar and (scalar or empty.norms.shape == (0,))
    assert q.decode(empty).shape == (0, d)
    row = data.draw(corpora(d, n_min=1, n_max=1))
    one = q.encode(row[0]) if not scalar else q.encode(row[0].astype(np.float64))
    batch = q.encode(row if not scalar else row.astype(np.float64))
    assert one.n == 1 and np.array_equal(one.indices, batch.indices)
    with pytest.raises(ValueError):
        q.encode(np.zeros((2, d + 1), np.float32))
    with pytest.raises(ValueError):
        q.encode(np.zeros((0, d + 1), np.float32))


@S(150)
@given(st.data())
def test_input_dtype_and_layout_do_not_change_codes_and_input_is_untouched(data):
    d, bits, seed, rotation = data.draw(configs(rotations=("rht", "none")))
    scalar = data.draw(st.booleans())
    q = Q(d, bits, seed, rotation, False, 1.0) if scalar else Q(d, bits, seed, rotation)
    X = data.draw(corpora(d, n_min=1, kinds=("gauss", "grid", "onehot", "zero_rows", "dups")))
    cast = np.float64 if scalar else np.float32
    kind = data.draw(st.sampled_from(["float64", "float16", "int8", "int32", "int64", "uint8", "bool", "float32"]))
    event(f"dtype={kind}")
    if kind == "uint8":
        Xs = np.clip(np.rint(np.abs(X)), 0, 200).astype(np.uint8)
    elif kind in ("int8", "int32", "int64"):
        Xs = np.clip(np.rint(X), -100, 100).astype(kind)
    elif kind == "bool":
        Xs = X > 0
    else:
        Xs = X.astype(kind)
    layout = data.draw(st.sampled_from(["c", "fortran", "strided", "reversed", "readonly", "transposed"]))
    event(f"layout={layout}")
    if layout == "fortran":
        V = np.asfortranarray(Xs)
    elif layout == "strided":
        big = np.zeros((Xs.shape[0] * 2, Xs.shape[1] * 2), Xs.dtype)
        big[::2, ::2] = Xs
        V = big[::2, ::2]
    elif layout == "reversed":
        V = np.ascontiguousarray(Xs[::-1, ::-1])[::-1, ::-1]
    elif layout == "transposed":
        V = np.ascontiguousarray(Xs.T).T
    elif layout == "readonly":
        V = Xs.copy()
        V.setflags(write=False)
    else:
        V = Xs
    assert np.array_equal(V, Xs)
    before = np.ascontiguousarray(V).view(np.uint8).copy()
    ref = q.encode(np.ascontiguousarray(Xs).astype(cast))
    got = q.encode(V)
    assert_same_container(ref, got, f"{kind}/{layout}")
    assert np.array_equal(np.ascontiguousarray(V).view(np.uint8), before), "encode mutated its input"


@S(120)
@given(st.data())
def test_non_finite_input_is_total_and_row_local(data):
    """Contract probe: NaN/inf never raise and never leak across rows; in scalar
    mode with rotation none NaN/+inf land in the top cell and -inf in cell 0
    (documented in _encode_scalar). Normalized mode documents nothing: this only
    pins totality, locality and the non-finite norm."""
    d, bits, seed, rotation = data.draw(configs(rotations=("rht", "none", "haar")))
    scalar = data.draw(st.booleans())
    X = data.draw(corpora(d, n_min=2, min_norm=1e-3)).astype(np.float64)
    n = len(X)
    bad_rows = data.draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=3, unique=True))
    X2 = X.copy()
    for r in bad_rows:
        X2[r, data.draw(st.integers(0, d - 1))] = data.draw(st.sampled_from([np.nan, np.inf, -np.inf]))
    q = Q(d, bits, seed, rotation, False, 1.0) if scalar else Q(d, bits, seed, rotation)
    event(f"nonfinite.scalar={scalar}")
    m = (1 << bits) - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = q.encode(X2)
        clean = [i for i in range(n) if i not in bad_rows]
        ref = q.encode(X[clean] if scalar else X[clean].astype(np.float32))
        refm = q.encode(X2[clean] if scalar else X2[clean].astype(np.float32))
    assert got.indices.max(initial=0) <= m
    assert np.array_equal(got.indices[clean], ref.indices) or rotation == "haar"
    assert np.array_equal(refm.indices, ref.indices)
    if not scalar:
        assert not np.isfinite(got.norms[bad_rows]).any()
    if scalar and rotation == "none":
        for r in bad_rows:
            c = int(np.flatnonzero(~np.isfinite(X2[r]))[0])
            v = X2[r, c]
            assert got.indices[r, c] == (0 if v == -np.inf else m)


@S(100)
@given(st.data())
def test_corpus_mean_and_anisotropy_properties(data):
    d = data.draw(st.integers(1, 12))
    n = data.draw(st.integers(1, 30))
    kind = data.draw(st.sampled_from(["normal", "zeros", "same_row", "lowrank", "ints"]))
    event(f"stats.kind={kind}")
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    if kind == "zeros":
        X = np.zeros((n, d))
    elif kind == "same_row":
        X = np.tile(rng.standard_normal(d), (n, 1))
    elif kind == "lowrank":
        X = rng.standard_normal((n, 1)) * 0.01 + rng.standard_normal(d)
    elif kind == "ints":
        X = rng.integers(-5, 6, (n, d)).astype(np.float64)
    else:
        X = rng.standard_normal((n, d)) * rng.lognormal(0, 2, (n, 1))
    X32 = X.astype(np.float32)
    m = corpus_mean(X32)
    assert m.dtype == np.float32 and m.shape == (d,)
    exact = np.array([math.fsum(col) / n for col in X32.astype(np.float64).T])
    assert np.all(np.abs(m.astype(np.float64) - exact) <= np.spacing(np.abs(exact).astype(np.float32)) + 1e-45)
    perm = rng.permutation(n)
    ulp = lambda a: np.spacing(np.abs(a)) + 1e-45
    assert np.all(np.abs(corpus_mean(X32[perm]).astype(np.float64) - m) <= ulp(m).astype(np.float64))
    if n == 1:
        assert np.array_equal(m, X32[0])
    r = anisotropy(X32)
    allzero = not X32.any()
    if n < 2 or allzero:
        assert r is None
    else:
        assert 0.0 <= r <= 1.0 + 1e-9
        assert abs(anisotropy(X32.astype(np.float64)) - r) < 1e-12
        assert abs(anisotropy(X32[perm]) - r) < 1e-9
        assert abs(anisotropy(-X32) - r) < 1e-12
        assert anisotropy(X32 * np.float32(4.0)) == pytest.approx(r, abs=1e-12)
        if kind == "same_row":
            assert abs(r - 1.0) < 1e-9
    q = Q(d, 2, 3, "haar")
    qm = with_mean(q, m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = qm.encode(X32)
    assert np.all(np.isfinite(cv.norms)) and np.all(np.isfinite(cv.mean_dots))
    assert np.all(np.isfinite(qm.decode(cv)))
    if kind == "same_row":
        assert np.all(cv.norms == 0) or np.all(cv.norms < 1e-5 * (1 + np.abs(X32).max()))


@S(80)
@given(st.data())
def test_anisotropy_check_agrees_with_anisotropy_and_never_changes_codes(data):
    d = data.draw(st.integers(2, 12))
    bits = data.draw(st.sampled_from([1, 2, 3, 4, 8]))
    rotation = data.draw(st.sampled_from(["haar", "none"] + (["rht"] if d % 2 == 0 else [])))
    X = data.draw(corpora(d, n_min=100, n_max=130, kinds=("gauss", "lowrank", "heavy", "const", "dups", "zero_rows", "zeros"),
                          min_norm=1e-3))
    q = Q(d, bits, 5, rotation)
    seen = []
    orig = core._warn_if_anisotropic
    core._warn_if_anisotropic = lambda ratio, n, b: (seen.append((ratio, n, b)), orig(ratio, n, b))[1]
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = q.encode(X)
        old = core.ANISOTROPY_MIN_ROWS
        core.ANISOTROPY_MIN_ROWS = 10 ** 9
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = q.encode(X)
        finally:
            core.ANISOTROPY_MIN_ROWS = old
    finally:
        core._warn_if_anisotropic = orig
    assert_same_container(a, b, "check on vs off")
    warned = any(issubclass(x.category, AnisotropyWarning) for x in w)
    expected = anisotropy(X)
    if bits > 4 or expected is None:
        assert not seen
        return
    assert len(seen) == 1
    ratio = seen[0][0]
    assert abs(ratio - expected) < 1e-6
    event(f"anisotropy.warned={warned}")
    if abs(expected - core.ANISOTROPY_WARN_RATIO) > 1e-6:
        assert warned == (expected >= core.ANISOTROPY_WARN_RATIO)


@S(120)
@given(st.data())
def test_renorm_zero_mean_and_container_metadata(data):
    """renorm is not part of the encoding (README: 'changes how codes are read,
    never what they are'); a zero mean is the uncentred encoding; the container
    records exactly what the quantizer is."""
    d, bits, seed, rotation = data.draw(configs())
    q = Q(d, bits, seed, rotation)
    X = data.draw(corpora(d, n_min=0, q=q))
    a = q.encode(X)
    nr = Quantizer(d, bits, seed=seed, rotation=rotation, renorm=False)
    assert_same_container(a, nr.encode(X), "renorm=False")
    if len(X):
        zm = with_mean(q, np.zeros(d, np.float32)).encode(X)
        assert np.array_equal(a.indices, zm.indices) and bit_equal(a.norms, zm.norms), "zero mean != uncentred"
        assert np.all(zm.mean_dots == 0)
    assert a.indices.dtype == np.uint8 and a.indices.shape == (len(X), d)
    assert a.norms.dtype == np.float32 and a.norms.shape == (len(X),)
    assert (a.d, a.bits, a.rotation, a.seed, a.n) == (d, bits, rotation, seed, len(X))
    assert a.scale is None and a.mean is None and a.mean_dots is None
    assert a.indices.max(initial=0) < 2 ** bits
    s = Q(d, bits, seed, rotation, False, 2.0).encode(X.astype(np.float64))
    assert s.norms is None and s.scale == 2.0 and s.seed == seed and s.rotation == rotation


def test_unknown_rotation_and_bits_are_rejected_everywhere():
    from remex.rotation import ROTATION_CODES, rotation_from_code, validate_rotation
    for bad in ("foo", "", "RHT", None, 3):
        if bad is None:
            continue
        with pytest.raises(ValueError):
            validate_rotation(bad)
    with pytest.raises(ValueError):
        Quantizer(8, 4, rotation="foo")
    with pytest.raises(ValueError):
        rotation_from_code(max(ROTATION_CODES.values()) + 1)
    for bits in (0, 5, 6, 7, 9, -1, 16):
        with pytest.raises(ValueError):
            Quantizer(8, bits)


@S(120)
@given(st.data())
def test_dense_matrix_path_agrees_with_operator_except_at_boundaries(data):
    """Assigning the dense matrix (``q.R = ...``) swaps the operator for
    ``X @ R.T``. The two are the same transform to float32 rounding, so codes may
    differ only on coordinates within ~1e-5 of a boundary (hug corpora put many
    coordinates exactly there)."""
    d, bits, seed, rotation = data.draw(configs(rotations=("rht",)))
    q = Q(d, bits, seed, rotation)
    X = data.draw(corpora(d, n_min=1, min_norm=1e-6, q=q))
    dense = copy.copy(q)
    dense._op = None
    dense._R = rht_rotation(d, seed)
    a, b = q.encode(X).indices, dense.encode(X).indices
    diff = a != b
    event(f"dense_vs_operator.differs={'yes' if diff.any() else 'no'}")
    if diff.any():
        v = semantic_values(q, X)
        margin = np.abs(v[..., None] - q.boundaries.astype(np.float64)).min(axis=-1)
        assert margin[diff].max() < 1e-5


@S(25)
@given(d=st.sampled_from([256, 384, 768, 1000, 1024, 1536, 2048, 3072]),
       seed=st.sampled_from(SEED_POOL), n=st.integers(1, 4),
       dt=st.sampled_from([np.float32, np.float64]))
def test_rht_operator_at_embedding_sized_d(d, seed, n, dt):
    """Native vs NumPy bit-for-bit and vs the dense matrix at real embedding
    sizes (B = 8, 32, 128, 256, 1024 ... all occur)."""
    op = RHTOperator(d, seed)
    X = np.random.default_rng(seed + d).standard_normal((n, d)).astype(dt)
    R = _dense_rht(d, seed)
    rr, ur = op.rotate_rows(X), op.unrotate_rows(X)
    assert np.abs(rr - X.astype(np.float64) @ R.T).max() < 1e-5
    assert np.abs(ur - X.astype(np.float64) @ R).max() < 1e-5
    with numpy_path():
        op2 = RHTOperator(d, seed)
        assert bit_equal(op2.rotate_rows(X), rr) and bit_equal(op2.unrotate_rows(X), ur)
    assert np.abs(op.unrotate_rows(rr) - X).max() < 1e-5


@functools.lru_cache(maxsize=16)
def _dense_rht(d, seed):
    return rht_rotation(d, seed).astype(np.float64)


@pytest.mark.parametrize("bits", BITS)
def test_assign_codes_exhaustive_float16_and_ulp_windows(bits):
    """Bounded-exhaustive: every float16 bit pattern (incl. NaN, inf, subnormal,
    +/-0) and every float32 within 64 ulps of each boundary, for several tables."""
    f16 = np.arange(65536, dtype=np.uint16).view(np.float16)
    for d, sigma in [(1, None), (64, None), (768, None), (3, 0.37), (3, 300.0)]:
        b, _ = lloyd_max_codebook(d, bits, sigma=sigma)
        for hay in (f16, f16.astype(np.float32).reshape(256, 256)):
            assert np.array_equal(assign_codes(hay, b, bits), np.searchsorted(b, hay).astype(np.uint8))
        steps = np.arange(-64, 65)
        win = (b.view(np.int32)[:, None].astype(np.int64) + np.where(b[:, None] < 0, -steps, steps)).astype(np.int32)
        win = np.concatenate([win.view(np.float32).ravel(), (-b).ravel(), [0.0, -0.0]]).astype(np.float32)
        for hay in (win, win.astype(np.float64)):
            assert np.array_equal(assign_codes(hay, b, bits), np.searchsorted(b, hay).astype(np.uint8))
        with numpy_path():
            assert np.array_equal(assign_codes(win, b, bits), np.searchsorted(b, win).astype(np.uint8))


# ---- subprocess: the real REMEX_NO_NATIVE switch --------------------------
def test_env_switch_remex_no_native_gives_identical_codes(tmp_path):
    """The env var is read once, at first kernel(); a fresh interpreter proves
    the switch (not just our in-process monkeypatch) yields the same bits."""
    if not have_native():
        pytest.skip("no compiled kernel")
    rng = np.random.default_rng(11)
    cases = []
    for (d, bits, rotation, normalize) in [(96, 4, "rht", True), (130, 2, "rht", False), (64, 8, "none", False),
                                           (33, 3, "haar", True), (10, 1, "rht", True), (2, 4, "rht", True)]:
        X = (rng.standard_normal((300, d)) * rng.lognormal(0, 3, (300, 1))).astype(np.float64)
        X[0, :min(d, 3)] = [np.nan, np.inf, -np.inf][:min(d, 3)]
        cases.append((d, bits, rotation, normalize, X))
    np.savez(tmp_path / "in.npz", **{f"x{i}": c[4] for i, c in enumerate(cases)})
    spec = [(c[0], c[1], c[2], c[3]) for c in cases]
    code = textwrap.dedent(f"""
        import numpy as np, warnings
        warnings.simplefilter("ignore")
        from remex import Quantizer, _native
        data = np.load({str(tmp_path / 'in.npz')!r})
        out = {{}}
        for i, (d, bits, rotation, normalize) in enumerate({spec!r}):
            X = data[f"x{{i}}"]
            q = Quantizer(d, bits, seed=9, rotation=rotation, normalize=normalize)
            X = X if not normalize else X.astype(np.float32)
            cv = q.encode(X)
            out[f"i{{i}}"] = cv.indices
            if cv.norms is not None:
                out[f"n{{i}}"] = cv.norms.view(np.uint8)
        np.savez({str(tmp_path / 'out.npz')!r}, **out, killed=np.array(_native.kernel() is None))
    """)
    subprocess.run([sys.executable, "-c", code], env=dict(os.environ, REMEX_NO_NATIVE="1"), check=True)
    out = np.load(tmp_path / "out.npz")
    assert bool(out["killed"]), "REMEX_NO_NATIVE=1 did not disable the kernel"
    for i, (d, bits, rotation, normalize, X) in enumerate(cases):
        q = Quantizer(d, bits, seed=9, rotation=rotation, normalize=normalize)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = q.encode(X if not normalize else X.astype(np.float32))
        assert np.array_equal(cv.indices, out[f"i{i}"]), (d, bits, rotation, normalize)
        if cv.norms is not None:
            assert np.array_equal(cv.norms.view(np.uint8), out[f"n{i}"])


def test_concurrent_encodes_from_python_threads_match_serial():
    """The thread pool is process-global and re-created when the worker count
    changes; concurrent callers with different row counts must not corrupt
    each other's output."""
    if not have_native():
        pytest.skip("no compiled kernel")
    import threading
    d = 64
    q = Q(d, 4, 7, "rht")
    rng = np.random.default_rng(3)
    batches = [rng.standard_normal((n, d)).astype(np.float32) for n in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)]
    serial = [q.encode(b).indices.copy() for b in batches]
    results = [None] * len(batches)
    errors = []
    with encode_env(threads=4, parallel_min=1):
        def work(i):
            try:
                for _ in range(20):
                    results[i] = q.encode(batches[i]).indices
                    assert np.array_equal(results[i], serial[i])
            except BaseException as exc:   # noqa: BLE001
                errors.append(exc)
        ts = [threading.Thread(target=work, args=(i,)) for i in range(len(batches))]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert not errors, errors[:1]
