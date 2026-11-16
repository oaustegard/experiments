"""Deterministic regressions for the findings of the area-D property hunt.

Every test here is a plain pytest function with fixed inputs and an explicit
expectation. Each FAILS on remex 2.2.0 (checkout b8d527a) for the reason stated
in its docstring, and should pass once the stated contract holds. Nothing in
this file needs Hypothesis or the discovery harness.

    CONFIRMED  a stated or tested contract is violated (cited in the docstring)
    AMBIGUOUS  behaviour is surprising but no document pins the expectation
    LIMITATION documented, not a bug: marked xfail(strict) so it stays visible

Run:  python3 -m pytest test_encode_regressions.py -q
"""
import math
import warnings

import numpy as np
import pytest
from scipy.special import ndtr

from remex import AnisotropyWarning, Quantizer, corpus_mean
from remex.codebook import assign_codes, coordinate_sigma, lloyd_max_codebook
from remex.rotation import rht_rotation


# ---------------------------------------------------------------------------
# F1  CONFIRMED  rht_rounds >= 2 does not make the seed count at d = 2 (and 4)
# ---------------------------------------------------------------------------
def test_F1_rht_d2_decode_depends_on_the_seed():
    """rotation.py:186-199 and CLAUDE.md ('"rht" takes at least two rounds at every
    d ... with two the seed counts', issue #89): at d=2 the rotation is always a
    signed permutation (H P H stays in the dihedral group D4), so the symmetric
    Lloyd-Max codebook cannot see the seed and every seed decodes bit-identically.
    tests/test_rht_seed.py::test_decode_depends_on_the_seed iterates POW2[1:] and so
    silently skips d=2."""
    X = np.random.default_rng(0).standard_normal((200, 2)).astype(np.float32)
    decodes = set()
    for seed in range(20):
        q = Quantizer(2, bits=2, seed=seed, rotation="rht")
        decodes.add(q.decode(q.encode(X)).tobytes())
    assert len(decodes) > 1, "all 20 seeds decode bit-identically at d=2"


def test_F1_rht_d2_rotation_is_not_a_signed_permutation():
    """Same contract: a randomized rotation of R^2 must be able to leave the
    coordinate axes. rht_rotation(2, s) is in {+-I, +-swap, ...} for every s."""
    for seed in range(50):
        R = rht_rotation(2, seed)
        if np.abs(R).max() < 0.999:
            return
    pytest.fail("rht_rotation(2, seed) is a signed permutation for all 50 seeds")


# ---------------------------------------------------------------------------
# F1b AMBIGUOUS  'enough rounds to mix every coordinate' is only probabilistic
# ---------------------------------------------------------------------------
def test_F1b_rht_small_even_d_maps_no_basis_vector_to_a_basis_vector():
    """rotation.py:186 ('enough to mix every coordinate'). At d=10 (B=2, 4 rounds)
    about 14% of seeds leave some basis vector mapped to +-one coordinate
    (max |R| == 1): that input stays one-hot, far outside a Gaussian cell
    structure. Measured over 300 seeds: d=2 100%, d=4 53%, d=8 27%, d=10 14%,
    d=12 2%, d=14 8%, d=20 0.3%, d>=16 power-of-two none. rht_rounds returns the
    minimum for *reachability* (B**rounds >= d), not for mixing."""
    unmixed = [s for s in range(64) if np.abs(rht_rotation(10, s)).max() > 0.999]
    assert not unmixed, f"seeds with an unmixed basis vector at d=10: {unmixed[:8]} ({len(unmixed)}/64)"


# ---------------------------------------------------------------------------
# F2  CONFIRMED  the 1e-8 absolute length floor breaks scale invariance
# ---------------------------------------------------------------------------
def test_F2_direction_codes_are_scale_invariant_below_1e8():
    """core.py:1093 ``X_unit = target / np.maximum(lengths, 1e-8)``. The codes are
    meant to be those of the unit direction (encode docstring, CLAUDE.md data
    flow: normalize, store norms separately), so encode(c * x) and encode(x) must
    share codes for any c > 0 (exact for c = 2^k). They do for ||c x|| >= 1e-8 and
    not below: x_unit then has length ||x||/1e-8 < 1 and its coordinates fall
    into the central cells. Measured at d=64, 4 bits: mean cosine of
    decode(encode(x)) to x is 0.9956 for ||x|| ~ 7, still 0.9956 at 5e-8, 0.9936
    at 7e-9, 0.80 (1-bit quality) at 7e-12."""
    d = 64
    q = Quantizer(d, 4, seed=3, rotation="rht")
    x = np.random.default_rng(0).standard_normal((50, d)).astype(np.float32)
    base = q.encode(x).indices
    scaled = (x * np.float32(2.0) ** -30)
    assert np.sqrt((scaled.astype(np.float64) ** 2).sum(1)).max() < 1e-8
    got = q.encode(scaled).indices
    assert np.array_equal(got, base), f"{int((got != base).sum())} of {base.size} codes changed under x -> x * 2**-30"


# ---------------------------------------------------------------------------
# F3  CONFIRMED  scale outside the float32-representable range is accepted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1e-300, 1e-46, 1e39])
def test_F3_accepted_scale_gives_a_strictly_increasing_finite_codebook(scale):
    """codebook.py:84-86 casts the float64 table to float32; ``coordinate_sigma``
    (codebook.py:17-22) accepts any finite positive sigma. For sigma < ~1e-43 the
    centroids collapse to 0 (non-strict cells), for sigma > ~7e37 they overflow to
    inf and the midpoints to NaN. tests/test_polar_embed.py::TestCodebook asserts
    strict monotonicity, so the invariant is established; the constructor neither
    keeps it nor raises. (scale=1e300 raises an OverflowError from scipy instead.)"""
    try:
        q = Quantizer(8, 2, normalize=False, rotation="none", scale=scale)
    except ValueError:
        return                       # a clear rejection would also satisfy the contract
    assert np.all(np.isfinite(q.boundaries)), "NaN/inf boundaries"
    assert np.all(np.diff(q.centroids) > 0), "collapsed centroids"
    assert np.all(np.diff(q.boundaries) > 0), "collapsed boundaries"


# ---------------------------------------------------------------------------
# F4  CONFIRMED  assign_codes is not 'identical' to searchsorted outside f32/f64
# ---------------------------------------------------------------------------
def test_F4a_assign_codes_float64_boundaries_are_not_rounded():
    """codebook.py:161-186 docstring: 'np.searchsorted(boundaries, X_rot) ... Identical
    output, not merely close.' ``b = np.ascontiguousarray(boundaries, dtype=float32)``
    for a float32 haystack rounds float64 boundaries; searchsorted compares in
    float64."""
    b = np.array([np.nextafter(float(np.float32(0.1)), -np.inf)])   # float64, just below float32(0.1)
    x = np.array([np.float32(0.1)])
    got, want = assign_codes(x, b, 1), np.searchsorted(b, x).astype(np.uint8)
    assert np.array_equal(got, want), f"assign_codes {got.tolist()} vs searchsorted {want.tolist()}"


def test_F4b_assign_codes_integer_haystack_is_not_truncated_to_float32():
    """Same contract; int64 haystacks are cast to float32 (codebook.py:179-182) while
    searchsorted promotes to float64: 16777217 > 2**24 but float32(16777217) == 2**24."""
    b = np.array([2.0 ** 24], dtype=np.float32)
    x = np.array([2 ** 24 + 1], dtype=np.int64)
    got, want = assign_codes(x, b, 1), np.searchsorted(b, x).astype(np.uint8)
    assert np.array_equal(got, want), f"assign_codes {got.tolist()} vs searchsorted {want.tolist()}"


def test_F4c_assign_codes_big_endian_float64_is_not_truncated_to_float32():
    """Same contract; ``X_rot.dtype == np.float64`` is False for '>f8', so the
    haystack is cast to float32 (codebook.py:179-181)."""
    b = np.array([5.0], dtype=np.float32)
    x = np.array([5.0 + 1e-12], dtype=">f8")
    got, want = assign_codes(x, b, 1), np.searchsorted(b, x).astype(np.uint8)
    assert np.array_equal(got, want), f"assign_codes {got.tolist()} vs searchsorted {want.tolist()}"


def test_F4d_assign_codes_rejects_boundaries_shorter_than_2_pow_bits_minus_1():
    """The C kernel indexes b[c-1] up to c = 2**bits - 1 with no length check
    (_native.py quant kernel, codebook.py:185-195): a shorter table is an out-of-bounds
    read. Here the 'out of bounds' slots are inside a sentinel buffer (-inf), so
    the test itself stays memory-safe while the wrong answer is deterministic.
    Expected: ValueError, or at least searchsorted's answer (<= 3)."""
    buf = np.full(16, -np.inf, dtype=np.float32)
    buf[:3] = [-1.0, 0.0, 1.0]
    b = buf[:3]
    x = np.array([[0.5, -5.0, 5.0]], dtype=np.float32)
    try:
        got = assign_codes(x, b, 4)
    except ValueError:
        return
    assert np.array_equal(got, np.searchsorted(b, x).astype(np.uint8)), f"got {got.tolist()}"


# ---------------------------------------------------------------------------
# F5  CONFIRMED  spurious AnisotropyWarning ('nan') on non-finite input
# ---------------------------------------------------------------------------
def test_F5_non_finite_row_does_not_trigger_a_common_direction_warning():
    """core.py:168-171 ``if ratio is None or ratio < ANISOTROPY_WARN_RATIO: return``
    lets NaN through (nan < 0.75 is False), so an inf entry in a >=100-row, <=4-bit
    batch emits "These 120 vectors share a large common direction (||mean|| / mean
    ||x|| = nan)". The claim is false, and NaN input takes a different branch
    (norm_total > 0.0 is False) and stays silent. ``anisotropy()`` documents None for
    'says nothing'; the warning should key on a finite ratio."""
    X = np.random.default_rng(0).standard_normal((120, 16)).astype(np.float32)
    X[3, 2] = np.inf
    q = Quantizer(16, 2, seed=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        q.encode(X)
    msgs = [str(x.message) for x in w if issubclass(x.category, AnisotropyWarning)]
    assert not msgs, msgs[0][:120]


# ---------------------------------------------------------------------------
# F6  CONFIRMED (design-level)  8-bit Lloyd-Max is not converged at n_iter=300
# ---------------------------------------------------------------------------
def _mse_unit(c):
    c = np.asarray(c, dtype=np.float64)
    b = np.concatenate([[-np.inf], (c[:-1] + c[1:]) / 2, [np.inf]])
    lo, hi = b[:-1], b[1:]
    phi = lambda x: np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def xphi(x):
        f = np.isfinite(x)
        xs = np.where(f, x, 0.0)
        return np.where(f, xs * phi(xs), 0.0)

    i0, i1 = ndtr(hi) - ndtr(lo), phi(lo) - phi(hi)
    i2 = i0 - (xphi(hi) - xphi(lo))
    return float(np.sum(i2 - 2 * c * i1 + c * c * i0))


def test_F6_8bit_codebook_is_within_1_percent_of_the_lloyd_max_optimum():
    """codebook.py:25-47 documents 'optimal Lloyd-Max codebook' (README/CLAUDE.md:
    Lloyd-Max optimal). The simultaneous (Jacobi) update started from
    linspace(-3, 3) converges slowly in the tails: after 300 iterations the 8-bit
    table has MSE 4.941e-5 (sigma = 1) against 4.1185e-5 at the fixed point
    (200,000 iterations, independent ndtr implementation; the high-resolution
    theory value is 4.15e-5): +20%. At 4 bits the gap is 1e-6 relative.
    NOTE the fixed-point residual is only 6e-4 sigma, so a 'centroid = conditional
    mean' check does not see this; MSE does. Fixing it changes every stored 8-bit
    code, so it needs a versioned codebook."""
    _, c = lloyd_max_codebook(1, 8, sigma=1.0)
    optimum = 4.1185082867e-05
    assert _mse_unit(c) <= optimum * 1.01, f"MSE {_mse_unit(c):.4e} vs optimum {optimum:.4e}"


# ---------------------------------------------------------------------------
# F7  CONFIRMED (corner)  scalar mode: saturation claim fails under rotation
# ---------------------------------------------------------------------------
def test_F7_scalar_mode_saturates_huge_finite_values_under_rht():
    """core.py:947-950 ('values that would overflow either ... quantize to the
    extreme cell instead of to inf/nan') and README 'saturates at the outermost
    cell'. With rotation='rht' the *intermediate* FWHT sums overflow float64 once
    |x| > ~1.8e308 / sqrt(B) (d=64: ~2e307): inf - inf = NaN, and NaN goes to the
    top cell, so ~30-50% of coordinates get the wrong extreme (sign lost) instead
    of saturating by sign. rotation='none' is fine; 'haar' fails only at 1.7e308."""
    d = 64
    q = Quantizer(d, 2, seed=3, rotation="rht", normalize=False, scale=1.0)
    X = np.clip(np.random.default_rng(1).standard_normal((20, d)) * 1e307, -1.7e308, 1.7e308)
    # same directions, scaled down so nothing overflows; saturation depends only on sign
    got, ref = q.encode(X).indices, q.encode(X * 2.0 ** -200).indices
    assert np.array_equal(got, ref), f"{int((got != ref).sum())} of {got.size} codes lost their sign to overflow"


# ---------------------------------------------------------------------------
# F8  CONFIRMED (minor)  coordinate_sigma / lloyd_max_codebook do not validate d
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [-4, 0])
def test_F8_coordinate_sigma_rejects_non_positive_d(d):
    """codebook.py:17-22 validates an explicit sigma ('must be finite and
    positive') but not the default 1/sqrt(d): d=-4 returns NaN (plus a NumPy
    RuntimeWarning) and lloyd_max_codebook(-4, 2) returns an all-NaN table; d=0
    raises ZeroDivisionError. Both functions are public (remex.__all__).
    CLAUDE.md: 'A failure degrades to None ... never to a legal-looking value'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            coordinate_sigma(d)
        with pytest.raises(ValueError):
            lloyd_max_codebook(d, 2)


# ---------------------------------------------------------------------------
# F9  AMBIGUOUS  seed=None is accepted but R and the operator disagree
# ---------------------------------------------------------------------------
def test_F9_seed_none_R_is_the_rotation_encode_applies():
    """core.py:975 types seed as int, but None passes validation. For rotation='rht'
    the operator draws default_rng(None) in __init__ and ``Quantizer.R`` draws a
    second, unrelated one on first read (core.py:1149-1151), so the matrix that
    ``GPUSearcher``/``save_params`` use is not the one ``encode`` applied
    (max |difference| = 1.0). Either reject None or draw one SeedSequence."""
    q = Quantizer(16, 4, seed=None, rotation="rht")
    I = np.eye(16, dtype=np.float32)
    assert np.abs(q._op.rotate_rows(I) - q.R.T).max() < 1e-5


# ---------------------------------------------------------------------------
# F10 AMBIGUOUS  corpus_mean(empty) returns NaN with RuntimeWarnings
# ---------------------------------------------------------------------------
def test_F10_corpus_mean_of_an_empty_corpus_is_not_a_nan_vector():
    """core.py:113-122. corpus_mean(np.empty((0, d))) returns a NaN vector and
    emits 'Mean of empty slice'. Quantizer(mean=...) then raises 'mean must be
    finite', so the failure is only caught one call later and far from its cause.
    anisotropy() returns None for the same input. Expected: ValueError."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = corpus_mean(np.empty((0, 4), np.float32))
        except ValueError:
            return
    assert np.all(np.isfinite(m)), "corpus_mean returned %r" % (m,)


# ---------------------------------------------------------------------------
# F11 AMBIGUOUS  normalized encode accepts NaN/inf silently
# ---------------------------------------------------------------------------
def test_F11_normalized_encode_flags_non_finite_input():
    """Scalar mode documents NaN -> top cell (core.py:1128-1132). Normalized mode
    documents nothing: a NaN/inf entry yields NaN/inf in that row's norm and
    all-top-cell codes, with no error and no warning, so search scores for the row
    become NaN. mean= and scale= both reject non-finite values (core.py:108,
    codebook.py:20). Expected: ValueError (or a documented sentinel)."""
    X = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
    X[1, 2] = np.nan
    q = Quantizer(16, 4, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # a warning is an acceptable flag too
        try:
            cv = q.encode(X)
        except (ValueError, RuntimeWarning):
            return
    pytest.fail(f"encode accepted a NaN entry silently (norms={cv.norms.tolist()})")


# ---------------------------------------------------------------------------
# L1  LIMITATION (documented)  Haar codes depend on the BLAS batch shape
# ---------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason="documented: CLAUDE.md 'haar still depends on the BLAS kernel'; "
                                       "encode comment at core.py:1052-1055 claims block-size independence")
def test_L1_haar_encode_is_independent_of_batch_shape():
    """d=8, 2 bits, seed 1: row 0 encoded alone vs as part of a 2-row batch differs
    at coordinate 5 (code 1 vs 2) because X @ R.T takes a different BLAS path for
    n=1 (gemv) than n=2 (gemm). Only coordinates within float32 rounding of a
    boundary can differ (the harness asserts that). Not a bug under the stated
    contract; recorded so nobody reads core.py:1052 as covering Haar."""
    pair = np.array([[-0.006528499070554972, 0.09520027041435242, 0.14480805397033691, 0.03586587682366371,
                      0.11877724528312683, -0.03600447624921799, -0.4123893678188324, -0.18212418258190155],
                     [-0.4331320524215698, 0.5595385432243347, -0.9699552059173584, 0.9363618493080139,
                      -0.040080152451992035, -0.6242416501045227, -0.2245211899280548, -1.113567590713501]],
                    dtype=np.float32)
    q = Quantizer(8, 2, seed=1, rotation="haar")
    assert np.array_equal(q.encode(pair).indices[0], q.encode(pair[:1]).indices[0])
