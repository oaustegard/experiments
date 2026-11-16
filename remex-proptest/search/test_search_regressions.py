"""Deterministic regression tests for the findings of the area-A property hunt.

Plain pytest, no Hypothesis, no shared helpers: each test is the minimal
standalone repro for one finding and FAILS on remex 2.2.0 (checkout at
/home/user/remex, HEAD b8d527a) for the reason in its docstring.  It should
pass once the fix sketched in the report lands.

Confirmed findings:      test_F2_*, test_F3_*, test_F5_*, test_F4_*
Ambiguous (see report):  test_AMBIGUOUS_*

Run:  pytest -q test_search_regressions.py -p no:cacheprovider
"""
import math
import warnings

import numpy as np
import pytest

from remex import IVFCoarseIndex, PackedVectors, Quantizer

warnings.simplefilter("ignore")


# --------------------------------------------------------------------------
# F1  Quantizer.search skips the bits/mean guards once the container cache is warm
# --------------------------------------------------------------------------
def _writer_and_reader(kind):
    d = 8
    if kind == "scalar_bits":
        return (Quantizer(d, bits=4, seed=1, normalize=False),
                Quantizer(d, bits=2, seed=1, normalize=False))
    if kind == "renorm_off_bits":
        return (Quantizer(d, bits=4, seed=1, renorm=False),
                Quantizer(d, bits=2, seed=1, renorm=False))
    if kind == "renorm_off_mean_dropped":
        mean = np.random.default_rng(5).standard_normal(d).astype(np.float32)
        return (Quantizer(d, bits=4, seed=1, mean=mean),
                Quantizer(d, bits=4, seed=1, renorm=False))
    raise KeyError(kind)


KINDS_F1 = ["scalar_bits", "renorm_off_bits", "renorm_off_mean_dropped"]


@pytest.mark.parametrize("kind", KINDS_F1)
def test_F1_control_cold_container_refuses_a_mismatched_reader(kind):
    """Control: with nothing cached, `search` refuses a reader whose bits (or
    mean) differ from the writer's, as README.md:23 promises."""
    writer, reader = _writer_and_reader(kind)
    rng = np.random.default_rng(0)
    cv = writer.encode(rng.standard_normal((20, 8)))
    with pytest.raises(ValueError, match="mismatch"):
        reader.search(cv, rng.standard_normal(8), k=3)


@pytest.mark.parametrize("kind", KINDS_F1)
def test_F1_search_refuses_a_mismatched_reader_after_the_cache_is_warm(kind):
    """The same call on the same container, after the legitimate writer has
    run one `search` (which fills `_x_hat_rot`), is ACCEPTED and returns the
    writer's scores.  `search` only runs `_check_rotation` itself and relies on
    `_get_x_hat_rot -> _resolve_centroids` for `_check_bits`/`_check_mean`
    (core.py:1665-1669, 2005-2006: the cache returns before the guards), and
    `_effective_norms`'s own guard (core.py:1408-1412, added for exactly this
    reason) is skipped by its early return for scalar mode and renorm=False
    (core.py:1406).  `search_adc`, `search_batch`, `search_twostage`, IVF and
    GPUSearcher all refuse."""
    writer, reader = _writer_and_reader(kind)
    rng = np.random.default_rng(0)
    cv = writer.encode(rng.standard_normal((20, 8)))
    q = rng.standard_normal(8)
    writer.search(cv, q, k=3)                       # legitimate use warms the cache
    with pytest.raises(ValueError, match="mismatch"):
        reader.search(cv, q, k=3)                   # returns scores today


# --------------------------------------------------------------------------
# F2  Quantizer._score_vectors rejects PackedVectors unless the codes are centred
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["uncentred", "uncentred_norenorm", "scalar"])
def test_F2_score_vectors_accepts_packed_for_every_mode(kind):
    """`_score_vectors` has an explicit PackedVectors branch and
    tests/test_v2_fixes.py::test_score_vectors_accepts_packed pins it, but only
    for centred codes.  Uncentred and scalar codes fall through to
    `self.decode(packed)`, which reads `packed.indices` -> AttributeError
    (core.py:1617-1618 -> 1595 -> _resolve_indices core.py:2038-2045)."""
    rng = np.random.default_rng(0)
    d = 8
    kw = {"uncentred": {}, "uncentred_norenorm": {"renorm": False},
          "scalar": {"normalize": False}}[kind]
    pq = Quantizer(d, bits=2, seed=3, **kw)
    cv = pq.encode(rng.standard_normal((5, d)))
    packed = PackedVectors.from_compressed(cv)
    for precision in (None, 1):
        Y, bias = pq._score_vectors(cv, precision)
        Yp, biasp = pq._score_vectors(packed, precision)   # AttributeError today
        np.testing.assert_allclose(Yp, Y, rtol=1e-6, atol=1e-6)
        assert (bias is None) and (biasp is None)


# --------------------------------------------------------------------------
# F3  IVF at nprobe = 2**n_bits is not byte-identical to the flat paths (centred)
# --------------------------------------------------------------------------
def _centred_setup():
    rng = np.random.default_rng(0)
    d, n = 8, 40
    X = rng.standard_normal((n, d)).astype(np.float32)
    pq = Quantizer(d, bits=2, seed=0, mean=X.mean(0))
    cv = pq.encode(X)
    queries = rng.standard_normal((20, d)).astype(np.float32)
    return pq, cv, queries, n


def _by_row(ids, scores, n):
    out = np.full(n, np.nan, np.float32)
    out[ids] = scores
    return out


def test_F3_ivf_flat_scan_is_byte_identical_to_search_adc_for_centred_codes():
    """README.md:395-398 ("tests verify byte-identical agreement with
    Quantizer.search_adc / Quantizer.search_twostage") and the table row
    "bit-identical to flat".  True for uncentred codes; for centred codes
    `IVFCoarseIndex.search_coarse` adds `+ offset` then `+ bias`
    (ivf.py:374-379) while `search_adc` adds `+ bias` then `+ offset`
    (core.py:1732-1733), so float32 rounding differs by 1 ulp."""
    pq, cv, queries, n = _centred_setup()
    ivf = IVFCoarseIndex(pq, cv, n_bits=3, mode="rotated_prefix")
    for q in queries:
        for precision in (None, 1):
            got = _by_row(*ivf.search_coarse(q, k=n, nprobe=ivf.n_cells, precision=precision), n)
            want = _by_row(*pq.search_adc(cv, q, k=n, precision=precision), n)
            assert np.array_equal(got, want), (
                f"max |diff| {np.abs(got - want).max():.3e} "
                f"({np.abs(got.view(np.int32) - want.view(np.int32)).max()} ulp)")


def test_F3_ivf_twostage_flat_scan_is_byte_identical_to_quantizer_twostage_for_centred_codes():
    """Same root cause (F3) on the stage-2 rerank: ivf.py:447-454 adds offset before
    bias, Quantizer.search_twostage (core.py:1826-1829) the other way round."""
    pq, cv, queries, n = _centred_setup()
    ivf = IVFCoarseIndex(pq, cv, n_bits=3, mode="lsh")
    for q in queries:
        got = _by_row(*ivf.search_twostage(q, k=n, candidates=n, nprobe=ivf.n_cells, coarse_precision=1), n)
        want = _by_row(*pq.search_twostage(cv, q, k=n, candidates=n, coarse_precision=1), n)
        assert np.array_equal(got, want)


def test_F3_control_uncentred_ivf_flat_scan_is_byte_identical():
    """Control: the claim holds for uncentred codes, so a failure above is
    specific to the centred add-order, not to IVF in general."""
    rng = np.random.default_rng(0)
    d, n = 8, 40
    pq = Quantizer(d, bits=2, seed=0)
    cv = pq.encode(rng.standard_normal((n, d)))
    ivf = IVFCoarseIndex(pq, cv, n_bits=3, mode="rotated_prefix")
    for q in rng.standard_normal((20, d)):
        got = _by_row(*ivf.search_coarse(q, k=n, nprobe=ivf.n_cells, precision=1), n)
        want = _by_row(*pq.search_adc(cv, q, k=n, precision=1), n)
        assert np.array_equal(got, want)


# --------------------------------------------------------------------------
# F5  candidate_indices(nprobe=1) hands out a view of the index's own storage
# --------------------------------------------------------------------------
def test_F5_candidate_indices_does_not_alias_the_index():
    """`candidate_indices` returns `sorted_idx[a:b]` (a view) when exactly one
    cell is probed (ivf.py:274-278) but a fresh array for two or more cells
    (np.concatenate, ivf.py:279-283).  A caller that edits the result in place
    (sort, zero, shuffle) silently corrupts the index for every later query."""
    rng = np.random.default_rng(0)
    pq = Quantizer(8, bits=2, seed=1)
    cv = pq.encode(rng.standard_normal((50, 8)))
    ivf = IVFCoarseIndex(pq, cv, n_bits=2, mode="rotated_prefix")
    before = ivf.sorted_idx.copy()
    cell = int(ivf.cell_ids[0])                      # a non-empty cell
    cand = ivf.candidate_indices(cell, 1)
    assert len(cand) > 0
    cand[:] = 0                                      # caller scribbles on its result
    assert np.array_equal(ivf.sorted_idx, before), "index storage was modified through the returned array"


def test_F5_control_two_cells_are_already_a_copy():
    rng = np.random.default_rng(0)
    pq = Quantizer(8, bits=2, seed=1)
    cv = pq.encode(rng.standard_normal((50, 8)))
    ivf = IVFCoarseIndex(pq, cv, n_bits=2, mode="rotated_prefix")
    assert not np.shares_memory(ivf.candidate_indices(0, 2), ivf.sorted_idx)


# --------------------------------------------------------------------------
# F4  q.mu, a row-constant, is added in float32 *before* top-k and can reorder
# --------------------------------------------------------------------------
def test_F4_constant_query_offset_cannot_reorder_rows(monkeypatch):
    """core.py:1337-1346: `_query_offset` is "constant across the corpus for a
    given query, so it cannot reorder anything".  Every path adds it to the
    float32 score vector *before* selecting the top-k (core.py:1672, 1733,
    1829, 1887-1890; ivf.py:374, 451; gpu.py:375, 406, 455).  At
    q.mu ~ 1e12 (||mu|| = 1e6, ||q|| ~ ||mu||) the float32 spacing is 65536, the
    two rows' score gap (~4e4) disappears, the scores tie and index order
    decides: the worse row wins.  Without the offset the better row wins."""
    d = 4
    mu = np.array([1e6, 0.0, 0.0, 0.0])
    pq = Quantizer(d, bits=8, seed=1, rotation="haar", mean=mu.astype(np.float32))
    X = mu + np.random.default_rng(11).uniform(-1, 1, (2, d))   # two rows, residual ~ 1
    cv = pq.encode(X)
    q = mu + np.array([0.0, 1.0, 0.0, 0.0])

    with_offset, scores = pq.search_adc(cv, q, k=2)
    assert scores[0] == scores[1], "precondition: the offset rounds both rows to one float32"

    monkeypatch.setattr(Quantizer, "_query_offset", lambda self, query: 0.0)
    ranked_without, s0 = pq.search_adc(cv, q, k=2)
    assert abs(float(s0[0]) - float(s0[1])) > 1e4, "precondition: rows are clearly separated without the offset"

    assert list(with_offset) == list(ranked_without), (
        f"adding the constant q.mu reordered the rows: {list(with_offset)} vs {list(ranked_without)}")


# --------------------------------------------------------------------------
# Ambiguous: documented limitation or defect, maintainer's call
# --------------------------------------------------------------------------
def test_AMBIGUOUS_at_precision_accepts_every_matryoshka_level():
    """`search(precision=5)` is accepted on an 8-bit container (nested tables
    exist for 7..1 bits) and `PackedVectors.at_precision` documents
    "target_bits: 1 to self.bits" and passes its own range check, then fails in
    `packed_nbytes` with "Bit width 5 is not supported" (core.py:605 ->
    packing.py:144) because only SUPPORTED_BITS = (1, 2, 3, 4, 8) are
    packable.  Either the docstring should say 1-4 or 8, or at_precision should
    refuse earlier with its own message."""
    rng = np.random.default_rng(0)
    pq = Quantizer(8, bits=8, seed=0)
    cv = pq.encode(rng.standard_normal((6, 8)))
    pq.search(cv, rng.standard_normal(8), k=3, precision=5)           # accepted
    packed = PackedVectors.from_compressed(cv)
    low = packed.at_precision(5)                                       # ValueError today
    assert low.bits == 5


def test_AMBIGUOUS_ivf_query_hash_matches_corpus_hash_for_centred_codes():
    """Corpus cells hash the direction of the *residual* x - mu (the MSBs of
    the centred codes), but `query_cell` hashes the raw query (ivf.py:232-252).
    A corpus vector used as its own query therefore lands in a different cell
    usually (measured 5-22% self-hit centred vs 100% uncentred,
    rotated_prefix, d=64, n_bits=6, anisotropy ~0.9).  IVF is approximate by
    design, so this is a recall-quality observation, not a flat-scan
    violation; hashing `q - mu` makes the self-hit rate 100%."""
    rng = np.random.default_rng(0)
    d, n = 64, 600
    mu = rng.standard_normal(d)
    mu *= 3.0 / np.linalg.norm(mu)
    X = mu + 0.3 * rng.standard_normal((n, d))
    pq = Quantizer(d, bits=4, seed=1, mean=X.mean(axis=0))
    cv = pq.encode(X)
    ivf = IVFCoarseIndex(pq, cv, n_bits=6, mode="rotated_prefix")
    hit = np.mean([ivf.query_cell(X[i]) == int(ivf.cell_ids[i]) for i in range(200)])
    assert hit > 0.9, f"a stored vector queried as itself hashes to its own cell only {hit:.0%} of the time"
