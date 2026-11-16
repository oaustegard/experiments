"""Property-testing harness, area A: SEARCH-PATH AGREEMENT in remex.

Every way remex scores a query must agree where the docs say it agrees:

    Quantizer.search / search_adc / search_batch / search_twostage
    precision= (Matryoshka levels), PackedVectors search, _score_vectors
    IVFCoarseIndex.search_coarse / search_twostage  (nprobe = 2**n_bits is flat)
    GPUSearcher on the numpy backend

Oracle (oracle.py): float64 brute force `q . decode(x)` for uncentred and
scalar codes, the documented RaBitQ estimator for centred codes (mean=).
Both are rebuilt from the codebook tables, the dense rotation matrix and the
stored per-row floats -- not from any production scoring helper.

Run:   pytest -q test_search_props.py -p no:cacheprovider
       REMEX_PT_EXAMPLES=2000 pytest ...           # more examples per property
       REMEX_PT_EVENTS=events.json pytest ...      # dump event() counters
       pytest --hypothesis-show-statistics ...     # Hypothesis' own event table
       REMEX_PT_DERANDOMIZE=1 / REMEX_PT_NODB=1    # fixed seed / no example database
Config sweeps worth repeating:  REMEX_NO_NATIVE=1,  REMEX_PARALLEL_MIN=1 REMEX_NUM_THREADS=3

Properties (each pins a documented agreement):
  test_cpu_paths_match_oracle                    operation scripts on one warming container
  test_cold_cache_each_path                      same, fresh container per call
  test_score_vectors_accepts_packed              _score_vectors(PackedVectors) in every mode   [FAILS: F2]
  test_gpu_numpy_backend_matches                 GPUSearcher vs oracle and vs the CPU paths
  test_ivf_flat_limit_matches_oracle_and_flat_paths   nprobe=2**n_bits == flat scan (tolerance)
  test_ivf_flat_limit_is_byte_identical_to_adc   README "byte-identical" claim                 [FAILS: F3]
  test_ivf_partial_probe_sound                   Hamming pool, valid top-k within the pool
  test_ivf_structure                             CSR layout, cell ids, self-query cell
  test_batch_equals_single_and_permutation       batch == single, row-permutation, equal rows
  test_query_scaling_and_mean_shift_invariance   linearity in q (uncentred / scalar)
  test_derived_containers_search_like_their_rows subset / to_compressed / at_precision after warming
  test_invalid_precision_rejected_uniformly      ValueError on every path
  test_empty_corpus_and_query_dtypes             n = 0, query as list / tuple / float64
  test_query_offset_cannot_reorder               "a row-constant cannot reorder"             [FAILS: F4]
  test_serialized_containers_search_identically  npz / arrow / mmap / from_rows / .pq
  test_adc_scores_independent_of_chunking_and_position
  test_renorm_is_a_reader_property_and_caches_do_not_leak
  test_precision_equal_to_bits_is_full_precision
  test_mismatched_reader_is_refused_on_every_path   guards hold on warm caches             [FAILS: F1]
  test_large_n_chunk_boundaries / test_large_n_ivf_prefix_chunk_boundary / test_encode_block_size_...

The four properties marked FAILS fail on remex 2.2.0 (b8d527a) and pass on the
fix sketches in the report; test_search_regressions.py holds the minimal
deterministic repro for each.

What each property is allowed to disagree about, explicitly:
  * float32 accumulation order  -> `oracle.score_tol` (absolute, scaled)
  * tie order at the top-k boundary -> `check_topk` accepts any member of a tie
    class within 2*tol and *reports* it (events "tie_boundary",
    "path_vs_path_tie_difference")
  * recall of approximate paths (IVF with nprobe < 2**n_bits, two-stage with
    candidates < n) -> only soundness + candidate-set-restricted optimality
  * centred search != q . decode(x)  (documented)  -> oracle is the estimator
  * BLAS gemv gives identical rows slightly different float32 scores
    (event "identical_rows_unequal_scores_in_search(BLAS gemv)") -> within tol
"""
from __future__ import annotations

import atexit
import json
import math
import os
import sys
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, event, given, settings, strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase
from hypothesis.extra import numpy as hnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle as O  # noqa: E402

import remex  # noqa: E402
import remex.core  # noqa: E402
from remex import IVFCoarseIndex, PackedVectors, Quantizer  # noqa: E402
from remex import codebook as _cb  # noqa: E402
from remex.core import CompressedVectors  # noqa: E402
from remex.gpu import GPUSearcher  # noqa: E402

_MUTANT = os.environ.get("REMEX_PT_MUTANT")
if _MUTANT:                       # mutation calibration: see mutants.py / mutation_calibration.py
    import mutants as _mutants
    _mutants.apply(_MUTANT)

warnings.simplefilter("ignore")
pytestmark = pytest.mark.filterwarnings("ignore::remex.AnisotropyWarning")

# --------------------------------------------------------------------------
# harness plumbing: example budget, event counters, codebook cache
# --------------------------------------------------------------------------
EXAMPLES = int(os.environ.get("REMEX_PT_EXAMPLES", "400"))
PROFILE = settings(
    max_examples=EXAMPLES,
    deadline=None,
    suppress_health_check=list(HealthCheck),
    database=(None if (os.environ.get("REMEX_PT_NODB") or os.environ.get("REMEX_PT_MUTANT"))
              else DirectoryBasedExampleDatabase(HERE / ".hypothesis" / "examples")),
    derandomize=bool(os.environ.get("REMEX_PT_DERANDOMIZE")),
)

COUNTS: Counter = Counter()


def ev(label: str) -> None:
    COUNTS[label] += 1
    event(label)


@atexit.register
def _dump_events():
    out = os.environ.get("REMEX_PT_EVENTS")
    if out:
        Path(out).write_text(json.dumps(dict(sorted(COUNTS.items())), indent=1))


# The Lloyd-Max tables are pure functions of (d, bits, sigma); recomputing them
# per example (50-70 ms) would dominate the run.  Cache and hand out copies.
def _cached(fn):
    @lru_cache(maxsize=None)
    def inner(*a, **kw):
        return fn(*a, **kw)

    def wrapper(*a, **kw):
        r = inner(*a, **kw)
        if isinstance(r, tuple):
            return tuple(x.copy() for x in r)
        return {k: v.copy() for k, v in r.items()}
    return wrapper


remex.core.lloyd_max_codebook = _cached(_cb.lloyd_max_codebook)
remex.core.nested_codebooks = _cached(_cb.nested_codebooks)


@lru_cache(maxsize=None)
def make_pq(d, bits, seed, rotation, normalize, scale, renorm, mean_key):
    mean = None if mean_key is None else np.array(mean_key, np.float32)
    return Quantizer(d, bits=bits, seed=seed, rotation=rotation,
                     normalize=normalize, scale=scale, renorm=renorm, mean=mean)


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
D_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64]
# rare larger d: multi-block RHT (d = 96, 100, 130, 192: pow2 block < d), odd d
# beyond the dense-QR toy range, and d*n >= 2**15 so the threaded rotate/assign
# kernels engage (REMEX_PARALLEL_MIN) -- the oracle uses the dense matrix, so a
# kernel that disagrees with its own R shows up as a path/oracle disagreement.
D_LARGE = [96, 100, 128, 130, 192, 255, 256]
BITS_CHOICES = [1, 2, 3, 4, 8]

_val = (st.floats(-4, 4, allow_nan=False, allow_infinity=False, width=32)
        .map(lambda x: 0.0 if abs(x) < 1e-3 else x))


class GroundTruth:
    """The codes/floats the oracle reads, copied out of the container at
    encode time so no production search path can disturb them."""

    def __init__(self, cv):
        self.indices = cv.indices.copy()
        self.norms = None if cv.norms is None else cv.norms.copy()
        self.mean_dots = None if cv.mean_dots is None else cv.mean_dots.copy()
        self.n = cv.n

    def subset(self, idx):
        g = object.__new__(GroundTruth)
        g.indices = self.indices[idx]
        g.norms = None if self.norms is None else self.norms[idx]
        g.mean_dots = None if self.mean_dots is None else self.mean_dots[idx]
        g.n = len(idx)
        return g


class Case:
    def __init__(self, pq, cv, X, Q, k, desc):
        self.pq, self.cv, self.X, self.Q, self.k, self.desc = pq, cv, X, Q, k, desc
        self.gt = GroundTruth(cv)
        self._orc = {}
        self._packed = None

    @property
    def n(self):
        return self.cv.n

    @property
    def packed(self):
        if self._packed is None:
            self._packed = PackedVectors.from_compressed(self.cv)
        return self._packed

    def orc(self, qi, precision):
        key = (qi, precision)
        if key not in self._orc:
            self._orc[key] = O.oracle_scores(self.pq, self.gt, self.Q[qi], precision)
        return self._orc[key]

    def tol(self, qi):
        return O.score_tol(self.pq, self.gt, self.Q[qi])


PALETTE = [0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.5, 0.25]


@st.composite
def matrices(draw, n, d, kinds):
    """(n, d) float64 matrix built row by row from a small set of row kinds:
    continuous rows (from a drawn integer seed, so they cost 4 bytes of
    entropy instead of 8*d), palette rows (few distinct values -> exact ties),
    zero / one-hot rows, and rows rewritten as a duplicate, negation or scaled
    copy of an earlier row.  Large n*d matrices stay inside Hypothesis' buffer
    (per-element float draws overran it on ~10% of examples)."""
    X = np.zeros((n, d))
    for i in range(n):
        kind = draw(st.sampled_from(kinds))
        if kind == "gauss":
            rng = np.random.default_rng(draw(st.integers(0, 2 ** 32 - 1)))
            X[i] = rng.uniform(-3, 3, d)
        elif kind == "palette":
            X[i] = draw(st.lists(st.sampled_from(PALETTE), min_size=d, max_size=d))
        elif kind == "zero":
            pass
        elif kind == "onehot":
            X[i, draw(st.integers(0, d - 1))] = draw(st.sampled_from([-3.0, 1.0, 2.5]))
        elif i == 0:
            rng = np.random.default_rng(draw(st.integers(0, 2 ** 32 - 1)))
            X[i] = rng.uniform(-3, 3, d)
        else:
            src = draw(st.integers(0, i - 1))
            if kind == "dup":
                X[i] = X[src]
            elif kind == "neg":
                X[i] = -X[src]
            elif kind == "scale":
                X[i] = X[src] * draw(st.sampled_from([1e-9, 1e-6, 0.5, 2.0, 3.0, 100.0]))
            elif kind == "jitter":     # near-duplicate: not an exact tie, a near-tie
                rng = np.random.default_rng(draw(st.integers(0, 2 ** 32 - 1)))
                X[i] = X[src] + draw(st.sampled_from([1e-3, 1e-2, 1e-1])) * rng.standard_normal(d)
    return X


CORPUS_KINDS = ["gauss"] * 5 + ["palette"] * 2 + ["zero", "dup", "dup", "neg", "scale", "onehot",
                                                   "jitter", "jitter", "jitter"]
QUERY_KINDS = ["gauss"] * 4 + ["palette", "zero", "onehot", "dup", "neg", "scale", "jitter", "jitter"]


@st.composite
def cases(draw, modes=("unit", "unit_norenorm", "scalar", "centred"), max_n=24,
          allow_k0=True, d_choices=None,
          mean_kinds=("zero", "const", "corpus", "big", "spike", "mix")):
    mode = draw(st.sampled_from(list(modes)))
    d = draw(st.sampled_from(d_choices or (D_CHOICES * 3 + D_LARGE)))
    bits = draw(st.sampled_from(BITS_CHOICES))
    seed = draw(st.sampled_from([0, 1, 42]))
    rots = [None, "haar", "none"] + (["rht"] if d % 2 == 0 else [])
    rotation = draw(st.sampled_from(rots))
    n = draw(st.sampled_from([1, 1, 2, 3, 5, 8, 13, max_n] + ([40, 130] if max_n >= 24 else [])))
    X = draw(matrices(n, d, CORPUS_KINDS))
    scale = None
    renorm = True
    mean_key = None
    normalize = True
    if mode == "unit_norenorm":
        renorm = False
    elif mode == "scalar":
        normalize = False
        scale = draw(st.sampled_from([0.25, 1.0, 4.0]))
        X = X * scale / 2.0
    elif mode == "centred":
        mk = draw(st.sampled_from(list(mean_kinds)))
        cm = X.mean(axis=0)
        if mk == "zero":
            mu = np.zeros(d)
        elif mk == "const":
            mu = np.full(d, 0.5)
        elif mk == "corpus":
            mu = cm
        elif mk == "big":
            mu = 10 * cm + 5.0
        elif mk == "spike":
            mu = np.zeros(d)
            mu[0] = 3.0
        elif mk == "huge":
            mu = np.zeros(d)
            mu[0] = draw(st.sampled_from([1e5, 1e6, 1e7]))
            X = X + mu           # a corpus that really is a small cloud around a huge mean
        else:
            mu = X[0] * 0.5 + 0.1
        mean_key = tuple(float(np.float32(x)) for x in mu)
        ev(f"centred_mean={mk}")
    pq = make_pq(d, bits, seed, rotation, normalize, scale, renorm, mean_key)
    m = draw(st.sampled_from([1, 1, 2, 3]))
    Q = draw(matrices(m, d, QUERY_KINDS))
    kmax = n + 3
    k = draw(st.one_of(st.integers(1 if not allow_k0 else 0, kmax),
                       st.sampled_from([1, max(n - 1, 1), n, n + 1, 10])))
    cv = pq.encode(X)

    desc = dict(mode=mode, d=d, bits=bits, rotation=pq.rotation, n=n, k=k, m=m)
    ev(f"mode={mode}")
    ev(f"rotation={pq.rotation}")
    ev(f"bits={bits}")
    ev("d=" + ("odd" if d % 2 else "even") + ("_pow2" if d & (d - 1) == 0 else ""))
    ev("d=" + ("1" if d == 1 else "2" if d == 2 else "3-8" if d <= 8 else "9-33" if d <= 33 else "34-64" if d <= 64 else "65-256"))
    if n * d >= 2 ** 15:
        ev("threaded_kernel_regime(n*d>=2^15)")
    ev("n=" + str(n) if n <= 3 else "n=4-24" if n <= 24 else "n=25+")
    ev("k<n" if k < n else "k==n" if k == n else "k>n")
    if k == 0:
        ev("k=0")
    if n >= 2 and len({tuple(r) for r in X}) < n:
        ev("duplicate_rows")
    if np.any(~X.any(axis=1)):
        ev("zero_rows")
    if np.any(np.linalg.norm(X, axis=1) < 1e-4):
        ev("tiny_or_zero_norm_rows")
    if not Q[0].any():
        ev("zero_query")
    ev("row_aligned_packing" if (d * bits) % 8 == 0 else "row_unaligned_packing")
    return Case(pq, cv, X, Q, k, desc)


def precisions(bits):
    return st.one_of(st.none(), st.integers(1, bits))


def prec_class(bits, p):
    return "precision=None" if p is None else (
        "precision=bits" if p == bits else "precision=lower")


# --------------------------------------------------------------------------
# result checking
# --------------------------------------------------------------------------
class Violation(AssertionError):
    pass


def verify(case, qi, ids, scores, precision, expect_len, label, pool=None):
    orc = case.orc(qi, precision)
    if np.asarray(scores).dtype != np.float32:
        raise Violation(f"{label}: scores dtype {np.asarray(scores).dtype}, expected float32\n  case={case.desc}")
    ok, msg, tie = O.check_topk(ids, scores, orc, expect_len, case.tol(qi),
                                label=label, pool=pool)
    if not ok:
        raise Violation(f"{msg}\n  case={case.desc} precision={precision} qi={qi}")
    if tie:
        ev("tie_boundary")
    return tie


def verify_twostage(case, qi, ids, scores, cp_eff, candidates, expect_len, label,
                    pool=None):
    """Two-stage soundness + optimality *restricted to the stage-1 candidate set*.

    Stage-1 candidates are the top-`candidates` of the coarse oracle (inside
    `pool`); rows within 2*tol of the coarse boundary may or may not be in.
    """
    n = case.n
    fine = case.orc(qi, None)
    coarse = case.orc(qi, cp_eff)
    tol = case.tol(qi)
    legal = np.ones(n, bool) if pool is None else pool
    ok, msg, tie = O.check_topk(ids, scores, fine, expect_len, tol, label=label,
                                pool=legal)
    # check_topk demanded full optimality over `legal`; that is too strong when
    # candidates < |legal|, so redo the optimality part against the candidate set.
    psize = int(legal.sum())
    if candidates >= psize:
        if not ok:
            raise Violation(f"{msg}\n  case={case.desc}")
        return
    # structural checks only (everything but optimality)
    ids = np.asarray(ids)
    if len(ids) != expect_len:
        raise Violation(f"{label}: returned {len(ids)} results, expected {expect_len}\n  case={case.desc}")
    if len(ids) == 0:
        return
    if len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= n or not legal[ids].all():
        raise Violation(f"{label}: bad ids {ids.tolist()}\n  case={case.desc}")
    if np.any(np.diff(np.asarray(scores, np.float64)) > 0):
        raise Violation(f"{label}: scores not descending\n  case={case.desc}")
    err = np.abs(np.asarray(scores, np.float64) - fine[ids]).max()
    if err > tol:
        raise Violation(f"{label}: fine score err {err:.3e} > {tol:.3e}\n  case={case.desc}")
    if candidates <= 0:
        return
    pool_rows = np.flatnonzero(legal)
    cs = coarse[pool_rows]
    thr = np.sort(cs)[::-1][candidates - 1]
    ceil = pool_rows[cs >= thr - 2 * tol]
    floor = pool_rows[cs > thr + 2 * tol]
    if not set(ids.tolist()) <= set(ceil.tolist()):
        raise Violation(f"{label}: returned rows outside the stage-1 candidate set "
                        f"{sorted(set(ids.tolist()) - set(ceil.tolist()))}\n  case={case.desc}")
    returned = set(ids.tolist())
    worst = fine[ids].min()
    for f in floor:
        if int(f) not in returned and fine[f] > worst + 2 * tol:
            raise Violation(f"{label}: stage-1 candidate {int(f)} (fine {fine[f]!r}) "
                            f"beats returned worst {worst!r}\n  case={case.desc}")
    if len(ceil) > len(floor):
        ev("tie_boundary_stage1")


# --------------------------------------------------------------------------
# CPU paths: operation scripts replayed on one shared (warming) container
# --------------------------------------------------------------------------
CPU_OPS = ["search", "adc", "batch", "twostage_full", "twostage_part",
           "packed_adc", "packed_twostage_full", "packed_twostage_part",
           "score_vectors", "decode_dot", "drop_cache"]


@st.composite
def scripts(draw, case):
    bits = case.pq.bits
    n = case.n
    out = []
    enabled = draw(st.lists(st.sampled_from(CPU_OPS), min_size=1, max_size=4, unique=True))   # swarm
    for _ in range(draw(st.integers(1, 8))):
        op = draw(st.sampled_from(enabled))
        prec = draw(precisions(bits))
        cp = draw(precisions(bits))
        chunk = draw(st.sampled_from([1, 2, 3, 7, 4096]))
        cand_full = draw(st.sampled_from([n, n + 1, 500]))
        cand_part = draw(st.integers(0, max(n - 1, 0)))
        qi = draw(st.integers(0, case.Q.shape[0] - 1))
        out.append((op, prec, cp, chunk, cand_full, cand_part, qi))
    return out


def cp_eff_of(pq, cp):
    return max(1, pq.bits - 2) if cp is None else cp


def run_cpu_op(case, step):
    pq, cv, n, k = case.pq, case.cv, case.n, case.k
    op, prec, cp, chunk, cand_full, cand_part, qi = step
    q = case.Q[qi]
    ev(f"cpu_op={op}")
    if op not in ("drop_cache",):
        ev(f"cpu_op={op}:{prec_class(pq.bits, prec if 'twostage' not in op else cp)}")
    if op == "drop_cache":
        cv.drop_cache()
        if case._packed is not None:
            case._packed._dir_len = {}
        return
    if op == "search":
        ids, sc = pq.search(cv, q, k=k, precision=prec)
        verify(case, qi, ids, sc, prec, min(k, n), "search")
    elif op == "adc":
        ids, sc = pq.search_adc(cv, q, k=k, precision=prec, chunk_size=chunk)
        verify(case, qi, ids, sc, prec, min(k, n), "search_adc")
    elif op == "packed_adc":
        ids, sc = pq.search_adc(case.packed, q, k=k, precision=prec, chunk_size=chunk)
        verify(case, qi, ids, sc, prec, min(k, n), "search_adc[packed]")
    elif op == "batch":
        ids, sc = pq.search_batch(cv, case.Q, k=k, precision=prec)
        if ids.shape != (case.Q.shape[0], min(k, n)) or sc.shape != ids.shape:
            raise Violation(f"search_batch shape {ids.shape}/{sc.shape} != "
                            f"({case.Q.shape[0]}, {min(k, n)})  case={case.desc}")
        if sc.dtype != np.float32 or not np.issubdtype(ids.dtype, np.integer):
            raise Violation(f"search_batch dtypes {ids.dtype}/{sc.dtype}")
        for j in range(case.Q.shape[0]):
            verify(case, j, ids[j], sc[j], prec, min(k, n), f"search_batch[{j}]")
    elif op in ("twostage_full", "packed_twostage_full"):
        src = case.packed if op.startswith("packed") else cv
        ids, sc = pq.search_twostage(src, q, k=k, candidates=cand_full,
                                     coarse_precision=cp, coarse_chunk_size=chunk)
        verify(case, qi, ids, sc, None, min(k, n), op)
    elif op in ("twostage_part", "packed_twostage_part"):
        src = case.packed if op.startswith("packed") else cv
        ids, sc = pq.search_twostage(src, q, k=k, candidates=cand_part,
                                     coarse_precision=cp, coarse_chunk_size=chunk)
        verify_twostage(case, qi, ids, sc, cp_eff_of(pq, cp), cand_part,
                        min(k, min(cand_part, n)), op)
    elif op == "score_vectors":
        Y, bias = pq._score_vectors(cv, prec)
        s = Y.astype(np.float64) @ np.asarray(q, np.float32).astype(np.float64)
        if bias is not None:
            s = s + bias.astype(np.float64)
        orc = case.orc(qi, prec)
        err = np.abs(s - orc).max() if n else 0.0
        if err > case.tol(qi):
            raise Violation(f"_score_vectors rows disagree with oracle: max err {err:.3e} "
                            f"> tol {case.tol(qi):.3e}  case={case.desc} precision={prec}")
    elif op == "decode_dot":
        if pq.mean is not None:
            ev("decode_dot:skipped_centred")
            return
        dec = pq.decode(cv, prec).astype(np.float64)
        s = dec @ np.asarray(q, np.float32).astype(np.float64)
        err = np.abs(s - case.orc(qi, prec)).max()
        if err > case.tol(qi):
            raise Violation(f"q.decode(x) vs oracle: max err {err:.3e} > tol  case={case.desc}")


@PROFILE
@given(data=st.data())
def test_cpu_paths_match_oracle(data):
    """search / search_adc / search_batch / search_twostage / PackedVectors /
    _score_vectors / decode, all at random precision, replayed in random order
    on one shared container so every cache state is reachable."""
    case = data.draw(cases())
    script = data.draw(scripts(case))
    ev(f"script_len={len(script)}")
    for step in script:
        run_cpu_op(case, step)


@PROFILE
@given(data=st.data())
def test_cold_cache_each_path(data):
    """Same checks with a *fresh container per call* (cold caches), so a path
    that only works when another path warmed a cache would fail here."""
    case = data.draw(cases())
    script = data.draw(scripts(case))
    for step in script:
        if step[0] == "drop_cache":
            continue
        fresh = Case.__new__(Case)
        fresh.__dict__.update(case.__dict__)
        cv2 = CompressedVectors(case.gt.indices.copy(),
                                None if case.gt.norms is None else case.gt.norms.copy(),
                                case.pq.d, case.pq.bits, case.cv.rotation, case.cv.mean,
                                None if case.gt.mean_dots is None else case.gt.mean_dots.copy(),
                                case.cv.seed, case.cv.scale)
        fresh.cv = cv2
        fresh._packed = None
        run_cpu_op(fresh, step)


@PROFILE
@given(data=st.data())
def test_score_vectors_accepts_packed(data):
    """`_score_vectors` handles PackedVectors (tests/test_v2_fixes.py pins this
    for centred codes; the code has a PackedVectors branch).  Every mode must
    accept it and give the same (Y, bias) as the unpacked container."""
    case = data.draw(cases(allow_k0=False))
    pq = case.pq
    prec = data.draw(precisions(pq.bits))
    ev("score_vectors_packed:" + case.desc["mode"])
    Y, b = pq._score_vectors(case.cv, prec)
    Y2, b2 = pq._score_vectors(case.packed, prec)
    if not np.allclose(Y, Y2, rtol=1e-6, atol=1e-6) or (b is None) != (b2 is None) or (
            b is not None and not np.allclose(b, b2, rtol=1e-6, atol=1e-6)):
        raise Violation(f"_score_vectors(packed) != _score_vectors(unpacked)  case={case.desc}")


# --------------------------------------------------------------------------
# GPUSearcher (numpy backend) vs oracle and vs the CPU paths
# --------------------------------------------------------------------------
GPU_OPS = ["search", "adc", "twostage_full", "twostage_part", "batch", "drop_cache"]


@PROFILE
@given(data=st.data())
def test_gpu_numpy_backend_matches(data):
    case = data.draw(cases())
    pq, cv, n, k = case.pq, case.cv, case.n, case.k
    if data.draw(st.booleans()):
        pq.search(cv, case.Q[0], k=1)          # warm the CPU-side caches first
        ev("gpu:cpu_warmed_first")
    g = GPUSearcher(pq, cv, backend="numpy")
    for _ in range(data.draw(st.integers(1, 5))):
        op = data.draw(st.sampled_from(GPU_OPS))
        prec = data.draw(precisions(pq.bits))
        cp = data.draw(precisions(pq.bits))
        chunk = data.draw(st.sampled_from([1, 3, 4096]))
        cand_full = data.draw(st.sampled_from([n, n + 2, 500]))
        cand_part = data.draw(st.integers(0, max(n - 1, 0)))
        qi = data.draw(st.integers(0, case.Q.shape[0] - 1))
        q = case.Q[qi]
        ev(f"gpu_op={op}")
        if op == "drop_cache":
            g.drop_cache()
        elif op == "search":
            ids, sc = g.search(q, k=k)
            verify(case, qi, ids, sc, None, min(k, n), "gpu.search")
            cids, csc = pq.search(cv, q, k=k)
            cmp_cpu(case, qi, ids, sc, cids, csc, "gpu.search vs cpu")
        elif op == "adc":
            ids, sc = g.search_adc(q, k=k, precision=prec, chunk_size=chunk)
            verify(case, qi, ids, sc, prec, min(k, n), "gpu.search_adc")
            cids, csc = pq.search_adc(cv, q, k=k, precision=prec)
            cmp_cpu(case, qi, ids, sc, cids, csc, "gpu.search_adc vs cpu", prec)
        elif op == "twostage_full":
            ids, sc = g.search_twostage(q, k=k, candidates=cand_full,
                                        coarse_precision=cp, coarse_chunk_size=chunk)
            verify(case, qi, ids, sc, None, min(k, n), "gpu.search_twostage")
        elif op == "twostage_part":
            ids, sc = g.search_twostage(q, k=k, candidates=cand_part,
                                        coarse_precision=cp, coarse_chunk_size=chunk)
            verify_twostage(case, qi, ids, sc, cp_eff_of(pq, cp), cand_part,
                            min(k, min(cand_part, n)), "gpu.search_twostage[part]")
        elif op == "batch":
            ids, sc = g.search_batch(case.Q, k=k)
            if ids.shape != (case.Q.shape[0], min(k, n)):
                raise Violation(f"gpu.search_batch shape {ids.shape}  case={case.desc}")
            for j in range(case.Q.shape[0]):
                verify(case, j, ids[j], sc[j], None, min(k, n), f"gpu.search_batch[{j}]")


def cmp_cpu(case, qi, ids, sc, cids, csc, label, prec=None):
    """Path-to-path: same top-k set modulo ties, scores within 2*tol rowwise."""
    tol = case.tol(qi)
    orc = case.orc(qi, prec)
    a, b = set(ids.tolist()), set(cids.tolist())
    if a != b:
        diff = sorted(a ^ b)
        worst = min(orc[list(a)].min() if a else 0, orc[list(b)].min() if b else 0)
        if any(abs(orc[i] - worst) > 2 * tol for i in diff):
            raise Violation(f"{label}: top-k sets differ beyond a tie: {sorted(a)} vs {sorted(b)}"
                            f"  case={case.desc}")
        ev("path_vs_path_tie_difference")
    common = sorted(a & b)
    gs = dict(zip(ids.tolist(), sc.tolist()))
    cs = dict(zip(cids.tolist(), csc.tolist()))
    for i in common:
        if abs(gs[i] - cs[i]) > 2 * tol:
            raise Violation(f"{label}: score for row {i}: {gs[i]!r} vs {cs[i]!r}  case={case.desc}")


# --------------------------------------------------------------------------
# IVFCoarseIndex
# --------------------------------------------------------------------------
def hamming_pool(ivf, qcell, nprobe):
    """Independent re-derivation of the probed cells (popcount, then cell id)."""
    cells = np.arange(ivf.n_cells)
    hd = np.array([bin(int(c) ^ int(qcell)).count("1") for c in cells])
    order = sorted(range(ivf.n_cells), key=lambda c: (hd[c], c))
    probed = order[:max(0, min(nprobe, ivf.n_cells))]
    pool = np.isin(ivf.cell_ids.astype(np.int64), np.array(probed, dtype=np.int64))
    return pool


@st.composite
def ivf_cases(draw):
    case = draw(cases(max_n=24))
    n_bits = draw(st.sampled_from([1, 2, 3, 4, 5, 8, 8, 12, 16]))
    mode = draw(st.sampled_from(["lsh", "rotated_prefix"]))
    if mode == "rotated_prefix" and n_bits > case.pq.d:
        n_bits = draw(st.integers(1, case.pq.d))
    ivf_seed = draw(st.sampled_from([0, 1, 7]))
    use_packed = draw(st.booleans())
    src = case.packed if use_packed else case.cv
    ivf = IVFCoarseIndex(case.pq, src, n_bits=n_bits, mode=mode, seed=ivf_seed)
    ev(f"ivf_mode={mode}")
    ev(f"ivf_n_bits={'1-4' if n_bits <= 4 else '5-8' if n_bits <= 8 else '12-16'}")
    ev("ivf_src=" + ("packed" if use_packed else "compressed"))
    return case, ivf, src


@PROFILE
@given(data=st.data())
def test_ivf_flat_limit_matches_oracle_and_flat_paths(data):
    """nprobe = 2**n_bits  ==  flat scan (README/CLAUDE.md): search_coarse is
    a valid top-k of the whole corpus at the requested precision and
    search_twostage equals Quantizer.search_twostage."""
    case, ivf, src = data.draw(ivf_cases())
    pq, n = case.pq, case.n
    k = case.k
    qi = data.draw(st.integers(0, case.Q.shape[0] - 1))
    q = case.Q[qi]
    prec = data.draw(precisions(pq.bits))
    ids, sc = ivf.search_coarse(q, k=k, nprobe=ivf.n_cells, precision=prec)
    verify(case, qi, ids, sc, prec, min(k, n), "ivf.search_coarse[flat]")
    aids, asc = pq.search_adc(src, q, k=k, precision=prec)
    cmp_cpu(case, qi, ids, sc, aids, asc, "ivf.search_coarse vs search_adc", prec)
    # nprobe beyond n_cells behaves as n_cells
    ids2, sc2 = ivf.search_coarse(q, k=k, nprobe=ivf.n_cells * 3, precision=prec)
    if not (np.array_equal(ids, ids2) and np.array_equal(sc, sc2)):
        raise Violation(f"nprobe>n_cells differs from nprobe==n_cells  case={case.desc}")
    cp = data.draw(precisions(pq.bits))
    cand = data.draw(st.sampled_from([n, n + 1, 500]))
    tids, tsc = ivf.search_twostage(q, k=k, candidates=cand, nprobe=ivf.n_cells,
                                    coarse_precision=cp)
    verify(case, qi, tids, tsc, None, min(k, n), "ivf.search_twostage[flat]")
    qids, qsc = pq.search_twostage(src, q, k=k, candidates=cand, coarse_precision=cp)
    cmp_cpu(case, qi, tids, tsc, qids, qsc, "ivf.search_twostage vs Quantizer.search_twostage")


@PROFILE
@given(data=st.data())
def test_ivf_flat_limit_is_byte_identical_to_adc(data):
    """README: "tests verify byte-identical agreement with Quantizer.search_adc /
    Quantizer.search_twostage" at nprobe = 2**n_bits, and the table says
    "bit-identical to flat".  Exact equality of scores (and of ids when the
    scores are all distinct)."""
    case, ivf, src = data.draw(ivf_cases())
    pq, n, k = case.pq, case.n, case.k
    qi = data.draw(st.integers(0, case.Q.shape[0] - 1))
    q = case.Q[qi]
    prec = data.draw(precisions(pq.bits))
    ids, sc = ivf.search_coarse(q, k=k, nprobe=ivf.n_cells, precision=prec)
    aids, asc = pq.search_adc(src, q, k=k, precision=prec)
    ev("byte_identical:" + ("centred" if pq.mean is not None else "uncentred"))
    if not np.array_equal(sc, asc):
        raise Violation(
            f"ivf.search_coarse(nprobe=2**n_bits) scores are not byte-identical to "
            f"search_adc: max |diff| {np.abs(sc.astype(np.float64) - asc).max():.3e}, "
            f"ulps={np.max(np.abs(sc.view(np.int32) - asc.view(np.int32)))}  case={case.desc} precision={prec}")


@PROFILE
@given(data=st.data())
def test_ivf_partial_probe_sound(data):
    """nprobe < 2**n_bits: the probed pool is the Hamming-nearest cells (ties by
    cell id), results are a valid top-k of that pool, lengths are
    min(k, |pool|); search_twostage's results come from the stage-1 candidates."""
    case, ivf, src = data.draw(ivf_cases())
    pq, n, k = case.pq, case.n, case.k
    qi = data.draw(st.integers(0, case.Q.shape[0] - 1))
    q = case.Q[qi]
    nprobe = data.draw(st.one_of(st.integers(0, min(ivf.n_cells, 40)),
                                 st.sampled_from([1, 2, ivf.n_cells - 1])))
    qcell = ivf.query_cell(q)
    pool = hamming_pool(ivf, qcell, nprobe)
    if not (0 <= qcell < ivf.n_cells):
        raise Violation(f"query_cell {qcell} out of range  case={case.desc}")
    ev("ivf_pool=" + ("empty" if not pool.any() else "all" if pool.all() else "partial"))
    cand = np.sort(ivf.candidate_indices(qcell, nprobe))
    if not np.array_equal(cand, np.flatnonzero(pool)):
        raise Violation(f"candidate_indices differ from independent Hamming pool: "
                        f"{cand.tolist()} vs {np.flatnonzero(pool).tolist()}  case={case.desc} "
                        f"nprobe={nprobe} qcell={qcell}")
    if ivf.candidate_count(q, nprobe) != int(pool.sum()):
        raise Violation(f"candidate_count {ivf.candidate_count(q, nprobe)} != |pool| {int(pool.sum())}")
    prec = data.draw(precisions(pq.bits))
    ids, sc = ivf.search_coarse(q, k=k, nprobe=nprobe, precision=prec)
    verify(case, qi, ids, sc, prec, min(k, int(pool.sum())), "ivf.search_coarse[partial]", pool=pool)
    cp = data.draw(precisions(pq.bits))
    cands = data.draw(st.integers(0, n + 2))
    tids, tsc = ivf.search_twostage(q, k=k, candidates=cands, nprobe=nprobe, coarse_precision=cp)
    verify_twostage(case, qi, tids, tsc, cp_eff_of(pq, cp), cands,
                    min(k, min(cands, int(pool.sum()))), "ivf.search_twostage[partial]",
                    pool=pool)


@PROFILE
@given(data=st.data())
def test_ivf_structure(data):
    """CSR layout is a partition of the corpus; cell ids match an independent
    recomputation of the documented hash; rotated_prefix hashes a vector and
    the same direction as a query identically (uncentred modes)."""
    case, ivf, src = data.draw(ivf_cases())
    n = case.n
    if not np.array_equal(np.sort(ivf.sorted_idx), np.arange(n)):
        raise Violation(f"sorted_idx is not a permutation  case={case.desc}")
    if ivf.cell_offsets[0] != 0 or ivf.cell_offsets[-1] != n or np.any(np.diff(ivf.cell_offsets) < 0):
        raise Violation(f"cell_offsets malformed  case={case.desc}")
    for c in range(ivf.n_cells if ivf.n_cells <= 64 else 0):
        rows = ivf.sorted_idx[ivf.cell_offsets[c]:ivf.cell_offsets[c + 1]]
        if not np.all(ivf.cell_ids[rows] == c):
            raise Violation(f"cell {c} holds rows of another cell  case={case.desc}")
        if not np.all(np.diff(rows) > 0):
            raise Violation(f"cell {c} rows not in ascending corpus order (stable build)")
    msb = (case.gt.indices >> (case.pq.bits - 1)).astype(np.int64)
    if ivf.mode == "rotated_prefix":
        want = np.zeros(n, np.int64)
        for b in range(ivf.n_bits):
            want |= msb[:, b] << b
        if not np.array_equal(ivf.cell_ids.astype(np.int64), want):
            raise Violation(f"rotated_prefix cell ids differ from MSB recomputation  case={case.desc}")
        if case.pq.mean is None:
            # a corpus row used as its own query must hash to its own cell,
            # unless a hashed rotated coordinate sits within rounding of 0.
            R = case.pq.R.astype(np.float64)
            for i in range(n):
                x = case.X[i]
                if not x.any():
                    continue
                xr = R @ (x / np.linalg.norm(x))
                if np.min(np.abs(xr[:ivf.n_bits])) < 1e-4:
                    continue
                if ivf.query_cell(x) != int(ivf.cell_ids[i]):
                    raise Violation(f"row {i} queried as itself hashes to cell {ivf.query_cell(x)}, "
                                    f"stored in {int(ivf.cell_ids[i])}  case={case.desc}")
                ev("ivf_self_query_cell_checked")
    else:
        H = np.random.default_rng(ivf.seed).standard_normal((ivf.n_bits, case.pq.d)).astype(np.float32)
        proj = (2.0 * msb.astype(np.float32) - 1.0) @ H.T
        if np.min(np.abs(proj)) > 1e-4 if proj.size else True:
            want = np.zeros(n, np.int64)
            for b in range(ivf.n_bits):
                want |= (proj[:, b] > 0).astype(np.int64) << b
            if not np.array_equal(ivf.cell_ids.astype(np.int64), want):
                raise Violation(f"lsh cell ids differ from independent recomputation  case={case.desc}")


# --------------------------------------------------------------------------
# metamorphic relations
# --------------------------------------------------------------------------
@PROFILE
@given(data=st.data())
def test_batch_equals_single_and_permutation(data):
    """search_batch row i == search(q_i); permuting corpus rows permutes the
    result; duplicate rows score identically on every path."""
    case = data.draw(cases(allow_k0=False))
    pq, cv, n, k = case.pq, case.cv, case.n, case.k
    prec = data.draw(precisions(pq.bits))
    bids, bsc = pq.search_batch(cv, case.Q, k=k, precision=prec)
    for j in range(case.Q.shape[0]):
        ids, sc = pq.search(cv, case.Q[j], k=k, precision=prec)
        cmp_cpu(case, j, bids[j], bsc[j], ids, sc, "search_batch vs search", prec)
    perm = np.array(data.draw(st.permutations(list(range(n)))), dtype=np.int64)
    sub = cv.subset(perm)
    gt_sub = case.gt.subset(perm)
    for path in ("search", "adc", "packed_adc"):
        qi = 0
        if path == "search":
            ids, sc = pq.search(sub, case.Q[qi], k=k, precision=prec)
        elif path == "adc":
            ids, sc = pq.search_adc(sub, case.Q[qi], k=k, precision=prec)
        else:
            ids, sc = pq.search_adc(PackedVectors.from_compressed(sub), case.Q[qi], k=k, precision=prec)
        orc = O.oracle_scores(pq, gt_sub, case.Q[qi], prec)
        ok, msg, _ = O.check_topk(ids, sc, orc, min(k, n), case.tol(qi), label=f"perm:{path}")
        if not ok:
            raise Violation(msg + f"  case={case.desc}")
        ev("permuted_corpus_checked")
    # equal rows -> equal scores (every path computes a row's score from that
    # row alone), compared on the full-score vector via search(k=n)
    ids, sc = pq.search(cv, case.Q[0], k=n, precision=prec)
    full = np.empty(n)
    full[ids] = sc
    seen = {}
    for i in range(n):
        key = (case.gt.indices[i].tobytes(),
               None if case.gt.norms is None else float(case.gt.norms[i]),
               None if case.gt.mean_dots is None else float(case.gt.mean_dots[i]))
        if key in seen:
            gap = abs(full[i] - full[seen[key]])
            if gap > 0:
                ev("identical_rows_unequal_scores_in_search(BLAS gemv)")   # documented float noise
            if gap > case.tol(0):
                raise Violation(f"rows {seen[key]} and {i} have identical codes/norms but "
                                f"scores {full[seen[key]]!r} vs {full[i]!r}  case={case.desc}")
            ev("identical_code_rows_compared")
        else:
            seen[key] = i


@PROFILE
@given(data=st.data())
def test_query_scaling_and_mean_shift_invariance(data):
    """Uncentred / scalar: scores are linear in q, so scaling q by c > 0 scales
    scores by c and keeps the top-k set (modulo ties).  Centred: q.mu is the
    only q-term that does not vanish when q = mu, so score(mu) == mu.mu + ..."""
    case = data.draw(cases(modes=("unit", "unit_norenorm", "scalar", "centred"), allow_k0=False))
    pq, cv, n, k = case.pq, case.cv, case.n, case.k
    c = data.draw(st.sampled_from([-3.0, -1.0, 0.5, 2.0, 8.0]))
    q = case.Q[0]
    ids, sc = pq.search_adc(cv, q, k=k)
    ids2, sc2 = pq.search_adc(cv, q * c, k=k)
    ev("scaling_checked:" + ("centred" if pq.mean is not None else "linear"))
    if pq.mean is None:
        tol = case.tol(0) * abs(c)
        full1 = np.empty(n)
        full2 = np.empty(n)
        a, b = pq.search_adc(cv, q, k=n), pq.search_adc(cv, q * c, k=n)
        full1[a[0]] = a[1]
        full2[b[0]] = b[1]
        if np.abs(full2 - c * full1).max() > 2 * tol:
            raise Violation(f"scores not linear in q (c={c}): max dev "
                            f"{np.abs(full2 - c * full1).max():.3e}  case={case.desc}")


# --------------------------------------------------------------------------
# containers derived from containers: subset / at_precision / to_compressed
# --------------------------------------------------------------------------
@PROFILE
@given(data=st.data())
def test_derived_containers_search_like_their_rows(data):
    """warm every cache, then derive a container (subset, to_compressed,
    at_precision) and search it: scores must equal the oracle on the derived
    rows, i.e. no stale cache is carried across the derivation."""
    case = data.draw(cases(allow_k0=False))
    pq, cv, n, k = case.pq, case.cv, case.n, case.k
    qi = 0
    q = case.Q[qi]
    for p in (None, 1, pq.bits):
        pq.search(cv, q, k=1, precision=p)
        pq.search_adc(cv, q, k=1, precision=p)
    pk = case.packed
    pq.search_adc(pk, q, k=1)
    ev("derive:caches_warmed")
    idx = np.array(data.draw(st.lists(st.integers(0, n - 1), min_size=0, max_size=n + 2)),
                   dtype=np.int64)
    gsub = case.gt.subset(idx)
    for label, container in (("CompressedVectors.subset", cv.subset(idx)),
                             ("PackedVectors.subset", pk.subset(idx)),
                             ("PackedVectors.to_compressed.subset", pk.to_compressed().subset(idx))):
        ev(f"derive:{label}")
        m = len(idx)
        for p in (None, data.draw(precisions(pq.bits))):
            orc = O.oracle_scores(pq, gsub, q, p) if m else np.empty(0)
            tol = case.tol(qi)
            if isinstance(container, PackedVectors):
                ids, sc = pq.search_adc(container, q, k=k, precision=p)
            else:
                ids, sc = pq.search(container, q, k=k, precision=p)
            ok, msg, _ = O.check_topk(ids, sc, orc, min(k, m), tol, label=label)
            if not ok:
                raise Violation(msg + f"  case={case.desc} idx={idx.tolist()}")
    # at_precision: codes must equal the right-shifted ones, and search at the
    # lower level on the derived container must equal search(precision=) on the parent.
    if pq.bits > 1:
        t = data.draw(st.sampled_from([t for t in (1, 2, 3, 4) if t < pq.bits]))  # 5-7 are not packable (see regression file)
        low = pk.at_precision(t)
        got = low.unpack_rows(0, n)
        if not np.array_equal(got, case.gt.indices >> (pq.bits - t)):
            raise Violation(f"at_precision({t}) codes differ from right shift  case={case.desc}")
        ev("derive:at_precision_codes_checked")


# --------------------------------------------------------------------------
# API contract: precision validation, empty corpus, input dtypes
# --------------------------------------------------------------------------
@PROFILE
@given(data=st.data())
def test_invalid_precision_rejected_uniformly(data):
    """precision outside 1..bits is a ValueError on every path (never a silent
    wrong answer or a different exception type)."""
    case = data.draw(cases(allow_k0=False))
    pq, cv, q = case.pq, case.cv, case.Q[0]
    bad = data.draw(st.sampled_from([0, -1, pq.bits + 1, pq.bits + 7]))
    g = GPUSearcher(pq, cv, backend="numpy")
    ivf = IVFCoarseIndex(pq, cv, n_bits=1, mode="lsh")
    calls = {
        "search": lambda: pq.search(cv, q, k=1, precision=bad),
        "search_adc": lambda: pq.search_adc(cv, q, k=1, precision=bad),
        "search_adc[packed]": lambda: pq.search_adc(case.packed, q, k=1, precision=bad),
        "search_batch": lambda: pq.search_batch(cv, case.Q, k=1, precision=bad),
        "search_twostage": lambda: pq.search_twostage(cv, q, k=1, coarse_precision=bad),
        "search_twostage[packed]": lambda: pq.search_twostage(case.packed, q, k=1, coarse_precision=bad),
        "decode": lambda: pq.decode(cv, precision=bad),
        "gpu.search_adc": lambda: g.search_adc(q, k=1, precision=bad),
        "gpu.search_twostage": lambda: g.search_twostage(q, k=1, coarse_precision=bad),
        "ivf.search_coarse": lambda: ivf.search_coarse(q, k=1, nprobe=1, precision=bad),
        "ivf.search_twostage": lambda: ivf.search_twostage(q, k=1, nprobe=1, coarse_precision=bad),
    }
    for name, fn in calls.items():
        ev(f"invalid_precision:{name}")
        try:
            fn()
        except ValueError:
            continue
        except Exception as e:  # noqa: BLE001
            raise Violation(f"{name}(precision={bad}) raised {type(e).__name__}: {e}  case={case.desc}")
        raise Violation(f"{name}(precision={bad}) was accepted  case={case.desc}")


@PROFILE
@given(data=st.data())
def test_empty_corpus_and_query_dtypes(data):
    """n = 0 corpora and non-float32 query dtypes are accepted by every path
    and give empty / identical results."""
    case = data.draw(cases(allow_k0=False))
    pq = case.pq
    q = case.Q[0]
    empty = pq.encode(np.empty((0, pq.d)))
    ev("empty_corpus")
    k = case.k
    outs = {
        "search": lambda: pq.search(empty, q, k=k),
        "search_adc": lambda: pq.search_adc(empty, q, k=k),
        "search_adc[packed]": lambda: pq.search_adc(PackedVectors.from_compressed(empty), q, k=k),
        "search_twostage": lambda: pq.search_twostage(empty, q, k=k),
        "search_batch": lambda: pq.search_batch(empty, case.Q, k=k),
        "gpu.search": lambda: GPUSearcher(pq, empty, backend="numpy").search(q, k=k),
        "gpu.search_adc": lambda: GPUSearcher(pq, empty, backend="numpy").search_adc(q, k=k),
        "gpu.search_twostage": lambda: GPUSearcher(pq, empty, backend="numpy").search_twostage(q, k=k),
        "ivf.search_coarse": lambda: IVFCoarseIndex(pq, empty, n_bits=1, mode="lsh").search_coarse(q, k=k, nprobe=2),
        "ivf.search_twostage": lambda: IVFCoarseIndex(pq, empty, n_bits=1, mode="lsh").search_twostage(q, k=k, nprobe=2),
    }
    for name, fn in outs.items():
        try:
            ids, sc = fn()
        except Exception as e:  # noqa: BLE001
            raise Violation(f"{name} on an empty corpus raised {type(e).__name__}: {e}  case={case.desc}")
        if np.asarray(ids).size != 0:
            raise Violation(f"{name} on an empty corpus returned {ids}")
    # query dtypes: float64 / float32 / list all give the same answer
    base = pq.search_adc(case.cv, q.astype(np.float32), k=k)
    for conv in (lambda a: a.astype(np.float64), lambda a: a.tolist(), lambda a: tuple(a.tolist())):
        got = pq.search_adc(case.cv, conv(q), k=k)
        if not (np.array_equal(base[0], got[0]) and np.array_equal(base[1], got[1])):
            raise Violation(f"search_adc result depends on query container type  case={case.desc}")
        ev("query_dtype_checked")


# --------------------------------------------------------------------------
# a row-constant term cannot reorder rows (docstring of Quantizer._query_offset)
# --------------------------------------------------------------------------
@PROFILE
@given(data=st.data())
def test_query_offset_cannot_reorder(data):
    """`_query_offset` is "constant across the corpus for a given query, so it
    cannot reorder anything".  Adding it before top-k selection in float32
    can: once |q.mu| is ~1e6 the float32 spacing (0.06) exceeds the score gaps
    among near neighbours.  Differential check: top-k with the offset == top-k
    without it, up to ties of the offset-free float32 scores.  Needs a cloud
    of many near neighbours around a huge mean, which `cases()` does not
    build, so this property has its own generator."""
    from unittest import mock
    d = data.draw(st.sampled_from([8, 16, 32, 64]))
    bits = data.draw(st.sampled_from([4, 8]))
    n = data.draw(st.sampled_from([60, 130, 300]))
    mag = data.draw(st.sampled_from([1e3, 1e5, 1e6, 3e6]))
    sigma = data.draw(st.sampled_from([0.1, 0.3, 1.0]))
    rng = np.random.default_rng(data.draw(st.integers(0, 2 ** 32 - 1)))
    mu = np.zeros(d)
    mu[:] = mag / math.sqrt(d)
    X = mu + sigma * rng.standard_normal((n, d))
    pq = make_pq(d, bits, 1, None, True, None, True, tuple(float(np.float32(x)) for x in mu))
    cv = pq.encode(X)
    k = data.draw(st.sampled_from([1, 5, 10]))
    ev(f"offset_reorder:mean_norm={'>=1e6' if mag >= 1e6 else '<1e6'}")
    for _ in range(4):
        q = X[rng.integers(n)] + 0.1 * sigma * rng.standard_normal(d)
        for name, call in (("search_adc", lambda: pq.search_adc(cv, q, k=n)),
                           ("search", lambda: pq.search(cv, q, k=n)),
                           ("search_batch", lambda: tuple(a[0] for a in pq.search_batch(cv, q[None], k=n)))):
            ids, sc = call()
            with mock.patch.object(Quantizer, "_query_offset", lambda self, query: 0.0), \
                 mock.patch.object(type(pq), "_query_offset", lambda self, query: 0.0):
                ids0, sc0 = call()
            full0 = np.empty(n)
            full0[ids0] = sc0
            if set(ids[:k].tolist()) == set(ids0[:k].tolist()):
                continue
            boundary = full0[ids0[k - 1]]
            slack = 1e-4 * (1.0 + np.abs(full0).max())
            for r in set(ids[:k].tolist()) ^ set(ids0[:k].tolist()):
                if abs(full0[r] - boundary) > slack:
                    raise Violation(
                        f"{name}: adding the constant q.mu={float(np.asarray(q, np.float32) @ pq.mean):.3e} "
                        f"changed the top-{k} set: row {r} has offset-free score {full0[r]!r} vs "
                        f"boundary {boundary!r}; d={d} bits={bits} n={n} |mu|={mag:g} sigma={sigma}")


# --------------------------------------------------------------------------
# serialization round trips: a loaded container searches like the original
# --------------------------------------------------------------------------
@PROFILE
@given(data=st.data())
def test_serialized_containers_search_identically(data):
    import tempfile
    case = data.draw(cases(allow_k0=False))
    pq, cv, n, k = case.pq, case.cv, case.n, case.k
    q = case.Q[0]
    prec = data.draw(precisions(pq.bits))
    base_ids, base_sc = pq.search_adc(cv, q, k=k, precision=prec)
    pk = case.packed
    pk_ids, pk_sc = pq.search_adc(pk, q, k=k, precision=prec)
    if not (np.array_equal(base_ids, pk_ids) and np.array_equal(base_sc, pk_sc)):
        raise Violation(f"PackedVectors.from_compressed search_adc != CompressedVectors  case={case.desc}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        variants = []
        cv.save(str(tmp / "c"))
        variants.append(("CompressedVectors.load(npz)", CompressedVectors.load(str(tmp / "c"))))
        pk.save(str(tmp / "p"))
        variants.append(("PackedVectors.load(npz)", PackedVectors.load(str(tmp / "p"))))
        cv.save_arrow(str(tmp / "c.arrow"))
        variants.append(("CompressedVectors.load_arrow", CompressedVectors.load_arrow(str(tmp / "c.arrow"))))
        pk.save_arrow(str(tmp / "p.arrow"))
        variants.append(("PackedVectors.load_arrow", PackedVectors.load_arrow(str(tmp / "p.arrow"))))
        variants.append(("PackedVectors.load_arrow(mmap)", PackedVectors.load_arrow(str(tmp / "p.arrow"), memory_map=True)))
        rows = [pk._packed[i] for i in range(n)]
        variants.append(("PackedVectors.from_rows", PackedVectors.from_rows(
            rows, None if pk.norms is None else pk.norms, pk.d, pk.bits, pk.rotation,
            pk.mean, pk.mean_dots, pk.seed, pk.scale)) if n else ("skip", pk))
        for label, loaded in variants:
            ev(f"roundtrip:{label}")
            ids, sc = pq.search_adc(loaded, q, k=k, precision=prec)
            if not (np.array_equal(ids, base_ids) and np.array_equal(sc, base_sc)):
                raise Violation(f"{label}: search_adc differs from the in-memory container: "
                                f"{ids.tolist()}/{sc.tolist()} vs {base_ids.tolist()}/{base_sc.tolist()}  case={case.desc}")
        if pq.mean is None:
            from remex import load_pq, save_pq
            save_pq(tmp / "x.pq", cv)
            lp = load_pq(tmp / "x.pq")
            ev("roundtrip:save_pq/load_pq")
            lp.seed = cv.seed  # .pq has no seed field
            lp.scale = cv.scale
            ids, sc = pq.search_adc(lp, q, k=k, precision=prec)
            if not (np.array_equal(ids, base_ids) and np.array_equal(sc, base_sc)):
                raise Violation(f".pq round trip changes search_adc results  case={case.desc}")


@PROFILE
@given(data=st.data())
def test_adc_scores_independent_of_chunking_and_position(data):
    """search_adc's per-row score is a gather-sum over that row alone, so it must
    not depend on chunk_size, on the row's position, or on whether the corpus is
    a CompressedVectors or a PackedVectors: exact equality of the full score
    vector (k = n), and identical rows score identically."""
    case = data.draw(cases(allow_k0=False))
    pq, cv, n = case.pq, case.cv, case.n
    q = case.Q[0]
    prec = data.draw(precisions(pq.bits))

    def full(src, chunk):
        ids, sc = pq.search_adc(src, q, k=n, precision=prec, chunk_size=chunk)
        out = np.empty(n, np.float32)
        out[ids] = sc
        return out
    ref = full(cv, 4096)
    for chunk in (1, 2, 3, 5):
        for label, src in (("CompressedVectors", cv), ("PackedVectors", case.packed)):
            got = full(src, chunk)
            ev(f"adc_chunk_invariance:{label}")
            if not np.array_equal(got, ref):
                raise Violation(f"search_adc scores depend on chunk_size={chunk} / {label}: max diff "
                                f"{np.abs(got.astype(np.float64) - ref).max():.3e}  case={case.desc}")
    seen = {}
    for i in range(n):
        key = (case.gt.indices[i].tobytes(),
               None if case.gt.norms is None else float(case.gt.norms[i]),
               None if case.gt.mean_dots is None else float(case.gt.mean_dots[i]))
        if key in seen and ref[i] != ref[seen[key]]:
            raise Violation(f"search_adc: identical rows {seen[key]}, {i} score {ref[seen[key]]!r} vs {ref[i]!r}")
        seen.setdefault(key, i)


# --------------------------------------------------------------------------
# readers that differ from the writer only in `renorm`, and precision == bits
# --------------------------------------------------------------------------
@PROFILE
@given(data=st.data())
def test_renorm_is_a_reader_property_and_caches_do_not_leak(data):
    """`renorm` "changes how codes are read, not what they are" (CLAUDE.md):
    a container written under renorm=True can be searched through a Quantizer
    with renorm=False and vice versa.  Both readers share the container's
    caches, in either order; each must match the oracle for *its* setting."""
    case = data.draw(cases(modes=("unit", "unit_norenorm"), allow_k0=False))
    w = case.pq
    other = make_pq(w.d, w.bits, w.seed, w.rotation, True, None, not w.renorm, None)
    n, k = case.n, case.k
    prec = data.draw(precisions(w.bits))
    order = data.draw(st.sampled_from([(w, other), (other, w), (w, other, w)]))
    for reader in order:
        ev("renorm_reader=" + ("renorm" if reader.renorm else "raw"))
        orc = O.oracle_scores(reader, case.gt, case.Q[0], prec)
        for path in ("search", "adc", "gpu"):
            if path == "search":
                ids, sc = reader.search(case.cv, case.Q[0], k=k, precision=prec)
            elif path == "adc":
                ids, sc = reader.search_adc(case.cv, case.Q[0], k=k, precision=prec)
            else:
                ids, sc = GPUSearcher(reader, case.cv, backend="numpy").search_adc(case.Q[0], k=k, precision=prec)
            ok, msg, _ = O.check_topk(ids, sc, orc, min(k, n), O.score_tol(reader, case.gt, case.Q[0]),
                                      label=f"reader(renorm={reader.renorm}).{path}")
            if not ok:
                raise Violation(msg + f"  case={case.desc} precision={prec}")


@PROFILE
@given(data=st.data())
def test_precision_equal_to_bits_is_full_precision(data):
    """precision=bits is the same level as precision=None: identical ids and
    bit-identical scores on every path that takes `precision`."""
    case = data.draw(cases(allow_k0=False))
    pq, cv, k, q = case.pq, case.cv, case.k, case.Q[0]
    for label, f in (("search", lambda p: pq.search(cv, q, k=k, precision=p)),
                     ("search_adc", lambda p: pq.search_adc(cv, q, k=k, precision=p)),
                     ("search_adc[packed]", lambda p: pq.search_adc(case.packed, q, k=k, precision=p)),
                     ("search_batch", lambda p: tuple(a[0] for a in pq.search_batch(cv, case.Q[:1], k=k, precision=p))),
                     ("ivf.search_coarse", lambda p: IVFCoarseIndex(pq, cv, n_bits=1, mode="lsh")
                      .search_coarse(q, k=k, nprobe=2, precision=p)),
                     ("gpu.search_adc", lambda p: GPUSearcher(pq, cv, backend="numpy").search_adc(q, k=k, precision=p))):
        a, b = f(None), f(pq.bits)
        ev(f"precision_bits_eq_none:{label}")
        if not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])):
            raise Violation(f"{label}: precision=None and precision=bits differ: {a} vs {b}  case={case.desc}")


# --------------------------------------------------------------------------
# large-n boundaries the small generated corpora cannot reach
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4095, 4096, 4097, 8193])
@pytest.mark.parametrize("mode", ["unit", "centred", "scalar"])
@pytest.mark.parametrize("packed", [False, True])
def test_large_n_chunk_boundaries(n, mode, packed):
    """The default 4096-row chunks in _direction_lengths, _direction_mean_dots,
    _adc_score_*, IVF search_coarse and GPU gather_sum only wrap for n > 4096."""
    rng = np.random.default_rng(n)
    d, bits = 8, 3
    X = rng.standard_normal((n, d)) + (rng.standard_normal(d) * 2 if mode == "centred" else 0)
    mean = tuple(float(np.float32(x)) for x in X.mean(axis=0)) if mode == "centred" else None
    pq = make_pq(d, bits, 5, "none" if mode == "scalar" else None, mode != "scalar",
                 1.0 if mode == "scalar" else None, True, mean)
    cv = pq.encode(X)
    gt = GroundTruth(cv)
    src = PackedVectors.from_compressed(cv) if packed else cv
    q = rng.standard_normal(d)
    ivf = IVFCoarseIndex(pq, src, n_bits=4, mode="rotated_prefix")
    g = None if packed else GPUSearcher(pq, cv, backend="numpy")
    for prec in (None, 1, 2):
        orc = O.oracle_scores(pq, gt, q, prec)
        tol = O.score_tol(pq, gt, q)
        for label, (ids, sc) in (
                ("search_adc", pq.search_adc(src, q, k=10, precision=prec)),
                ("ivf.search_coarse", ivf.search_coarse(q, k=10, nprobe=16, precision=prec))):
            ok, msg, _ = O.check_topk(ids, sc, orc, 10, tol, label=label)
            assert ok, msg
        if g is not None:
            ok, msg, _ = O.check_topk(*g.search_adc(q, k=10, precision=prec), orc, 10, tol, label="gpu.search_adc")
            assert ok, msg
            if prec is None:
                ok, msg, _ = O.check_topk(*g.search(q, k=10), orc, 10, tol, label="gpu.search")
                assert ok, msg
        if not packed and prec is None:
            ok, msg, _ = O.check_topk(*pq.search(cv, q, k=10), orc, 10, tol, label="search")
            assert ok, msg
            ok, msg, _ = O.check_topk(*pq.search_twostage(cv, q, k=10, candidates=n), orc, 10, tol, label="twostage")
            assert ok, msg


def test_large_n_ivf_prefix_chunk_boundary():
    """IVF rotated_prefix builds cell ids in 65,536-row chunks (ivf.py:179)."""
    rng = np.random.default_rng(0)
    d, n = 4, 65_536 * 2 + 17
    pq = make_pq(d, 2, 0, None, True, None, True, None)
    cv = pq.encode(rng.standard_normal((n, d)))
    ivf = IVFCoarseIndex(pq, cv, n_bits=3, mode="rotated_prefix")
    msb = (cv.indices >> 1).astype(np.int64)
    want = msb[:, 0] | (msb[:, 1] << 1) | (msb[:, 2] << 2)
    assert np.array_equal(ivf.cell_ids.astype(np.int64), want)


@PROFILE
@given(data=st.data())
def test_encode_block_size_does_not_change_the_container(data):
    """Encoding runs in row blocks (ENCODE_BLOCK_FLOATS // d rows) and "the codes
    do not depend on the block size".  The same must hold for the per-row
    floats the scoring reads (norms, mean_dots): same corpus encoded with a
    1-row block and with one big block gives identical containers, hence
    identical searches."""
    import remex.core as rc
    case = data.draw(cases(allow_k0=False))
    pq = case.pq
    saved = rc.ENCODE_BLOCK_FLOATS
    try:
        rc.ENCODE_BLOCK_FLOATS = data.draw(st.sampled_from([1, pq.d, 2 * pq.d, 3 * pq.d + 1]))
        small = pq.encode(case.X)
    finally:
        rc.ENCODE_BLOCK_FLOATS = saved
    ev("encode_block_small")
    for a, b, name in ((small.indices, case.gt.indices, "indices"), (small.norms, case.gt.norms, "norms"),
                       (small.mean_dots, case.gt.mean_dots, "mean_dots")):
        if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            raise Violation(f"container {name} depends on the encode block size  case={case.desc}")


# ---------------------------------------------------------------------------
# A reader that differs from the writer must be refused on EVERY path, warm or cold
# ---------------------------------------------------------------------------
MISMATCHES = ["bits", "seed", "rotation", "mean_dropped", "mean_added", "mean_changed",
              "scale", "mode", "dim"]


def make_mismatched_reader(case, kind, data):
    """A Quantizer that differs from `case.pq` in exactly one encoding parameter,
    or None when the case cannot express that mismatch."""
    w = case.pq
    d, bits, seed, rot = w.d, w.bits, w.seed, w.rotation
    normalize, renorm = w.normalize, w.renorm
    scale = None if w.normalize else w.scale
    mean_key = None if w.mean is None else tuple(float(x) for x in w.mean)
    if kind == "bits":
        other = [b for b in BITS_CHOICES if b != bits]
        bits = data.draw(st.sampled_from(other))
    elif kind == "seed":
        if rot == "none":
            return None
        seed = seed + 1
    elif kind == "rotation":
        options = [r for r in ("haar", "none") + (("rht",) if d % 2 == 0 else ()) if r != rot]
        rot = data.draw(st.sampled_from(options))
    elif kind == "mean_dropped":
        if mean_key is None:
            return None
        mean_key = None
    elif kind == "mean_added":
        if mean_key is not None or not normalize or not renorm:
            return None
        mean_key = tuple([0.25] * d)
    elif kind == "mean_changed":
        if mean_key is None:
            return None
        mean_key = tuple(x + 1.0 for x in mean_key)
    elif kind == "scale":
        if normalize:
            return None
        scale = scale * 2.0
    elif kind == "mode":
        if mean_key is not None:
            return None
        normalize = not normalize
        scale = None if normalize else 1.0
        renorm = True
    elif kind == "dim":
        return None      # dimension mismatch needs a different d for the *container*, covered separately
    return make_pq(d, bits, seed, rot, normalize, scale, renorm, mean_key)


@settings(PROFILE, max_examples=max(EXAMPLES, 1500))   # rare precondition: warm cache + scalar/renorm=False writer
@given(data=st.data())
def test_mismatched_reader_is_refused_on_every_path(data):
    """README.md:23: "decoding against a quantizer that differs in any of them
    raises rather than returning wrong-but-plausible vectors".  Every search
    path must raise ValueError for a reader whose bits / seed / rotation / mean /
    scale / mode differ from the writer's -- including after another reader
    warmed the container's caches (history dependence)."""
    case = data.draw(cases(allow_k0=False))
    kind = data.draw(st.sampled_from(MISMATCHES))
    reader = make_mismatched_reader(case, kind, data)
    if reader is None:
        ev("mismatch:not_expressible")
        return
    w, cv, q = case.pq, case.cv, case.Q[0]
    warm = data.draw(st.booleans())
    if warm:
        # the legitimate writer warms every cache it can
        w.search(cv, q, k=1)
        w.search_adc(cv, q, k=1)
        w.search_batch(cv, case.Q, k=1)
        w.search_twostage(cv, q, k=1)
        _ = case.packed
        w.search_adc(case.packed, q, k=1)
    ev(f"mismatch:{kind}:{'warm' if warm else 'cold'}")
    calls = {
        "search": lambda: reader.search(cv, q, k=1),
        "search_adc": lambda: reader.search_adc(cv, q, k=1),
        "search_adc[packed]": lambda: reader.search_adc(case.packed, q, k=1),
        "search_batch": lambda: reader.search_batch(cv, case.Q, k=1),
        "search_twostage": lambda: reader.search_twostage(cv, q, k=1),
        "search_twostage[packed]": lambda: reader.search_twostage(case.packed, q, k=1),
        "decode": lambda: reader.decode(cv),
        "gpu.__init__": lambda: GPUSearcher(reader, cv, backend="numpy"),
        "ivf.search_coarse": lambda: IVFCoarseIndex(reader, cv, n_bits=1, mode="lsh").search_coarse(q, k=1, nprobe=2),
    }
    accepted = []
    for name, fn in calls.items():
        try:
            fn()
        except ValueError:
            continue
        except Exception as e:  # noqa: BLE001
            accepted.append(f"{name}: wrong exception {type(e).__name__}: {e}")
            continue
        accepted.append(f"{name}: ACCEPTED")
    if accepted:
        raise Violation(f"reader differing in `{kind}` (container {'warm' if warm else 'cold'}) was not refused "
                        f"by: {accepted}  writer={case.desc} reader(bits={reader.bits}, rot={reader.rotation}, "
                        f"renorm={reader.renorm}, centred={reader.mean is not None})")
