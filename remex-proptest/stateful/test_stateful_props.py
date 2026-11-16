"""Stateful / model-based property tests for remex (area C: indexes and caches).

Three Hypothesis RuleBasedStateMachines, plus a few plain properties:

1. ``RemexIndexMachine``     remex.st.RemexIndex driven by tests/_fake_st.py.
   Reference model: the document list re-encoded from scratch by a separate
   Quantizer and scored by an independent float64 brute force (see _oracle.py).
2. ``CacheMachine``          Quantizer / CompressedVectors / PackedVectors caches
   (_x_hat_rot, _dir_len per precision, drop_cache, subset, save/load, the
   Quantizer.R setter, the lazily built rht R, second quantizers with equal
   parameters).  Every answer is compared to a cache-free float64 recomputation
   from the container's raw arrays, and repeated questions must be bit-identical.
3. ``GpuIvfMachine``         GPUSearcher (numpy backend) and IVFCoarseIndex
   interleaved with CPU searches and drop_cache.

Machines named ``*KnownBug`` reproduce findings that are present in the remex
revision under test; they are ``xfail(strict=True)`` so they flip red the day the
bug is fixed.  See test_stateful_regressions.py for the deterministic repros.

Environment knobs:  REMEX_PROP_EXAMPLES (default 60), REMEX_PROP_STEPS (default 25),
REMEX_PROP_MUTANTS=1 enables the mutation-calibration tests.
Coverage counters are written to results/coverage_events.json at module teardown.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase
from hypothesis.stateful import (RuleBasedStateMachine, initialize, invariant,
                                 precondition, rule, run_state_machine_as_test)

import remex
import remex.st as remex_st
from remex import (AnisotropyWarning, CompressedVectors, PackedVectors, Quantizer,
                   RemexIndex, corpus_mean)
from remex.gpu import GPUSearcher
from remex.ivf import IVFCoarseIndex

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(remex.__file__).resolve().parents[1] / "tests"))
from _fake_st import FakeModel, FakeMultiVectorModel, FakeTokenPooling  # noqa: E402
from _oracle import EV, Oracle, check_topk, check_twostage, codes_of, tolerance  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EXAMPLES = int(os.environ.get("REMEX_PROP_EXAMPLES", "60"))
STEPS = int(os.environ.get("REMEX_PROP_STEPS", "25"))
MUTANTS = os.environ.get("REMEX_PROP_MUTANTS") == "1"
NOSHRINK = os.environ.get("REMEX_PROP_NOSHRINK") == "1"

_TMP_ROOT: Optional[Path] = None
_ids = itertools.count()


def machine_settings(**kw) -> settings:
    base = dict(
        max_examples=EXAMPLES, stateful_step_count=STEPS, deadline=None,
        database=DirectoryBasedExampleDatabase(HERE / ".hypothesis-examples"),
        suppress_health_check=list(HealthCheck), print_blob=True,
    )
    if NOSHRINK:
        from hypothesis import Phase
        base["phases"] = [Phase.explicit, Phase.reuse, Phase.generate]
    base.update(kw)
    return settings(**base)


@pytest.fixture(scope="module", autouse=True)
def _module_env(tmp_path_factory):
    """All files the machines write live under pytest's tmp_path machinery; the
    AnisotropyWarning RemexIndex.add emits on shifted corpora is noise here."""
    global _TMP_ROOT
    _TMP_ROOT = tmp_path_factory.mktemp("stateful")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AnisotropyWarning)
        yield
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "coverage_events.json"
    prior = json.loads(out.read_text()) if out.exists() and os.environ.get("REMEX_PROP_APPEND") else {}
    for k, v in EV.counts.items():
        prior[k] = prior.get(k, 0) + v
    out.write_text(json.dumps(dict(sorted(prior.items())), indent=1))


_OWNERS: dict = {}


def _cached_quantizer(params: dict) -> Quantizer:
    """Foreign-encoding owners are read-only apart from the containers they encode, so
    they are built once per parameter set (a Quantizer costs ~50 ms to construct)."""
    from _oracle import freeze
    key = freeze(params)
    if key not in _OWNERS:
        _OWNERS[key] = Quantizer(**params)
        if len(_OWNERS) > 200:
            _OWNERS.pop(next(iter(_OWNERS)))
    return _OWNERS[key]


def _tmpdir() -> Path:
    d = Path(tempfile.mkdtemp(dir=_TMP_ROOT, prefix=f"m{next(_ids)}_"))
    return d


# --------------------------------------------------------------------------
# corpus / query generation (deterministic from integer seeds, shrinks cleanly)
# --------------------------------------------------------------------------

def make_corpus(kind: str, n: int, d: int, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    if kind == "dups":
        for i in range(1, n):
            if rng.random() < 0.4:
                X[i] = X[rng.integers(0, i)]
    elif kind == "zeros":
        X[rng.random(n) < 0.25] = 0.0
    elif kind == "shifted":
        X += 3.0
    elif kind == "scaled":
        X *= (10.0 ** rng.uniform(-3, 3, (n, 1))).astype(np.float32)
    return (X * np.float32(scale)).astype(np.float32)


def make_queries(X: np.ndarray, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919)
    n, d = X.shape
    g = rng.standard_normal((2, d)).astype(np.float32) * scale
    rows = [X[rng.integers(0, n)].copy(),
            X[rng.integers(0, n)] + 0.3 * rng.standard_normal(d).astype(np.float32) * scale,
            g[0], g[1], np.zeros(d, np.float32), 50.0 * g[0]]
    return np.stack(rows).astype(np.float32)


CONFIG = st.fixed_dictionaries({
    "d": st.sampled_from([4, 6, 8, 9, 16, 24]),
    "bits": st.sampled_from([1, 2, 3, 4, 8]),
    "mode": st.sampled_from(["plain", "plain", "centered", "scalar"]),
    "rotation": st.sampled_from([None, None, "haar", "none", "rht"]),
    "renorm": st.booleans(),
    "seed": st.sampled_from([0, 1, 42]),
    "n": st.integers(1, 24),
    "data_seed": st.integers(0, 2 ** 31 - 1),
    "kind": st.sampled_from(["gauss", "dups", "zeros", "shifted", "scaled"]),
    "scale": st.sampled_from([None, 0.5, 2.0]),
    "mean_kind": st.sampled_from(["corpus", "zero", "random", "row0"]),
})


def resolve_config(cfg: dict, force_renorm_false: bool = False) -> dict:
    """Turn a drawn config into Quantizer kwargs + corpus, fixing illegal combos."""
    d, mode = cfg["d"], cfg["mode"]
    rot = cfg["rotation"]
    if rot == "rht" and d % 2:
        rot = None
    if force_renorm_false and mode == "centered":
        mode = "plain"
    params = dict(d=d, bits=cfg["bits"], seed=cfg["seed"], rotation=rot)
    scalar = mode == "scalar"
    scale = cfg["scale"] if scalar else None
    kind = cfg["kind"]
    X = make_corpus(kind, cfg["n"], d, cfg["data_seed"], scale=(scale or 1.0) if scalar else 1.0)
    if scalar:
        params.update(normalize=False)
        if scale is not None:
            params.update(scale=scale)
        mean = None
    elif mode == "centered":
        rng = np.random.default_rng(cfg["data_seed"] + 1)
        mk = cfg["mean_kind"]
        mean = {"corpus": lambda: corpus_mean(X), "zero": lambda: np.zeros(d, np.float32),
                "random": lambda: (2.0 * rng.standard_normal(d)).astype(np.float32),
                "row0": lambda: X[0].copy()}[mk]()
        params.update(mean=mean)
    else:
        mean = None
    if mode != "centered" and (force_renorm_false or not cfg["renorm"]) and not scalar:
        params.update(renorm=False)
    Q = make_queries(X, cfg["data_seed"], scale=(scale or 1.0) if scalar else 1.0)
    return dict(params=params, X=X, Q=Q, mode=mode, scalar=scalar, scale=scale)


# --------------------------------------------------------------------------
# Machine 2: Quantizer / container caches
# --------------------------------------------------------------------------

@dataclass
class CRec:
    c: object
    label: str
    owner: int                      # index of the quantizer that encoded it
    packed: bool = False
    rtag: Optional[int] = None      # replaced-R tag of the encoding quantizer
    fresh_ok: bool = True           # codes should equal a fresh encode of X[rows]
    rows: Optional[np.ndarray] = None
    last_p: object = "unset"
    dropped_since_search: bool = False
    stamp: int = 0


class CacheMachine(RuleBasedStateMachine):
    """Interleave searches at different precisions with drop_cache, subset,
    save/load, packed conversion, second quantizers and R assignment; every
    answer must equal a cache-free recomputation from the raw codes."""

    FORCE_RENORM_FALSE = False
    GUARD_KINDS = ("bits", "mean", "seed", "rotation", "mode", "d")
    GUARD_WARM = None            # None: draw; True/False: fixed

    def __init__(self):
        super().__init__()
        self.tmp = _tmpdir()
        self.cs: list[CRec] = []
        self.qs: list[Quantizer] = []
        self.qmeta: list[dict] = []
        self.memo: dict = {}
        self.R_by_tag: dict = {}
        self.foreign: dict = {}
        self.ready = False

    # -- setup ---------------------------------------------------------------
    @initialize(cfg=CONFIG)
    def setup(self, cfg):
        r = resolve_config(cfg, self.FORCE_RENORM_FALSE)
        self.params, self.X, self.Q = r["params"], r["X"], r["Q"]
        self.mode, self.scalar = r["mode"], r["scalar"]
        self.d, self.bits = self.params["d"], self.params["bits"]
        self.rotation = self.params["rotation"]
        self.oracle = Oracle(self.params)
        self.qs = [Quantizer(**self.params)]
        self.qmeta = [dict(dense=False, rtag=None)]
        self.cs = [CRec(self.qs[0].encode(self.X), "c0", 0, rows=np.arange(len(self.X)))]
        self.ready = True
        EV(f"cache:mode={self.mode}")
        EV(f"cache:bits={self.bits}")
        EV(f"cache:rotation={self.oracle.pq.rotation}")
        EV(f"cache:renorm={self.oracle.pq.renorm}")
        EV(f"cache:corpus_kind={cfg['kind']}")

    # -- helpers -------------------------------------------------------------
    def _precision(self, code: int):
        """0 -> None; else 1..bits (bits itself included: the explicit full width)."""
        if code == 0:
            return None
        return (code - 1) % self.bits + 1

    def _eligible(self, qi, packed_ok):
        """Containers this quantizer may read, most recently used first (so a
        container warmed a moment ago is the likely next target)."""
        tag = self.qmeta[qi]["rtag"]
        els = [r for r in self.cs if r.rtag == tag and (packed_ok or not r.packed)]
        return sorted(els, key=lambda r: -r.stamp)

    def _R(self, rec):
        return None if rec.rtag is None else self.R_by_tag[rec.rtag]

    def _mru(self, i, dense_only=False):
        """A container, biased to the most recently searched ones (i is small)."""
        recs = sorted((r for r in self.cs if not (dense_only and r.packed)), key=lambda r: -r.stamp)
        return recs[i % len(recs)]

    def _note_use(self, rec, qi, p, method):
        self.clock = getattr(self, "clock", 0) + 1
        rec.stamp = self.clock
        EV(f"cache:method={method}")
        if rec.last_p != "unset" and rec.last_p != p:
            EV("cache:precision_switch")
        if not rec.packed:
            EV("cache:xhat_warm_at_call" if rec.c._x_hat_rot is not None else "cache:xhat_cold_at_call")
        if rec.dropped_since_search:
            EV("cache:search_after_drop_cache")
        if rec.owner != qi:
            EV("cache:other_quantizer_than_encoder")
        if self.qmeta[qi]["dense"]:
            EV("cache:dense_R_quantizer")
        rec.last_p, rec.dropped_since_search = p, False

    def _memo_check(self, key, ids, sc):
        ids, sc = np.asarray(ids).copy(), np.asarray(sc).copy()
        if key in self.memo:
            EV("cache:memo_repeat_checked")
            pi, ps = self.memo[key]
            assert np.array_equal(pi, ids) and np.array_equal(ps, sc), (
                f"same question {key} answered differently the second time:\n"
                f"  first : {pi.tolist()} {ps.tolist()}\n  second: {ids.tolist()} {sc.tolist()}")
        else:
            self.memo[key] = (ids, sc)

    # -- structural invariants (read-only: must not warm any cache) ------------
    @invariant()
    def containers_are_consistent(self):
        if not self.ready:
            return
        for r in self.cs:
            c = r.c
            codes = codes_of(c)
            assert codes.shape == (c.n, c.d)
            if c.norms is not None:
                assert c.norms.shape == (c.n,)
            if r.packed:
                continue
            if c._x_hat_rot is not None:
                assert c._x_hat_rot.shape == (c.n, c.d), f"{r.label}: cached x_hat_rot has shape {c._x_hat_rot.shape}"
                assert np.array_equal(c._x_hat_rot, self.oracle.pq.centroids[c.indices]), (
                    f"{r.label}: cached x_hat_rot is not centroids[indices]")
        for r in self.cs:
            for key, v in r.c._dir_len.items():
                assert np.asarray(v).shape == (r.c.n,), f"{r.label}: _dir_len[{key!r}] has shape {np.asarray(v).shape}, container has n={r.c.n}"
                if key is None or isinstance(key, (int, np.integer)):
                    want = self.oracle.direction_lengths(r.c, key)
                    assert np.allclose(v, want, rtol=1e-5, atol=1e-6), f"{r.label}: _dir_len[{key!r}] is stale"

    # -- container-producing rules --------------------------------------------
    @rule(qi=st.integers(0, 9), sel=st.integers(0, 2 ** 20), size=st.integers(1, 24))
    def encode_rows(self, qi, sel, size):
        qi %= len(self.qs)
        if len(self.cs) >= 7:
            return
        n = len(self.X)
        rows = np.sort(np.random.default_rng(sel).choice(n, size=min(size, n), replace=False))
        q = self.qs[qi]
        tag = self.qmeta[qi]["rtag"]
        c = q.encode(self.X[rows])
        self.cs.append(CRec(c, f"c{len(self.cs)}", qi, rtag=tag, rows=rows,
                            fresh_ok=not self.qmeta[qi]["dense"] and tag is None))
        EV("cache:container_encoded")

    @rule(i=st.integers(0, 3), kind=st.sampled_from(["sorted", "unsorted", "dups", "mask", "negative", "empty", "all"]),
          seed=st.integers(0, 2 ** 20))
    def subset(self, i, kind, seed):
        if len(self.cs) >= 7:
            return
        rec = self._mru(i)
        n = rec.c.n
        rng = np.random.default_rng(seed)
        if n == 0:
            kind = "mask" if kind == "mask" else "empty"
        if kind == "mask" and n == 0:
            idx = np.zeros(0, dtype=bool)
        elif kind == "sorted":
            idx = np.sort(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False))
        elif kind == "unsorted":
            idx = rng.permutation(n)[: int(rng.integers(0, n + 1))]
        elif kind == "dups":
            idx = rng.integers(0, n, size=int(rng.integers(1, 2 * n + 1)))
        elif kind == "mask":
            idx = rng.random(n) < 0.5
        elif kind == "negative":
            idx = -1 - rng.integers(0, n, size=int(rng.integers(1, n + 1)))
        elif kind == "empty":
            idx = np.array([], dtype=np.int64)
        else:
            idx = np.arange(n)
        warm = (not rec.packed) and rec.c._x_hat_rot is not None
        sub = rec.c.subset(idx)
        flat = np.flatnonzero(idx) if idx.dtype == bool else np.asarray(idx, dtype=np.int64)
        assert sub.n == len(flat)
        assert np.array_equal(codes_of(sub), codes_of(rec.c)[flat])
        if rec.c.norms is not None:
            assert np.array_equal(sub.norms, rec.c.norms[flat])
        if rec.c.mean_dots is not None:
            assert np.array_equal(sub.mean_dots, rec.c.mean_dots[flat])
        if not rec.packed and sub._x_hat_rot is not None:
            assert np.array_equal(sub._x_hat_rot, self.oracle.pq.centroids[sub.indices])
        EV(f"cache:subset_{kind}")
        if warm:
            EV("cache:subset_of_warm_container")
        self.cs.append(CRec(sub, f"c{len(self.cs)}", rec.owner, packed=rec.packed, rtag=rec.rtag,
                            fresh_ok=rec.fresh_ok, rows=None if rec.rows is None else rec.rows[flat]))

    @rule(i=st.integers(0, 3))
    def save_and_load(self, i):
        if len(self.cs) >= 7:
            return
        rec = self._mru(i)
        path = str(self.tmp / f"saved{len(self.cs)}_{next(_ids)}.npz")
        cls = PackedVectors if rec.packed else CompressedVectors
        got = rec.c.save(path)
        assert got == path
        back = cls.load(path)
        assert back.n == rec.c.n and back.d == rec.c.d and back.bits == rec.c.bits
        assert back.rotation == rec.c.rotation and back.seed == rec.c.seed and back.scale == rec.c.scale
        assert np.array_equal(codes_of(back), codes_of(rec.c))
        for name in ("norms", "mean_dots", "mean"):
            a, b = getattr(rec.c, name), getattr(back, name)
            assert (a is None) == (b is None), f"{name} presence changed by save/load"
            if a is not None:
                assert np.array_equal(a, b), f"{name} changed by save/load"
        assert not back._dir_len and (rec.packed or back._x_hat_rot is None)
        EV("cache:save_load")
        self.cs.append(CRec(back, f"c{len(self.cs)}", rec.owner, packed=rec.packed, rtag=rec.rtag,
                            fresh_ok=rec.fresh_ok, rows=rec.rows))

    @rule(i=st.integers(0, 3))
    def to_packed(self, i):
        if len(self.cs) >= 7:
            return
        rec = self._mru(i, dense_only=True)
        pv = PackedVectors.from_compressed(rec.c)
        assert np.array_equal(pv.unpack_rows(0, pv.n), rec.c.indices)
        EV("cache:to_packed")
        self.cs.append(CRec(pv, f"p{len(self.cs)}", rec.owner, packed=True, rtag=rec.rtag,
                            fresh_ok=rec.fresh_ok, rows=rec.rows))

    @rule()
    def new_quantizer_same_params(self):
        if len(self.qs) >= 3:
            return
        self.qs.append(Quantizer(**self.params))
        self.qmeta.append(dict(dense=False, rtag=None))
        EV("cache:second_quantizer_built")

    # -- the R setter ------------------------------------------------------------
    @rule(qi=st.integers(0, 9))
    def assign_own_R_to_quantizer(self, qi):
        """q.R = q.R: the setter drops the RHT operator; the dense matrix it
        keeps must score and encode like the operator did (up to rounding)."""
        qi %= len(self.qs)
        q = self.qs[qi]
        q.R = q.R
        self.qmeta[qi]["dense"] = True
        EV("cache:R_setter_same_matrix")

    @rule(qi=st.integers(0, 9))
    def read_R(self, qi):
        qi %= len(self.qs)
        R = np.asarray(self.qs[qi].R, np.float64)
        tag = self.qmeta[qi]["rtag"]
        want = self.oracle.R if tag is None else self.R_by_tag[tag]
        assert np.allclose(R, want, atol=1e-5), "lazily built R differs from the reference rotation"
        assert np.allclose(R @ R.T, np.eye(self.d), atol=1e-4), "R is not orthogonal"
        EV("cache:R_read")

    @rule(seed=st.integers(0, 2 ** 20), sel=st.integers(0, 2 ** 20))
    def quantizer_with_replaced_R(self, seed, sel):
        """A new quantizer whose R is replaced by a different orthogonal matrix:
        it must encode and search with that matrix (and not with the operator)."""
        if self.rotation == "none" or len(self.qs) >= 3 or len(self.cs) >= 7:
            return
        rng = np.random.default_rng(seed)
        M, _ = np.linalg.qr(rng.standard_normal((self.d, self.d)))
        M = M.astype(np.float32)
        q = Quantizer(**self.params)
        q.R = M
        tag = next(_ids)
        self.R_by_tag[tag] = M.astype(np.float64)
        self.qs.append(q)
        self.qmeta.append(dict(dense=True, rtag=tag))
        qi = len(self.qs) - 1
        self.cs.append(CRec(q.encode(self.X), f"r{len(self.cs)}", qi, rtag=tag, fresh_ok=False,
                            rows=np.arange(len(self.X))))
        EV("cache:R_setter_replaced_matrix")

    # -- cache control -------------------------------------------------------------
    @rule(i=st.integers(0, 3))
    def drop_cache(self, i):
        rec = self._mru(i, dense_only=True)      # PackedVectors has no drop_cache (README documents it on CompressedVectors)
        rec.c.drop_cache()
        assert rec.c._dir_len == {}, "drop_cache left per-precision state behind"
        assert rec.c._x_hat_rot is None, "drop_cache left the float32 cache behind"
        rec.dropped_since_search = True
        EV("cache:drop_cache")

    @rule(i=st.integers(0, 99), qi=st.integers(0, 9), prec=st.integers(0, 12), k=st.integers(1, 30),
          qsel=st.integers(0, 5))
    def warm_only(self, i, qi, prec, k, qsel):
        """Populate caches without checking the answer (so checks do not mask staleness)."""
        qi %= len(self.qs)
        els = self._eligible(qi, packed_ok=False)
        if not els:
            return
        rec = els[i % len(els)]
        self.qs[qi].search(rec.c, self.Q[qsel], k=k, precision=self._precision(prec))
        EV("cache:warm_only_search")

    # -- searches ---------------------------------------------------------------------
    def _answer(self, i, qi, prec, k, qsel, method, cand=2, cprec=0, chunk=4096, rec=None):
        qi %= len(self.qs)
        packed_ok = method in ("search_adc", "search_twostage")
        if method == "decode" and self.mode == "centered":
            return
        if rec is None:
            els = self._eligible(qi, packed_ok)
            if not els:
                return
            rec = els[i % len(els)]
        q, p = self.qs[qi], self._precision(prec)
        R = self._R(rec)
        n = rec.c.n
        qv = self.Q[qsel]
        self._note_use(rec, qi, p, method)
        dense = self.qmeta[qi]["dense"]
        base = (rec.label, dense, self.qmeta[qi]["rtag"], method, p, k, qsel, chunk, cand, cprec)
        if method == "decode":
            got = q.decode(rec.c, precision=p)
            want = self.oracle.decode(rec.c, p, R)
            assert got.shape == want.shape
            assert np.allclose(got, want, rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(want).max(initial=0.0)))), (
                f"decode at precision {p} disagrees with the cache-free reconstruction")
            return
        if method == "search_batch":
            nq = 1 + qsel % 3
            Qb = self.Q[qsel: qsel + nq] if qsel + nq <= len(self.Q) else self.Q[:nq]
            EV(f"cache:batch_nq={len(Qb)}")
            ids, sc = q.search_batch(rec.c, Qb, k=k, precision=p)
            assert ids.shape == sc.shape == (len(Qb), min(k, n))
            for r_i in range(len(Qb)):
                ref = self.oracle.scores(rec.c, Qb[r_i], p, R)
                check_topk(ref, ids[r_i], sc[r_i], k, tolerance(ref), what=f"search_batch row {r_i} p={p}")
            self._memo_check(base + (len(Qb),), ids.ravel(), sc.ravel())
            return
        ref = self.oracle.scores(rec.c, qv, p, R)
        tol = tolerance(ref)
        if method == "search":
            ids, sc = q.search(rec.c, qv, k=k, precision=p)
            check_topk(ref, ids, sc, k, tol, what=f"search p={p}")
        elif method == "search_adc":
            ids, sc = q.search_adc(rec.c, qv, k=k, precision=p, chunk_size=chunk)
            check_topk(ref, ids, sc, k, tol, what=f"search_adc p={p}")
        else:
            candidates = [1, max(1, n // 2), n, n + 7][cand]
            cp = self._precision(cprec)
            ref_c = self.oracle.scores(rec.c, qv, self.bits - 2 if cp is None and self.bits > 2 else (1 if cp is None else cp), R)
            ref_f = self.oracle.scores(rec.c, qv, None, R)
            ids, sc = q.search_twostage(rec.c, qv, k=k, candidates=candidates, coarse_precision=cp)
            check_twostage(ref_f, ref_c, ids, sc, k, candidates, max(tol, tolerance(ref_f), tolerance(ref_c)),
                           what=f"search_twostage cp={cp} cand={candidates}")
            EV(f"cache:twostage_candidates={'exhaustive' if candidates >= n else 'partial'}")
        self._memo_check(base, ids, sc)

    _ANS = dict(i=st.integers(0, 3), qi=st.integers(0, 9), prec=st.integers(0, 12), k=st.integers(1, 30),
                qsel=st.integers(0, 5))

    @rule(**_ANS)
    def answer_search(self, i, qi, prec, k, qsel):
        self._answer(i, qi, prec, k, qsel, "search")

    @rule(**_ANS)
    def answer_search_batch(self, i, qi, prec, k, qsel):
        self._answer(i, qi, prec, k, qsel, "search_batch")

    @rule(chunk=st.sampled_from([1, 3, 4096]), **_ANS)
    def answer_search_adc(self, i, qi, prec, k, qsel, chunk):
        self._answer(i, qi, prec, k, qsel, "search_adc", chunk=chunk)

    @rule(cand=st.integers(0, 3), cprec=st.integers(0, 12), **_ANS)
    def answer_search_twostage(self, i, qi, prec, k, qsel, cand, cprec):
        self._answer(i, qi, prec, k, qsel, "search_twostage", cand=cand, cprec=cprec)

    @rule(**_ANS)
    def answer_decode(self, i, qi, prec, k, qsel):
        self._answer(i, qi, prec, k, qsel, "decode")

    @rule(i=st.integers(0, 3), qi=st.integers(0, 9), p1=st.integers(0, 12), p2=st.integers(0, 12),
          drop=st.booleans(), method=st.sampled_from(["search", "search_batch", "search_adc", "decode"]),
          k=st.integers(1, 30), qsel=st.integers(0, 5))
    def precision_cycle(self, i, qi, p1, p2, drop, method, k, qsel):
        """Targeted pattern: answer at p1 (warming the caches), optionally drop them,
        answer at p2, at full precision, then at p1 again.  Every answer is checked
        against the cache-free reference and the two p1 answers must be identical."""
        qi %= len(self.qs)
        els = self._eligible(qi, packed_ok=method == "search_adc")
        if not els:
            return
        rec = els[i % len(els)]
        for step, p in enumerate((p1, p2, 0, p1)):
            if drop and step == 1 and not rec.packed:
                rec.c.drop_cache()
                rec.dropped_since_search = True
                EV("cache:drop_inside_cycle")
            self._answer(i, qi, p, k, qsel, method, rec=rec)
        EV("cache:precision_cycle")

    @rule(i=st.integers(0, 3), qi=st.integers(0, 9), kind=st.sampled_from(["sorted", "unsorted", "dups", "mask", "negative"]),
          seed=st.integers(0, 2 ** 20), p=st.integers(0, 12), k=st.integers(1, 30), qsel=st.integers(0, 5),
          via=st.sampled_from(["subset", "save_load", "packed"]))
    def warm_derive_search(self, i, qi, kind, seed, p, k, qsel, via):
        """Targeted pattern: warm a container's caches (search at precision p), derive a new
        container from it (subset / save+load / pack), then search the derived one."""
        qi %= len(self.qs)
        els = self._eligible(qi, packed_ok=False)
        if not els or len(self.cs) >= 7:
            return
        rec = els[i % len(els)]
        self._answer(0, qi, p, k, qsel, "search", rec=rec)
        n = rec.c.n
        rng = np.random.default_rng(seed)
        if via == "subset" and n > 0:
            idx = {"sorted": lambda: np.sort(rng.choice(n, size=max(1, n // 2), replace=False)),
                   "unsorted": lambda: rng.permutation(n)[: max(1, n - 1)],
                   "dups": lambda: rng.integers(0, n, size=n + 2),
                   "mask": lambda: rng.random(n) < 0.5,
                   "negative": lambda: -1 - rng.integers(0, n, size=max(1, n // 2))}[kind]()
            sub = rec.c.subset(idx)
        elif via == "save_load":
            path = str(self.tmp / f"w{next(_ids)}.npz")
            rec.c.save(path)
            sub = CompressedVectors.load(path)
        else:
            sub = PackedVectors.from_compressed(rec.c)
        new = CRec(sub, f"w{len(self.cs)}", rec.owner, packed=isinstance(sub, PackedVectors), rtag=rec.rtag,
                   fresh_ok=rec.fresh_ok, rows=None)
        self.cs.append(new)
        EV(f"cache:warm_then_derive_{via}")
        if isinstance(sub, PackedVectors):
            self._answer(0, qi, p, k, qsel, "search_adc", rec=new)
        else:
            self._answer(0, qi, p, k, qsel, "search", rec=new)
            self._answer(0, qi, 0, k, qsel, "search_batch", rec=new)

    @rule(i=st.integers(0, 99), qi=st.integers(0, 9), prec=st.integers(1, 40),
          method=st.sampled_from(["search", "search_batch", "search_adc", "search_twostage", "decode"]))
    def invalid_precision_is_refused(self, i, qi, prec, method):
        qi %= len(self.qs)
        els = self._eligible(qi, packed_ok=method in ("search_adc", "search_twostage"))
        if not els:
            return
        rec = els[i % len(els)]
        bad = (0, -1, self.bits + 1, self.bits + prec)[prec % 4]
        q = self.qs[qi]
        with pytest.raises(ValueError):
            if method == "search":
                q.search(rec.c, self.Q[0], k=3, precision=bad)
            elif method == "search_batch":
                q.search_batch(rec.c, self.Q[:2], k=3, precision=bad)
            elif method == "search_adc":
                q.search_adc(rec.c, self.Q[0], k=3, precision=bad)
            elif method == "search_twostage":
                q.search_twostage(rec.c, self.Q[0], k=3, coarse_precision=bad)
            else:
                q.decode(rec.c, precision=bad)
        EV("cache:invalid_precision_refused")

    # -- guards: a container another encoding wrote is refused, warm or cold ----------
    def _foreign(self, kind: str, warm: bool):
        key = (kind, warm)
        if key in self.foreign:
            return self.foreign[key]
        p = dict(self.params)
        if kind == "bits":
            p["bits"] = 2 if self.bits != 2 else 3
        elif kind == "seed":
            p["seed"] = self.params["seed"] + 1
        elif kind == "rotation":
            cur = self.oracle.pq.rotation
            p["rotation"] = "haar" if cur != "haar" else "none"
        elif kind == "d":
            p["d"] = self.d + 2
            if "mean" in p:
                p["mean"] = np.full(p["d"], 0.25, np.float32)
        elif kind == "mode":
            if self.scalar:
                p.pop("normalize", None); p.pop("scale", None)
            else:
                p.update(normalize=False); p.pop("mean", None); p.pop("renorm", None)
        elif kind == "mean":
            if self.mode == "centered":
                p.pop("mean", None)
            else:
                p.update(mean=np.ones(self.d, np.float32) * 0.5, renorm=True)
        owner = _cached_quantizer(p)
        d = p["d"]
        Xf = np.random.default_rng(5).standard_normal((9, d)).astype(np.float32)
        c = owner.encode(Xf)
        if warm:
            qf = np.random.default_rng(6).standard_normal(d).astype(np.float32)
            owner.search(c, qf, k=3)
            owner.search(c, qf, k=3, precision=1)
        self.foreign[key] = (owner, c)
        return self.foreign[key]

    GUARD_METHODS = ("decode", "search_adc", "search_twostage", "search_batch", "search")

    def _guard(self, kind, warm, qi):
        """A container written under another encoding must be refused by every read path,
        whether or not its owner has warmed its caches."""
        if kind in ("rotation", "seed") and self.rotation == "none":
            return                                  # identity rotation ignores the seed
        if kind == "mean" and self.scalar:
            return                                  # scalar mode cannot carry a mean at all
        f2 = (kind == "bits" and (self.scalar or not self.oracle.pq.renorm)) or \
             (kind == "mean" and not self.oracle.pq.renorm)
        if f2 and not self.GUARD_KINDS_INCLUDE_RENORM_FALSE:
            return                                  # finding F2 territory, see CacheMachineGuardKnownBug
        owner, c = self._foreign(kind, warm)
        q = self.qs[qi % len(self.qs)]
        qv = np.random.default_rng(1).standard_normal(self.d).astype(np.float32)
        EV(f"cache:guard_{kind}_{'warm' if warm else 'cold'}")
        for method in self.GUARD_METHODS:
            with pytest.raises(ValueError, match="mismatch"):
                if method == "search":
                    q.search(c, qv, k=3)
                elif method == "search_batch":
                    q.search_batch(c, qv[None, :], k=3)
                elif method == "search_adc":
                    q.search_adc(c, qv, k=3)
                elif method == "search_twostage":
                    q.search_twostage(c, qv, k=3)
                else:
                    q.decode(c)

    @rule(qi=st.integers(0, 9))
    def every_foreign_encoding_is_refused(self, qi):
        """Cross every kind of encoding mismatch with a cold and a warm foreign container."""
        for kind in self.GUARD_KINDS:
            for warm in ((False, True) if self.GUARD_WARM is None else (self.GUARD_WARM,)):
                self._guard(kind, warm, qi)

    GUARD_KINDS_INCLUDE_RENORM_FALSE = False

    @rule(i=st.integers(0, 99))
    def verify_content_equals_fresh_encode(self, i):
        rec = self.cs[i % len(self.cs)]
        if not rec.fresh_ok or rec.rows is None:
            return
        fresh = self.oracle.pq.encode(self.X[rec.rows])
        a, b = codes_of(rec.c), fresh.indices
        frac = float((a != b).mean()) if a.size else 0.0
        assert frac <= (5e-4 if self.oracle.pq.rotation == "haar" else 0.0), (
            f"{rec.label}: {frac:.2%} of codes differ from a fresh from-scratch encode of the same rows")
        if rec.c.norms is not None:
            assert np.array_equal(rec.c.norms, fresh.norms)
        EV("cache:content_equals_fresh_encode_checked")


class CacheMachineGuardKnownBug(CacheMachine):
    """Finding F2: with renorm=False, Quantizer.search on a warm container skips the
    bits and mean guards that a cold search (and search_batch / search_adc / decode)
    enforce.  Only the guard rule runs here, warm, for the two affected kinds."""

    FORCE_RENORM_FALSE = True
    GUARD_KINDS = ("bits", "mean")
    GUARD_METHODS = ("search",)
    GUARD_WARM = True
    GUARD_KINDS_INCLUDE_RENORM_FALSE = True

    # the guard check is one rule among ~28; give it weight so 60 examples find the bug
    @rule(qi=st.integers(0, 9))
    def focus_a(self, qi):
        self.every_foreign_encoding_is_refused(qi)

    @rule(qi=st.integers(0, 9))
    def focus_b(self, qi):
        self.every_foreign_encoding_is_refused(qi)

    @rule(qi=st.integers(0, 9))
    def focus_c(self, qi):
        self.every_foreign_encoding_is_refused(qi)


TestCacheMachine = CacheMachine.TestCase
TestCacheMachine.settings = machine_settings()

TestCacheMachineGuardKnownBug = pytest.mark.xfail(
    strict=True, reason="F2: Quantizer.search skips _check_bits/_check_mean on a warm cache when renorm=False "
                        "(core.py:1665-1672)")(CacheMachineGuardKnownBug.TestCase)
TestCacheMachineGuardKnownBug.settings = machine_settings(max_examples=60)


# --------------------------------------------------------------------------
# Machine 1: RemexIndex against a from-scratch reference
# --------------------------------------------------------------------------

VOCAB = [f"w{i}" for i in range(40)]

INDEX_CONFIG = st.fixed_dictionaries({
    "kind": st.sampled_from(["cosine", "dot", "maxsim", "meanmaxsim"]),
    "d": st.sampled_from([4, 6, 8, 9, 16, 24]),
    "bits": st.sampled_from([1, 2, 3, 4, 8]),
    "center": st.booleans(),
    "low_memory": st.booleans(),
    "seed": st.sampled_from([0, 1, 42]),
    "with_model": st.sampled_from([True, True, False]),
    "dim_given": st.booleans(),
    "shared": st.sampled_from([0.0, 0.5, 2.0]),
    "pooling": st.sampled_from([None, None, "instance", "dict"]),
    "block": st.sampled_from([None, 8, 64]),
})


def _fake_st_modules():
    """sentence_transformers.multi_vector_encoder.modules, as test_st_pooling.py builds it."""
    mods = types.ModuleType("sentence_transformers.multi_vector_encoder.modules")
    mods.FakeTokenPooling = FakeTokenPooling
    mods.HierarchicalTokenPooling = FakeTokenPooling
    pkg = types.ModuleType("sentence_transformers.multi_vector_encoder")
    pkg.modules = mods
    root = types.ModuleType("sentence_transformers")
    root.multi_vector_encoder = pkg
    return {"sentence_transformers": root, "sentence_transformers.multi_vector_encoder": pkg,
            "sentence_transformers.multi_vector_encoder.modules": mods}


def _unit(X):                                   # remex.st._unit_rows, restated
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class RemexIndexMachine(RuleBasedStateMachine):
    """History-dependent behaviour of RemexIndex: add batches (varied sizes, text and
    precomputed), searches between adds, save/load mid-history (and rewinding to an
    older snapshot), single-vector and MaxSim, centred and low-memory.

    Reference: the document list, re-encoded from scratch in one ``Quantizer.encode``
    call (centred: the mean of the *first* batch, as documented) and scored by an
    independent float64 brute force; MaxSim is a plain Python reduction."""

    MODEL_LESS_UNKNOWN_DIM = False      # KnownBug machine: no model, dim not given

    def __init__(self):
        super().__init__()
        self.tmp = _tmpdir()
        self.patches = []
        self.snaps: dict = {}
        self.ready = False
        self.adds_since_search = 0

    def teardown(self):
        for p in self.patches:
            p.stop()

    @initialize(cfg=INDEX_CONFIG)
    def setup(self, cfg):
        import unittest.mock as mock
        kind = cfg["kind"]
        self.kind, self.d, self.bits = kind, cfg["d"], cfg["bits"]
        self.multi = kind in ("maxsim", "meanmaxsim")
        self.center, self.seed = cfg["center"], cfg["seed"]
        with_model = cfg["with_model"] and not self.MODEL_LESS_UNKNOWN_DIM
        self.has_model = with_model
        if self.multi:
            self.model = FakeMultiVectorModel(dim=self.d, shared=cfg["shared"]) if with_model else None
        else:
            self.model = FakeModel(dim=self.d, similarity_fn_name=kind) if with_model else None
        self.dim_known = with_model or (cfg["dim_given"] and not self.MODEL_LESS_UNKNOWN_DIM)
        self.pool_factor = None
        pooling = None
        if self.multi and cfg["pooling"] is not None:
            self.pool_factor = 2 + (cfg["seed"] % 2)
            # an index loaded from a file holds its pooling as a settings dict and builds the
            # module on the next add, whichever way the original was given, so the stand-in
            # sentence_transformers modules must be importable for every pooled configuration
            p = mock.patch.dict(sys.modules, _fake_st_modules())
            p.start()
            self.patches.append(p)
            if cfg["pooling"] == "instance":
                pooling = FakeTokenPooling(pool_factor=self.pool_factor)
            else:
                pooling = {"pool_factor": self.pool_factor}
        if cfg["block"] is not None:
            p = mock.patch.object(remex_st, "_SCORE_BLOCK_FLOATS", cfg["block"])
            p.start()
            self.patches.append(p)
        self.block = cfg["block"]
        self.idx = RemexIndex(self.model, bits=self.bits, similarity=kind, center=self.center,
                              low_memory=cfg["low_memory"], seed=self.seed,
                              dim=self.d if self.dim_known and not with_model else None,
                              token_pooling=pooling)
        self.low_memory = cfg["low_memory"]
        self.pool_mode = cfg["pooling"] if self.multi else None
        self.docs: list = []              # reference: one entry per document, in insertion order
        self.first_n = 0                  # documents in the first successful batch
        self.ref_q: Optional[Quantizer] = None
        self.ready = True
        for lab in (f"idx:kind={kind}", f"idx:bits={self.bits}", f"idx:center={self.center}",
                    f"idx:low_memory={self.low_memory}", f"idx:model={with_model}",
                    f"idx:pooling={cfg['pooling'] if self.multi else None}", f"idx:block={cfg['block']}",
                    f"idx:d_parity={'odd(haar)' if self.d % 2 else 'even(rht)'}"):
            EV(lab)

    # -- reference model -----------------------------------------------------------
    def _norm_flag(self):
        return self.kind == "cosine"

    def _ref_rows(self, items):
        """Reference embeddings of documents -> (n, d) float32 (single) or list of mats (multi)."""
        if not self.multi:
            rows = []
            for kind, payload in items:
                if kind == "text":
                    rows.append(self.model.encode([payload], normalize_embeddings=self._norm_flag())[0])
                else:
                    rows.append(np.asarray(payload, np.float32))
            X = np.stack(rows).astype(np.float32)
            return _unit(X) if self.kind == "cosine" else X
        mats = []
        for kind, payload in items:
            if kind == "text":
                m = self.model.encode([payload])[0]
                if self.pool_factor is not None:
                    m = FakeTokenPooling(pool_factor=self.pool_factor).pool([m])[0]
                mats.append(np.asarray(m, np.float32))
            else:
                mats.append(np.asarray(payload, np.float32))
        return mats

    def _ref_build(self, docs, first_n):
        """Re-encode every document from scratch -> (oracle, codes, offsets or None)."""
        items = self._ref_rows(docs)
        first = self._ref_rows(docs[:first_n])
        if self.multi:
            X = np.concatenate(items)
            first_X = np.concatenate(first)
            counts = np.array([len(m) for m in items])
            offsets = np.concatenate([[0], np.cumsum(counts)])
        else:
            X, first_X, offsets = items, first, None
        mean = corpus_mean(first_X) if self.center else None
        params = dict(d=self.d, bits=self.bits, seed=self.seed)
        if mean is not None:
            params["mean"] = mean
        oracle = Oracle(params)
        return oracle, oracle.pq.encode(X), offsets

    # -- invariants (cheap, read-only) ---------------------------------------------
    @invariant()
    def length_and_bookkeeping(self):
        if not self.ready:
            return
        assert len(self.idx) == len(self.docs)
        if self.idx.compressed is not None and self.multi:
            off = self.idx.offsets
            assert off[0] == 0 and off[-1] == self.idx.compressed.n and np.all(np.diff(off) > 0)
            assert len(off) - 1 == len(self.docs)
        if self.idx.compressed is not None:
            assert self.idx.compressed.norms.shape == (self.idx.compressed.n,)
            assert self.idx.compressed.indices.shape[0] == self.idx.compressed.n

    # -- document generation ---------------------------------------------------------
    def _make_docs(self, size, seed, as_arrays, dup):
        rng = np.random.default_rng(seed)
        docs = []
        for _ in range(size):
            if as_arrays or not self.has_model:
                if self.multi:
                    t = int(rng.integers(1, 7))
                    m = rng.standard_normal((t, self.d)).astype(np.float32)
                    m /= np.linalg.norm(m, axis=1, keepdims=True)
                    docs.append(("array", m))
                else:
                    v = rng.standard_normal(self.d).astype(np.float32) * float(rng.choice([1.0, 3.0]))
                    docs.append(("array", v))
            else:
                n_words = int(rng.integers(1, 7)) if self.multi else 3
                docs.append(("text", " ".join(rng.choice(VOCAB, n_words))))
        same = [d for d in self.docs if d[0] == docs[0][0]]
        if dup and same:
            docs[0] = same[int(rng.integers(0, len(same)))]
        return docs

    @staticmethod
    def _payload(docs):
        return [p for _, p in docs]

    # -- rules ---------------------------------------------------------------------------
    def _add(self, size, seed, as_arrays, dup, shape="list"):
        docs = self._make_docs(size, seed, as_arrays, dup)
        before = len(self.idx)
        if shape == "single" and size == 1 and docs[0][0] == "array" and not self.multi:
            payload = docs[0][1]                       # a bare (d,) vector
        else:
            payload = self._payload(docs)
            if not self.multi and docs[0][0] == "array":
                payload = np.stack(payload)            # an (n, d) array
        ids = self.idx.add(payload)
        assert list(map(int, ids)) == list(range(before, before + len(docs))), (
            f"add returned ids {list(ids)}, expected {before}..{before + len(docs) - 1}")
        if not self.docs:
            self.first_n = len(docs)
        self.docs.extend(docs)
        self.adds_since_search += 1
        EV(f"idx:add_size={min(len(docs), 4)}")
        EV(f"idx:add_input={docs[0][0]}")
        if dup:
            EV("idx:add_with_duplicate_document")
        if self.snaps:
            EV("idx:add_after_snapshot_taken")

    _SIZE = st.sampled_from([1, 1, 2, 3, 5, 6])

    @rule(size=_SIZE, seed=st.integers(0, 2 ** 20), dup=st.booleans())
    def add_text(self, size, seed, dup):
        """Add documents the model encodes (needs a model; else precomputed)."""
        self._add(size, seed, as_arrays=False, dup=dup)

    @rule(size=_SIZE, seed=st.integers(0, 2 ** 20), dup=st.booleans(), shape=st.sampled_from(["list", "single"]))
    def add_embeddings(self, size, seed, dup, shape):
        """Add precomputed embeddings / token matrices."""
        self._add(size, seed, as_arrays=True, dup=dup, shape=shape)

    @rule(sizes=st.lists(_SIZE, min_size=1, max_size=3), seed=st.integers(0, 2 ** 20),
          qmode=st.sampled_from(["text", "list", "array"]), top_k=st.integers(1, 45), warm=st.booleans())
    def warm_add_search(self, sizes, seed, qmode, top_k, warm):
        """Targeted pattern: search (warming any cache), add one or more batches, search again
        -- the search must see every new document."""
        if warm and self.docs:
            self._check_search(self.idx, self.docs, self.first_n, qmode, seed, 1, top_k, "warm search before adds")
            EV("idx:warm_search_before_adds")
        for j, size in enumerate(sizes):
            self._add(size, seed + j, as_arrays=not self.has_model, dup=False)
        EV(f"idx:adds_between_searches={min(len(sizes), 3)}")
        self.adds_since_search = 0
        self._check_search(self.idx, self.docs, self.first_n, qmode, seed + 99, 2, top_k, "search after adds")

    def _queries(self, qmode, seed, nq):
        """-> (arg for search, [reference query vectors/matrices], single?)"""
        rng = np.random.default_rng(seed)
        texts = []
        for _ in range(nq):
            if self.docs and rng.random() < 0.4 and self.has_model:
                kind, payload = self.docs[int(rng.integers(0, len(self.docs)))]
                texts.append(payload if kind == "text" else " ".join(rng.choice(VOCAB, 2)))
            else:
                texts.append(" ".join(rng.choice(VOCAB, int(rng.integers(1, 5)) if self.multi else 2)))
        use_text = self.has_model and qmode != "array"
        if self.multi:
            if use_text:
                refs = [self.model.encode([t])[0] for t in texts]
                return (texts[0], refs[:1], True) if qmode == "text" else (texts, refs, False)
            mats = [(rng.standard_normal((int(rng.integers(1, 5)), self.d)).astype(np.float32)) for _ in range(nq)]
            mats = [m / np.linalg.norm(m, axis=1, keepdims=True) for m in mats]
            return (mats[0], mats[:1], True) if qmode == "text" else (mats, mats, False)
        if use_text:
            refs = [self.model.encode([t], normalize_embeddings=self._norm_flag())[0] for t in texts]
            refs = [r for r in refs]
            return (texts[0], refs[:1], True) if qmode == "text" else (texts, refs, False)
        arr = rng.standard_normal((nq, self.d)).astype(np.float32)
        if qmode == "text":                              # single 1-D embedding
            return arr[0], [arr[0]], True
        return arr, list(arr), False

    def _ref_doc_scores(self, oracle, codes, offsets, q):
        if not self.multi:
            qv = _unit(np.asarray(q, np.float32)[None, :])[0] if self.kind == "cosine" else np.asarray(q, np.float32)
            return oracle.scores(codes, qv)
        S = oracle.matrix(codes, np.asarray(q, np.float32))             # (query tokens, corpus tokens)
        out = np.empty(len(offsets) - 1)
        for i in range(len(offsets) - 1):
            out[i] = S[:, offsets[i]:offsets[i + 1]].max(axis=1).sum()
        if self.kind == "meanmaxsim":
            out /= len(S)
        return out

    def _check_search(self, idx, docs, first_n, qmode, seed, nq, top_k, what):
        oracle, codes, offsets = self._ref_build(docs, first_n)
        arg, refs, single = self._queries(qmode, seed, nq)
        got = idx.search(arg, top_k=top_k)
        got = [got] if single else got
        if single:
            assert isinstance(got[0], list)
        assert len(got) == len(refs), f"{what}: {len(got)} result lists for {len(refs)} queries"
        for hits, q in zip(got, refs):
            ref = self._ref_doc_scores(oracle, codes, offsets, q)
            tol = tolerance(ref)
            assert all(set(h) == {"corpus_id", "score"} and isinstance(h["corpus_id"], int)
                       and isinstance(h["score"], float) for h in hits)
            check_topk(ref, [h["corpus_id"] for h in hits], [h["score"] for h in hits], top_k, tol, what=what)
        return oracle, codes, offsets

    @rule(qmode=st.sampled_from(["text", "list", "array"]), seed=st.integers(0, 2 ** 20),
          nq=st.integers(1, 3), top_k=st.integers(1, 45))
    def search(self, qmode, seed, nq, top_k):
        if not self.docs:
            return
        EV(f"idx:adds_before_search={min(self.adds_since_search, 3)}")
        EV(f"idx:search_query={qmode}")
        EV("idx:search_top_k_exceeds_corpus" if top_k >= len(self.docs) else "idx:search_top_k_within_corpus")
        self.adds_since_search = 0
        self._check_search(self.idx, self.docs, self.first_n, qmode, seed, nq, top_k, "RemexIndex.search")

    @rule()
    def codes_equal_from_scratch(self):
        """Incremental construction == one-shot construction, bit for bit."""
        if not self.docs:
            return
        oracle, codes, offsets = self._ref_build(self.docs, self.first_n)
        c = self.idx.compressed
        assert c.n == codes.n
        diff = float((c.indices != codes.indices).mean())
        assert diff <= (5e-4 if oracle.pq.rotation == "haar" else 0.0), (
            f"{diff:.2%} of codes differ from encoding all {len(self.docs)} documents at once")
        assert np.array_equal(c.norms, codes.norms)
        if codes.mean_dots is not None:
            assert np.array_equal(c.mean_dots, codes.mean_dots)
        assert (c.mean is None) == (codes.mean is None)
        if codes.mean is not None:
            assert np.array_equal(c.mean, codes.mean), "first-batch mean changed"
        if self.multi:
            assert np.array_equal(self.idx.offsets, offsets)
        assert self.idx.nbytes == codes.nbytes
        EV("idx:codes_equal_from_scratch_checked")

    @rule(name=st.sampled_from(["a", "b"]), suffix=st.booleans())
    def save(self, name, suffix):
        if not self.docs:
            return
        path = str(self.tmp / (name + (".npz" if suffix else "")))
        got = self.idx.save(path)
        assert got.endswith(".npz") and Path(got).exists()
        self.snaps[name] = (got, list(self.docs), self.first_n)
        EV("idx:save")
        if self.adds_since_search:
            EV("idx:save_with_unsearched_adds")

    @rule(name=st.sampled_from(["a", "b"]), low_memory=st.booleans(),
          qmode=st.sampled_from(["text", "list", "array"]), seed=st.integers(0, 2 ** 20))
    def load(self, name, low_memory, qmode, seed):
        # (loading by the suffix-less stem is finding F3: see test_stateful_regressions.py)
        if name not in self.snaps:
            return
        path, docs, first_n = self.snaps[name]
        new = RemexIndex.load(path, self.model, low_memory=low_memory)
        assert (new.bits, new.seed, new.center, new.similarity, new.dim) == \
               (self.bits, self.seed, self.center, self.kind, self.d)
        assert new.low_memory == low_memory
        assert len(new) == len(docs)
        if self.pool_mode is not None:
            spec = remex_st._pooling_spec(new.token_pooling)
            assert spec.get("pool_factor") == self.pool_factor, "token_pooling not restored by load"
        self._check_search(new, docs, first_n, qmode, seed, 2, 7, "RemexIndex.load -> search")
        self.idx, self.docs, self.first_n, self.low_memory = new, list(docs), first_n, low_memory
        self.adds_since_search = 0
        EV("idx:load")
        EV("idx:load_rewinds_history" if len(docs) < len(self.docs) or True else "")

    # -- rejected operations must leave the index usable and unchanged --------------------
    def _state(self):
        i = self.idx
        return (len(i), i.dim, i.quantizer is None, id(i.compressed),
                None if i.offsets is None else i.offsets.tolist())

    @rule(which=st.integers(0, 4))
    def rejected_operation(self, which):
        before = self._state()
        if which == 0 and (self.dim_known or self.MODEL_LESS_UNKNOWN_DIM):
            with pytest.raises(ValueError):
                self.idx.add([])
            EV("idx:rejected_empty_add")
        elif which == 1 and self.dim_known and (self.docs or self.has_model):
            bad = [np.ones((2, self.d + 1), np.float32)] if self.multi else np.ones((2, self.d + 1), np.float32)
            with pytest.raises(ValueError):
                self.idx.add(bad)
            EV("idx:rejected_wrong_dim_add")
        elif which == 2 and self.docs:
            with pytest.raises(ValueError):
                self.idx.search(np.ones((1, self.d + 1), np.float32) if self.multi else np.ones(self.d + 1, np.float32))
            q = np.ones((1, self.d), np.float32) if self.multi else np.ones(self.d, np.float32)
            with pytest.raises(ValueError):
                self.idx.search(q, top_k=0)
            EV("idx:rejected_query")
        elif which == 3 and not self.has_model:
            with pytest.raises(ValueError):
                self.idx.add(["some text"])
            EV("idx:rejected_text_without_model")
        elif which == 4 and self.multi and self.MODEL_LESS_UNKNOWN_DIM and not self.docs:
            rng = np.random.default_rng(0)
            with pytest.raises(ValueError):
                self.idx.add([rng.standard_normal((2, 8)).astype(np.float32),
                              rng.standard_normal((2, 5)).astype(np.float32)])
            EV("idx:rejected_mixed_dim_batch")
        else:
            return
        assert self._state() == before, "a rejected operation changed the index"


class RemexIndexRejectsKnownBug(RemexIndexMachine):
    """Finding F1: an index with no model and no ``dim`` learns its dimension from the
    first batch it sees -- even a batch that is then rejected.  add([]) (and a
    MaxSim batch whose matrices disagree on d) leave ``dim`` poisoned."""

    MODEL_LESS_UNKNOWN_DIM = True


TestRemexIndexMachine = RemexIndexMachine.TestCase
TestRemexIndexMachine.settings = machine_settings()

TestRemexIndexRejectsKnownBug = pytest.mark.xfail(
    strict=True, reason="F1: RemexIndex._check_dim records dim before the batch is accepted "
                        "(st.py:304-308, 310-319, 321-336)")(RemexIndexRejectsKnownBug.TestCase)
TestRemexIndexRejectsKnownBug.settings = machine_settings(max_examples=60)


# --------------------------------------------------------------------------
# Machine 3: GPUSearcher (numpy backend) and IVFCoarseIndex vs Quantizer
# --------------------------------------------------------------------------

GPU_CONFIG = st.fixed_dictionaries({
    "d": st.sampled_from([8, 9, 16, 24]),
    "bits": st.sampled_from([1, 2, 3, 4, 8]),
    "mode": st.sampled_from(["plain", "plain", "centered", "scalar"]),
    "rotation": st.sampled_from([None, None, "haar", "none", "rht"]),
    "renorm": st.booleans(),
    "seed": st.sampled_from([0, 1, 42]),
    "n": st.integers(1, 40),
    "data_seed": st.integers(0, 2 ** 31 - 1),
    "kind": st.sampled_from(["gauss", "dups", "zeros", "shifted", "scaled"]),
    "scale": st.sampled_from([None, 0.5, 2.0]),
    "mean_kind": st.sampled_from(["corpus", "zero", "random", "row0"]),
})


def _popcount(x: int) -> int:
    return bin(x).count("1")


class GpuIvfMachine(RuleBasedStateMachine):
    """Build GPUSearcher / IVFCoarseIndex over one corpus, search, drop caches (CPU and
    GPU side), rebuild, search again.  Every answer is checked against the cache-free
    reference; IVF additionally against an independent re-derivation of its cells."""

    def __init__(self):
        super().__init__()
        self.ready = False
        self.gpus: list = []
        self.ivfs: list = []

    @initialize(cfg=GPU_CONFIG)
    def setup(self, cfg):
        r = resolve_config(cfg)
        self.params, self.X, self.Q = r["params"], r["X"], r["Q"]
        self.mode, self.scalar = r["mode"], r["scalar"]
        self.d, self.bits, self.n = self.params["d"], self.params["bits"], len(self.X)
        self.pq = Quantizer(**self.params)
        self.oracle = Oracle(self.params)
        self.c = self.pq.encode(self.X)
        self.packed = PackedVectors.from_compressed(self.c)
        self.ready = True
        EV(f"gpuivf:mode={self.mode}")
        EV(f"gpuivf:bits={self.bits}")
        EV(f"gpuivf:n={'1' if self.n == 1 else ('<=8' if self.n <= 8 else '>8')}")

    def _prec(self, code):
        return None if code == 0 else (code - 1) % self.bits + 1

    # -- builders -----------------------------------------------------------------
    @rule()
    def build_gpu(self):
        if len(self.gpus) >= 3:
            return
        warm_cpu = self.c._x_hat_rot is not None
        g = GPUSearcher(self.pq, self.c, backend="numpy")
        self.gpus.append(g)
        EV("gpuivf:gpu_built_on_warm_cpu_cache" if warm_cpu else "gpuivf:gpu_built_on_cold_cpu_cache")

    @rule(mode=st.sampled_from(["lsh", "rotated_prefix"]), n_bits=st.integers(1, 4), use_packed=st.booleans(),
          seed=st.sampled_from([0, 3]))
    def build_ivf(self, mode, n_bits, use_packed, seed):
        if len(self.ivfs) >= 4:
            return
        n_bits = min(n_bits, self.d)
        comp = self.packed if use_packed else self.c
        ivf = IVFCoarseIndex(self.pq, comp, n_bits=n_bits, mode=mode, seed=seed)
        self.ivfs.append((ivf, comp, mode, n_bits, seed))
        EV(f"gpuivf:ivf_{mode}_n_bits={n_bits}")
        EV("gpuivf:ivf_on_packed" if use_packed else "gpuivf:ivf_on_dense")
        self._check_ivf_structure(len(self.ivfs) - 1)

    # -- cache control ---------------------------------------------------------------
    @rule()
    def drop_cpu_cache(self):
        self.c.drop_cache()
        assert self.c._x_hat_rot is None and self.c._dir_len == {}
        EV("gpuivf:cpu_drop_cache")

    @rule(i=st.integers(0, 9))
    def drop_gpu_cache(self, i):
        if not self.gpus:
            return
        g = self.gpus[i % len(self.gpus)]
        g.drop_cache()
        assert g._x_hat_rot_gpu is None
        EV("gpuivf:gpu_drop_cache")

    @rule(prec=st.integers(0, 12), qsel=st.integers(0, 5), k=st.integers(1, 40))
    def warm_cpu(self, prec, qsel, k):
        self.pq.search(self.c, self.Q[qsel], k=k, precision=self._prec(prec))
        EV("gpuivf:cpu_warm_search")

    # -- checks -----------------------------------------------------------------------
    def _ref(self, qv, p):
        return self.oracle.scores(self.c, qv, p)

    @rule(i=st.integers(0, 9), method=st.sampled_from(["search", "search_adc", "search_twostage", "search_batch"]),
          prec=st.integers(0, 12), cprec=st.integers(0, 12), k=st.integers(1, 45), qsel=st.integers(0, 5),
          cand=st.integers(0, 3), chunk=st.sampled_from([1, 3, 4096]))
    def gpu_answer(self, i, method, prec, cprec, k, qsel, cand, chunk):
        if not self.gpus:
            return
        g = self.gpus[i % len(self.gpus)]
        qv, p, n = self.Q[qsel], self._prec(prec), self.n
        EV(f"gpuivf:gpu_{method}")
        EV("gpuivf:gpu_xhat_warm" if g._x_hat_rot_gpu is not None else "gpuivf:gpu_xhat_cold")
        if method == "search":
            ids, sc = g.search(qv, k=k)
            ref = self._ref(qv, None)
            check_topk(ref, ids, sc, k, tolerance(ref), what="GPUSearcher.search")
        elif method == "search_adc":
            ids, sc = g.search_adc(qv, k=k, precision=p, chunk_size=chunk)
            ref = self._ref(qv, p)
            check_topk(ref, ids, sc, k, tolerance(ref), what=f"GPUSearcher.search_adc p={p}")
        elif method == "search_batch":
            Qb = self.Q[: 1 + qsel % 3]
            ids, sc = g.search_batch(Qb, k=k)
            assert ids.shape == (len(Qb), min(k, n))
            for r_i in range(len(Qb)):
                ref = self._ref(Qb[r_i], None)
                check_topk(ref, ids[r_i], sc[r_i], k, tolerance(ref), what="GPUSearcher.search_batch")
        else:
            cp = self._prec(cprec)
            candidates = [1, max(1, n // 2), n, n + 7][cand]
            ref_f = self._ref(qv, None)
            ref_c = self._ref(qv, max(1, self.bits - 2) if cp is None else cp)
            ids, sc = g.search_twostage(qv, k=k, candidates=candidates, coarse_precision=cp)
            check_twostage(ref_f, ref_c, ids, sc, k, candidates, max(tolerance(ref_f), tolerance(ref_c)),
                           what="GPUSearcher.search_twostage")
            # and it must agree with the CPU two-stage on the same corpus
            ids2, sc2 = self.pq.search_twostage(self.c, qv, k=k, candidates=candidates, coarse_precision=cp)
            check_twostage(ref_f, ref_c, ids2, sc2, k, candidates, max(tolerance(ref_f), tolerance(ref_c)),
                           what="Quantizer.search_twostage")

    @rule(method=st.sampled_from(["search", "search_adc", "search_twostage", "search_batch"]),
          prec=st.integers(0, 12), k=st.integers(1, 45), qsel=st.integers(0, 5), cand=st.integers(0, 3),
          use_packed=st.booleans())
    def cpu_answer(self, method, prec, k, qsel, cand, use_packed):
        comp = self.packed if use_packed and method in ("search_adc", "search_twostage") else self.c
        qv, p, n = self.Q[qsel], self._prec(prec), self.n
        EV("gpuivf:cpu_xhat_warm" if self.c._x_hat_rot is not None else "gpuivf:cpu_xhat_cold")
        if method == "search":
            ids, sc = self.pq.search(comp, qv, k=k, precision=p)
            ref = self._ref(qv, p)
            check_topk(ref, ids, sc, k, tolerance(ref), what="Quantizer.search")
        elif method == "search_adc":
            ids, sc = self.pq.search_adc(comp, qv, k=k, precision=p)
            ref = self._ref(qv, p)
            check_topk(ref, ids, sc, k, tolerance(ref), what="Quantizer.search_adc")
        elif method == "search_batch":
            ids, sc = self.pq.search_batch(comp, self.Q[:2], k=k, precision=p)
            for r_i in range(2):
                ref = self._ref(self.Q[r_i], p)
                check_topk(ref, ids[r_i], sc[r_i], k, tolerance(ref), what="Quantizer.search_batch")
        else:
            candidates = [1, max(1, n // 2), n, n + 7][cand]
            ref_f = self._ref(qv, None)
            ref_c = self._ref(qv, max(1, self.bits - 2))
            ids, sc = self.pq.search_twostage(comp, qv, k=k, candidates=candidates)
            check_twostage(ref_f, ref_c, ids, sc, k, candidates, max(tolerance(ref_f), tolerance(ref_c)),
                           what="Quantizer.search_twostage")

    # -- IVF -----------------------------------------------------------------------------
    def _ivf_cells(self, ivf, comp, mode, n_bits, seed):
        codes = codes_of(comp)
        msb = (codes >> (self.bits - 1)).astype(np.int64)
        if mode == "rotated_prefix":
            return sum(msb[:, b] << b for b in range(n_bits))
        H = np.random.default_rng(seed).standard_normal((n_bits, self.d)).astype(np.float32)
        proj = (2.0 * msb.astype(np.float32) - 1.0) @ H.T
        return sum((proj[:, b] > 0).astype(np.int64) << b for b in range(n_bits))

    def _check_ivf_structure(self, j):
        ivf, comp, mode, n_bits, seed = self.ivfs[j]
        cells = self._ivf_cells(ivf, comp, mode, n_bits, seed)
        assert np.array_equal(ivf.cell_ids.astype(np.int64), cells), "IVF cell ids differ from an independent hash"
        assert ivf.n == comp.n and ivf.n_cells == 1 << n_bits
        assert ivf.cell_offsets[0] == 0 and ivf.cell_offsets[-1] == comp.n
        assert sorted(ivf.sorted_idx.tolist()) == list(range(comp.n)), "sorted_idx is not a permutation"
        for cell in range(ivf.n_cells):
            members = ivf.sorted_idx[ivf.cell_offsets[cell]:ivf.cell_offsets[cell + 1]]
            assert np.array_equal(members, np.flatnonzero(cells == cell)), f"cell {cell} holds the wrong rows"
        EV("gpuivf:ivf_structure_checked")

    @rule(j=st.integers(0, 9), qsel=st.integers(0, 5), nprobe=st.integers(0, 20), k=st.integers(1, 45),
          prec=st.integers(0, 12), chunk=st.sampled_from([1, 3, 4096]))
    def ivf_search_coarse(self, j, qsel, nprobe, k, prec, chunk):
        if not self.ivfs:
            return
        ivf, comp, mode, n_bits, seed = self.ivfs[j % len(self.ivfs)]
        qv, p = self.Q[qsel], self._prec(prec)
        cells = self._ivf_cells(ivf, comp, mode, n_bits, seed)
        q_cell = ivf.query_cell(qv)
        order = sorted(range(1 << n_bits), key=lambda c: (_popcount(c ^ q_cell), c))
        probed = set(order[:max(0, nprobe)]) if nprobe < (1 << n_bits) else set(range(1 << n_bits))
        allowed = {int(i) for i in np.flatnonzero(np.isin(cells, list(probed)))}
        got_cand = set(ivf.candidate_indices(q_cell, nprobe).tolist())
        assert got_cand == allowed, f"candidate set for nprobe={nprobe} differs from an independent derivation"
        assert ivf.candidate_count(qv, nprobe) == len(allowed)
        ids, sc = ivf.search_coarse(qv, k=k, nprobe=nprobe, precision=p, chunk_size=chunk)
        ref = self._ref(qv, p)
        check_topk(ref, ids, sc, k, tolerance(ref), allowed=allowed, what=f"IVF.search_coarse nprobe={nprobe}")
        EV(f"gpuivf:ivf_nprobe={'all' if nprobe >= (1 << n_bits) else ('0' if nprobe <= 0 else 'partial')}")
        if nprobe >= (1 << n_bits):
            ids2, sc2 = self.pq.search_adc(comp, qv, k=k, precision=p)
            ref_ids = ids2
            check_topk(ref, ref_ids, sc2, k, tolerance(ref), what="flat ADC")
            assert len(ids) == len(ids2)
            EV("gpuivf:ivf_full_probe_matches_flat_scan")

    @rule(j=st.integers(0, 9), qsel=st.integers(0, 5), nprobe=st.integers(1, 20), k=st.integers(1, 45),
          cprec=st.integers(0, 12), cand=st.integers(0, 3))
    def ivf_twostage(self, j, qsel, nprobe, k, cprec, cand):
        if not self.ivfs:
            return
        ivf, comp, mode, n_bits, seed = self.ivfs[j % len(self.ivfs)]
        qv, cp, n = self.Q[qsel], self._prec(cprec), self.n
        candidates = [1, max(1, n // 2), n, n + 7][cand]
        cells = self._ivf_cells(ivf, comp, mode, n_bits, seed)
        q_cell = ivf.query_cell(qv)
        order = sorted(range(1 << n_bits), key=lambda c: (_popcount(c ^ q_cell), c))
        probed = set(order[:nprobe]) if nprobe < (1 << n_bits) else set(range(1 << n_bits))
        allowed = {int(i) for i in np.flatnonzero(np.isin(cells, list(probed)))}
        ref_f = self._ref(qv, None)
        ref_c = self._ref(qv, max(1, self.bits - 2) if cp is None else cp)
        ids, sc = ivf.search_twostage(qv, k=k, candidates=candidates, nprobe=nprobe, coarse_precision=cp)
        check_twostage(ref_f, ref_c, ids, sc, k, candidates, max(tolerance(ref_f), tolerance(ref_c)),
                       allowed=allowed, what=f"IVF.search_twostage nprobe={nprobe}")
        if nprobe >= (1 << n_bits):
            flat_ids, flat_sc = self.pq.search_twostage(comp, qv, k=k, candidates=candidates, coarse_precision=cp)
            check_twostage(ref_f, ref_c, flat_ids, flat_sc, k, candidates, max(tolerance(ref_f), tolerance(ref_c)),
                           what="flat twostage")
        EV("gpuivf:ivf_twostage")

    @rule(i=st.integers(0, 9), pick=st.integers(0, 3))
    def invalid_precision_is_refused(self, i, pick):
        bad = (0, -1, self.bits + 1, self.bits + 5)[pick]
        qv = self.Q[0]
        if self.gpus:
            g = self.gpus[i % len(self.gpus)]
            with pytest.raises(ValueError):
                g.search_adc(qv, k=3, precision=bad)
            with pytest.raises(ValueError):
                g.search_twostage(qv, k=3, coarse_precision=bad)
        if self.ivfs:
            ivf = self.ivfs[i % len(self.ivfs)][0]
            with pytest.raises(ValueError):
                ivf.search_coarse(qv, k=3, nprobe=1 << 4, precision=bad)
            with pytest.raises(ValueError):
                ivf.search_twostage(qv, k=3, nprobe=1 << 4, coarse_precision=bad)
        EV("gpuivf:invalid_precision_refused")

    @rule(kind=st.sampled_from(["bits", "seed", "d", "rotation"]), which=st.sampled_from(["gpu", "ivf"]))
    def foreign_container_refused_at_build(self, kind, which):
        """GPUSearcher / IVF search refuse codes another encoding wrote."""
        if kind in ("seed", "rotation") and self.params["rotation"] == "none":
            return
        p = dict(self.params)
        if kind == "bits":
            p["bits"] = 2 if self.bits != 2 else 3
        elif kind == "seed":
            p["seed"] = self.params["seed"] + 1
        elif kind == "rotation":
            p["rotation"] = "haar" if self.oracle.pq.rotation != "haar" else "none"
        else:
            p["d"] = self.d + 2
            if "mean" in p:
                p["mean"] = np.full(p["d"], 0.25, np.float32)
        owner = Quantizer(**p)
        c = owner.encode(np.random.default_rng(3).standard_normal((6, p["d"])).astype(np.float32)
                         * (p.get("scale") or 1.0))
        owner.search(c, np.ones(p["d"], np.float32), k=2)             # warm it
        if which == "gpu":
            with pytest.raises(ValueError):
                GPUSearcher(self.pq, c, backend="numpy")
        else:
            # IVFCoarseIndex.__init__ runs none of the quantizer's guards (finding F5): for some
            # mismatches it fails obscurely while hashing, for others it builds a garbage index.
            # What must hold is that no *answer* comes out of it.
            try:
                ivf = IVFCoarseIndex(self.pq, c, n_bits=1, mode="rotated_prefix") if kind != "d" else None
            except ValueError:
                ivf = None
            if ivf is not None:
                with pytest.raises(ValueError):
                    ivf.search_coarse(self.Q[0], k=2, nprobe=2)
        EV(f"gpuivf:foreign_{kind}_{which}_refused")


TestGpuIvfMachine = GpuIvfMachine.TestCase
TestGpuIvfMachine.settings = machine_settings()


# --------------------------------------------------------------------------
# Plain properties
# --------------------------------------------------------------------------

from hypothesis import given  # noqa: E402


@settings(max_examples=40, deadline=None, database=None, suppress_health_check=list(HealthCheck))
@given(d=st.sampled_from([4, 6, 12, 24, 48, 96]), n=st.integers(1, 64), seed=st.sampled_from([0, 1, 42]),
       threads=st.sampled_from([1, 2, 3, 4]), threshold=st.sampled_from([1, 64, 1 << 20]),
       blocks=st.sampled_from([1, 7, 1 << 30]), x_seed=st.integers(0, 10 ** 6))
def test_rht_encode_is_independent_of_thread_pool_history(d, n, seed, threads, threshold, blocks, x_seed):
    """The RHT operator's thread pool, parallel threshold and encode block size are history
    (module-level state); codes and norms must not depend on any of them."""
    import remex.core as core
    import remex.rotation as rot
    X = np.random.default_rng(x_seed).standard_normal((n, d)).astype(np.float32)
    ref = Quantizer(d, 4, seed=seed, rotation="rht").encode(X)
    old_t, old_min, old_blk = rot._num_threads, rot.PARALLEL_MIN_FLOATS, core.ENCODE_BLOCK_FLOATS
    try:
        rot.set_num_threads(threads)
        rot.PARALLEL_MIN_FLOATS = threshold
        core.ENCODE_BLOCK_FLOATS = blocks * d
        got = Quantizer(d, 4, seed=seed, rotation="rht").encode(X)
    finally:
        rot._num_threads, rot.PARALLEL_MIN_FLOATS, core.ENCODE_BLOCK_FLOATS = old_t, old_min, old_blk
    assert np.array_equal(ref.indices, got.indices) and np.array_equal(ref.norms, got.norms)
    EV(f"plain:rht_threads={threads}_threshold={'tiny' if threshold == 1 else 'default-ish'}")


@pytest.mark.skipif(not MUTANTS, reason="set REMEX_PROP_MUTANTS=1 (or run calibrate_mutants.py)")
@pytest.mark.parametrize("mutant_id", ["M01", "M02", "M05", "M07", "M08", "M09", "M10", "M12", "M13", "M15", "M16", "M17"])
def test_mutant_is_caught(mutant_id):
    """Harness calibration: each planted defect must make its machine fail."""
    import calibrate_mutants as C
    m = next(x for x in C.MUTANTS if x["id"] == mutant_id)
    with m["patch"]():
        r = C.run_machine(C.MACHINES[m["machine"]], examples=200, steps=25)
    assert r["caught"], f"{mutant_id} ({m['what']}) survived {r['examples']} examples"


@settings(max_examples=40, deadline=None, database=None, suppress_health_check=list(HealthCheck))
@given(cfg=CONFIG, blocks=st.sampled_from([1, 2, 5, 1 << 20]))
def test_encode_is_independent_of_row_block_size(cfg, blocks):
    """``ENCODE_BLOCK_FLOATS`` only bounds peak memory: codes, norms and centred mean_dots must
    not depend on it, in every mode (plain, centred, scalar) and rotation."""
    import remex.core as core
    r = resolve_config(cfg)
    ref = Quantizer(**r["params"]).encode(r["X"])
    old = core.ENCODE_BLOCK_FLOATS
    try:
        core.ENCODE_BLOCK_FLOATS = blocks * r["params"]["d"]
        got = Quantizer(**r["params"]).encode(r["X"])
    finally:
        core.ENCODE_BLOCK_FLOATS = old
    frac = float((ref.indices != got.indices).mean())
    assert frac <= (5e-4 if ref.rotation == "haar" else 0.0), f"{frac:.2%} of codes depend on the row block size"
    if ref.norms is not None:
        assert np.array_equal(ref.norms, got.norms)
    if ref.mean_dots is not None:
        assert np.array_equal(ref.mean_dots, got.mean_dots)
    EV(f"plain:encode_block_rows={'all' if blocks > 100 else blocks}")


@pytest.mark.parametrize("n", [4095, 4096, 4097, 8192, 8193])
@pytest.mark.parametrize("mode", ["plain", "centered", "scalar"])
def test_chunk_boundaries_of_direction_lengths_and_adc(n, mode):
    """The machines use n <= 40, so the 4096-row chunks in ``_direction_lengths``,
    ``_direction_mean_dots``, ``_adc_score_*`` and IVF hashing never split a container.  Cross them:
    dense and packed containers at n = 4095..8193 against the cache-free reference, at two precisions."""
    d, bits = 8, 4
    rng = np.random.default_rng(n)
    X = rng.standard_normal((n, d)).astype(np.float32) + (2.0 if mode == "centered" else 0.0)
    params = dict(d=d, bits=bits, seed=1)
    if mode == "centered":
        params["mean"] = corpus_mean(X)
    elif mode == "scalar":
        params.update(normalize=False)
    q = Quantizer(**params)
    oracle = Oracle(params)
    c = q.encode(X)
    pv = PackedVectors.from_compressed(c)
    qv = rng.standard_normal(d).astype(np.float32)
    for p in (None, 2):
        ref = oracle.scores(c, qv, p)
        for cont in (c, pv):
            ids, sc = q.search_adc(cont, qv, k=7, precision=p, chunk_size=4096 if cont is c else 1000)
            check_topk(ref, ids, sc, 7, tolerance(ref), what=f"{type(cont).__name__}.search_adc n={n} p={p}")
        ids, sc = q.search(c, qv, k=7, precision=p)
        check_topk(ref, ids, sc, 7, tolerance(ref), what=f"search n={n} p={p}")
    ivf = IVFCoarseIndex(q, pv, n_bits=3, mode="lsh")
    assert ivf.cell_ids.shape == (n,) and int(ivf.cell_offsets[-1]) == n
    EV(f"plain:chunk_boundary_n={n}")
