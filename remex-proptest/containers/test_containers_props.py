"""Property-based bug hunt on remex area B: containers, serialization,
Matryoshka precision and bit packing.

Run:   pytest test_containers_props.py -q --hypothesis-show-statistics
Time:  about 2 minutes on one core (all 27 properties + 7 xfail-strict known findings;
       measured 133 s wall at load average 7 on 4 cores, fresh example database).

Method (proptest-praxis): independent reference models, operation-sequence
tests, near-valid / malformed inputs, boundary-value analysis, metamorphic
equivalence between API paths, and mutation calibration (see mutants.py).

Layout
------
1. reference models   (ref_pack / ref_unpack / ref_nbytes / index resolution)
2. generators         (cfgs -> quantizer configuration; datasets; selectors; ops)
3. state + checkers   (Held: container + model; structure and functional checks)
4. properties         (test_*)
5. KNOWN findings     (test_known_*, xfail(strict=True) so a fix flips them to XPASS)

Finding ids (full write-ups and deterministic repros: test_containers_regressions.py)
  F1 .npz packed layout differs between CompressedVectors.save and PackedVectors.save
  F2 Quantizer.search skips the bits/mean guards on a warm container (scalar / renorm=False)
  F3 .npz cannot record a seed >= 2**63
  F4 save_arrow(**extra_metadata) can overwrite reserved keys
  F5 PackedVectors.load of a legacy indices-only .npz leaks KeyError
  F6 a damaged (not truncated) .npz member leaks BadZipFile / zlib.error / ...
  A1.. ambiguous behaviours (regressions file, class TestAmbiguous)

Mutation calibration:  python mutants.py all      (see mutation_results.txt)
Event counts:          pytest ... --hypothesis-show-statistics   (see coverage_events.txt)

Where a property exposes a finding its domain is *restricted* in the main test
(with a comment naming the finding) and the failing region is exercised by a
dedicated ``test_known_*`` property marked xfail(strict=True).  When remex is
fixed the xfail turns into a strict XPASS failure, which is the cue to delete
the marker and widen the main property.
"""
from __future__ import annotations

import functools
import io
import os
import random
import struct
import sys
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, event, given, settings, strategies as st

import remex
from remex import (
    CompressedVectors, PackedVectors, Quantizer, load_pq, pack, packed_nbytes,
    save_params, save_pq, unpack,
)

pytest.importorskip("pyarrow")

warnings.filterwarnings("ignore", category=remex.AnisotropyWarning)

HERE = Path(__file__).resolve().parent
SUPPORTED = (1, 2, 3, 4, 8)

# One profile for everything: no per-example deadline (codebook builds and
# file I/O are slow and noisy), failure persistence on (default example DB, in
# ./.hypothesis next to wherever pytest is launched from), and the health
# checks that the per-example tmp dir and expensive generators would trip.
SUPPRESS = [
    HealthCheck.too_slow,
    HealthCheck.data_too_large,
    HealthCheck.function_scoped_fixture,
    HealthCheck.filter_too_much,
]


def prop(max_examples: int, **kw):
    return settings(max_examples=max_examples, deadline=None,
                    suppress_health_check=SUPPRESS, **kw)


# The test_known_* properties only need to fail at least once (strict xfail); shrinking
# and the explain phase would just spend minutes minimising an already-known bug.
NO_SHRINK = dict(phases=[Phase.explicit, Phase.reuse, Phase.generate])


# ---------------------------------------------------------------------------
# 1. Reference models (independent of remex.packing / remex.core)
# ---------------------------------------------------------------------------

_GROUP = {1: (8, 1), 2: (4, 1), 3: (8, 3), 4: (2, 1), 8: (1, 1)}  # values, bytes


def ref_nbytes(total_values: int, bits: int) -> int:
    """Bytes for a flat run of ``total_values`` codes: whole groups only."""
    vals, nb = _GROUP[bits]
    return -(-total_values // vals) * nb


def ref_pack(values, bits: int) -> bytes:
    """MSB-first bit stream, zero-padded to a whole group: a big-endian integer."""
    vals = [int(v) for v in values]
    gvals, _ = _GROUP[bits]
    vals += [0] * ((-len(vals)) % gvals)
    acc = 0
    for v in vals:
        assert 0 <= v < (1 << bits)
        acc = (acc << bits) | v
    nbytes = len(vals) * bits // 8
    return acc.to_bytes(nbytes, "big")


def ref_unpack(buf: bytes, bits: int, n_values: int) -> list:
    acc = int.from_bytes(buf, "big")
    total = len(buf) * 8
    assert n_values * bits <= total
    mask = (1 << bits) - 1
    return [(acc >> (total - (i + 1) * bits)) & mask for i in range(n_values)]


def resolve_rows(idx, n: int) -> list:
    """Documented ``subset`` semantics, re-derived without numpy indexing:
    ints may be negative; a bool mask must have length n; anything out of
    range raises IndexError."""
    items = list(idx)
    if items and all(isinstance(i, (bool, np.bool_)) for i in items):
        if len(items) != n:
            raise IndexError("mask length")
        return [i for i, b in enumerate(items) if b]
    out = []
    for i in items:
        i = int(i)
        j = i + n if i < 0 else i
        if not (0 <= j < n):
            raise IndexError("out of range")
        out.append(j)
    return out


# ---------------------------------------------------------------------------
# 2. Generators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cfg:
    d: int
    bits: int
    seed: int
    rotation: str
    normalize: bool
    scale: Optional[float]     # declared scale (scalar mode only)
    renorm: bool
    mean_kind: str             # none | zero | unit | big
    mean_seed: int = 0

    @property
    def mean(self):
        if self.mean_kind == "none":
            return None
        if self.mean_kind == "zero":
            return np.zeros(self.d, np.float32)
        rng = np.random.default_rng(self.mean_seed)
        m = rng.standard_normal(self.d)
        if self.mean_kind == "big":
            m = m * 40.0 + 15.0
        # keep |mean| comfortably above the guard's 1e-6 comparison tolerance
        m = np.where(np.abs(m) < 0.5, np.sign(m + 1e-9) * 0.5, m)
        return m.astype(np.float32)

    @property
    def row_aligned(self):
        return (self.d * self.bits) % 8 == 0

    def replace(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return Cfg(**d)


@functools.lru_cache(maxsize=None)
def make_q(cfg: Cfg, bits: Optional[int] = None) -> Quantizer:
    c = cfg if bits is None else cfg.replace(bits=bits)
    return Quantizer(c.d, c.bits, seed=c.seed, rotation=c.rotation,
                     normalize=c.normalize, scale=c.scale, renorm=c.renorm,
                     mean=c.mean)


INTERESTING_D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 24, 31, 32, 33, 48, 63, 64]
SEEDS = [0, 1, 42, 7, 2**31 - 1, 2**32, 2**63 - 1]


@st.composite
def cfgs(draw, *, scalar=True, mean=True, max_d=64, bits=None, seeds=None,
         unaligned=None, rotation=None, normalize=None):
    rot = draw(st.sampled_from(["haar", "rht", "none"])) if rotation is None else rotation
    d = draw(st.one_of(st.sampled_from(INTERESTING_D), st.integers(1, max_d)))
    d = min(d, max_d)
    if rot == "rht" and d % 2:
        d += 1 if d < max_d else -1
    b = draw(st.sampled_from(SUPPORTED)) if bits is None else bits
    if unaligned is True and (d * b) % 8 == 0:
        # shift d to the nearest value whose rows are not byte-aligned (rht needs even d)
        for dd in range(d, d + 9):
            if (dd * b) % 8 != 0 and not (rot == "rht" and dd % 2):
                d = dd
                break
        else:
            d, b = (6 if rot == "rht" else 3), 3
    seed = draw(st.one_of(st.sampled_from(SEEDS), st.integers(0, 2**31))) \
        if seeds is None else draw(seeds)
    norm_mode = draw(st.sampled_from([True, True, True, False])) if normalize is None else normalize
    if not scalar:
        norm_mode = True
    if not norm_mode:
        scale = draw(st.sampled_from([0.05, 0.5, 1.0, 3.0, 17.0]))
        return Cfg(d, b, seed, rot, False, scale, draw(st.booleans()), "none")
    renorm = draw(st.sampled_from([True, True, True, True, True, False]))
    mk = "none"
    if mean and renorm:
        mk = draw(st.sampled_from(["none", "none", "zero", "unit", "unit", "unit", "big", "big"]))
    return Cfg(d, b, seed, rot, True, None, renorm, mk,
               draw(st.integers(0, 50)))


def ev_cfg(cfg: Cfg, n: int):
    event(f"rot:{cfg.rotation}")
    event(f"bits:{cfg.bits}")
    event("mode:scalar" if not cfg.normalize else "mode:normalized")
    event(f"mean:{cfg.mean_kind}")
    event(f"renorm:{'on' if cfg.renorm else 'off'}")
    event("layout:row-aligned" if cfg.row_aligned else "layout:UNALIGNED-rows")
    event("n:0" if n == 0 else "n:1" if n == 1 else "n:many")
    event("d:1" if cfg.d == 1 else "d:odd" if cfg.d % 2 else "d:even")


@st.composite
def datasets(draw, cfg: Cfg, n=None):
    """(X, n): small float matrices of several structural classes."""
    n = draw(st.one_of(st.just(0), st.just(1), st.integers(2, 40),
                       st.integers(2, 12), st.integers(3, 40))) if n is None else n
    kind = draw(st.sampled_from(
        ["gauss", "gauss", "gauss", "zeros", "dups", "onehot", "big", "tiny",
         "sparse", "mixed"] + ([] if cfg.normalize else ["extreme"])))
    rng = np.random.default_rng(draw(st.integers(0, 2**31)))
    d = cfg.d
    spread = 1.0 if cfg.normalize else (cfg.scale or 1.0)
    X = rng.standard_normal((n, d)) * spread
    if kind == "zeros":
        X[:] = 0.0
    elif kind == "dups" and n:
        X[:] = X[0]
    elif kind == "onehot":
        X[:] = 0.0
        for i in range(n):
            X[i, i % d] = spread * (1.0 if i % 2 == 0 else -1.0)
    elif kind == "big":
        X *= 1e20
    elif kind == "tiny":
        X *= 1e-20
    elif kind == "sparse":
        X[rng.random((n, d)) < 0.7] = 0.0
    elif kind == "mixed" and n:
        X[rng.random(n) < 0.3] = 0.0
        X[rng.random(n) < 0.3] *= 1e8
    elif kind == "extreme" and n:
        flat = X.ravel()
        flat[rng.random(flat.size) < 0.15] = rng.choice(
            [1e300, -1e300, np.inf, -np.inf, np.nan], size=int((rng.random(flat.size) < 0.15).sum() or 1))[0]
    event(f"data:{kind}")
    return X


def encode(cfg: Cfg, X) -> CompressedVectors:
    with np.errstate(all="ignore"):
        return make_q(cfg).encode(X)


@st.composite
def hand_cvs(draw):
    """Containers built without a Quantizer: arbitrary uint8 codes, norms,
    mean, mean_dots, seed, scale.  Reaches states encode() rarely makes
    (all codes equal, extreme codes, odd n/d/bits combos, hand-set metadata)."""
    bits = draw(st.sampled_from(SUPPORTED))
    d = draw(st.one_of(st.sampled_from(INTERESTING_D), st.integers(1, 40)))
    n = draw(st.one_of(st.just(0), st.just(1), st.integers(2, 25)))
    rng = np.random.default_rng(draw(st.integers(0, 2**31)))
    style = draw(st.sampled_from(["rand", "zeros", "max", "alt"]))
    hi = 1 << bits
    if style == "rand":
        idx = rng.integers(0, hi, (n, d))
    elif style == "zeros":
        idx = np.zeros((n, d), int)
    elif style == "max":
        idx = np.full((n, d), hi - 1)
    else:
        idx = np.tile(np.array([0, hi - 1]), (n, d))[:, :d]
    scalar = draw(st.booleans())
    centred = (not scalar) and draw(st.booleans())
    norms = None if scalar else rng.random(n).astype(np.float32) * 10
    mean = rng.standard_normal(d).astype(np.float32) if centred else None
    md = rng.standard_normal(n).astype(np.float32) if centred else None
    rot = draw(st.sampled_from(["haar", "rht", "none"]))
    seed = draw(st.one_of(st.none(), st.sampled_from(SEEDS)))
    scale = draw(st.sampled_from([0.5, 1.0, 3.25])) if scalar else None
    event(f"hand:bits={bits}")
    event("hand:scalar" if scalar else "hand:centred" if centred else "hand:plain")
    event("hand:aligned" if (d * bits) % 8 == 0 else "hand:UNALIGNED")
    return CompressedVectors(idx.astype(np.uint8), norms, d, bits, rot, mean, md, seed, scale)


# --- subset selectors --------------------------------------------------------

VALID_FORMS = ["ints", "ints", "neg", "perm", "all", "head", "bool", "boollist",
               "empty_list", "empty_arr", "empty_mask", "dups", "tuple", "range"]
# in operation sequences the empties are drawn rarely so later ops still see data
SEQ_FORMS = ["ints", "ints", "ints", "neg", "perm", "perm", "all", "all", "head", "bool",
             "boollist", "dups", "tuple", "range", "empty_arr"]
INVALID_FORMS = ["oob_hi", "oob_lo", "mask_long", "mask_short"]


def build_selector(form: str, k: int, n: int):
    """(selector, expected_rows or None for must-raise).  Deterministic in
    (form, k, n): shrinking an op never changes how it is interpreted."""
    r = random.Random(k)
    if form == "ints":
        sel = [r.randint(-n, n - 1) for _ in range(r.randint(1, 2 * n))] if n else []
        sel = np.asarray(sel, dtype=np.int64) if r.random() < 0.5 else sel
    elif form == "neg":
        sel = np.asarray([-1 - r.randrange(n) for _ in range(r.randint(1, 4))]) if n else []
    elif form == "perm":
        p = list(range(n)); r.shuffle(p); sel = np.asarray(p, dtype=np.int64)
    elif form == "all":
        sel = np.arange(n)
    elif form == "head":
        sel = np.arange(0, r.randint(1, n)) if n else np.arange(0)
    elif form == "range":
        sel = range(r.randint(1, n)) if n else range(0)
    elif form == "bool":
        sel = np.asarray([r.random() < 0.6 for _ in range(n)], dtype=bool)
    elif form == "boollist":
        sel = [r.random() < 0.6 for _ in range(n)]
    elif form == "empty_list":
        sel = []
    elif form == "empty_arr":
        sel = np.array([], dtype=np.int64)
    elif form == "empty_mask":
        sel = np.zeros(n, dtype=bool)
    elif form == "dups":
        sel = [r.randrange(n)] * r.randint(1, 4) if n else []
    elif form == "tuple":
        sel = tuple(r.randrange(n) for _ in range(r.randint(1, 3))) if n else ()
    elif form == "oob_hi":
        sel = [n + r.randint(0, 3)]
    elif form == "oob_lo":
        sel = [-n - 1 - r.randint(0, 3)]
    elif form == "mask_long":
        sel = np.ones(n + 1, dtype=bool)
    elif form == "mask_short":
        sel = np.ones(max(n - 1, 0), dtype=bool) if n else np.ones(1, dtype=bool)
        if n == 0:
            sel = np.ones(1, dtype=bool)
    else:
        raise AssertionError(form)
    if form in INVALID_FORMS:
        return sel, None
    if form in ("bool", "boollist", "empty_mask"):
        return sel, [i for i, b in enumerate(list(sel)) if b]
    if form in ("empty_list", "empty_arr"):
        return sel, []
    return sel, resolve_rows(sel, n)


# --- operations --------------------------------------------------------------

OP_KINDS = ["subset", "subset", "subset", "warm", "warm", "convert", "rows", "npz",
            "arrow", "pq", "at_precision", "unpack", "drop_cache", "check"]


@st.composite
def op_lists(draw, max_len=8):
    out = []
    for _ in range(draw(st.integers(1, max_len))):
        kind = draw(st.sampled_from(OP_KINDS))
        form = draw(st.sampled_from(SEQ_FORMS + INVALID_FORMS[:2])) if kind == "subset" \
            else draw(st.sampled_from(VALID_FORMS[:8]))
        out.append((kind, form, draw(st.integers(0, 10**6)), draw(st.integers(0, 3))))
    return out


# ---------------------------------------------------------------------------
# 3. State + checkers
# ---------------------------------------------------------------------------

def f32bits(a):
    return None if a is None else np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def same_f32(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and np.array_equal(f32bits(a), f32bits(b))


@dataclass
class Model:
    """Plain-numpy ground truth for a container, updated by explicit row lists."""
    idx: np.ndarray
    norms: Optional[np.ndarray]
    mean: Optional[np.ndarray]
    mean_dots: Optional[np.ndarray]
    d: int
    bits: int
    rotation: str
    seed: Optional[int]
    scale: Optional[float]

    @classmethod
    def of(cls, cv):
        return cls(cv.indices.copy(), None if cv.norms is None else cv.norms.copy(),
                   None if cv.mean is None else cv.mean.copy(),
                   None if cv.mean_dots is None else cv.mean_dots.copy(),
                   cv.d, cv.bits, cv.rotation, cv.seed, cv.scale)

    @property
    def n(self):
        return self.idx.shape[0]

    def rows(self, rows):
        rows = list(rows)
        idx = np.empty((len(rows), self.d), np.uint8)
        for o, i in enumerate(rows):
            idx[o] = self.idx[i]
        pick = lambda a: None if a is None else np.asarray([a[i] for i in rows], np.float32)
        out = Model(idx, pick(self.norms), self.mean, pick(self.mean_dots),
                    self.d, self.bits, self.rotation, self.seed, self.scale)
        return out

    def fresh(self) -> CompressedVectors:
        return CompressedVectors(
            self.idx.copy(), None if self.norms is None else self.norms.copy(),
            self.d, self.bits, self.rotation,
            None if self.mean is None else self.mean.copy(),
            None if self.mean_dots is None else self.mean_dots.copy(),
            self.seed, self.scale)


class Held:
    """A container under test plus its model and the quantizer config."""

    def __init__(self, obj, model: Model, cfg: Cfg, qry, tmp: Path):
        self.obj, self.model, self.cfg, self.qry, self.tmp = obj, model, cfg, qry, tmp
        self.log = []

    @property
    def is_cv(self):
        return isinstance(self.obj, CompressedVectors)

    def q(self) -> Quantizer:
        return make_q(self.cfg, self.model.bits)

    def path(self, suffix):
        return self.tmp / f"{uuid.uuid4().hex[:10]}{suffix}"


def packed_rows_of(obj: PackedVectors):
    return [obj._packed[i].tobytes() for i in range(obj.n)]


def check_structure(h: Held):
    o, m = h.obj, h.model
    tag = " | ".join(h.log[-3:])
    assert o.n == m.n, f"n {o.n} != {m.n} after {tag}"
    assert (o.d, o.bits) == (m.d, m.bits), tag
    assert o.rotation == m.rotation, tag
    assert o.seed == m.seed, f"seed {o.seed} != {m.seed} after {tag}"
    assert o.scale == m.scale, f"scale {o.scale} != {m.scale} after {tag}"
    assert same_f32(o.mean, m.mean), f"mean differs after {tag}"
    assert same_f32(o.mean_dots, m.mean_dots), f"mean_dots differs after {tag}"
    assert same_f32(o.norms, m.norms), f"norms differ after {tag}"
    assert (o.norms is None) == (not o.has_norms)
    n, d, bits = m.n, m.d, m.bits
    extra = (0 if m.norms is None else 4 * n) + (0 if m.mean_dots is None else 4 * n)
    if h.is_cv:
        assert o.indices.dtype == np.uint8 and o.indices.shape == (n, d), tag
        assert np.array_equal(o.indices, m.idx), f"indices differ after {tag}"
        assert o.nbytes == ref_nbytes(n * d, bits) + extra, (o.nbytes, tag)
        assert o.nbytes_unpacked == n * d + extra
        xhat = 0 if o._x_hat_rot is None else o._x_hat_rot.nbytes
        assert o.resident_bytes == n * d + extra + xhat
    else:
        rb = ref_nbytes(d, bits)
        assert o._packed.shape == (n, rb), (o._packed.shape, n, rb, tag)
        got = o.unpack_rows(0, n)
        assert got.dtype == np.uint8 and got.shape == (n, d)
        assert np.array_equal(got, m.idx), f"unpacked rows differ after {tag}"
        # the packed bytes themselves, against an independent bit-stream packer
        for i in range(n):
            assert o._packed[i].tobytes() == ref_pack(m.idx[i], bits), \
                f"row {i} packed layout wrong after {tag}"
        assert o.nbytes == n * rb + extra, (o.nbytes, tag)
        assert o.resident_bytes == o.nbytes
    if n > 0:
        assert o.compression_ratio == (n * d * 4) / o.nbytes


def eq_result(a, b, what, tag):
    assert np.array_equal(a[0], b[0]), f"{what}: indices differ ({tag})\n{a[0]}\n{b[0]}"
    assert np.array_equal(a[1], b[1]), f"{what}: scores differ ({tag})\n{a[1]}\n{b[1]}"


def check_functional(h: Held, precision=None):
    """Search/decode on the held container must equal the same call on a fresh
    container built from the model: cached state carried by subset / reloads /
    conversions must give the answers a cold container gives."""
    if h.cfg is None:            # hand-built container: no matching Quantizer, structure only
        return
    o, m = h.obj, h.model
    q = h.q()
    fresh = m.fresh()
    qry = h.qry
    tag = " | ".join(h.log[-4:])
    k = 5
    with np.errstate(all="ignore"):
        if h.is_cv:
            for p in (None, precision):
                if p is not None and not (1 <= p <= m.bits):
                    continue
                a, b = q.decode(o, precision=p), q.decode(fresh, precision=p)
                assert np.array_equal(a, b, equal_nan=True), f"decode(p={p}) differs ({tag})"
                eq_result(q.search(o, qry, k=k, precision=p), q.search(fresh, qry, k=k, precision=p), f"search p={p}", tag)
                eq_result(q.search_adc(o, qry, k=k, precision=p), q.search_adc(fresh, qry, k=k, precision=p), f"adc p={p}", tag)
                eq_result(q.search_batch(o, qry[None], k=k, precision=p), q.search_batch(fresh, qry[None], k=k, precision=p), f"batch p={p}", tag)
            eq_result(q.search_twostage(o, qry, k=k, candidates=7),
                      q.search_twostage(fresh, qry, k=k, candidates=7), "twostage", tag)
        else:
            for p in (None, precision):
                if p is not None and not (1 <= p <= m.bits):
                    continue
                eq_result(q.search_adc(o, qry, k=k, precision=p), q.search_adc(fresh, qry, k=k, precision=p), f"adc(PV) p={p}", tag)
            eq_result(q.search_twostage(o, qry, k=k, candidates=7),
                      q.search_twostage(fresh, qry, k=k, candidates=7), "twostage(PV)", tag)


def to_cv(h: Held):
    if h.is_cv:
        return
    h.obj = h.obj.to_compressed()
    h.log.append("to_compressed")


def to_pv(h: Held):
    if not h.is_cv:
        return
    h.obj = PackedVectors.from_compressed(h.obj)
    h.log.append("from_compressed")


def apply_op(h: Held, op):
    kind, form, k, small = op
    m = h.model
    n = m.n
    d, bits = m.d, m.bits
    h.log.append(f"{kind}:{form}:{k}")
    event(f"op:{kind}")

    if kind == "subset":
        sel, rows = build_selector(form, k, n)
        event(f"subset:{form}")
        if rows is None:
            before = h.obj
            with pytest.raises(IndexError):
                h.obj.subset(sel)
            assert h.obj is before          # a failed subset leaves the container alone
            check_structure(h)
            return
        h.obj = h.obj.subset(sel)
        h.model = m.rows(rows)
        if n and rows:
            event("subset:nonempty-of-nonempty")
        if len(rows) == 0:
            event("subset:to-empty")

    elif kind == "warm":
        event("warm:cache-populated")
        h.log.append("warm")
        check_functional(h)

    elif kind == "convert":
        if h.is_cv:
            to_pv(h)
        else:
            to_cv(h)

    elif kind == "rows":
        # PackedVectors.from_rows fed with independently packed per-row bytes
        to_pv(h)
        rows = [ref_pack(m.idx[i], bits) for i in range(n)]
        if n == 0:
            event("rows:n=0 (from_rows cannot build empty; uses subset)")
            return                                  # from_rows([]) is known finding A9
        flavour = small % 3
        rows = [r if flavour == 0 else bytearray(r) if flavour == 1
                else np.frombuffer(r, np.uint8) for r in rows]
        h.obj = PackedVectors.from_rows(
            rows, m.norms, d, bits, m.rotation, m.mean, m.mean_dots, m.seed, m.scale)

    elif kind == "npz":
        form_ = small % 4
        event(f"npz:path-form-{form_}")
        cls = type(h.obj)
        if form_ == 3:
            buf = io.BytesIO()
            h.obj.save(buf)
            buf.seek(0)
            h.obj = cls.load(buf)
        else:
            p = h.path("" if form_ == 1 else ".npz")
            if form_ == 2:
                p = Path(str(p))
            written = h.obj.save(p if form_ == 2 else str(p))
            assert os.fspath(written).endswith(".npz")
            assert os.path.exists(written)
            h.obj = cls.load(str(p) if form_ != 2 else p)   # no-suffix path must resolve too
            assert not [f for f in os.listdir(h.tmp) if f.endswith(".tmp") or ".tmp" in f]

    elif kind == "arrow":
        reader_cv = bool(small & 1)
        mmap = bool(small & 2)
        event(f"arrow:{'cv' if h.is_cv else 'pv'}->{'cv' if reader_cv else 'pv'}"
              f"{'+mmap' if mmap and not reader_cv else ''}")
        p = str(h.path(".arrow"))
        seed_arg = m.seed if (k % 2 == 0) else None
        h.obj.save_arrow(p, seed=seed_arg)
        h.obj = (CompressedVectors.load_arrow(p) if reader_cv
                 else PackedVectors.load_arrow(p, memory_map=mmap))
        if mmap and not reader_cv:
            os.remove(p)       # a memory-mapped load must not depend on the file afterwards

    elif kind == "pq":
        to_cv(h)
        p = str(h.path(".pq"))
        if m.mean is not None:
            event("pq:centred-refused")
            with pytest.raises(ValueError):
                save_pq(p, h.obj)
            assert not os.path.exists(p), "refused save_pq left a file behind"
            return
        save_pq(p, h.obj)
        h.obj = load_pq(p)
        # .pq has no field for seed or scale: documented, they come back unrecorded
        h.model = Model(m.idx, m.norms, None, None, d, bits, m.rotation, None, None)
        event("pq:roundtrip")

    elif kind == "at_precision":
        to_pv(h)
        valid = [t for t in SUPPORTED if t < bits]
        if k % 5 == 0 or not valid:
            bad = [0, bits + 1, -1][small % 3]
            event("at_precision:invalid-raises")
            with pytest.raises(ValueError):
                h.obj.at_precision(bad)
            check_structure(h)
            return
        tgt = valid[k % len(valid)]
        event(f"at_precision:{bits}->{tgt}")
        shift = bits - tgt
        new = h.obj.at_precision(tgt)
        assert new is not h.obj
        h.obj = new
        h.model = Model((m.idx >> shift).astype(np.uint8), m.norms, m.mean, m.mean_dots,
                        d, tgt, m.rotation, m.seed, m.scale)
        if k % 7 == 0:
            same = h.obj.at_precision(tgt)       # at_precision(bits) is the identity
            assert same.bits == tgt and same.n == n

    elif kind == "unpack":
        to_pv(h)
        o = h.obj
        if n:
            a = k % (n + 1)
            b = a + (small + k) % (n - a + 1)
            assert np.array_equal(o.unpack_rows(a, b), m.idx[a:b]), f"unpack_rows({a},{b})"
            event("unpack:rows-slice")
        assert o.unpack_rows(0, 0).shape == (0, d)
        sel, rows = build_selector(["ints", "perm", "dups", "neg", "empty_arr"][small % 5], k, n)
        if rows is not None:
            got = o.unpack_at(np.asarray(sel, dtype=np.int64) if len(sel) else np.array([], dtype=np.int64))
            exp = np.empty((len(rows), d), np.uint8)
            for t, i in enumerate(rows):
                exp[t] = m.idx[i]
            assert np.array_equal(got, exp), "unpack_at"
            event("unpack:at")

    elif kind == "drop_cache":
        if h.is_cv:
            h.obj.drop_cache()
            assert h.obj._x_hat_rot is None and h.obj._dir_len == {}

    elif kind == "check":
        check_functional(h, precision=1 + (k % max(bits, 1)))

    check_structure(h)


# ---------------------------------------------------------------------------
# 4. Properties
# ---------------------------------------------------------------------------

# ---- 4.1 packing ------------------------------------------------------------

@given(bits=st.sampled_from(SUPPORTED), n=st.one_of(st.just(0), st.just(1), st.integers(2, 80)),
       seed=st.integers(0, 2**31), style=st.sampled_from(["rand", "zeros", "max"]))
@prop(250)
def test_pack_matches_bitstream_reference_and_roundtrips(bits, n, seed, style):
    rng = np.random.default_rng(seed)
    hi = 1 << bits
    vals = (rng.integers(0, hi, n) if style == "rand"
            else np.zeros(n, int) if style == "zeros" else np.full(n, hi - 1)).astype(np.uint8)
    event(f"pack:bits={bits}")
    event("pack:n=0" if n == 0 else "pack:n%group!=0" if n % _GROUP[bits][0] else "pack:n%group==0")
    p = pack(vals, bits)
    assert p.dtype == np.uint8 and p.ndim == 1
    assert p.tobytes() == ref_pack(vals, bits), "layout differs from MSB-first bit stream"
    assert len(p) == ref_nbytes(n, bits)
    assert len(p) == packed_nbytes(n, 1, bits) == packed_nbytes(1, n, bits)
    assert np.array_equal(unpack(p, bits, n), vals)
    assert ref_unpack(p.tobytes(), bits, n) == [int(v) for v in vals]
    # a prefix of the values: unpack(n_values=k) is the first k
    for kk in {0, n // 2, max(n - 1, 0)}:
        assert np.array_equal(unpack(p, bits, kk), vals[:kk]), (kk, n)
    # pack ravel()s: any shape packs identically to its flat form
    if n >= 2 and n % 2 == 0:
        assert pack(vals.reshape(2, n // 2), bits).tobytes() == p.tobytes()


@given(bits=st.sampled_from(SUPPORTED), n=st.integers(0, 40), d=st.integers(1, 70))
@prop(300)
def test_packed_nbytes_arithmetic(bits, n, d):
    got = packed_nbytes(n, d, bits)
    assert got == ref_nbytes(n * d, bits)
    assert got == len(pack(np.zeros(n * d, np.uint8), bits))
    # monotone in n, and within one group of the information-theoretic size
    assert packed_nbytes(n + 1, d, bits) >= got
    assert got * 8 >= n * d * bits and got * 8 < n * d * bits + 8 * _GROUP[bits][1] + 8 * _GROUP[bits][1]


@pytest.mark.parametrize("bits", [0, 5, 6, 7, 9, 16, -1])
def test_unsupported_bit_widths_are_refused_everywhere(bits, tmp_path):
    """Documented limitation: only 1,2,3,4,8 pack.  Every entry point must say
    so with ValueError rather than mispacking."""
    with pytest.raises(ValueError):
        Quantizer(8, bits)
    with pytest.raises(ValueError):
        pack(np.zeros(4, np.uint8), bits)
    with pytest.raises(ValueError):
        unpack(np.zeros(4, np.uint8), bits, 4)
    with pytest.raises(ValueError):
        packed_nbytes(4, 4, bits)
    # a .pq whose header names the width is refused at load
    c = Quantizer(8, 4, seed=1).encode(np.random.default_rng(0).standard_normal((3, 8)))
    p = tmp_path / "x.pq"
    save_pq(p, c)
    raw = bytearray(p.read_bytes())
    raw[16] = bits & 0xFF
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        load_pq(p)


# ---- 4.2 encode produces well-formed containers ------------------------------

@given(data=st.data())
@prop(150)
def test_encode_produces_wellformed_containers(data):
    cfg = data.draw(cfgs())
    X = data.draw(datasets(cfg))
    n = X.shape[0]
    ev_cfg(cfg, n)
    c = encode(cfg, X)
    q = make_q(cfg)
    assert isinstance(c, CompressedVectors)
    assert c.indices.dtype == np.uint8 and c.indices.shape == (n, cfg.d)
    assert (c.n, c.d, c.bits) == (n, cfg.d, cfg.bits)
    assert c.rotation == cfg.rotation == q.rotation
    assert c.seed == cfg.seed
    if n:
        assert int(c.indices.max()) < (1 << cfg.bits)
    if cfg.normalize:
        assert c.norms is not None and c.norms.dtype == np.float32 and c.norms.shape == (n,)
        assert c.scale is None
    else:
        assert c.norms is None and not c.has_norms
        assert c.scale == cfg.scale
        assert c.mean is None and c.mean_dots is None
    if cfg.mean_kind != "none":
        assert same_f32(c.mean, cfg.mean)
        assert c.mean_dots is not None and c.mean_dots.shape == (n,) and c.mean_dots.dtype == np.float32
    else:
        assert c.mean is None and c.mean_dots is None
    # a 1-D input is one row; encoding is row-independent: encode(X)[i] == encode(X[i])
    if n >= 1:
        one = encode(cfg, X[n - 1])
        assert np.array_equal(one.indices[0], c.indices[n - 1])
        assert same_f32(one.norms, None if c.norms is None else c.norms[n - 1:n])
        assert same_f32(one.mean_dots, None if c.mean_dots is None else c.mean_dots[n - 1:n])


@given(data=st.data(), block=st.integers(1, 7))
@prop(60)
def test_encode_is_independent_of_row_block_size(data, block):
    """ENCODE_BLOCK_FLOATS only bounds peak memory; the container must be
    identical for any block size (boundary of the row-block loop)."""
    import remex.core as core
    cfg = data.draw(cfgs())
    X = data.draw(datasets(cfg, n=data.draw(st.integers(2, 25))))
    ev_cfg(cfg, X.shape[0])
    ref = encode(cfg, X)
    old = core.ENCODE_BLOCK_FLOATS
    try:
        core.ENCODE_BLOCK_FLOATS = block * cfg.d      # `block` rows per block
        got = encode(cfg, X)
    finally:
        core.ENCODE_BLOCK_FLOATS = old
    assert np.array_equal(ref.indices, got.indices)
    assert same_f32(ref.norms, got.norms) and same_f32(ref.mean_dots, got.mean_dots)


# ---- 4.3 serialization round trips -------------------------------------------

def assert_container_equal(a, b, *, seed=True, scale=True, mean=True, what=""):
    unpack_ = lambda c: c.indices if isinstance(c, CompressedVectors) else c.unpack_rows(0, c.n)
    assert (a.n, a.d, a.bits, a.rotation) == (b.n, b.d, b.bits, b.rotation), what
    assert np.array_equal(unpack_(a), unpack_(b)), f"indices {what}"
    assert same_f32(a.norms, b.norms), f"norms {what}"
    assert a.has_norms == b.has_norms
    if mean:
        assert same_f32(a.mean, b.mean), f"mean {what}"
        assert same_f32(a.mean_dots, b.mean_dots), f"mean_dots {what}"
    if seed:
        assert a.seed == b.seed, f"seed {what}"
    if scale:
        assert a.scale == b.scale, f"scale {what}"


def scores_close(q, cv_a, cv_b, qry, tag):
    with np.errstate(all="ignore"):
        assert np.array_equal(q.decode(cv_a), q.decode(cv_b), equal_nan=True), tag
        eq_result(q.search(cv_a, qry, k=5), q.search(cv_b, qry, k=5), "search", tag)
        eq_result(q.search_adc(cv_a, qry, k=5), q.search_adc(cv_b, qry, k=5), "adc", tag)


@given(data=st.data())
@prop(120)
def test_every_format_roundtrips_losslessly(data, tmp_path):
    # NOTE: CompressedVectors.save -> PackedVectors.load is NOT in this property:
    # the two .npz layouts disagree when d*bits % 8 != 0 (finding F1,
    # test_known_npz_layouts_disagree).  Same-class npz and both Arrow
    # directions are exercised.  Seeds stay below 2**63 (finding F3).
    cfg = data.draw(cfgs(seeds=st.one_of(st.sampled_from(SEEDS), st.integers(0, 2**31))))
    X = data.draw(datasets(cfg))
    n = X.shape[0]
    ev_cfg(cfg, n)
    c = encode(cfg, X)
    q = make_q(cfg)
    qry = np.random.default_rng(1).standard_normal(cfg.d)
    d_ = tmp_path / uuid.uuid4().hex
    d_.mkdir()

    # .npz, CompressedVectors
    r = CompressedVectors.load(c.save(str(d_ / "c.npz")))
    assert_container_equal(c, r, what="CV npz")
    scores_close(q, c, r, qry, "CV npz")
    # .npz, PackedVectors
    pv = PackedVectors.from_compressed(c)
    r = PackedVectors.load(pv.save(str(d_ / "p.npz")))
    assert_container_equal(pv, r, what="PV npz")
    assert_container_equal(c, r, what="PV npz vs CV")
    assert np.array_equal(r._packed, pv._packed)
    # Arrow both ways, with and without memory_map
    for writer, wname in ((c, "cv"), (pv, "pv")):
        p = str(d_ / f"{wname}.arrow")
        writer.save_arrow(p)
        for mm in (False, True):
            ra = PackedVectors.load_arrow(p, memory_map=mm)
            assert_container_equal(pv, ra, what=f"arrow {wname}->pv mm={mm}")
            assert np.array_equal(ra._packed, pv._packed)
        rc = CompressedVectors.load_arrow(p)
        assert_container_equal(c, rc, what=f"arrow {wname}->cv")
        scores_close(q, c, rc, qry, "arrow")
        event(f"arrow:writer={wname}")
    # ADC from a reloaded packed container equals the original's
    if n:
        r = PackedVectors.load_arrow(str(d_ / "pv.arrow"), memory_map=True)
        with np.errstate(all="ignore"):
            eq_result(q.search_adc(r, qry, k=5), q.search_adc(c, qry, k=5), "adc reload", "")
    # .pq: indices/norms/d/bits/n/rotation/mode survive; seed, scale unrecorded;
    # a centred container is refused and leaves no file
    pqp = str(d_ / "c.pq")
    if cfg.mean_kind != "none":
        event("pq:centred-refused")
        with pytest.raises(ValueError):
            save_pq(pqp, c)
        assert not os.path.exists(pqp)
    else:
        save_pq(pqp, c)
        lp = load_pq(pqp)
        assert_container_equal(c, lp, seed=False, scale=False, what="pq")
        assert lp.seed is None and lp.scale is None
        assert (lp.norms is None) == (not cfg.normalize)
        assert os.path.getsize(pqp) == 32 + ref_nbytes(n * cfg.d, cfg.bits) + (4 * n if cfg.normalize else 0)
        # unrecorded seed/scale are not checked: decode matches the original
        with np.errstate(all="ignore"):
            assert np.array_equal(q.decode(lp), q.decode(c), equal_nan=True)


@given(cv=hand_cvs())
@prop(150)
def test_hand_built_containers_roundtrip_every_writer(cv, tmp_path):
    """Containers that encode() would never emit (arbitrary codes, hand-set
    seed/scale/mean_dots) still survive every same-class path bit for bit.
    seed is restricted below 2**63 (finding F3); scalar-with-centre is not built."""
    d_ = tmp_path / uuid.uuid4().hex
    d_.mkdir()
    pv = PackedVectors.from_compressed(cv)
    assert_container_equal(cv, pv, what="from_compressed")
    back = pv.to_compressed()
    assert_container_equal(cv, back, what="to_compressed")
    assert_container_equal(cv, CompressedVectors.load(cv.save(str(d_ / "a.npz"))), what="cv npz")
    assert_container_equal(pv, PackedVectors.load(pv.save(str(d_ / "b.npz"))), what="pv npz")
    cv.save_arrow(str(d_ / "c.arrow"))
    assert_container_equal(cv, CompressedVectors.load_arrow(str(d_ / "c.arrow")), what="arrow")
    assert_container_equal(pv, PackedVectors.load_arrow(str(d_ / "c.arrow"), memory_map=True), what="arrow mmap")
    if cv.mean is None:
        save_pq(str(d_ / "d.pq"), cv)
        assert_container_equal(cv, load_pq(str(d_ / "d.pq")), seed=False, scale=False, what="pq")


# ---- 4.4 operation sequences --------------------------------------------------

@given(data=st.data(), ops=op_lists(max_len=9))
@prop(220)
def test_operation_sequences_keep_container_and_model_in_agreement(data, ops, tmp_path):
    hand = data.draw(st.sampled_from([False, False, False, True]))
    work = tmp_path / uuid.uuid4().hex
    work.mkdir()
    if hand:
        # a container encode() would never emit; structure/model checks only
        cv = data.draw(hand_cvs())
        cfg, qry = None, None
        event("start-state:hand-built")
        h = Held(cv, Model.of(cv), None, None, work)
    else:
        cfg = data.draw(cfgs(seeds=st.one_of(st.sampled_from(SEEDS), st.integers(0, 2**31))))
        X = data.draw(datasets(cfg))
        n = X.shape[0]
        ev_cfg(cfg, n)
        cv = encode(cfg, X)
        qry = np.random.default_rng(cfg.seed % (2**31)).standard_normal(cfg.d)
        event("start-state:encoded")
        h = Held(cv, Model.of(cv), cfg, qry, work)
    check_structure(h)
    # half the cases start from a PackedVectors, half warm the CV caches first
    if data.draw(st.booleans()):
        to_pv(h)
        event("start:pv")
    elif data.draw(st.booleans()):
        event("start:warmed-cv")
        check_functional(h)
    else:
        event("start:cold-cv")
    for op in ops:
        apply_op(h, op)
    check_structure(h)
    check_functional(h)
    event(f"final-n:{min(h.model.n, 2) if h.model.n < 2 else 'many'}")
    event(f"final-bits:{h.model.bits}")
    event("final:" + ("cv" if h.is_cv else "pv"))


# ---- 4.5 subset selector forms ---------------------------------------------------

@given(data=st.data(), form=st.sampled_from(VALID_FORMS + INVALID_FORMS), k=st.integers(0, 10**6))
@prop(220)
def test_subset_accepts_documented_forms_and_rejects_the_rest(data, form, k, tmp_path):
    """Both containers: bool masks, negatives, duplicates, empties, ranges and
    tuples select what an independent resolver says; out-of-range and wrong-length
    masks raise IndexError and never mutate the container.  Warm CV caches
    (x_hat_rot) are sliced with the rows."""
    cfg = data.draw(cfgs())
    X = data.draw(datasets(cfg))
    n = X.shape[0]
    ev_cfg(cfg, n)
    cv = encode(cfg, X)
    qry = np.random.default_rng(3).standard_normal(cfg.d)
    sel, rows = build_selector(form, k, n)
    event(f"selector:{form}")
    pv = PackedVectors.from_compressed(cv)
    if data.draw(st.booleans()):
        with np.errstate(all="ignore"):
            make_q(cfg).search(cv, qry, k=3)          # warm: _x_hat_rot and _dir_len populated
        event("cache:warm")
    for obj in (cv, pv):
        before = (obj.n, obj.indices.copy() if obj is cv else obj._packed.copy())
        if rows is None:
            with pytest.raises(IndexError):
                obj.subset(sel)
            assert obj.n == before[0]
            continue
        sub = obj.subset(sel)
        m = Model.of(cv).rows(rows)
        h = Held(sub, m, cfg, qry, tmp_path)
        h.log.append(f"subset:{form}")
        check_structure(h)
        check_functional(h)
        # original untouched
        assert obj.n == before[0]
        assert np.array_equal(before[1], obj.indices if obj is cv else obj._packed)


# ---- 4.6 Matryoshka / at_precision ------------------------------------------------

@given(data=st.data())
@prop(150)
def test_at_precision_is_right_shift_and_composes(data, tmp_path):
    cfg = data.draw(cfgs(bits=data.draw(st.sampled_from([2, 3, 4, 8]))))
    X = data.draw(datasets(cfg))
    ev_cfg(cfg, X.shape[0])
    cv = encode(cfg, X)
    pv = PackedVectors.from_compressed(cv)
    levels = [t for t in SUPPORTED if t <= cfg.bits]
    for t in levels:
        sh = cfg.bits - t
        lo = pv.at_precision(t)
        assert lo.bits == t and lo.n == pv.n and lo.d == pv.d
        assert np.array_equal(lo.unpack_rows(0, lo.n), (cv.indices >> sh).astype(np.uint8))
        assert lo._packed.shape == (pv.n, ref_nbytes(cfg.d, t))
        for i in range(lo.n):
            assert lo._packed[i].tobytes() == ref_pack(cv.indices[i] >> sh, t)
        event(f"precision:{cfg.bits}->{t}")
        # composition: at_precision(j).at_precision(t) == at_precision(t)
        for j in levels:
            if j >= t:
                mid = pv.at_precision(j).at_precision(t)
                assert np.array_equal(mid._packed, lo._packed), (cfg.bits, j, t)
        # derived container keeps every encoding field
        assert (lo.rotation, lo.seed, lo.scale) == (pv.rotation, pv.seed, pv.scale)
        assert same_f32(lo.norms, pv.norms) and same_f32(lo.mean, pv.mean) and same_f32(lo.mean_dots, pv.mean_dots)
        # ... and does not alias its parent's arrays (at_precision(self.bits) returns
        # self by design: tests/test_packed_vectors.py::test_at_full_precision_returns_self)
        if t == cfg.bits:
            assert lo is pv
        elif lo.norms is not None and lo.n:
            lo.norms[0] = np.float32(1234.5)
            assert pv.norms[0] != np.float32(1234.5)
    for bad in (0, -1, cfg.bits + 1):
        with pytest.raises(ValueError):
            pv.at_precision(bad)
    # precision= reads on the original agree between CV and PV paths
    q = make_q(cfg)
    qry = np.random.default_rng(5).standard_normal(cfg.d)
    with np.errstate(all="ignore"):
        for t in levels:
            eq_result(q.search_adc(cv, qry, k=4, precision=t),
                      q.search_adc(pv, qry, k=4, precision=t), f"adc precision={t}", "")
            sh = cfg.bits - t
            assert np.array_equal(q.decode(cv, precision=t),
                                  q.decode(cv.subset(np.arange(cv.n)), precision=t), equal_nan=True)


@given(data=st.data())
@prop(100)
def test_one_bit_matryoshka_level_is_exactly_the_standalone_one_bit_code(data):
    """Documented (CLAUDE.md design decision 2): the MSB of an n-bit code *is* the
    1-bit code.  Holds in every mode: haar/rht/none, normalized, scalar, centred."""
    cfg = data.draw(cfgs(bits=data.draw(st.sampled_from([2, 3, 4, 8]))))
    X = data.draw(datasets(cfg, n=data.draw(st.integers(1, 20))))
    ev_cfg(cfg, X.shape[0])
    big = encode(cfg, X)
    one = encode(cfg.replace(bits=1), X)
    assert np.array_equal(big.indices >> (cfg.bits - 1), one.indices)
    assert same_f32(big.norms, one.norms) and same_f32(big.mean_dots, one.mean_dots)


# ---- 4.7 mismatch guards (cold and warm) ------------------------------------------

def other_rotation(cfg):
    for r in ("haar", "none", "rht"):
        if r != cfg.rotation and not (r == "rht" and cfg.d % 2):
            return r


def mismatched(cfg: Cfg, kind: str) -> Optional[Cfg]:
    if kind == "seed":
        return cfg.replace(seed=cfg.seed + 1 if cfg.seed < 2**63 - 1 else cfg.seed - 1)
    if kind == "rotation":
        return cfg.replace(rotation=other_rotation(cfg))
    if kind == "mean_drop":
        return cfg.replace(mean_kind="none") if cfg.mean_kind != "none" else None
    if kind == "mean_add":
        return cfg.replace(mean_kind="unit", mean_seed=cfg.mean_seed + 1, renorm=True) \
            if cfg.normalize and cfg.mean_kind == "none" else None
    if kind == "mean_other":
        return cfg.replace(mean_seed=cfg.mean_seed + 1, mean_kind="big" if cfg.mean_kind == "unit" else "unit") \
            if cfg.mean_kind in ("unit", "big") else None
    if kind == "mode":
        return (cfg.replace(normalize=False, scale=1.0, mean_kind="none") if cfg.normalize
                else cfg.replace(normalize=True, scale=None, renorm=True, mean_kind="none"))
    if kind == "d":
        nd = cfg.d + 1 if cfg.d < 64 else cfg.d - 1
        rot = cfg.rotation
        if rot == "rht" and nd % 2:
            rot = "haar"
        return cfg.replace(d=nd, rotation=rot, mean_kind="none") if cfg.mean_kind == "none" else None
    if kind == "bits":
        nb = [b for b in SUPPORTED if b != cfg.bits][0]
        return cfg.replace(bits=nb)
    if kind == "scale":
        return cfg.replace(scale=cfg.scale * 2) if not cfg.normalize else None
    raise AssertionError(kind)


MISMATCH_KINDS = ["seed", "rotation", "mean_drop", "mean_add", "mean_other", "mode", "d", "bits", "scale"]


def call_all(q, obj, qry, is_cv):
    """Every decode/search entry that accepts this container type."""
    q_in = qry[: q.d] if len(qry) >= q.d else np.resize(qry, q.d)
    calls = []
    if is_cv:
        calls += [("decode", lambda: q.decode(obj)), ("search", lambda: q.search(obj, q_in, k=3)),
                  ("search_batch", lambda: q.search_batch(obj, q_in[None], k=3))]
    calls += [("search_adc", lambda: q.search_adc(obj, q_in, k=3)),
              ("search_twostage", lambda: q.search_twostage(obj, q_in, k=3, candidates=4))]
    return calls


@given(data=st.data(), kind=st.sampled_from(MISMATCH_KINDS), warm=st.booleans(),
       packed=st.booleans(), unrecorded=st.booleans())
@prop(600)
def test_mismatched_quantizers_are_refused_cold_and_warm(data, kind, warm, packed, unrecorded):
    """seed/rotation/mean/mode/d/bits/scale mismatches must raise ValueError on
    every path, on a container that has been searched before as well as one
    that has not.  renorm=False mismatching readers on a *warm* CompressedVectors.search
    are finding F2 and are excluded here (test_known_search_guards_skipped_on_warm_container)."""
    cfg = data.draw(cfgs(seeds=st.sampled_from([1, 2, 5, 99])))
    other = mismatched(cfg, kind)
    if other is None:
        return
    # scalar-mode scale mismatch / 'mode' etc need a valid mismatching quantizer
    if other.normalize:
        other = other.replace(renorm=True if other.mean_kind != "none" else data.draw(st.booleans()))
    X = data.draw(datasets(cfg, n=data.draw(st.integers(1, 12))))
    ev_cfg(cfg, X.shape[0])
    event(f"mismatch:{kind}")
    event(f"reader-renorm:{other.renorm}")
    event("state:warm" if warm else "state:cold")
    cv = encode(cfg, X)
    if unrecorded and kind in ("seed", "scale"):
        cv.seed = None          # a .pq / pre-2.1 container: seed unrecorded
        cv.scale = None
    obj = PackedVectors.from_compressed(cv) if packed else cv
    qry = np.random.default_rng(2).standard_normal(max(cfg.d, other.d))
    good = make_q(cfg)
    if warm:
        with np.errstate(all="ignore"):
            for _, fn in call_all(good, obj, qry, not packed):
                fn()
    bad_q = make_q(other)
    # rotation="none" ignores seed; unrecorded seed/scale is not checked
    expect_raise = True
    if kind == "seed" and (cfg.rotation == "none" or cv.seed is None):
        expect_raise = False
    if kind == "scale" and cv.scale is None:
        expect_raise = False
    for name, fn in call_all(bad_q, obj, qry, not packed):
        if (expect_raise and warm and name == "search" and kind in ("bits", "mean_drop")
                and (not other.renorm or not cfg.normalize)):
            # Finding F2: CompressedVectors.search skips _check_bits/_check_mean on a warm
            # container whenever _effective_norms returns early (renorm=False or scalar mode).
            event("skipped:F2 (search, warm, renorm=False/scalar)")
            continue
        if expect_raise:
            with np.errstate(all="ignore"), pytest.raises(ValueError):
                fn()
            event(f"raised:{name}")
        else:
            try:
                with np.errstate(all="ignore"):
                    fn()
            except ValueError as e:
                if "mismatch" in str(e):
                    raise AssertionError(f"{name}: refused although {kind} is unchecked here: {e}")


@given(data=st.data(), key=st.sampled_from(["rotation", "seed", "mean", "scale", "bits", "d"]))
@prop(80)
def test_roundtripped_containers_keep_guards_armed(data, key, tmp_path):
    """The guard must survive every serializer that claims to record the field."""
    cfg = data.draw(cfgs(seeds=st.sampled_from([3, 4, 6])))
    other = mismatched(cfg, {"rotation": "rotation", "seed": "seed", "mean": "mean_drop",
                             "scale": "scale", "bits": "bits", "d": "d"}[key])
    if other is None:
        return
    X = data.draw(datasets(cfg, n=data.draw(st.integers(1, 8))))
    cv = encode(cfg, X)
    work = tmp_path / uuid.uuid4().hex
    work.mkdir()
    loaded = {
        "cv_npz": CompressedVectors.load(cv.save(str(work / "a.npz"))),
        "pv_npz": PackedVectors.load(PackedVectors.from_compressed(cv).save(str(work / "b.npz"))),
    }
    cv.save_arrow(str(work / "c.arrow"))
    loaded["arrow_pv"] = PackedVectors.load_arrow(str(work / "c.arrow"))
    loaded["arrow_cv"] = CompressedVectors.load_arrow(str(work / "c.arrow"))
    for name, obj in loaded.items():
        event(f"guard-after:{name}")
        for cname, fn in call_all(make_q(other), obj, np.ones(max(cfg.d, other.d)), isinstance(obj, CompressedVectors)):
            if key == "seed" and cfg.rotation == "none":
                continue
            with np.errstate(all="ignore"), pytest.raises(ValueError):
                fn()


# ---- 4.8 reader robustness: truncated / damaged files ------------------------------

@given(data=st.data(), what=st.sampled_from(["truncate", "flip", "append", "zero-header"]))
@prop(150)
def test_damaged_pq_is_a_valid_container_or_a_value_error(data, what, tmp_path):
    cfg = data.draw(cfgs(scalar=True, mean=False))
    X = data.draw(datasets(cfg, n=data.draw(st.integers(0, 10))))
    c = encode(cfg, X)
    p = tmp_path / (uuid.uuid4().hex + ".pq")
    save_pq(p, c)
    raw = bytearray(p.read_bytes())
    pos = data.draw(st.integers(0, max(len(raw) - 1, 0)))
    if what == "truncate":
        raw = raw[:pos]
    elif what == "flip":
        raw[pos] ^= 1 << data.draw(st.integers(0, 7))
    elif what == "append":
        raw += bytes([data.draw(st.integers(0, 255))]) * data.draw(st.integers(1, 9))
    else:
        raw[:data.draw(st.integers(1, 32))] = b"\x00" * 1
    event(f"pq-damage:{what}")
    p.write_bytes(bytes(raw))
    try:
        got = load_pq(p)
    except ValueError:
        event("pq-damage:refused")
        return
    event("pq-damage:accepted")
    assert got.indices.dtype == np.uint8 and got.indices.shape == (got.n, got.d)
    assert got.bits in SUPPORTED
    if got.n:
        assert int(got.indices.max()) < (1 << got.bits)
    assert got.norms is None or got.norms.shape == (got.n,)


@given(data=st.data(), what=st.sampled_from(["truncate", "flip"]))
@prop(120)
def test_damaged_npz_and_arrow_never_yield_a_wrong_shaped_container(data, what, tmp_path):
    """Any exception is acceptable here (see test_known_damaged_npz... for the
    exception *type* promise); what is not acceptable is a container whose
    shape disagrees with its own n/d, or a hang."""
    cfg = data.draw(cfgs(mean=True))
    X = data.draw(datasets(cfg, n=data.draw(st.integers(1, 10))))
    c = encode(cfg, X)
    for ext, save, load in (
        (".npz", c.save, CompressedVectors.load),
        (".arrow", c.save_arrow, CompressedVectors.load_arrow),
    ):
        p = tmp_path / (uuid.uuid4().hex + ext)
        save(str(p))
        raw = bytearray(p.read_bytes())
        pos = data.draw(st.integers(0, len(raw) - 1))
        if what == "truncate":
            raw = raw[:pos]
        else:
            raw[pos] ^= 1 << data.draw(st.integers(0, 7))
        p.write_bytes(bytes(raw))
        event(f"damage:{ext}:{what}")
        try:
            got = load(str(p))
        except Exception:
            event(f"damage:{ext}:refused")
            continue
        event(f"damage:{ext}:accepted")
        assert got.indices.shape == (got.n, got.d)


# ---- 4.9 all-or-nothing writers ---------------------------------------------------

class _Boom(RuntimeError):
    pass


@given(data=st.data(), writer=st.sampled_from(["cv_npz", "pv_npz", "cv_arrow", "pv_arrow", "pq", "params"]),
       preexisting=st.booleans(), where=st.sampled_from(["partial", "rename"]))
@prop(60)
def test_writers_are_all_or_nothing(data, writer, preexisting, where, tmp_path, monkeypatch):
    cfg = data.draw(cfgs(mean=writer not in ("pq", "params"), scalar=writer != "params",
                         rotation=("haar" if writer == "params" else None)))
    if writer == "params":
        cfg = cfg.replace(rotation=data.draw(st.sampled_from(["haar", "rht"])))
        if cfg.rotation == "rht" and cfg.d % 2:
            cfg = cfg.replace(d=cfg.d + 1)
    X = data.draw(datasets(cfg, n=data.draw(st.integers(1, 8))))
    c = encode(cfg, X)
    pv = PackedVectors.from_compressed(c)
    ext = {"cv_npz": ".npz", "pv_npz": ".npz", "cv_arrow": ".arrow", "pv_arrow": ".arrow",
           "pq": ".pq", "params": ".bin"}[writer]
    work = tmp_path / uuid.uuid4().hex
    work.mkdir()
    dest = work / f"dest{ext}"
    sentinel = b"previous contents"
    if preexisting:
        dest.write_bytes(sentinel)
    do = {
        "cv_npz": lambda: c.save(str(dest)),
        "pv_npz": lambda: pv.save(str(dest)),
        "cv_arrow": lambda: c.save_arrow(str(dest)),
        "pv_arrow": lambda: pv.save_arrow(str(dest)),
        "pq": lambda: save_pq(str(dest), c),
        "params": lambda: save_params(str(dest), make_q(cfg)),
    }[writer]
    event(f"writer:{writer}")
    event("dest:preexisting" if preexisting else "dest:absent")
    event(f"fail-at:{where}")
    with monkeypatch.context() as mp:
        import remex._io as rio
        if where == "rename":
            def boom(*a, **k):
                raise _Boom("rename failed")
            mp.setattr(rio.os, "replace", boom)
        elif writer.endswith("npz"):
            real = np.savez_compressed

            def partial(tmp, **kw):
                real(tmp, **kw)
                with open(tmp, "r+b") as f:       # file exists and is complete, then the process "dies"
                    f.truncate(5)
                raise _Boom("killed mid-write")
            mp.setattr(np, "savez_compressed", partial)
        elif writer.endswith("arrow"):
            import pyarrow.feather as feather
            real_w = feather.write_feather

            def partial_w(table, tmp, **kw):
                real_w(table, tmp, **kw)
                with open(tmp, "r+b") as f:
                    f.truncate(5)
                raise _Boom("killed mid-write")
            mp.setattr(feather, "write_feather", partial_w)
        else:
            import builtins
            mod = sys.modules["remex.pq_format"]
            real_open = builtins.open

            class Dying:
                def __init__(self, f):
                    self.f, self.calls = f, 0

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    self.f.close()

                def write(self, b):
                    self.calls += 1
                    if self.calls >= 2 or len(b) > 20:
                        self.f.write(b[:3])
                        raise _Boom("killed mid-write")
                    return self.f.write(b)
            mp.setattr(mod, "open", lambda p, m="r", *a, **k: Dying(real_open(p, m, *a, **k)), raising=False)
        with pytest.raises(_Boom):
            do()
    leftovers = sorted(os.listdir(work))
    assert leftovers == (["dest" + ext] if preexisting else []), f"temp litter or missing dest: {leftovers}"
    if preexisting:
        assert dest.read_bytes() == sentinel, "old file damaged by a failed overwrite"
    # and the same call succeeds afterwards, leaving exactly one file
    do()
    assert sorted(os.listdir(work)) == ["dest" + ext]


@given(data=st.data())
@prop(60)
def test_save_params_file_contents(data, tmp_path):
    rot = data.draw(st.sampled_from(["haar", "rht"]))
    cfg = data.draw(cfgs(scalar=False, mean=False, rotation=rot))
    q = make_q(cfg)
    work = tmp_path / uuid.uuid4().hex
    work.mkdir()
    p = work / "p.bin"
    save_params(str(p), q)
    raw = p.read_bytes()
    d, bits = cfg.d, cfg.bits
    assert len(raw) == 16 + 4 * (d * d + (2 ** bits - 1) + 2 ** bits)
    assert raw[:4] == b"PR\x00\x01" and struct.unpack("<I", raw[4:8])[0] == d
    assert raw[8] == bits and raw[9] == {"haar": 0, "rht": 1}[rot]
    body = np.frombuffer(raw[16:], dtype="<f4")
    assert np.array_equal(body[:d * d].reshape(d, d), q.R.astype(np.float32))
    assert np.array_equal(body[d * d:d * d + 2 ** bits - 1], q.boundaries.astype(np.float32))
    assert np.array_equal(body[d * d + 2 ** bits - 1:], q.centroids.astype(np.float32))
    # refused configurations create nothing
    for bad in (make_q(cfg.replace(normalize=False, scale=1.0, mean_kind="none", renorm=True)),
                make_q(cfg.replace(rotation="none"))):
        with pytest.raises(ValueError):
            save_params(str(work / "bad.bin"), bad)
        assert not (work / "bad.bin").exists()


# ---- 4.10 size arithmetic and chunk boundaries -------------------------------------

@given(cv=hand_cvs())
@prop(200)
def test_nbytes_and_compression_ratio_arithmetic(cv):
    n, d, bits = cv.n, cv.d, cv.bits
    pv = PackedVectors.from_compressed(cv)
    extra = (0 if cv.norms is None else 4 * n) + (0 if cv.mean_dots is None else 4 * n)
    assert cv.nbytes == ref_nbytes(n * d, bits) + extra
    assert cv.nbytes_unpacked == n * d + extra
    assert pv.nbytes == n * ref_nbytes(d, bits) + extra
    assert pv.resident_bytes == pv.nbytes
    assert pv.nbytes >= cv.nbytes                      # row padding can only add
    if (d * bits) % 8 == 0:
        assert pv.nbytes == cv.nbytes                   # aligned rows: no padding
        event("size:aligned-equal")
    else:
        event("size:unaligned")
    if n:
        assert cv.compression_ratio == n * d * 4 / cv.nbytes
        assert pv.compression_ratio == n * d * 4 / pv.nbytes


CHUNK_NS = [4095, 4096, 4097, 8191, 8192, 8193]


@given(n=st.sampled_from(CHUNK_NS), d=st.sampled_from([2, 3, 5, 8]), bits=st.sampled_from([3, 4, 8]),
       seed=st.integers(0, 10**6), chunk=st.sampled_from([1, 7, 4096, 5000, 10**6]))
@prop(24)
def test_chunk_boundaries_at_4096(n, d, bits, seed, chunk):
    """at_precision, _direction_lengths and the ADC scan loop over 4096-row chunks
    (literal in core.py).  n straddles the boundary; search chunk_size is varied."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 1 << bits, (n, d)).astype(np.uint8)
    norms = rng.random(n).astype(np.float32) + 0.1
    rot = "none" if d == 3 or d == 5 else "rht"
    cv = CompressedVectors(idx, norms, d, bits, rot if d % 2 == 0 or rot == "none" else "haar", seed=7)
    pv = PackedVectors.from_compressed(cv)
    event(f"chunk:n={n}")
    for t in [t for t in SUPPORTED if t < bits]:
        lo = pv.at_precision(t)
        assert np.array_equal(lo.unpack_rows(0, n), idx >> (bits - t))
        assert np.array_equal(lo.unpack_rows(n - 3, n), idx[n - 3:] >> (bits - t))
    q = Quantizer(d, bits, seed=7, rotation=cv.rotation)
    qry = rng.standard_normal(d)
    a = q.search_adc(cv, qry, k=9, chunk_size=chunk)
    b = q.search_adc(pv, qry, k=9, chunk_size=chunk)
    eq_result(a, b, "adc chunked", f"n={n} chunk={chunk}")
    eq_result(a, q.search_adc(cv, qry, k=9), "adc chunk-size invariance", f"chunk={chunk}")
    for t in range(1, bits):
        eq_result(q.search_adc(cv, qry, k=9, precision=t, chunk_size=chunk),
                  q.search_adc(pv, qry, k=9, precision=t, chunk_size=chunk), f"adc p={t}", "")
    eq_result(q.search_twostage(cv, qry, k=9, candidates=50), q.search_twostage(pv, qry, k=9, candidates=50), "twostage", "")


# ---- 4.11 independent decode / score oracle -----------------------------------------

def dense_decode(q: Quantizer, cv, precision=None) -> np.ndarray:
    """Uncentred / scalar decode from the definition, using the dense R."""
    p = q.bits if precision is None else precision
    cent = (q.centroids if p == q.bits else q._nested[p]).astype(np.float64)
    codes = (cv.indices >> (q.bits - p)).astype(np.int64)
    C = cent[codes]
    X = C @ q.R.astype(np.float64)
    if cv.norms is None:
        return X
    scale = cv.norms.astype(np.float64)
    if q.renorm:
        scale = scale / np.maximum(np.sqrt((C ** 2).sum(axis=1)), 1e-12)
    return X * scale[:, None]


@given(data=st.data())
@prop(150)
def test_decode_and_scores_match_an_independent_oracle_after_reload(data, tmp_path):
    cfg = data.draw(cfgs())
    X = data.draw(datasets(cfg, n=data.draw(st.integers(1, 15))))
    # float32 overflow guard for 'big' data in the independent float64 oracle
    ev_cfg(cfg, X.shape[0])
    cv = encode(cfg, X)
    q = make_q(cfg)
    work = tmp_path / uuid.uuid4().hex
    work.mkdir()
    cv.save_arrow(str(work / "x.arrow"))
    reloaded = PackedVectors.load_arrow(str(work / "x.arrow")).to_compressed()
    qry = np.random.default_rng(11).standard_normal(cfg.d)
    prec = data.draw(st.sampled_from([None] + [t for t in range(1, cfg.bits + 1)]))
    with np.errstate(all="ignore"):
        dec = q.decode(reloaded, precision=prec).astype(np.float64)
        if cfg.mean_kind == "none":
            ref = dense_decode(q, reloaded, prec)
            tol = 2e-4 * max(1.0, float(np.abs(ref).max()) if ref.size else 1.0)
            np.testing.assert_allclose(dec, ref, rtol=2e-4, atol=tol)
            event("oracle:plain/scalar decode")
            # search scores are q . decode(x) for uncentred / scalar containers
            idx, sc = q.search(reloaded, qry, k=5, precision=prec)
            oracle = ref @ qry
            np.testing.assert_allclose(sc, oracle[idx], rtol=3e-4, atol=3e-4 * max(1.0, float(np.abs(oracle).max())))
            assert list(np.argsort(-sc, kind="stable")) == list(range(len(sc))) or np.all(np.diff(sc) <= 0)
            assert len(set(idx.tolist())) == len(idx)
            ia, sa = q.search_adc(PackedVectors.from_compressed(reloaded), qry, k=5, precision=prec)
            np.testing.assert_allclose(sa, oracle[ia], rtol=3e-4, atol=3e-4 * max(1.0, float(np.abs(oracle).max())))
        else:
            # documented invariant of centred decode: mu . (decode(x) - mu) == mu . (x - mu)
            mu = cfg.mean.astype(np.float64)
            lhs = (dec - mu) @ mu
            rhs = reloaded.mean_dots.astype(np.float64)
            scale = np.abs(mu).sum() * (np.abs(dec).max() + np.abs(mu).max()) + np.abs(rhs).max()
            np.testing.assert_allclose(lhs, rhs, rtol=1e-3, atol=1e-3 * scale + 1e-6)
            event("oracle:centred invariant")


# ---- 4.12 isolation (clone / copy-on-write) ------------------------------------------

@given(data=st.data())
@prop(80)
def test_derived_containers_do_not_alias_their_parent(data):
    cfg = data.draw(cfgs(scalar=False))
    X = data.draw(datasets(cfg, n=data.draw(st.integers(2, 8))))
    cv = encode(cfg, X)
    pv = PackedVectors.from_compressed(cv)
    snap = (cv.indices.copy(), None if cv.norms is None else cv.norms.copy(),
            None if cv.mean is None else cv.mean.copy(),
            None if cv.mean_dots is None else cv.mean_dots.copy(), pv._packed.copy())
    # PackedVectors conversions copy; CompressedVectors.subset must too (indices
    # and norms are fancy-indexed copies).  mean aliasing is finding A8 (ambiguous)
    # and is deliberately not asserted here.
    derived = [cv.subset(np.arange(cv.n)), pv.subset(np.arange(pv.n)), pv.to_compressed(),
               PackedVectors.from_compressed(cv)]
    for dv in derived:
        idx = dv.indices if isinstance(dv, CompressedVectors) else dv._packed
        idx[...] = 0
        if dv.norms is not None:
            dv.norms[...] = -1
        if dv.mean_dots is not None:
            dv.mean_dots[...] = -1
    assert np.array_equal(cv.indices, snap[0]) and np.array_equal(pv._packed, snap[4])
    assert same_f32(cv.norms, snap[1]) and same_f32(cv.mean_dots, snap[3])
    event("alias-check:ok")


# ===========================================================================
# 5. KNOWN findings (xfail strict): each is a property over the failing region
# ===========================================================================

def xf(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


@xf("F1: CompressedVectors.save and PackedVectors.save write different packed_indices "
    "layouts (flat vs row-padded) under the same .npz keys when d*bits % 8 != 0")
@given(data=st.data(), direction=st.sampled_from(["cv->pv", "pv->cv"]))
@prop(40, **NO_SHRINK)
def test_known_npz_layouts_disagree(data, direction, tmp_path):
    cfg = data.draw(cfgs(unaligned=True, scalar=False, mean=False))
    X = data.draw(datasets(cfg, n=data.draw(st.integers(2, 12))))
    c = encode(cfg, X)
    pv = PackedVectors.from_compressed(c)
    p = str(tmp_path / (uuid.uuid4().hex + ".npz"))
    if direction == "cv->pv":
        got = PackedVectors.load(c.save(p)).unpack_rows(0, c.n)
    else:
        got = CompressedVectors.load(pv.save(p)).indices
    assert np.array_equal(got, c.indices)


@xf("F2: Quantizer.search consults only _check_rotation before using a warm _x_hat_rot; "
    "_effective_norms returns before its guards under renorm=False or in scalar mode, so the "
    "mean and bits guards are skipped")
@given(data=st.data(), kind=st.sampled_from(["mean_drop", "bits"]), via_subset=st.booleans())
@prop(40, **NO_SHRINK)
def test_known_search_guards_skipped_on_warm_container(data, kind, via_subset):
    cfg = data.draw(cfgs(seeds=st.just(1)))
    if kind == "mean_drop" and cfg.mean_kind == "none":
        cfg = cfg.replace(normalize=True, scale=None, mean_kind="unit", renorm=True)
    other = mismatched(cfg, kind)
    if other.normalize:
        other = other.replace(renorm=False)       # (scalar containers hit it with any reader)
    if other.mean_kind != "none":
        other = other.replace(renorm=True)       # mean requires renorm
    event(f"F2:{'scalar' if not cfg.normalize else 'normalized'}:{kind}:reader-renorm-{other.renorm}")
    X = data.draw(datasets(cfg, n=data.draw(st.integers(2, 10))))
    cv = encode(cfg, X)
    qry = np.ones(cfg.d)
    make_q(cfg).search(cv, qry, k=2)                # warm: populates _x_hat_rot
    target = cv.subset(np.arange(cv.n)) if via_subset else cv    # subset carries the cache
    with pytest.raises(ValueError):                   # a cold container would raise
        make_q(other).search(target, qry, k=2)


@xf("F3: .npz stores the seed as np.int64, so a seed in [2**63, inf) that Quantizer and "
    "Arrow accept overflows at save")
@given(seed=st.sampled_from([2**63, 2**64 - 1, 2**64 + 5]), rot=st.sampled_from(["haar", "rht", "none"]),
       packed=st.booleans())
@prop(20, **NO_SHRINK)
def test_known_large_seeds_roundtrip_npz(seed, rot, packed, tmp_path):
    q = Quantizer(8, 3, seed=seed, rotation=rot)
    c = q.encode(np.random.default_rng(0).standard_normal((3, 8)))
    obj = PackedVectors.from_compressed(c) if packed else c
    p = str(tmp_path / (uuid.uuid4().hex + ".npz"))
    back = type(obj).load(obj.save(p))
    assert back.seed == seed


@xf("F4: save_arrow(**extra_metadata) overwrites the reserved d/n/bits/rotation/mean/scale keys, "
    "so a caller keyword can silently relabel the encoding")
@given(data=st.data(), key=st.sampled_from(["d", "n", "bits", "rotation", "mean", "scale"]))
@prop(40, **NO_SHRINK)
def test_known_arrow_extra_metadata_cannot_override_encoding(data, key, tmp_path):
    cfg = data.draw(cfgs(scalar=False, mean=False, seeds=st.just(1)))
    c = encode(cfg, np.random.default_rng(0).standard_normal((4, cfg.d)))
    p = str(tmp_path / (uuid.uuid4().hex + ".arrow"))
    val = {"d": cfg.d + 1, "n": c.n + 1, "bits": 3 if cfg.bits != 3 else 4,
           "rotation": "haar" if cfg.rotation != "haar" else "none",
           "mean": "xx", "scale": 2.0}[key]
    try:
        c.save_arrow(p, **{key: val})
    except ValueError:
        return                                    # refusing the reserved key is a fix
    back = PackedVectors.load_arrow(p)           # ... and so is ignoring it
    assert (back.n, back.d, back.bits, back.rotation) == (c.n, c.d, c.bits, c.rotation)
    assert back.mean is None and back.scale is None


@xf("F5: open_npz accepts the legacy indices-only layout but PackedVectors.load indexes "
    "'packed_indices' unconditionally -> raw KeyError")
@given(data=st.data())
@prop(30, **NO_SHRINK)
def test_known_packedvectors_load_of_legacy_indices_npz(data, tmp_path):
    cfg = data.draw(cfgs(scalar=False, mean=False))
    c = encode(cfg, np.random.default_rng(0).standard_normal((3, cfg.d)))
    p = str(tmp_path / (uuid.uuid4().hex + ".npz"))
    np.savez(p, indices=c.indices, norms=c.norms, d=np.int32(c.d), bits=np.int32(c.bits), n=np.int32(c.n))
    assert CompressedVectors.load(p).n == c.n         # the CV reader supports it
    try:
        got = PackedVectors.load(p)
    except ValueError:
        return                                        # a named refusal would also be acceptable
    assert np.array_equal(got.unpack_rows(0, got.n), c.indices)


@xf("A1: compression_ratio divides 0/0 for an empty container (ZeroDivisionError)")
@given(data=st.data(), packed=st.booleans())
@prop(20, **NO_SHRINK)
def test_known_compression_ratio_of_empty_container(data, packed):
    cfg = data.draw(cfgs(scalar=False, mean=False))
    c = encode(cfg, np.empty((0, cfg.d)))
    obj = PackedVectors.from_compressed(c) if packed else c
    r = obj.compression_ratio
    assert r == r


@xf("F6: open_npz wraps BadZipFile/EOFError/ValueError only around np.load; a damaged member "
    "(flipped byte) fails later, in the lazy member read, as BadZipFile / zlib.error / "
    "NotImplementedError / RuntimeError / OSError instead of the named ValueError")
@given(data=st.data())
@prop(40, **NO_SHRINK)
def test_known_damaged_npz_member_names_the_file(data, tmp_path):
    cfg = data.draw(cfgs(mean=False))
    c = encode(cfg, data.draw(datasets(cfg, n=data.draw(st.integers(1, 10)))))
    p = tmp_path / (uuid.uuid4().hex + ".npz")
    c.save(str(p))
    raw = bytearray(p.read_bytes())
    pos = data.draw(st.integers(0, len(raw) - 1))
    raw[pos] ^= 1 << data.draw(st.integers(0, 7))
    p.write_bytes(bytes(raw))
    try:
        CompressedVectors.load(str(p))
    except ValueError:
        event("flip:named ValueError")
    except Exception as e:                           # BadZipFile, zlib.error, EOFError ...
        event(f"flip:{type(e).__name__} escaped")
        raise AssertionError(f"{type(e).__name__} escaped open_npz: {e}")
    else:
        event("flip:loaded (harmless bit)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--hypothesis-show-statistics"]))
