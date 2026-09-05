"""A RASP subset with one extra primitive: numeric select.

RASP (Weiss, Goldberg, Yahav 2021) builds programs out of s-ops -- sequence
operations that map a length-n input to a length-n output. This module carries
the pieces a compiler needs and nothing else: `tokens`, `indices`, elementwise
`smap` / `sequence_map`, the categorical `select` / `aggregate` pair, and two
additions -- `select_all` (RASP's constant-true selector, kept separate so a
compiler need not tabulate it) and `select_at`.

`select_at(addr)` is the new one. `addr` is a numeric s-op giving, at every
query position i, an integer target position. `aggregate(select_at(addr), v)`
returns `v[addr[i]]`, or 0 when `addr[i]` falls outside `0 .. n-1`.

Programs are built as graphs so the same object can be run by `evaluate` here
and compiled to weight matrices by `compile_ns.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

Value = float


# --------------------------------------------------------------------------
# s-ops
# --------------------------------------------------------------------------


@dataclass(eq=False)
class SOp:
    """A sequence operation. Subclasses carry the arguments; `kind` is
    'numerical' for a real-valued stream and 'categorical' for a symbol
    stream that a compiler one-hot encodes."""

    name: str = field(default="sop", kw_only=True)
    kind: str = field(default="numerical", kw_only=True)

    def inputs(self) -> list["SOp"]:
        return []


@dataclass(eq=False)
class Tokens(SOp):
    """The input sequence itself."""


@dataclass(eq=False)
class Indices(SOp):
    """0, 1, ..., n-1."""


@dataclass(eq=False)
class Map(SOp):
    f: Callable[[Value], Value] = None
    x: SOp = None

    def inputs(self) -> list[SOp]:
        return [self.x]


@dataclass(eq=False)
class SequenceMap(SOp):
    f: Callable[[Value, Value], Value] = None
    x: SOp = None
    y: SOp = None

    def inputs(self) -> list[SOp]:
        return [self.x, self.y]


@dataclass(eq=False)
class Categorize(SOp):
    """Turn a numeric s-op into a categorical one over a fixed value list.

    RASP's own numerical-to-categorical conversion, and Tracr's: the compiler
    spends one residual dimension per candidate value. Semantically the
    identity, so the interpreter just passes the numbers through.
    """

    x: SOp = None
    values: tuple = ()

    def inputs(self) -> list["SOp"]:
        return [self.x]


@dataclass(eq=False)
class Aggregate(SOp):
    selector: "Selector" = None
    values: SOp = None

    def inputs(self) -> list[SOp]:
        return self.selector.inputs() + [self.values]


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------


@dataclass(eq=False)
class Selector:
    def inputs(self) -> list[SOp]:
        return []


@dataclass(eq=False)
class Select(Selector):
    """RASP's categorical select: matrix[i][j] = predicate(keys[j], queries[i])."""

    keys: SOp = None
    queries: SOp = None
    predicate: Callable[[Value, Value], bool] = None

    def inputs(self) -> list[SOp]:
        return [self.keys, self.queries]


@dataclass(eq=False)
class SelectAll(Selector):
    """The constant-true selector. `aggregate(select_all(), v)` is the mean of v."""


@dataclass(eq=False)
class SelectAt(Selector):
    """The numeric selector. Row i selects the single position `addr[i]`."""

    addr: SOp = None

    def inputs(self) -> list[SOp]:
        return [self.addr]


# --------------------------------------------------------------------------
# surface API
# --------------------------------------------------------------------------


def tokens(name: str = "tokens", kind: str = "numerical") -> Tokens:
    return Tokens(name=name, kind=kind)


def indices(name: str = "indices") -> Indices:
    return Indices(name=name)


def smap(f: Callable[[Value], Value], x: SOp, name: str = "map", kind: str = "numerical") -> Map:
    return Map(f=f, x=x, name=name, kind=kind)


def sequence_map(
    f: Callable[[Value, Value], Value],
    x: SOp,
    y: SOp,
    name: str = "seqmap",
    kind: str = "numerical",
) -> SequenceMap:
    return SequenceMap(f=f, x=x, y=y, name=name, kind=kind)


def categorize(x: SOp, values, name: str = "cat") -> Categorize:
    return Categorize(x=x, values=tuple(float(v) for v in values), name=name, kind="categorical")


def select(keys: SOp, queries: SOp, predicate: Callable[[Value, Value], bool]) -> Select:
    return Select(keys=keys, queries=queries, predicate=predicate)


def select_all() -> SelectAll:
    return SelectAll()


def select_at(addr: SOp) -> SelectAt:
    return SelectAt(addr=addr)


def aggregate(
    selector: Selector, values: SOp, name: str = "agg", kind: str = "numerical"
) -> Aggregate:
    return Aggregate(selector=selector, values=values, name=name, kind=kind)


# convenience predicates
def EQ(k: Value, q: Value) -> bool:  # noqa: N802 - RASP spells these in caps
    return k == q


def LEQ(k: Value, q: Value) -> bool:  # noqa: N802
    return k <= q


def TRUE(k: Value, q: Value) -> bool:  # noqa: N802
    return True


# --------------------------------------------------------------------------
# interpreter
# --------------------------------------------------------------------------


def selector_matrix(selector: Selector, seq: Sequence[Value], cache: dict) -> np.ndarray:
    """The n x n boolean selection matrix, row = query position."""
    n = len(seq)
    if isinstance(selector, SelectAll):
        return np.ones((n, n), dtype=bool)
    if isinstance(selector, Select):
        keys = _eval(selector.keys, seq, cache)
        queries = _eval(selector.queries, seq, cache)
        m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                m[i, j] = bool(selector.predicate(keys[j], queries[i]))
        return m
    if isinstance(selector, SelectAt):
        addr = _eval(selector.addr, seq, cache)
        m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            a = addr[i]
            ai = int(round(float(a)))
            if ai != a:
                raise ValueError(f"select_at got a non-integer address {a!r} at position {i}")
            if 0 <= ai < n:
                m[i, ai] = True
        return m
    raise TypeError(f"unknown selector {selector!r}")


def _eval(sop: SOp, seq: Sequence[Value], cache: dict) -> np.ndarray:
    key = id(sop)
    if key in cache:
        return cache[key]
    n = len(seq)
    if isinstance(sop, Tokens):
        out = np.asarray(seq, dtype=float)
    elif isinstance(sop, Indices):
        out = np.arange(n, dtype=float)
    elif isinstance(sop, Map):
        x = _eval(sop.x, seq, cache)
        out = np.asarray([sop.f(v) for v in x], dtype=float)
    elif isinstance(sop, SequenceMap):
        x = _eval(sop.x, seq, cache)
        y = _eval(sop.y, seq, cache)
        out = np.asarray([sop.f(a, b) for a, b in zip(x, y)], dtype=float)
    elif isinstance(sop, Categorize):
        out = _eval(sop.x, seq, cache)
    elif isinstance(sop, Aggregate):
        m = selector_matrix(sop.selector, seq, cache)
        v = _eval(sop.values, seq, cache)
        counts = m.sum(axis=1)
        # RASP's aggregate averages the selected values; an empty row gives the
        # default 0. select_at selects at most one position, so its aggregate is
        # a plain read.
        sums = m.astype(float) @ v
        out = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    else:
        raise TypeError(f"unknown s-op {sop!r}")
    cache[key] = out
    return out


def evaluate(sop: SOp, seq: Sequence[Value]) -> np.ndarray:
    """Run a program on one input sequence."""
    return _eval(sop, seq, {})


def topo(sop: SOp) -> list[SOp]:
    """Every s-op the program reaches, inputs before consumers."""
    order: list[SOp] = []
    seen: set[int] = set()

    def visit(node: SOp) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        for dep in node.inputs():
            visit(dep)
        order.append(node)

    visit(sop)
    return order
