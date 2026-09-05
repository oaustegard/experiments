"""Compile a `rasp_ns` program to explicit numpy weight matrices.

The target is a standard encoder transformer: residual stream, multi-head
attention with per-head W_Q / W_K / W_V / W_O, and a ReLU MLP per block. Every
weight is a numpy array built at compile time; the forward pass in `Model.run`
does nothing a transformer does not do.

Two selector compilations live here.

Categorical `select`, compiled the Tracr way. Keys and queries occupy one-hot
subspaces of width |values|, W_K reads the key one-hot, W_Q reads the query
one-hot through the predicate matrix P[q, k], and the score is `S * P[q, k]`
for a large S. A BOS column scoring `S / 2` and carrying value 0 supplies the
default for a query that matches nothing. The one-hot subspace is the cost, and
`categorize` shows it directly: one residual dimension per candidate value,
fixed when the weights are built.

Numeric `select_at`, compiled to one head of head_dim 2. Position j writes the
key `(2j, -j^2)`; the query for a computed address x is `(x, 1)`; the score is
`2xj - j^2 = x^2 - (j - x)^2`, maximized at j = x. The head needs three
position features -- 1, j, j^2 -- and that count does not depend on sequence
length, so a model compiled once runs at any length.

`select_at` out of range returns 0. A range gate does that: the MLP computes
`below = relu(-x)`, `above = relu(x - L + 1)`, `v = relu(1 - below - above)`,
which is exactly 1 for an integer address in `0 .. L-1` and exactly 0
otherwise, then multiplies the head output by v with two more ReLUs. L itself
comes from a uniform-attention head reading `indices`: its mean is (L-1)/2.

Attention runs in one of two modes. `hard` is average-hard attention -- uniform
over the argmax set, the setting B-RASP works in. `soft` is
`softmax(beta * score)` for a key scale beta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

import rasp_ns as R

GATE_BOUND = 1.0e3  # |value| ceiling of the range gate; see _build_mlp
CATEGORICAL_SCALE = 100.0

DimRef = "str | tuple[str, int]"


# --------------------------------------------------------------------------
# residual layout
# --------------------------------------------------------------------------


class Layout:
    """Named slices of the residual stream."""

    def __init__(self) -> None:
        self.width = 0
        self.slots: dict[str, slice] = {}

    def alloc(self, name: str, size: int = 1) -> slice:
        if name in self.slots:
            raise KeyError(f"duplicate residual slot {name}")
        s = slice(self.width, self.width + size)
        self.slots[name] = s
        self.width += size
        return s

    def __getitem__(self, name: str) -> slice:
        return self.slots[name]

    def index(self, ref) -> int:
        if isinstance(ref, tuple):
            name, off = ref
            return self.slots[name].start + off
        s = self.slots[ref]
        assert s.stop - s.start == 1, f"slot {ref} is not scalar"
        return s.start

    def size(self, name: str) -> int:
        s = self.slots[name]
        return s.stop - s.start


# --------------------------------------------------------------------------
# stage descriptions
# --------------------------------------------------------------------------


@dataclass
class Head:
    W_Q: np.ndarray
    W_K: np.ndarray
    W_V: np.ndarray
    W_O: np.ndarray
    bos_score: float | None = None  # None means no BOS default column
    label: str = ""


@dataclass
class MLP:
    W_in: np.ndarray
    b_in: np.ndarray
    W_out: np.ndarray
    label: str = ""


@dataclass
class Stage:
    kind: str  # 'attn' or 'mlp'
    heads: list[Head] = field(default_factory=list)
    mlp: MLP | None = None


# --------------------------------------------------------------------------
# the compiled model
# --------------------------------------------------------------------------


@dataclass
class Model:
    layout: Layout
    stages: list[Stage]
    W_embed_num: np.ndarray  # (4, d) applied to [token, 1, j, j^2]
    W_embed_cat: np.ndarray | None  # (V, d) applied to one-hot(token)
    vocab: list | None
    out_slot: str
    dtype: type = np.float64

    def embed(self, seq: Sequence[float]) -> np.ndarray:
        n = len(seq)
        j = np.arange(n, dtype=self.dtype)
        feats = np.stack(
            [np.asarray(seq, dtype=self.dtype), np.ones(n, dtype=self.dtype), j, j * j],
            axis=1,
        )
        x = feats @ self.W_embed_num.astype(self.dtype)
        if self.W_embed_cat is not None:
            onehot = np.zeros((n, len(self.vocab)), dtype=self.dtype)
            for i, t in enumerate(seq):
                if t not in self.vocab:
                    raise ValueError(f"token {t!r} is outside the compiled vocabulary")
                onehot[i, self.vocab.index(t)] = 1.0
            x = x + onehot @ self.W_embed_cat.astype(self.dtype)
        return x

    def attend(self, x: np.ndarray, head: Head, mode: str, beta: float) -> np.ndarray:
        n = x.shape[0]
        q = x @ head.W_Q.astype(self.dtype)
        k = x @ head.W_K.astype(self.dtype)
        scores = q @ k.T
        if head.bos_score is not None:
            bos = np.full((n, 1), head.bos_score, dtype=self.dtype)
            scores = np.concatenate([scores, bos], axis=1)
        if mode == "hard":
            top = scores.max(axis=1, keepdims=True)
            tol = 1e-6 * np.maximum(1.0, np.abs(top))
            win = (scores >= top - tol).astype(self.dtype)
            probs = win / win.sum(axis=1, keepdims=True)
        elif mode == "soft":
            s = beta * scores
            s = s - s.max(axis=1, keepdims=True)
            e = np.exp(s)
            probs = e / e.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"unknown attention mode {mode!r}")
        v = x @ head.W_V.astype(self.dtype)
        out = probs[:, :n] @ v  # the BOS column carries value 0
        return out @ head.W_O.astype(self.dtype)

    def run(self, seq: Sequence[float], mode: str = "hard", beta: float = 100.0) -> np.ndarray:
        x = self.embed(seq)
        for stage in self.stages:
            if stage.kind == "attn":
                delta = np.zeros_like(x)
                for head in stage.heads:
                    delta = delta + self.attend(x, head, mode, beta)
                x = x + delta
            else:
                m = stage.mlp
                h = np.maximum(x @ m.W_in.astype(self.dtype) + m.b_in.astype(self.dtype), 0.0)
                x = x + h @ m.W_out.astype(self.dtype)
        return x[:, self.layout.index(self.out_slot)]

    @property
    def n_params(self) -> int:
        total = self.W_embed_num.size + (
            self.W_embed_cat.size if self.W_embed_cat is not None else 0
        )
        for stage in self.stages:
            for h in stage.heads:
                total += h.W_Q.size + h.W_K.size + h.W_V.size + h.W_O.size
            if stage.mlp is not None:
                total += stage.mlp.W_in.size + stage.mlp.b_in.size + stage.mlp.W_out.size
        return total


# --------------------------------------------------------------------------
# affine fitting
# --------------------------------------------------------------------------

_PROBE = np.arange(-6, 7, dtype=float)


def fit_affine_1(f) -> tuple[float, float]:
    """Recover (a, b) with f(x) = a x + b, or raise."""
    b = f(0.0)
    a = f(1.0) - b
    for x in _PROBE:
        if abs(f(float(x)) - (a * x + b)) > 1e-9:
            raise NotImplementedError(f"map {f!r} is not affine at x={x}")
    return float(a), float(b)


def fit_affine_2(f) -> tuple[float, float, float]:
    """Recover (a, b, c) with f(x, y) = a x + b y + c, or raise."""
    c = f(0.0, 0.0)
    a = f(1.0, 0.0) - c
    b = f(0.0, 1.0) - c
    for x in _PROBE:
        for y in _PROBE:
            if abs(f(float(x), float(y)) - (a * x + b * y + c)) > 1e-9:
                raise NotImplementedError(f"sequence_map {f!r} is not affine at ({x}, {y})")
    return float(a), float(b), float(c)


# --------------------------------------------------------------------------
# scheduling
# --------------------------------------------------------------------------


class _Builder:
    """Places tasks on alternating attention / MLP stages.

    Even stages hold attention heads, odd stages hold one MLP. A task lands in
    the first stage of its own kind that follows every stage its inputs were
    written by.
    """

    def __init__(self) -> None:
        self.stage_heads: dict[int, list[Head]] = {}
        self.stage_units: dict[int, list[tuple]] = {}
        self.ready: dict[str, int] = {}

    def when(self, kind: str, deps: Sequence[str]) -> int:
        base = max([self.ready.get(d, -1) for d in deps], default=-1) + 1
        if (base % 2 == 0) != (kind == "attn"):
            base += 1
        return base

    def add_head(self, head: Head, deps: Sequence[str], produces: Sequence[str]) -> None:
        s = self.when("attn", deps)
        self.stage_heads.setdefault(s, []).append(head)
        for p in produces:
            self.ready[p] = s

    def add_unit(self, unit: tuple, deps: Sequence[str], produces: Sequence[str]) -> None:
        s = self.when("mlp", deps)
        self.stage_units.setdefault(s, []).append(unit)
        for p in produces:
            self.ready[p] = s


def _proj(layout: Layout, pairs: Sequence[tuple], cols: int) -> np.ndarray:
    w = np.zeros((layout.width, cols))
    for ref, col, weight in pairs:
        w[layout.index(ref), col] = weight
    return w


# --------------------------------------------------------------------------
# the compiler
# --------------------------------------------------------------------------


def compile_program(
    program: R.SOp,
    vocab: Sequence | None = None,
    dtype: type = np.float64,
    gate_bound: float = GATE_BOUND,
) -> Model:
    """Compile a rasp_ns program into a Model.

    `vocab` is required only when the program uses `tokens(kind="categorical")`.
    """
    nodes = R.topo(program)
    layout = Layout()
    layout.alloc("one")
    layout.alloc("pos")
    layout.alloc("pos2")

    # -- categorical blocks ----------------------------------------------
    cat_values: dict[int, tuple] = {}
    slot: dict[int, str] = {}
    needs_vocab = False
    for n in nodes:
        if n.kind != "categorical":
            continue
        if isinstance(n, R.Tokens):
            if vocab is None:
                raise ValueError("a categorical `tokens` needs an explicit vocab")
            needs_vocab = True
            cat_values[id(n)] = tuple(float(v) for v in vocab)
        elif isinstance(n, R.Categorize):
            cat_values[id(n)] = n.values
        else:
            raise NotImplementedError(f"cannot one-hot encode {n!r}")

    if needs_vocab:
        vocab = [float(v) for v in vocab]
        layout.alloc("tok_onehot", len(vocab))

    for i, n in enumerate(nodes):
        if n.kind == "categorical":
            if isinstance(n, R.Tokens):
                slot[id(n)] = "tok_onehot"
            else:
                name = f"c{i}_{n.name}"
                layout.alloc(name, len(cat_values[id(n)]))
                slot[id(n)] = name
            continue
        name = f"v{i}_{n.name}"
        layout.alloc(name)
        slot[id(n)] = name

    # -- scratch, allocated before any matrix is sized --------------------
    uses_select_at = any(
        isinstance(n, R.Aggregate) and isinstance(n.selector, R.SelectAt) for n in nodes
    )
    if uses_select_at:
        layout.alloc("len_raw")
        layout.alloc("length")
    for i, n in enumerate(nodes):
        if isinstance(n, R.Aggregate) and isinstance(n.selector, R.SelectAt):
            for suffix in ("read", "below", "above", "valid"):
                layout.alloc(f"s{i}_{suffix}")
        elif isinstance(n, R.Categorize):
            k = len(cat_values[id(n)])
            layout.alloc(f"c{i}_lo", k)
            layout.alloc(f"c{i}_hi", k)

    builder = _Builder()

    # -- embedding ---------------------------------------------------------
    W_num = np.zeros((4, layout.width))
    W_num[1, layout.index("one")] = 1.0
    W_num[2, layout.index("pos")] = 1.0
    W_num[3, layout.index("pos2")] = 1.0
    W_cat = np.zeros((len(vocab), layout.width)) if needs_vocab else None
    if needs_vocab:
        blk = layout["tok_onehot"]
        for vi in range(len(vocab)):
            W_cat[vi, blk.start + vi] = 1.0
        builder.ready["tok_onehot"] = -1

    for n in nodes:
        if isinstance(n, R.Tokens) and n.kind == "numerical":
            W_num[0, layout.index(slot[id(n)])] = 1.0
            builder.ready[slot[id(n)]] = -1
        elif isinstance(n, R.Indices):
            W_num[2, layout.index(slot[id(n)])] = 1.0
            builder.ready[slot[id(n)]] = -1
    for base in ("one", "pos", "pos2"):
        builder.ready[base] = -1

    # -- length, needed by every range gate --------------------------------
    if uses_select_at:
        head = Head(
            W_Q=np.zeros((layout.width, 1)),
            W_K=np.zeros((layout.width, 1)),
            W_V=_proj(layout, [("pos", 0, 1.0)], 1),
            W_O=np.zeros((1, layout.width)),
            label="length/uniform-mean",
        )
        head.W_O[0, layout.index("len_raw")] = 1.0
        builder.add_head(head, ["pos"], ["len_raw"])
        builder.add_unit(("affine", "length", [("len_raw", 2.0)], 1.0), ["len_raw"], ["length"])

    # -- one task per node -------------------------------------------------
    for i, n in enumerate(nodes):
        if isinstance(n, (R.Tokens, R.Indices)):
            continue
        out = slot[id(n)]
        if isinstance(n, R.Map):
            a, b = fit_affine_1(n.f)
            src = slot[id(n.x)]
            builder.add_unit(("affine", out, [(src, a)], b), [src], [out])
        elif isinstance(n, R.SequenceMap):
            a, b, c = fit_affine_2(n.f)
            sx, sy = slot[id(n.x)], slot[id(n.y)]
            builder.add_unit(("affine", out, [(sx, a), (sy, b)], c), [sx, sy], [out])
        elif isinstance(n, R.Categorize):
            _compile_categorize(n, i, slot, builder, cat_values[id(n)])
        elif isinstance(n, R.Aggregate):
            _compile_aggregate(n, i, layout, slot, builder, cat_values, out)
        else:
            raise NotImplementedError(f"cannot compile {n!r}")

    # -- materialize -------------------------------------------------------
    last = max(list(builder.stage_heads) + list(builder.stage_units), default=-1)
    stages: list[Stage] = []
    for s in range(last + 1):
        if s % 2 == 0:
            stages.append(Stage("attn", heads=builder.stage_heads.get(s, [])))
        else:
            stages.append(
                Stage("mlp", mlp=_build_mlp(layout, builder.stage_units.get(s, []), gate_bound))
            )

    return Model(
        layout=layout,
        stages=stages,
        W_embed_num=W_num,
        W_embed_cat=W_cat,
        vocab=list(vocab) if needs_vocab else None,
        out_slot=slot[id(program)],
        dtype=dtype,
    )


def _compile_categorize(n, i, slot, builder, values) -> None:
    """One-hot a numeric s-op with a triangular hat per candidate value.

    `hat_p(a) = relu(1 - relu(v_p - a) - relu(a - v_p))` is 1 at `a = v_p` and 0
    at every other integer. Two MLP stages, three hidden units and three
    residual dimensions per value.
    """
    src = slot[id(n.x)]
    out = slot[id(n)]
    lo, hi = f"c{i}_lo", f"c{i}_hi"
    for p, v in enumerate(values):
        builder.add_unit(("relu", (lo, p), [(src, -1.0)], float(v)), [src], [lo])
        builder.add_unit(("relu", (hi, p), [(src, 1.0)], -float(v)), [src], [hi])
    for p in range(len(values)):
        builder.add_unit(
            ("relu", (out, p), [((lo, p), -1.0), ((hi, p), -1.0)], 1.0),
            [lo, hi],
            [out],
        )


def _compile_aggregate(n, i, layout, slot, builder, cat_values, out) -> None:
    sel = n.selector
    if n.values.kind != "numerical":
        raise NotImplementedError("aggregate values must be numerical")
    vals = slot[id(n.values)]

    if isinstance(sel, R.SelectAll):
        head = Head(
            W_Q=np.zeros((layout.width, 1)),
            W_K=np.zeros((layout.width, 1)),
            W_V=_proj(layout, [(vals, 0, 1.0)], 1),
            W_O=np.zeros((1, layout.width)),
            label="uniform-mean",
        )
        head.W_O[0, layout.index(out)] = 1.0
        builder.add_head(head, [vals], [out])
        return

    if isinstance(sel, R.Select):
        if sel.keys.kind != "categorical" or sel.queries.kind != "categorical":
            raise NotImplementedError("categorical select needs categorical keys and queries")
        kv = cat_values[id(sel.keys)]
        qv = cat_values[id(sel.queries)]
        if kv != qv:
            raise NotImplementedError("keys and queries must share one value list")
        V = len(kv)
        P = np.zeros((V, V))
        for qi, q in enumerate(qv):
            for ki, k in enumerate(kv):
                P[qi, ki] = CATEGORICAL_SCALE if sel.predicate(k, q) else 0.0
        kslot, qslot = slot[id(sel.keys)], slot[id(sel.queries)]
        W_Q = np.zeros((layout.width, V))
        W_Q[layout[qslot], :] = P
        W_K = np.zeros((layout.width, V))
        W_K[layout[kslot], :] = np.eye(V)
        head = Head(
            W_Q=W_Q,
            W_K=W_K,
            W_V=_proj(layout, [(vals, 0, 1.0)], 1),
            W_O=np.zeros((1, layout.width)),
            bos_score=CATEGORICAL_SCALE / 2.0,
            label=f"categorical-select[{V}]",
        )
        head.W_O[0, layout.index(out)] = 1.0
        builder.add_head(head, [kslot, qslot, vals], [out])
        return

    if isinstance(sel, R.SelectAt):
        addr = slot[id(sel.addr)]
        read, below, above, valid = (
            f"s{i}_read",
            f"s{i}_below",
            f"s{i}_above",
            f"s{i}_valid",
        )
        head = Head(
            W_Q=_proj(layout, [(addr, 0, 1.0), ("one", 1, 1.0)], 2),
            W_K=_proj(layout, [("pos", 0, 2.0), ("pos2", 1, -1.0)], 2),
            W_V=_proj(layout, [(vals, 0, 1.0)], 1),
            W_O=np.zeros((1, layout.width)),
            label="parabolic-select_at",
        )
        head.W_O[0, layout.index(read)] = 1.0
        builder.add_head(head, [addr, vals], [read])
        builder.add_unit(("relu", below, [(addr, -1.0)], 0.0), [addr], [below])
        builder.add_unit(
            ("relu", above, [(addr, 1.0), ("length", -1.0)], 1.0), [addr, "length"], [above]
        )
        builder.add_unit(
            ("relu", valid, [(below, -1.0), (above, -1.0)], 1.0), [below, above], [valid]
        )
        builder.add_unit(("gate", out, read, valid), [read, valid], [out])
        return

    raise NotImplementedError(f"cannot compile selector {sel!r}")


def _build_mlp(layout: Layout, units: Sequence[tuple], gate_bound: float = GATE_BOUND) -> MLP:
    """Pack units into one ReLU MLP.

    affine: u = sum(c*x) + b, as relu(u) - relu(-u), exact for any u.
    relu:   u = relu(sum(c*x) + b), one hidden unit.
    gate:   v * val for v in {0, 1}, as relu(val + Mv - M) - relu(-val + Mv - M),
            exact whenever |val| <= M. M also sets how sharp softmax attention
            has to be: an error eps in v moves the output by M * eps, so a large
            M buys value range at the price of a larger beta.
    """
    cols_in: list[np.ndarray] = []
    bias: list[float] = []
    cols_out: list[np.ndarray] = []
    d = layout.width

    def push(w_in: np.ndarray, b: float, w_out: np.ndarray) -> None:
        cols_in.append(w_in)
        bias.append(b)
        cols_out.append(w_out)

    def onehot(ref) -> np.ndarray:
        o = np.zeros(d)
        o[layout.index(ref)] = 1.0
        return o

    def combo(terms) -> np.ndarray:
        w = np.zeros(d)
        for ref, c in terms:
            w[layout.index(ref)] += c
        return w

    for unit in units:
        if unit[0] == "affine":
            _, out, terms, b = unit
            w, o = combo(terms), onehot(out)
            push(w, b, o)
            push(-w, -b, -o)
        elif unit[0] == "relu":
            _, out, terms, b = unit
            push(combo(terms), b, onehot(out))
        elif unit[0] == "gate":
            _, out, val, v = unit
            o = onehot(out)
            push(combo([(val, 1.0), (v, gate_bound)]), -gate_bound, o)
            push(combo([(val, -1.0), (v, gate_bound)]), -gate_bound, -o)
        else:
            raise NotImplementedError(f"unknown MLP unit {unit[0]!r}")

    if not cols_in:
        return MLP(np.zeros((d, 0)), np.zeros(0), np.zeros((0, d)), label="empty")
    return MLP(
        W_in=np.stack(cols_in, axis=1),
        b_in=np.asarray(bias, dtype=float),
        W_out=np.stack(cols_out, axis=0),
        label=f"{len(cols_in)} hidden",
    )
