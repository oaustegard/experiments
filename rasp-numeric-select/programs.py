"""Three programs whose addresses are computed, plus one categorical control.

Each of the three reads position `f(i, x)` where f depends on the input or on
arithmetic over the index. A categorical compiler reaches such a position only
by one-hot encoding it, which fixes a maximum length at compile time. The
`select_at` compilation carries three position features regardless of length,
so the same weights run at any n.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import rasp_ns as R


@dataclass
class Program:
    key: str
    title: str
    build: Callable[[], R.SOp]
    sample: Callable[[np.random.Generator, int], list[float]]
    vocab: list | None = None
    note: str = ""


# -- (a) gather by a computed index ---------------------------------------


def _gather() -> R.SOp:
    idx = R.indices()
    addr = R.smap(lambda i: 2 * i + 1, idx, name="addr")
    return R.aggregate(R.select_at(addr), R.tokens(), name="gather")


def _sample_small(rng: np.random.Generator, n: int) -> list[float]:
    return [float(v) for v in rng.integers(0, 10, size=n)]


# -- (b) pointer chase -----------------------------------------------------


def _chase() -> R.SOp:
    tok = R.tokens()
    return R.aggregate(R.select_at(tok), tok, name="chase")


def _sample_pointers(rng: np.random.Generator, n: int) -> list[float]:
    return [float(v) for v in rng.integers(0, n, size=n)]


# -- (c) shift by a data-dependent offset ---------------------------------


def _shift() -> R.SOp:
    tok = R.tokens()
    idx = R.indices()
    zero = R.smap(lambda i: 0.0 * i, idx, name="zero")
    head = R.aggregate(R.select_at(zero), tok, name="x0")
    addr = R.sequence_map(lambda i, x0: i - x0, idx, head, name="addr")
    return R.aggregate(R.select_at(addr), tok, name="shift")


def _sample_shift(rng: np.random.Generator, n: int) -> list[float]:
    seq = [float(v) for v in rng.integers(0, 10, size=n)]
    seq[0] = float(rng.integers(0, min(8, n)))
    return seq


# -- (d) categorical control, compiled the Tracr way ----------------------


def _match() -> R.SOp:
    tok = R.tokens(kind="categorical")
    return R.aggregate(R.select(tok, tok, R.EQ), R.indices(), name="match")


def _sample_vocab(rng: np.random.Generator, n: int) -> list[float]:
    return [float(v) for v in rng.integers(0, 5, size=n)]


PROGRAMS: dict[str, Program] = {
    "gather": Program(
        key="gather",
        title="y[i] = x[2i + 1]",
        build=_gather,
        sample=_sample_small,
        note="address is arithmetic in the index and leaves the sequence for i >= (n-1)/2",
    ),
    "chase": Program(
        key="chase",
        title="y[i] = x[x[i]]",
        build=_chase,
        sample=_sample_pointers,
        note="address is the token itself; the read value grows with n",
    ),
    "shift": Program(
        key="shift",
        title="y[i] = x[i - x[0]]",
        build=_shift,
        sample=_sample_shift,
        note="two chained numeric selects; the offset is broadcast by select_at(0)",
    ),
    "match": Program(
        key="match",
        title="y[i] = mean{ j : x[j] == x[i] }",
        build=_match,
        sample=_sample_vocab,
        vocab=[0.0, 1.0, 2.0, 3.0, 4.0],
        note="categorical control, compiled the Tracr way over a 5-symbol one-hot",
    ),
}

NUMERIC_KEYS = ["gather", "chase", "shift"]


def categorical_gather(n_max: int) -> R.SOp:
    """`gather` written with a categorical select over positions.

    This is the only route a Tracr-shaped compiler has to a computed position:
    one-hot the index, one-hot the address, match them with EQ. Both one-hots
    are `n_max` wide and are fixed when the weights are built.
    """
    idx = R.indices()
    addr = R.smap(lambda i: 2 * i + 1, idx, name="addr")
    grid = list(range(n_max))
    return R.aggregate(
        R.select(
            R.categorize(idx, grid, name="keypos"),
            R.categorize(addr, grid, name="qpos"),
            R.EQ,
        ),
        R.tokens(),
        name="cat_gather",
    )


def cost_comparison(n_max_values=(8, 16, 32, 64), probe_lengths=(8, 16, 24, 40, 96)) -> list[dict]:
    """Width and reach of the two compilations of the same program."""
    import compile_ns as C

    rng = np.random.default_rng(1)
    rows = []
    numeric = PROGRAMS["gather"].build()
    nm = C.compile_program(numeric)
    rows.append(
        {
            "route": "select_at (parabolic)",
            "n_max": None,
            "d_model": nm.layout.width,
            "n_params": nm.n_params,
            "correct_up_to": max(probe_lengths),
        }
    )
    for n_max in n_max_values:
        prog = categorical_gather(n_max)
        model = C.compile_program(prog)
        reach = 0
        for n in sorted(probe_lengths):
            seq = [float(v) for v in rng.integers(0, 10, size=n)]
            err = float(
                np.abs(R.evaluate(PROGRAMS["gather"].build(), seq) - model.run(seq, mode="hard")).max()
            )
            if err > 1e-9:
                break
            reach = n
        rows.append(
            {
                "route": f"select(indices, addr, EQ), n_max={n_max}",
                "n_max": n_max,
                "d_model": model.layout.width,
                "n_params": model.n_params,
                "correct_up_to": reach,
            }
        )
    return rows


def check(key: str, lengths=(8, 32, 128), trials: int = 8, seed: int = 0) -> dict:
    """Compile once, then compare the compiled model against the interpreter."""
    import compile_ns as C

    prog = PROGRAMS[key]
    sop = prog.build()
    model = C.compile_program(sop, vocab=prog.vocab)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n in lengths:
        for _ in range(trials):
            seq = prog.sample(rng, n)
            ref = R.evaluate(sop, seq)
            got = model.run(seq, mode="hard")
            worst = max(worst, float(np.abs(ref - got).max()))
    return {
        "key": key,
        "title": prog.title,
        "d_model": model.layout.width,
        "n_stages": len(model.stages),
        "n_params": model.n_params,
        "max_abs_error": worst,
    }


if __name__ == "__main__":
    rows = [check(k) for k in PROGRAMS]
    print(f"{'program':10s} {'d_model':>8s} {'stages':>7s} {'params':>8s} {'max err':>10s}")
    for r in rows:
        print(
            f"{r['key']:10s} {r['d_model']:8d} {r['n_stages']:7d} "
            f"{r['n_params']:8d} {r['max_abs_error']:10.3e}   {r['title']}"
        )
    print()
    print(f"{'route':40s} {'d_model':>8s} {'params':>8s} {'correct up to n':>16s}")
    for r in cost_comparison():
        print(
            f"{r['route']:40s} {r['d_model']:8d} {r['n_params']:8d} {r['correct_up_to']:16d}"
        )
