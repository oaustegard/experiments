"""Guards for the interpreter, the compiler, the three programs and the margin."""

from __future__ import annotations

import numpy as np
import pytest

import compile_ns as C
import margin as M
import programs as P
import rasp_ns as R

LENGTHS = (8, 32, 128)


# -- 1. interpreter semantics ---------------------------------------------


def test_tokens_and_indices():
    seq = [3.0, 1.0, 4.0, 1.0]
    assert R.evaluate(R.tokens(), seq).tolist() == seq
    assert R.evaluate(R.indices(), seq).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_map_and_sequence_map():
    seq = [3.0, 1.0, 4.0, 1.0]
    doubled = R.smap(lambda v: 2 * v, R.tokens())
    assert R.evaluate(doubled, seq).tolist() == [6.0, 2.0, 8.0, 2.0]
    summed = R.sequence_map(lambda a, b: a + b, R.tokens(), R.indices())
    assert R.evaluate(summed, seq).tolist() == [3.0, 2.0, 6.0, 4.0]


def test_categorical_aggregate_averages_and_defaults():
    seq = [1.0, 2.0, 1.0, 5.0]
    tok = R.tokens(kind="categorical")
    same = R.aggregate(R.select(tok, tok, R.EQ), R.indices())
    # positions 0 and 2 both hold 1, so each reads the mean of {0, 2}
    assert R.evaluate(same, seq).tolist() == [1.0, 1.0, 1.0, 3.0]
    never = R.aggregate(R.select(tok, tok, lambda k, q: False), R.indices())
    assert R.evaluate(never, seq).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_select_all_is_the_mean():
    seq = [1.0, 2.0, 3.0, 4.0]
    mean = R.aggregate(R.select_all(), R.tokens())
    assert R.evaluate(mean, seq).tolist() == [2.5] * 4


def test_select_at_reads_and_defaults_out_of_range():
    seq = [10.0, 20.0, 30.0, 40.0]
    addr = R.smap(lambda i: 2 * i, R.indices())
    read = R.aggregate(R.select_at(addr), R.tokens())
    assert R.evaluate(read, seq).tolist() == [10.0, 30.0, 0.0, 0.0]
    back = R.aggregate(R.select_at(R.smap(lambda i: i - 1, R.indices())), R.tokens())
    assert R.evaluate(back, seq).tolist() == [0.0, 10.0, 20.0, 30.0]


def test_select_at_rejects_a_non_integer_address():
    seq = [1.0, 2.0]
    half = R.aggregate(R.select_at(R.smap(lambda i: i + 0.5, R.indices())), R.tokens())
    with pytest.raises(ValueError):
        R.evaluate(half, seq)


def test_categorize_is_semantically_the_identity():
    seq = [0.0, 1.0, 2.0]
    cat = R.categorize(R.indices(), [0, 1, 2])
    assert R.evaluate(cat, seq).tolist() == [0.0, 1.0, 2.0]


# -- 2. compiler ----------------------------------------------------------


def test_parabolic_head_has_head_dim_two():
    model = C.compile_program(P.PROGRAMS["gather"].build())
    heads = [h for st in model.stages for h in st.heads if "parabolic" in h.label]
    assert len(heads) == 1
    assert heads[0].W_Q.shape[1] == 2
    assert heads[0].W_K.shape[1] == 2


def test_residual_width_is_independent_of_length():
    """The whole claim in one assertion: nothing in the numeric compilation is
    sized by a sequence length, because no length is passed to the compiler."""
    widths = {C.compile_program(P.PROGRAMS[k].build()).layout.width for k in P.NUMERIC_KEYS}
    assert widths == {13, 11, 19}


def test_categorical_route_costs_a_position_table():
    small = C.compile_program(P.categorical_gather(8)).layout.width
    large = C.compile_program(P.categorical_gather(32)).layout.width
    # two categorize nodes (key and query), three dimensions per candidate each
    assert large - small == 6 * (32 - 8)


def test_categorical_route_breaks_past_its_table():
    model = C.compile_program(P.categorical_gather(16))
    ref = P.PROGRAMS["gather"].build()
    rng = np.random.default_rng(0)
    inside = [float(v) for v in rng.integers(0, 10, size=16)]
    assert np.abs(R.evaluate(ref, inside) - model.run(inside, mode="hard")).max() == 0.0
    outside = [float(v) for v in rng.integers(0, 10, size=40)]
    assert np.abs(R.evaluate(ref, outside) - model.run(outside, mode="hard")).max() > 0.0


def test_non_affine_map_is_refused():
    prog = R.aggregate(R.select_at(R.smap(lambda i: i * i, R.indices())), R.tokens())
    with pytest.raises(NotImplementedError):
        C.compile_program(prog)


# -- 3. compiled == interpreted -------------------------------------------


@pytest.mark.parametrize("key", ["gather", "chase", "shift", "match"])
@pytest.mark.parametrize("n", LENGTHS)
def test_compiled_matches_interpreter(key, n):
    prog = P.PROGRAMS[key]
    sop = prog.build()
    model = C.compile_program(sop, vocab=prog.vocab)
    rng = np.random.default_rng(hash((key, n)) % 2**32)
    for _ in range(4):
        seq = prog.sample(rng, n)
        ref = R.evaluate(sop, seq)
        got = model.run(seq, mode="hard")
        assert np.abs(ref - got).max() < 1e-9


@pytest.mark.parametrize("key", P.NUMERIC_KEYS)
def test_one_compilation_runs_far_past_any_compile_time_length(key):
    """Compiled once, no length ever supplied, then run at 1000."""
    prog = P.PROGRAMS[key]
    sop = prog.build()
    model = C.compile_program(sop)
    rng = np.random.default_rng(7)
    for n in (3, 257, 1000):
        seq = prog.sample(rng, n)
        assert np.abs(R.evaluate(sop, seq) - model.run(seq, mode="hard")).max() < 1e-9


@pytest.mark.parametrize("key", P.NUMERIC_KEYS)
def test_float32_weights_still_read_exactly_at_128(key):
    """j^2 = 16129 at n = 128, well under the 2^24 float32 integer ceiling."""
    prog = P.PROGRAMS[key]
    sop = prog.build()
    model = C.compile_program(sop, dtype=np.float32)
    rng = np.random.default_rng(11)
    seq = prog.sample(rng, 128)
    assert np.abs(R.evaluate(sop, seq) - model.run(seq, mode="hard")).max() < 1e-3


# -- 4. margin ------------------------------------------------------------


def test_gap_is_exactly_one_for_every_integer_address():
    for row in M.gap_table(lengths=(8, 32, 128, 512)):
        assert row["exactly_one"], row


def test_beta_threshold_exists_for_every_program_and_length():
    table = M.sweep(lengths=LENGTHS)
    for key, row in table.items():
        for n, thr in row.items():
            assert thr is not None, (key, n)
            assert thr <= 12.0, (key, n, thr)


def test_beta_threshold_grows_with_length_when_the_read_values_do():
    """`chase` reads a token drawn from 0..n-1, so the softmax leak it has to
    beat scales with n and the threshold has to rise. `gather` and `shift` read
    from a fixed 0..9 alphabet, so their thresholds stay flat -- within one
    0.25 grid step across a 16x change in length."""
    chase = [M.beta_threshold("chase", n) for n in LENGTHS]
    assert chase == sorted(chase)
    assert chase[-1] > chase[0]
    for key in ("gather", "shift"):
        flat = [M.beta_threshold(key, n) for n in LENGTHS]
        assert max(flat) - min(flat) <= 0.5, (key, flat)


def test_a_larger_range_gate_bound_costs_more_beta():
    seen = [M.beta_threshold("shift", 32, gate_bound=m) for m in (1e2, 1e3, 1e4, 1e6)]
    assert seen == sorted(seen)
    assert seen[-1] > seen[0]
