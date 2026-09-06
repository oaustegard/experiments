"""Gates for the ALTA residual-code experiment.

Two of these are the pre-registered gates: the compiled model has to agree
with the symbolic interpreter on 50 random inputs per program, and the
identity code at `d = D` has to reproduce the compiled model exactly. The rest
guard the instrumentation that the trainer's claims rest on.

Run: ``pytest -q`` in this directory. SUBLEQ compiles 43k lookup rules, so the
whole file takes a few minutes.
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

import alta_common as A
import train_code as T

GATE_INPUTS = 50


@functools.lru_cache(maxsize=None)
def compiled(name: str) -> A.Compiled:
    return A.compile_case(A.get_case(name))


@pytest.mark.parametrize("name", A.CASE_ORDER)
def test_compiled_matches_interpreter(name):
    """The pre-registered gate: compiled weights == symbolic interpreter."""
    model = compiled(name)
    for input_ids in A.sample_inputs(model.case, GATE_INPUTS, seed=7):
        want = [int(v) for v in A.run_interpreter(model, input_ids)]
        got = [int(v) for v in A.run_compiled(model, input_ids)]
        assert got == want, f"{name}: compiled output diverged from interpreter"
        assert model.case.reference(input_ids, want), f"{name}: interpreter is wrong"


@pytest.mark.parametrize("name", A.CASE_ORDER)
def test_identity_code_reproduces_compiled(name):
    """The identity code is the compiled model, to the last bit."""
    model = compiled(name)
    identity = np.eye(model.dim)
    for input_ids in A.sample_inputs(model.case, 5, seed=11):
        want = [int(v) for v in A.run_compiled(model, input_ids)]
        got = [int(v) for v in A.run_compressed(model, input_ids, identity, identity)]
        assert got == want


@pytest.mark.parametrize("name", A.CASE_ORDER)
def test_gauge_removal_is_lossless(name):
    """Dropping never-used dimensions is exact, not a compression."""
    model = compiled(name)
    train_inputs = A.training_inputs(model.case)
    problem = T.Problem(model, train_inputs)
    code = np.eye(model.dim)[:, problem.used]
    margin, tol = problem.parts(*(2 * [__import__("torch").tensor(code)]))
    assert float(margin) == 0.0 and float(tol) == 0.0
    for input_ids in A.sample_inputs(model.case, 5, seed=13, exclude=train_inputs):
        want = [int(v) for v in A.run_compiled(model, input_ids)]
        assert [int(v) for v in A.run_compressed(model, input_ids, code, code)] == want


@pytest.mark.parametrize("name", A.CASE_ORDER)
def test_live_dims_are_read_and_used(name):
    model = compiled(name)
    problem = T.Problem(model, A.training_inputs(model.case))
    assert (problem.live <= problem.read).all()
    assert (problem.live <= problem.used).all()
    assert problem.live.sum() > 0
    assert (A.tolerances(model) > 0).all()


@pytest.mark.parametrize("name", A.CASE_ORDER)
def test_training_set_covers_evaluation_set(name):
    """A code cannot be asked to preserve a feature it never saw."""
    model = compiled(name)
    train_inputs = A.training_inputs(model.case)
    eval_inputs = A.sample_inputs(model.case, 20, seed=2, exclude=train_inputs)
    train_live = T.Problem(model, train_inputs).live
    eval_live = T.Problem(model, eval_inputs).live
    assert not (eval_live & ~train_live).any()


@pytest.mark.parametrize("name", A.CASE_ORDER)
def test_sparse_ffn_matches_alta(name):
    """The batched sparse forward pass is ALTA's, not an approximation of it."""
    model = compiled(name)
    for input_ids in A.sample_inputs(model.case, 3, seed=17):
        want = [int(v) for v in A.run_compiled_reference(model, input_ids)]
        assert [int(v) for v in A.run_compiled(model, input_ids)] == want
