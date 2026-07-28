"""Calibration gates, per issue #163. Must all pass before any search runs.

Implemented (gate numbering matches the issue):
  1. Reproduce Fact 7.1 for Schrijver's (D1,u1): nu=1, tau=2.
  3. Verify the odd-ring generalization (i=5) is a counterexample and the
     even-ring analogue (i=4) is not, using the same ring/path construction
     pattern as D1 (generalized programmatically, not re-transcribed by hand
     from Figure 8 -- see generator/ring.py for the construction and its
     own docstring on what is and isn't verified against the figure).
  4. Sanity: random small DAGs passing the hard pruning filters (source-
     sink-connected, or single-source/sink) should always yield a packing
     (conjecture-consistent, since these classes are proven good).

NOT implemented in this pass (see experiments/woodall/README.md "Deferred"):
  2. Cornuejols-Guenin (D2,u2)/(D3,u3) calibration. Figure 9/10 in the
     survey involve ~14+ vertices with vertex numbers referenced directly
     in the text (Williams' vertices 14 and 8) -- transcribing those
     correctly from the rendered figure needs the same careful
     crop-and-cross-check process used for D1 below, which is its own
     multi-hour task. Flagged, not silently skipped: gate 2 is RED.
"""

import random
import sys

sys.path.insert(0, "..")

from verifier.digraph import Digraph
from verifier.packing import nu, nu_at_least
from calibration.schrijver_d1 import build as build_d1, SPECIAL_JOINS, ACTIVE_ARCS
from generator.ring import build_ring_counterexample


def gate1_schrijver():
    D = build_d1()
    assert D.is_dag(), "D1 must be a DAG"
    tau, _ = D.tau()
    assert tau == 2, f"Fact 7.1 requires tau(D1,u1)=2, got {tau}"
    n = nu(D)
    assert n == 1, f"Fact 7.1 requires nu(D1,u1)=1, got {n}"
    for name, J in SPECIAL_JOINS.items():
        assert D.is_dijoin(J), f"{name} must be a valid dijoin"
        assert D.contract_and_check_strong(J), f"{name} must make D1 strongly connected when contracted"
    assert D.is_dijoin(ACTIVE_ARCS)
    print("GATE 1 (Schrijver D1): PASS  tau=2 nu=1, all special joins verified")


def gate3_ring_parity():
    for i, expect_counterexample in [(3, True), (4, False), (5, True)]:
        D = build_ring_counterexample(i)
        assert D.is_dag()
        tau, _ = D.tau()
        n = nu(D)
        is_ce = n < tau
        status = "counterexample" if is_ce else "NOT a counterexample (nu=tau)"
        print(f"  ring i={i}: tau={tau} nu={n} -> {status}")
        assert is_ce == expect_counterexample, (
            f"ring i={i}: expected counterexample={expect_counterexample}, "
            f"got nu={n} tau={tau}"
        )
    print("GATE 3 (ring parity, i=3,4,5): PASS")


def random_dag(n_vertices, p_edge, seed):
    rng = random.Random(seed)
    verts = list(range(n_vertices))
    order = verts[:]
    rng.shuffle(order)
    arcs = []
    lab = 0
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if rng.random() < p_edge:
                arcs.append((order[i], order[j], f"e{lab}", 1))
                lab += 1
    return Digraph(verts, arcs)


def is_source_sink_connected(D):
    sources = [v for v in D.vertices if not D.in_arcs(v)]
    sinks = [v for v in D.vertices if not D.out_arcs(v)]
    if not sources or not sinks:
        return False
    adj = {v: set(v2 for _, v2, *_ in D.out_arcs(v)) for v in D.vertices}

    def reach(v):
        seen = {v}
        stack = [v]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return seen

    for s in sources:
        r = reach(s)
        if not all(t in r for t in sinks):
            return False
    return True


def gate4_random_sanity(trials=25, n_vertices=7, p_edge=0.35):
    checked = 0
    seed = 0
    while checked < trials:
        seed += 1
        D = random_dag(n_vertices, p_edge, seed)
        if not D.is_dag() or not D.arcs:
            continue
        if not is_source_sink_connected(D):
            continue
        tau, _ = D.tau()
        if tau == 0:
            continue
        n = nu(D)
        assert n == tau, (
            f"seed={seed}: source-sink-connected DAG should satisfy EG "
            f"(nu=tau) but got nu={n} tau={tau} -- either a verifier bug "
            f"or a genuine counterexample (extremely unlikely at this size, "
            f"would itself be a major finding)"
        )
        checked += 1
    print(f"GATE 4 (random sanity, {checked} source-sink-connected DAGs): PASS, all satisfy nu=tau")


if __name__ == "__main__":
    gate1_schrijver()
    print("GATE 2 (Cornuejols-Guenin D2/D3): SKIPPED -- not transcribed this pass, see README")
    gate3_ring_parity()
    gate4_random_sanity()
    print()
    print("ALL IMPLEMENTED GATES PASSED")
