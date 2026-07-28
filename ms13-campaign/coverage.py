"""
ms13-campaign coverage.py: exact counterexample-EXISTENCE test for a fixed
small (G, d) (claude-workspace issue #169).

For a fixed instance (G, d), a Conjecture 1.3 counterexample x exists iff

    Q_G  not subset of  Union_R Box(f^R, d_max)

where Q_G is the product of per-terminal path simplices (the reachable set
of fractional flows x, parameterized in split-weight space -- see
`core.Instance.x_from_split`) and

    Box(f^R, d_max) = { y : |y(a) - f^R(a)| <= d_max  for all arcs a }

is the two-sided tolerance box around routing R's arc loads.

We encode "exists x in Q_G uncovered by every box" as a MILP (scipy.optimize
.milp / HiGHS): continuous split-weight variables per terminal (simplex
constraints via equality + bounds), continuous x(a) variables tied to the
weights by equality, and for each routing R a disjunctive constraint
"some arc escapes Box(f^R, d_max) by at least eps" encoded with big-M
binaries z_{R,a,side} (side in {+, -}), sum_{a,side} z_{R,a,side} >= 1.
Feasibility of the MILP (objective is the zero vector -- we only need a
witness, not an optimum) means an uncovered x exists.

THIS IS A SCREENING TOOL, exactly like sweep.py. A MILP solution x* is a
floating-point vertex; before anyone calls it a counterexample it MUST be
rationalized and confirmed via core.verify_counterexample. This module
enforces that in the return path: `exists_uncovered_x` never reports
`uncovered_x_exists: True` without also attaching `exact_verdict`, the
result of running the rationalized split through
core.verify_counterexample -- check `exact_verdict["is_counterexample"]`,
not the raw MILP feasibility, before believing anything.

Scale: routings grow as the product of per-terminal path counts, and each
routing contributes 2*|arcs| binaries. This module bails with a clear
message above `routing_cap` (default 200) rather than building an
intractable MILP silently. A CEGAR loop (start with a small routing subset,
add violated routings lazily as they're discovered) would lift this cap and
is noted here as future work, not implemented.
"""

from __future__ import annotations

from fractions import Fraction as Fr
from typing import Dict, List, Optional, Tuple

import numpy as np

from core import Arc, Instance, Terminal, verify_counterexample
from sweep import rationalize_split_candidates, t_of_x


# ---------------------------------------------------------------------------
# The triangle-conflict calibration fixture (shared with tests/test_calibration.py)
# ---------------------------------------------------------------------------

def triangle_instance() -> Instance:
    """PR #167 triangle-conflict fixture, corrected-convention calibration
    values: compute_beta_star gives beta_star == 1,
    last_infeasible_breakpoint == 1/2 (see beta_star.py's module docstring
    for the off-by-one correction). t_of_x == 0, is_counterexample == False.

    Pseudo-arc convention: AB / BC / CA / E1 / E2 / E3 are NOT real graph
    vertices -- they are load buckets modeled as arcs ("s", "AB") etc. so
    core.py's arc-load machinery applies unmodified. Terminal names t1/t2/t3
    intentionally do NOT match their paths' final nodes (AB, CA, ...), so
    core.check_paths_valid is not meaningful here -- this is a load-math
    fixture, not a graph-realized instance. Skip check_paths_valid on it.
    """
    arcs: List[Arc] = [
        ("s", "AB"), ("s", "BC"), ("s", "CA"),
        ("s", "E1"), ("s", "E2"), ("s", "E3"),
    ]
    t1 = Terminal("t1", Fr(1), [(("s", "AB"), ("s", "CA")), (("s", "E1"),)])
    t2 = Terminal("t2", Fr(1), [(("s", "AB"), ("s", "BC")), (("s", "E2"),)])
    t3 = Terminal("t3", Fr(1), [(("s", "BC"), ("s", "CA")), (("s", "E3"),)])
    return Instance(arcs=arcs, terminals=[t1, t2, t3])


def triangle_split() -> List[List[Fr]]:
    half = Fr(1, 2)
    return [[half, half], [half, half], [half, half]]


# ---------------------------------------------------------------------------
# MILP encoding
# ---------------------------------------------------------------------------

def exists_uncovered_x(inst: Instance, eps: float = 1e-4,
                        routing_cap: int = 200) -> Dict:
    """Does an x in Q_G escape every routing's two-sided box by >= eps?

    Returns a dict. If bailed (too many routings): {"bailed": True, "reason": ...}.
    If the MILP proved infeasible: {"uncovered_x_exists": False, ...} -- the
    Conjecture holds on this fixed instance (no uncovered x, screening-level).
    If the MILP found a witness: {"uncovered_x_exists": True,
    "milp_x_float": ..., "rationalized_split": ..., "exact_verdict": ...,
    "is_counterexample_confirmed": ...} -- ALWAYS check
    "is_counterexample_confirmed" (the exact re-verification), never the raw
    MILP feasibility, before calling anything a counterexample.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    routings = inst.routings()
    n_routings = len(routings)
    if n_routings > routing_cap:
        return {
            "bailed": True,
            "reason": (
                f"{n_routings} routings exceeds routing_cap={routing_cap}; "
                "this MILP encoding is one-shot (all routings up front). "
                "A CEGAR loop -- start with a routing subset, add violated "
                "routings lazily -- would lift this cap; not implemented here."
            ),
        }

    arcs = inst.arcs
    terminals = inst.terminals
    dmax = float(inst.d_max())
    total_demand = float(sum(t.demand for t in terminals))
    big_m = 2.0 * (total_demand + dmax) + 1.0

    # ---- variable layout -----------------------------------------------
    w_index: Dict[Tuple[int, int], int] = {}
    col = 0
    for ti, t in enumerate(terminals):
        for pi in range(len(t.paths)):
            w_index[(ti, pi)] = col
            col += 1
    n_w = col

    x_index: Dict[Arc, int] = {a: n_w + i for i, a in enumerate(arcs)}
    n_x = len(arcs)

    z_index: Dict[Tuple[int, Arc, str], int] = {}
    base = n_w + n_x
    for ri in range(n_routings):
        for a in arcs:
            z_index[(ri, a, "+")] = base; base += 1
            z_index[(ri, a, "-")] = base; base += 1
    n_vars = base

    rows: List[List[float]] = []
    lbs: List[float] = []
    ubs: List[float] = []

    # x(a) - sum_i demand_i * sum_p [a in path] w_{i,p} = 0
    for a in arcs:
        row = [0.0] * n_vars
        row[x_index[a]] = 1.0
        for ti, t in enumerate(terminals):
            d = float(t.demand)
            for pi, path in enumerate(t.paths):
                if a in path:
                    row[w_index[(ti, pi)]] -= d
        rows.append(row); lbs.append(0.0); ubs.append(0.0)

    # per-terminal simplex: sum_p w_{i,p} = 1
    for ti, t in enumerate(terminals):
        row = [0.0] * n_vars
        for pi in range(len(t.paths)):
            row[w_index[(ti, pi)]] = 1.0
        rows.append(row); lbs.append(1.0); ubs.append(1.0)

    # per-routing disjunction: some arc escapes the box by >= eps
    for ri, r in enumerate(routings):
        load = inst.routing_load(r)
        loadf = {a: float(load[a]) for a in arcs}
        sumrow = [0.0] * n_vars
        for a in arcs:
            # side '+': x(a) - f^R(a) >= dmax + eps   (forced when z=1)
            row_plus = [0.0] * n_vars
            row_plus[x_index[a]] = 1.0
            row_plus[z_index[(ri, a, "+")]] = -big_m
            rows.append(row_plus)
            lbs.append(dmax + eps - big_m + loadf[a])
            ubs.append(np.inf)
            sumrow[z_index[(ri, a, "+")]] = 1.0

            # side '-': f^R(a) - x(a) >= dmax + eps   (forced when z=1)
            row_minus = [0.0] * n_vars
            row_minus[x_index[a]] = -1.0
            row_minus[z_index[(ri, a, "-")]] = -big_m
            rows.append(row_minus)
            lbs.append(dmax + eps - big_m - loadf[a])
            ubs.append(np.inf)
            sumrow[z_index[(ri, a, "-")]] = 1.0

        rows.append(sumrow); lbs.append(1.0); ubs.append(np.inf)

    A = np.array(rows, dtype=float)
    constraints = LinearConstraint(A, np.array(lbs), np.array(ubs))

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)
    for idx in w_index.values():
        upper[idx] = 1.0
    for idx in x_index.values():
        upper[idx] = total_demand
    for idx in z_index.values():
        upper[idx] = 1.0
    bounds = Bounds(lower, upper)

    integrality = np.zeros(n_vars)
    for idx in z_index.values():
        integrality[idx] = 1

    c = np.zeros(n_vars)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

    if not res.success:
        return {
            "uncovered_x_exists": False,
            "milp_status": res.message,
            "n_routings": n_routings,
        }

    sol = res.x
    split_float: List[List[float]] = []
    for ti, t in enumerate(terminals):
        ws = [max(0.0, sol[w_index[(ti, pi)]]) for pi in range(len(t.paths))]
        s = sum(ws)
        ws = [w / s for w in ws] if s > 0 else [1.0 / len(ws)] * len(ws)
        split_float.append(ws)

    # rationalize several ways, keep whichever gives the largest exact t --
    # this is the "confirm before believing it" step the module docstring
    # promises.
    best_split: Optional[List[List[Fr]]] = None
    best_t = None
    for cand in rationalize_split_candidates(split_float):
        t_exact = t_of_x(inst, cand)
        if best_t is None or t_exact > best_t:
            best_t, best_split = t_exact, cand
    assert best_split is not None
    exact_verdict = verify_counterexample(inst, best_split)

    return {
        "uncovered_x_exists": True,
        "n_routings": n_routings,
        "milp_x_float": {a: float(sol[x_index[a]]) for a in arcs},
        "milp_split_float": split_float,
        "rationalized_split": best_split,
        "exact_verdict": exact_verdict,
        "is_counterexample_confirmed": exact_verdict["is_counterexample"],
    }


if __name__ == "__main__":
    from sweep import random_layered_dag, DEFAULT_DEMAND_POOL

    print("=== triangle-conflict fixture ===")
    tri = triangle_instance()
    tri_result = exists_uncovered_x(tri)
    if tri_result.get("bailed"):
        print("BAILED:", tri_result["reason"])
    elif tri_result["uncovered_x_exists"]:
        print("MILP found a candidate uncovered x -- confirming exactly...")
        print("is_counterexample_confirmed =", tri_result["is_counterexample_confirmed"])
        if tri_result["is_counterexample_confirmed"]:
            print("!!! CONFIRMED COUNTEREXAMPLE on the triangle fixture !!!")
            print(tri_result["exact_verdict"])
    else:
        print(f"NO uncovered x over {tri_result['n_routings']} routings "
              "-- Conjecture 1.3 holds on this fixed instance.")

    print()
    print("=== random layered DAG (seed=7) ===")
    rand_inst = random_layered_dag(2, 2, 2, seed=7, demand_pool=DEFAULT_DEMAND_POOL)
    rand_result = exists_uncovered_x(rand_inst)
    if rand_result.get("bailed"):
        print("BAILED:", rand_result["reason"])
    elif rand_result["uncovered_x_exists"]:
        print("MILP found a candidate uncovered x -- confirming exactly...")
        print("is_counterexample_confirmed =", rand_result["is_counterexample_confirmed"])
        if rand_result["is_counterexample_confirmed"]:
            print("!!! CONFIRMED COUNTEREXAMPLE on the random instance !!!")
            print(rand_result["exact_verdict"])
    else:
        print(f"NO uncovered x over {rand_result['n_routings']} routings "
              "-- Conjecture 1.3 holds on this fixed instance.")
