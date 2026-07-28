"""
beta* engine for kill-path B (issue #169), ported from experiments/ssuf-beta
(PR #167) with one substantive correction, flagged loudly because it changes
reported values:

    beta*(G,d,x) = sup { beta >= 0 : x not in conv U_beta(x) },
    U_beta(x)    = { f^P : f^P(a) <= x(a) + beta * d_max  for all a }.

The good-set U_beta is constant on half-open intervals between consecutive
breakpoints beta = (f^P(a) - x(a)) / d_max, and membership is monotone in
beta. If the membership LP is infeasible at breakpoint b_i and first becomes
feasible at breakpoint b_{i+1}, then x is outside the hull on the whole
interval [b_i, b_{i+1}) and inside from b_{i+1} on -- so the sup is
b_{i+1}, the FIRST FEASIBLE breakpoint. The PR #167 engine returned b_i
(the last infeasible breakpoint) and its self-hosted calibration hand-derived
the same off-by-one convention, so the two agreed while both sat one
breakpoint low. On the Rybin anchor the corrected convention gives
beta* = 16/15 (= overshoot 16 / d_max 15, an exact breakpoint); the old one
would report whatever breakpoint sits below it.

Refutation semantics are unaffected in direction: beta* > 2 still means the
membership LP is infeasible at beta = 2 (x outside the hull on [.., beta*)),
which is what the STVZ contrapositive needs -- but the reported number now
matches the sup definition and the 16/15 anchor.

Exact rationals everywhere; sympy's simplex is the LP core (same choice and
rationale as PR #167 -- a hand-rolled simplex failed its own sanity test
there, and a silently wrong LP corrupts every certificate downstream).
"""

from __future__ import annotations

from fractions import Fraction as Fr
from typing import Dict, List, Optional, Sequence, Tuple

from core import Arc, Instance

from sympy import symbols as _sp_symbols
from sympy.solvers.simplex import lpmin as _sp_lpmin, InfeasibleLPError as _SpInfeasible


def exact_lp_feasible(A_rows: List[List[Fr]], b: List[Fr], n_vars: int
                      ) -> Tuple[bool, Optional[List[Fr]]]:
    """Exists w >= 0, sum(w) = 1, A_rows . w <= b componentwise?"""
    ws = _sp_symbols(f"w0:{n_vars}")
    constraints = []
    for row, rhs in zip(A_rows, b):
        expr = sum(Fr(c) * w for c, w in zip(row, ws)) - Fr(rhs)
        constraints.append(expr <= 0)
    constraints.append(sum(ws) - 1 >= 0)
    constraints.append(sum(ws) - 1 <= 0)
    for w in ws:
        constraints.append(w >= 0)
    try:
        _, solution = _sp_lpmin(0, constraints)
    except _SpInfeasible:
        return False, None
    return True, [Fr(solution[w]) for w in ws]


def membership_lp_feasible(inst: Instance, x: Dict[Arc, Fr], beta: Fr
                           ) -> Tuple[bool, Optional[List[Fr]]]:
    """Is x in conv{ f^P : f^P(a) <= x(a) + beta*d_max for all a }?"""
    dmax = inst.d_max()
    routings = inst.routings()
    loads = [inst.routing_load(r) for r in routings]
    good_idx = [i for i, load in enumerate(loads)
                if all(load[a] <= x[a] + beta * dmax for a in inst.arcs)]
    if not good_idx:
        return False, None
    good = [loads[i] for i in good_idx]
    n = len(good)
    A_rows: List[List[Fr]] = []
    b: List[Fr] = []
    for a in inst.arcs:
        A_rows.append([good[i][a] for i in range(n)])
        b.append(x[a])
        A_rows.append([-good[i][a] for i in range(n)])
        b.append(-x[a])
    feasible, w = exact_lp_feasible(A_rows, b, n)
    if not feasible:
        return False, None
    full_w = [Fr(0)] * len(routings)
    for i, gi in enumerate(good_idx):
        full_w[gi] = w[i]
    return True, full_w


def breakpoints(inst: Instance, x: Dict[Arc, Fr]) -> List[Fr]:
    """All beta where the good-set changes: (f^P(a) - x(a)) / d_max, >= 0."""
    dmax = inst.d_max()
    bps = set()
    for r in inst.routings():
        load = inst.routing_load(r)
        for a in inst.arcs:
            bps.add((load[a] - x[a]) / dmax)
    return sorted(b for b in bps if b >= 0)


def compute_beta_star(inst: Instance, split: Sequence[Sequence[Fr]],
                      verbose: bool = False) -> Dict:
    """beta* = first breakpoint at which the membership LP is feasible
    (the sup of the infeasible region -- see module docstring).

    x always lies in conv of ALL routing loads (it is the expectation of the
    independent per-terminal random path choice), so the LP is feasible at
    the largest breakpoint and beta* is always attained by this scan.
    """
    x = inst.x_from_split(split)
    dmax = inst.d_max()
    routings = inst.routings()
    loads = [inst.routing_load(r) for r in routings]

    # per-routing minimum overshoot: the smallest beta*d_max slack that makes
    # this routing ceiling-good. Cross-checks "min overshoots 26,16,16,16"
    # style certificates.
    overshoots = [max(load[a] - x[a] for a in inst.arcs) for load in loads]

    bps = breakpoints(inst, x)
    # Test beta = 0 explicitly: if x is already in the hull with zero slack,
    # the infeasible region is empty and beta* = 0 by convention -- without
    # this the scan would wrongly report the first positive breakpoint.
    if not bps or bps[0] != 0:
        bps = [Fr(0)] + bps
    last_infeasible: Optional[Fr] = None
    for beta in bps:
        feasible, w = membership_lp_feasible(inst, x, beta)
        if verbose:
            print(f"  beta={beta} ({float(beta):.4f}) feasible={feasible}", flush=True)
        if feasible:
            return {
                "beta_star": beta,
                "last_infeasible_breakpoint": last_infeasible,
                "witness_weights": w,
                "overshoots": overshoots,
                "routings": routings,
                "breakpoints": bps,
            }
        last_infeasible = beta
    raise AssertionError(
        "membership LP infeasible at every breakpoint -- impossible, since x "
        "is the split-expectation of routing loads; instance or split is malformed")
