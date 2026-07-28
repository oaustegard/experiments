"""
beta* engine for single-source unsplittable flow (SSUF) cost-preserving violation.

Given a DAG G with source s, terminals t_1..t_k with demands d_1..d_k, a fractional
flow x on the arcs of G (x = sum of per-terminal path flows), and d_max = max(d_i):

    U_beta(x) = { load vectors f^P of unsplittable routings P :
                  f^P(a) <= x(a) + beta * d_max  for every arc a }

    beta*(G, d, x) = sup { beta : x is NOT in conv(U_beta(x)) }

By the DAG potential-shift argument (issue #165), x not-in conv(U_beta(x)) is exactly
"the membership LP is infeasible", and beta* is the largest breakpoint at which that
LP is still infeasible. Costs never need to be searched directly for beta* itself
(they fall out of the LP dual / are only needed if you want the explicit certificate).

Exact rational arithmetic throughout (fractions.Fraction) -- no floats anywhere in
the certificate path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as Fr
from itertools import product
from typing import Dict, List, Tuple, Optional

Arc = Tuple[str, str]


@dataclass
class Terminal:
    name: str
    demand: Fr
    # each path is a tuple of arcs (Arc, ...) from s to this terminal
    paths: List[Tuple[Arc, ...]]
    # fractional split across paths, same length/order as `paths`, must sum to 1
    split: List[Fr]


@dataclass
class Instance:
    arcs: List[Arc]
    terminals: List[Terminal]

    def d_max(self) -> Fr:
        return max(t.demand for t in self.terminals)

    def x(self) -> Dict[Arc, Fr]:
        """Fractional arc loads induced by each terminal's path split."""
        load: Dict[Arc, Fr] = {a: Fr(0) for a in self.arcs}
        for t in self.terminals:
            for path, frac in zip(t.paths, t.split):
                for a in path:
                    load[a] += t.demand * frac
        return load

    def routings(self) -> List[Tuple[int, ...]]:
        """All unsplittable routings: one path-index per terminal."""
        return list(product(*(range(len(t.paths)) for t in self.terminals)))

    def routing_load(self, routing: Tuple[int, ...]) -> Dict[Arc, Fr]:
        load: Dict[Arc, Fr] = {a: Fr(0) for a in self.arcs}
        for t, path_idx in zip(self.terminals, routing):
            for a in t.paths[path_idx]:
                load[a] += t.demand
        return load

    def routing_label(self, routing: Tuple[int, ...], names: Optional[List[str]] = None) -> str:
        names = names or [t.name for t in self.terminals]
        return "".join(
            (str(path_idx) if len(self.terminals[i].paths) > 2 else "ZE"[path_idx])
            for i, path_idx in enumerate(routing)
        )


from sympy import symbols as _sp_symbols
from sympy.solvers.simplex import lpmin as _sp_lpmin, InfeasibleLPError as _SpInfeasible


def exact_lp_feasible(A_rows: List[List[Fr]], b: List[Fr], n_vars: int) -> Tuple[bool, Optional[List[Fr]]]:
    """
    Exact rational feasibility test for: does there exist w >= 0, sum(w) = 1,
    with A_rows . w <= b (componentwise)?  A_rows has one row per constraint,
    n_vars = number of routings (columns). The simplex constraint sum(w)=1, w>=0
    is added on top of the caller's rows.

    Delegates to sympy.solvers.simplex.lpmin (exact Fraction/Rational arithmetic,
    part of sympy's tested public API) rather than a hand-rolled simplex -- an
    early hand-rolled two-phase implementation here had a live bug (failed its
    own 1-D sanity test), and a silently-wrong LP solver would corrupt every
    downstream beta* certificate without any visible symptom. Small instances
    only (<=10^4 routings per issue #165 scope) -- not built for scale.
    """
    ws = _sp_symbols(f"w0:{n_vars}")
    constraints = []
    for row, rhs in zip(A_rows, b):
        expr = sum(Fr(coef) * w for coef, w in zip(row, ws)) - Fr(rhs)
        constraints.append(expr <= 0)
    constraints.append(sum(ws) - 1 >= 0)
    constraints.append(sum(ws) - 1 <= 0)
    for w in ws:
        constraints.append(w >= 0)

    try:
        _, solution = _sp_lpmin(0, constraints)
    except _SpInfeasible:
        return False, None

    w_values = [Fr(solution[w]) for w in ws]
    return True, w_values


def membership_lp_feasible(inst: Instance, beta: Fr) -> Tuple[bool, Optional[List[Fr]]]:
    """Is x in conv{ F_r : F_r(a) <= x(a) + beta*dmax for all a }?"""
    x = inst.x()
    dmax = inst.d_max()
    routings = inst.routings()
    loads = [inst.routing_load(r) for r in routings]

    good_idx = [
        i for i, load in enumerate(loads)
        if all(load[a] <= x[a] + beta * dmax for a in inst.arcs)
    ]
    if not good_idx:
        return False, None

    good_loads = [loads[i] for i in good_idx]
    n = len(good_loads)

    # Need: exists w>=0, sum w=1, sum_i w_i * good_loads[i](a) == x(a) for all a.
    # Equality -> encode as two <= (>=  and <=) per arc.
    A_rows: List[List[Fr]] = []
    b: List[Fr] = []
    for a in inst.arcs:
        A_rows.append([good_loads[i][a] for i in range(n)])
        b.append(x[a])
        A_rows.append([-good_loads[i][a] for i in range(n)])
        b.append(-x[a])

    feasible, w = exact_lp_feasible(A_rows, b, n)
    if not feasible:
        return False, None
    full_w = [Fr(0)] * len(routings)
    for i, gi in enumerate(good_idx):
        full_w[gi] = w[i]
    return True, full_w


def breakpoints(inst: Instance) -> List[Fr]:
    """All beta values where the good-set changes: (F_r(a) - x(a)) / dmax."""
    x = inst.x()
    dmax = inst.d_max()
    routings = inst.routings()
    bps = set()
    for r in routings:
        load = inst.routing_load(r)
        for a in inst.arcs:
            bps.add((load[a] - x[a]) / dmax)
    return sorted(b for b in bps if b >= 0)


def compute_beta_star(inst: Instance, verbose: bool = False) -> Dict:
    """
    beta* = largest breakpoint at which the membership LP is still infeasible.
    Returns a dict with beta_star, the routing-by-routing minimum overshoot table
    (for cross-checking against a "per-routing minimum overshoot" style certificate),
    and the good/bad split at beta*.
    """
    x = inst.x()
    dmax = inst.d_max()
    routings = inst.routings()
    loads = [inst.routing_load(r) for r in routings]

    # per-routing overshoot = max_a (F_r(a) - x(a)), in units of dmax the routing
    # is "beta-good" once beta >= overshoot/dmax.
    overshoots = []
    for load in loads:
        worst = max(load[a] - x[a] for a in inst.arcs)
        overshoots.append(worst)

    bps = breakpoints(inst)
    beta_star = None
    last_feasible_w = None
    for beta in bps:
        feasible, w = membership_lp_feasible(inst, beta)
        if verbose:
            print(f"  beta={float(beta):.4f} ({beta}) feasible={feasible}")
        if feasible:
            beta_star = beta
            last_feasible_w = w
            break
    # beta* per the definition is the SUP of betas where x is NOT in the hull,
    # i.e. the largest breakpoint strictly below the first feasible one.
    # bps is sorted ascending; find the breakpoint immediately before the first
    # feasible beta.
    if beta_star is None:
        # never feasible even at largest breakpoint -- beta* is unbounded within
        # tested range; report the max breakpoint tested as a lower bound.
        return {
            "beta_star": None,
            "beta_star_lower_bound": bps[-1] if bps else None,
            "overshoots": overshoots,
            "routings": routings,
            "note": "membership LP infeasible at all tested breakpoints",
        }
    idx = bps.index(beta_star)
    beta_star_value = bps[idx - 1] if idx > 0 else Fr(0)

    return {
        "beta_star": beta_star_value,
        "first_feasible_beta": beta_star,
        "witness_weights": last_feasible_w,
        "overshoots": overshoots,
        "routings": routings,
        "breakpoints": bps,
    }
