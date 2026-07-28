"""
Phase 4 (issue #169): the reformulated question, and certified per-instance
decisions for it.

QUESTION Q4 (NG-9/NG-10's residue -- the campaign's whole remaining target).
Let T be a tree with network matrix C w.r.t. chords (u_i, v_i), i = 1..k
(C[a,i] in {-1,0,+1}, nonzero exactly on the tree path between u_i and v_i,
signed by side). Given demands d_i > 0 and w in [0,1]^k, is there always
z in {0,1}^k with

    max_a | sum_i C[a,i] d_i (z_i - w_i) |  <=  d_max := max_i d_i ?

YES for every (T, chords, d) would close the entire 2-path grammar of
Conjecture 1.3 (Lemma 4's class), forcing any counterexample to use >= 3
paths per terminal. NO -- an explicit (T, chords, d, w) -- IS a counterexample
to Conjecture 1.3. Known: YES for equal demands (Lemma 6, Hoffman-Kruskal),
YES whenever the chord graph is series-parallel (NG-9, MSW25), and Doerr's
TU linear-discrepancy bound covers only the unit-box (equal-demand) case.

FLOW READING (useful for intuition): orient chord i from v_i to u_i and send
xi_i = d_i(z_i - w_i) along its unique tree path. Then arc a's net load is
exactly sum_i C[a,i] xi_i, and each xi_i is chosen from the two-point set
{-d_i w_i, +d_i(1-w_i)}, which straddles 0 with spread d_i. At z = w every
load is exactly 0; the question is how far rounding must push them.

LEMMA 7 (cycle elimination -- proved and implemented below). If the chords
with fractional zeta_i contain a cycle in the multigraph H on tree nodes
(one edge {u_i, v_i} per chord), then shifting xi around that cycle changes
NO tree-arc load: a cycle of chords means their tree paths concatenate into a
closed walk, so the signed combination sum_j sigma_j C[.,i_j] vanishes
identically (verified numerically as well as argued). Shift until some
zeta_i reaches 0 or 1. Hence:

    WLOG the fractional chords form a FOREST on the tree's nodes,
    with every arc load unchanged -- in particular still all-zero
    when starting from zeta = w.

COROLLARY 7.1: at most n-1 chords are fractional at the critical point
(a forest on n tree nodes has < n edges), independently of k. With NG-9
(a counterexample needs a K4 minor, hence n >= 4 tree nodes), the first
possible critical configurations have 3 fractional chords on 4 nodes.
"""

from __future__ import annotations

from fractions import Fraction as Fr
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from outtree import OutTreeInstance


# ---------------------------------------------------------------------------
# Lemma 7: cycle elimination
# ---------------------------------------------------------------------------

def find_chord_cycle(attach: Sequence[Tuple[str, str]],
                     active: Sequence[int]) -> Optional[List[Tuple[int, int]]]:
    """A cycle among the `active` chords, as [(chord_index, orientation)].

    Orientation +1 means the chord is traversed u_i -> v_i in the cycle walk,
    -1 the other way. Union-find over the active chords; the edge that closes
    a component gives the cycle, recovered by walking the spanning forest.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj: Dict[str, List[Tuple[str, int]]] = {}
    for i in active:
        u, v = attach[i]
        ru, rv = find(u), find(v)
        if ru == rv:
            # closes a cycle: path from u to v in the forest built so far
            # walk u -> v through the forest, then close via chord i (v -> u)
            path = _forest_path(adj, attach, u, v)
            if path is None:
                continue
            return path + [(i, _orientation(attach[i], v))]
        parent[ru] = rv
        adj.setdefault(u, []).append((v, i))
        adj.setdefault(v, []).append((u, i))
    return None


def _orientation(attach_i: Tuple[str, str], frm: str) -> int:
    """+1 when the walk traverses chord i from v_i to u_i (the direction in
    which xi_i is positive, since C[a,i] = [u_i in S_a] - [v_i in S_a]);
    -1 for the reverse."""
    u, v = attach_i
    return +1 if frm == v else -1


def _forest_path(adj: Dict[str, List[Tuple[str, int]]],
                 attach: Sequence[Tuple[str, str]], src: str, dst: str
                 ) -> Optional[List[Tuple[int, int]]]:
    """Path src->dst in the chord forest, as [(chord_index, orientation)]."""
    stack = [(src, [])]
    seen = {src}
    while stack:
        node, acc = stack.pop()
        if node == dst:
            return acc
        for (nxt, ci) in adj.get(node, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            stack.append((nxt, acc + [(ci, _orientation(attach[ci], node))]))
    return None


def loads(C: Sequence[Sequence[int]], demands: Sequence[Fr],
          zeta: Sequence[Fr], w: Sequence[Fr]) -> List[Fr]:
    return [sum(c * d * (z - wi) for c, d, z, wi in zip(row, demands, zeta, w))
            for row in C]


def cycle_shift_preserves_loads(inst: OutTreeInstance,
                                cycle: Sequence[Tuple[int, int]]) -> bool:
    """Lemma 7's content, checked exactly: the signed chord combination of a
    cycle is identically zero on every tree arc."""
    C = inst.coefficient_matrix()
    for row in C:
        if sum(sigma * row[i] for (i, sigma) in cycle) != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Certified per-instance decision (phase 3 used grid+polish; this is exact)
# ---------------------------------------------------------------------------

def exists_bad_w(inst: OutTreeInstance, eps_floor: float = 1e-6,
                 bound_factor: float = 1.0) -> Dict:
    """Does ANY w in [0,1]^k make every rounding violate? (MILP over w.)

    For each z in {0,1}^k the instance must violate on SOME arc and side:
        sum_i C[a,i] d_i (z_i - w_i)  >=  d_max + eps      (side +)
     or -sum_i C[a,i] d_i (z_i - w_i) >=  d_max + eps      (side -)
    encoded with one binary per (z, arc, side) and a big-M relaxation, plus
    "at least one selected" per z. Maximizes eps; eps > 0 means a
    counterexample x exists for this (tree, chords, demands).

    SCREENING ONLY -- floats. Any hit must be rationalized and re-verified
    exactly (core.verify_counterexample on to_core_instance) before the word
    counterexample is used. Returns the raw MILP witness for that purpose.
    """
    C = inst.coefficient_matrix()
    d = [float(x) for x in inst.demands]
    dmax = max(d) * bound_factor   # bound_factor < 1 = calibration knob only;
                                   # 1.0 is the real Conjecture 1.3 threshold
    k = len(d)
    zs = list(product((0, 1), repeat=k))
    n_arc = len(C)
    if not n_arc:
        return {"feasible": False, "reason": "no tree arcs"}

    # variables: w_0..w_{k-1}, eps, then binaries y[z][a][side]
    n_bin = len(zs) * n_arc * 2
    n_var = k + 1 + n_bin
    big_m = 4 * (sum(d) + dmax) + 10

    def bin_index(zi: int, a: int, side: int) -> int:
        return k + 1 + (zi * n_arc + a) * 2 + side

    A_rows, lb, ub = [], [], []
    for zi, z in enumerate(zs):
        for a, row in enumerate(C):
            # sum_i C[a,i] d_i z_i - sum_i C[a,i] d_i w_i  >= dmax + eps - M(1-y)
            const = sum(c * di * zzi for c, di, zzi in zip(row, d, z))
            for side in (0, 1):
                sgn = 1.0 if side == 0 else -1.0
                # want (when selected):  sgn*sum_i C[a,i] d_i (z_i - w_i) >= dmax + eps
                # rearranged:            sgn*sum_i C[a,i] d_i w_i + eps <= sgn*const - dmax
                # big-M, enforced iff y = 1:   LHS + M*y <= RHS + M
                r = np.zeros(n_var)
                for i in range(k):
                    r[i] = sgn * (row[i] * d[i])
                r[k] = 1.0                             # +eps
                r[bin_index(zi, a, side)] = big_m
                A_rows.append(r)
                lb.append(-np.inf)
                ub.append(sgn * const - dmax + big_m)
        # at least one (arc, side) selected for this z
        r = np.zeros(n_var)
        for a in range(n_arc):
            for side in (0, 1):
                r[bin_index(zi, a, side)] = 1.0
        A_rows.append(r)
        lb.append(1.0)
        ub.append(np.inf)

    integrality = np.zeros(n_var)
    integrality[k + 1:] = 1
    bounds = Bounds(lb=[0.0] * k + [0.0] + [0.0] * n_bin,
                    ub=[1.0] * k + [dmax] + [1.0] * n_bin)
    c_obj = np.zeros(n_var)
    c_obj[k] = -1.0  # maximize eps

    res = milp(c=c_obj, constraints=LinearConstraint(np.array(A_rows), lb, ub),
               integrality=integrality, bounds=bounds)
    if not res.success or res.x is None:
        # For this encoding infeasibility is MEANINGFUL: no w makes every
        # rounding violate, i.e. the instance satisfies the bound. Distinguish
        # it from a solver failure so a certified "no" is never confused with
        # an error.
        proved = "infeasible" in (res.message or "").lower()
        return {"feasible": False,
                "proved_no_bad_w": proved,
                "reason": res.message}
    eps = float(res.x[k])
    return {
        "feasible": eps > eps_floor,
        "eps": eps,
        "w_float": [float(v) for v in res.x[:k]],
    }


def rationalize_and_verify(inst: OutTreeInstance, w_float: Sequence[float],
                           denominators: Sequence[int] = (1, 2, 3, 4, 6, 8, 12,
                                                          16, 24, 48, 120)
                           ) -> Optional[Dict]:
    """Exact re-check of a MILP witness across candidate rationalizations.
    Returns the first exactly-verified counterexample, else None."""
    from core import verify_counterexample, check_path_closure, check_paths_valid
    C = inst.coefficient_matrix()
    ci = inst.to_core_instance()
    for den in denominators:
        w = [Fr(round(v * den), den) for v in w_float]
        w = [min(max(x, Fr(0)), Fr(1)) for x in w]
        if not inst.is_counterexample(w, C):
            continue
        problems = check_paths_valid(ci, 's') + check_path_closure(ci, 's')
        res = verify_counterexample(ci, inst.split_from_w(w))
        if res["is_counterexample"] and not problems:
            return {"w": [str(x) for x in w], "t": str(res["t_of_x"]),
                    "denominator": den, "table": res["table"]}
    return None


# ---------------------------------------------------------------------------
# Quantifying over demands too (session 4, second pass)
# ---------------------------------------------------------------------------

def exists_bad_instance(inst: OutTreeInstance, eps_floor: float = 1e-6,
                        bound_factor: float = 1.0) -> Dict:
    """Decide a COMBINATORIAL STRUCTURE, not just one demand vector.

    `exists_bad_w` fixes demands and searches w. But demands are continuous,
    so sweeping a finite sample of them leaves the demand axis open. Change
    variables: p_i = d_i w_i, q_i = d_i (1 - w_i), so p_i, q_i >= 0 and
    d_i = p_i + q_i. The rounding contribution becomes

        delta_i(z_i) = +q_i  if z_i = 1,   -p_i  if z_i = 0,

    which is LINEAR in (p, q) -- the bilinear d_i * w_i is gone. The only
    nonlinearity left is d_max = max_i (p_i + q_i), and that is removed by
    scaling: a counterexample scaled by 1/d_max has d_max = 1, so impose
    p_i + q_i <= 1 and demand a violation > 1. Sound and complete for the
    existence question (any (p,q) with violation > 1 >= d_max is itself a
    counterexample, normalized or not).

    So this MILP decides: does ANY choice of demands AND w make every
    rounding violate, for this tree + chord structure? Infeasible = the
    structure is safe for EVERY demand vector -- an infinitely stronger
    statement than a finite demand sweep.
    """
    C = inst.coefficient_matrix()
    k = len(inst.attach)
    zs = list(product((0, 1), repeat=k))
    n_arc = len(C)
    if not n_arc:
        return {"feasible": False, "reason": "no tree arcs"}
    thresh = 1.0 * bound_factor

    # variables: p_0..p_{k-1}, q_0..q_{k-1}, eps, binaries y[z][a][side]
    n_bin = len(zs) * n_arc * 2
    n_var = 2 * k + 1 + n_bin
    big_m = 4 * k + 10

    def bidx(zi: int, a: int, side: int) -> int:
        return 2 * k + 1 + (zi * n_arc + a) * 2 + side

    A_rows, lb, ub = [], [], []
    # d_i = p_i + q_i <= 1  (normalization d_max = 1)
    for i in range(k):
        r = np.zeros(n_var)
        r[i] = 1.0
        r[k + i] = 1.0
        A_rows.append(r)
        lb.append(-np.inf)
        ub.append(1.0)
    for zi, z in enumerate(zs):
        for a, row in enumerate(C):
            for side in (0, 1):
                sgn = 1.0 if side == 0 else -1.0
                # want: sgn * sum_i C[a,i] * delta_i(z_i) >= thresh + eps
                # => -sgn*sum(...) + eps <= -thresh   (enforced iff y = 1)
                r = np.zeros(n_var)
                for i in range(k):
                    if not row[i]:
                        continue
                    if z[i]:
                        r[k + i] = -sgn * row[i]      # coefficient on q_i
                    else:
                        r[i] = +sgn * row[i]          # delta = -p_i
                r[2 * k] = 1.0                        # +eps
                r[bidx(zi, a, side)] = big_m
                A_rows.append(r)
                lb.append(-np.inf)
                ub.append(-thresh + big_m)
        r = np.zeros(n_var)
        for a in range(n_arc):
            for side in (0, 1):
                r[bidx(zi, a, side)] = 1.0
        A_rows.append(r)
        lb.append(1.0)
        ub.append(np.inf)

    integrality = np.zeros(n_var)
    integrality[2 * k + 1:] = 1
    bounds = Bounds(lb=[0.0] * (2 * k) + [0.0] + [0.0] * n_bin,
                    ub=[1.0] * (2 * k) + [1.0] + [1.0] * n_bin)
    c_obj = np.zeros(n_var)
    c_obj[2 * k] = -1.0

    res = milp(c=c_obj, constraints=LinearConstraint(np.array(A_rows), lb, ub),
               integrality=integrality, bounds=bounds)
    if not res.success or res.x is None:
        proved = "infeasible" in (res.message or "").lower()
        return {"feasible": False, "proved_no_bad_instance": proved,
                "reason": res.message}
    eps = float(res.x[2 * k])
    p = [float(v) for v in res.x[:k]]
    q = [float(v) for v in res.x[k:2 * k]]
    return {"feasible": eps > eps_floor, "eps": eps, "p": p, "q": q,
            "demands": [pi + qi for pi, qi in zip(p, q)],
            "w": [(pi / (pi + qi)) if (pi + qi) > 1e-12 else 0.0
                  for pi, qi in zip(p, q)]}
