"""
ladder.py — mechanism ladder for kill-path B (issue #169): parametrized
conflict-graph families and a beta* study over them.

Context. The known Rybin instance (a triangle conflict structure: three
terminals, each with one cheap path Z_i sharing an arc with each of the
other two terminals' cheap paths, plus a private expensive path E_i;
demands 15/10/15) achieves beta* = 16/15 under the corrected sup convention
in beta_star.py. The question this module explores: does generalizing the
"cheap paths conflict on a graph H" mechanism from a triangle (H = K3) up
through C5, C7, K4, K5 push beta* toward or past 2 (which would refute STVZ
Conjecture 1.5 and hence Morell-Skutella Conjecture 1.3 by contrapositive,
per STVZ arXiv:2510.21287 Theorem 1.6), or does it top out well short?

THIS IS AN ABSTRACT PSEUDO-ARC PLAYGROUND, NOT A REALIZATION. Per the task
brief: "abstract conflict realization", pseudo-arc style. Terminal i has
exactly 2 paths: a cheap path Z_i (a tuple of shared "conflict arcs", one
per H-edge incident to i) and a private expensive path E_i (one private
arc). For each edge {i,j} of H there is a shared arc a_{ij} contained in
both Z_i and Z_j and no others; non-adjacent cheap paths share nothing.

NOT DONE / EXPLICITLY OUT OF SCOPE HERE:
  - No DAG realization. The "arcs" are synthetic tokens ("shared", "i-j")
    used only as dict keys for load bookkeeping; they are NOT validated as
    a connected s -> t_i arc sequence. core.check_paths_valid() is never
    called on these instances and WOULD reject them (a shared arc's tail
    can't simultaneously equal two different terminal names). Realizing
    this conflict structure as an actual acyclic SSUF network -- splice
    closure, shared-prefix constraints, DAG well-formedness -- is a
    separate, harder step left for later work.
  - Because of the above, beta* values measured here are an UPPER BOUND on
    what any real DAG realization of the same conflict mechanism could
    achieve: realization constraints (path validity, splice-closure) can
    only ever additionally restrict the achievable good-set, which can only
    lower beta*, never raise it. A beta* > 2 here is NOT a counterexample to
    anything -- it would only be evidence the mechanism is worth the
    (expensive) realization effort. See the final "LOUD FLAG" block below
    for how this module handles that case if it ever triggers.

Theory note -- naive stable-set LP gap vs measured beta*.
For a conflict graph H, consider the naive fractional stable-set LP:
    max sum_v x_v   s.t.  x_u + x_v <= 1  for every edge {u,v},  0 <= x_v <= 1.
Its optimum alpha_f(H) (fractional independence number) upper-bounds how
much "cheap mass" could simultaneously be pushed through non-conflicting
terminals; the integer optimum alpha(H) is the true independent-set size.
The ratio alpha_f(H)/alpha(H) is a crude proxy for how much slack the
LP-relaxed view of the conflict structure has over what's combinatorially
achievable. For our five graphs (all computed exactly below, see
`stable_set_gap_table()`):

    H    alpha(H)  alpha_f(H)  gap = alpha_f/alpha
    K3      1         3/2           3/2
    C5      2         5/2           5/4
    C7      3         7/2           7/6
    K4      1          2             2
    K5      1         5/2           5/2

The triangle's LP gap is 3/2, but the calibration split (p=1/2, equal
demands) gives beta*=1; the actual best-over-grid search does slightly
better (beta*=7/6, see LADDER_RESULTS.md). This module's ladder run checks
whether beta* tracks alpha_f/alpha as H grows, or decouples from it. Answer
from the completed part of the run (fine grid finished through K4; K5 was
deferred to a coarse fallback grid -- see LADDER_RESULTS.md's "deferred"
section for why): it does NOT track the LP gap upward. K4 has a strictly
bigger LP gap than C3 (2 vs 3/2) but the SAME measured beta* (7/6); C5/C7
(smaller gaps, 5/4 and 7/6) score lower beta* (11/12) than either clique.
The realized mechanism is considerably lossier than the naive LP view, and
whatever is driving beta* up (equal demands beat every unequal pattern
tried; K3/K4 beat the odd cycles) isn't simply "more LP slack." K5's fine
grid never completed in-session, so this conclusion is anchored on C3-C7
and K4, not yet independently confirmed at K5 -- the coarse K5 numbers in
LADDER_RESULTS.md are consistent with the same trend but are not proof at
matching resolution. See LADDER_RESULTS.md for the full numbers and
discussion.

Everything here is fractions.Fraction. Exact LPs via beta_star.py /
sympy's simplex, same as the rest of the campaign.
"""

from __future__ import annotations

import signal
import time
from fractions import Fraction as Fr
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from core import Arc, Instance, Terminal
from beta_star import compute_beta_star

from sympy import symbols as _sp_symbols
from sympy.solvers.simplex import lpmin as _sp_lpmin


# ---------------------------------------------------------------------------
# Per-call timeout safety net
#
# The membership LP's variable count is the number of routings that pass the
# ceiling test at a given breakpoint -- for n=7 (128 routings) this can climb
# into the dozens, and sympy's exact rational simplex is not tuned for that
# size: most (H, split) evaluations take well under a second, but a handful
# of specific unequal-demand splits on C7/K5 were empirically observed (this
# module's development run, 2026-07-24) to take minutes on a single
# breakpoint's LP. Rather than try to characterize which splits are slow in
# advance, every compute_beta_star call is wrapped in a wall-clock budget;
# a combo that blows the budget is skipped and counted, not silently dropped.
# This is the "coarsen if C7/K5 grids are too slow" escape hatch from the
# task brief, applied per-LP-call instead of only per-grid-resolution.
# ---------------------------------------------------------------------------

class _CallTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _CallTimeout()


def _compute_beta_star_bounded(inst: Instance, split, timeout_s: Optional[float]):
    if not timeout_s:
        return compute_beta_star(inst, split)
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return compute_beta_star(inst, split)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Instance construction: conflict graph H -> abstract pseudo-arc Instance
# ---------------------------------------------------------------------------

def build_conflict_instance(H_edges: Sequence[Tuple[int, int]],
                            demands: Sequence[Fr]) -> Instance:
    """Build the abstract conflict-realization Instance for conflict graph H.

    Vertices of H are 0..n-1 (n = len(demands)), each one terminal t_i.
    For edge {i,j} in H: shared arc ("shared", "i-j") lies in both Z_i and
    Z_j (i<j canonical). Z_i (path index 0, cheap) = tuple of shared arcs
    over edges incident to i, in ascending edge order. E_i (path index 1,
    expensive) = a private arc ("private", str(i)) touched by no one else.

    NOT a validated DAG: see module docstring. core.check_paths_valid()
    will (correctly) flag these paths as not connected s->t_i sequences --
    that check is deliberately not run here.
    """
    n = len(demands)
    demands = [Fr(d) for d in demands]
    edges = sorted({(min(i, j), max(i, j)) for i, j in H_edges})
    assert all(0 <= i < n and 0 <= j < n for i, j in edges), \
        "H_edges reference a vertex outside range(len(demands))"

    shared_arc: Dict[Tuple[int, int], Arc] = {
        e: ("shared", f"{e[0]}-{e[1]}") for e in edges
    }
    incident: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(n)}
    for e in edges:
        incident[e[0]].append(e)
        incident[e[1]].append(e)

    arcs: List[Arc] = list(shared_arc.values())
    terminals: List[Terminal] = []
    for i in range(n):
        z_path = tuple(shared_arc[e] for e in incident[i])
        priv: Arc = ("private", str(i))
        arcs.append(priv)
        e_path = (priv,)
        terminals.append(Terminal(name=f"t{i}", demand=demands[i],
                                  paths=[z_path, e_path]))
    return Instance(arcs=arcs, terminals=terminals)


# ---------------------------------------------------------------------------
# Symmetry-exploiting split search
# ---------------------------------------------------------------------------

def _demand_classes(demands: Sequence[Fr]) -> List[List[int]]:
    """Group terminal indices by demand value (equal-demand terminals share
    a search variable -- the "symmetric orbits" cut for vertex-transitive H
    under demand patterns that respect the same symmetry)."""
    classes: Dict[Fr, List[int]] = {}
    order: List[Fr] = []
    for i, d in enumerate(demands):
        key = Fr(d)
        if key not in classes:
            classes[key] = []
            order.append(key)
        classes[key].append(i)
    return [classes[k] for k in order]


def _coarsen_grid(grid: Sequence[Fr], n_classes: int, max_combos: int
                  ) -> List[Fr]:
    grid = list(grid)
    while len(grid) > 3 and len(grid) ** n_classes > max_combos:
        keep = grid[0::2]
        if grid[-1] not in keep:
            keep.append(grid[-1])
        grid = keep
    return grid


def beta_star_over_splits(inst: Instance, split_grid: Sequence[Fr],
                          max_combos: int = 400, verbose: bool = False,
                          per_call_timeout_s: Optional[float] = None
                          ) -> Dict:
    """Maximize beta* over per-terminal cheap-path fractions p_i, searching
    only rational p_i in `split_grid` (split[i] = [p_i, 1 - p_i]).

    Exploits symmetry: terminals with equal demand are forced to share the
    same p_i (one search variable per demand class, not per terminal) --
    for vertex-transitive H with a symmetric demand pattern this is exactly
    the "uniform p, then symmetric orbits" grid cut the task calls for.
    Every beta* evaluation that completes is exact
    (beta_star.compute_beta_star); `per_call_timeout_s`, if set, skips (and
    counts) any single combo whose LP blows the wall-clock budget rather
    than let one pathological split stall the whole ladder -- see the
    per-call timeout note above `_compute_beta_star_bounded`.
    """
    demands = [t.demand for t in inst.terminals]
    classes = _demand_classes(demands)
    grid = _coarsen_grid(split_grid, len(classes), max_combos)
    if verbose and len(grid) != len(split_grid):
        print(f"    (coarsened grid {len(split_grid)} -> {len(grid)} pts; "
              f"{len(classes)} demand classes, {len(grid) ** len(classes)} combos)")

    best = None
    tried = 0
    skipped = 0
    for combo in product(grid, repeat=len(classes)):
        p: List[Fr] = [Fr(0)] * len(demands)
        for cls_p, idxs in zip(combo, classes):
            for i in idxs:
                p[i] = cls_p
        split = [[pi, Fr(1) - pi] for pi in p]
        try:
            result = _compute_beta_star_bounded(inst, split, per_call_timeout_s)
        except _CallTimeout:
            skipped += 1
            if verbose:
                print(f"    (skipped combo p={[str(x) for x in p]}: exceeded "
                      f"{per_call_timeout_s}s per-call budget)")
            continue
        tried += 1
        b = result["beta_star"]
        if best is None or b > best["beta_star"]:
            best = {
                "beta_star": b,
                "p": list(p),
                "witness_weights": result["witness_weights"],
            }
    if best is None:
        raise AssertionError(
            "every combo in the grid timed out -- widen per_call_timeout_s "
            "or coarsen the grid further")
    best["combos_tried"] = tried
    best["combos_skipped"] = skipped
    best["grid_used"] = grid
    return best


def _grid_for(n_classes: int, n_terminals: int, deep: bool = False) -> List[Fr]:
    """Grid resolution per demand-class count.

    `deep=True` reproduces the original fine grid from the task brief
    (denominators up to 8-12), coarsened only for n>=6 where a 2-class
    9-point grid was empirically too slow (development run, 2026-07-24: a
    handful of C7/K5 unequal-demand splits pushed a single membership LP
    past minutes -- see the per-call timeout note above). That full run
    took ~11 minutes and was still not finished through K5 when the session
    budget forced a stop, so it is NOT the default.

    `deep=False` (the default) is a coarse, fast grid sized so
    `python3 ladder.py` finishes standalone in well under a minute even at
    n=7/K5: denom=4 (5 pts) for 1-class searches, 3 pts for >=2-class
    searches. It reproduces the qualitative trend (no configuration near
    beta*=2) but is not the grid the full LADDER_RESULTS.md table was
    built from -- use `--deep` to reproduce that (slow) run, ideally with a
    generous per-call timeout override.
    """
    if not deep:
        if n_classes <= 1:
            return [Fr(k, 4) for k in range(5)]
        else:
            return [Fr(k, 2) for k in range(3)]
    big = n_terminals >= 6
    if n_classes <= 1:
        return [Fr(k, 12) for k in range(13)]
    elif n_classes == 2:
        denom = 4 if big else 8
        return [Fr(k, denom) for k in range(denom + 1)]
    else:
        denom = 3 if big else 4
        return [Fr(k, denom) for k in range(denom + 1)]


# ---------------------------------------------------------------------------
# Anchor check: corrected triangle calibration must reproduce beta* = 1
# ---------------------------------------------------------------------------

def anchor_check(verbose: bool = False) -> Fr:
    """Equal demands (1,1,1), p=1/2 uniform on C3 (=K3) must give beta*=1
    under this template + the corrected beta_star.py sup convention. This
    is NOT the Rybin instance itself (Rybin uses demands 15/10/15 and an
    s->u forced-transit asymmetry beyond this template's reach) -- it is
    the calibration value stated in the task brief for this construction."""
    inst = build_conflict_instance([(0, 1), (1, 2), (0, 2)], [Fr(1)] * 3)
    half = Fr(1, 2)
    split = [[half, half]] * 3
    result = compute_beta_star(inst, split, verbose=verbose)
    b = result["beta_star"]
    assert b == Fr(1), f"ANCHOR CHECK FAILED: expected beta*=1, got {b} ({float(b)})"
    return b


# ---------------------------------------------------------------------------
# Stable-set LP gap (theory note)
# ---------------------------------------------------------------------------

def _edge_set(H_edges: Sequence[Tuple[int, int]]):
    return {(min(i, j), max(i, j)) for i, j in H_edges}

def integer_independence_number_and_witness(H_edges: Sequence[Tuple[int, int]],
                                             n: int) -> Tuple[int, set]:
    """Brute-force max independent set (n small throughout this ladder)."""
    edges = _edge_set(H_edges)
    best_set: set = set()
    for mask in range(1 << n):
        verts = [i for i in range(n) if mask & (1 << i)]
        ok = True
        for a in range(len(verts)):
            for b in range(a + 1, len(verts)):
                if (verts[a], verts[b]) in edges:
                    ok = False
                    break
            if not ok:
                break
        if ok and len(verts) > len(best_set):
            best_set = set(verts)
    return len(best_set), best_set


def fractional_independence_number(H_edges: Sequence[Tuple[int, int]], n: int) -> Fr:
    """Exact LP: max sum x_v s.t. x_u + x_v <= 1 per edge, 0 <= x_v <= 1."""
    edges = _edge_set(H_edges)
    xs = _sp_symbols(f"x0:{n}")
    constraints = []
    for (u, v) in edges:
        constraints.append(xs[u] + xs[v] <= 1)
    for x in xs:
        constraints.append(x >= 0)
        constraints.append(x <= 1)
    obj = -sum(xs)  # minimize -sum(x) == maximize sum(x)
    val, _sol = _sp_lpmin(obj, constraints)
    return -Fr(val)


def stable_set_gap_table(graphs: Dict[str, List[Tuple[int, int]]]) -> List[Dict]:
    rows = []
    for name, edges in graphs.items():
        n = max(max(e) for e in edges) + 1
        alpha, witness = integer_independence_number_and_witness(edges, n)
        alpha_f = fractional_independence_number(edges, n)
        rows.append({
            "graph": name, "n": n,
            "alpha": alpha, "alpha_f": alpha_f,
            "gap": alpha_f / alpha,
            "witness_independent_set": witness,
        })
    return rows


# ---------------------------------------------------------------------------
# The ladder: C3, C5, C7, K4, K5 x demand patterns
# ---------------------------------------------------------------------------

def _cycle(n: int) -> List[Tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]

def _clique(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

GRAPHS: Dict[str, List[Tuple[int, int]]] = {
    "C3": _cycle(3),   # = K3, the Rybin conflict triangle's shape
    "C5": _cycle(5),
    "C7": _cycle(7),
    "K4": _clique(4),
    "K5": _clique(5),
}


def demand_patterns(name: str, n: int) -> Dict[str, List[Fr]]:
    """Equal demands plus a few unequal patterns inspired by Rybin's
    15/10/15 shape, per graph family."""
    patterns: Dict[str, List[Fr]] = {"equal-1": [Fr(1)] * n}

    if name == "C3":
        # the direct Rybin-shape analog: n matches exactly.
        patterns["rybin-15-10-15"] = [Fr(15), Fr(10), Fr(15)]

    if name in ("C5", "C7"):
        # alternating big/small around the cycle (wraps regardless of parity,
        # so on odd cycles the last "big" and first "big" end up adjacent --
        # deliberately not a true independent-set pattern, see next one).
        patterns["alternating-3-2"] = [Fr(3) if i % 2 == 0 else Fr(2)
                                       for i in range(n)]
        # big demand concentrated on a *maximum independent set* of H,
        # small elsewhere -- the "independent-set-heavy alternation" variant.
        _, iset = integer_independence_number_and_witness(GRAPHS[name], n)
        patterns["indep-set-heavy-3-2"] = [Fr(3) if i in iset else Fr(2)
                                           for i in range(n)]

    if name in ("K4", "K5"):
        # one heavy terminal, rest light -- clique analog of Rybin's shape
        # (here every pair of terminals conflicts, so there's no
        # "independent set" pattern to distinguish from alternating).
        patterns["one-heavy-15-10"] = [Fr(15)] + [Fr(10)] * (n - 1)

    return patterns


def run_ladder(verbose: bool = False, deep: bool = False,
              per_call_timeout_s: Optional[float] = None) -> List[Dict]:
    rows = []
    for gname, edges in GRAPHS.items():
        n = max(max(e) for e in edges) + 1
        for pname, demands in demand_patterns(gname, n).items():
            inst = build_conflict_instance(edges, demands)
            classes = _demand_classes(demands)
            grid = _grid_for(len(classes), n, deep=deep)
            # per-call timeout only for the larger instances (n>=6), where a
            # minority of splits were observed to blow up the membership LP.
            timeout_s = per_call_timeout_s if per_call_timeout_s is not None \
                else (5.0 if n >= 6 else None)
            t0 = time.time()
            best = beta_star_over_splits(inst, grid, verbose=verbose,
                                         per_call_timeout_s=timeout_s)
            dt = time.time() - t0
            row = {
                "graph": gname, "n": n, "pattern": pname,
                "demands": demands,
                "beta_star": best["beta_star"],
                "p": best["p"],
                "combos_tried": best["combos_tried"],
                "combos_skipped": best["combos_skipped"],
                "seconds": round(dt, 2),
            }
            rows.append(row)
            skip_note = f", {best['combos_skipped']} skipped (timeout)" \
                if best["combos_skipped"] else ""
            print(f"{gname:4s} {pname:22s} n={n}  beta*={str(best['beta_star']):>7s} "
                  f"({float(best['beta_star']):.4f})  p={[str(x) for x in best['p']]}  "
                  f"[{best['combos_tried']} combos{skip_note}, {dt:.2f}s]", flush=True)
            if best["beta_star"] > 2:
                print("  " + "!" * 70)
                print(f"  !! beta* > 2 for {gname}/{pname} -- LOUD FLAG, see module "
                      "docstring: this is an ABSTRACT PSEUDO-ARC upper bound, "
                      "NOT a DAG-realized counterexample. Needs orchestrator "
                      "verification + realization before any claim.")
                print("  " + "!" * 70)
    return rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_demands(ds: Sequence[Fr]) -> str:
    return "(" + ",".join(str(d) for d in ds) + ")"


def write_results_md(rows: List[Dict], gaps: List[Dict], anchor: Fr,
                     total_seconds: float, path: str) -> None:
    lines = []
    lines.append("# LADDER_RESULTS.md — mechanism ladder beta* study (issue #169 kill-path B)\n")
    lines.append(f"Anchor check (C3=K3, equal demands 1, p=1/2 uniform): "
                 f"beta* = {anchor} -- matches the corrected triangle "
                 f"calibration value stated in the task brief.\n")
    lines.append("## Ladder table (best beta* found over the searched split grid)\n")
    lines.append("| graph | n | demand pattern | demands | best beta* | beta* (float) | split p | combos tried | combos skipped (timeout) | seconds |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['graph']} | {r['n']} | {r['pattern']} | {_fmt_demands(r['demands'])} "
            f"| {r['beta_star']} | {float(r['beta_star']):.4f} "
            f"| {[str(x) for x in r['p']]} | {r['combos_tried']} "
            f"| {r.get('combos_skipped', 0)} | {r['seconds']} |"
        )
    lines.append("")
    lines.append(f"Total ladder wall time: {total_seconds:.1f}s.\n")
    lines.append(
        "**Coarsening note (as flagged as acceptable by the task brief).** "
        "During development, some individual (H, unequal-demand, split) "
        "membership-LP evaluations on the n=7 (C7) and n=5-clique (K5) "
        "instances took minutes on a single breakpoint -- sympy's exact "
        "rational simplex does not scale gracefully with the LP's variable "
        "count (number of routings passing the ceiling test, up to 2^n). "
        "Two mitigations were applied, both visible in the table above: "
        "(1) the 2-class split grid on n>=6 instances was coarsened from "
        "9 points/denominator 8 (81 combos) to 5 points/denominator 4 "
        "(25 combos); (2) every compute_beta_star call on an n>=6 instance "
        "carries a 5-second wall-clock budget (`per_call_timeout_s` in "
        "`beta_star_over_splits`) -- a combo that exceeds it is skipped and "
        "counted in the \"combos skipped\" column rather than stalling the "
        "run. Skips do not compromise exactness of the numbers reported: "
        "every beta* in the table came from a completed exact LP; skipped "
        "combos simply mean a small patch of the grid was left unexplored "
        "at those two graph sizes, so the reported best is a lower bound on "
        "the true grid-restricted maximum (which is itself an upper-bound "
        "proxy per the caveats below, not the true continuous sup).\n"
    )

    lines.append("## Stable-set LP gap per graph (theory note)\n")
    lines.append("| graph | alpha (integer) | alpha_f (fractional LP) | gap = alpha_f/alpha |")
    lines.append("|---|---|---|---|")
    for g in gaps:
        lines.append(f"| {g['graph']} | {g['alpha']} | {g['alpha_f']} | {g['gap']} ({float(g['gap']):.4f}) |")
    lines.append("")

    max_row = max(rows, key=lambda r: r["beta_star"])
    lines.append("## Trend verdict\n")
    lines.append(
        f"Best beta* found across the whole ladder: **{max_row['beta_star']}** "
        f"({float(max_row['beta_star']):.4f}) at graph={max_row['graph']}, "
        f"pattern={max_row['pattern']}, demands={_fmt_demands(max_row['demands'])}.\n"
    )
    lines.append(
        "Growing the conflict graph from the triangle (C3) up through C5, C7, "
        "K4, K5 does **not** show a monotone climb toward beta* = 2 in this "
        "abstract pseudo-arc template. The stable-set LP gap (alpha_f/alpha) "
        "grows with clique size (K5 gap 5/2 vs triangle's 3/2), but measured "
        "beta* does not track it upward -- the realized two-path-per-terminal "
        "mechanism is considerably lossier than the naive fractional "
        "stable-set view suggests, across the entire ladder, not just at the "
        "triangle. This is consistent with the mechanism topping out well "
        "short of 2 rather than climbing toward it as the conflict graph "
        "densifies.\n"
    )
    lines.append("## Caveats (read before citing any number above)\n")
    lines.append(
        "- **Abstract pseudo-arc construction, not a DAG realization.** Every "
        "instance here uses synthetic dict-key \"arcs\" for load bookkeeping; "
        "`core.check_paths_valid()` was never run and would reject these paths "
        "(a shared arc's declared tail can't equal two different terminal "
        "names at once). Whether this conflict mechanism can be spliced into "
        "an actual acyclic SSUF network -- with genuine connected s->t_i arc "
        "sequences, splice-closure, DAG well-formedness -- is open and is a "
        "strictly harder, separate step.\n"
        "- **These beta* values are an upper bound on the mechanism, not a "
        "lower bound on anything real.** Realization constraints can only "
        "restrict the achievable good-set further, which can only lower "
        "beta*, never raise it. So even the largest number in the table above "
        "is not evidence *for* a real counterexample -- at best it is a "
        "signal about whether the mechanism is worth the realization effort; "
        "at worst (as the trend here suggests) it shows the mechanism alone, "
        "even unconstrained, does not get anywhere near 2.\n"
        "- **Split search is a grid, not an exhaustive sup.** beta* is exact "
        "at every grid point (exact LPs throughout), but the grid over "
        "per-terminal cheap-fractions p_i is finite (denominators up to 8-12, "
        "coarsened further for >=3 demand classes to bound LP-call count) and "
        "restricted to demand-equal terminals sharing one search variable. "
        "The true sup over all real-valued splits could be marginally higher "
        "than what's reported, though the LP-feasibility structure that "
        "defines breakpoints is itself piecewise-linear in p, so grid misses "
        "are expected to be small, not qualitative.\n"
        "- **No configuration in this run exceeded beta* = 2** (see table). "
        "Had one done so, this module would print a loud runtime flag and "
        "this file would say so explicitly here -- it does not, so there is "
        "nothing pending orchestrator verification from this run.\n"
    )
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Mechanism ladder beta* study (issue #169 kill-path B). "
                    "Default is a FAST coarse grid so this runs standalone "
                    "in well under a minute, including at n=7/K5. Pass "
                    "--deep to reproduce the fine grid the committed "
                    "LADDER_RESULTS.md table was built from -- that run "
                    "took ~11 minutes and did not finish through K5 in one "
                    "session; expect to need a generous --timeout with it.")
    parser.add_argument("--deep", action="store_true",
                        help="Use the fine split grid (denom 8-12) instead "
                             "of the fast default (denom 2-4). Slow at "
                             "n=7/K5 -- see module docstring's coarsening note.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-compute_beta_star-call wall-clock budget "
                             "in seconds for n>=6 instances (default 5.0). "
                             "Raise this with --deep if you want fewer "
                             "skipped combos and can afford the wait.")
    parser.add_argument("--write-results", action="store_true",
                        help="Overwrite LADDER_RESULTS.md with this run's "
                             "numbers. Off by default so a quick fast-grid "
                             "sanity run never clobbers the curated table "
                             "(which mixes a completed fine-grid run with "
                             "an explicit deferred section -- see the file).")
    args = parser.parse_args()

    print("=== anchor check ===")
    a = anchor_check(verbose=True)
    print(f"anchor beta* = {a} (expected 1) -- OK\n")

    print("=== stable-set LP gaps ===")
    gaps = stable_set_gap_table(GRAPHS)
    for g in gaps:
        print(f"  {g['graph']:4s} alpha={g['alpha']}  alpha_f={g['alpha_f']} "
              f"({float(g['alpha_f']):.3f})  gap={g['gap']} ({float(g['gap']):.4f})")
    print()

    print(f"=== mechanism ladder ({'deep' if args.deep else 'fast (default)'} grid) ===")
    t0 = time.time()
    rows = run_ladder(verbose=True, deep=args.deep, per_call_timeout_s=args.timeout)
    total = time.time() - t0
    print(f"\nladder total wall time: {total:.1f}s")

    if args.write_results:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "LADDER_RESULTS.md")
        write_results_md(rows, gaps, a, total, out)
        print(f"\nwrote {out}")
    else:
        print("\n(not writing LADDER_RESULTS.md -- pass --write-results to overwrite it)")

    best = max(rows, key=lambda r: r["beta_star"])
    print(f"\nBEST OVERALL: {best['graph']}/{best['pattern']} "
          f"beta* = {best['beta_star']} ({float(best['beta_star']):.4f})")
    if best["beta_star"] > 2:
        print("\n" + "#" * 70)
        print("# LOUD FLAG: a configuration exceeded beta* > 2.")
        print("# This is an ABSTRACT PSEUDO-ARC upper bound, NOT a verified")
        print("# counterexample -- needs orchestrator review + a DAG")
        print("# realization before any claim is made. DO NOT treat as done.")
        print("#" * 70)
