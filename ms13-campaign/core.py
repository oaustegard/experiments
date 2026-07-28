"""
ms13-campaign core: exact machinery for Morell-Skutella Conjecture 1.3
(claude-workspace issue #169).

Conjecture 1.3 (Morell-Skutella): for every acyclic SSUF network (DAG G,
source s, terminals T with demands d) and every fractional flow x in Q_G,
there is an unsplittable routing P with

    x(a) - d_max  <=  f^P(a)  <=  x(a) + d_max      for all arcs a.

A counterexample is (G, d, x) where EVERY routing violates the two-sided
bound on some arc. Verification of a candidate is finite: enumerate all
routings, check in exact rationals.

Everything in the certificate path is fractions.Fraction. Floats appear
only in screening heuristics (sweep.py), never in a decision.

Vocabulary used throughout the campaign (NOGOS.md, gadgets.py):
  ceiling at a:  f^P(a) <= x(a) + d_max   (binds when co-resident demands stack)
  floor at a:    f^P(a) >= x(a) - d_max   (non-vacuous only when x(a) > d_max,
                                           in which case SOME terminal must use a)
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Fr
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

Arc = Tuple[str, str]


@dataclass
class Terminal:
    name: str
    demand: Fr
    # each path is a tuple of arcs from s to this terminal
    paths: List[Tuple[Arc, ...]]


@dataclass
class Instance:
    """SSUF instance: arc set + terminals with enumerated s->t_i path sets.

    The fractional flow x is supplied separately (a point in the product of
    per-terminal path simplices) so one instance can be tested against many
    adversarial x. Use `x_from_split` for the induced arc loads.
    """
    arcs: List[Arc]
    terminals: List[Terminal]

    def d_max(self) -> Fr:
        return max(t.demand for t in self.terminals)

    def routings(self) -> List[Tuple[int, ...]]:
        """All unsplittable routings: one path index per terminal."""
        return list(product(*(range(len(t.paths)) for t in self.terminals)))

    def routing_load(self, routing: Sequence[int]) -> Dict[Arc, Fr]:
        load = {a: Fr(0) for a in self.arcs}
        for t, pi in zip(self.terminals, routing):
            for a in t.paths[pi]:
                load[a] += t.demand
        return load

    def x_from_split(self, split: Sequence[Sequence[Fr]]) -> Dict[Arc, Fr]:
        """Arc loads of the fractional flow given per-terminal path weights.

        split[i] must be a distribution over terminals[i].paths (sum 1, >= 0);
        validated here because a silently-unnormalized split corrupts every
        downstream verdict.
        """
        assert len(split) == len(self.terminals)
        x = {a: Fr(0) for a in self.arcs}
        for t, ws in zip(self.terminals, split):
            assert len(ws) == len(t.paths)
            assert all(w >= 0 for w in ws) and sum(ws) == 1, \
                f"split for {t.name} is not a distribution: {ws}"
            for path, w in zip(t.paths, ws):
                for a in path:
                    x[a] += t.demand * Fr(w)
        return x

    def routing_label(self, routing: Sequence[int]) -> str:
        """Z = path 0 (by convention the cheap option), E = path 1; digits
        when a terminal has >2 paths. Matches the ZZZ/EZZ/... naming used in
        issues #165/#169."""
        return "".join(
            str(pi) if len(t.paths) > 2 else "ZE"[pi]
            for t, pi in zip(self.terminals, routing)
        )


# ---------------------------------------------------------------------------
# Two-sided violation (the Conjecture 1.3 objective)
# ---------------------------------------------------------------------------

def two_sided_violation(inst: Instance, x: Dict[Arc, Fr],
                        routing: Sequence[int]) -> Fr:
    """max_a |f^P(a) - x(a)| for this routing."""
    load = inst.routing_load(routing)
    return max(abs(load[a] - x[a]) for a in inst.arcs)


def best_routing_violation(inst: Instance, x: Dict[Arc, Fr]
                           ) -> Tuple[Fr, Tuple[int, ...]]:
    """min over routings of the worst-arc two-sided deviation, with argmin.

    (G, d, x) is a Conjecture 1.3 counterexample iff this value > d_max.
    """
    best: Optional[Fr] = None
    best_r: Optional[Tuple[int, ...]] = None
    for r in inst.routings():
        v = two_sided_violation(inst, x, r)
        if best is None or v < best:
            best, best_r = v, r
    assert best is not None and best_r is not None
    return best, best_r


def verify_counterexample(inst: Instance, split: Sequence[Sequence[Fr]],
                          ) -> Dict:
    """Exact verdict on a candidate (G, d, x): is every routing violating?

    Returns the full per-routing table (label, worst deviation, worst arc)
    so a positive verdict is already most of a certificate. Exact rationals
    throughout; no tolerance anywhere.
    """
    x = inst.x_from_split(split)
    dmax = inst.d_max()
    table = []
    all_violate = True
    for r in inst.routings():
        load = inst.routing_load(r)
        dev, arc = max(
            ((abs(load[a] - x[a]), a) for a in inst.arcs),
            key=lambda p: p[0],
        )
        violates = dev > dmax
        all_violate = all_violate and violates
        table.append({
            "routing": r,
            "label": inst.routing_label(r),
            "worst_deviation": dev,
            "worst_arc": arc,
            "violates": violates,
        })
    return {
        "is_counterexample": all_violate,
        "d_max": dmax,
        "x": x,
        "table": table,
        # t-value of this particular x: min_P max_a |f - x| - d_max.
        "t_of_x": min(row["worst_deviation"] for row in table) - dmax,
    }


# ---------------------------------------------------------------------------
# Flow-conservation sanity check (structural, catches malformed instances)
# ---------------------------------------------------------------------------

def check_paths_valid(inst: Instance, source: str) -> List[str]:
    """Every declared path must be a connected arc sequence from `source` to
    its terminal. Returns a list of problems (empty = clean). Instances built
    by generators MUST pass this before any verdict is trusted -- a path with
    a gap silently under-loads arcs and fakes low violations.
    """
    problems = []
    for t in inst.terminals:
        if not t.paths:
            problems.append(f"{t.name}: no paths")
        for k, path in enumerate(t.paths):
            if not path:
                problems.append(f"{t.name} path {k}: empty")
                continue
            if path[0][0] != source:
                problems.append(f"{t.name} path {k}: starts at {path[0][0]}, not {source}")
            for (u, v), (u2, v2) in zip(path, path[1:]):
                if v != u2:
                    problems.append(f"{t.name} path {k}: gap {v} -> {u2}")
            if path[-1][1] != t.name:
                problems.append(f"{t.name} path {k}: ends at {path[-1][1]}, not {t.name}")
            unknown = [a for a in path if a not in inst.arcs]
            if unknown:
                problems.append(f"{t.name} path {k}: arcs not in instance: {unknown}")
    return problems


# ---------------------------------------------------------------------------
# Path closure (added session 2 after the NG-7 false positive)
# ---------------------------------------------------------------------------

def graph_paths(inst: Instance, source: str, target: str) -> List[Tuple[Arc, ...]]:
    """ALL source->target paths in the instance's arc set (graph level)."""
    succ: Dict[str, List[str]] = {}
    for (u, v) in inst.arcs:
        succ.setdefault(u, []).append(v)
    out: List[Tuple[Arc, ...]] = []
    stack: List[Tuple[str, Tuple[Arc, ...]]] = [(source, ())]
    while stack:
        node, acc = stack.pop()
        if node == target:
            out.append(acc)
            continue
        for w in succ.get(node, []):
            stack.append((w, acc + ((node, w),)))
    return out

def check_path_closure(inst: Instance, source: str) -> List[str]:
    """MS 1.3 quantifies over ALL s->t_i paths of the DAG, not a declared
    menu. A declared path system that is a strict subset of the graph's
    paths makes verify_counterexample check the wrong routing space --
    the NG-7 false positive (2026-07-24) passed dual code paths this way:
    20-arc realization, 2 declared paths/terminal, 4x8x8 real routings,
    23 of them satisfying. Returns problems; empty = closed. Run this on
    every realized instance BEFORE any counterexample verdict."""
    problems = []
    for t in inst.terminals:
        gp = set(graph_paths(inst, source, t.name))
        declared = set(t.paths)
        extra = gp - declared
        missing = declared - gp
        if extra:
            problems.append(f"{t.name}: {len(extra)} graph paths not declared")
        if missing:
            problems.append(f"{t.name}: {len(missing)} declared paths not in graph")
    return problems
