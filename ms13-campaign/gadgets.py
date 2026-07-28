"""
Kill-path A framework (issue #169, phase 2): implication-gadget clause systems.

A gadget design is abstracted as a CLAUSE SYSTEM over per-terminal Boolean
path choices (every terminal has exactly 2 paths, per the splice-closure
guidance -- fewest paths safest):

  floor clause   (arc F, eligible set E ⊆ {(i, pi)}):
      the routing must pick some (i, pi) ∈ E, else the floor f(F) >= x(F) - d_max
      is violated. Realizing it costs the mass constraint x(F) > d_max.
  conflict clause (arc c, {(i, pi), (j, pj)}):
      routings choosing both paths violate c's ceiling. Realizing it costs
      x(c) < d_i + d_j - d_max.

A design is a COUNTEREXAMPLE CANDIDATE iff (a) the clause system is UNSAT
over the 2^k assignments, and (b) the mass constraints admit a consistent
fractional flow (exact LP below), and (c) the abstract arc-sharing pattern is
realizable as an acyclic DAG (separate step -- NOT checked here; candidates
surviving (a)+(b) still need DAG realization before any counterexample talk).

Pruning applied before the LP (NOGOS.md):
  Lemma 2 (NG-3): floors form an antichain here (distinct abstract arcs), so
      more than ceil(D/d_max)-1 floors is infeasible outright.
  Lemma 3 (NG-6): summation over a fully-blocked terminal's two paths yields
      demand-only inequalities; contradictions kill without an LP call.

Everything exact (fractions.Fraction); sympy lpmin solves the strict-
inequality system via an epsilon-margin maximization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as Fr
from itertools import product
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from sympy import symbols as _sp_symbols
from sympy.solvers.simplex import lpmax as _sp_lpmax, InfeasibleLPError as _SpInfeasible

PathRef = Tuple[int, int]  # (terminal index, path index in {0,1})


@dataclass(frozen=True)
class Floor:
    arc: str
    eligible: FrozenSet[PathRef]  # paths that traverse the floor arc


@dataclass(frozen=True)
class Conflict:
    arc: str
    pair: FrozenSet[PathRef]      # exactly two PathRefs of distinct terminals


@dataclass
class ClauseSystem:
    demands: List[Fr]
    floors: List[Floor]
    conflicts: List[Conflict]

    @property
    def k(self) -> int:
        return len(self.demands)

    def d_max(self) -> Fr:
        return max(self.demands)

    # -- (a) combinatorial UNSAT ---------------------------------------------

    def assignment_ok(self, choice: Sequence[int]) -> bool:
        for fl in self.floors:
            if not any(choice[i] == pi for (i, pi) in fl.eligible):
                return False
        for cf in self.conflicts:
            if all(choice[i] == pi for (i, pi) in cf.pair):
                return False
        return True

    def is_unsat(self) -> bool:
        return not any(self.assignment_ok(c) for c in product((0, 1), repeat=self.k))

    # -- (b) exact mass-consistency LP ---------------------------------------

    def arc_mass(self, refs: FrozenSet[PathRef], ws) -> object:
        """Mass contributed to an arc traversed by exactly `refs`:
        sum_i d_i * (w_i if path 0 else 1 - w_i). Terminals not in refs are
        assumed to avoid the arc entirely (the design's sharing pattern is
        exact -- extra traversals would only RAISE conflict-arc mass and make
        realization harder, so this is the design's best case)."""
        total = 0
        for (i, pi) in refs:
            total += self.demands[i] * (ws[i] if pi == 0 else 1 - ws[i])
        return total

    def realizable_x(self) -> Optional[Dict]:
        """Exact feasibility of the mass constraints with maximum margin eps:
            x(F)  >= d_max + eps            for every floor
            x(c)  <= d_i + d_j - d_max - eps  for every conflict
            0 <= w_i <= 1
        Returns {'eps': Fr, 'w': [Fr..]} if a strictly positive margin exists,
        else None. Strictness matters: eps = 0 boundary points realize nothing
        (floors/ceilings would bind with equality and the routing survives)."""
        ws = _sp_symbols(f"w0:{self.k}")
        eps = _sp_symbols("eps")
        dmax = self.d_max()
        cons = []
        for w in ws:
            cons.append(w >= 0)
            cons.append(w <= 1)
        cons.append(eps >= 0)
        cons.append(eps <= max(self.demands))  # bounded objective
        for fl in self.floors:
            cons.append(self.arc_mass(fl.eligible, ws) - dmax - eps >= 0)
        for cf in self.conflicts:
            (i, _), (j, _) = sorted(cf.pair)
            cap = self.demands[i] + self.demands[j] - dmax
            cons.append(self.arc_mass(cf.pair, ws) - cap + eps <= 0)
        try:
            best, sol = _sp_lpmax(eps, cons)
        except _SpInfeasible:
            return None
        if Fr(best) <= 0:
            return None
        return {"eps": Fr(best), "w": [Fr(sol[w]) for w in ws]}

    # -- pruning -------------------------------------------------------------

    def floor_budget_ok(self) -> bool:
        """Lemma 2: sum of floor masses <= D, each floor needs > d_max."""
        D = sum(self.demands)
        return len(self.floors) * self.d_max() < D

    def lemma3_prune(self) -> Optional[str]:
        """Detect a fully-blocked terminal whose summation inequality
        contradicts a floor. Returns the killing reason, or None.
        Covers the NG-6 shape: floor with eligible {(i,pi),(j,pj)} plus
        conflicts pairing BOTH paths of some terminal t against those same
        two paths. The general LP subsumes this; the point is a free, exact
        reason string for the ledger."""
        dmax = self.d_max()
        D = sum(self.demands)
        for fl in self.floors:
            if len(fl.eligible) != 2:
                continue
            elig = sorted(fl.eligible)
            for t in range(self.k):
                if any(t == i for (i, _) in elig):
                    continue
                # need all four conflicts (elig path) x (both t paths)
                needed = {frozenset({e, (t, b)}) for e in elig for b in (0, 1)}
                have = {cf.pair for cf in self.conflicts}
                if needed <= have:
                    # sum the four conflicts: mass(elig) < D - 2*dmax <= dmax
                    if D - 2 * dmax <= dmax:
                        return (f"NG-6: floor {fl.arc} eligible {elig} fully "
                                f"blocked vs terminal {t}; summation gives "
                                f"mass < D-2dmax <= dmax, contradicting floor")
        return None

    def attack(self) -> Dict:
        """Full abstract-level verdict for this design."""
        if not self.is_unsat():
            return {"verdict": "SAT", "reason": "some assignment satisfies all clauses"}
        if not self.floor_budget_ok():
            return {"verdict": "KILLED", "reason":
                    "NG-3 floor budget: #floors * d_max >= D"}
        reason = self.lemma3_prune()
        if reason is not None:
            return {"verdict": "KILLED", "reason": reason}
        x = self.realizable_x()
        if x is None:
            return {"verdict": "KILLED", "reason":
                    "mass LP infeasible (no consistent x)"}
        return {"verdict": "CANDIDATE", "x": x, "reason":
                "UNSAT + consistent x at abstract level; needs DAG realization"}


def ng6_instance() -> ClauseSystem:
    """The NG-6 shape, kept as an executable regression: must be KILLED."""
    d = [Fr(1), Fr(1), Fr(1)]
    fl = Floor("F", frozenset({(0, 0), (1, 0)}))
    cfs = [Conflict(f"c{n}", frozenset({e, (2, b)}))
           for n, (e, b) in enumerate(((e, b) for e in ((0, 0), (1, 0)) for b in (0, 1)))]
    return ClauseSystem(d, [fl], cfs)


if __name__ == "__main__":
    cs = ng6_instance()
    print("NG-6 regression:", cs.attack())
