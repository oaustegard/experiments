"""
tests/test_calibration.py -- the campaign's verifier gate (gate 2 of issue
#169). Runnable directly:

    python3 tests/test_calibration.py     # from the ms13-campaign directory

Four gates, in order:

  GATE 1  triangle-conflict calibration (PR #167 fixture, corrected
          off-by-one sup convention -- see beta_star.py's module docstring)
  GATE 2  membership_lp_feasible brute-force agreement, 8 tiny random
          instances (independent scipy.optimize.linprog cross-check against
          the sympy-exact LP)
  GATE 3  hand-derived two-sided verifier sanity (t_of_x computed by hand
          against a small graph-realized instance)
  GATE 4  check_paths_valid catches a deliberately malformed (gap) instance

Exits 0 if every gate passes, 1 otherwise.
"""

import os
import sys
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Instance, Terminal, check_paths_valid, verify_counterexample  # noqa: E402
from beta_star import breakpoints, compute_beta_star, membership_lp_feasible  # noqa: E402
from coverage import triangle_instance, triangle_split  # noqa: E402
from sweep import DEFAULT_DEMAND_POOL, random_layered_dag  # noqa: E402

from scipy.optimize import linprog  # noqa: E402


GATES = []  # (name, passed, detail) accumulated for the summary table


def _record(name: str, passed: bool, detail: str = "") -> None:
    GATES.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# GATE 1: triangle-conflict calibration
# ---------------------------------------------------------------------------

def gate_triangle() -> bool:
    """PR #167 reported beta* = 1/2 under an off-by-one sup convention.

    Hand derivation under the CORRECTED convention (beta* = sup of the
    INFEASIBLE region = the first FEASIBLE breakpoint, per beta_star.py's
    module docstring): the membership LP is infeasible at every breakpoint
    strictly below 1 (in particular at 0 and at 1/2, the two breakpoints
    below 1 that this instance produces) and first becomes feasible exactly
    at beta=1, where the routing that puts every terminal on its E-path
    (loads E1=E2=E3=1, AB=BC=CA=0) plus the routing that puts every terminal
    on its pair-path (loads AB=BC=CA=2, E1=E2=E3=0) can be combined by the
    LP to reproduce x=(AB,BC,CA,E1,E2,E3)=(1,1,1,.5,.5,.5) exactly at
    beta=1 slack. So sup(infeasible region) = beta_star = 1, and the last
    breakpoint scanned before that (1/2) is last_infeasible_breakpoint.
    """
    inst = triangle_instance()
    split = triangle_split()

    result = compute_beta_star(inst, split)
    ok = True
    detail = []
    if result["beta_star"] != Fr(1):
        ok = False
        detail.append(f"beta_star={result['beta_star']} != 1")
    if result["last_infeasible_breakpoint"] != Fr(1, 2):
        ok = False
        detail.append(
            f"last_infeasible_breakpoint={result['last_infeasible_breakpoint']} != 1/2"
        )

    verdict = verify_counterexample(inst, split)
    if verdict["t_of_x"] != Fr(0):
        ok = False
        detail.append(f"t_of_x={verdict['t_of_x']} != 0")
    if verdict["is_counterexample"] is not False:
        ok = False
        detail.append("is_counterexample is not False")

    # NOTE: terminal names (t1/t2/t3) intentionally do not match the pseudo-
    # arc endpoints (AB/CA/...) that their paths end at -- triangle_instance
    # is a load-math fixture (pseudo-arcs), not a graph-realized instance, so
    # check_paths_valid is deliberately not run against it here.

    _record("GATE 1: triangle-conflict calibration (beta*, t_of_x)", ok, "; ".join(detail))
    return ok


# ---------------------------------------------------------------------------
# GATE 2: membership_lp_feasible brute-force agreement
# ---------------------------------------------------------------------------

def _brute_force_membership(inst: Instance, x, beta, tol: float = 1e-7) -> bool:
    """Independent SECOND path to the same yes/no answer as
    beta_star.membership_lp_feasible: is x in conv{f^R : f^R(a) <= x(a) +
    beta*d_max}? Solved via scipy.optimize.linprog (HiGHS simplex/IPM) on a
    phase-1 feasibility formulation, instead of beta_star.py's sympy exact
    LP -- a genuinely different code path, floats with tolerance, per the
    task's explicit allowance ("float cross-check with tolerance is
    acceptable as the SECOND path since the first is exact").
    """
    dmax = float(inst.d_max())
    arcs = inst.arcs
    loads = [inst.routing_load(r) for r in inst.routings()]
    xf = {a: float(x[a]) for a in arcs}
    good = [
        ld for ld in loads
        if all(float(ld[a]) <= xf[a] + float(beta) * dmax + tol for a in arcs)
    ]
    if not good:
        return False
    n = len(good)
    A_eq = [[float(g[a]) for g in good] for a in arcs] + [[1.0] * n]
    b_eq = [xf[a] for a in arcs] + [1.0]
    res = linprog(c=[0.0] * n, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n, method="highs")
    return bool(res.success)


def gate_brute_force() -> bool:
    ok = True
    mismatches = []
    n_checked = 0
    n_built = 0
    attempt = 0
    seed_base = 20260724

    while n_built < 8 and attempt < 400:
        attempt += 1
        n_terminals = 2 if attempt % 2 == 0 else 3
        try:
            inst = random_layered_dag(
                n_layers=2, width=max(2, n_terminals), n_terminals=n_terminals,
                seed=seed_base + attempt, demand_pool=DEFAULT_DEMAND_POOL,
                max_routings=50,
            )
        except RuntimeError:
            continue
        # keep it tiny per the gate's spec: <= 3 terminals, <= 2-3 paths each
        if any(len(t.paths) > 3 for t in inst.terminals):
            continue
        n_built += 1

        split = [[Fr(1, len(t.paths))] * len(t.paths) for t in inst.terminals]
        x = inst.x_from_split(split)
        bps = breakpoints(inst, x)
        top = bps[-1] + 1 if bps else Fr(1)
        uniq = sorted(set([Fr(0)] + bps + [top]))
        mids = [(uniq[i] + uniq[i + 1]) / 2 for i in range(len(uniq) - 1)]

        for beta in uniq + mids:
            exact_feasible, _ = membership_lp_feasible(inst, x, beta)
            brute_feasible = _brute_force_membership(inst, x, beta)
            n_checked += 1
            if exact_feasible != brute_feasible:
                ok = False
                mismatches.append((n_built, str(beta), exact_feasible, brute_feasible))

    detail = f"{n_checked} (instance,beta) points checked across {n_built} tiny random instances"
    if mismatches:
        detail += f"; mismatches (first 5): {mismatches[:5]}"
    _record("GATE 2: membership_lp_feasible brute-force agreement", ok, detail)
    return ok


# ---------------------------------------------------------------------------
# GATE 3: hand-derived two-sided verifier sanity
# ---------------------------------------------------------------------------

def gate_two_sided_hand() -> bool:
    """
    Hand-built 2-terminal, graph-realized instance:

        s -> m -> A     (shared trunk s-m, then private hop m-A)
        s -> m -> B     (shared trunk s-m, then private hop m-B)
        s -> A          (private bypass for A)

      Terminal A: demand 2, paths = [ (s-m, m-A),  (s-A,) ]
      Terminal B: demand 2, single path = [ (s-m, m-B) ]   (no alternative)
      Split:      A = [1/2, 1/2],  B = [1]

    Fractional loads:
      x(s-m) = 2*(1/2)[A via trunk] + 2*1[B, forced] = 1 + 2 = 3
      x(m-A) = 2*(1/2)[A via trunk]                  = 1
      x(m-B) = 2*1[B, forced]                        = 2
      x(s-A) = 2*(1/2)[A via bypass]                 = 1
      d_max  = max(2, 2) = 2

    Two routings exist (A in {path0, path1}; B has no choice):

      A=path0 (trunk): loads s-m=2+2=4, m-A=2, m-B=2, s-A=0
        deviations: |4-3|=1, |2-1|=1, |2-2|=0, |0-1|=1  -> worst = 1
      A=path1 (bypass): loads s-m=0+2=2, m-A=0, m-B=2, s-A=2
        deviations: |2-3|=1, |0-1|=1, |2-2|=0, |2-1|=1  -> worst = 1

    min over routings = 1 (a tie);  t_of_x = 1 - d_max = 1 - 2 = -1.
    """
    arcs = [("s", "m"), ("m", "A"), ("m", "B"), ("s", "A")]
    term_a = Terminal("A", Fr(2), [(("s", "m"), ("m", "A")), (("s", "A"),)])
    term_b = Terminal("B", Fr(2), [(("s", "m"), ("m", "B"))])
    inst = Instance(arcs=arcs, terminals=[term_a, term_b])

    ok = True
    detail = []

    problems = check_paths_valid(inst, "s")
    if problems:
        ok = False
        detail.append(f"check_paths_valid found problems on a well-formed fixture: {problems}")

    split = [[Fr(1, 2), Fr(1, 2)], [Fr(1)]]
    result = verify_counterexample(inst, split)
    if result["t_of_x"] != Fr(-1):
        ok = False
        detail.append(f"t_of_x={result['t_of_x']} != -1")

    _record("GATE 3: hand-derived two-sided verifier sanity", ok, "; ".join(detail))
    return ok


# ---------------------------------------------------------------------------
# GATE 4: check_paths_valid catches a malformed (gap) instance
# ---------------------------------------------------------------------------

def gate_malformed() -> bool:
    # arc "m -> A" is declared but the terminal's path instead jumps m -> x
    # (a node it never actually has an arc for) -- a gap.
    arcs = [("s", "m"), ("m", "A"), ("x", "A")]
    bad_terminal = Terminal("A", Fr(1), [(("s", "m"), ("x", "A"))])
    inst = Instance(arcs=arcs, terminals=[bad_terminal])

    problems = check_paths_valid(inst, "s")
    ok = len(problems) > 0 and any("gap" in p for p in problems)
    _record(
        "GATE 4: check_paths_valid catches a malformed (gap) instance",
        ok, f"problems={problems}",
    )
    return ok


# ---------------------------------------------------------------------------

def gate5_milp_two_sided() -> bool:
    """GATE 5 (phase 4): the certified MILP decision procedure must detect
    BOTH directions, with its transition at an independently known value.

    On the Rybin instance the best w achieves min-max deviation exactly 10
    against d_max = 15 (t* = -5, established in phase 1 and re-derived by the
    out-tree reduction in phase 3). So `exists_bad_w` must report a bad w for
    every threshold below 10 and prove none for every threshold above it.
    A one-sided test would pass on a trivially-infeasible (broken) encoding --
    the polarity bug this gate was written to catch did exactly that.
    """
    from fractions import Fraction as Fr
    from outtree import OutTreeInstance
    from discrepancy import exists_bad_w

    ryb = OutTreeInstance('s', [('s', 'a'), ('a', 'b'), ('b', 'c')],
                          [('c', 'a'), ('c', 's'), ('b', 's')],
                          [Fr(15), Fr(10), Fr(15)])
    ok = True
    for factor, expect_bad in ((0.4, True), (0.6, True), (0.65, True),
                               (0.70, False), (1.0, False)):
        r = exists_bad_w(ryb, bound_factor=factor)
        got = r['feasible']
        if got != expect_bad:
            print(f"  [FAIL] bound_factor={factor}: expected feasible={expect_bad}, got {got}")
            ok = False
        if not got and not r.get('proved_no_bad_w'):
            print(f"  [FAIL] bound_factor={factor}: negative was not PROVED "
                  f"(solver error, not infeasibility): {r.get('reason')}")
            ok = False
    print(f"[{'PASS' if ok else 'FAIL'}] GATE 5: certified MILP two-sided calibration "
          f"(transition brackets Rybin's known optimum 10)")
    return ok


def _record_gate5() -> bool:
    ok = gate5_milp_two_sided()
    GATES.append(("GATE 5: certified MILP two-sided calibration", ok, ""))
    return ok


def main() -> int:
    results = [
        gate_triangle(),
        gate_brute_force(),
        gate_two_sided_hand(),
        gate_malformed(),
        _record_gate5(),
    ]

    print()
    print("=" * 70)
    print("GATE SUMMARY")
    print("=" * 70)
    for name, passed, _detail in GATES:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print("=" * 70)

    if all(results):
        print("ALL GATES PASS")
        return 0
    print("GATE FAILURE -- see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
