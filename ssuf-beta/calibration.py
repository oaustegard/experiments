"""
Self-hosted calibration gate for the beta* engine.

IMPORTANT — why this is NOT the Rybin instance:
Issue #165 specifies calibration against Dmitry Rybin's published counterexample
(7 vertices, 9 arcs, demands 15/10/15, beta* = 16/15 exactly). That instance has
NO arXiv writeup (confirmed via literature-gate agent, 2026-07-24) and exists only
as an X/Twitter thread (x.com/DmitryRybin1/status/2079904005652893709) which this
session's WebFetch cannot reach (HTTP 402), with no working nitter mirror either.
Secondary sources (officechai.com, vibemathed.com) confirm only the high-level
facts (7 nodes, demands 15/10/15, fractional cost 58, unsplittable-min cost 60)
but do NOT reproduce arc-level topology or flow values. Prior in-session memory
(2026-07-22) confirms these facts were independently verified against an uploaded
PDF certificate that session, but the PDF itself does not persist across sessions
and is not retrievable here.

Rather than guess arc-level topology and risk silently mis-reporting a "calibration
pass" against a fabricated instance (the exact failure mode this whole research
programme exists to avoid -- see memory 8eb1f1c8, "let the AI report zero" /
CDC-episode contrast), this file instead hand-derives an INDEPENDENT small SSUF
instance with the same qualitative mechanism (three pairwise-conflicting cheap
paths forming a triangle conflict), computes its beta* by hand below, and asserts
the engine reproduces that hand-derived value exactly. This validates the full
pipeline (Instance -> routings -> breakpoints -> exact rational membership LP)
end-to-end; it does NOT validate against the Rybin ground truth, and no claim in
this repo should be read as reproducing or exceeding beta*=16/15 until the real
instance data can be retrieved (see RESULTS.md "Blocked" section).

HAND DERIVATION (triangle-conflict instance, demands 1/1/1, split p=1/2):
  3 terminals t1,t2,t3, demand 1 each (dmax=1). Each has 2 paths:
    Z (cheap): t1 uses arcs {AB,CA}; t2 uses {AB,BC}; t3 uses {BC,CA}
    E (expensive, dedicated): t1->E1, t2->E2, t3->E3
  Fractional split: each terminal sends 1/2 on Z, 1/2 on E.
    x(AB)=x(BC)=x(CA)=1, x(E1)=x(E2)=x(E3)=1/2.
  Per-routing worst-arc overshoot (worked by hand, see experiments/ssuf-beta/
  RESULTS.md "Self-hosted calibration" section for the full derivation table):
    ZZZ and every one-E routing:  worst overshoot = 1        (breakpoint beta=1)
    every two-E routing and EEE:  worst overshoot = 1/2      (breakpoint beta=1/2)
  At beta=1/2, the only available (beta-good) routings are the 3 two-E routings
  + EEE (4 total). Solving membership exactly: the three "shared triangle arc"
  equality constraints force w1+w2+w3 = 3/2 while total weight is capped at 1,
  giving w4 = 1 - 3/2 = -1/2 < 0 -- INFEASIBLE. So x is not in conv(good set) at
  beta=1/2; the LP first becomes feasible at beta=1 (all 8 routings available,
  and x is exactly the product-distribution average over independent p=1/2
  Bernoulli choices per terminal, feasible with uniform weight 1/8 each).
  Therefore beta* = 1/2 by hand, first_feasible_beta = 1.
"""

from fractions import Fraction as Fr
from engine import Instance, Terminal, compute_beta_star


def build_triangle_instance() -> Instance:
    half = Fr(1, 2)
    arcs = [("s", "AB"), ("s", "BC"), ("s", "CA"), ("s", "E1"), ("s", "E2"), ("s", "E3")]
    AB, BC, CA, E1, E2, E3 = arcs

    t1 = Terminal("t1", Fr(1), paths=[(AB, CA), (E1,)], split=[half, half])
    t2 = Terminal("t2", Fr(1), paths=[(AB, BC), (E2,)], split=[half, half])
    t3 = Terminal("t3", Fr(1), paths=[(BC, CA), (E3,)], split=[half, half])

    return Instance(arcs=arcs, terminals=[t1, t2, t3])


def run_calibration(verbose: bool = True) -> bool:
    inst = build_triangle_instance()
    result = compute_beta_star(inst, verbose=verbose)
    expected_beta_star = Fr(1, 2)
    expected_first_feasible = Fr(1)

    ok = (result["beta_star"] == expected_beta_star and
          result["first_feasible_beta"] == expected_first_feasible)

    if verbose:
        print(f"beta_star computed = {result['beta_star']} (expected {expected_beta_star})")
        print(f"first_feasible_beta computed = {result['first_feasible_beta']} "
              f"(expected {expected_first_feasible})")
        print("witness weights at first-feasible beta:",
              [str(w) for w in result["witness_weights"]])
        print("CALIBRATION " + ("PASS" if ok else "FAIL"))

    return ok


if __name__ == "__main__":
    passed = run_calibration()
    if not passed:
        raise SystemExit(1)
