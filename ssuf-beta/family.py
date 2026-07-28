"""
Parametrized generalization of the triangle-conflict construction in calibration.py,
following the shape issue #165 requested for the K4-subdivision family: normalized
demands (1, b, 1), per-terminal cheap-path ("Z") split fractions (r, q, r).

NOTE ON PROVENANCE: this is NOT Rybin's K4-subdivision graph (that topology is
unverified in this session -- see calibration.py docstring and RESULTS.md). This
is an independently-constructed family with the same qualitative mechanism (three
pairwise-conflicting cheap paths on a triangle of shared arcs). Findings here are
honest results about THIS family, not about the Rybin instance or any claim to
beat/reproduce beta*=16/15.

Topology (fixed across the family, only demands/splits vary):
  t1: Z touches {AB, CA}, E touches {E1}, demand 1
  t2: Z touches {AB, BC}, E touches {E2}, demand b
  t3: Z touches {BC, CA}, E touches {E3}, demand 1
"""

from fractions import Fraction as Fr
from typing import Tuple

from engine import Instance, Terminal, compute_beta_star


def build_family_instance(b: Fr, r: Fr, q: Fr) -> Instance:
    arcs = [("s", "AB"), ("s", "BC"), ("s", "CA"), ("s", "E1"), ("s", "E2"), ("s", "E3")]
    AB, BC, CA, E1, E2, E3 = arcs

    t1 = Terminal("t1", Fr(1), paths=[(AB, CA), (E1,)], split=[r, 1 - r])
    t2 = Terminal("t2", Fr(b), paths=[(AB, BC), (E2,)], split=[q, 1 - q])
    t3 = Terminal("t3", Fr(1), paths=[(BC, CA), (E3,)], split=[r, 1 - r])

    return Instance(arcs=arcs, terminals=[t1, t2, t3])


def beta_star_at(b: Fr, r: Fr, q: Fr) -> Tuple[Fr, Fr]:
    inst = build_family_instance(b, r, q)
    result = compute_beta_star(inst, verbose=False)
    return result["beta_star"], result["first_feasible_beta"]


def grid_search(b_values, rq_values, verbose: bool = True, label: str = "grid"):
    """Rational grid search over (b, r, q) for max beta*, printing progress
    incrementally (flush=True) so a time-limited run still yields partial
    results instead of nothing -- the first version of this function printed
    only at the end and lost an entire ~9-minute background run to a timeout
    with zero captured output.
    """
    best = None
    results = []
    total = len(b_values) * len(rq_values) * len(rq_values)
    n_done = 0
    for b in b_values:
        for r in rq_values:
            for q in rq_values:
                n_done += 1
                try:
                    bstar, first_feas = beta_star_at(b, r, q)
                except Exception:
                    continue
                if bstar is None:
                    continue
                results.append((bstar, b, r, q, first_feas))
                if best is None or bstar > best[0]:
                    best = (bstar, b, r, q, first_feas)
                    if verbose:
                        print(f"[{label} {n_done}/{total}] new best beta*={bstar} "
                              f"({float(bstar):.4f}) at b={b} r={r} q={q}", flush=True)
                elif verbose and n_done % 25 == 0:
                    print(f"[{label} {n_done}/{total}] ...", flush=True)

    results.sort(key=lambda t: -t[0])
    if verbose:
        print(f"[{label}] done: searched {len(results)}/{total} feasible points.")
        if best:
            print(f"[{label}] best: beta*={best[0]} ({float(best[0]):.4f}) "
                  f"at b={best[1]}, r={best[2]}, q={best[3]}", flush=True)
    return best, results


def coarse_then_refine(verbose: bool = True):
    """Stage 1: coarse denom=4 grid over a wide b range. Stage 2: finer
    denom=16 grid restricted to a neighborhood of the coarse best. Bounded
    total LP-call count (~150 + ~150) so it finishes well inside a few
    minutes even with sympy's pure-Python exact simplex as the bottleneck.
    """
    coarse_b = [Fr(n, 4) for n in range(1, 5)] + [Fr(n) for n in range(2, 6)]
    coarse_rq = [Fr(n, 4) for n in range(1, 4)]
    if verbose:
        print(f"Stage 1 (coarse): {len(coarse_b)} b-values x {len(coarse_rq)}^2 (r,q) "
              f"= {len(coarse_b) * len(coarse_rq) ** 2} points", flush=True)
    coarse_best, coarse_results = grid_search(coarse_b, coarse_rq, verbose=verbose, label="coarse")

    if coarse_best is None:
        return None, coarse_results

    _, b0, r0, q0, _ = coarse_best
    denom = 16
    step = Fr(1, denom)

    def neighborhood(center, lo=Fr(1, denom), hi=None):
        vals = set()
        for k in (-2, -1, 0, 1, 2):
            v = center + k * step
            if v > 0 and (hi is None or v < hi):
                vals.add(v)
        return sorted(vals)

    fine_b = neighborhood(b0, hi=None) or [b0]
    fine_rq = neighborhood(r0, hi=1) or [r0]
    fine_rq2 = neighborhood(q0, hi=1) or [q0]
    fine_rq_union = sorted(set(fine_rq) | set(fine_rq2))
    if verbose:
        print(f"Stage 2 (refine around b={b0},r={r0},q={q0}): "
              f"{len(fine_b)} b-values x {len(fine_rq_union)}^2 (r,q) "
              f"= {len(fine_b) * len(fine_rq_union) ** 2} points", flush=True)
    fine_best, fine_results = grid_search(fine_b, fine_rq_union, verbose=verbose, label="fine")

    overall_best = fine_best if (fine_best and fine_best[0] > coarse_best[0]) else coarse_best
    all_results = coarse_results + fine_results
    all_results.sort(key=lambda t: -t[0])

    if verbose:
        print(f"\nOverall best: beta*={overall_best[0]} ({float(overall_best[0]):.4f}) "
              f"at b={overall_best[1]}, r={overall_best[2]}, q={overall_best[3]}", flush=True)
        print("Calibration triangle case (b=1,r=q=1/2) gives beta*=1/2 -- "
              "this search's max is the comparison point.", flush=True)

    return overall_best, all_results


if __name__ == "__main__":
    coarse_then_refine()
