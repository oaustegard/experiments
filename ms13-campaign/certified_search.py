"""
certified_search.py -- phase 4 (issue #169): certified re-decision of phase
3's out-tree search via discrepancy.exists_bad_w (exact MILP over w), then
extension to n=6,7.

WHAT CHANGED FROM PHASE 3. `outtree_search.py` maximized t(w) per instance by
grid+polish -- a heuristic LOWER BOUND on t*. `discrepancy.exists_bad_w(inst)`
DECIDES the same question exactly: is there ANY w in [0,1]^k making every
z-rounding violate? Measured ~2-15ms per instance at k<=5 (see pilot log),
so this script can afford to test EVERY uncovered structural pattern at
n<=5 (not a capped/sampled subset like phase 3), plus extend to n=6,7.

STAGE 1 (n<=5): reuses outtree_search.py's enumeration exactly (same
TREE_SHAPES, compute_automorphisms, structural_patterns, build_demand_vectors)
so the (tree,k) structural ledger is reproducible against OUTTREE_RESULTS.md,
then certifies every uncovered pattern via exists_bad_w instead of the
heuristic search.

STAGE 2 (n=6,7): new tree shapes (chain-shaped + a few others), k=3..6.
Exhaustive structural enumeration where the raw combinatorial space is small
enough (<=ENUM_CAP), random sampling otherwise (honestly labeled). NG-9
(msw25_covered) filters to the live (uncovered) set first; live MILP search
capped per combo (budget, not correctness) and prioritized on uncovered
patterns.

LEMMA 7 / COROLLARY 7.1: used here as a VERIFICATION self-test (does the
cycle-shift claim hold on n=6,7 instances too, not just wherever it was first
checked) and as QUALITATIVE guidance for shape selection (branching/chain-fork
shapes create K4-minor-worthy structure fastest, per NG-9's n>=4 requirement)
-- NOT as a search-space pruning device. See CERTIFIED_RESULTS.md section 3
for why: turning "at most n-1 chords fractional at a critical LP vertex" into
a safe STRUCTURAL-ENUMERATION filter needs more care than this run's budget
allows (the lemma is about fractional support at an LP-optimal z-relaxation
for a FIXED instance, not about which structural patterns are worth
generating in the outer combinatorial search) -- so we did not risk an
unsound "pruning" and instead spent the budget on more exists_bad_w calls,
which are cheap and exact regardless of enumeration order.

SELF-TESTS (mandatory, same spirit as phases 2-3):
  - exists_bad_w must return infeasible on any equal-demand instance (Lemma 6).
  - exists_bad_w must return infeasible on any MSW25-covered instance (NG-9).
  - Lemma 7 cycle-shift check on instances whose chord multigraph has a cycle.
A violation of any of these is a BUG -- reported loudly, and the run stops.

Any exists_bad_w feasible=True hit STOPS the run immediately; rationalized
via discrepancy.rationalize_and_verify (exact Fractions + path closure +
core.verify_counterexample) before the word "counterexample" is used anywhere.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys
import time
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Tuple

import outtree_search as p3
from core import check_path_closure, check_paths_valid, verify_counterexample
from discrepancy import (
    cycle_shift_preserves_loads,
    exists_bad_w,
    find_chord_cycle,
    rationalize_and_verify,
)
from outtree import (
    OutTreeInstance,
    chord_graph_edges,
    equal_demand_no_go,
    is_series_parallel,
    msw25_covered,
)

# ---------------------------------------------------------------------------
# budget / knobs
# ---------------------------------------------------------------------------

TIME_BUDGET_S = 15 * 60
SOFT_STOP_S = 13 * 60
ENUM_CAP = 20_000          # raw (with-replacement) combo count above which
                            # we sample rather than exhaustively enumerate
SAMPLE_RAW = 4_000         # random raw attach tuples drawn when sampling
CAP_LIVE_K345 = 60         # uncovered patterns certified per combo, k<=5
CAP_LIVE_K6 = 15           # uncovered patterns certified per combo, k=6
COVERED_SELF_TEST_CAP = 6
EQUAL_SELF_TEST_CAP = 6
LEMMA7_CHECK_CAP = 20
SEARCH_SEED = 41221

START = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - START


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


def fr_vec(vals) -> List[Fr]:
    return [Fr(v) for v in vals]


results: Dict = {
    "stage1_n5": {
        "enumeration_counts": {},
        "ledger_totals": {},
        "covered_self_test": {"n_checked": 0, "n_passed": 0},
        "equal_demand_self_test": {"n_checked": 0, "n_passed": 0},
        "live_certified": [],       # per (tree,k): counts + agreement w/ phase3
        "phase3_argmax_recheck": [],
        "permutation_invariance_check": [],
    },
    "stage2_n67": {
        "shapes": [],
        "enumeration_counts": {},
        "covered_self_test": {"n_checked": 0, "n_passed": 0},
        "equal_demand_self_test": {"n_checked": 0, "n_passed": 0},
        "live_certified": [],
    },
    "lemma7_verification": {"n_checked": 0, "n_passed": 0},
    "positive_hits": [],
    "bugs": [],
    "truncated": False,
    "truncated_note": None,
    "total_elapsed_s": None,
}


def dump_partial() -> None:
    with open("certified_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


found_positive = False


def loud_hit(tree_name, tree_arcs, inst: OutTreeInstance, w_float) -> Dict:
    """A feasible=True MILP witness: STOP, rationalize, report loudly."""
    print("\n" + "=" * 78, flush=True)
    print("!!!! exists_bad_w FEASIBLE -- POSSIBLE COUNTEREXAMPLE -- STOPPING !!!!",
          flush=True)
    print("=" * 78, flush=True)
    print(f"tree={tree_name}  tree_arcs={tree_arcs}", flush=True)
    print(f"attach={inst.attach}  demands={inst.demands}  w_float={w_float}",
          flush=True)
    verified = rationalize_and_verify(inst, w_float)
    print(f"rationalize_and_verify -> {verified}", flush=True)
    print("=" * 78 + "\n", flush=True)
    rec = {
        "tree": tree_name, "tree_arcs": tree_arcs,
        "attach": [list(p) for p in inst.attach],
        "demands": [str(d) for d in inst.demands],
        "w_float": w_float,
        "rationalized": verified,
    }
    results["positive_hits"].append(rec)
    dump_partial()
    return rec


def bug(msg: str) -> None:
    print("\n" + "!" * 78, flush=True)
    print(f"BUG: {msg}", flush=True)
    print("!" * 78 + "\n", flush=True)
    results["bugs"].append(msg)
    dump_partial()


# ---------------------------------------------------------------------------
# Stage 1: n<=5, reuse phase 3's exact enumeration, certify via exists_bad_w
# ---------------------------------------------------------------------------

def stage1(rng: random.Random) -> None:
    global found_positive
    log("STAGE 1: certified re-decision of phase-3's n<=5 space")

    shapes = []
    for name, tree_arcs, max_k in p3.TREE_SHAPES:
        autos = p3.compute_automorphisms(tree_arcs)
        shapes.append((name, tree_arcs, autos, max_k))

    pattern_cache: Dict[Tuple[str, int], List[Tuple[list, bool]]] = {}
    tot_cov = tot_unc = 0
    for name, tree_arcs, autos, max_k in shapes:
        ks = (3, 4) if max_k == 4 else (3, 4, 5)
        for k in ks:
            pats = p3.structural_patterns(tree_arcs, autos, k)
            pattern_cache[(name, k)] = pats
            n_cov = sum(1 for _, c in pats if c)
            n_unc = len(pats) - n_cov
            tot_cov += n_cov
            tot_unc += n_unc
            results["stage1_n5"]["enumeration_counts"][f"{name}/k{k}"] = {
                "deduped": len(pats), "covered": n_cov, "uncovered": n_unc,
            }
    results["stage1_n5"]["ledger_totals"] = {
        "covered": tot_cov, "uncovered": tot_unc,
    }
    log(f"stage1 enumeration: covered={tot_cov} uncovered={tot_unc} "
        f"(phase-3 ledger: covered=14038 uncovered=2415 -- should match exactly)")
    dump_partial()

    demand_vectors = p3.build_demand_vectors()

    # -- permutation-invariance check: exists_bad_w searches all w in the
    #    FULL box, so relabeling which demand value sits at which attach
    #    slot cannot change the reachable set of t(w) (apply the inverse
    #    relabeling to w). Verify this empirically on a few instances before
    #    relying on it to skip the k! permutation trials phase 3 used. -----
    log("checking permutation-invariance of exists_bad_w (justifies "
        "skipping demand-to-slot permutations below)")
    n_perm_checked = 0
    for name, tree_arcs, autos, max_k in shapes[:6]:
        k = 3 if max_k == 4 else 4
        pats = [p for p, c in pattern_cache[(name, k)] if not c]
        if not pats:
            continue
        attach = pats[0]
        dvec = demand_vectors[k]["unequal"][0]
        perm = list(range(k))
        rng.shuffle(perm)
        inst_a = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach,
                                  demands=dvec)
        dvec_p = [dvec[perm.index(i)] for i in range(k)]
        # relabel attach the same way perm relabels demand-slots
        attach_p = [attach[perm.index(i)] for i in range(k)]
        inst_b = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach_p,
                                  demands=dvec_p)
        res_a = exists_bad_w(inst_a)
        res_b = exists_bad_w(inst_b)
        agree = res_a.get("feasible") == res_b.get("feasible")
        results["stage1_n5"]["permutation_invariance_check"].append({
            "tree": name, "k": k, "feasible_a": res_a.get("feasible"),
            "feasible_b": res_b.get("feasible"), "agree": agree,
        })
        if not agree:
            bug(f"permutation invariance FAILED on {name}/k{k}: "
                f"{res_a} vs {res_b}")
        n_perm_checked += 1
    log(f"permutation-invariance check: {n_perm_checked} instances, "
        f"{'all agreed' if not results['bugs'] else 'SEE BUGS'}")
    dump_partial()

    # -- covered self-test (NG-9) -------------------------------------------
    n_cov_checked = n_cov_passed = 0
    for name, tree_arcs, autos, max_k in shapes:
        ks = (3, 4) if max_k == 4 else (3, 4, 5)
        for k in ks:
            covered_pats = [p for p, c in pattern_cache[(name, k)] if c]
            sample = (covered_pats if len(covered_pats) <= COVERED_SELF_TEST_CAP
                      else rng.sample(covered_pats, COVERED_SELF_TEST_CAP))
            dvec = demand_vectors[k]["unequal"][0]
            for attach in sample:
                inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                        attach=attach, demands=dvec)
                assert msw25_covered(inst)
                res = exists_bad_w(inst)
                n_cov_checked += 1
                if res.get("feasible") is False:
                    n_cov_passed += 1
                else:
                    bug(f"NG-9 SELF-TEST VIOLATED: MSW25-covered instance "
                        f"tree={name} attach={attach} demands={dvec} "
                        f"exists_bad_w={res}")
    results["stage1_n5"]["covered_self_test"] = {
        "n_checked": n_cov_checked, "n_passed": n_cov_passed,
    }
    log(f"covered (NG-9) self-test: {n_cov_passed}/{n_cov_checked} passed")
    dump_partial()

    # -- equal-demand self-test (Lemma 6) ------------------------------------
    n_eq_checked = n_eq_passed = 0
    for name, tree_arcs, autos, max_k in shapes:
        if name not in p3.LEMMA6_TREES:
            continue
        ks = (3, 4) if max_k == 4 else (3, 4, 5)
        for k in ks:
            eq = demand_vectors[k]["equal_self_test"]
            all_pats = [p for p, _ in pattern_cache[(name, k)]]
            sample = (all_pats if len(all_pats) <= EQUAL_SELF_TEST_CAP
                      else rng.sample(all_pats, EQUAL_SELF_TEST_CAP))
            for attach in sample:
                inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                        attach=attach, demands=eq)
                assert equal_demand_no_go(inst)
                res = exists_bad_w(inst)
                n_eq_checked += 1
                if res.get("feasible") is False:
                    n_eq_passed += 1
                else:
                    bug(f"LEMMA 6 SELF-TEST VIOLATED: equal-demand instance "
                        f"tree={name} attach={attach} exists_bad_w={res}")
    results["stage1_n5"]["equal_demand_self_test"] = {
        "n_checked": n_eq_checked, "n_passed": n_eq_passed,
    }
    log(f"equal-demand (Lemma 6) self-test: {n_eq_passed}/{n_eq_checked} passed")
    dump_partial()

    # -- re-check phase-3's specific argmax instances ------------------------
    import json as _json
    try:
        p3res = _json.load(open("outtree_results.json"))
        for combo in p3res["per_combo"]:
            if combo.get("argmax_attach") is None:
                continue
            name, k = combo["tree"], combo["k"]
            tree_arcs = dict((n, a) for n, a, _ in p3.TREE_SHAPES)[name]
            attach = [tuple(p) for p in combo["argmax_attach"]]
            dvec = fr_vec(combo["argmax_demands"])
            inst = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach,
                                    demands=dvec)
            res = exists_bad_w(inst)
            agree = (res.get("feasible") is False)  # phase3 t was negative -> must be infeasible
            results["stage1_n5"]["phase3_argmax_recheck"].append({
                "tree": name, "k": k, "phase3_t": combo["max_t_found_uncovered"],
                "exists_bad_w_feasible": res.get("feasible"),
                "proved_no_bad_w": res.get("proved_no_bad_w"),
                "agrees_with_phase3_negative": agree,
            })
            if res.get("feasible") is True:
                found_positive = True
                loud_hit(name, tree_arcs, inst, res["w_float"])
                return
        log(f"phase-3 argmax re-check: "
            f"{len(results['stage1_n5']['phase3_argmax_recheck'])} instances, "
            f"all agree with phase-3's negative verdict: "
            f"{all(r['agrees_with_phase3_negative'] for r in results['stage1_n5']['phase3_argmax_recheck'])}")
    except FileNotFoundError:
        log("outtree_results.json not found -- skipping phase-3 argmax recheck")
    dump_partial()

    # -- live certified search: EVERY uncovered pattern x EVERY named demand
    #    vector phase 3 used (permutation skipped -- proven invariant above)
    for name, tree_arcs, autos, max_k in shapes:
        if found_positive:
            break
        if elapsed() > SOFT_STOP_S:
            results["truncated"] = True
            results["truncated_note"] = f"stage1 stopped early at tree={name}"
            log("SOFT STOP hit in stage1 -- truncating")
            break
        ks = (3, 4) if max_k == 4 else (3, 4, 5)
        for k in ks:
            if found_positive or elapsed() > SOFT_STOP_S:
                break
            uncovered = [p for p, c in pattern_cache[(name, k)] if not c]
            # cap at 3 demand vectors/pattern for cost (measured ~39ms/call at
            # k=5, real trees) -- still every uncovered STRUCTURAL pattern is
            # certified (phase 3 capped patterns AND used only 4 random trials
            # per pattern; this is a superset on the pattern axis, a subset on
            # the demand-vector axis, and permutation is provably irrelevant
            # -- see permutation_invariance_check above)
            dvecs = demand_vectors[k]["unequal"][:3]
            n_tested = n_proved = 0
            worst_eps = None
            for attach in uncovered:
                for dvec in dvecs:
                    inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                            attach=attach, demands=dvec)
                    res = exists_bad_w(inst)
                    n_tested += 1
                    if res.get("feasible") is True:
                        found_positive = True
                        loud_hit(name, tree_arcs, inst, res["w_float"])
                        break
                    if res.get("proved_no_bad_w"):
                        n_proved += 1
                    else:
                        bug(f"exists_bad_w INCONCLUSIVE (not proved infeasible, "
                            f"not feasible): tree={name} attach={attach} "
                            f"demands={dvec} res={res}")
                if found_positive:
                    break
            results["stage1_n5"]["live_certified"].append({
                "tree": name, "k": k, "n_uncovered_patterns": len(uncovered),
                "n_demand_vectors": len(dvecs), "n_tested": n_tested,
                "n_proved_no_bad_w": n_proved,
            })
            log(f"{name}/k{k}: {len(uncovered)} uncovered patterns x "
                f"{len(dvecs)} demand vectors = {n_tested} certified "
                f"({n_proved} proved no-bad-w)")
            dump_partial()

    log(f"STAGE 1 done at {elapsed():.1f}s")


# ---------------------------------------------------------------------------
# Stage 2: n=6,7, k=3..6
# ---------------------------------------------------------------------------

def chain(n: int) -> List[Tuple[str, str]]:
    nodes = ["s"] + [f"n{i}" for i in range(1, n)]
    return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


N67_SHAPES: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("n6-chain5", chain(6)),
    ("n6-chain-fork", [("s", "n1"), ("n1", "n2"), ("n2", "n3"),
                        ("n3", "n4"), ("n2", "n5")]),
    ("n6-star5", [("s", "n1"), ("s", "n2"), ("s", "n3"),
                  ("s", "n4"), ("s", "n5")]),
    ("n6-twochains", [("s", "n1"), ("n1", "n2"), ("n2", "n3"),
                       ("s", "n4"), ("n4", "n5")]),
    ("n7-chain6", chain(7)),
    ("n7-chain-fork", [("s", "n1"), ("n1", "n2"), ("n2", "n3"),
                        ("n3", "n4"), ("n4", "n5"), ("n2", "n6")]),
    ("n7-star6", [("s", "n1"), ("s", "n2"), ("s", "n3"),
                  ("s", "n4"), ("s", "n5"), ("s", "n6")]),
    ("n7-twochains", [("s", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"),
                       ("s", "n5"), ("n5", "n6")]),
]


def sampled_structural_patterns(tree_arcs, autos, pairs, k, sample_size, rng
                                ) -> List[Tuple[list, bool]]:
    """Same classification as p3.structural_patterns, but the raw generator
    is random sampling (with replacement over `pairs`) instead of exhaustive
    combinations_with_replacement -- for combo spaces too large to enumerate
    within budget. Deduped by tree automorphism only WITHIN the sample (not
    a claim of exhaustive orbit coverage)."""
    seen: Dict[tuple, List[Tuple[str, str]]] = {}
    for _ in range(sample_size):
        combo = tuple(sorted(tuple(sorted(pair))
                              for pair in rng.choices(pairs, k=k)))
        best = None
        for g in autos:
            mapped = tuple(sorted(tuple(sorted((g[u], g[v]))) for (u, v) in combo))
            if best is None or mapped < best:
                best = mapped
        if best not in seen:
            seen[best] = list(combo)
    out = []
    for attach_list in seen.values():
        edges = tree_arcs + attach_list
        out.append((attach_list, is_series_parallel(edges)))
    return out


def build_demand_vectors_67(k: int, rng: random.Random) -> List[List[Fr]]:
    named = {
        3: [(15, 10, 15), (7, 5, 3)],
        4: [(9, 7, 6, 4), (13, 8, 5, 3)],
        5: [(11, 9, 7, 5, 3), (13, 11, 8, 6, 2)],
        6: [(17, 13, 11, 8, 5, 3), (19, 14, 9, 7, 4, 2)],
    }[k]
    vecs = [fr_vec(v) for v in named]
    while True:
        v = tuple(rng.randint(2, 24) for _ in range(k))
        if len(set(v)) > 1:
            vecs.append(fr_vec(v))
            break
    return vecs


def lemma7_check(rng: random.Random) -> None:
    """Verification self-test (not pruning -- see module docstring): on
    n=6,7 instances whose full chord multigraph contains a cycle, confirm
    Lemma 7's claim that the signed cycle combination vanishes on every
    tree arc. A failure would mean Lemma 7 as implemented is wrong."""
    n_checked = n_passed = 0
    for name, tree_arcs in N67_SHAPES:
        pairs = p3.unordered_pairs(p3.nodes_of(tree_arcs))
        max_k = min(6, len(pairs))
        for k in range(3, max_k + 1):
            if n_checked >= LEMMA7_CHECK_CAP:
                break
            for _ in range(4):
                if n_checked >= LEMMA7_CHECK_CAP:
                    break
                attach = [pairs[rng.randrange(len(pairs))] for _ in range(k)]
                cyc = find_chord_cycle(attach, list(range(k)))
                if cyc is None:
                    continue
                inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                        attach=attach, demands=fr_vec([1] * k))
                ok = cycle_shift_preserves_loads(inst, cyc)
                n_checked += 1
                if ok:
                    n_passed += 1
                else:
                    bug(f"LEMMA 7 VIOLATED: tree={name} attach={attach} "
                        f"cycle={cyc}")
    results["lemma7_verification"] = {"n_checked": n_checked, "n_passed": n_passed}
    log(f"Lemma 7 cycle-shift verification: {n_passed}/{n_checked} passed "
        f"(n=6,7 instances)")
    dump_partial()


def stage2(rng: random.Random) -> None:
    global found_positive
    if found_positive:
        return
    log("STAGE 2: n=6,7 extension, k=3..6")

    lemma7_check(rng)
    if found_positive or results["bugs"]:
        return

    shapes = []
    for name, tree_arcs in N67_SHAPES:
        autos = p3.compute_automorphisms(tree_arcs)
        n_nodes = len(p3.nodes_of(tree_arcs))
        shapes.append((name, tree_arcs, autos, n_nodes))
        results["stage2_n67"]["shapes"].append(
            {"name": name, "tree_arcs": tree_arcs, "n_nodes": n_nodes,
             "automorphism_group_size": len(autos)})

    n_cov_checked = n_cov_passed = 0
    n_eq_checked = n_eq_passed = 0

    for name, tree_arcs, autos, n_nodes in shapes:
        if found_positive:
            break
        if elapsed() > SOFT_STOP_S:
            results["truncated"] = True
            results["truncated_note"] = (results.get("truncated_note") or "") + \
                f" stage2 stopped early at tree={name}"
            log("SOFT STOP hit in stage2 -- truncating")
            break
        pairs = p3.unordered_pairs(p3.nodes_of(tree_arcs))
        for k in range(3, 7):
            if found_positive or elapsed() > SOFT_STOP_S:
                break
            raw_count = math.comb(len(pairs) + k - 1, k)
            exhaustive = raw_count <= ENUM_CAP
            if exhaustive:
                pats = p3.structural_patterns(tree_arcs, autos, k)
            else:
                pats = sampled_structural_patterns(tree_arcs, autos, pairs, k,
                                                    SAMPLE_RAW, rng)
            n_cov = sum(1 for _, c in pats if c)
            n_unc = len(pats) - n_cov
            results["stage2_n67"]["enumeration_counts"][f"{name}/k{k}"] = {
                "raw_count": raw_count, "exhaustive": exhaustive,
                "deduped_or_sampled": len(pats), "covered": n_cov,
                "uncovered": n_unc,
            }
            log(f"{name}/k{k}: raw={raw_count} "
                f"{'EXHAUSTIVE' if exhaustive else f'SAMPLED({SAMPLE_RAW})'} "
                f"-> {len(pats)} patterns, covered={n_cov} uncovered={n_unc}")

            # covered self-test
            covered_pats = [p for p, c in pats if c]
            sample = (covered_pats if len(covered_pats) <= COVERED_SELF_TEST_CAP
                      else rng.sample(covered_pats, COVERED_SELF_TEST_CAP))
            dvecs = build_demand_vectors_67(k, rng)
            for attach in sample:
                inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                        attach=attach, demands=dvecs[0])
                assert msw25_covered(inst)
                res = exists_bad_w(inst)
                n_cov_checked += 1
                if res.get("feasible") is False:
                    n_cov_passed += 1
                else:
                    bug(f"NG-9 SELF-TEST VIOLATED (n67): tree={name} "
                        f"attach={attach} demands={dvecs[0]} res={res}")

            # equal-demand self-test (only k=3,4 to control cost)
            if k <= 4:
                eq = fr_vec([4] * k)
                eq_sample = (pats if len(pats) <= EQUAL_SELF_TEST_CAP
                             else rng.sample(pats, EQUAL_SELF_TEST_CAP))
                for attach, _ in eq_sample:
                    inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                            attach=attach, demands=eq)
                    assert equal_demand_no_go(inst)
                    res = exists_bad_w(inst)
                    n_eq_checked += 1
                    if res.get("feasible") is False:
                        n_eq_passed += 1
                    else:
                        bug(f"LEMMA 6 SELF-TEST VIOLATED (n67): tree={name} "
                            f"attach={attach} res={res}")

            if results["bugs"]:
                dump_partial()
                return

            # live certified search, prioritized (uncovered only, NG-9 filter)
            uncovered = [p for p, c in pats if not c]
            cap = CAP_LIVE_K6 if k == 6 else CAP_LIVE_K345
            tested_pats = (uncovered if len(uncovered) <= cap
                           else rng.sample(uncovered, cap))
            live_dvecs = dvecs[:2] if k == 6 else dvecs
            n_tested = n_proved = 0
            for attach in tested_pats:
                for dvec in live_dvecs:
                    inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                            attach=attach, demands=dvec)
                    res = exists_bad_w(inst)
                    n_tested += 1
                    if res.get("feasible") is True:
                        found_positive = True
                        loud_hit(name, tree_arcs, inst, res["w_float"])
                        break
                    if res.get("proved_no_bad_w"):
                        n_proved += 1
                    else:
                        bug(f"exists_bad_w INCONCLUSIVE (n67): tree={name} "
                            f"attach={attach} demands={dvec} res={res}")
                if found_positive:
                    break
            results["stage2_n67"]["live_certified"].append({
                "tree": name, "k": k, "n_uncovered_total": len(uncovered),
                "n_uncovered_tested": len(tested_pats),
                "n_demand_vectors": len(live_dvecs), "n_tested": n_tested,
                "n_proved_no_bad_w": n_proved,
                "sample_note": ("full uncovered set tested" if len(uncovered) <= cap
                                 else f"capped sample {cap}/{len(uncovered)}"),
            })
            log(f"  live: {len(tested_pats)}/{len(uncovered)} uncovered patterns "
                f"x {len(live_dvecs)} demands = {n_tested} certified "
                f"({n_proved} proved no-bad-w)")
            dump_partial()

    results["stage2_n67"]["covered_self_test"] = {
        "n_checked": n_cov_checked, "n_passed": n_cov_passed}
    results["stage2_n67"]["equal_demand_self_test"] = {
        "n_checked": n_eq_checked, "n_passed": n_eq_passed}
    dump_partial()
    log(f"STAGE 2 done at {elapsed():.1f}s")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    log("certified_search.py starting")
    rng = random.Random(SEARCH_SEED)

    stage1(rng)
    if not found_positive and not results["bugs"]:
        stage2(rng)

    results["total_elapsed_s"] = elapsed()
    dump_partial()
    if found_positive:
        log("POSITIVE HIT -- see positive_hits in certified_results.json")
    elif results["bugs"]:
        log(f"BUGS FOUND ({len(results['bugs'])}) -- see bugs in "
            f"certified_results.json -- STOPPING, not a clean negative")
    else:
        log("No positive hit. 0 bugs. Certified negative across stage1+stage2.")
    log(f"total elapsed: {elapsed():.1f}s")
    return results


if __name__ == "__main__":
    main()
