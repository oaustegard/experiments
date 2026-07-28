"""
outtree_search.py -- phase-3 computational search over the out-tree normal
form (outtree.py, NOGOS.md Lemma 4) for Conjecture 1.3 counterexamples in
path-closed 2-path instances, NOW FILTERED by the MSW25 literature gate.

WHAT THIS SEARCHES.  Path-closed instances with exactly 2 s->t_i paths per
terminal are exactly out-trees: an arborescence rooted at s with terminals
as in-degree-2 sinks attached from two tree nodes (u_i, v_i) (Lemma 4).

MSW25 PRUNE (session 3 literature gate, folded in mid-run). Majthoub
Almoghrabi, Skutella, Warode (arXiv:2412.05182, Thm 2) PROVE Conjecture 1.3
with the tight two-sided d_max bound for multiflows on series-parallel
digraphs. Suppressing an out-tree instance's terminal sinks (undirected
degree 2 each) turns it into "tree + one chord {u_i,v_i} per terminal". If
THAT graph is series-parallel (K4-minor-free), MSW25 already proves no
counterexample can exist there -- `outtree.msw25_covered` decides this via
the classical SP reduction (`outtree.is_series_parallel`).

STRUCTURAL CONSEQUENCE (derived here, drives the whole redesign): coverage
is a property of the ATTACH STRUCTURE alone -- demand values never enter
`chord_graph_edges`. And a K4 minor needs >=4 branch vertices, so a tree
with <4 total nodes (root + <3 more) can NEVER be uncovered, regardless of
how many parallel chords land on its 1-2 available pairs. Concretely: n=2
and n=3 trees (3 of our 7 original shapes) are ALWAYS 100% MSW25-covered --
dead on arrival for counterexample-hunting, though still useful as cheap
covered-self-test fodder. n=4 trees (the other 4 shapes) turn out to have a
TINY uncovered fraction (measured: 1-6 uncovered structural patterns out of
14-126 deduped ones per tree/k -- essentially "the Rybin K4 completion" and
its k=4 near-neighbors). This frees enormous budget, which is redirected
(per coordinator instruction) into NEW n=5 tree shapes (root + 4 more nodes,
9 shapes by the a(5)=9 rooted-tree count) at k=3,4,5, where the uncovered
fraction is much richer (measured up to 378/2002 at k=5 on chain-like
trees) -- genuine "more tree nodes, crossing chords" live territory.

SEARCH STRATEGY (post-redesign).
  1. Enumerate ALL structural attach patterns (demand-agnostic multisets of
     k node-pairs, deduped by tree automorphism) for every tree shape and
     every applicable k. Classify each as covered/uncovered via
     `is_series_parallel` -- cheap (no t(w) call), reported as a ledger.
  2. Equal-demand Lemma 6 self-test (unchanged from the original design,
     lightly sampled): a bug detector, not a discovery target.
  3. MSW25-covered self-test: a light sample of covered patterns run
     through the SAME search + `msw25_covered` re-check; t(w) > 0 there is
     a bug (proved impossible by MSW25), asserted loudly.
  4. Live search: EVERY uncovered pattern that fits the per-combo cap is a
     candidate; each gets several random (demand-vector, slot-permutation)
     trials through the full w-search (corners + balanced heuristic +
     random rational grid + coordinate polish, all exact Fractions). Any
     t(w) > 0 STOPS the run immediately and cross-checks through
     `to_core_instance()` + `core.py` -- reported loudly, never called a
     verified counterexample here (that's the orchestrator's certificate
     gate).

SANITY ANCHOR. The Rybin instance (tree_arcs = s-a-b-c chain, attach =
{t1:(a,c), t2:(s,c), t3:(s,b)}, demands (15,10,15)) has chord graph = K4 on
{s,a,b,c} exactly -- `msw25_covered` must return False. Asserted at startup;
if it ever returns True the filter is wired backwards and nothing downstream
can be trusted.

COMPUTE BUDGET (measured, not guessed). t_of_w pilot cost in this
container: ~250-260us at k=3, ~630us at k=4, and (extrapolating the same
~2x/level trend, spot-checked) roughly ~1ms at k=5. Because the MSW25 filter
shrinks the live set so much, the dominant cost is now the *live* search,
not enumeration (enumeration of every structural pattern up to n=5,k=5 --
2002 raw combos per tree in the worst case -- timed at <0.1s per tree/k,
negligible). With CAP_PATTERNS_PER_COMBO=20 uncovered patterns sampled per
(tree,k) and TRIALS_PER_PATTERN=6 random (demand,permutation) trials each:
    m=4 combos (8 total): all uncovered counts already <=6, no capping,
        22 patterns x6 trials = 132 instance-searches
    m=5 combos (27 total): capped sum ~384 patterns x6 trials
        = 2304 instance-searches
    total ~2436 instance-searches x ~150-230 evals x ~0.3-1ms/eval
        ~= 290s (~4.8min) of the 720s (12min) ceiling -- large margin left
        for the covered self-tests, enumeration passes, and I/O.
A wall-clock guard (SOFT_STOP_S) truncates any remaining combos if this
projection runs long, logging what was skipped rather than silently
overrunning.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from collections import Counter
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Sequence, Tuple

from core import check_path_closure, check_paths_valid, verify_counterexample
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

TIME_BUDGET_S = 12 * 60
SOFT_STOP_S = 10 * 60
CAP_PATTERNS_PER_COMBO = 15      # uncovered structural patterns explored / (tree,k)
TRIALS_PER_PATTERN = 4           # random (demand-vector, permutation) trials / pattern
COVERED_SELF_TEST_CAP = 5        # covered structural patterns self-tested / (tree,k)
RANDOM_SAMPLES = 12              # random rational w points / instance search
POLISH_DENOM = 12
POLISH_PASSES = 2
SEARCH_SEED = 1729
LEMMA6_TREES = {"n2-chain", "n3-cherry", "n4-star3", "n5-star4",
                 "n4-chain3", "n5-chain4"}  # representative subset for the
                                             # equal-demand self-test (Lemma 6
                                             # is a general structural claim,
                                             # not tied to a specific tree's
                                             # symmetry -- full-tree coverage
                                             # isn't needed for a bug-catch)

START = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - START


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Tree shapes. n=2..4 (7 shapes, original) + n=5 (9 shapes, MSW25 redirect).
#    Verified against OEIS A000081 (unlabeled rooted trees): a(2)=1, a(3)=2,
#    a(4)=4, a(5)=9, all cross-checked by hand via the subtree-size-partition
#    argument (see campaign transcript). Hardcoded because the space is tiny
#    and this avoids algorithmic risk from a general free-standing generator.
# ---------------------------------------------------------------------------

TREE_SHAPES: List[Tuple[str, List[Tuple[str, str]], int]] = [
    # (name, tree_arcs, max_k) -- max_k caps which k values are swept for it
    ("n2-chain",        [("s", "n1")], 4),
    ("n3-chain",        [("s", "n1"), ("n1", "n2")], 4),
    ("n3-cherry",       [("s", "n1"), ("s", "n2")], 4),
    ("n4-chain3",       [("s", "n1"), ("n1", "n2"), ("n2", "n3")], 4),
    ("n4-fork-at-n1",   [("s", "n1"), ("n1", "n2"), ("n1", "n3")], 4),
    ("n4-chain2+leaf",  [("s", "n1"), ("n1", "n2"), ("s", "n3")], 4),
    ("n4-star3",        [("s", "n1"), ("s", "n2"), ("s", "n3")], 4),
    # n=5 shapes (redirect: MSW25 leaves this the richest live territory)
    ("n5-chain4",       [("s", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4")], 5),
    ("n5-chain-fork",   [("s", "n1"), ("n1", "n2"), ("n2", "n3"), ("n2", "n4")], 5),
    ("n5-chain-leaf",   [("s", "n1"), ("n1", "n2"), ("n2", "n3"), ("n1", "n4")], 5),
    ("n5-chain-star",   [("s", "n1"), ("n1", "n2"), ("n1", "n3"), ("n1", "n4")], 5),
    ("n5-chain3-leaf",  [("s", "n1"), ("n1", "n2"), ("n2", "n3"), ("s", "n4")], 5),
    ("n5-cherry-leaf",  [("s", "n1"), ("n1", "n2"), ("n1", "n3"), ("s", "n4")], 5),
    ("n5-twochains",    [("s", "n1"), ("n1", "n2"), ("s", "n3"), ("n3", "n4")], 5),
    ("n5-chain2-2leaf", [("s", "n1"), ("n1", "n2"), ("s", "n3"), ("s", "n4")], 5),
    ("n5-star4",        [("s", "n1"), ("s", "n2"), ("s", "n3"), ("s", "n4")], 5),
]


def nodes_of(tree_arcs: List[Tuple[str, str]]) -> List[str]:
    ns = {"s"}
    for u, v in tree_arcs:
        ns.add(u)
        ns.add(v)
    return sorted(ns)


def compute_automorphisms(tree_arcs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Brute-force: permutations of non-root nodes fixing root that preserve
    the parent-child edge set. <=4 non-root nodes here (n=5 shapes), so
    <=24 permutations to test -- exhaustive and cheap."""
    nodes = nodes_of(tree_arcs)
    non_root = [n for n in nodes if n != "s"]
    edge_set = set(tree_arcs)
    autos = []
    for perm in itertools.permutations(non_root):
        m = dict(zip(non_root, perm))
        m["s"] = "s"
        mapped = {(m[u], m[v]) for (u, v) in tree_arcs}
        if mapped == edge_set:
            autos.append(m)
    return autos


def unordered_pairs(nodes: List[str]) -> List[Tuple[str, str]]:
    return [tuple(sorted((u, v))) for u, v in itertools.combinations(nodes, 2)]


# ---------------------------------------------------------------------------
# 2. Structural (demand-agnostic) attach-pattern enumeration, deduped by
#    tree automorphism, classified covered/uncovered via MSW25.
# ---------------------------------------------------------------------------

def structural_patterns(tree_arcs, autos, k) -> List[Tuple[List[Tuple[str, str]], bool]]:
    """[(attach_list, covered)] -- one representative per automorphism orbit
    of k-terminal attach multisets, classified by is_series_parallel on the
    resulting chord graph (demand-agnostic: coverage never depends on
    which demand values later get assigned to these slots)."""
    pairs = unordered_pairs(nodes_of(tree_arcs))
    seen: Dict[tuple, List[Tuple[str, str]]] = {}
    for combo in itertools.combinations_with_replacement(pairs, k):
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


# ---------------------------------------------------------------------------
# 3. w-search: corners, balanced heuristic, random grid, coordinate polish.
#    Exact Fraction arithmetic throughout (floats never touch a decision).
# ---------------------------------------------------------------------------

def balanced_candidates(inst: OutTreeInstance, C, k) -> List[Tuple[Fr, ...]]:
    """For each tree arc and each pair of terminals co-resident on it
    (nonzero coefficient), the point balancing their demands on that arc:
    w_i = d_j/(d_i+d_j), w_j = d_i/(d_i+d_j), others at 1/2."""
    d = inst.demands
    half = Fr(1, 2)
    cands = []
    for row in C:
        idxs = [i for i, c in enumerate(row) if c]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                total = d[i] + d[j]
                w = [half] * k
                if total > 0:
                    w[i] = d[j] / total
                    w[j] = d[i] / total
                cands.append(tuple(w))
    return cands


def search_max_t(inst: OutTreeInstance, rng: random.Random
                 ) -> Tuple[Fr, Tuple[Fr, ...], int]:
    """Returns (best_t, best_w, n_evals). Short-circuits the moment a
    positive t(w) is found -- the caller must STOP that branch."""
    k = len(inst.demands)
    C = inst.coefficient_matrix()
    best_t: Optional[Fr] = None
    best_w: Optional[Tuple[Fr, ...]] = None
    n_evals = 0

    def consider(w) -> Fr:
        nonlocal best_t, best_w, n_evals
        t = inst.t_of_w(w, C)
        n_evals += 1
        if best_t is None or t > best_t:
            best_t, best_w = t, tuple(w)
        return t

    def positive() -> bool:
        return best_t is not None and best_t > 0

    for w in itertools.product((Fr(0), Fr(1, 2), Fr(1)), repeat=k):
        consider(w)
        if positive():
            return best_t, best_w, n_evals

    for w in balanced_candidates(inst, C, k):
        consider(w)
        if positive():
            return best_t, best_w, n_evals

    for _ in range(RANDOM_SAMPLES):
        denom = rng.choice((4, 6, 8, 12))
        w = tuple(Fr(rng.randint(0, denom), denom) for _ in range(k))
        consider(w)
        if positive():
            return best_t, best_w, n_evals

    polish_vals = [Fr(i, POLISH_DENOM) for i in range(POLISH_DENOM + 1)]
    cur = list(best_w)
    for _ in range(POLISH_PASSES):
        improved = False
        for i in range(k):
            local_best_val = cur[i]
            local_best_t = best_t
            for val in polish_vals:
                trial = list(cur)
                trial[i] = val
                t = consider(trial)
                if positive():
                    return best_t, best_w, n_evals
                if t > local_best_t:
                    local_best_t, local_best_val = t, val
            if local_best_val != cur[i]:
                cur[i] = local_best_val
                improved = True
        if not improved:
            break

    return best_t, best_w, n_evals


# ---------------------------------------------------------------------------
# 4. Demand vectors
# ---------------------------------------------------------------------------

def fr_vec(vals) -> List[Fr]:
    return [Fr(v) for v in vals]


def build_demand_vectors():
    rng = random.Random(SEARCH_SEED)

    def rand_unequal(length, lo=2, hi=20):
        while True:
            v = tuple(rng.randint(lo, hi) for _ in range(length))
            if len(set(v)) > 1:
                return v

    k3_named = [(15, 10, 15), (7, 5, 3), (5, 5, 6), (3, 2, 2)]
    k3_random = [rand_unequal(3) for _ in range(3)]
    k3_unequal = [fr_vec(v) for v in (k3_named + k3_random)]
    k3_equal = fr_vec((4, 4, 4))

    k4_named = [(9, 7, 6, 4), (11, 9, 6, 2), (13, 8, 5, 3)]
    k4_random = [rand_unequal(4) for _ in range(2)]
    k4_unequal = [fr_vec(v) for v in (k4_named + k4_random)]
    k4_equal = fr_vec((4, 4, 4, 4))

    k5_named = [(11, 9, 7, 5, 3), (13, 11, 8, 6, 2), (9, 7, 6, 5, 4)]
    k5_random = [rand_unequal(5) for _ in range(2)]
    k5_unequal = [fr_vec(v) for v in (k5_named + k5_random)]
    k5_equal = fr_vec((4, 4, 4, 4, 4))

    return {
        3: {"unequal": k3_unequal, "equal_self_test": k3_equal},
        4: {"unequal": k4_unequal, "equal_self_test": k4_equal},
        5: {"unequal": k5_unequal, "equal_self_test": k5_equal},
    }


# ---------------------------------------------------------------------------
# 5. Cross-check path for any positive hit
# ---------------------------------------------------------------------------

def loud_report(tree_name, tree_arcs, inst: OutTreeInstance, best_t, best_w):
    print("\n" + "=" * 78, flush=True)
    print("!!!! t(w) > 0 FOUND -- POSSIBLE COUNTEREXAMPLE BRANCH -- STOPPING !!!!",
          flush=True)
    print("=" * 78, flush=True)
    print(f"tree shape: {tree_name}  tree_arcs={tree_arcs}", flush=True)
    print(f"attach={inst.attach}  demands={inst.demands}", flush=True)
    print(f"w = {best_w}", flush=True)
    print(f"t_of_w(w) = {best_t}  (d_max = {max(inst.demands)})", flush=True)
    print(f"msw25_covered(inst) = {msw25_covered(inst)}  (must be False here)", flush=True)

    core_inst = inst.to_core_instance()
    split = inst.split_from_w(best_w)
    problems_valid = check_paths_valid(core_inst, "s")
    problems_closed = check_path_closure(core_inst, "s")
    verdict = verify_counterexample(core_inst, split)

    print(f"core.check_paths_valid: {problems_valid or 'CLEAN'}", flush=True)
    print(f"core.check_path_closure: {problems_closed or 'CLEAN'}", flush=True)
    print(f"core.verify_counterexample: is_counterexample="
          f"{verdict['is_counterexample']}  t_of_x={verdict['t_of_x']}", flush=True)
    print("=" * 78 + "\n", flush=True)

    return {
        "tree_name": tree_name, "tree_arcs": tree_arcs,
        "attach": [list(p) for p in inst.attach],
        "demands": [str(d) for d in inst.demands],
        "w": [str(x) for x in best_w], "t_of_w": str(best_t),
        "msw25_covered": msw25_covered(inst),
        "check_paths_valid_problems": problems_valid,
        "check_path_closure_problems": problems_closed,
        "core_is_counterexample": verdict["is_counterexample"],
        "core_t_of_x": str(verdict["t_of_x"]),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    log("outtree_search.py starting (MSW25-filtered redesign)")
    projection = (
        "COST PROJECTION (revised after a full 41-combo run measured "
        "26/41 combos costing 615s at CAP=20/TRIALS=6 -- truncated by the "
        "soft-stop guard, 0 assertion failures, 0 positive hits): knobs "
        "cut to CAP_PATTERNS_PER_COMBO=15, TRIALS_PER_PATTERN=4, "
        "RANDOM_SAMPLES=12, COVERED_SELF_TEST_CAP=5, Lemma-6 self-test "
        "restricted to 6 representative trees -- targeting full 41-combo "
        "coverage inside the 720s (12min) ceiling this run."
    )
    log(projection)

    results = {
        "budget_projection_note": projection,
        "tree_shapes": [],
        "sanity_anchor": {},
        "enumeration_counts": {},   # "{tree}/k{k}" -> {raw, deduped, covered, uncovered}
        "equal_demand_self_test": [],
        "msw25_covered_self_test": [],
        "per_combo": [],
        "positive_hits": [],
        "truncated": False,
        "truncated_note": None,
    }

    def dump_partial():
        with open("outtree_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # -- sanity anchor: Rybin's chord graph is K4 --> msw25_covered False --
    rybin = OutTreeInstance(
        root="s", tree_arcs=[("s", "a"), ("a", "b"), ("b", "c")],
        attach=[("a", "c"), ("s", "c"), ("s", "b")],
        demands=fr_vec((15, 10, 15)),
    )
    rybin_covered = msw25_covered(rybin)
    assert rybin_covered is False, (
        "SANITY ANCHOR FAILED: msw25_covered(rybin) should be False "
        "(chord graph is exactly K4) -- the filter is wired backwards"
    )
    log(f"sanity anchor OK: chord_graph_edges(rybin)={chord_graph_edges(rybin)}, "
        f"msw25_covered={rybin_covered} (expected False)")
    results["sanity_anchor"] = {
        "instance": "rybin (s-a-b-c chain, attach={t1:(a,c),t2:(s,c),t3:(s,b)}, d=(15,10,15))",
        "chord_graph_edges": chord_graph_edges(rybin),
        "msw25_covered": rybin_covered,
        "expected": False,
        "passed": rybin_covered is False,
    }

    # -- tree shapes + automorphisms ----------------------------------------
    shapes = []
    for name, tree_arcs, max_k in TREE_SHAPES:
        autos = compute_automorphisms(tree_arcs)
        shapes.append((name, tree_arcs, autos, max_k))
        n_nodes = len(nodes_of(tree_arcs))
        results["tree_shapes"].append({
            "name": name, "tree_arcs": tree_arcs, "n_nodes": n_nodes,
            "automorphism_group_size": len(autos),
        })
        log(f"tree {name}: n_nodes={n_nodes} |Aut|={len(autos)}")
    assert len(shapes) == 16, "expected 7 (n=2..4) + 9 (n=5) = 16 tree shapes"

    # -- structural enumeration + covered/uncovered ledger -------------------
    pattern_cache: Dict[Tuple[str, int], List[Tuple[list, bool]]] = {}
    tot_raw = tot_ded = tot_cov = tot_unc = 0
    for name, tree_arcs, autos, max_k in shapes:
        ks = (3, 4) if max_k == 4 else (3, 4, 5)
        for k in ks:
            pats = structural_patterns(tree_arcs, autos, k)
            pattern_cache[(name, k)] = pats
            n_ded = len(pats)
            n_cov = sum(1 for _, c in pats if c)
            n_unc = n_ded - n_cov
            raw = len(list(itertools.combinations_with_replacement(
                unordered_pairs(nodes_of(tree_arcs)), k)))
            tot_raw += raw; tot_ded += n_ded; tot_cov += n_cov; tot_unc += n_unc
            results["enumeration_counts"][f"{name}/k{k}"] = {
                "raw": raw, "deduped": n_ded, "covered": n_cov, "uncovered": n_unc,
            }
    log(f"structural enumeration done: raw={tot_raw} deduped={tot_ded} "
        f"covered={tot_cov} ({100*tot_cov/tot_ded:.1f}%) uncovered={tot_unc} "
        f"({100*tot_unc/tot_ded:.1f}%)")
    results["ledger_totals"] = {
        "raw": tot_raw, "deduped": tot_ded, "covered": tot_cov, "uncovered": tot_unc,
        "covered_fraction": tot_cov / tot_ded, "uncovered_fraction": tot_unc / tot_ded,
    }
    dump_partial()

    demand_vectors = build_demand_vectors()
    for k in (3, 4, 5):
        log(f"k={k} unequal demand vectors: {demand_vectors[k]['unequal']}")
        log(f"k={k} equal-demand self-test vector: {demand_vectors[k]['equal_self_test']}")

    rng = random.Random(SEARCH_SEED + 1)
    found_positive = False

    # -- equal-demand self test (Lemma 6) -- lightly sampled, representative
    #    subset of trees only (Lemma 6 is a general structural claim, not
    #    tied to a specific tree's automorphisms; this is a bug-catch, not
    #    an exhaustiveness requirement -- see LEMMA6_TREES above) ----------
    LEMMA6_CAP = 8
    for name, tree_arcs, autos, max_k in shapes:
        if name not in LEMMA6_TREES:
            continue
        ks = (3, 4) if max_k == 4 else (3, 4, 5)
        for k in ks:
            eq = demand_vectors[k]["equal_self_test"]
            all_patterns = [p for p, _ in pattern_cache[(name, k)]]
            sample = (all_patterns if len(all_patterns) <= LEMMA6_CAP
                      else rng.sample(all_patterns, LEMMA6_CAP))
            worst = None
            for attach_list in sample:
                inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                        attach=attach_list, demands=eq)
                assert equal_demand_no_go(inst), \
                    "equal_demand_no_go() False on all-equal demands -- BUG"
                best_t, best_w, _ = search_max_t(inst, rng)
                assert best_t <= 0, (
                    f"LEMMA 6 VIOLATED (bug, not a discovery): tree={name} "
                    f"attach={attach_list} demands={eq} w={best_w} t={best_t}"
                )
                if worst is None or best_t > worst:
                    worst = best_t
            results["equal_demand_self_test"].append({
                "tree": name, "k": k, "n_checked": len(sample),
                "max_t_found": str(worst) if worst is not None else None,
            })
    log("equal-demand self-test (Lemma 6) passed on every sampled tree/k -- no bug detected")
    dump_partial()

    # -- MSW25-covered self-test + live uncovered search ---------------------
    global_best = None  # (t, tree_name, attach, demands, w)
    per_tree_best: Dict[str, Optional[Fr]] = {name: None for name, _, _, _ in shapes}
    n_covered_self_tested = 0
    n_uncovered_searched = 0

    combos = []
    for name, tree_arcs, autos, max_k in shapes:
        ks = (3, 4) if max_k == 4 else (3, 4, 5)
        for k in ks:
            combos.append((name, tree_arcs, k))
    log(f"total (tree, k) combos to sweep: {len(combos)}")

    for idx, (name, tree_arcs, k) in enumerate(combos):
        if found_positive:
            break
        if elapsed() > SOFT_STOP_S:
            results["truncated"] = True
            skipped = len(combos) - idx
            results["truncated_note"] = (
                f"Stopped after {idx}/{len(combos)} combos at {elapsed():.1f}s "
                f"(soft stop {SOFT_STOP_S}s); {skipped} combos NOT swept."
            )
            log(results["truncated_note"])
            break

        pats = pattern_cache[(name, k)]
        covered_pats = [p for p, c in pats if c]
        uncovered_pats = [p for p, c in pats if not c]

        # -- covered self-test (small sample, expects t<=0 always) ---------
        cov_sample = (covered_pats if len(covered_pats) <= COVERED_SELF_TEST_CAP
                      else rng.sample(covered_pats, COVERED_SELF_TEST_CAP))
        dvec_pool = demand_vectors[k]["unequal"]
        cov_worst = None
        for attach_list in cov_sample:
            dvec = list(dvec_pool[rng.randrange(len(dvec_pool))])
            rng.shuffle(dvec)
            inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                    attach=attach_list, demands=dvec)
            assert msw25_covered(inst), "msw25_covered() False on a covered pattern -- BUG"
            best_t, best_w, _ = search_max_t(inst, rng)
            n_covered_self_tested += 1
            assert best_t <= 0, (
                f"MSW25 SELF-TEST VIOLATED (bug, not a discovery -- MSW25 proves "
                f"this is impossible): tree={name} attach={attach_list} "
                f"demands={dvec} w={best_w} t={best_t}"
            )
            if cov_worst is None or best_t > cov_worst:
                cov_worst = best_t

        # -- live uncovered search ------------------------------------------
        unc_sample = (uncovered_pats if len(uncovered_pats) <= CAP_PATTERNS_PER_COMBO
                      else rng.sample(uncovered_pats, CAP_PATTERNS_PER_COMBO))
        combo_best_t = None
        combo_best_w = None
        combo_best_inst = None
        for attach_list in unc_sample:
            for _ in range(TRIALS_PER_PATTERN):
                dvec = list(dvec_pool[rng.randrange(len(dvec_pool))])
                rng.shuffle(dvec)
                inst = OutTreeInstance(root="s", tree_arcs=tree_arcs,
                                        attach=attach_list, demands=dvec)
                assert not msw25_covered(inst), \
                    "msw25_covered() True on an uncovered pattern -- BUG"
                best_t, best_w, _ = search_max_t(inst, rng)
                n_uncovered_searched += 1
                if combo_best_t is None or best_t > combo_best_t:
                    combo_best_t, combo_best_w, combo_best_inst = best_t, best_w, inst
                if best_t > 0:
                    hit = loud_report(name, tree_arcs, inst, best_t, best_w)
                    results["positive_hits"].append(hit)
                    found_positive = True
                    break
            if found_positive:
                break

        results["msw25_covered_self_test"].append({
            "tree": name, "k": k, "n_checked": len(cov_sample),
            "max_t_found": str(cov_worst) if cov_worst is not None else None,
        })
        dmax_note = None
        results["per_combo"].append({
            "tree": name, "k": k,
            "n_uncovered_structural": len(uncovered_pats),
            "n_uncovered_searched": len(unc_sample) * TRIALS_PER_PATTERN if unc_sample else 0,
            "n_covered_structural": len(covered_pats),
            "max_t_found_uncovered": str(combo_best_t) if combo_best_t is not None else None,
            "argmax_w": [str(x) for x in combo_best_w] if combo_best_w else None,
            "argmax_attach": [list(p) for p in combo_best_inst.attach] if combo_best_inst else None,
            "argmax_demands": [str(d) for d in combo_best_inst.demands] if combo_best_inst else None,
        })
        if combo_best_t is not None:
            if per_tree_best[name] is None or combo_best_t > per_tree_best[name]:
                per_tree_best[name] = combo_best_t
            if global_best is None or combo_best_t > global_best[0]:
                global_best = (combo_best_t, name, combo_best_inst.attach,
                               combo_best_inst.demands, combo_best_w)

        if idx % 3 == 0 or idx == len(combos) - 1:
            log(f"combo {idx+1}/{len(combos)}: tree={name} k={k} "
                f"uncovered={len(uncovered_pats)}(searched {len(unc_sample)}) "
                f"covered={len(covered_pats)}(self-tested {len(cov_sample)}) "
                f"best_t_uncovered={combo_best_t} "
                f"global_best={global_best[0] if global_best else None}")
            dump_partial()

    dump_partial()

    log("=" * 60)
    if found_positive:
        log("POSITIVE HIT FOUND -- see positive_hits in outtree_results.json "
            "and the LOUD REPORT above. Orchestrator must run the certificate gate.")
    else:
        log(f"No positive t(w) found. covered_self_tested={n_covered_self_tested} "
            f"uncovered_searched={n_uncovered_searched}")
        log(f"Global best t (uncovered set) = "
            f"{global_best[0] if global_best else 'n/a'} on tree "
            f"{global_best[1] if global_best else 'n/a'}")
    log(f"total elapsed: {elapsed():.1f}s")

    results["global_best_uncovered"] = None
    if global_best is not None:
        t, name, attach, dvec, w = global_best
        results["global_best_uncovered"] = {
            "t": str(t), "tree": name, "attach": [list(p) for p in attach],
            "demands": [str(d) for d in dvec], "w": [str(x) for x in w],
            "t_over_dmax": str(t / max(dvec)),
        }
    results["per_tree_best_uncovered"] = {name: (str(v) if v is not None else None)
                                           for name, v in per_tree_best.items()}
    results["n_covered_self_tested"] = n_covered_self_tested
    results["n_uncovered_searched"] = n_uncovered_searched
    results["total_elapsed_s"] = elapsed()
    dump_partial()
    log("wrote outtree_results.json")
    return results


if __name__ == "__main__":
    main()
