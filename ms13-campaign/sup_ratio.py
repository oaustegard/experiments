"""
sup_ratio.py -- phase 4 sup-ratio sweep (issue #169), superseding
certified_search.py's fixed-demand-vector decisions.

WHAT CHANGED. certified_search.py decided instances at FIXED demand vectors
(3 samples/pattern) via discrepancy.exists_bad_w -- closing the w-axis
exactly but leaving the continuous demand axis open. discrepancy's OTHER
decision procedure, exists_bad_instance(inst, bound_factor=0.0), closes that
axis too: substituting p_i = d_i w_i, q_i = d_i (1-w_i) makes the MILP
linear in (p,q), demands normalize to d_max = 1 via p_i+q_i <= 1, and
maximizing eps at threshold 0 returns the EXACT SUPREMUM, over ALL demand
vectors and ALL w simultaneously, of

    ratio(structure) = sup_{d, w}  [ min_z max_a |load_a(z,w,d)| ] / d_max

for a fixed tree+chord STRUCTURE (attach pattern), independent of demand
values entirely. ratio < 1 => that structure can NEVER yield a
counterexample, for ANY demand vector -- infinitely stronger than a finite
demand sweep. ratio >= 1 => candidate counterexample, STOP.

CALIBRATION ANCHOR. Rybin's structure (tree s->a->b->c; chords (c,a),(c,s),
(b,s)) gives ratio = 0.7500 exactly, witness demands (1, 0.5, 1) -- note
this EXCEEDS Rybin's own demands' 2/3, because the tool also optimizes
demands, not just w. Verified below as the first self-test.

COST. Empirically measured (this session, this container): k=3 ~150-250ms,
k=4 ~400-900ms, k=5 ~4-8s, k=6 ~36s+ EVEN AT THE SMALLEST (3-arc) tree --
MILP size grows with 2^k * n_arc * 2 binaries, and branch-and-bound cost
grows faster than that count (k=5->k=6 alone was a ~5-9x jump, not the ~2x
the raw binary count would suggest). This dictates the strata design below:
n=4,5 at k<=4 are cheap enough to sweep exhaustively over the uncovered set;
k=5 needs a per-tree sample cap; k=6 gets a single spot-check (not a sweep);
k=7 is not attempted at all (see the k7_note in the results and the final
report) -- the projected cost of even one call exceeds what the remaining
budget could safely absorb.

A rare but DETERMINISTIC HiGHS "Solve error" occurs on some (mostly
degenerate/repeated-chord) attach patterns -- not a timeout, not a
convergence issue, reproducible on retry. Handled by one retry then
recording as a solver_error (excluded from ratio stats, counted
separately) -- NOT treated as a self-test bug, since it is orthogonal to
correctness of the encoding (Rybin calibration + covered/consistency
self-tests all pass) and not something this read-only script can fix in
discrepancy.py.

Any ratio >= 1 hit STOPS the run immediately, is rationalized to exact
Fractions (both demands AND w, since both are free here, unlike
discrepancy.rationalize_and_verify which only rationalizes w for a FIXED
demand vector), and is verified through core.check_paths_valid,
core.check_path_closure, core.verify_counterexample before the word
"counterexample" is used anywhere.
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import random
import time
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Tuple

import outtree_search as p3
import certified_search as cs  # N67_SHAPES, chain(), sampled_structural_patterns
from core import check_path_closure, check_paths_valid, verify_counterexample
from discrepancy import exists_bad_instance
from outtree import OutTreeInstance, msw25_covered

# ---------------------------------------------------------------------------
# budget / knobs
# ---------------------------------------------------------------------------

TIME_BUDGET_S = 14 * 60
SOFT_STOP_S = 13 * 60
ENUM_CAP = 20_000
SAMPLE_RAW = 4_000

N45_K5_PER_TREE_CAP = 4      # n=5, k=5: 1,939 uncovered patterns total --
                              # exhaustive is ~2.7h at ~5s/call; sample instead
N67_K3_PER_TREE_CAP = 25
N67_K4_PER_TREE_CAP = 6
N67_K5_PER_TREE_CAP = 1  # empirically ~0% success at n=6,7 within HARD_TIMEOUT_S
                          # (both n6 trees tried so far: 1/1 timed out) -- kept
                          # nonzero (not silently dropped) so this is visible
                          # as an attempted-but-largely-unsuccessful stratum,
                          # not a stratum skipped outright

HARD_TIMEOUT_S = 45   # per-call subprocess kill switch -- see ratio_of docstring:
                      # discrepancy.exists_bad_instance has no internal time
                      # limit, and the big-M MILP encoding occasionally hits a
                      # pathological branch-and-bound blowup (observed: one
                      # call ran >160s with no sign of returning, versus a
                      # typical 4-8s and a previously-seen worst case of 58.6s
                      # at k=5). Isolated in a forked subprocess so a hung
                      # solve can be killed outright instead of stalling the
                      # whole sweep indefinitely.

SEARCH_SEED = 90210
_MP_CTX = mp.get_context("fork")

START = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - START


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


def fr_vec(vals) -> List[Fr]:
    return [Fr(v) for v in vals]


results: Dict = {
    "calibration": {},
    "self_tests": {},
    "strata": [],
    "k6_spot_check": None,
    "k7_note": None,
    "global_best": None,
    "count_ge_075": 0,
    "count_ge_090": 0,
    "count_ge_099": 0,
    "n_ratios_computed": 0,
    "total_solver_errors": 0,
    "positive_hits": [],
    "bugs": [],
    "truncated": False,
    "truncated_note": None,
    "total_elapsed_s": None,
}

found_hit = False


def dump_partial() -> None:
    with open("sup_ratio_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


def bug(msg: str) -> None:
    print("\n" + "!" * 78, flush=True)
    print(f"BUG: {msg}", flush=True)
    print("!" * 78 + "\n", flush=True)
    results["bugs"].append(msg)
    dump_partial()


# ---------------------------------------------------------------------------
# core decision wrapper: robust to the deterministic HiGHS solve-error tail
# ---------------------------------------------------------------------------

def _mp_worker(tree_arcs, attach, k, conn) -> None:
    try:
        inst = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach,
                               demands=fr_vec([1] * k))
        conn.send(exists_bad_instance(inst, bound_factor=0.0))
    except Exception as e:  # noqa: BLE001 -- surface any worker-side exception
        conn.send({"reason": f"worker exception: {e!r}"})
    finally:
        conn.close()


def _call_with_timeout(tree_arcs, attach, k) -> Dict:
    """Isolate exists_bad_instance in a forked subprocess with a hard kill
    switch. discrepancy.py has no internal MILP time limit and is read-only
    (not to be modified here), so the timeout has to live at this layer."""
    parent_conn, child_conn = _MP_CTX.Pipe()
    proc = _MP_CTX.Process(target=_mp_worker, args=(tree_arcs, attach, k, child_conn))
    proc.start()
    proc.join(HARD_TIMEOUT_S)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"reason": f"timeout after {HARD_TIMEOUT_S}s (killed)"}
    if parent_conn.poll():
        return parent_conn.recv()
    return {"reason": f"worker exited (code {proc.exitcode}) without a result"}


def ratio_of(inst: OutTreeInstance) -> Dict:
    """exists_bad_instance(inst, bound_factor=0.0)['eps'] IS the sup ratio
    (demands normalize to d_max<=1 via p_i+q_i<=1, and the optimum always
    pushes some demand to the boundary, so eps directly equals the ratio --
    no further division needed). One retry on solver error (rare,
    deterministic on the exact same instance, but cheap enough that a retry
    costs little and occasionally clears transient HiGHS state) -- but NOT
    on a timeout, which would just time out again identically."""
    last = None
    for _ in range(2):
        res = _call_with_timeout(inst.tree_arcs, inst.attach, len(inst.attach))
        if "eps" in res:
            return {"ok": True, "ratio": res["eps"], "p": res["p"], "q": res["q"],
                    "demands": res["demands"], "w": res["w"]}
        last = res
        if "timeout" in (res.get("reason") or "").lower():
            break
    return {"ok": False, "reason": last.get("reason") if last else "unknown",
            "raw": last}


def update_counts(ratio: float) -> None:
    results["n_ratios_computed"] += 1
    if ratio >= 0.75:
        results["count_ge_075"] += 1
    if ratio >= 0.90:
        results["count_ge_090"] += 1
    if ratio >= 0.99:
        results["count_ge_099"] += 1


# ---------------------------------------------------------------------------
# rationalization + verification of a ratio >= 1 hit (both demands AND w are
# free here, unlike discrepancy.rationalize_and_verify which only
# rationalizes w for a caller-fixed demand vector)
# ---------------------------------------------------------------------------

def rationalize_sup_hit(tree_name: str, tree_arcs, attach, p_float, q_float,
                        denominators=(1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 60, 120)
                        ) -> Optional[Dict]:
    for den in denominators:
        d = []
        w = []
        for pi, qi in zip(p_float, q_float):
            tot = pi + qi
            di = Fr(round(tot * den), den)
            if di <= 0:
                di = Fr(1, den)
            wf = (pi / tot) if tot > 1e-9 else 0.0
            wi = min(max(Fr(round(wf * den), den), Fr(0)), Fr(1))
            d.append(di)
            w.append(wi)
        inst = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach, demands=d)
        C = inst.coefficient_matrix()
        if not inst.is_counterexample(w, C):
            continue
        ci = inst.to_core_instance()
        problems = check_paths_valid(ci, "s") + check_path_closure(ci, "s")
        vres = verify_counterexample(ci, inst.split_from_w(w))
        if vres["is_counterexample"] and not problems:
            return {"tree": tree_name, "tree_arcs": tree_arcs,
                    "attach": [list(a) for a in attach],
                    "demands": [str(x) for x in d], "w": [str(x) for x in w],
                    "denominator": den, "t_of_x": str(vres["t_of_x"]),
                    "table": vres["table"]}
    return None


def loud_hit(tree_name, tree_arcs, attach, r: Dict, k: int) -> None:
    print("\n" + "=" * 78, flush=True)
    print("!!!! sup ratio >= 1 -- POSSIBLE COUNTEREXAMPLE STRUCTURE -- STOPPING !!!!",
          flush=True)
    print("=" * 78, flush=True)
    print(f"tree={tree_name}  tree_arcs={tree_arcs}  k={k}", flush=True)
    print(f"attach={attach}  ratio={r['ratio']}  p={r['p']}  q={r['q']}  "
          f"demands={r['demands']}  w={r['w']}", flush=True)
    verified = rationalize_sup_hit(tree_name, tree_arcs, attach, r["p"], r["q"])
    print(f"rationalize_and_verify -> {verified}", flush=True)
    print("=" * 78 + "\n", flush=True)
    rec = {"tree": tree_name, "tree_arcs": tree_arcs, "k": k,
           "attach": [list(a) for a in attach], "ratio": r["ratio"],
           "p": r["p"], "q": r["q"], "demands": r["demands"], "w": r["w"],
           "rationalized": verified}
    results["positive_hits"].append(rec)
    dump_partial()


# ---------------------------------------------------------------------------
# self-tests
# ---------------------------------------------------------------------------

def calibration_rybin() -> bool:
    tree_arcs = [("s", "a"), ("a", "b"), ("b", "c")]
    attach = [("c", "a"), ("c", "s"), ("b", "s")]
    inst = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach,
                            demands=fr_vec([1, 1, 1]))
    r = ratio_of(inst)
    results["calibration"]["rybin"] = r
    log(f"CALIBRATION Rybin: {r}")
    ok = r["ok"] and abs(r["ratio"] - 0.75) < 1e-6
    if not ok:
        bug(f"CALIBRATION FAILED: Rybin structure ratio != 0.7500, got {r}")
    dump_partial()
    return ok


def covered_self_test(rng: random.Random) -> bool:
    trees = [(name, arcs) for name, arcs, _ in p3.TREE_SHAPES
             if len(p3.nodes_of(arcs)) >= 4]
    trees += list(cs.N67_SHAPES)
    sample_trees = rng.sample(trees, min(10, len(trees)))
    n_checked = n_passed = 0
    for name, tree_arcs in sample_trees:
        pairs = p3.unordered_pairs(p3.nodes_of(tree_arcs))
        if len(pairs) < 3:
            continue
        for _ in range(40):
            attach = [pairs[rng.randrange(len(pairs))] for _ in range(3)]
            inst = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach,
                                    demands=fr_vec([1, 1, 1]))
            if not msw25_covered(inst):
                continue
            r = ratio_of(inst)
            n_checked += 1
            if r["ok"] and r["ratio"] < 1.0 - 1e-6:
                n_passed += 1
            else:
                bug(f"NG-9 SELF-TEST VIOLATED (sup_ratio): tree={name} "
                    f"attach={attach} r={r}")
            break
    results["self_tests"]["covered"] = {"n_checked": n_checked, "n_passed": n_passed}
    log(f"covered (NG-9) self-test: {n_passed}/{n_checked} passed")
    dump_partial()
    return n_checked == n_passed


def consistency_self_test(rng: random.Random) -> bool:
    try:
        p3res = json.load(open("outtree_results.json"))
    except FileNotFoundError:
        log("outtree_results.json not found -- skipping consistency self-test")
        results["self_tests"]["consistency"] = {"n_checked": 0, "n_passed": 0,
                                                 "note": "outtree_results.json missing"}
        return True
    tree_by_name = {name: arcs for name, arcs, _ in p3.TREE_SHAPES}
    candidates = [c for c in p3res["per_combo"] if c.get("argmax_attach")]
    sample = rng.sample(candidates, min(8, len(candidates)))
    n_checked = n_passed = 0
    detail = []
    for combo in sample:
        name = combo["tree"]
        tree_arcs = tree_by_name[name]
        attach = [tuple(p) for p in combo["argmax_attach"]]
        t = Fr(combo["max_t_found_uncovered"])
        dmax = max(Fr(x) for x in combo["argmax_demands"])
        ratio_fixed = float((t + dmax) / dmax)
        inst = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach,
                                demands=fr_vec([1] * len(attach)))
        r = ratio_of(inst)
        n_checked += 1
        ok = r["ok"] and r["ratio"] >= ratio_fixed - 1e-6
        if ok:
            n_passed += 1
        else:
            bug(f"CONSISTENCY SELF-TEST VIOLATED: tree={name} attach={attach} "
                f"phase3-fixed-demand ratio={ratio_fixed:.4f} but "
                f"sup_ratio={r.get('ratio')} (sup optimizes over more, so "
                f"sup_ratio must be >=)")
        detail.append({"tree": name, "k": combo["k"], "fixed_demand_ratio": ratio_fixed,
                        "sup_ratio": r.get("ratio"), "ok": ok})
    results["self_tests"]["consistency"] = {"n_checked": n_checked, "n_passed": n_passed}
    results["self_tests"]["consistency_detail"] = detail
    log(f"consistency self-test (sup_ratio >= phase3 fixed-demand heuristic): "
        f"{n_passed}/{n_checked} passed")
    dump_partial()
    return n_checked == n_passed


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------

def maybe_update_global_best(name, k, category, attach, r) -> None:
    gb = results["global_best"]
    if gb is None or r["ratio"] > gb["ratio"]:
        results["global_best"] = {
            "tree": name, "k": k, "category": category,
            "attach": [list(a) for a in attach], "ratio": r["ratio"],
            "demands": r["demands"], "w": r["w"],
        }


def sweep_stratum(name, tree_arcs, autos, k, per_tree_cap, category,
                  rng: random.Random) -> Dict:
    global found_hit
    pairs = p3.unordered_pairs(p3.nodes_of(tree_arcs))
    raw_count = math.comb(len(pairs) + k - 1, k)
    exhaustive_enum = raw_count <= ENUM_CAP
    if exhaustive_enum:
        pats = p3.structural_patterns(tree_arcs, autos, k)
    else:
        pats = cs.sampled_structural_patterns(tree_arcs, autos, pairs, k,
                                              SAMPLE_RAW, rng)
    uncovered = [p for p, c in pats if not c]
    n_uncovered_total = len(uncovered)

    if per_tree_cap is not None and len(uncovered) > per_tree_cap:
        tested_patterns = rng.sample(uncovered, per_tree_cap)
        sample_note = f"capped sample {per_tree_cap}/{len(uncovered)}"
    else:
        tested_patterns = uncovered
        sample_note = ("full uncovered set tested" if uncovered
                       else "no uncovered patterns (structurally dead)")

    stratum_best = None
    n_tested = n_errors = 0
    first_logged = False
    for attach in tested_patterns:
        if found_hit:
            break
        if elapsed() > SOFT_STOP_S:
            sample_note += " | STOPPED EARLY (soft time budget)"
            results["truncated"] = True
            results["truncated_note"] = ((results.get("truncated_note") or "") +
                                         f" {name}/k{k} truncated mid-stratum;")
            break
        inst = OutTreeInstance(root="s", tree_arcs=tree_arcs, attach=attach,
                                demands=fr_vec([1] * k))
        t0 = time.perf_counter()
        r = ratio_of(inst)
        dt = time.perf_counter() - t0
        if not first_logged:
            proj = dt * len(tested_patterns)
            log(f"  {name}/k{k} [{category}]: first call {dt*1000:.0f}ms, "
                f"{len(tested_patterns)} planned ({sample_note}) -> "
                f"projected ~{proj:.1f}s for this stratum")
            first_logged = True
        n_tested += 1
        if not r["ok"]:
            n_errors += 1
            results["total_solver_errors"] += 1
            continue
        ratio = r["ratio"]
        update_counts(ratio)
        maybe_update_global_best(name, k, category, attach, r)
        if stratum_best is None or ratio > stratum_best["ratio"]:
            stratum_best = {"attach": [list(a) for a in attach], "ratio": ratio,
                            "demands": r["demands"], "w": r["w"]}
        if ratio >= 1.0 - 1e-9:
            found_hit = True
            loud_hit(name, tree_arcs, attach, r, k)
            break
        if n_tested % 25 == 0:
            dump_partial()

    record = {"tree": name, "k": k, "category": category, "n_nodes": len(p3.nodes_of(tree_arcs)),
              "raw_combo_count": raw_count, "enum_exhaustive": exhaustive_enum,
              "n_uncovered_total": n_uncovered_total, "n_tested": n_tested,
              "n_solver_errors": n_errors, "sample_note": sample_note, "best": stratum_best}
    results["strata"].append(record)
    log(f"{name}/k{k} [{category}] done: tested={n_tested} errors={n_errors} "
        f"best_ratio={stratum_best['ratio'] if stratum_best else None} ({sample_note})")
    dump_partial()
    return record


RESUME_SNAPSHOT = "sup_ratio_results.stage1.json"


def load_resume() -> set:
    """Continuation support: stage 1 of this run hit a genuine MILP
    pathology (one call ran >160s with no sign of returning -- see
    HARD_TIMEOUT_S's docstring) and was killed after 19 strata / 257 ratios
    of clean, self-test-passed progress (0 bugs, max ratio 0.75 throughout).
    Rather than re-spend that budget, resume from the saved snapshot and
    subprocess-timeout-protect the remainder."""
    import os
    if not os.path.exists(RESUME_SNAPSHOT):
        return set()
    with open(RESUME_SNAPSHOT) as f:
        prev = json.load(f)
    for key in ("calibration", "self_tests", "strata", "count_ge_075", "count_ge_090",
                "count_ge_099", "n_ratios_computed", "total_solver_errors",
                "global_best", "positive_hits", "bugs"):
        if key in prev:
            results[key] = prev[key]
    done = {(s["tree"], s["k"]) for s in results["strata"]}
    log(f"RESUMED from {RESUME_SNAPSHOT}: {len(done)} strata already done, "
        f"{results['n_ratios_computed']} ratios already computed, "
        f"{len(results['bugs'])} bugs so far")
    results["stage1_wall_time_s_note"] = (
        "stage 1 (pre-resume) ran ~640s of real wall time before being killed "
        "on a single pathological MILP call; that time is not included in "
        "this process's elapsed()/total_elapsed_s, which only covers the "
        "continuation. See the final report for the combined wall-clock total.")
    return done


def main():
    global found_hit
    rng = random.Random(SEARCH_SEED)
    done_strata = load_resume()

    if not done_strata:
        log("sup_ratio.py starting (fresh run)")
        if not calibration_rybin():
            results["total_elapsed_s"] = elapsed()
            dump_partial()
            log("CALIBRATION FAILED -- stopping, not a valid run")
            return results
        if not covered_self_test(rng):
            results["total_elapsed_s"] = elapsed()
            dump_partial()
            log("NG-9 SELF-TEST FAILED -- stopping, not a valid run")
            return results
        if not consistency_self_test(rng):
            results["total_elapsed_s"] = elapsed()
            dump_partial()
            log("CONSISTENCY SELF-TEST FAILED -- stopping, not a valid run")
            return results
        log("all self-tests green -- starting main sweep")
    else:
        log("sup_ratio.py starting (CONTINUATION run, resumed strata skipped, "
            "self-tests already green in the snapshot)")

    # -- n=4,5: exhaustive over uncovered where budget allows -----------------
    for name, tree_arcs, max_k in p3.TREE_SHAPES:
        n_nodes = len(p3.nodes_of(tree_arcs))
        if n_nodes < 4:
            continue  # NG-9: n<=3 can never be uncovered -- structural fact
        if found_hit or elapsed() > SOFT_STOP_S:
            if not found_hit:
                results["truncated"] = True
                results["truncated_note"] = ((results.get("truncated_note") or "") +
                                             f" n45 sweep stopped before {name};")
            break
        autos = p3.compute_automorphisms(tree_arcs)
        ks = (3, 4) if max_k == 4 else (3, 4, 5)
        for k in ks:
            if (name, k) in done_strata:
                continue
            if found_hit or elapsed() > SOFT_STOP_S:
                break
            cap = N45_K5_PER_TREE_CAP if k == 5 else None
            sweep_stratum(name, tree_arcs, autos, k, cap, f"n{n_nodes}", rng)

    # -- n=6,7: sampled, honestly labeled -------------------------------------
    if not found_hit and not results["bugs"]:
        for name, tree_arcs in cs.N67_SHAPES:
            n_nodes = len(p3.nodes_of(tree_arcs))
            if found_hit or elapsed() > SOFT_STOP_S:
                results["truncated"] = True
                results["truncated_note"] = ((results.get("truncated_note") or "") +
                                             f" n67 sweep stopped before {name};")
                break
            autos = p3.compute_automorphisms(tree_arcs)
            for k in (3, 4, 5):
                if (name, k) in done_strata:
                    continue
                if found_hit or elapsed() > SOFT_STOP_S:
                    break
                cap = {3: N67_K3_PER_TREE_CAP, 4: N67_K4_PER_TREE_CAP,
                       5: N67_K5_PER_TREE_CAP}[k]
                sweep_stratum(name, tree_arcs, autos, k, cap, f"n{n_nodes}", rng)

    # -- k=6: single spot check, NOT a sweep -----------------------------------
    if not found_hit and not results["bugs"] and elapsed() < SOFT_STOP_S:
        spot_tree = [("s", "n1"), ("n1", "n2"), ("n1", "n3")]  # n4-fork, 3 arcs
        pairs = p3.unordered_pairs(p3.nodes_of(spot_tree))
        attach = [pairs[i % len(pairs)] for i in range(6)]
        inst = OutTreeInstance(root="s", tree_arcs=spot_tree, attach=attach,
                                demands=fr_vec([1] * 6))
        log("k=6 SPOT CHECK starting (smallest available 3-arc structure -- "
            "NOT representative of n=6/7 k=6 cost; single data point, per budget)")
        t0 = time.perf_counter()
        r = ratio_of(inst)
        dt = time.perf_counter() - t0
        log(f"k=6 spot check done: {dt:.1f}s, ratio={r.get('ratio')}")
        results["k6_spot_check"] = {"tree": "n4-fork-at-n1(spot,3-arc)",
                                    "attach": [list(a) for a in attach],
                                    "elapsed_s": dt, **r}
        if r.get("ok"):
            update_counts(r["ratio"])
            maybe_update_global_best("n4-fork-spot-k6", 6, "k6-spot", attach, r)
        if r.get("ok") and r["ratio"] >= 1.0 - 1e-9:
            found_hit = True
            loud_hit("n4-fork-spot-k6", spot_tree, attach, r, 6)
    elif not found_hit and not results["bugs"]:
        results["k6_spot_check"] = {"skipped": True, "reason": "soft time budget exhausted"}

    results["k7_note"] = (
        "k=7 not attempted at all -- not a silent cap. Measured growth k=5->k=6 "
        "was a ~5-9x jump (4-8s -> 36s+) even at the smallest 3-arc tree, faster "
        "than the ~2x the raw 2^k binary count alone would predict (branch-and-"
        "bound cost, not just problem size, blows up). Extrapolating the same "
        "factor projects k=7 at several hundred seconds PER CALL on the cheapest "
        "structure available -- a single call would risk consuming most of "
        "whatever budget remained for one uncertain data point. Left untouched.")

    results["total_elapsed_s"] = elapsed()
    dump_partial()
    if found_hit:
        log("POSITIVE HIT -- see positive_hits in sup_ratio_results.json")
    elif results["bugs"]:
        log(f"BUGS FOUND ({len(results['bugs'])}) -- see bugs in sup_ratio_results.json")
    else:
        log(f"No ratio>=1 hit. 0 bugs. {results['n_ratios_computed']} ratios computed, "
            f"{results['total_solver_errors']} solver errors excluded.")
    log(f"total elapsed: {elapsed():.1f}s")
    return results


if __name__ == "__main__":
    main()
