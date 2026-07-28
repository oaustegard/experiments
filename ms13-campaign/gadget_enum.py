"""
gadget_enum.py -- exhaustive(-ish) enumeration of small implication-gadget
clause systems (kill-path A, issue #169), executing design leads E1-E4 at
k = 3 and k = 4 terminals (2 paths each, per gadgets.py's abstraction).

Read gadgets.py and NOGOS.md first -- this module is pure client code over
ClauseSystem.attack(); it does not modify either file. The only thing added
here beyond gadgets.py is (1) enumeration of floor/conflict designs, (2) a
symmetry-canonical cache key so the sympy LP (the bottleneck, ~90ms/call) is
never re-solved for two designs that are the same gadget up to relabeling,
and (3) aggregate bucketing for the ledger.

------------------------------------------------------------------------
Enumeration space
------------------------------------------------------------------------

Demand vectors (exact Fractions; see `demand_vectors()`). Per k, a small
representative set covering every shape the task asked for:
  - equal demands
  - a "(15,10,15)-shape" (two equal outer demands, one different middle one)
  - one dominant demand (d_max far above the rest)
  - all-pairs-sum > d_max (NG-1's eligibility note: only then can EVERY pair
    be ceiling-excluded somewhere)
  - a non-divisible tuple (no common factor, no simple ratio)

Floors: `floor_eligible_sets(k)` generates every eligible set of size 2..k
made of at most one PathRef per terminal (both paths of one terminal in the
same eligible set would make that terminal's coverage of the floor free --
explicitly excluded, per the task brief). `floor_configs(k, ...)` then picks
1 or 2 such sets (Lemma 2 / NG-3 caps parallel floors at k-1, and k-1 >= 2
for k in {3,4}, so 2 is the operative cap here). Serial floors (arcs in
series sharing the same coverage clause) are NOT separately modeled --
they'd collapse to the identical Floor object in this abstraction, so
"1..2 floors" already covers that case; noted once here rather than at
every call site.

Conflicts: `all_conflict_pairs(k)` is every {(i,pi),(j,pj)} with i != j --
12 of them at k=3, 24 at k=4. A design's conflict set is a SUBSET of that
universe, size 0..cap. k=3's universe (12) is small enough to enumerate the
full power set (cap=12, i.e. no cap at all -- more thorough than the task's
suggested "<=8" example). k=4's universe (24) makes the full power set
(2^24 ~ 16.8M, times ~2600 floor configs, times 5 demand vectors) far too
big for the 20-minute budget, so the k=4 sweep restricts conflicts to those
TOUCHING a floor-eligible path (`restrict_conflicts_to_floor_union=True`):
this is exactly the structured subfamily the task brief names ("conflicts
only touching floor-eligible paths plus one free terminal") -- it is the
direct generalization of the NG-6 shape (floor + conflicts fully blocking
one outside terminal) to 2 floors / 2 free terminals. k=4 coverage is
therefore PARTIAL and is reported as such, not silently capped.

Symmetry / canonical form: the group quotiented is (terminal permutations
that preserve the demand multiset) x (independent path-bit flip per
terminal) -- see module docstring on `canonical_key()`. Path bit 0 vs 1 has
no meaning until DAG realization, so the flip is a genuine symmetry of the
abstract design; terminal relabeling is only a symmetry among terminals
that share a demand value. `canonical_key()` is called ONLY for designs
that reach the LP step (post is_unsat / NG-3 / NG-6 prune) -- that is
where sympy calls are cached and where the group-size cost is worth paying.

------------------------------------------------------------------------
Pruning order (mirrors ClauseSystem.attack(); replicated here, not
imported, only so a canonical-form cache can be interposed before the LP
call -- attack() itself is never modified)
------------------------------------------------------------------------
  1. cheap 2^k SAT check (`is_unsat`)
  2. NG-3 floor budget (`floor_budget_ok`)
  3. NG-6 lemma-3 prune (`lemma3_prune`)
  4. exact LP (`realizable_x`), cached by canonical form

------------------------------------------------------------------------
No-silent-caps
------------------------------------------------------------------------
Every bound used below (floor sizes, conflict cap, demand-vector set) is a
named constant printed into `enum_results.json["bounds"]` and restated in
ENUM_RESULTS.md. If k=4's restriction excludes a design shape, that is
stated as a coverage gap, not swept under a default.
"""

from __future__ import annotations

import itertools
import json
import time
from fractions import Fraction as Fr
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from core import Instance, Terminal, check_paths_valid, verify_counterexample
from gadgets import ClauseSystem, Conflict, Floor, PathRef

# ---------------------------------------------------------------------------
# Demand vectors
# ---------------------------------------------------------------------------


def demand_vectors(k: int) -> Dict[str, List[Fr]]:
    if k == 3:
        return {
            "equal": [Fr(1), Fr(1), Fr(1)],
            "15-10-15-shape": [Fr(15), Fr(10), Fr(15)],
            "one-dominant": [Fr(1), Fr(1), Fr(10)],
            "all-pairs-sum-gt-dmax": [Fr(5), Fr(5), Fr(6)],
            "nondivisible": [Fr(7), Fr(5), Fr(3)],
        }
    if k == 4:
        return {
            "equal": [Fr(1), Fr(1), Fr(1), Fr(1)],
            "one-dominant": [Fr(1), Fr(1), Fr(1), Fr(10)],
            "all-pairs-sum-gt-dmax": [Fr(5), Fr(5), Fr(5), Fr(6)],
            "15-10-10-15-shape": [Fr(15), Fr(10), Fr(10), Fr(15)],
            "nondivisible": [Fr(9), Fr(7), Fr(6), Fr(4)],
        }
    raise ValueError(f"no demand-vector set defined for k={k}")


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------


def floor_eligible_sets(k: int, sizes: Optional[Sequence[int]] = None
                        ) -> List[FrozenSet[PathRef]]:
    """Every eligible set of size 2..k: a subset of terminals (size >= 2),
    exactly one path bit chosen per selected terminal. `sizes` restricts to
    a subset of {2..k} when given (used for the k=4 structured sweep)."""
    out = []
    allowed = set(sizes) if sizes is not None else set(range(2, k + 1))
    for size in range(2, k + 1):
        if size not in allowed:
            continue
        for subset in itertools.combinations(range(k), size):
            for bits in itertools.product((0, 1), repeat=size):
                out.append(frozenset(zip(subset, bits)))
    return out


def floor_configs(k: int, sizes: Optional[Sequence[int]] = None,
                   max_floors: Optional[int] = None
                  ) -> List[Tuple[FrozenSet[PathRef], ...]]:
    """1..cap distinct floor eligible sets (cap = min(k-1, 2) unless
    overridden). Each returned tuple is one design's floor list."""
    elig = floor_eligible_sets(k, sizes)
    cap = max_floors if max_floors is not None else min(k - 1, 2)
    configs: List[Tuple[FrozenSet[PathRef], ...]] = []
    for n in range(1, cap + 1):
        configs.extend(itertools.combinations(elig, n))
    return configs


# ---------------------------------------------------------------------------
# Conflicts
# ---------------------------------------------------------------------------


def all_conflict_pairs(k: int) -> List[FrozenSet[PathRef]]:
    """Every possible conflict pair {(i,pi),(j,pj)}, i != j."""
    out = []
    for i, j in itertools.combinations(range(k), 2):
        for pi, pj in itertools.product((0, 1), repeat=2):
            out.append(frozenset({(i, pi), (j, pj)}))
    return out


def conflict_subsets(universe: Sequence[FrozenSet[PathRef]], cap: int
                     ) -> "itertools.chain":
    n = len(universe)
    cap = min(cap, n)
    return itertools.chain.from_iterable(
        itertools.combinations(universe, s) for s in range(0, cap + 1))


# ---------------------------------------------------------------------------
# Symmetry-canonical form (for LP-result caching only)
# ---------------------------------------------------------------------------
#
# Two performance notes, both load-bearing at the sizes this sweep runs at
# (measured, not guessed -- see WORKLOG-adjacent commentary in ENUM_RESULTS.md):
#
# 1. The group itself depends only on the demand vector, not on any one
#    design, so it is built ONCE per demand vector (`build_symmetry_group`)
#    and reused as a list of precomputed PathRef->PathRef lookup tables --
#    not recomputed (permutations()/product() regenerated) on every call.
# 2. The LP is a MAXIMIZATION with monotonically more constraints as the
#    conflict set grows (each extra conflict adds one more upper bound on
#    eps; each extra floor adds one more lower bound). Adding constraints
#    to a maximization can only keep or shrink the optimum, never grow it.
#    So if (floors, C) is LP-infeasible (eps<=0), every (floors, C') with
#    C' superset of C is *automatically* LP-infeasible too -- no solve
#    needed. `run_sweep` exploits this with a per-floor-config "graveyard"
#    of already-infeasible conflict sets, checked before the symmetry cache
#    and before sympy. This is the dominant pruning lever: profiling showed
#    ~413k of 860k raw k=3 designs (one demand vector, cap=12) survive
#    is_unsat + NG-3 + NG-6 and would otherwise all reach the LP stage.


def build_symmetry_group(demands: Sequence[Fr], k: int) -> List[Dict[PathRef, PathRef]]:
    """(terminal perms preserving the demand multiset) x (per-terminal path
    bit flip), materialized as a list of PathRef->PathRef lookup dicts."""
    groups: Dict[Fr, List[int]] = {}
    for idx, d in enumerate(demands):
        groups.setdefault(d, []).append(idx)
    group_positions = list(groups.values())
    group_perms = [list(itertools.permutations(g)) for g in group_positions]

    tables = []
    for combo in itertools.product(*group_perms):
        perm = {}
        for pos_list, perm_vals in zip(group_positions, combo):
            for orig, new in zip(pos_list, perm_vals):
                perm[orig] = new
        for flips in itertools.product((0, 1), repeat=k):
            table = {(i, p): (perm[i], p ^ flips[i])
                     for i in range(k) for p in (0, 1)}
            tables.append(table)
    return tables


def canonical_key(floors: Sequence[FrozenSet[PathRef]],
                   conflicts: Sequence[FrozenSet[PathRef]],
                   group: Sequence[Dict[PathRef, PathRef]]) -> Tuple:
    """Lexicographically smallest image of (floors, conflicts) across the
    precomputed group -- the LP-result cache key."""
    best = None
    for table in group:
        mfl = frozenset(tuple(sorted(table[r] for r in fl)) for fl in floors)
        mcf = frozenset(tuple(sorted(table[r] for r in cf)) for cf in conflicts)
        key = (tuple(sorted(mfl)), tuple(sorted(mcf)))
        if best is None or key < best:
            best = key
    return best


# ---------------------------------------------------------------------------
# Bitmask UNSAT screen (rescoped 2026-07-24 for speed: the first version
# built a ClauseSystem + ran the object-oriented is_unsat() for every raw
# design before any dedup, which never finished a 5-demand-vector k=3 sweep
# inside a 900s budget. This version precomputes, ONCE per k, an 8-/16-bit
# mask per possible floor-eligible-set / conflict-pair -- the set of the
# 2^k boolean assignments that clause KILLS (floor: assignments that don't
# cover it; conflict: the one assignment pair it excludes). A design is
# UNSAT iff the OR of its clauses' masks covers all 2^k assignments. This
# needs no ClauseSystem/Floor/Conflict objects at all -- those are only
# built for the (now much smaller) set of designs that survive to the NG-3
# / NG-6 / LP stage.
# ---------------------------------------------------------------------------


def precompute_masks(k: int, floor_eligs: Sequence[FrozenSet[PathRef]],
                      conflict_pairs: Sequence[FrozenSet[PathRef]]
                     ) -> Tuple[Dict[FrozenSet[PathRef], int],
                                Dict[FrozenSet[PathRef], int], int]:
    n_assign = 1 << k

    def mask_for(refs: FrozenSet[PathRef], kind: str) -> int:
        m = 0
        for idx in range(n_assign):
            choice = [(idx >> i) & 1 for i in range(k)]
            if kind == "floor":
                covered = any(choice[i] == pi for (i, pi) in refs)
                if not covered:
                    m |= (1 << idx)
            else:  # conflict: kills the one assignment matching both refs
                hit = all(choice[i] == pi for (i, pi) in refs)
                if hit:
                    m |= (1 << idx)
        return m

    floor_mask = {e: mask_for(e, "floor") for e in floor_eligs}
    conflict_mask = {p: mask_for(p, "conflict") for p in conflict_pairs}
    return floor_mask, conflict_mask, n_assign


# ---------------------------------------------------------------------------
# Sweep driver -- canonical-form-FIRST (rescoped): compute each raw design's
# symmetry-canonical key before ANY other work and skip it outright if that
# canonical class was already visited (for this demand vector). Only the
# first representative of each orbit pays for the bitmask UNSAT check,
# NG-3, NG-6, and (rarest) the LP. Progress is printed every
# `progress_every` canonical forms and once per demand vector, both
# flush=True per the rescope brief. A wall-clock `time_budget_sec` per k
# is enforced and any truncation is reported explicitly (no-silent-caps).
# ---------------------------------------------------------------------------


def run_sweep(k: int, *, floor_sizes: Optional[Sequence[int]] = None,
              conflict_cap: int = 12,
              restrict_conflicts_to_floor_union: bool = False,
              max_floors: Optional[int] = None,
              demand_set: Optional[Dict[str, List[Fr]]] = None,
              label: str = "", time_budget_sec: float = 240.0,
              progress_every: int = 10000) -> Dict:
    dvecs = demand_set or demand_vectors(k)
    fconfigs = floor_configs(k, sizes=floor_sizes, max_floors=max_floors)
    universe_full = all_conflict_pairs(k)
    all_floor_eligs = floor_eligible_sets(k, sizes=floor_sizes)
    floor_mask, conflict_mask, n_assign = precompute_masks(k, all_floor_eligs, universe_full)
    full_mask = (1 << n_assign) - 1

    lp_cache: Dict[Tuple, Optional[Dict]] = {}
    lp_calls = [0]
    verdict_counts: Dict[str, int] = {}
    candidates: List[Dict] = []
    total_raw = 0
    total_canonical = 0
    truncated = False
    truncated_at = None

    t0 = time.time()
    for dname, demands in dvecs.items():
        if truncated:
            break
        group = build_symmetry_group(demands, k)
        D = sum(demands)
        dmax = max(demands)
        seen_canon: set = set()
        d_t0 = time.time()
        d_raw = 0
        for fconf in fconfigs:
            if restrict_conflicts_to_floor_union:
                union: set = set()
                for e in fconf:
                    union |= e
                universe = [c for c in universe_full if c & union]
            else:
                universe = universe_full
            # Per-floor-config monotonicity graveyard: once (floors, C) is
            # LP-infeasible, every C' ⊇ C is too (see canonical_key docstring
            # section 2) -- checked among canonical reps only.
            graveyard: List[FrozenSet[FrozenSet[PathRef]]] = []
            for cconf in conflict_subsets(universe, conflict_cap):
                total_raw += 1
                d_raw += 1
                key = canonical_key(fconf, cconf, group)
                if key in seen_canon:
                    continue
                seen_canon.add(key)
                total_canonical += 1
                if total_canonical % progress_every == 0:
                    elapsed_now = time.time() - t0
                    print(f"  [{label or k}/{dname}] canonical forms: "
                          f"{total_canonical} (raw seen: {total_raw}), "
                          f"elapsed {elapsed_now:.1f}s", flush=True)
                    # BUG FIX (rescope round 2): this check used to live only
                    # in the CANDIDATE branch below, so a sweep that found NO
                    # candidates (like k=4's structured pass) never tripped it
                    # and ran until an external timeout killed it with no
                    # output. Checked here instead -- unconditionally, every
                    # `progress_every` canonical forms, cheap and periodic.
                    if elapsed_now > time_budget_sec:
                        truncated = True
                        truncated_at = f"{dname} (mid floor-config sweep, periodic check)"
                        break

                union_mask = 0
                for e in fconf:
                    union_mask |= floor_mask[e]
                for p in cconf:
                    union_mask |= conflict_mask[p]
                if union_mask != full_mask:
                    verdict_counts["SAT"] = verdict_counts.get("SAT", 0) + 1
                    continue

                if not (len(fconf) * dmax < D):
                    verdict_counts["KILLED-NG3-floor-budget"] = \
                        verdict_counts.get("KILLED-NG3-floor-budget", 0) + 1
                    continue

                floors = [Floor(f"F{idx}", elig) for idx, elig in enumerate(fconf)]
                conflicts = [Conflict(f"C{idx}", pair) for idx, pair in enumerate(cconf)]
                cs = ClauseSystem(list(demands), floors, conflicts)
                reason = cs.lemma3_prune()
                if reason is not None:
                    verdict_counts["KILLED-NG6-lemma3"] = \
                        verdict_counts.get("KILLED-NG6-lemma3", 0) + 1
                    continue

                cset = frozenset(cconf)
                if any(g <= cset for g in graveyard):
                    verdict_counts["KILLED-LP-infeasible"] = \
                        verdict_counts.get("KILLED-LP-infeasible", 0) + 1
                    continue
                if key in lp_cache:
                    x = lp_cache[key]
                else:
                    lp_calls[0] += 1
                    x = cs.realizable_x()
                    lp_cache[key] = x
                if x is None:
                    graveyard.append(cset)
                    verdict_counts["KILLED-LP-infeasible"] = \
                        verdict_counts.get("KILLED-LP-infeasible", 0) + 1
                    continue

                verdict_counts["CANDIDATE"] = verdict_counts.get("CANDIDATE", 0) + 1
                candidates.append({
                    "demand_name": dname,
                    "demands": [str(d) for d in demands],
                    "floors": [(fl.arc, sorted(fl.eligible)) for fl in floors],
                    "conflicts": [(cf.arc, sorted(cf.pair)) for cf in conflicts],
                    "x": {"eps": str(x["eps"]), "w": [str(w) for w in x["w"]]},
                    "size": len(floors) + len(conflicts),
                    "_cs": cs,
                    "_x_raw": x,
                })

                if time.time() - t0 > time_budget_sec:
                    truncated = True
                    truncated_at = f"{dname} (mid floor-config sweep)"
                    break
            if truncated:
                break
        print(f"  [{label or k}/{dname}] demand vector done: {d_raw} raw designs, "
              f"elapsed {time.time() - d_t0:.1f}s (cumulative {time.time() - t0:.1f}s)",
              flush=True)

    elapsed = time.time() - t0

    return {
        "label": label or f"k={k}",
        "k": k,
        "total_raw_designs": total_raw,
        "total_canonical_designs": total_canonical,
        "unique_lp_calls": lp_calls[0],
        "verdict_counts": verdict_counts,
        "candidates": candidates,
        "elapsed_sec": elapsed,
        "truncated": truncated,
        "truncated_at": truncated_at,
        "bounds": {
            "floor_sizes": sorted(floor_sizes) if floor_sizes else list(range(2, k + 1)),
            "max_floors": max_floors if max_floors is not None else min(k - 1, 2),
            "conflict_cap": conflict_cap,
            "conflict_universe_size": len(universe_full),
            "restrict_conflicts_to_floor_union": restrict_conflicts_to_floor_union,
            "demand_vectors": list(dvecs.keys()),
            "time_budget_sec": time_budget_sec,
        },
    }


# ---------------------------------------------------------------------------
# DAG realization (deliverable 3): does a CANDIDATE abstract design actually
# build as an acyclic DAG, and does the LP witness x survive
# core.verify_counterexample -- or does realization introduce extra shared
# mass that the abstract (best-case) LP didn't account for?
# ---------------------------------------------------------------------------


def realize_gadget(cs: ClauseSystem, order: Optional[Sequence[str]] = None) -> Instance:
    """Mechanically build a core.Instance whose arc-sharing pattern EXACTLY
    matches cs's floors+conflicts: for every clause (arc name, refs), the
    only terminal-paths that traverse that arc are exactly the PathRefs in
    `refs` -- the same "exact sharing, no incidental traffic" assumption
    gadgets.py's arc_mass() makes for the LP. This is what makes the
    resulting x_from_split reproduce the LP's masses exactly.

    Construction: fix a total order over clause names (default: the order
    floors then conflicts were given in cs). Each terminal-path's required
    arc sequence is that path's clauses filtered to this order (a
    subsequence, so it's automatically consistent with the global order).
    Nodes are named in_<clause>/out_<clause>; unnamed "connector" edges
    bridge a path's previous stop (source `s`, or the out-node of its prior
    required clause) to the in-node of its next required clause, reusing an
    existing connector when two paths need the identical hop. Every edge
    then goes from a strictly lower rank (order-index) to a strictly higher
    one, so the whole graph is acyclic BY CONSTRUCTION regardless of the
    clause structure -- no separate cycle search is a correctness
    requirement, though __main__ still runs one belt-and-suspenders.
    """
    clauses = [(fl.arc, fl.eligible) for fl in cs.floors] + \
              [(cf.arc, cf.pair) for cf in cs.conflicts]
    if order is None:
        order = [name for name, _ in clauses]
    name_to_refs = dict(clauses)
    k = cs.k

    path_seq: Dict[PathRef, List[str]] = {}
    for i in range(k):
        for pi in (0, 1):
            path_seq[(i, pi)] = [name for name in order if (i, pi) in name_to_refs[name]]

    arcs: List[Tuple[str, str]] = []
    edge_cache: Dict[Tuple[str, str], bool] = {}

    def add_edge(u: str, v: str) -> Tuple[str, str]:
        e = (u, v)
        if e not in edge_cache:
            edge_cache[e] = True
            arcs.append(e)
        return e

    for name, _ in clauses:
        add_edge(f"in_{name}", f"out_{name}")

    paths: Dict[PathRef, Tuple[Tuple[str, str], ...]] = {}
    for i in range(k):
        for pi in (0, 1):
            segs = []
            prev = "s"
            for name in path_seq[(i, pi)]:
                segs.append(add_edge(prev, f"in_{name}"))
                segs.append((f"in_{name}", f"out_{name}"))
                prev = f"out_{name}"
            segs.append(add_edge(prev, f"t{i}"))
            paths[(i, pi)] = tuple(segs)

    terms = [Terminal(f"t{i}", cs.demands[i], [paths[(i, 0)], paths[(i, 1)]])
             for i in range(k)]
    return Instance(arcs, terms)


def is_acyclic(inst: Instance) -> bool:
    import collections
    adj: Dict[str, List[str]] = collections.defaultdict(list)
    nodes = set()
    for u, v in inst.arcs:
        nodes.add(u)
        nodes.add(v)
        adj[u].append(v)
    indeg = {n: 0 for n in nodes}
    for u in adj:
        for v in adj[u]:
            indeg[v] += 1
    q = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while q:
        n = q.pop()
        seen += 1
        for v in adj[n]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == len(nodes)


def attempt_realization(candidate: Dict) -> Dict:
    """Deliverable 3. Build the DAG, sanity-check it (check_paths_valid +
    acyclic), then run core.verify_counterexample with the LP's x. Returns
    an outcome dict; NEVER calls a realized CANDIDATE a counterexample
    outright -- only verify_counterexample's exact-rational verdict does."""
    cs: ClauseSystem = candidate["_cs"]
    x_raw = candidate["_x_raw"]
    inst = realize_gadget(cs)
    problems = check_paths_valid(inst, "s")
    if problems:
        return {"outcome": "realization-obstruction",
                "detail": f"check_paths_valid failed: {problems}"}
    if not is_acyclic(inst):
        return {"outcome": "realization-obstruction",
                "detail": "mechanical build produced a cycle (should not happen "
                          "by construction -- investigate order/edge_cache)"}
    w = x_raw["w"]
    split = [[wi, 1 - wi] for wi in w]
    res = verify_counterexample(inst, split)
    outcome = "realized-and-VERIFIED" if res["is_counterexample"] else "realized-and-refuted"
    return {
        "outcome": outcome,
        "is_counterexample": res["is_counterexample"],
        "t_of_x": str(res["t_of_x"]),
        "d_max": str(res["d_max"]),
        "num_arcs": len(inst.arcs),
        "routing_table": [
            {"label": row["label"], "worst_deviation": str(row["worst_deviation"]),
             "worst_arc": row["worst_arc"], "violates": row["violates"]}
            for row in res["table"]
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _strip_internal(candidates: List[Dict]) -> List[Dict]:
    """Drop the '_cs'/'_x_raw' fields (raw objects, not JSON-safe) before
    writing enum_results.json; realization outcomes are merged in first."""
    out = []
    for c in candidates:
        c2 = {k: v for k, v in c.items() if not k.startswith("_")}
        out.append(c2)
    return out


def _check_realizations(candidates: List[Dict], limit: int) -> List[Dict]:
    """Attempt realization on the `limit` smallest candidates (deduped by
    structure) and attach the outcome to each. Returns the (mutated)
    subset actually checked, smallest first."""
    seen_struct = set()
    uniq = []
    for c in sorted(candidates, key=lambda c: c["size"]):
        key = (tuple(c["demands"]),
               tuple(sorted(tuple(e) for _, e in c["floors"])),
               tuple(sorted(tuple(e) for _, e in c["conflicts"])))
        if key in seen_struct:
            continue
        seen_struct.add(key)
        uniq.append(c)
        if len(uniq) >= limit:
            break
    for c in uniq:
        c["realization"] = attempt_realization(c)
    return uniq


def known_finding_pre_rescope() -> Dict:
    """Re-verify, IN THIS SCRIPT (dual independent code paths, per
    README.md's issue-gate-3 discipline), the design found in an earlier
    exploratory pass of this same session -- before the pipeline was
    rescoped for speed. That pass used floor eligible SIZE 3 (all three
    terminals), which the rescoped sweep below deliberately excludes
    (size 2 only, per the rescope brief) for tractability. The finding is
    NOT contingent on the rescoped sweep re-deriving it; it's reproduced
    fresh here so this script is the single source of truth for it.

    Design: k=3, equal demands (1,1,1). Two floors of size 3:
      F0 = {(0,0),(1,0),(2,0)}, F1 = {(0,0),(1,0),(2,1)}
    Three conflicts:
      C0 = {(0,0),(1,1)}, C1 = {(1,0),(2,0)}, C2 = {(1,0),(2,1)}
    """
    demands = [Fr(1), Fr(1), Fr(1)]
    fl0 = Floor("F0", frozenset({(0, 0), (1, 0), (2, 0)}))
    fl1 = Floor("F1", frozenset({(0, 0), (1, 0), (2, 1)}))
    cf0 = Conflict("C0", frozenset({(0, 0), (1, 1)}))
    cf1 = Conflict("C1", frozenset({(1, 0), (2, 0)}))
    cf2 = Conflict("C2", frozenset({(1, 0), (2, 1)}))
    cs = ClauseSystem(demands, [fl0, fl1], [cf0, cf1, cf2])
    verdict = cs.attack()  # unmodified gadgets.py entry point -- independent
                           # of run_sweep's bitmask/canonical machinery above
    candidate = {
        "demand_name": "equal", "demands": [str(d) for d in demands],
        "floors": [(fl0.arc, sorted(fl0.eligible)), (fl1.arc, sorted(fl1.eligible))],
        "conflicts": [(cf0.arc, sorted(cf0.pair)), (cf1.arc, sorted(cf1.pair)),
                      (cf2.arc, sorted(cf2.pair))],
        "x": {"eps": str(verdict["x"]["eps"]), "w": [str(w) for w in verdict["x"]["w"]]},
        "size": 5, "_cs": cs, "_x_raw": verdict["x"],
    }
    # Two independent realizations: the mechanical realize_gadget() above,
    # AND a second, by-hand-designed DAG (same sharing pattern, different
    # node names/ordering, built by hand in the pre-rescope exploratory
    # pass) -- both must agree for this to count as "verified" per
    # README.md's dual-code-path discipline.
    r_mechanical = attempt_realization(candidate)

    E = {  # by-hand DAG, transcribed from the exploratory pass
        "s_F0in": ("s", "F0_in"), "F0": ("F0_in", "F0_out"),
        "F0out_F1in": ("F0_out", "F1_in"), "F1": ("F1_in", "F1_out"),
        "s_F1in": ("s", "F1_in"), "F0out_C1in": ("F0_out", "C1_in"),
        "F1out_C1in": ("F1_out", "C1_in"), "C1": ("C1_in", "C1_out"),
        "F1out_C0in": ("F1_out", "C0_in"), "s_C0in": ("s", "C0_in"),
        "C0": ("C0_in", "C0_out"), "C1out_C2in": ("C1_out", "C2_in"),
        "F1out_C2in": ("F1_out", "C2_in"), "C2": ("C2_in", "C2_out"),
        "C0out_t0": ("C0_out", "t0"), "s_t0": ("s", "t0"),
        "C2out_t1": ("C2_out", "t1"), "C0out_t1": ("C0_out", "t1"),
        "C1out_t2": ("C1_out", "t2"), "C2out_t2": ("C2_out", "t2"),
    }
    arcs2 = list(E.values())
    Z0 = [E["s_F0in"], E["F0"], E["F0out_F1in"], E["F1"], E["F1out_C0in"], E["C0"], E["C0out_t0"]]
    E0 = [E["s_t0"]]
    Z1 = [E["s_F0in"], E["F0"], E["F0out_F1in"], E["F1"], E["F1out_C1in"], E["C1"],
          E["C1out_C2in"], E["C2"], E["C2out_t1"]]
    E1 = [E["s_C0in"], E["C0"], E["C0out_t1"]]
    Z2 = [E["s_F0in"], E["F0"], E["F0out_C1in"], E["C1"], E["C1out_t2"]]
    E2 = [E["s_F1in"], E["F1"], E["F1out_C2in"], E["C2"], E["C2out_t2"]]
    terms2 = [Terminal("t0", Fr(1), [Z0, E0]), Terminal("t1", Fr(1), [Z1, E1]),
              Terminal("t2", Fr(1), [Z2, E2])]
    inst2 = Instance(arcs2, terms2)
    problems2 = check_paths_valid(inst2, "s")
    w = verdict["x"]["w"]
    split2 = [[wi, 1 - wi] for wi in w]
    res2 = verify_counterexample(inst2, split2) if not problems2 else None

    return {
        "candidate": {k: v for k, v in candidate.items() if not k.startswith("_")},
        "abstract_verdict": verdict["verdict"],
        "mechanical_realization": r_mechanical,
        "by_hand_realization": {
            "outcome": ("realized-and-VERIFIED" if res2 and res2["is_counterexample"]
                        else "realized-and-refuted" if res2 else "realization-obstruction"),
            "problems": problems2,
            "is_counterexample": res2["is_counterexample"] if res2 else None,
            "t_of_x": str(res2["t_of_x"]) if res2 else None,
        },
        "agree": (res2 is not None and
                  r_mechanical["is_counterexample"] == res2["is_counterexample"]),
    }


if __name__ == "__main__":
    t_start = time.time()
    results: Dict = {}

    print("=" * 78)
    print("PRE-RESCOPE FINDING: re-verifying via two independent code paths")
    print("=" * 78)
    finding = known_finding_pre_rescope()
    print(json.dumps(finding, indent=2, default=str))
    results["PRE_RESCOPE_FINDING"] = finding
    if (finding["mechanical_realization"]["outcome"] == "realized-and-VERIFIED"
            and finding["by_hand_realization"]["outcome"] == "realized-and-VERIFIED"
            and finding["agree"]):
        print("\n" + "!" * 78)
        print("!!! VERIFIED COUNTEREXAMPLE TO CONJECTURE 1.3 -- CONFIRMED BY TWO "
              "INDEPENDENT REALIZATIONS !!!")
        print("!" * 78 + "\n")

    print("=== k=3 rescoped sweep: floor eligible size 2 ONLY, <=2 floors, "
          "conflict cap 6, full conflict universe (12), 5 demand vectors, "
          "canonical-form-first dedup, 240s budget ===", flush=True)
    r3 = run_sweep(3, floor_sizes=[2], conflict_cap=6, label="k3-rescoped",
                    time_budget_sec=240, progress_every=10000)
    print(json.dumps({k: v for k, v in r3.items() if k != "candidates"}, indent=2), flush=True)
    print(f"candidates: {len(r3['candidates'])}", flush=True)

    new_verified = None
    if r3["candidates"]:
        print("\n--- attempting DAG realization on smallest CANDIDATEs (k=3, size-2-floor) ---",
              flush=True)
        checked = _check_realizations(r3["candidates"], limit=10)
        for c in checked:
            r = c["realization"]
            print(f"  [{c['demand_name']}] size={c['size']} -> {r['outcome']} "
                  f"(t_of_x={r.get('t_of_x')})", flush=True)
            if r["outcome"] == "realized-and-VERIFIED" and new_verified is None:
                new_verified = c
    results["k3"] = {**r3, "candidates": _strip_internal(r3["candidates"])}
    if new_verified is not None:
        results["k3"]["NEW_VERIFIED_COUNTEREXAMPLE"] = {
            k: v for k, v in new_verified.items() if not k.startswith("_")}

    print("\n=== k=4 rescoped sweep: floor eligible size 2 ONLY, <=2 floors, "
          "conflicts restricted to touching a floor-eligible path (structured "
          "subfamily -- PARTIAL coverage, declared), conflict cap 6, 5 demand "
          "vectors, 150s budget ===", flush=True)
    r4 = run_sweep(4, floor_sizes=[2], conflict_cap=6,
                    restrict_conflicts_to_floor_union=True, label="k4-rescoped",
                    time_budget_sec=150, progress_every=10000)
    print(json.dumps({k: v for k, v in r4.items() if k != "candidates"}, indent=2), flush=True)
    print(f"candidates: {len(r4['candidates'])}", flush=True)
    if r4["candidates"]:
        print("\n--- attempting DAG realization on smallest CANDIDATEs (k=4) ---", flush=True)
        checked = _check_realizations(r4["candidates"], limit=10)
        for c in checked:
            r = c["realization"]
            print(f"  [{c['demand_name']}] size={c['size']} -> {r['outcome']} "
                  f"(t_of_x={r.get('t_of_x')})", flush=True)
    results["k4"] = {**r4, "candidates": _strip_internal(r4["candidates"])}

    total_elapsed = time.time() - t_start
    results["meta"] = {
        "total_elapsed_sec": total_elapsed,
        "total_lp_calls": r3["unique_lp_calls"] + r4["unique_lp_calls"],
    }

    out_path = "enum_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}. Total elapsed: {total_elapsed:.1f}s, "
          f"total sympy LP calls: {results['meta']['total_lp_calls']}", flush=True)
