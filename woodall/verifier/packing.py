"""CEGAR SAT loop for nu(D,u) >= k feasibility (dijoin packing).

Per issue #163 architecture: dicuts can be exponentially many, so we don't
enumerate them all upfront. Instead:
  1. Maintain a working set C of dicuts (seeded with a few obvious ones).
  2. Encode: each capacity-1 (active) arc gets a color in {0..k}, 0 = unused,
     at most one color per arc (arcs never used by 2 packed dijoins at once,
     matching u=1; an arc with u=0 is simply excluded from the variable set
     entirely -- it can never be colored, exactly modeling "null arcs can't
     be used by any packed dijoin"). For each dicut in C and each color
     c in 1..k: clause OR_{a in dicut, a active} x[a][c] (some active arc of
     that color must hit the dicut).
  3. On SAT: decode k candidate color classes, verify each is a real dijoin
     via Digraph.is_dijoin (poly, exact -- not part of the SAT encoding).
     If all k check out: nu >= k, done. If some color class fails, extract
     a violated dicut (a dicut disjoint from that class -- find one via the
     closed-set enumeration) and add it to C; loop.
  4. On UNSAT: nu < k.

This only handles capacity-1 active arcs colored 1..k (0=unused) plus
uncapacitated (no explicit u given -> treated as u=1) arcs the same way.
Arcs with u=0 are never given a SAT variable and can never appear in any
color class, matching "null arcs cannot be used by any dijoin" exactly.

For n small (<=20) exhaustive closed-set enumeration is cheap enough to run
every CEGAR round from scratch; a persistent cache isn't needed at this
scale.
"""

from __future__ import annotations

from pysat.solvers import Glucose3


def nu_at_least(D, k, max_rounds=200, verbose=False):
    """Return (bool nu>=k, witness) where witness is a list of k dijoins
    (each a list of arc labels) if SAT, else the final dicut set C if UNSAT.
    """
    active = [a for a in D.arcs if a[3] > 0]
    active_labels = [a[2] for a in active]
    if not active_labels:
        return (k == 0, [])

    # var(label, color) -> unique SAT int, colors 1..k
    def var(label, color):
        i = active_labels.index(label)
        return i * k + color  # 1-indexed color, so var >= 1

    nvars = len(active_labels) * k

    # seed C with the dicuts from all singleton predecessor-closed sets
    # (cheap, always exist for any DAG source) -- not exhaustive, CEGAR
    # will add more as needed.
    C = []
    seen = set()
    for v in D.vertices:
        S = frozenset([v])
        # only valid if predecessor-closed
        if all(u in S for (u, v2, *_ ) in D.arcs if v2 in S):
            d = D.dicut(S)
            labels = frozenset(a[2] for a in d if a[3] > 0)
            if labels and labels not in seen:
                seen.add(labels)
                C.append((S, labels))

    for round_ in range(max_rounds):
        solver = Glucose3()
        # at most one color per arc (allow 0 = no clause forcing a color)
        for lab in active_labels:
            lits = [var(lab, c) for c in range(1, k + 1)]
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    solver.add_clause([-lits[i], -lits[j]])

        # each dicut in C must be hit by every color class
        for S, labels in C:
            for c in range(1, k + 1):
                clause = [var(lab, c) for lab in labels]
                solver.add_clause(clause)

        sat = solver.solve()
        if not sat:
            solver.delete()
            return False, C

        model = set(solver.get_model())
        solver.delete()
        classes = {c: [] for c in range(1, k + 1)}
        for lab in active_labels:
            for c in range(1, k + 1):
                if var(lab, c) in model:
                    classes[c].append(lab)

        all_ok = True
        added_this_round = 0
        for c, arcs_c in classes.items():
            if D.is_dijoin(arcs_c):
                continue
            all_ok = False
            # Find A violated dicut: a predecessor-closed S whose dicut
            # (restricted to active arcs) doesn't intersect arcs_c. It's
            # fine if every such dicut is already in `seen` -- it may have
            # just been added while fixing an *earlier* class this same
            # round, against the stale pre-fix model; the constraint still
            # applies to every color once the solver re-runs next round, so
            # that's not an error. We only need >=1 genuinely new dicut
            # added somewhere in the round to guarantee progress.
            for S in D.closed_sets():
                d = D.dicut(S)
                labels = frozenset(a[2] for a in d if a[3] > 0)
                if labels and not (labels & set(arcs_c)):
                    if labels not in seen:
                        seen.add(labels)
                        C.append((S, labels))
                        added_this_round += 1
                    break
        if all_ok:
            return True, [classes[c] for c in range(1, k + 1)]
        if added_this_round == 0:
            raise RuntimeError(
                "round made no progress (no new dicut found for any failing "
                "color class, though at least one class isn't a dijoin) "
                "-- verifier bug"
            )
        if verbose:
            print(f"round {round_}: refined, |C|={len(C)}")

    raise RuntimeError(f"CEGAR did not converge in {max_rounds} rounds")


def nu(D, upper_bound=None, verbose=False):
    """Compute exact nu(D,u) by increasing k until nu_at_least(k) fails."""
    if upper_bound is None:
        upper_bound = len([a for a in D.arcs if a[3] > 0]) + 1
    k = 0
    while k < upper_bound:
        ok, _ = nu_at_least(D, k + 1, verbose=verbose)
        if not ok:
            return k
        k += 1
    return k
