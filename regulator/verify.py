"""Prove the case: ground truth satisfies every honest statement, exactly one
suspect can be the liar, and the attack time is unique. For every innocent,
extract a minimal set of testimony that rules them out (a deletion-minimised
unsat core) - the gear train that turns against that hypothesis."""
import json, sys, itertools
from z3 import Int, Solver, And, Or, Not, Bool, sat, unsat, Implies
from world import S, EVENTS, CLOCKS, PEOPLE, TRUE_OFF, SOLUTION, m, hhmm

T = {e: Int("t_" + e) for e in EVENTS}
OFF = {c: Int("off_" + c) for c in CLOCKS}


def atom_z3(a):
    k = a[0]
    if k == "read":
        _, e, c, r = a; return T[e] + OFF[c] == m(r)
    if k == "true":
        _, e, r = a; return T[e] == m(r)
    if k == "gap":
        _, e1, e2, lo, hi = a
        d = T[e2] - T[e1]
        return And(d >= lo, d <= hi) if hi is not None else d >= lo
    if k == "offdiff":
        _, a1, b1, d = a; return OFF[a1] - OFF[b1] == d
    if k == "during":
        _, e, e1, e2 = a; return And(T[e1] < T[e], T[e] < T[e2])
    if k == "at":
        return True  # handled structurally
    raise ValueError(a)


def atom_true(a):
    tv = {e: v for e, (_, v) in EVENTS.items()}
    k = a[0]
    if k == "read":
        _, e, c, r = a; return tv[e] + TRUE_OFF[c] == m(r)
    if k == "true":
        _, e, r = a; return tv[e] == m(r)
    if k == "gap":
        _, e1, e2, lo, hi = a; d = tv[e2] - tv[e1]; return d >= lo and (hi is None or d <= hi)
    if k == "offdiff":
        _, a1, b1, d = a; return TRUE_OFF[a1] - TRUE_OFF[b1] == d
    if k == "during":
        _, e, e1, e2 = a; return tv[e1] < tv[e] < tv[e2]
    if k == "at":
        return True
    raise ValueError(a)


def ats(stmts):
    return [(s["id"], a) for s in stmts for a in s["atoms"] if a[0] == "at"]


def structural(stmts, culprit):
    """Person non-overlap across rooms (1 min to move) and the culprit's
    presence in the atelier when the repeater sounds."""
    cons = []
    A = ats(stmts)
    for (i1, a1), (i2, a2) in itertools.combinations(A, 2):
        if a1[1] == a2[1] and a1[2] != a2[2]:
            s1, e1, s2, e2 = T[a1[3]], T[a1[4]], T[a2[3]], T[a2[4]]
            cons.append(((i1, i2), Or(e1 <= s2, e2 <= s1)))
    for i, a in A:
        if a[1] == culprit and a[2] != "atelier":
            cons.append(((i,), Or(T["chime"] <= T[a[3]] - 1, T["chime"] >= T[a[4]] + 1)))
    return cons


def base(s):
    for e in T: s.add(T[e] >= 0, T[e] <= 300)
    for c in OFF: s.add(OFF[c] >= -45, OFF[c] <= 45)


def retained(k):
    return [x for x in S if x["who"] != k]


def solver_for(k, track=False):
    s = Solver(); base(s)
    st = retained(k)
    lits = {}
    for x in st:
        f = And([atom_z3(a) for a in x["atoms"]] or [True])
        if track and x["who"] is not None:
            b = Bool("s_" + x["id"]); lits[x["id"]] = b; s.add(Implies(b, f))
        else:
            s.add(f)
    for ids, f in structural(st, k):
        if track and any(i in lits for i in ids):
            s.add(Implies(And([lits[i] for i in ids if i in lits]), f))
        else:
            s.add(f)
    return s, lits


def mus(k, pool=None):
    s, lits = solver_for(k, track=True)
    pool = list(lits) if pool is None else pool
    if s.check([lits[i] for i in pool]) != unsat:
        return None
    core = [str(c)[2:] for c in s.unsat_core()]
    for i in list(core):
        trial = [c for c in core if c != i]
        if s.check([lits[c] for c in trial]) == unsat:
            core = trial
    return sorted(core)


def all_mus(k, limit=12):
    """Distinct minimal refutations: find one, then search again with each of its
    statements withheld in turn (breadth-first, bounded)."""
    s, lits = solver_for(k, track=True)
    full = list(lits); found = []; seen = set(); queue = [frozenset()]
    while queue and len(found) < limit:
        banned = queue.pop(0)
        if banned in seen: continue
        seen.add(banned)
        m_ = mus(k, [i for i in full if i not in banned])
        if m_ is None: continue
        if m_ not in found: found.append(m_)
        for i in m_: queue.append(banned | {i})
    return sorted(found, key=len)


def main():
    report = {}
    # 1. ground truth: honest statements true, lies false
    bad = [(x["id"], a) for x in S for a in x["atoms"] if atom_true(a) == x["lie"] and not (x["lie"] and a[0] in ("at",))]
    lies_ok = all(not all(atom_true(a) for a in x["atoms"]) for x in S if x["lie"])
    honest_ok = all(all(atom_true(a) for a in x["atoms"]) for x in S if not x["lie"])
    print("ground truth: honest statements all true:", honest_ok, "| every lie false:", lies_ok)
    # 2. hypotheses
    feas = {}
    for k in PEOPLE:
        s, _ = solver_for(k)
        r = s.check()
        feas[k] = (r == sat)
        print(f"  {PEOPLE[k]:22s} as murderer: {'POSSIBLE' if r == sat else 'impossible'}")
    # 3. attack time under the solution
    k = SOLUTION["murderer"]
    s, _ = solver_for(k)
    times = []
    while s.check() == sat:
        v = s.model().eval(T["chime"]).as_long(); times.append(v)
        s.add(T["chime"] != v)
    print("  attack times consistent with the solution:", [hhmm(v) for v in times])
    # offsets determined?
    s, _ = solver_for(k)
    offs = {}
    for c in OFF:
        s2, _ = solver_for(k); vals = []
        while s2.check() == sat and len(vals) < 5:
            v = s2.model().eval(OFF[c]).as_long(); vals.append(v); s2.add(OFF[c] != v)
        offs[c] = vals
    print("  offsets pinned:", {c: v for c, v in offs.items()})
    # 4. gear trains
    trains = {}
    for kk in PEOPLE:
        if kk == k: continue
        trains[kk] = all_mus(kk)
        print(f"  rules out {PEOPLE[kk]:22s}: {len(trains[kk])} minimal chains, shortest", trains[kk][0])
    # 5. the trap: believe Rosa about her clock
    for kk in PEOPLE:
        s, _ = solver_for(kk); s.add(OFF["kitchen"] == 15)
        print(f"  believing Rosa's quarter-hour, {PEOPLE[kk]} as murderer:", s.check())
    # 6. naive world: every clock true
    s, _ = solver_for(None)
    for c in OFF: s.add(OFF[c] == 0)
    print("  trusting every clock, all six honest:", s.check())
    ok = honest_ok and lies_ok and [p for p in feas if feas[p]] == [k] and times == [SOLUTION["attack"]]
    report = dict(ok=ok, feasible=feas, attack=[hhmm(v) for v in times], offsets=offs, trains=trains)
    json.dump(report, open("work/verify.json", "w"), indent=1, ensure_ascii=False)
    print("UNIQUE AND CORRECT" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import os; os.makedirs("work", exist_ok=True)
    sys.exit(0 if main() else 1)
