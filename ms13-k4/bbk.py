"""Fail-first exact B&B for R <= v (same disjunctive system as k4_proof.exact_upper_bound_bb,
same exact Fraction simplex, but the next rounding z is chosen dynamically: the z with the
fewest (row, side) options that hold at the parent LP's margin-maximising point). Ordering
only affects speed; every prune is still an exact LP verdict."""
import json, sys, time
from fractions import Fraction as Fr
import kproof as m

K, ZS, SHIFT, VARN = m.K, m._ZS, m._SHIFT, m._VARN

def dev_at(row, z, p, q):
    return sum(c * (q[i] if z[i] else -p[i]) for i, c in enumerate(row) if c)

def prove(rows, v, verbose_every=20000, max_nodes=None):
    r = len(rows)
    c_obj = [Fr(0)] * VARN; c_obj[2 * K] = Fr(1)
    A0, b0 = m._box_rows()
    st = {"nodes": 0, "lp": 0, "pruned": 0}
    ce = [None]
    t0 = time.time()
    opts = [(a, s) for a in range(r) for s in (1, -1)]

    def dfs(A, b, remaining, point):
        st["nodes"] += 1
        if max_nodes and st["nodes"] >= max_nodes:
            raise RuntimeError("cap")
        if st["nodes"] % verbose_every == 0:
            print(f"    [bb2 v={v}] nodes={st['nodes']} lp={st['lp']} pruned={st['pruned']} left={len(remaining)} t={time.time()-t0:.0f}s", flush=True)
        if not remaining:
            return
        # fail-first: z with fewest options satisfied at the parent's optimum point
        p, q = point[:K], point[K:2 * K]
        def n_ok(zj):
            zz = ZS[zj]
            return sum(1 for a, s in opts if s * dev_at(rows[a], zz, p, q) > v)
        zi = min(remaining, key=n_ok)
        z = ZS[zi]
        rest = [x for x in remaining if x != zi]
        for a, s in opts:
            mrow, mrhs = m._margin_row(rows[a], z, s, v)
            A2, b2 = A + [mrow], b + [mrhs]
            st["lp"] += 1
            res = m.simplex_max(c_obj, A2, b2, want_solution=True)
            if res is None:
                st["pruned"] += 1; continue
            val, x = res
            if val - SHIFT <= 0:
                st["pruned"] += 1; continue
            if not rest:
                ce[0] = {"margin": str(val - SHIFT), "p": [str(t) for t in x[:K]], "q": [str(t) for t in x[K:2*K]]}
                return
            dfs(A2, b2, rest, x)
            if ce[0] is not None:
                return

    try:
        x0 = [Fr(1, 2)] * (2 * K) + [SHIFT - v]
        dfs(A0, b0, list(range(len(ZS))), x0)
    except RuntimeError:
        pass
    return {"v": str(v), "proved_le_v": ce[0] is None, "counterexample": ce[0], "stats": st, "elapsed_s": time.time() - t0}

if __name__ == "__main__":
    v = Fr(sys.argv[1]); sel = [int(x) for x in sys.argv[2].split(",")]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "0" else None
    T = json.load(open(sys.argv[4]))["maximal"]
    res = {}
    for i in sel:
        rows = tuple(tuple(r) for r in T[i])
        out = prove(rows, v, max_nodes=cap)
        print(f"type {i}: proved_le_{v}={out['proved_le_v']} nodes={out['stats']['nodes']} lp={out['stats']['lp']} t={out['elapsed_s']:.0f}s ce={out['counterexample']}", flush=True)
        res[i] = out
        json.dump(res, open("bb2_k%d_%d_%d_%s.json" % (K, v.numerator, v.denominator, sys.argv[2].replace(",", "_")), "w"), indent=1, default=str)
    print("DONE")
