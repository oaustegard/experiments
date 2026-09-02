"""Heuristic adversary: for each maximal type, maximise phi(p,q) = min_z max_a |L_a(z)| over the
box {p,q>=0, p_i+q_i<=1} by random restarts + coordinate hill-climbing in floats, then re-evaluate
the best point in exact rationals (denominators up to 60) via verify_witness.min_max_dev.
A value > k/(k+1) would refute Q7' at that k. Lower bounds only; never an upper bound."""
import json, sys, random, itertools
from fractions import Fraction as F
import numpy as np
from verify_witness import min_max_dev

def phi_np(C, Z, p, q):
    # C: rows x k, Z: 2^k x k in {0,1}. contribution: z*q - (1-z)*p
    D = Z * q - (1 - Z) * p            # 2^k x k
    L = np.abs(D @ C.T)                # 2^k x rows
    return L.max(axis=1).min()

def climb(C, Z, k, rng, iters=400):
    d = rng.random(k); w = rng.random(k)
    p = d * w; q = d * (1 - w)
    best = phi_np(C, Z, p, q)
    step = 0.25
    for it in range(iters):
        i = rng.integers(k)
        p2, q2 = p.copy(), q.copy()
        p2[i] = min(1, max(0, p[i] + rng.normal(0, step)))
        q2[i] = min(1 - p2[i], max(0, q[i] + rng.normal(0, step)))
        v = phi_np(C, Z, p2, q2)
        if v >= best:
            best, p, q = v, p2, q2
        if it % 100 == 99: step *= 0.6
    return best, p, q

def rationalize(p, q, dens=(2,3,4,5,6,8,10,12,15,20,24,30,60)):
    out = []
    for den in dens:
        P = [F(round(x*den), den) for x in p]; Q = [F(round(x*den), den) for x in q]
        P = [min(F(1), max(F(0), x)) for x in P]; Q = [min(1 - a, max(F(0), b)) for a, b in zip(P, Q)]
        out.append((P, Q))
    return out

if __name__ == "__main__":
    path = sys.argv[1]; restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    T = json.load(open(path)); k = T["k"]
    Z = np.array(list(itertools.product((0, 1), repeat=k)), dtype=float)
    rng = np.random.default_rng(0)
    target = F(k, k + 1)
    res = {}
    for i, t in enumerate(T["maximal"]):
        C = np.array(t, dtype=float)
        bestv, bp, bq = -1, None, None
        for r in range(restarts):
            v, p, q = climb(C, Z, k, rng)
            if v > bestv: bestv, bp, bq = v, p, q
        exact = max(((min_max_dev([tuple(r) for r in t], P, Q)[0], P, Q) for P, Q in rationalize(bp, bq)), key=lambda x: x[0])
        flag = "EXCEEDS" if exact[0] > target else ""
        res[i] = {"float": float(bestv), "exact": str(exact[0]), "p": [str(x) for x in exact[1]], "q": [str(x) for x in exact[2]]}
        print(f"type {i:3d} rows={len(t):2d} float={bestv:.4f} exact={exact[0]} {flag}", flush=True)
        json.dump(res, open(f"adversary_k{k}.json", "w"), indent=1)
    vals = [F(r["exact"]) for r in res.values()]
    print("max exact over types:", max(vals), "| target", target, "| n exceeding:", sum(v > target for v in vals))
