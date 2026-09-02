"""Check Doerr's extremal characterisation (Combinatorica 24 (2004): a TU matrix with n columns
has lindisc = 1 - 1/(n+1) iff it contains n+1 rows every n of which are linearly independent)
on every maximal k-chord type. Usage: python3 doerr_extremal.py types2_k4.json"""
import json, sys, itertools
from fractions import Fraction as F

def rank(rows):
    M = [[F(x) for x in r] for r in rows]; rk = 0
    for c in range(len(M[0])):
        piv = next((i for i in range(rk, len(M)) if M[i][c] != 0), None)
        if piv is None: continue
        M[rk], M[piv] = M[piv], M[rk]
        for i in range(len(M)):
            if i != rk and M[i][c] != 0:
                f = M[i][c] / M[rk][c]; M[i] = [a - f * b for a, b in zip(M[i], M[rk])]
        rk += 1
    return rk

def certificate(rows, k):
    for sub in itertools.combinations(rows, k + 1):
        if all(rank([r for j, r in enumerate(sub) if j != d]) == k for d in range(k + 1)):
            return sub
    return None

if __name__ == "__main__":
    T = json.load(open(sys.argv[1])); k = T["k"]
    for i, t in enumerate(T["maximal"]):
        c = certificate(t, k)
        print(f"k={k} type {i}: {'YES' if c else 'no'} {c or ''}")
