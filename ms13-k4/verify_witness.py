"""Independent, deliberately disjoint evaluator: exact min over roundings of the
max-row |deviation| for a row-set and a rational (p,q). Shares no code with
k4_proof / k3_proof. Usage: python3 verify_witness.py '<rows json>' '<p json>' '<q json>'"""
import sys, json
from fractions import Fraction as F
from itertools import product

def dev(row, z, p, q):
    return sum(c * (q[i] if z[i] else -p[i]) for i, c in enumerate(row) if c)

def min_max_dev(rows, p, q):
    best = None
    for z in product((0, 1), repeat=len(p)):
        m = max(abs(dev(r, z, p, q)) for r in rows)
        if best is None or m < best[0]:
            best = (m, z)
    return best

if __name__ == "__main__":
    rows = json.loads(sys.argv[1]); p = [F(x) for x in json.loads(sys.argv[2])]; q = [F(x) for x in json.loads(sys.argv[3])]
    assert all(0 <= a and 0 <= b and a + b <= 1 for a, b in zip(p, q)), "outside the normalized box"
    m, z = min_max_dev(rows, p, q)
    print(f"min over z of max |dev| = {m}  (attained at z={z}); d = {[str(a+b) for a,b in zip(p,q)]}")
