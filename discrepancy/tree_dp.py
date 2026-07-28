"""Optimal Kunisky-style tree value delta*(n) for every n, by DP.

Kunisky's construction (arXiv:2111.02974) applies to ANY rooted binary tree
with n leaves, not just complete ones.  With column scalings c_j >= 0 and
the unit-norm constraint, each internal vertex v (subtree leaf-count s_v)
caps at c_v <= 1/sqrt(s_v); constraints are independent, so the optimum
saturates them all, and the certified lower bound is

  delta*(T) = 1/sqrt(n)  [dummy column]  +  min over leaves of
              sum over internal vertices v on the root path of 1/sqrt(s_v).

Maximizing over tree shapes is the DP
  g(1) = 0,   g(s) = 1/sqrt(s) + max_{s1+s2=s} min(g(s1), g(s2)),
  delta*(n) = 1/sqrt(n) + g(n).

This answers: can a NON-complete tree beat padding a complete one at
intermediate n?  (Padding = zero columns, disc unchanged.)
"""

import sys
from functools import lru_cache


@lru_cache(maxsize=None)
def g(s: int) -> float:
    if s == 1:
        return 0.0
    return s ** -0.5 + max(min(g(a), g(s - a)) for a in range(1, s // 2 + 1))


def delta_star(n: int) -> float:
    return n ** -0.5 + g(n)


if __name__ == "__main__":
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    pad_best = 0.0
    print(" n  delta*(n)   padded-best   winner")
    for n in range(2, n_max + 1):
        d = delta_star(n)
        pad_best = max(pad_best, d)
        w = "tree" if d > pad_best - 1e-12 and d == pad_best else "pad"
        print(f"{n:3d}  {d:.6f}    {pad_best:.6f}    "
              f"{'tree(new)' if d == pad_best else 'pad'}")
