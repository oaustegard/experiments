"""Guards for the claims RESULTS.md rests on. No network, no data, no deps
beyond numpy/scipy. Run: python3 test_decoders.py
"""
import numpy as np

from decoders import (
    DECODERS,
    d1_hungarian,
    d2_row_greedy,
    dcg,
    ndcg,
    rank1_energy,
    recall_at_k,
    row_argmax_collisions,
)

rng = np.random.default_rng(0)
fails = []


def check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


print("every decoder returns a permutation of 0..N-1")
for N, K in [(5, 5), (12, 12), (50, 50), (8, 3), (8, 1)]:
    M = rng.normal(size=(N, K))
    for n, f in DECODERS.items():
        r = f(M)
        check(f"{n} N={N} K={K}", sorted(r.tolist()) == list(range(N)))

print("\nrank-1 M with a decreasing position factor: D1 is exactly argsort")
agree = 0
for _ in range(200):
    a = rng.normal(size=20)
    b = np.sort(rng.uniform(0.5, 2.0, size=20))[::-1]
    agree += int(np.array_equal(d1_hungarian(np.outer(a, b)), np.argsort(-a, kind="stable")))
check("200/200 draws", agree == 200)
check("rank1_energy of an outer product is 1", abs(rank1_energy(np.outer(a, b)) - 1) < 1e-9)

print("\nD1 attains at least D2's assignment value (it is the optimum)")
worse = 0
for _ in range(300):
    M = rng.normal(size=(15, 15))
    p1 = np.empty(15, dtype=int); p1[d1_hungarian(M)] = np.arange(15)
    p2 = np.empty(15, dtype=int); p2[d2_row_greedy(M)] = np.arange(15)
    worse += int(M[np.arange(15), p1].sum() + 1e-9 < M[np.arange(15), p2].sum())
check("D1 never scores below D2 on the assignment objective", worse == 0)

print("\ncollision diagnostic")
M = np.array([[3.0, 0.0], [2.0, 0.0]])          # both items prefer column 0
check("two items, one preferred column", row_argmax_collisions(M)[0] == 1)
M = np.array([[3.0, 0.0], [0.0, 2.0]])
check("no collision when preferences differ", row_argmax_collisions(M)[0] == 0)

print("\nmetrics")
lab = np.array([3.0, 0.0, 1.0, 0.0])
check("NDCG of the ideal order is 1", abs(ndcg(lab, np.array([0, 2, 1, 3]), 4) - 1) < 1e-12)
check("NDCG is <= 1 for a bad order", ndcg(lab, np.array([1, 3, 2, 0]), 4) < 1.0)
check("NDCG is nan with no relevant item", np.isnan(ndcg(np.zeros(4), np.arange(4), 4)))
check("DCG gain is 2^l - 1", abs(dcg(np.array([1.0]), 1) - 1.0) < 1e-12)
check("recall@2 counts relevant in the prefix",
      abs(recall_at_k(lab, np.array([0, 2, 1, 3]), 2) - 1.0) < 1e-12)

print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'all checks passed'}")
raise SystemExit(1 if fails else 0)
