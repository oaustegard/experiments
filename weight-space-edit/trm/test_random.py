"""Hand-written TRM vs BFS on random grids of several sizes and wall densities."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from handtrm import *
m = model_from(torch.load('weights/hand.pt'))
rng = np.random.default_rng(0)
for n, dens, nsup in [(16, 0.25, 6), (32, 0.3, 12), (32, 0.15, 12), (64, 0.3, 30)]:
    X, Y = [], []
    for _ in range(64):
        g = np.where(rng.random(n * n) < dens, WALL, rng.choice([FREE, FREE, FREE, DOOR, HAZARD], n * n)).astype(np.uint8)
        a, b = rng.choice(n * n, 2, replace=False); g[a] = START; g[b] = GOAL
        X.append(g); Y.append(bfs_union(g, n))
    X, Y = np.stack(X), np.stack(Y)
    P, Q = run(m, X, n, nsup)
    ex = (P == Y[None]).all(-1).mean(1)
    halted_ok = [(P[Q[:, i].argmax(), i] == Y[i]).all() for i in range(len(X)) if Q[:, i].any()]
    print(f"n={n} walls={dens} reachable={(Y.sum(1) > 0).mean():.2f} exact by step {np.round(ex, 3).tolist()} "
          f"halted={Q[-1].mean():.2f} exact-at-halt={np.mean(halted_ok):.3f}")
