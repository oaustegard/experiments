"""Halting on unreachable goals: random 32x32 grids, half with the goal walled off."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from handtrm import *
torch.set_num_threads(2)
m = model_from(torch.load('weights/hand.pt'))
rng = np.random.default_rng(1); n = 32; X, Y = [], []
for k in range(48):
    g = np.where(rng.random(n * n) < 0.25, WALL, FREE).astype(np.uint8)
    a, b = rng.choice(n * n, 2, replace=False); g[a] = START; g[b] = GOAL
    if k % 2:  # wall the goal in
        for o in (b - 1, b + 1, b - n, b + n):
            if 0 <= o < n * n and abs(o % n - b % n) <= 1 and g[o] != START: g[o] = WALL
    X.append(g); Y.append(bfs_union(g, n))
X, Y = np.stack(X), np.stack(Y)
P, Q = run(m, X, n, 12)
reach = Y.sum(1) > 0
first = np.where(Q.any(0), Q.argmax(0) + 1, -1)
ok = np.array([(P[first[i] - 1, i] == Y[i]).all() if first[i] > 0 else False for i in range(len(X))])
print("reachable", reach.sum(), "unreachable", (~reach).sum())
print("halted", (first > 0).sum(), "exact at halt", ok.mean(), "exact at step 12", (P[-1] == Y).all(-1).mean())
print("halt step, reachable", sorted(first[reach].tolist()))
print("halt step, unreachable", sorted(first[~reach].tolist()))
