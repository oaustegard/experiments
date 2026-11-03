"""Run the hand-written TRM on one held-out grid and dump its state channels for the hero shader.

Picks the reachable E1M9 grid with the longest route, runs rounds until the halting head fires,
and writes hero/state.json: per cell the map token, the wavefront label (0-2, or -1 unlabelled)
and the route mark, all read from the network's own y channels; plus BFS distance to the goal
for shading.
"""
import json, sys
from collections import deque
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trm"))
from handtrm import model_from, L as LCH, M as MCH, GOAL, FREE, BIG

name, n = sys.argv[1] if len(sys.argv) > 1 else "e1m9_96", int(sys.argv[2]) if len(sys.argv) > 2 else 96
X = np.load(HERE.parent / "trm" / "data" / name / "x.npy"); Y = np.load(HERE.parent / "trm" / "data" / name / "y.npy")
k = int(np.argmax(Y.sum(1))); g = X[k]
print("grid", k, "route cells", int(Y[k].sum()))
m = model_from(torch.load(HERE.parent / "trm" / "weights" / "hand.pt")); m.set_grid(n, n)
with torch.no_grad():
    x = torch.from_numpy(g[None].astype(np.int64)); h = m.embed(x); y, z = m.init_state(1, n * n, "cpu", h.dtype)
    for s in range(60):
        y, z, lg, q = m.step(h, y, z)
        if q.item() > 0: break
print("halted at round", s + 1)
r = (y[0] / y[0, :, :1] * BIG).numpy()
lab = np.where(r[:, LCH].max(1) > 0.5, r[:, LCH].argmax(1), -1)
mark = (r[:, MCH].max(1) > 0.5).astype(int)
assert (mark == Y[k]).all(), "network route differs from BFS"
t = int(np.where(g == GOAL)[0][0]); d = np.full(n * n, -1); d[t] = 0; q = deque([t])
while q:
    o = q.popleft(); i, j = o % n, o // n
    for p in (o - 1 if i else -1, o + 1 if i < n - 1 else -1, o - n if j else -1, o + n if j < n - 1 else -1):
        if p >= 0 and d[p] < 0 and g[p] >= FREE: d[p] = d[o] + 1; q.append(p)
assert ((d >= 0) == (lab >= 0)).all() and (d[d >= 0] % 3 == lab[d >= 0]).all(), "labels differ from BFS mod 3"
json.dump({"n": n, "rounds": s + 1, "tok": g.tolist(), "lab": lab.tolist(), "mark": mark.tolist(), "dist": d.tolist(),
           "dmax": int(d.max())}, open(HERE / "state.json", "w"))
print("labels and route match BFS; dmax", d.max())
