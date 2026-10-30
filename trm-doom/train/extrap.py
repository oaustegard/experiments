"""Run a 32×32-trained conv TRM on n×n grids: exact and join rates by supervision step. python extrap.py CKPT DIR N MAXSUP"""
import sys, json
from collections import deque
import numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from trm import TRM
ck = torch.load(sys.argv[1], map_location="cpu"); a = ck["args"]; n = int(sys.argv[3]); S = int(sys.argv[4])
m = TRM(d=a["d"], heads=a["heads"], layers=a["layers"], n=a["n"], T=a["T"], mixer=a.get("mixer", "attn")); m.load_state_dict(ck["ema"]); m.eval(); m.set_grid(n, n)
X = np.load(sys.argv[2] + "/x.npy"); Y = np.load(sys.argv[2] + "/y.npy")
def joins(g, s):
    st = int(np.where(g == 5)[0][0]); gl = int(np.where(g == 6)[0][0]); seen = {st}; q = deque([st])
    while q:
        o = q.popleft()
        if o == gl: return True
        i, j = o % n, o // n
        for p, ok in ((o - 1, i > 0), (o + 1, i < n - 1), (o - n, j > 0), (o + n, j < n - 1)):
            if ok and p not in seen and s[p] and g[p] >= 2: seen.add(p); q.append(p)
    return False
ex = np.zeros(S); jo = np.zeros(S)
with torch.no_grad():
    for k in range(len(X)):
        t = torch.from_numpy(X[k]).long()[None]; h = m.embed(t); y, z = m.init_state(1, n * n, "cpu", torch.float32)
        for s in range(S):
            y, z, lg, q = m.step(h, y, z); pred = lg[0].argmax(-1).numpy().astype(np.uint8)
            ex[s] += (pred == Y[k]).all(); jo[s] += joins(X[k], pred)
print(json.dumps({"n": n, "count": len(X), "exact_by_step": (ex / len(X)).round(3).tolist(), "join_by_step": (jo / len(X)).round(3).tolist()}))
