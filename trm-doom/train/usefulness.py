"""How often a checkpoint's predicted set joins START to GOAL (what the bot needs), vs exact match.
python usefulness.py CKPT DATA_DIR N SUP"""
import sys, json
from collections import deque
import numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from trm import TRM
ck = torch.load(sys.argv[1], map_location="cpu"); a = ck["args"]
m = TRM(d=a["d"], heads=a["heads"], layers=a["layers"], n=a["n"], T=a["T"], mixer=a.get("mixer", "attn")); m.load_state_dict(ck["ema"]); m.eval()
X = np.load(sys.argv[2] + "/x.npy"); Y = np.load(sys.argv[2] + "/y.npy"); n = int(sys.argv[3]); sup = int(sys.argv[4])
def joins(g, s):
    st = int(np.where(g == 5)[0][0]); gl = np.where(g == 6)[0]
    if not len(gl): return False
    gl = int(gl[0]); seen = {st}; q = deque([st])
    while q:
        o = q.popleft()
        if o == gl: return True
        i, j = o % 32, o // 32
        for p, ok in ((o - 1, i > 0), (o + 1, i < 31), (o - 32, j > 0), (o + 32, j < 31)):
            if ok and p not in seen and s[p] and g[p] >= 2: seen.add(p); q.append(p)
    return False
res = {"exact": 0, "join_when_reachable": 0, "reachable": 0, "empty_when_unreachable": 0, "unreachable": 0}
with torch.no_grad():
    for k in range(n):
        g = X[k]; t = torch.from_numpy(g).long()[None]
        h = m.embed(t); y, z = m.init_state(1, 1024, "cpu", torch.float32)
        for s in range(sup): y, z, lg, q = m.step(h, y, z)
        pred = (lg[0].argmax(-1).numpy()).astype(np.uint8)
        res["exact"] += int((pred == Y[k]).all())
        if Y[k].any(): res["reachable"] += 1; res["join_when_reachable"] += int(joins(g, pred))
        else: res["unreachable"] += 1; res["empty_when_unreachable"] += int(not pred.any())
print(json.dumps(res))
