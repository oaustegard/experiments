"""Random-init TRM (training config shapes), exported, plus a torch f32 reference for parity."""
import json, sys, subprocess
from pathlib import Path
import numpy as np, torch
here = Path(__file__).resolve().parent; root = here.parent
sys.path.insert(0, str(root / "train"))
from trm import TRM
torch.manual_seed(0)
cfg = dict(d=128, heads=4, layers=2, n=4, T=3, nsup=8)
src = sys.argv[1] if len(sys.argv) > 1 else ""
m = TRM(L=1024, d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"], n=cfg["n"], T=cfg["T"]).eval()
if src:
    ck = torch.load(src, map_location="cpu"); m.load_state_dict(ck["ema"]); ck = {"ema": ck["ema"], "args": ck["args"], "step": ck.get("step")}
else:
    ck = {"ema": m.state_dict(), "args": cfg, "step": 0}
(here / "tmp").mkdir(exist_ok=True)
torch.save(ck, here / "tmp/ck.pt")
subprocess.run([sys.executable, str(root / "train/export.py"), str(here / "tmp/ck.pt"), str(here / "tmp/w")], check=True)
x = np.load(root / "data/val/x.npy")[:4]
outs = []
with torch.no_grad():
    for g in x:
        t = torch.from_numpy(g).long()[None]
        h = m.embed(t); y, z = m.init_state(1, 1024, "cpu", torch.float32)
        for s in range(2):
            y, z, lg, q = m.step(h, y, z)
        outs.append({"grid": g.tolist(), "diff": (lg[0, :, 1] - lg[0, :, 0]).tolist(), "q": float(q[0])})
json.dump(outs, open(here / "tmp/ref.json", "w"))
print("ref written")
