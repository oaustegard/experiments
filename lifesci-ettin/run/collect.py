"""Fetch a finished bridge training job and assemble its params: python3 run/collect.py JOB_ID OUT.f32"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
jid, out = sys.argv[1], Path(sys.argv[2])
d = Path(__file__).resolve().parents[1] / ".work/res" / jid
r = subprocess.run([sys.executable, "/home/user/claude-workspace/scripts/gpu_bridge.py", "fetch", jid, "--out", str(d)], capture_output=True, text=True, check=True)
res = json.loads(r.stdout)
o = res["outputs"]
p = np.concatenate([np.load(o[f"p{i}"]["npy"]) for i in range(o["nPieces"])]).astype(np.float32)
assert np.isfinite(p).all()
p.tofile(out)
L = np.load(o["losses"]["npy"])
summary = {"job": jid, "device": res["device"], "adapter": res["adapter"], "msPerStep": o["msPerStep"], "trainMs": o["trainMs"], "loadMs": o["loadMs"],
           "steps": [o["from"], o["to"]], "loss_first100": float(L[:100].mean()), "loss_last100": float(L[-100:].mean()),
           "loss_by_500": [round(float(L[i:i + 500].mean()), 4) for i in range(0, len(L), 500)]}
(d.parent / f"{jid}.summary.json").write_text(json.dumps(summary, indent=1))
np.save(d.parent / f"{jid}.losses.npy", L)
print(json.dumps(summary))
