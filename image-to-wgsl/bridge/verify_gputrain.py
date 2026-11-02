"""Re-render the splats a gputrain bridge job returned with the torch decoder (no footprint cutoff), score
them against the target, and save a small summary (no splat lists) to results/bridge_<id>.json.

    python3 bridge/verify_gputrain.py RESULT.json bridge/mona_512.bin W H
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import codec  # noqa: E402

res_path, rgba_path, W, H = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
t = Path(res_path).read_text()
res = json.loads(t[t.index("{"):])
out_ = res.get("outputs") or {}
tgt = np.fromfile(rgba_path, np.uint8).reshape(H, W, 4)[..., :3]
L = codec.Layout()
summary = {k: res.get(k) for k in ("id", "device", "tabName", "adapter", "ua", "status", "error", "runMs")}
summary.update(logs=res.get("logs"), probe=out_.get("probe"), draws=out_.get("draws"), limits=out_.get("limits"), image=f"{Path(rgba_path).name} {W}x{H}", runs=[])
for r in out_.get("runs", []):
    P = torch.tensor(r["P"], dtype=torch.float32).view(-1, 10)
    bg = torch.tensor(r["bg"], dtype=torch.float32)
    pts = []
    for p in r["points"]:
        img = codec.render_image(P, bg, H, W, r["A"], L, n=p["n"]).numpy()
        pts.append({"n": p["n"], "psnr_gpu": round(p["psnr"], 3), "psnr_torch": round(codec.psnr8(img, tgt), 3)})
    summary["runs"].append({"cfg": r["cfg"], "seconds": round(r["seconds"], 2), "steps": r["steps"], "overflows": r["overflows"], "points": pts})
    print(r["cfg"], [(p["n"], p["psnr_gpu"], p["psnr_torch"]) for p in pts], flush=True)
dest = HERE / "results" / f"bridge_{summary['id']}.json"
dest.write_text(json.dumps(summary, indent=1))
print("wrote", dest)
