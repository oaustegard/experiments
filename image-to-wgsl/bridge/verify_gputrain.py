"""Re-render the splats a gputrain bridge job returned with the torch decoder (no footprint cutoff), score
them against the target, and save a small summary (no splat lists) to results/bridge_<id>.json.

    python3 bridge/verify_gputrain.py RESULT.json bridge/mona_512.bin W H [FOCUS.bin]

With FOCUS.bin (u8 weights, W*H), points also get the focus-weighted PSNR, the map floored and normalised
as trainer.js normaliseFocus does.
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
fw = None
if len(sys.argv) > 5:
    f = np.fromfile(sys.argv[5], np.uint8).astype(np.float64).reshape(H, W) / 255
    f = np.maximum(f / f.max(), 0.05); fw = f / f.mean()


def psnr8w(img, ref, w):
    a = np.clip(np.rint(img * 255), 0, 255); d2 = ((a - ref.astype(np.float64)) ** 2).sum(-1)
    return float(10 * np.log10(255 ** 2 / max((w * d2).sum() / (3 * w.sum()), 1e-12)))
L = codec.Layout()
summary = {k: res.get(k) for k in ("id", "device", "tabName", "adapter", "ua", "status", "error", "runMs")}
summary.update(logs=res.get("logs"), probe=out_.get("probe"), draws=out_.get("draws"), limits=out_.get("limits"), image=f"{Path(rgba_path).name} {W}x{H}", runs=[])
for r in out_.get("runs", []):
    P = torch.tensor(r["P"], dtype=torch.float32).view(-1, 10)
    bg = torch.tensor(r["bg"], dtype=torch.float32)
    pts = []
    for p in r["points"]:
        img = codec.render_image(P, bg, H, W, r["A"], L, n=p["n"]).numpy()
        pt = {"n": p["n"], "psnr_gpu": round(p["psnr"], 3), "psnr_torch": round(codec.psnr8(img, tgt), 3)}
        if fw is not None and "psnrFocus" in p:
            pt.update(psnr_focus_gpu=round(p["psnrFocus"], 3), psnr_focus_torch=round(psnr8w(img, tgt, fw), 3))
        pts.append(pt)
    summary["runs"].append({"cfg": r["cfg"], "seconds": round(r["seconds"], 2), "steps": r["steps"], "overflows": r["overflows"], "points": pts})
    print(r["cfg"], [(p["n"], p["psnr_gpu"], p["psnr_torch"], p.get("psnr_focus_gpu"), p.get("psnr_focus_torch")) for p in pts], flush=True)
dest = HERE / "results" / f"bridge_{summary['id']}.json"
dest.write_text(json.dumps(summary, indent=1))
print("wrote", dest)
