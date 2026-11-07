"""Submit one stretch of a contrastive arm to the GPU Bridge (web/bridge_contrast.js).

    python3 run/submit_contrast.py ARM FROM TO --params NAME --sha SHA [--save] [--id ID] [--total N]

ARM is stock or ls32 (which shards and table); --params is the start under R2 v1/ (round 2's
self_stock_e2.f32 / self_ls32_e2.f32). Pairs: v1/contrast/pairs.u32 (run/pairs.py order, flattened).
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
U = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/lifesci-ettin/v1"
META = json.loads((HERE / "results/shards_meta.json").read_text())
LR, WARM, SCALE = 5e-5, 100, 20.0

ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("frm", type=int); ap.add_argument("to", type=int)
ap.add_argument("--params", required=True); ap.add_argument("--sha", required=True)
ap.add_argument("--save", action="store_true"); ap.add_argument("--id"); ap.add_argument("--total", type=int, default=2428)
ap.add_argument("--timeout", type=int, default=1800000)
a = ap.parse_args()
m = META["arms"][a.arm]
cfg = {"sha": a.sha, "arm": a.arm, "V": m["vocab"], "pad": m["pad"], "from": a.frm, "to": a.to, "total": a.total,
       "lr": LR, "warm": WARM, "scale": SCALE, "save": a.save,
       "urls": {"params": f"{U}/{a.params}", "ids": f"{U}/{a.arm}.ids.u16", "lens": f"{U}/{a.arm}.lens.u16", "pairs": f"{U}/contrast/pairs.u32"}}
jid = a.id or f"lsc-{a.arm}-{a.frm}-{a.to}"
subprocess.run([sys.executable, "/home/user/claude-workspace/scripts/gpu_bridge.py", "submit", str(HERE / "web/bridge_contrast.js"), "--id", jid,
                "--note", f"lifesci-ettin contrastive {a.arm} steps {a.frm}-{a.to}", "--target", "Mac", "--timeout", str(a.timeout),
                "--input", "cfg=@" + json.dumps(cfg)], check=True)
