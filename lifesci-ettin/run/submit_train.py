"""Submit one stretch of an arm to the GPU Bridge (web/bridge_train.js).

    python3 run/submit_train.py ARM FROM TO [--save] [--id ID] --sha SHA [--teacher DIR --mu FILE --params NAME]
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
U = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/lifesci-ettin/v1"
META = json.loads((HERE / ".work/shards/meta.json").read_text())
LR, WARM = 3e-4, 300

ap = argparse.ArgumentParser()
ap.add_argument("arm"); ap.add_argument("frm", type=int); ap.add_argument("to", type=int)
ap.add_argument("--save", action="store_true"); ap.add_argument("--id"); ap.add_argument("--sha", required=True)
ap.add_argument("--timeout", type=int, default=1800000)
ap.add_argument("--teacher", default="teacher", help="R2 dir under v1/ holding t0..t9.bin")
ap.add_argument("--mu", default=str(HERE / ".work/teacher/mu_all.npy"))
ap.add_argument("--params", help="R2 file under v1/ to start from (default ARM_h.f32)")
a = ap.parse_args()
m = META["arms"][a.arm]
cfg = {"sha": a.sha, "arm": a.arm, "V": m["vocab"] if a.arm != "stock" else 50368, "pad": m["pad"], "from": a.frm, "to": a.to,
       "total": META["steps"], "lr": LR, "warm": WARM, "save": a.save,
       "urls": {"params": f"{U}/{a.params or a.arm + '_h.f32'}", "ids": f"{U}/{a.arm}.ids.u16", "lens": f"{U}/{a.arm}.lens.u16", "order": f"{U}/order.u32",
                "teacher": [f"{U}/{a.teacher}/t{i}.bin" for i in range(10)]}}
jid = a.id or f"lse-{a.arm}-{a.frm}-{a.to}"
subprocess.run([sys.executable, "/home/user/claude-workspace/scripts/gpu_bridge.py", "submit", str(HERE / "web/bridge_train.js"), "--id", jid,
                "--note", f"lifesci-ettin {a.arm} steps {a.frm}-{a.to}", "--target", "Mac", "--timeout", str(a.timeout),
                "--input", "cfg=@" + json.dumps(cfg), "--input", f"mu={a.mu}"], check=True)
