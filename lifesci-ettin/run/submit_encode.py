"""Submit a TREC-COVID doc encode to the bridge: python3 run/submit_encode.py TOKARM (--params NAME.f32 | --cached ARM) --sha SHA [--id ID]"""
import argparse, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
U = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/lifesci-ettin/v1"
META = json.loads((HERE / ".work/shards/meta.json").read_text())
ap = argparse.ArgumentParser(); ap.add_argument("tokarm"); ap.add_argument("--params"); ap.add_argument("--cached"); ap.add_argument("--sha", required=True); ap.add_argument("--id")
a = ap.parse_args()
m = META["arms"][a.tokarm]
cfg = {"sha": a.sha, "V": m["vocab"], "pad": m["pad"], "texts": {"ids": f"{U}/evaltok/{a.tokarm}.trec.ids.u16", "lens": f"{U}/evaltok/{a.tokarm}.trec.lens.u16"}}
if a.cached: cfg["arm"] = a.cached
else: cfg["params"] = f"{U}/{a.params}"
jid = a.id or f"lse-enc-{a.cached or a.params.replace('.f32', '')}"
subprocess.run([sys.executable, "/home/user/claude-workspace/scripts/gpu_bridge.py", "submit", str(HERE / "web/bridge_encode.js"), "--id", jid,
                "--note", f"lifesci-ettin TREC-COVID encode {jid}", "--target", "Mac", "--timeout", "1800000", "--input", "cfg=@" + json.dumps(cfg)], check=True)
