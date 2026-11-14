#!/usr/bin/env python3
"""Run one level headless on SwiftShader and save the result: play.py NAME 'map=E1M4&skill=2&...' [timeout_s]"""
import json, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
name, query = sys.argv[1], sys.argv[2]
tmo = int(sys.argv[3]) if len(sys.argv) > 3 else 7200
env = dict(os.environ, NODE_PATH=subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip())
t0 = time.time()
p = subprocess.run(["node", str(HERE / "run_page.cjs"), "test/play.html", str(tmo), query], capture_output=True, text=True, env=env, cwd=HERE.parent)
out = p.stdout
i = out.find("\n{")
try: r = json.loads(out[i:] if i >= 0 else out[out.find("{"):])
except Exception:
    sys.exit("no result:\n" + out[-3000:] + p.stderr[-2000:])
r["query"], r["wall_s"] = query, round(time.time() - t0, 1)
dst = HERE.parent / "results" / "runs" / f"{name}.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(r))
if "error" in r: sys.exit(r["error"])
ro = r["stats"].get("rollout", {})
print(name, r["map"], r["result"], f'{r["seconds"]}s', f'kills {r["kills"]}/{r["totalKills"]}', f'hp {r["health"]}', f'wall {r["wall_s"]}s',
      f'decisions {ro.get("decisions", 0)} {ro.get("meanMs", 0)}ms chosen {ro.get("chosen", {})}' if ro else "")
