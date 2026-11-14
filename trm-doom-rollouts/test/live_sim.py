#!/usr/bin/env python3
"""Run test/live-sim.html and summarise: live_sim.py NAME 'map=E1M4&seed=0&tics=700&fights=rollout' [timeout_s]"""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
name, query = sys.argv[1], sys.argv[2]
tmo = sys.argv[3] if len(sys.argv) > 3 else "2400"
env = dict(os.environ, NODE_PATH=subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip())
p = subprocess.run(["node", str(HERE / "run_page.cjs"), "test/live-sim.html", tmo, query], capture_output=True, text=True, env=env, cwd=HERE.parent)
out = p.stdout; i = out.find("\n{")
r = json.loads(out[i:] if i >= 0 else out[out.find("{"):])
dst = HERE.parent / "results" / "live" / f"{name}.json"; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(json.dumps(r))
if "error" in r: sys.exit(r["error"])
T = r["trace"]
fight = [t for t in T if t[5] or t[6]]
flips = sum(1 for a, b in zip(fight, fight[1:]) if a[2] * b[2] < 0)
stale = [t[7] for t in fight]
print(name, {k: r[k] for k in ("tics", "kills", "health", "dead", "maxStaleTics", "decisions", "decisionMs")})
print("  fight tics", len(fight), "side flips", flips, "mean stale tics in fights", round(sum(stale) / max(1, len(stale)), 1),
      "tics with turn 0 in fights", sum(1 for t in fight if t[3] == 0))
print("  status:", r.get("status"))
