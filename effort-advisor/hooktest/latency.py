#!/usr/bin/env python3
"""Question 5: wall time of minimal Haiku calls through `claude -p` (flags as in decide.py, prompt on stdin).
Usage: latency.py [out.json]"""
import json, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
ARGV = ["claude", "-p", "--model", "haiku", "--effort", "low", "--output-format", "json", "--tools", "",
        "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
        "--system-prompt", "You are a turn classifier. Reply with one line of JSON and nothing else."]
TASK = ("Refactor the sync engine in Sources/Sync so that calendar and mail share one delta-token store, "
        "keep the public API stable, add migration for existing users, and update the tests. ")
BIG = ("Classify the user prompt below for a coding agent. Decide the reasoning effort it deserves "
       "(low, medium, high, xhigh) and whether part of it should be delegated to a search subagent. "
       "Effort guide: low = lookup, rename, one-line fix, chit-chat; medium = a contained change in one or two "
       "files with a clear spec; high = multi-file change, debugging with unknown cause, design trade-offs; "
       "xhigh = architecture, concurrency, data migration, security-sensitive or irreversible work. "
       "Delegate when the prompt contains an independent read-only search over many files whose result is a "
       "short conclusion. Recent context: the session has touched 14 files in the last three turns, the last "
       "turn ended with two failing tests, and the user corrected the agent once. "
       'Answer as {"effort": "...", "delegate": true|false, "why": "<=12 words"}.\n\nUSER PROMPT:\n' + TASK * 3)
BIG = (BIG + "x" * 1500)[:1500] if len(BIG) < 1500 else BIG[:1500]
SMALL = 'Reply with exactly {"ok": true}'

def call(prompt):
    t = time.monotonic()
    r = subprocess.run(ARGV, input=prompt, capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=120)
    wall = time.monotonic() - t
    try:
        o = json.loads(r.stdout)
        return {"wall": round(wall, 2), "api_ms": o.get("duration_api_ms"), "ms": o.get("duration_ms"), "err": bool(o.get("is_error")),
                "cost": o.get("total_cost_usd"), "result": str(o.get("result"))[:120]}
    except ValueError:
        return {"wall": round(wall, 2), "err": True, "result": (r.stdout + r.stderr)[:200]}

def stats(rows):
    w = sorted(x["wall"] for x in rows)
    p90 = w[min(len(w) - 1, int(round(0.9 * (len(w) - 1))))]
    return {"n": len(w), "errors": sum(x["err"] for x in rows), "median": round(statistics.median(w), 2), "p90": p90, "min": w[0], "max": w[-1]}

out = {"prompt_chars": len(BIG)}
seq = [call(BIG) for _ in range(10)]
out["sequential_10"] = {"stats": stats(seq), "rows": seq}
t = time.monotonic()
with ThreadPoolExecutor(5) as ex:
    con = list(ex.map(call, [BIG] * 5))
out["concurrent_5"] = {"stats": stats(con), "batch_wall": round(time.monotonic() - t, 2), "rows": con}
small = [call(SMALL) for _ in range(5)]
out["small_5"] = {"stats": stats(small), "rows": small}
t = time.monotonic(); subprocess.run(["claude", "--version"], capture_output=True); out["claude_version_wall"] = round(time.monotonic() - t, 2)
s = json.dumps(out, indent=1)
open(sys.argv[1], "w").write(s) if len(sys.argv) > 1 else None
print(json.dumps({k: (v["stats"] if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
