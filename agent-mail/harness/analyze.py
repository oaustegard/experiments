import json, subprocess, pathlib, collections
S = pathlib.Path("/tmp/claude-0/-home-user-claude-workspace/7cfafecc-c68e-53f6-b3a5-6c31ebacf9b2")
agents = json.load(open("/tmp/ac/agents.json"))
H = S / "scratchpad/hidden"
# Haiku 5.5 $/MTok: in 0.10 out 0.50 (<=100K prompt), x5 above; cache read 0.1x, write 1.25x
def cost(u):
    p = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
    k = 5 if p > 100_000 else 1
    return k * (0.10 * u.get("input_tokens", 0) + 0.01 * u.get("cache_read_input_tokens", 0)
                + 0.125 * u.get("cache_creation_input_tokens", 0) + 0.50 * u.get("output_tokens", 0)) / 1e6
rows = []
for arm, team in agents.items():
    g = json.loads(subprocess.run(["python3", H / "grade.py", f"/tmp/ac/{arm}", H / "cases.json"], capture_output=True, text=True).stdout or "{}")
    tot = collections.Counter(); seen = set(); t0 = t1 = None
    for name, aid in team.items():
        f = S / "tasks" / f"{aid}.output"
        if not f.exists(): continue
        for line in f.read_text().splitlines():
            try: r = json.loads(line)
            except ValueError: continue
            ts = r.get("timestamp")
            if ts: t0 = min(t0 or ts, ts); t1 = max(t1 or ts, ts)
            m = r.get("message") or {}
            if r.get("type") == "assistant" and isinstance(m, dict):
                mid = m.get("id")
                if mid and mid not in seen:
                    seen.add(mid); tot["usd"] += cost(m.get("usage") or {}); tot["turns"] += 1
                    tot["out_tok"] += (m.get("usage") or {}).get("output_tokens", 0)
                for c in m.get("content") or []:
                    if isinstance(c, dict) and c.get("type") == "tool_use":
                        tot["tools"] += 1
                        if c["name"] == "SendMessage": tot["msgs"] += 1
                        cmd = (c.get("input") or {}).get("command", "")
                        if "/mail send" in cmd: tot["msgs"] += cmd.count("/mail send")
                        if "/mail recv" in cmd: tot["waits"] += 1
    from datetime import datetime
    wall = (datetime.fromisoformat(t1.replace("Z", "+00:00")) - datetime.fromisoformat(t0.replace("Z", "+00:00"))).seconds if t0 else 0
    rows.append((arm, g.get("pass"), g.get("total"), g.get("error"), g.get("mutates"), tot["msgs"], tot["waits"], tot["tools"], tot["turns"], tot["out_tok"], round(tot["usd"], 3), wall))
print(f"{'arm':16} pass  err  mut msgs waits tools turns out_tok   usd  wall_s")
for r in rows: print(f"{r[0]:16} {r[1]}/{r[2]} {str(r[3])[:30]:30} {r[4]} {r[5]:4} {r[6]:5} {r[7]:5} {r[8]:5} {r[9]:7} {r[10]:6} {r[11]}")
