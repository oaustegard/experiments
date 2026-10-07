"""Measured input-side cost of Agent-tool subagents, from their transcripts.
One usage per message id (blocks of one message repeat it). Output tokens in these
transcripts are streaming-start values, so the output figure is a floor, not a count.
Usage: subagent_cost.py <label>=<transcript> ...  -> JSON rows + totals."""
import json, sys
PRICE = {  # $/MTok: input, 5m cache write, cache read, output (claude-api skill, cached 2026-10-06)
    "claude-haiku-5-5": (0.10, 0.125, 0.01, 0.50),
    "claude-sonnet-5-5": (2.00, 2.50, 0.20, 10.0),
    "claude-opus-5-5": (4.00, 5.00, 0.20, 20.0),
}
rows = {}
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    seen, tot, model = set(), dict(inp=0, cw=0, cr=0, out_floor=0, turns=0), None
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = d.get("message")
        if not isinstance(m, dict) or m.get("role") != "assistant" or m.get("id") in seen:
            continue
        seen.add(m.get("id"))
        u = m.get("usage") or {}
        model = m.get("model") or model
        tot["inp"] += u.get("input_tokens", 0); tot["cw"] += u.get("cache_creation_input_tokens", 0)
        tot["cr"] += u.get("cache_read_input_tokens", 0); tot["out_floor"] += u.get("output_tokens", 0); tot["turns"] += 1
    p = PRICE[model]
    tot["model"] = model
    tot["usd_input_side"] = round((tot["inp"] * p[0] + tot["cw"] * p[1] + tot["cr"] * p[2]) / 1e6, 5)
    rows[label] = tot
print(json.dumps(rows, indent=0))
s = sum(r["usd_input_side"] for r in rows.values())
print(f"TOTAL input-side ${s:.4f} over {len(rows)} runs; mean ${s/len(rows):.5f}; mean cache-write tokens {sum(r['cw'] for r in rows.values())/len(rows):.0f}")
