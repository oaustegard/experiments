"""Collect the headline numbers into results/summary.json for RESULTS.md evidence refs."""
import json, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
out = {"lfm": {}, "trm": {}, "arena": {}}
for f in sorted((HERE / "runs").glob("base_*.jsonl")):
    rs = [json.loads(l) for l in open(f)]
    acc = [r["correct"] for r in rs if r["correct"] is not None]
    out["lfm"][f.stem.replace("base_", "")] = {"n": len(rs), "loops": sum(r["loop"] is not None for r in rs),
        "hit_limit": sum(r["hit_limit"] for r in rs), "acc": round(sum(acc) / len(acc), 3) if acc else None}
res = json.load(open(HERE / "trm" / "results.json"))
for k, v in res.items():
    s = v["exact_by_step"]; best = max(s.values())
    full = min((int(r) for r, x in s.items() if x >= 1.0), default=None)
    out["trm"][k.replace("/", "_")] = {"at_bot_budget": v["exact_at_bot_sup"], "bot_budget": v["bot_sup"],
        "at_16": s.get("16"), "at_2": s["2"], "best": best, "first_round_100pct": full,
        "halted": v.get("halted"), "exact_at_halt": v.get("exact_at_halt")}
logs = Path("/home/user/gpu-bridge-comms/doom-logs")
for name, sess in [("hand", "2026-11-03T00-46-41-pxcj"), ("run7_ghoz", "2026-10-30T15-42-57-ghoz"), ("run7_mtud", "2026-10-30T16-43-53-mtud")]:
    att = [json.load(open(p)) for p in sorted((logs / sess).glob("0*.json"))]
    st = [a.get("stats", {}) for a in att]
    out["arena"][name] = {"session": sess, "attempts": len(att), "exits": sum(a["result"] in ("exit", "secret-exit") for a in att),
        "dead": sum(a["result"] == "dead" for a in att), "stuck": sum(a["result"] == "stuck" for a in att),
        "no_route": sum(s.get("trmMiss") or 0 for s in st) + sum(s.get("trmShut") or 0 for s in st),
        "no_route_wrong": sum(s.get("trmMiss") or 0 for s in st)}
json.dump(out, open(HERE / "results" / "summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
