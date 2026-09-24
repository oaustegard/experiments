"""Offline half of the replay: how much time speculative execution could save, before any Jev call.

Writes results/offline.json (aggregates only; the transcripts are private and stay outside the repo).
  oracle:    every read-only call predicted perfectly, saving min(tool duration, generation window)
  reachable: only calls whose key is in the candidate pool built from earlier session context
"""
import collections, glob, json, statistics as st
from pathlib import Path
from parse import points
from candidates import candidates

TX = "/home/user/spec-data/tx/*.jsonl"
wall, n_points, per = 0.0, 0, collections.defaultdict(lambda: collections.Counter())
windows, pool = [], []
for f in sorted(glob.glob(TX)):
    ev, ps = points(f)
    ts = [e["t"] for e in ev]
    wall += sum(min(b - a, 600) for a, b in zip(ts, ts[1:]))  # gaps over 10 min are idle
    n_points += len(ps)
    for p in ps:
        if not p["labels"]:
            continue
        windows.append(p["window"])
        cands = candidates(ev, p["idx"]); pool.append(len(cands)); top60 = set(cands[:60]); cands = set(cands)
        for l in p["labels"]:
            c, s = per[l["name"]], min(l["dur"], l["window"])
            c["calls"] += 1; c["dur_s"] += l["dur"]; c["oracle_s"] += s
            if l["key"] in cands:
                c["in_pool"] += 1; c["reachable_s"] += s
            if l["key"] in top60:
                c["in_top60"] += 1
tot = collections.Counter()
for c in per.values():
    tot.update(c)
out = {
    "sessions": len(glob.glob(TX)), "active_hours": round(wall / 3600, 2), "points": n_points,
    "readonly_calls": tot["calls"], "readonly_tool_min": round(tot["dur_s"] / 60, 1),
    "oracle_min": round(tot["oracle_s"] / 60, 1), "oracle_share": round(tot["oracle_s"] / wall, 4),
    "reachable_calls": tot["in_pool"], "reachable_top60_calls": tot["in_top60"],
    "reachable_min": round(tot["reachable_s"] / 60, 1), "reachable_share": round(tot["reachable_s"] / wall, 4),
    "generation_window_s": {"median": round(st.median(windows), 2), "p90": round(sorted(windows)[int(.9 * len(windows))], 2)},
    "pool_size_median": st.median(pool),
    "by_tool": {n: {k: round(v, 1) if isinstance(v, float) else v for k, v in c.items()}
                for n, c in sorted(per.items(), key=lambda x: -x[1]["oracle_s"])},
}
Path("results/offline.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "by_tool"}, indent=1))
