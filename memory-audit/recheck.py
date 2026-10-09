"""Check the numbers RESULTS.md states against results/summary.json. Runs in well under a second."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
s = json.loads((HERE / "results/summary.json").read_text())
t = re.sub(r"\s+", " ", (HERE / "RESULTS.md").read_text())
p, f = s["pipeline"], s["final"]
n = s["claims_verified"]
fp, rc, hc = p["first_pass"], p["recheck"], p["haiku_conduct"]


def pct(a, b):
    return f"{round(100 * a / b)}%"


def kind_row(k):
    v = s["by_kind"][k]
    m = sum(v.values())
    return (f"| {k} | {m} | {pct(v.get('outdated', 0), m)} | {pct(v.get('holds', 0), m)} | "
            f"{pct(v.get('unverifiable', 0), m)} |")


def month_row(mo):
    v = s["by_month"][mo]
    dec = v.get("outdated", 0) + v.get("holds", 0)
    return f"| {mo} | {sum(v.values())} | {pct(v.get('outdated', 0), dec)} |"


checks = {
    "headline": f"({f['outdated']} of {n:,})" in t and f["outdated"] + f["holds"] + f["unverifiable"] == n,
    "headline pct": pct(f["outdated"], n) in t and pct(f["holds"], n) in t and pct(f["unverifiable"], n) in t,
    "first pass": f"{fp['holds']:,} holds, {fp['stale']} stale, {fp['contradicted']} contradicted, {fp['unverifiable']} unverifiable" in t,
    "recheck": f"{rc['confirmed']} confirmed, {rc['confirmed_bad_correction']} confirmed with a better correction, "
               f"{rc['overturned']} overturned, {rc['unsure']} unsure" in t,
    "recheck sum": sum(v for k, v in rc.items() if k != "rechecked") == rc["rechecked"] == fp["stale"] + fp["contradicted"],
    "final from recheck": f["outdated"] == rc["confirmed"] + rc["confirmed_bad_correction"],
    "applied": f"{p['superseded_memories']} memories superseded with {p['corrections']} corrections" in t,
    "agents": sum(p["agents"].values()) == 184 and "All 184 audit agents" in t,
    "kinds": all(kind_row(k) in t for k in s["by_kind"] if k != "?"),
    "months": all(month_row(m) in t for m in s["by_month"] if m != "?"),
    "conduct": f"{hc['agents_making_them']} of {hc['agents']} agents made {hc['out_of_scope_mcp_calls']} MCP calls" in t
               and f"{hc['empty_argument_calls']} of them had empty arguments" in t,
}
bad = [k for k, ok in checks.items() if not ok]
print("recheck:", "OK" if not bad else f"FAILED {bad}", f"({len(checks)} checks)")
raise SystemExit(1 if bad else 0)
