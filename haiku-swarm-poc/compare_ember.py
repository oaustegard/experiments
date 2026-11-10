"""Muse Ember swarm (arm E) against the Haiku swarm (B) and Sonnet's best effort level.

Arm E is arm B with every call on muse-ember-1.0 through the hub's muse.py:
same driver, prompts, K=4, N=8, proposers at effort low, writer at medium.
One seed per problem, so this is a sample, not a test.

Haiku's `spent` includes Claude Code's own ~1.4K tokens per call and its cache
writes; `spent_api` removes that overhead. Ember has none, so its two columns
are equal, and the Haiku `spent_api` column is the like-for-like one.

    python3 compare_ember.py            # reads results/runs/
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results" / "runs"
PROBLEMS = {"circle": "", "hexagon": "hx-", "heilbronn": "ht-"}
BEST_SONNET = {"circle": "Am", "hexagon": "A", "heilbronn": "Am"}   # as in RESULTS.md
HEX_MAX_SCORE = 3.930092 / 11 ** 0.5


def load(rid: str) -> dict | None:
    d = R / rid
    sfile = d / "summary.json" if (d / "summary.json").exists() else d / "DONE"
    if not sfile.exists() or not (d / "events.jsonl").exists():
        return None
    s = json.loads(sfile.read_text())
    ev = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
    wall = 3600 if s.get("arm") != "E" else 1e9   # E ran throttled; its axis is dollars
    ev = [e for e in ev if e.get("kind") == "eval" and e["te"] <= wall]
    for e in ev:
        if s.get("problem") == "hexagon" and e["score"] > HEX_MAX_SCORE:
            e["score"] = 0.0
    ld = [json.loads(l) for l in (d / "ledger.jsonl").read_text().splitlines()] if (d / "ledger.jsonl").exists() else []
    return {"s": s, "ev": sorted(ev, key=lambda e: e["spent"]), "ledger": ld}


def first(ev, thr, key):
    hits = [e[key] for e in ev if e["score"] >= thr]
    return min(hits) if hits else None


def best_at(ev, x, key="spent"):
    vals = [e["score"] for e in ev if e[key] <= x]
    return max(vals) if vals else None


def row(rid: str, run: dict) -> dict:
    ev, s, ld = run["ev"], run["s"], run["ledger"]
    roles = {}
    for r in ld:
        k = r.get("role") or "?"
        roles.setdefault(k, [0, 0.0, 0, 0])
        roles[k][0] += 1
        roles[k][1] += r.get("cost", 0.0)
        roles[k][2] += (r.get("usage") or {}).get("cacheReadInputTokens", 0)
        roles[k][3] += not r.get("ok")
    total = sum(v[1] for v in roles.values()) or float("nan")
    progs = len({e["sha"] for e in ev})
    return {
        "run": rid, "spent": s["spent"], "spent_api": s.get("spent_api", s["spent"]),
        "min": s.get("elapsed", 0) / 60, "programs": progs,
        "per_usd": progs / s["spent"] if s["spent"] else None,
        "best": max((e["score"] for e in ev), default=None),
        "at_0.05": best_at(ev, 0.05), "at_0.25": best_at(ev, 0.25),
        "usd99": first(ev, 0.99, "spent"), "usd99_api": first(ev, 0.99, "spent_api"),
        "usd999": first(ev, 0.999, "spent"), "min99": (first(ev, 0.99, "te") or 0) / 60 or None,
        "prop_share": roles.get("proposer", [0, 0])[1] / total,
        "failed_calls": sum(v[3] for v in roles.values()), "calls": len(ld),
    }


def f(v, nd=4):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    out, lines = {}, []
    for prob, tag in PROBLEMS.items():
        lines += [f"\n### {prob}\n",
                  "| run | spent $ | API-equiv $ | programs/$ | best | best at $0.05 | best at $0.25 | $ to 0.99 | API-equiv $ to 0.99 | $ to 0.999 | proposer share | failed calls |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rids = [f"{tag}E-s1"] + [f"{tag}B-s{s}" for s in (1, 2, 3)] + [f"{tag}{BEST_SONNET[prob]}-s{s}" for s in (1, 2, 3)]
        for rid in rids:
            run = load(rid)
            if not run:
                continue
            r = row(rid, run)
            out[rid] = r
            lines.append(f"| {rid} | {f(r['spent'])} | {f(r['spent_api'])} | {f(r['per_usd'], 0)} | {f(r['best'], 5)} | "
                         f"{f(r['at_0.05'], 4)} | {f(r['at_0.25'], 4)} | {f(r['usd99'])} | {f(r['usd99_api'])} | "
                         f"{f(r['usd999'])} | {f(r['prop_share'] * 100, 0)}% | {r['failed_calls']}/{r['calls']} |")
    (R.parent / "ember.json").write_text(json.dumps(out, indent=1))
    (R.parent / "ember_table.md").write_text("\n".join(lines).lstrip() + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
