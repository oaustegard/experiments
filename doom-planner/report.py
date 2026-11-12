#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Collect each run's numbers into results/summary.json (cited by RESULTS.md).

    python3 report.py e1m1-explore e1m1-assisted e1m1-judged e1-long
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_stats(name):
    res = json.loads((HERE / "results" / f"{name}.json").read_text())
    s = res["summary"]
    rundir = HERE / "runs" / name
    rows = [json.loads(l) for l in open(rundir / "decisions.jsonl")]
    events = (rundir / "events.log").read_text().splitlines()
    level_tics = [int(l.split()[-1]) for l in events if l.startswith("LEVEL")]
    deaths = [l for l in events if l.startswith("DIED")]
    options = len(rows[0]["all"]) if rows else 0
    rollouts = sum(len(r["all"]) for r in rows) * s["scenarios"]
    out = {
        "game_seconds": s["seconds"],
        "wall_seconds": res["wall_seconds"],
        "levels_completed": s["levels_completed"],
        "first_exit_tic": level_tics[0] if level_tics else None,
        "first_exit_seconds": round(level_tics[0] / 35, 1) if level_tics else None,
        "deaths": len(deaths),
        "kills": s["kills"],
        "e1m1_kills": max((r["kills"] for r in rows if r["map"] == 1), default=0),
        "final_map": s["map"],
        "damage_dealt": s["damage_dealt"],
        "decisions": s["decisions"],
        "options_per_decision": options,
        "scenarios": s["scenarios"],
        "rollouts": rollouts,
        "rollout_tics": rollouts * s["horizon"],
        "rollout_tics_per_wall_second": round(rollouts * s["horizon"] / res["wall_seconds"]),
        "giveups": sum(1 for l in events if l.startswith("GIVEUP")),
        "resets": sum(1 for l in events if l.startswith("RESET")),
        "video_url": res.get("video_url"),
        "chosen_by": dict(Counter(r["by"] for r in rows)),
    }
    jpath = rundir / "judge.jsonl"
    judged = [json.loads(l) for l in open(jpath)] if jpath.exists() else []
    if judged:
        picks = Counter(j["pick"] for j in judged)
        out["judge"] = {
            "calls": len(judged),
            "mean_seconds": round(sum(j["seconds"] for j in judged) / len(judged), 2),
            "picked_planner_top": picks.get(0, 0),
            "picked_planner_top_share": round(picks.get(0, 0) / len(judged), 3),
            "unparsed": picks.get(None, 0),
            "overrode_planner": sum(1 for j in judged if j["pick"] not in (0, None)),
            "seconds_total": round(sum(j["seconds"] for j in judged), 1),
        }
    return out


def main():
    names = sys.argv[1:]
    summary = {n: run_stats(n) for n in names}
    path = HERE / "results" / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
