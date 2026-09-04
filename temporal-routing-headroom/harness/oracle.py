#!/usr/bin/env python3
"""Stage-1 analysis: does any router have room to pay?

Reads graded results for the weak and strong arms across replicates plus measured output
tokens, and reports four things:

  per-arm      cost per completed task, the number the routing decision actually turns on
  oracle       route to weak iff weak solves it -- the upper bound on every router,
               learned or not. If this is not materially below all-strong, Stage 2 and
               Stage 3 cannot pay and the project stops here.
  disjointness |W \\ S|, the tasks weak solves and strong does not. SWE-Router's curve
               passes above its own all-strong marker only because this set is non-empty;
               our agent-routing skill assumes the opposite (escalation is monotone).
  stability    per-task solve agreement across replicates. A task that flips between
               replicates carries no routing signal, and the fraction that flip bounds
               how much of any measured routing gain is noise.

Inputs (all under data/):
  results_stage1*.json   {replicate: {arm: {task: {"passed": bool, ...}}}}
  tokens_stage1*.json    {replicate: {arm: {task: output_tokens}}}

Usage:
  python3 oracle.py                       # read data/, write data/analysis_stage1.json
  python3 oracle.py --demo                # run the arithmetic on synthetic input
"""
import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
P = json.loads((ROOT / "params.json").read_text())
PRICES = P["api_prices_usd_per_mtok"]
ARMS = P["arms"]


def load(pattern):
    merged = {}
    for f in sorted(DATA.glob(pattern)):
        for rep, per_arm in json.loads(f.read_text()).items():
            for arm, per_task in per_arm.items():
                merged.setdefault(rep, {}).setdefault(arm, {}).update(per_task)
    return merged


def out_price(arm):
    return PRICES[ARMS[arm]["model"]]["output"] / 1_000_000


def arm_summary(results, tokens, arm):
    """Cost per completed task, pooled over replicates."""
    solved = cost = 0
    per_rep = []
    for rep, per_arm in results.items():
        if arm not in per_arm:
            continue
        tasks = per_arm[arm]
        s = sum(1 for r in tasks.values() if r["passed"])
        c = sum(tokens.get(rep, {}).get(arm, {}).get(t, 0) for t in tasks) * out_price(arm)
        solved += s
        cost += c
        per_rep.append({"replicate": rep, "solved": s, "n": len(tasks), "cost_usd": round(c, 4)})
    return {
        "arm": arm,
        "model": ARMS[arm]["model"],
        "solved": solved,
        "attempted": sum(r["n"] for r in per_rep),
        "cost_usd": round(cost, 4),
        "cost_per_completed": round(cost / solved, 4) if solved else None,
        "per_replicate": per_rep,
    }


def solve_sets(results, arm):
    """task -> list of pass booleans, one per replicate."""
    out = {}
    for rep, per_arm in results.items():
        for task, r in per_arm.get(arm, {}).items():
            out.setdefault(task, []).append(bool(r["passed"]))
    return out


def majority(flags):
    return sum(flags) * 2 > len(flags)


def analyse(results, tokens):
    weak, strong = solve_sets(results, "weak"), solve_sets(results, "strong")
    tasks = sorted(set(weak) | set(strong))

    W = {t for t in tasks if majority(weak.get(t, []))}
    S = {t for t in tasks if majority(strong.get(t, []))}

    def mean_tokens(arm, task):
        vals = [tokens[rep][arm][task] for rep in tokens
                if arm in tokens[rep] and task in tokens[rep][arm]]
        return statistics.mean(vals) if vals else 0.0

    # Oracle: send each task to weak iff weak solves it, else to strong. The weak attempt
    # is NOT free on escalation in a real cascade, but the oracle never makes a losing
    # attempt, which is exactly why it is a bound and not a strategy.
    oracle_cost = oracle_solved = 0.0
    routed = {}
    for t in tasks:
        if t in W:
            oracle_cost += mean_tokens("weak", t) * out_price("weak")
            oracle_solved += 1
            routed[t] = "weak"
        else:
            oracle_cost += mean_tokens("strong", t) * out_price("strong")
            oracle_solved += 1 if t in S else 0
            routed[t] = "strong"

    flips = {t: flags for t, flags in
             {**{f"weak:{k}": v for k, v in weak.items()},
              **{f"strong:{k}": v for k, v in strong.items()}}.items()
             if len(set(flags)) > 1}

    strong_cost = sum(mean_tokens("strong", t) for t in tasks) * out_price("strong")
    return {
        "n_tasks": len(tasks),
        "arms": {a: arm_summary(results, tokens, a) for a in ("weak", "strong")},
        "solve_sets": {
            "weak_only": sorted(W - S),
            "strong_only": sorted(S - W),
            "both": sorted(W & S),
            "neither": sorted(set(tasks) - W - S),
        },
        "disjointness": {
            "weak_solves_strong_fails": len(W - S),
            "note": "non-empty is the precondition for any above-all-strong routing curve",
        },
        "oracle": {
            "solved": oracle_solved,
            "cost_usd": round(oracle_cost, 4),
            "cost_per_completed": round(oracle_cost / oracle_solved, 4) if oracle_solved else None,
            "vs_all_strong": round(oracle_cost / strong_cost, 3) if strong_cost else None,
            "routed": routed,
        },
        "stability": {
            "flipping": sorted(flips),
            "n_flipping": len(flips),
            "note": "tasks whose pass flips across replicates carry no routing signal",
        },
    }


def demo():
    """Exercise the arithmetic without model calls, so the analysis is testable offline."""
    reps = ["r1", "r2", "r3"]
    tasks = [f"t{i}" for i in range(6)]
    results, tokens = {}, {}
    for r in reps:
        results[r] = {"weak": {}, "strong": {}}
        tokens[r] = {"weak": {}, "strong": {}}
        for i, t in enumerate(tasks):
            results[r]["weak"][t] = {"passed": i < 3}
            results[r]["strong"][t] = {"passed": i != 0}
            tokens[r]["weak"][t] = 4000
            tokens[r]["strong"][t] = 3000
    return analyse(results, tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=str(DATA / "analysis_stage1.json"))
    args = ap.parse_args()

    if args.demo:
        print(json.dumps(demo(), indent=2))
        return 0

    results, tokens = load("results_stage1*.json"), load("tokens_stage1*.json")
    if not results:
        raise SystemExit("no data/results_stage1*.json yet -- run the arms first")
    out = analyse(results, tokens)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    o, a = out["oracle"], out["arms"]
    print(f"tasks={out['n_tasks']}  weak={a['weak']['solved']}/{a['weak']['attempted']}"
          f"  strong={a['strong']['solved']}/{a['strong']['attempted']}")
    print(f"cost/completed  weak={a['weak']['cost_per_completed']}"
          f"  strong={a['strong']['cost_per_completed']}  oracle={o['cost_per_completed']}")
    print(f"oracle vs all-strong: {o['vs_all_strong']}x   "
          f"weak-only solves: {out['disjointness']['weak_solves_strong_fails']}   "
          f"flipping: {out['stability']['n_flipping']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
