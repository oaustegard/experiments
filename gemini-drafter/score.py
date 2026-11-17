#!/usr/bin/env python3
"""Score arm G against arm B (raw drafts) and G-final against C (after the editor pass).

Arm B's and C's texts and newcomer reads come from ../positive-writer-brief; every judge rating comes from
this experiment's own calls, where B and G texts share one judge prompt. Deterministic (seeded bootstrap).

    python3 score.py            # prints the comparison table, writes results/
"""
from __future__ import annotations

import json
import random
import re
import statistics as st
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PWB = HERE.parent / "positive-writer-brief"
TASKS = json.loads((PWB / "tasks.json").read_text())
TIDS = [t["id"] for t in TASKS]
N_BOOT = 10_000
SEED = 7
JUDGE_SETS = {"claude": ("opus", "sonnet"), "gemini": ("gemini",)}
METRICS = ("register", "focus", "invented")


def lint(path: Path) -> tuple[int, Counter]:
    r = subprocess.run([sys.executable, str(PWB / "context" / "declaude_lint.py"), str(path), "--json"],
                       capture_output=True, text=True)
    seen, cats = set(), Counter()
    for h in json.loads(r.stdout)["hits"]:
        if h.get("line", 0) <= 0:
            continue
        pos = (h["line"], h.get("col"))
        if pos not in seen:
            seen.add(pos)
            cats[h["category"]] += 1
    return len(seen), cats


def words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def text_path(stage, arm, tid, k):
    root = PWB if arm == "B" else HERE
    return root / ("drafts" if stage == "raw" else "finals") / arm / f"{tid}__{k}.md"


def fresh_path(stage, arm, tid, k):
    root = PWB if arm == "B" else HERE
    return root / "scores" / "fresh" / f"{stage}__{arm}__{tid}__{k}.json"


def load_records() -> list[dict]:
    key = json.loads((HERE / "scores" / "judge_key.json").read_text())
    judged = defaultdict(dict)
    for f in sorted((HERE / "scores" / "judge").glob("*.json")):
        jname, stage, tid = f.stem.split("__")
        for r in json.loads(f.read_text())["ratings"]:
            m = key.get(f"{stage}__{tid}__{r['id']}")
            if m:
                judged[(stage, m["arm"], tid, m["k"])][jname] = r
    recs = []
    for stage in ("raw", "final"):
        for arm in ("B", "G"):
            for tid in TIDS:
                for k in range(4):
                    p = text_path(stage, arm, tid, k)
                    if not p.exists():
                        continue
                    n, cats = lint(p)
                    w = words(p.read_text())
                    rec = {"stage": stage, "arm": arm, "task": tid, "k": k, "words": w, "lint_hits": n,
                           "lint_rate": 1000 * n / max(w, 1), "lint_cats": dict(cats)}
                    j = judged.get((stage, arm, tid, k), {})
                    for m in METRICS:
                        for jn in ("opus", "sonnet", "gemini"):
                            rec[f"{m}_{jn}"] = j[jn][m] if jn in j else None
                        for js, members in JUDGE_SETS.items():
                            vals = [j[x][m] for x in members if x in j]
                            rec[f"{m}_{js}"] = st.mean(vals) if len(vals) == len(members) else None
                    fp = fresh_path(stage, arm, tid, k)
                    rec["stumbles"] = len(json.loads(fp.read_text())["stumbles"]) if fp.exists() else None
                    recs.append(rec)
    return recs


def diff_ci(recs, x, y, metric):
    """Mean over tasks of mean(x) - mean(y); x and y are (stage, arm). Two-level bootstrap 95% CI."""
    by = defaultdict(list)
    for r in recs:
        if r.get(metric) is None:
            continue
        for side, (s, a) in (("x", x), ("y", y)):
            if r["stage"] == s and r["arm"] == a:
                by[(r["task"], side)].append(r[metric])
    tasks = [t for t in TIDS if by[(t, "x")] and by[(t, "y")]]
    if not tasks:
        return None

    def est(ts, rng):
        d = []
        for t in ts:
            a, b = by[(t, "x")], by[(t, "y")]
            if rng:
                a = [rng.choice(a) for _ in a]
                b = [rng.choice(b) for _ in b]
            d.append(st.mean(a) - st.mean(b))
        return st.mean(d)

    rng = random.Random(SEED)
    boots = sorted(est([rng.choice(tasks) for _ in tasks], rng) for _ in range(N_BOOT))
    return {"x": round(st.mean(st.mean(by[(t, "x")]) for t in tasks), 3),
            "y": round(st.mean(st.mean(by[(t, "y")]) for t in tasks), 3),
            "diff": round(est(tasks, None), 3),
            "lo": round(boots[int(0.025 * N_BOOT)], 3), "hi": round(boots[int(0.975 * N_BOOT) - 1], 3),
            "n_tasks": len(tasks),
            "tasks_x_higher": sum(st.mean(by[(t, "x")]) > st.mean(by[(t, "y")]) for t in tasks)}


def main() -> int:
    recs = load_records()
    comps = {"raw_G_vs_B": (("raw", "G"), ("raw", "B")),
             "final_Gfinal_vs_C": (("final", "G"), ("final", "B")),
             "Gfinal_vs_rawB": (("final", "G"), ("raw", "B"))}
    metrics = [f"{m}_{js}" for m in METRICS for js in JUDGE_SETS] + \
              [f"{m}_{jn}" for m in ("register", "focus") for jn in ("opus", "sonnet", "gemini")] + \
              ["lint_rate", "stumbles", "words"]
    out = {"n_texts": len(recs), "comparisons": {}, "by_category": {}, "self_preference": {}}
    for cname, (x, y) in comps.items():
        out["comparisons"][cname] = {m: diff_ci(recs, x, y, m) for m in metrics}

    for stage in ("raw", "final"):
        for arm in ("B", "G"):
            c, w = Counter(), 0
            for r in recs:
                if r["stage"] == stage and r["arm"] == arm:
                    c.update(r["lint_cats"])
                    w += r["words"]
            out["by_category"][f"{stage}_{arm}"] = {"words": w, "hits": dict(c.most_common())}

    # Self-preference: how much more the Gemini judge favours G over B than the Claude judges do.
    for cname in ("raw_G_vs_B", "final_Gfinal_vs_C"):
        for m in ("register", "focus"):
            for r in recs:
                g, c = r.get(f"{m}_gemini"), r.get(f"{m}_claude")
                r["_gap"] = None if g is None or c is None else g - c
            out["self_preference"][f"{cname}_{m}"] = diff_ci(recs, *comps[cname], "_gap")
    for r in recs:
        r.pop("_gap", None)

    c = out["comparisons"]
    pre = {"P1": ("raw register G-B (Claude judges)", c["raw_G_vs_B"]["register_claude"], "lo", ">", 0),
           "P2": ("final register Gfinal-C (Claude judges)", c["final_Gfinal_vs_C"]["register_claude"], "lo", ">", 0),
           "P3": ("raw focus G-B (Claude judges)", c["raw_G_vs_B"]["focus_claude"], "lo", ">", -0.5),
           "P4": ("raw invented G-B (Claude judges)", c["raw_G_vs_B"]["invented_claude"], "hi", "<", 0.5)}
    out["predictions"] = {}
    for k, (label, d, bound, op, thr) in pre.items():
        held = None if d is None else (d[bound] > thr if op == ">" else d[bound] < thr)
        out["predictions"][k] = {"metric": label, **(d or {}), "held": held}

    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    with (res / "texts.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    print(f"{len(recs)} texts scored")
    print(f"{'comparison':18s} {'metric':18s} {'x':>7s} {'y':>7s} {'x-y':>7s}  95% CI")
    for cname in comps:
        for m in metrics:
            d = c[cname][m]
            if d:
                print(f"{cname:18s} {m:18s} {d['x']:7.2f} {d['y']:7.2f} {d['diff']:+7.2f}  "
                      f"[{d['lo']:+.2f}, {d['hi']:+.2f}]  ({d['tasks_x_higher']}/{d['n_tasks']} tasks x higher)")
    print("\nSelf-preference gap (Gemini judge minus Claude judges, on x-y):")
    for k, d in out["self_preference"].items():
        if d:
            print(f"  {k:30s} {d['diff']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]")
    print("\nPre-registered predictions")
    for k, v in out["predictions"].items():
        if "diff" in v:
            print(f"  {k} {v['metric']:42s} {v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]  held={v['held']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
