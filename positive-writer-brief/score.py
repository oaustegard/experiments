#!/usr/bin/env python3
"""Score every draft and final, compare arms, write results/summary.json.

Reads drafts/, finals/, scores/judge/, scores/fresh/ and scores/judge_key.json;
runs context/declaude_lint.py on each text. Deterministic: the bootstrap is
seeded, so a rerun prints the same numbers.

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
TASKS = json.loads((HERE / "tasks.json").read_text())
TRAP = {t["id"]: t["trap"] for t in TASKS}
N_BOOT = 10_000
SEED = 7
MARGIN = -0.5  # non-inferiority margin for focus, pre-registered


def lint(path: Path) -> tuple[int, Counter]:
    r = subprocess.run([sys.executable, str(HERE / "context" / "declaude_lint.py"), str(path), "--json"],
                       capture_output=True, text=True)
    hits = json.loads(r.stdout)["hits"]
    seen, cats = set(), Counter()
    for h in hits:
        if h.get("line", 0) <= 0:
            continue  # document-level density notes
        pos = (h["line"], h.get("col"))
        if pos in seen:
            continue
        seen.add(pos)
        cats[h["category"]] += 1
    return len(seen), cats


def words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def load_records() -> list[dict]:
    key = json.loads((HERE / "scores" / "judge_key.json").read_text())
    judged = defaultdict(dict)  # (stage, arm, task, k) -> {judge: rating}
    for f in sorted((HERE / "scores" / "judge").glob("*.json")):
        jname, stage, tid = f.stem.split("__")
        for r in json.loads(f.read_text())["ratings"]:
            m = key.get(f"{stage}__{tid}__{r['id']}")
            if m:
                judged[(stage, m["arm"], tid, m["k"])][jname] = r
    recs = []
    for stage, root in (("raw", "drafts"), ("final", "finals")):
        for arm in ("A", "B"):
            for p in sorted((HERE / root / arm).glob("*.md")):
                tid, k = p.stem.split("__")
                k = int(k)
                text = p.read_text()
                n, cats = lint(p)
                w = words(text)
                rec = {"stage": stage, "arm": arm, "task": tid, "k": k, "words": w,
                       "lint_hits": n, "lint_rate": 1000 * n / max(w, 1), "lint_cats": dict(cats)}
                j = judged.get((stage, arm, tid, k), {})
                for m in ("register", "focus", "invented"):
                    vals = [j[x][m] for x in sorted(j)]
                    rec[m] = st.mean(vals) if vals else None
                    for x in sorted(j):
                        rec[f"{m}_{x}"] = j[x][m]
                fp = HERE / "scores" / "fresh" / f"{stage}__{arm}__{tid}__{k}.json"
                rec["stumbles"] = len(json.loads(fp.read_text())["stumbles"]) if fp.exists() else None
                recs.append(rec)
    return recs


def diff_ci(recs, stage_b, stage_a, metric, tasks=None, rng_seed=SEED):
    """Mean over tasks of (mean B in stage_b) - (mean A in stage_a); two-level bootstrap CI."""
    tasks = tasks or [t["id"] for t in TASKS]
    by = defaultdict(list)
    for r in recs:
        if r[metric] is None:
            continue
        if r["stage"] == stage_b and r["arm"] == "B":
            by[(r["task"], "x")].append(r[metric])
        if r["stage"] == stage_a and r["arm"] == "A":
            by[(r["task"], "y")].append(r[metric])
    tasks = [t for t in tasks if by[(t, "x")] and by[(t, "y")]]

    def est(ts, resample):
        d = []
        for t in ts:
            x, y = by[(t, "x")], by[(t, "y")]
            if resample:
                x = [resample.choice(x) for _ in x]
                y = [resample.choice(y) for _ in y]
            d.append(st.mean(x) - st.mean(y))
        return st.mean(d)

    point = est(tasks, None)
    rng = random.Random(rng_seed)
    boots = sorted(est([rng.choice(tasks) for _ in tasks], rng) for _ in range(N_BOOT))
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
    a_mean = st.mean(st.mean(by[(t, "y")]) for t in tasks)
    b_mean = st.mean(st.mean(by[(t, "x")]) for t in tasks)
    wins = sum(1 for t in tasks if st.mean(by[(t, "x")]) < st.mean(by[(t, "y")]))
    return {"a": round(a_mean, 3), "b": round(b_mean, 3), "diff": round(point, 3),
            "lo": round(lo, 3), "hi": round(hi, 3), "n_tasks": len(tasks),
            "tasks_b_lower": wins}


def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else None


def main() -> int:
    recs = load_records()
    out = {"n_texts": len(recs), "comparisons": {}, "by_category": {}, "judges": {}}
    comps = {
        "raw_B_vs_A": ("raw", "raw"),          # P1, P2, P5
        "final_C_vs_Afinal": ("final", "final"),  # P3, P4
        "rawB_vs_Afinal": ("raw", "final"),    # exploratory: brief alone vs current full pipeline
    }
    trap_sets = {
        "all": None,
        "focus_traps": [t for t, v in TRAP.items() if v != "none"],
        "change_traps": [t for t, v in TRAP.items() if v == "change"],
        "author_traps": [t for t, v in TRAP.items() if v == "author"],
        "no_trap": [t for t, v in TRAP.items() if v == "none"],
    }
    for cname, (sb, sa) in comps.items():
        out["comparisons"][cname] = {}
        for sname, ts in trap_sets.items():
            out["comparisons"][cname][sname] = {
                m: diff_ci(recs, sb, sa, m, ts)
                for m in ("lint_rate", "register", "focus", "stumbles", "invented", "words")}

    # lint categories, summed per arm and stage
    for stage in ("raw", "final"):
        for arm in ("A", "B"):
            c = Counter()
            w = 0
            for r in recs:
                if r["stage"] == stage and r["arm"] == arm:
                    c.update(r["lint_cats"])
                    w += r["words"]
            out["by_category"][f"{stage}_{arm}"] = {"words": w, "hits": dict(c.most_common())}

    # judge agreement
    for m in ("register", "focus"):
        pairs = [(r[f"{m}_opus"], r[f"{m}_sonnet"]) for r in recs
                 if r.get(f"{m}_opus") is not None and r.get(f"{m}_sonnet") is not None]
        if pairs:
            out["judges"][m] = {"pearson": round(corr(*zip(*pairs)), 3), "n": len(pairs)}
            # per-judge raw B - A, so a reader can see whether both judges agree on direction
            for jn in ("opus", "sonnet"):
                for r in recs:
                    r["_j"] = r.get(f"{m}_{jn}")
                out["judges"][m][f"{jn}_raw_B_vs_A"] = diff_ci(recs, "raw", "raw", "_j")
                out["judges"][m][f"{jn}_final_C_vs_Afinal"] = diff_ci(recs, "final", "final", "_j")
    for r in recs:
        r.pop("_j", None)

    c = out["comparisons"]
    pre = {
        "P1": {"metric": "raw lint_rate B-A", **c["raw_B_vs_A"]["all"]["lint_rate"]},
        "P2": {"metric": "raw focus B-A", **c["raw_B_vs_A"]["all"]["focus"]},
        "P3": {"metric": "final lint_rate C-Afinal", **c["final_C_vs_Afinal"]["all"]["lint_rate"]},
        "P4": {"metric": "final focus C-Afinal", **c["final_C_vs_Afinal"]["all"]["focus"]},
        "P5": {"metric": "raw stumbles B-A", **c["raw_B_vs_A"]["all"]["stumbles"]},
    }
    pre["P1"]["held"] = pre["P1"]["hi"] < 0
    pre["P2"]["held"] = pre["P2"]["lo"] > MARGIN
    pre["P3"]["held"] = pre["P3"]["hi"] < 0
    pre["P4"]["held"] = pre["P4"]["lo"] > MARGIN
    pre["P5"]["held"] = None  # exploratory, direction only
    out["predictions"] = pre

    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    with (res / "texts.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    print(f"{len(recs)} texts scored")
    print(f"{'comparison':22s} {'subset':13s} {'metric':10s} {'A':>7s} {'B':>7s} {'B-A':>7s}  95% CI")
    for cname in comps:
        for sname in trap_sets:
            for m in ("lint_rate", "register", "focus", "stumbles", "invented", "words"):
                d = c[cname][sname][m]
                if sname != "all" and m in ("invented", "words"):
                    continue
                print(f"{cname:22s} {sname:13s} {m:10s} {d['a']:7.2f} {d['b']:7.2f} {d['diff']:+7.2f}"
                      f"  [{d['lo']:+.2f}, {d['hi']:+.2f}]  ({d['tasks_b_lower']}/{d['n_tasks']} tasks B lower)")
    print("\nPre-registered predictions")
    for k, v in pre.items():
        print(f"  {k} {v['metric']:26s} {v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]  held={v['held']}")
    print("\nJudge agreement (Pearson, opus vs sonnet):",
          {m: v["pearson"] for m, v in out["judges"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
