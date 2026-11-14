"""Score results/raw.jsonl: EM / F1 per arm, splits, re-ask effect, paired bootstrap.

    python3 run/score.py      # writes results/scores.json, prints a table
"""
import json
import random
import re
import string
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ARMS = ["CB", "PS", "AG3", "AG4", "PSAG3"]
PAIRS = [("PSAG3", "AG3"), ("PSAG3", "AG4"), ("AG4", "AG3"), ("AG3", "PS"), ("PS", "CB"), ("AG3", "CB")]


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold) -> float:
    if pred is None:
        return 0.0
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def em(pred, gold) -> float:
    return float(pred is not None and norm(pred) == norm(gold))


def best(fn, pred, answers):
    return max(fn(pred, a) for a in answers)


def boot(diffs, n=10000, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)]


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def main():
    qs = {q["id"]: q for q in map(json.loads, (HERE / "data" / "eval.jsonl").open())}
    recs = {}
    errors = Counter()
    for r in map(json.loads, (HERE / "results" / "raw.jsonl").open()):
        if "error" in r:
            errors[r["arm"]] += 1
            continue
        recs[(r["arm"], r["id"])] = r
    arms = [a for a in ARMS if any(k[0] == a for k in recs)]
    ids = [i for i in qs if all((a, i) in recs for a in arms)]  # paired: questions every arm finished

    score = {}
    for a in arms:
        for i in ids:
            r, ans = recs[(a, i)], qs[i]["answers"]
            score[(a, i)] = {"f1": best(f1, r["answer"], ans), "em": best(em, r["answer"], ans),
                             "f1_pre": best(f1, r["answer_pre"], ans)}
    cb_wrong = {i for i in ids if "CB" in arms and score[("CB", i)]["f1"] < 0.5}
    splits = {"all": set(ids), "2hop": {i for i in ids if qs[i]["hops"] == "2hop"},
              "3-4hop": {i for i in ids if qs[i]["hops"] != "2hop"}, "cb_wrong": cb_wrong}

    out = {"n_paired": len(ids), "errors_dropped": dict(errors), "arms": {}, "pairs": {}}
    for a in arms:
        rs = [recs[(a, i)] for i in ids]
        row = {s: {"n": len(m), "f1": mean([score[(a, i)]["f1"] for i in m]),
                   "em": mean([score[(a, i)]["em"] for i in m])} for s, m in splits.items()}
        row["cost_per_q"] = mean([r["cost"] for r in rs])
        row["api_s_per_q"] = mean([r["api_ms"] / 1000 for r in rs])
        row["model_searches_per_q"] = mean([len(r["searches"]) for r in rs])
        gold_hit = [len(set(r["retrieved"]) & set(qs[r["id"]]["gold_paragraphs"])) / len(qs[r["id"]]["gold_paragraphs"])
                    for r in rs]
        row["gold_paragraph_recall"] = mean(gold_hit)
        if a.startswith(("AG", "PSAG")):
            na = [r for r in rs if r["no_answer"]]
            row["reask"] = {
                "no_answer_at_cap": len(na), "no_answer_rate": mean([r["no_answer"] for r in rs]),
                "reask_got_answer": sum(r["answer"] is not None for r in na),
                "f1_before_reask": mean([score[(a, i)]["f1_pre"] for i in ids]),
                "f1_after_reask": row["all"]["f1"],
                "reask_gain_ci": boot([score[(a, i)]["f1"] - score[(a, i)]["f1_pre"] for i in ids]),
                "over_budget_rounds": sum(r["failures"]["over_budget"] for r in rs),
                "no_tag_rounds": sum(r["failures"]["no_tag"] for r in rs),
            }
        out["arms"][a] = row
    for x, y in PAIRS:
        if x in arms and y in arms:
            out["pairs"][f"{x}-{y}"] = {s: {"diff": mean([score[(x, i)]["f1"] - score[(y, i)]["f1"] for i in m]),
                                            "ci": boot([score[(x, i)]["f1"] - score[(y, i)]["f1"] for i in m])}
                                        for s, m in splits.items() if m}
    (HERE / "results" / "scores.json").write_text(json.dumps(out, indent=1))

    print(f"paired n={len(ids)}  dropped errors {dict(errors)}")
    print(f"{'arm':6} {'F1':>6} {'EM':>6} {'2hop':>6} {'3-4hop':>6} {'CBwrong':>7} {'srch':>5} {'goldR':>6} {'$/q':>7} {'api s':>6}")
    for a in arms:
        r = out["arms"][a]
        print(f"{a:6} {r['all']['f1']:6.3f} {r['all']['em']:6.3f} {r['2hop']['f1']:6.3f} {r['3-4hop']['f1']:6.3f} "
              f"{(r['cb_wrong']['f1'] or 0):7.3f} {r['model_searches_per_q']:5.2f} {r['gold_paragraph_recall']:6.3f} "
              f"{r['cost_per_q']:7.4f} {r['api_s_per_q']:6.1f}")
    for a in arms:
        if "reask" in out["arms"][a]:
            print(a, "reask", out["arms"][a]["reask"])
    for k, v in out["pairs"].items():
        print(k, {s: (d["diff"], d["ci"]) for s, d in v.items()})


if __name__ == "__main__":
    main()
