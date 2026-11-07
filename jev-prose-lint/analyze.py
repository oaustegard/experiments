#!/usr/bin/env python3
"""Judge agreement, gold labels, and Jev arms against gold.  python3 analyze.py > results/metrics.json"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "data"


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def coverage(scores, labels, budget=0.05):
    """Largest share of items settled by two thresholds (P<=lo -> legitimate, P>=hi -> tic)
    with at most `budget` of the settled items wrong."""
    best = (0.0, None, None)
    cuts = sorted(set([0.0, 1.0] + scores))
    for lo in cuts:
        for hi in cuts:
            if hi <= lo:
                continue
            settled = [(s >= hi, l) for s, l in zip(scores, labels) if s <= lo or s >= hi]
            if not settled:
                continue
            err = sum(p != l for p, l in settled) / len(settled)
            if err <= budget and len(settled) / len(scores) > best[0]:
                best = (len(settled) / len(scores), lo, hi)
    return {"coverage": round(best[0], 3), "lo": best[1], "hi": best[2]}


def main():
    items = {i["id"]: i for i in load(D / "items.jsonl")}
    judge = {}
    for name in ("opus", "sonnet"):
        rows = [r for b in range(4) for r in load(D / f"judges/{name}_{b}.jsonl")]
        judge[name] = {r["id"]: r["label"] == "tic" for r in rows}
    ids = [i for i in items if i in judge["opus"] and i in judge["sonnet"]]
    o = [judge["opus"][i] for i in ids]
    s = [judge["sonnet"][i] for i in ids]
    gold = {i: judge["opus"][i] for i in ids if judge["opus"][i] == judge["sonnet"][i]}
    out = {"items": len(items), "judged_by_both": len(ids),
           "judges": {"agreement": round(sum(x == y for x, y in zip(o, s)) / len(ids), 3),
                      "kappa": round(kappa(o, s), 3),
                      "tic_rate_opus": round(sum(o) / len(o), 3), "tic_rate_sonnet": round(sum(s) / len(s), 3)},
           "gold": {"n": len(gold), "tic_rate": round(sum(gold.values()) / len(gold), 3)}}
    rev = collections.defaultdict(list)
    for i, g in gold.items():
        rev[items[i]["revision"]].append(g)
    out["gold"]["tic_rate_by_revision"] = {k: f"{sum(v)}/{len(v)}" for k, v in rev.items()}

    gids = list(gold)
    gl = [gold[i] for i in gids]
    cats = [items[i]["category"] for i in gids]
    # per-category prior, leave-one-out
    prior = []
    for k, i in enumerate(gids):
        same = [gold[j] for j in gids if j != i and items[j]["category"] == cats[k]]
        prior.append(sum(same) / len(same) if same else sum(gl) / len(gl))
    maj = max(sum(gl), len(gl) - sum(gl)) / len(gl)
    out["baselines"] = {"majority_accuracy": round(maj, 3),
                        "category_prior_auroc": round(auroc(prior, gl), 3),
                        "category_prior_accuracy": round(sum((p >= 0.5) == l for p, l in zip(prior, gl)) / len(gl), 3)}
    # judges against each other's view of "gold" is circular; report each judge vs the other on all items
    for arm in ("A", "B"):
        runs = [{r["id"]: r["p"] for r in load(D / f"jev_{arm}_{k}.jsonl")} for k in (1, 2)]
        p = [(runs[0][i] + runs[1][i]) / 2 for i in gids]
        pred = [x >= 0.5 for x in p]
        tp = sum(a and b for a, b in zip(pred, gl))
        d = [abs(runs[0][i] - runs[1][i]) for i in items]
        cross = sum((runs[0][i] >= 0.5) != (runs[1][i] >= 0.5) for i in items)
        bycat = collections.defaultdict(list)
        for x, l, c in zip(pred, gl, cats):
            bycat[c].append(x == l)
        out[f"jev_{arm}"] = {
            "auroc": round(auroc(p, gl), 3),
            "accuracy@0.5": round(sum(x == l for x, l in zip(pred, gl)) / len(gl), 3),
            "kappa@0.5": round(kappa(pred, gl), 3),
            "precision_tic": round(tp / sum(pred), 3) if sum(pred) else None,
            "recall_tic": round(tp / sum(gl), 3) if sum(gl) else None,
            "predicted_tic_rate": round(sum(pred) / len(pred), 3),
            "coverage@5%": coverage(p, gl),
            "repeat_mean_abs_delta": round(sum(d) / len(d), 4),
            "repeat_cross_0.5": f"{cross}/{len(items)}",
            "accuracy_by_category": {c: f"{sum(v)}/{len(v)}" for c, v in
                                     sorted(bycat.items(), key=lambda kv: -len(kv[1])) if len(v) >= 5},
            "auroc_vs_opus_all": round(auroc([(runs[0][i] + runs[1][i]) / 2 for i in ids], o), 3),
            "auroc_vs_sonnet_all": round(auroc([(runs[0][i] + runs[1][i]) / 2 for i in ids], s), 3),
        }
    # each judge's agreement with the other as a reference ceiling, by category
    jc = collections.defaultdict(list)
    for i, x, y in zip(ids, o, s):
        jc[items[i]["category"]].append(x == y)
    out["judge_agreement_by_category"] = {c: f"{sum(v)}/{len(v)}" for c, v in
                                          sorted(jc.items(), key=lambda kv: -len(kv[1])) if len(v) >= 5}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
