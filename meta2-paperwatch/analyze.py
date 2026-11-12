#!/usr/bin/env python3
"""Metrics: gold vs ledger, gold self-agreement, and the four test conditions against gold.

Per condition and run: exact agreement, linear-weighted kappa, MEDIUM+ false-positive rate
(gold NONE/LOW rated MEDIUM+), MEDIUM+ recall, mean signed error, tokens and cost per paper.
Reported as mean and spread over runs, plus a paired bootstrap over papers for each condition
minus A. Writes results/metrics.json and per-run JSON in results/runs/.
"""
import json, random
from statistics import mean, pstdev

import common as c

R = c.HERE / "results"


def kappa_w(a, b, k=4):
    """Linear-weighted Cohen's kappa for two lists of ordinal ints in 0..k-1."""
    n = len(a)
    if n == 0:
        return None
    w = [[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    po = sum(w[x][y] for x, y in zip(a, b)) / n
    pa = [a.count(i) / n for i in range(k)]
    pb = [b.count(i) / n for i in range(k)]
    pe = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return None if pe == 1 else (po - pe) / (1 - pe)


def score(pred, gold):
    """pred, gold: dict pid -> rating. Missing predictions count as wrong and are reported."""
    ids = list(gold)
    have = [p for p in ids if pred.get(p)]
    g = [c.ORD[gold[p]] for p in have]
    y = [c.ORD[pred[p]] for p in have]
    neg = [p for p in ids if c.ORD[gold[p]] <= 1]
    pos = [p for p in ids if c.ORD[gold[p]] >= 2]
    return {"n": len(ids), "missing": len(ids) - len(have),
            "exact": sum(pred.get(p) == gold[p] for p in ids) / len(ids),
            "kappa_w": kappa_w(g, y),
            "fp_medium_plus": (sum(1 for p in neg if pred.get(p) and c.ORD[pred[p]] >= 2) / len(neg)) if neg else None,
            "recall_medium_plus": (sum(1 for p in pos if pred.get(p) and c.ORD[pred[p]] >= 2) / len(pos)) if pos else None,
            "signed_err": mean(yy - gg for yy, gg in zip(y, g)) if have else None,
            "n_gold_neg": len(neg), "n_gold_pos": len(pos)}


def confusion(pred, gold):
    m = {r: {s: 0 for s in c.RATINGS} for r in c.RATINGS}  # rows gold, cols pred
    for p, g in gold.items():
        if pred.get(p):
            m[g][pred[p]] += 1
    return m


def agg(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": mean(vals), "min": min(vals), "max": max(vals), "sd": pstdev(vals)} if vals else None


def boot_diff(per_paper_a, per_paper_b, iters=10000, seed=0):
    """Paired bootstrap over papers of mean(b - a), each a per-paper mean over runs."""
    rng = random.Random(seed)
    ids = list(per_paper_a)
    d = [per_paper_b[p] - per_paper_a[p] for p in ids]
    obs = mean(d)
    bs = sorted(mean(rng.choice(d) for _ in ids) for _ in range(iters))
    return {"diff": obs, "ci95": [bs[int(0.025 * iters)], bs[int(0.975 * iters)]]}


def main():
    gold_full = json.load(open(c.HERE / "gold.json"))
    gold = {p: v["practical"] for p, v in gold_full.items() if v}
    led = c.ledger()
    out = {}

    # gold vs ledger, all papers and by ledger stratum
    ledger_pred = {p: led[p]["practical"] for p in gold}
    hi = {p: g for p, g in gold.items() if c.ORD[led[p]["practical"]] >= 2}
    lo = {p: g for p, g in gold.items() if c.ORD[led[p]["practical"]] <= 1}
    out["gold_vs_ledger"] = {"all": score(ledger_pred, gold), "ledger_medium_plus": score(ledger_pred, hi),
                             "ledger_low_none": score(ledger_pred, lo),
                             "confusion_gold_rows_ledger_cols": confusion(ledger_pred, gold)}
    p1 = {p: v["passes"][0] for p, v in gold_full.items() if v and len(v["passes"]) >= 2}
    p2 = {p: v["passes"][1] for p, v in gold_full.items() if v and len(v["passes"]) >= 2}
    out["gold_self_agreement"] = score(p1, p2)
    out["gold_distribution"] = {r: list(gold.values()).count(r) for r in c.RATINGS}

    offline, test = c.split()
    tgold = {p: gold[p] for p in test if p in gold}
    out["test_n"] = len(tgold)
    out["point_value_pp"] = 100 / len(tgold)
    out["test_gold_distribution"] = {r: list(tgold.values()).count(r) for r in c.RATINGS}
    out["always_LOW_on_test"] = score({p: "LOW" for p in tgold}, tgold)
    out["ledger_on_test"] = score({p: led[p]["practical"] for p in tgold}, tgold)

    runs_dir = c.HERE / "cache" / "runs"
    per_paper_exact = {}
    out["conditions"] = {}
    for cond in "ABCD":
        runs = []
        for k in (1, 2, 3):
            recs = [json.loads((runs_dir / cond / f"r{k}" / f"{p}.json").read_text())
                    for p in tgold if (runs_dir / cond / f"r{k}" / f"{p}.json").exists()]
            if len(recs) < len(tgold):
                continue
            pred = {r["pid"]: r["final_rating"] for r in recs}
            s = score(pred, tgold)
            s["tokens_per_paper"] = mean(r["tokens"] for r in recs)
            s["cost_per_paper_usd"] = mean(r["cost_usd"] for r in recs)
            if cond in "CD":
                ev = [r for r in recs if r["decision"] == "EVOLVE"]
                s["evolve_rate"] = len(ev) / len(recs)
                s["evolver_failed"] = sum(1 for r in recs if r["decision"] is None)
                moved = [r for r in ev if r["draft_rating"] and r["final_rating"] and r["draft_rating"] != r["final_rating"]]
                s["evolve_changed_rating"] = len(moved)
                s["evolve_toward_gold"] = sum(1 for r in moved if abs(c.ORD[r["final_rating"]] - c.ORD[tgold[r["pid"]]])
                                              < abs(c.ORD[r["draft_rating"]] - c.ORD[tgold[r["pid"]]]))
                s["evolve_away_from_gold"] = sum(1 for r in moved if abs(c.ORD[r["final_rating"]] - c.ORD[tgold[r["pid"]]])
                                                 > abs(c.ORD[r["draft_rating"]] - c.ORD[tgold[r["pid"]]]))
                s["draft_exact"] = score({r["pid"]: r["draft_rating"] for r in recs}, tgold)["exact"]
            if cond == "D":
                s["standing_instructions_at_end"] = len(recs and max((r["standing_after"] for r in recs), key=len))
            s["confusion_gold_rows_pred_cols"] = confusion(pred, tgold)
            runs.append(s)
            c.save(R / "runs" / f"{cond}_r{k}.json",
                   {"cond": cond, "run": k, "metrics": s,
                    "papers": [{k2: r[k2] for k2 in ("pid", "draft_rating", "final_rating", "decision", "evolver",
                                                     "standing_before", "standing_after", "tokens", "cost_usd")}
                               | {"gold": tgold[r["pid"]],
                                  "draft_text": (r["draft"] or {}).get("text"),
                                  "final_text": ((r["sendback"] or r["draft"]) or {}).get("text")}
                               for r in recs]})
            for r in recs:
                per_paper_exact.setdefault(cond, {}).setdefault(r["pid"], []).append(r["final_rating"] == tgold[r["pid"]])
        if not runs:
            continue
        keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float)) and not isinstance(runs[0][k], bool)]
        out["conditions"][cond] = {"runs": len(runs), **{k: agg([r[k] for r in runs]) for k in keys}}

    ppe = {cond: {p: mean(v) for p, v in d.items()} for cond, d in per_paper_exact.items()}
    out["paired_vs_A_exact"] = {cond: boot_diff(ppe["A"], ppe[cond]) for cond in ppe if cond != "A" and "A" in ppe}
    if "B" in ppe and "C" in ppe:
        out["paired_C_vs_B_exact"] = boot_diff(ppe["B"], ppe["C"])
    if "C" in ppe and "D" in ppe:
        out["paired_D_vs_C_exact"] = boot_diff(ppe["C"], ppe["D"])
    c.save(R / "metrics.json", out)

    print(f"test n={len(tgold)}  one paper = {out['point_value_pp']:.1f} pp  gold dist {out['test_gold_distribution']}")
    print("gold vs ledger (all):", {k: round(v, 3) if isinstance(v, float) else v for k, v in out['gold_vs_ledger']['all'].items()})
    print("always LOW:", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["always_LOW_on_test"].items()})
    print("ledger on test:", {k: round(v, 3) if isinstance(v, float) else v for k, v in out["ledger_on_test"].items()})
    print("gold self-agreement:", round(out["gold_self_agreement"]["exact"], 3), "kappa", round(out["gold_self_agreement"]["kappa_w"], 3))
    hdr = ["exact", "kappa_w", "fp_medium_plus", "recall_medium_plus", "signed_err", "tokens_per_paper"]
    print("cond " + " ".join(f"{h:>22}" for h in hdr))
    for cond, d in out["conditions"].items():
        print(f"{cond:4} " + " ".join(f"{d[h]['mean']:>9.3f} [{d[h]['min']:.2f},{d[h]['max']:.2f}]".rjust(22)
                                       if d.get(h) else " " * 22 for h in hdr))
    for k, v in out.get("paired_vs_A_exact", {}).items():
        print(f"{k} - A exact: {v['diff']:+.3f} CI {v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}")
    for k in ("paired_C_vs_B_exact", "paired_D_vs_C_exact"):
        if k in out:
            print(k, f"{out[k]['diff']:+.3f} CI {out[k]['ci95'][0]:+.3f},{out[k]['ci95'][1]:+.3f}")


if __name__ == "__main__":
    main()
