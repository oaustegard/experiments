"""Score the life-science decision models against gold, on the items where Sonnet and Opus agree.

    python3 run/score.py      # results/b/ls/*_preds.json, results/b/ls_spikes/*_preds.json -> results/decisions.json

Per question: accuracy of each model's top answer on agreed items, beside Haiku and Jev (the two
candidate teachers) and the most-common-label rate. `role` is scored against the article's heading.
`immunoassay` is 98% "no", so it also gets AUROC. Size steps come with a paired bootstrap over
sections (10,000 resamples) on the questions the teacher labelled.
"""
import glob
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suite import ALL_QS, options

ROOT = Path(__file__).resolve().parents[1]
QS = ["role", "study_type", "field", "immunoassay", "human_samples", "quantitative_result", "detail", "evidence", "relevance"]


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    gold = json.loads((ROOT / "data" / "gold.json").read_text())
    truth = {}                                   # (case, question) -> agreed label
    agree = {}
    for sid, g in gold["sections"].items():
        for q in QS[:-1]:
            if q == "role":
                if g["heading_role"]:
                    truth[(sid, q)] = g["heading_role"]
                continue
            same = g["sonnet"][q] == g["opus"][q]
            agree.setdefault(q, []).append(same)
            if same:
                truth[(sid, q)] = g["opus"][q]
    for pid, g in gold["pairs"].items():
        same = g["sonnet"] == g["opus"]
        agree.setdefault("relevance", []).append(same)
        if same:
            truth[(pid, "relevance")] = g["opus"]

    def table(preds):
        """preds: {(case, q): label} -> per-question accuracy on agreed items, and the per-item hits."""
        out, hits = {}, {}
        for q in QS:
            keys = [k for k in truth if k[1] == q and k in preds]
            if keys:
                h = {k: preds[k] == truth[k] for k in keys}
                hits.update(h)
                out[q] = {"acc": sum(h.values()) / len(keys), "n": len(keys)}
        return out, hits

    res = {"agreement": {q: sum(v) / len(v) for q, v in agree.items()},
           "prior": {}, "models": {}, "n_agreed": Counter(k[1] for k in truth)}
    for q in QS:
        c = Counter(v for k, v in truth.items() if k[1] == q)
        res["prior"][q] = c.most_common(1)[0][1] / sum(c.values())

    # the labellers themselves, on the gold sections
    def lab(v):
        return str(v).lower() if isinstance(v, bool) else str(v)
    with open(ROOT / "data" / "gold_haiku.jsonl") as f:
        hk = {(r["id"], q): lab(r[q]) for r in map(json.loads, f) for q in QS[:-1]}
    jev = {}
    with open(ROOT / "out" / "jev_sections.jsonl") as f:
        for r in map(json.loads, f):
            if r["id"] in gold["sections"] and "probs" in r:
                for q, p in r["probs"].items():
                    jev[(r["id"], q)] = max(p, key=p.get)
    res["models"]["labeller-haiku"], _ = table(hk)
    res["models"]["labeller-jev"], _ = table(jev)

    allhits, imm = {}, {}
    files = sorted(glob.glob(str(ROOT / "results/b/ls/b-*_preds.json"))) + sorted(glob.glob(str(ROOT / "results/b/ls_spikes/spike-*_preds.json")))
    for f in files:
        name = Path(f).name.replace("_preds.json", "")
        preds, scores = {}, {}
        for r in json.loads(Path(f).read_text()):
            opts = options(ALL_QS[r["q"]])
            preds[(r["case"], r["q"])] = opts[max(range(len(opts)), key=lambda i: r["p"][i])]
            if r["q"] == "immunoassay":
                scores[r["case"]] = r["p"][0]
        t, hits = table(preds)
        res["models"][name] = t
        allhits[name] = hits
        ks = [k for k in truth if k[1] == "immunoassay" and k[0] in scores]
        if ks:
            imm[name] = auroc([scores[k[0]] for k in ks], [truth[k] == "true" for k in ks])
    res["immunoassay_auroc"] = imm

    # spikes are per group: merge the three groups of one seed into one row
    for seed in (0, 1):
        parts = [res["models"].get(f"spike-{g}-s{seed}") for g in ("classify", "rate", "relevance")]
        if all(parts):
            res["models"][f"spikes68-s{seed}"] = {q: v for p in parts for q, v in p.items()}
            allhits[f"spikes68-s{seed}"] = {k: v for g in ("classify", "rate", "relevance") for k, v in allhits[f"spike-{g}-s{seed}"].items()}

    def macro(t, qs):
        return st.mean(t[q]["acc"] for q in qs if q in t)
    TEACHER_QS = [q for q in QS if q != "role"]
    for t in res["models"].values():
        t["macro_all"] = macro(t, QS)
        t["macro_teacher_questions"] = macro(t, TEACHER_QS)

    # size means over seeds, and paired bootstrap of size steps (seed-averaged hits per item)
    sizes = {}
    for s in ("17", "32", "68"):
        names = [n for n in allhits if n.startswith(f"b-{s}-haiku-s")]
        if names:
            sizes[s] = {"seeds": len(names), "macro_all": st.mean(res["models"][n]["macro_all"] for n in names),
                        "per_question": {q: st.mean(res["models"][n][q]["acc"] for n in names) for q in QS},
                        "sd_macro": st.pstdev(res["models"][n]["macro_all"] for n in names)}
    res["sizes"] = sizes
    rng = random.Random(0)

    def item_mean(s):
        names = [n for n in allhits if n.startswith(f"b-{s}-haiku-s")]
        keys = set.intersection(*(set(allhits[n]) for n in names))
        return {k: st.mean(allhits[n][k] for n in names) for k in keys}
    boots = {}
    for a, b in (("17", "32"), ("32", "68"), ("17", "68")):
        if a in sizes and b in sizes:
            ha, hb = item_mean(a), item_mean(b)
            keys = sorted(set(ha) & set(hb))
            by_q = {q: [k for k in keys if k[1] == q] for q in QS}

            def stat(sample):
                return st.mean(st.mean(hb[k] - ha[k] for k in sample[q]) for q in QS if sample[q])
            point = stat(by_q)
            draws = sorted(stat({q: [rng.choice(v) for _ in v] for q, v in by_q.items()}) for _ in range(2000))
            boots[f"{b}_minus_{a}"] = {"macro_delta": point, "ci95": [draws[50], draws[1949]]}
    res["size_steps"] = boots
    (ROOT / "results" / "decisions.json").write_text(json.dumps(res, indent=1) + "\n")
    cols = ["labeller-haiku", "labeller-jev"] + [n for n in res["models"] if n.startswith(("b-", "spikes68"))]
    print("question            agree prior | " + " ".join(f"{c[-12:]:>12s}" for c in cols))
    for q in QS:
        print(f"{q:19s} {res['agreement'].get(q, float('nan')):.3f} {res['prior'][q]:.3f} | "
              + " ".join(f"{res['models'][c][q]['acc']:12.3f}" if q in res["models"][c] else " " * 12 for c in cols))
    print(f"{'macro (all)':31s} | " + " ".join(f"{res['models'][c]['macro_all']:12.3f}" for c in cols))
    print("sizes", json.dumps({s: round(v["macro_all"], 4) for s, v in sizes.items()}), "steps", json.dumps(boots))
    print("immunoassay AUROC", {k: round(v, 3) for k, v in imm.items() if v is not None})


if __name__ == "__main__":
    main()
