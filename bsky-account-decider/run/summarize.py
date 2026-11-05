"""Aggregate the account decider's arms into results/summary.json.

    python3 -I run/summarize.py

Per question: fine-tuned accuracy against Sonnet (mean and SD over seeds) for every training-set size,
the majority and frozen-probe baselines, the oracle's self-agreement, and bot precision/recall
(computed from the per-decision predictions against the test cases' Sonnet labels).
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
QS = ["account_type", "ai_share", "ai_role", "ai_stance", "original_content", "about_ai"]


def load(n):
    p = RES / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def agg(v):
    v = [x for x in v if x is not None]
    return {"mean": round(st.mean(v), 4), "sd": round(st.stdev(v), 4) if len(v) > 1 else None, "n": len(v),
            "values": [round(x, 4) for x in v]} if v else None


def qacc(r, q):
    k = [k for k in r["by"] if k.startswith("q=") and k.endswith("/" + q)]
    return r["by"][k[0]]["accuracy"] if k else None


def main():
    labels = json.loads((RES / "labels.json").read_text())
    base = json.loads((RES / "baselines.json").read_text())
    out = {"arms": {}, "baselines": base, "oracle_self_agreement": {}, "train_accounts": {}}
    for q in QS:
        sa = labels["self_agreement"]["items_about_ai" if q == "about_ai" else q]
        out["oracle_self_agreement"][q] = sa["argmax"]
    for arm in ("all", "n100", "n300", "n700"):
        runs = [r for r in (load(f"{arm}-s{s}") for s in range(3)) if r]
        if not runs:
            continue
        out["arms"][arm] = {q: agg([qacc(r, q) for r in runs]) for q in QS}
        out["arms"][arm]["overall"] = agg([r["test"]["accuracy"] for r in runs])
        out["arms"][arm]["account_mean"] = agg([st.mean(qacc(r, q) for q in QS[:5]) for r in runs])
        out["arms"][arm]["wall_min"] = agg([r["wall_sec"] / 60 for r in runs])
        out["arms"][arm]["best_epoch"] = [r["best_epoch"] for r in runs]
    out["train_accounts"] = {"all": labels["cases"]["account/train"], "n100": 100, "n300": 300, "n700": 700}
    out["baseline_account_mean"] = {
        k: round(st.mean(base[k][q]["accuracy"] for q in QS[:5]), 4) for k in ("majority", "probe")}
    # bot precision / recall per seed from predictions, against Sonnet's argmax on the test accounts
    cases = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data" / "cases.jsonl").open()}
    order = list(json.loads((ROOT / "run" / "questions.json").read_text())["account_type"]["criteria"])
    bots = []
    for s in range(3):
        p = RES / f"all-s{s}_preds.json"
        if not p.exists():
            continue
        tp = fp = fn = 0
        for d in json.loads(p.read_text()):
            if d["q"] != "account_type":
                continue
            g = cases[d["case"]]["gold"]["account_type"]["probabilities"]
            gold = max(g, key=g.get)
            pred = order[max(range(len(d["p"])), key=lambda i: d["p"][i])]
            tp += pred == "bot" and gold == "bot"
            fp += pred == "bot" and gold != "bot"
            fn += pred != "bot" and gold == "bot"
        bots.append({"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else None,
                     "recall": tp / (tp + fn) if tp + fn else None})
    out["bot_detection"] = {"per_seed": bots, "test_bots": bots[0]["tp"] + bots[0]["fn"] if bots else None}
    packed = load("packed_all-s0")
    if packed:
        out["packed_all-s0"] = {"overall": packed["overall"]["accuracy"],
                                "by_question": {q: v["accuracy"] for q, v in packed["by_question"].items()}}
    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({a: {q: (v[q]["mean"] if v.get(q) else None) for q in QS + ["account_mean"]} for a, v in out["arms"].items()}, indent=1))
    print("bots", out["bot_detection"])


if __name__ == "__main__":
    main()
