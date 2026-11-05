"""Turn the oracle's labels into training cases for browser-decider's trainer (--data).

    python3 -I run/build_data.py

Writes data/cases.jsonl. Two kinds of case, both in the Typed Decisions shape (id, workflow, state,
questions, gold, split, account):
  workflow "account": the fitted account state and the five questions of run/questions.json;
  workflow "post":    one feed item (run/state.py post_state) and the noul "this post is about AI".
Test = the 150 accounts of the oracle's second pass plus 150 more drawn with a fixed seed, all their
cases held out together. Training keeps up to POSTS_PER_ACCOUNT items per account (all 12 in test).
Also writes results/labels.json: label distribution per question and the oracle's self-agreement.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state import post_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Q = json.loads((ROOT / "run" / "questions.json").read_text())
POST_Q = {"about_ai": {"type": "noul", "instructions": "This post is about AI.", "criteria": {
    "true": "Its subject is artificial intelligence: machine learning, LLMs or chatbots, AI products, companies, "
            "research, policy, safety, art, jobs or energy use, or using AI tools.",
    "false": "It is about something else, or only mentions tech or science without AI, or its topic cannot be told."}}}
POSTS_PER_ACCOUNT = 6


def load(pas):
    out = {}
    for f in sorted((ROOT / "data" / f"labels{pas}").glob("b*.json")):
        d = json.loads(f.read_text())
        for aid, lab in d["labels"].items():
            out[aid] = (lab, d["states"][aid])
    return out


def argmax(d):
    return max(d, key=d.get)


def main():
    accts = {json.loads(l)["did"]: json.loads(l) for l in (ROOT / "data" / "accounts.jsonl").open()}
    L1, L2 = load(1), load(2)
    rng = random.Random(7)
    test = set(L2) & set(L1)
    rest = sorted(set(L1) - test)
    rng.shuffle(rest)
    test |= set(rest[: 300 - len(test)])
    recs = []
    for aid, (lab, st) in L1.items():
        split = "test" if aid in test else "train"
        gold = {n: {"type": Q[n]["type"], "probabilities": lab[n]} for n in Q}
        recs.append({"id": aid, "account": aid, "workflow": "account", "state": st, "questions": Q,
                     "gold": gold, "split": split, "source": accts[aid]["source"]})
        items = list(lab["items_about_ai"].items())
        if split == "train" and len(items) > POSTS_PER_ACCOUNT:
            items = rng.sample(items, POSTS_PER_ACCOUNT)
        for k, p in items:
            it = accts[aid]["items"][int(k) - 1]
            recs.append({"id": f"{aid}#{k}", "account": aid, "workflow": "post", "state": post_state(it),
                         "questions": POST_Q, "gold": {"about_ai": {"type": "noul", "probabilities": {"true": p, "false": 1 - p}}},
                         "split": split, "source": accts[aid]["source"]})
    with (ROOT / "data" / "cases.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    # label statistics and oracle self-agreement (argmax, and mean total-variation distance)
    stats = {"accounts": len(L1), "test_accounts": len(test), "cases": Counter(f"{r['workflow']}/{r['split']}" for r in recs),
             "distribution": {}, "self_agreement": {}}
    for n in Q:
        stats["distribution"][n] = dict(Counter(argmax(L1[a][0][n]) for a in L1).most_common())
        by_src = {}
        for a in L1:
            s = accts[a]["source"]
            by_src.setdefault(s, Counter())[argmax(L1[a][0][n])] += 1
        stats["distribution"][n + "_by_source"] = {s: dict(c.most_common()) for s, c in by_src.items()}
    both = sorted(set(L1) & set(L2))
    for n in list(Q) + ["items_about_ai"]:
        if n == "items_about_ai":
            pairs = [(L1[a][0][n][k] >= 0.5, L2[a][0][n][k] >= 0.5) for a in both for k in L1[a][0][n] if k in L2[a][0][n]]
            tv = [abs(L1[a][0][n][k] - L2[a][0][n][k]) for a in both for k in L1[a][0][n] if k in L2[a][0][n]]
        else:
            pairs = [(argmax(L1[a][0][n]), argmax(L2[a][0][n])) for a in both]
            tv = [0.5 * sum(abs(L1[a][0][n][o] - L2[a][0][n][o]) for o in L1[a][0][n]) for a in both]
        stats["self_agreement"][n] = {"argmax": round(sum(x == y for x, y in pairs) / len(pairs), 4), "n": len(pairs),
                                      "mean_tv": round(sum(tv) / len(tv), 4)}
    stats["cases"] = dict(stats["cases"])
    (ROOT / "results" / "labels.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({"cases": stats["cases"], "self_agreement": stats["self_agreement"],
                      "account_type": stats["distribution"]["account_type"]}, indent=1))


if __name__ == "__main__":
    main()
