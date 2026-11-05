"""Baselines for the account decider, on the same test accounts and the same Sonnet labels.

    python3 -I run/baseline_probe.py

1. Majority: each question's most frequent Sonnet label among training accounts.
2. Frozen-embedding probe: the 10 MB build's own Ettin-17m sentence vectors (ettin-native e17c,
   768-d unit vectors, the embedder List Scout runs) with a logistic regression per question
   (C=3, the bsky-list-ai-share recipe), fitted on training accounts' argmax labels. Account questions
   embed the account state; the post question embeds the post state.
Writes results/baselines.json.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
E17C = ROOT.parent / "browser-decider" / "out" / "base" / "st-ettin17m-tuned-e17-cap-c-epoch1" / "epoch1"


def argmax(d):
    return max(d, key=d.get)


def main():
    recs = [json.loads(l) for l in (ROOT / "data" / "cases.jsonl").open()]
    model = SentenceTransformer(str(E17C), device="cpu")
    model.max_seq_length = 1024
    out = {"majority": {}, "probe": {}}
    qnames = list(recs[0]["questions"]) if recs[0]["workflow"] == "account" else None
    acc_recs = [r for r in recs if r["workflow"] == "account"]
    post_recs = [r for r in recs if r["workflow"] == "post"]
    qnames = list(acc_recs[0]["questions"])
    jobs = [(q, acc_recs) for q in qnames] + [("about_ai", post_recs)]
    cache = {}
    for q, rs in jobs:
        tr = [r for r in rs if r["split"] == "train"]
        te = [r for r in rs if r["split"] == "test"]
        ytr = [argmax(r["gold"][q]["probabilities"]) for r in tr]
        yte = [argmax(r["gold"][q]["probabilities"]) for r in te]
        maj = Counter(ytr).most_common(1)[0][0]
        out["majority"][q] = {"label": maj, "accuracy": round(float(np.mean([y == maj for y in yte])), 4), "n_test": len(te)}
        key = id(rs)
        if key not in cache:
            texts = [r["state"] for r in rs]
            cache[key] = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        X = cache[key]
        idx = {id(r): i for i, r in enumerate(rs)}
        Xtr = X[[idx[id(r)] for r in tr]]
        Xte = X[[idx[id(r)] for r in te]]
        if len(set(ytr)) < 2:
            out["probe"][q] = {"accuracy": out["majority"][q]["accuracy"], "note": "one class in train"}
            continue
        lr = LogisticRegression(C=3, max_iter=3000).fit(Xtr, ytr)
        pred = lr.predict(Xte)
        out["probe"][q] = {"accuracy": round(float(np.mean([p == y for p, y in zip(pred, yte)])), 4), "n_test": len(te)}
        print(q, out["majority"][q]["accuracy"], out["probe"][q]["accuracy"], flush=True)
    (ROOT / "results" / "baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
