"""Real (human-written) query sets from MS MARCO, the held-out test set and a general training pool.

Usage: python3 -I run/make_real_queries.py

Writes out/queries/real_test.jsonl (1,000: all 238 AI-topical MS MARCO queries plus 762
tech/science ones) and out/queries/msmarco_train.jsonl (30,000 random others). No test
query, and no query matching either topical regex, enters the training pool, so the real
test set is unseen in topic as well as in text.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "queries"
AI = re.compile(r"\b(?:ai|artificial intelligence|machine learning|neural|deep learning|chatgpt|gpt|llm|"
                r"language model|robot\w*|algorithm\w*|data scien\w*|computer vision|automation|chatbot|"
                r"openai|nlp|natural language)\b", re.I)
TECH = re.compile(r"\b(?:software|computer\w*|programming|programmer|python|javascript|internet|google|"
                  r"facebook|twitter|microsoft|silicon|semiconductor|processor|cpu|gpu|nvidia|intel|"
                  r"database|cloud computing|server|encryption|cyber\w*|hacker|coding|code|app|smartphone|"
                  r"technology|startup|brain|cognitive|consciousness|neuron\w*|copyright|patent|data)\b", re.I)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(ROOT / "out" / "msmarco_queries.parquet")
    rng = np.random.default_rng(0)
    is_ai = d.text.str.contains(AI).to_numpy()
    is_tech = d.text.str.contains(TECH).to_numpy() & ~is_ai
    tech_pick = rng.choice(np.flatnonzero(is_tech), 1000 - is_ai.sum(), replace=False)
    test = np.concatenate([np.flatnonzero(is_ai), tech_pick])
    pool = np.flatnonzero(~is_ai & ~is_tech)
    train = rng.choice(pool, 30000, replace=False)
    for name, rows, tag in [("real_test", test, None), ("msmarco_train", train, "msmarco")]:
        with (OUT / f"{name}.jsonl").open("w") as f:
            for r in rows:
                kind = tag or ("ai" if is_ai[r] else "tech")
                f.write(json.dumps({"id": str(d._id.iloc[r]), "q": d.text.iloc[r], "kind": kind}) + "\n")
    print(json.dumps({"real_test": len(test), "ai": int(is_ai.sum()), "msmarco_train": len(train)}))


if __name__ == "__main__":
    main()
