"""Seeded sample of benchmark sentences and label sets for the browser runtime's timing runs.

    python3 run/bench_texts.py [--n 500] [--seed 0] [--out out/web_bench_texts.json]

Sentences come from the test splits of mit-movie and the five CrossNER sets in data/bench (written by
run/prep_data.py), drawn evenly per set with a seeded RNG and joined with single spaces (the
benchmark's word lists carry no original spacing). Label sets: `labels50` is every distinct label of
the seven benchmark sets in file order, padded with generic types to 50; `labels10` and `labels20`
are its first 10 and 20. The output is the GPU Bridge job's `bench` input and check.html's CPU timing set.
"""
import argparse
import json
import random
from pathlib import Path

D = Path(__file__).resolve().parents[1]
SETS = ["mit-movie", "CrossNER_AI", "CrossNER_literature", "CrossNER_music", "CrossNER_politics", "CrossNER_science"]
ALL = SETS + ["mit-restaurant"]
PAD = ["date", "time", "money", "percent", "quantity", "language", "nationality", "product", "vehicle", "food",
       "sport", "disease", "drug", "law", "facility", "religion", "website", "email address", "phone number"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=D / "out" / "web_bench_texts.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    per = -(-a.n // len(SETS))
    texts, src = [], []
    for s in SETS:
        rows = json.loads((D / "data" / "bench" / s / "test.json").read_text())
        for r in rng.sample(rows, per):
            texts.append(" ".join(r["tokenized_text"]))
            src.append(s)
    order = list(range(len(texts)))
    rng.shuffle(order)
    order = order[: a.n]
    labels = []
    for s in ALL:
        for lab in json.loads((D / "data" / "bench" / s / "labels.json").read_text()):
            if lab not in labels:
                labels.append(lab)
    labels += [x for x in PAD if x not in labels]
    labels = labels[:50]
    out = {"seed": a.seed, "n": len(order), "sources": [src[i] for i in order], "texts": [texts[i] for i in order],
           "labels10": labels[:10], "labels20": labels[:20], "labels50": labels,
           "note": "test-split sentences joined with single spaces; see run/bench_texts.py"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False))
    words = sorted(len(t.split()) for t in out["texts"])
    print(f"{a.out}: {len(order)} texts, words median {words[len(words) // 2]}, max {words[-1]}; {len(labels)} labels")


if __name__ == "__main__":
    main()
