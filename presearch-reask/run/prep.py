"""Build the eval set and the BM25 knowledge base from MuSiQue-Answerable dev.

    python3 run/prep.py      # writes data/eval.jsonl and data/corpus.jsonl
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
STRATA = {"2hop": 150, "3hop": 100, "4hop": 50}
KB_QUESTIONS = 1000


def hops(qid: str) -> str:
    return qid.split("_")[0][:4]


def main():
    rows = [json.loads(l) for l in (DATA / "musique_ans_v1.0_dev.jsonl").open()]
    rng = random.Random(0)
    pick = []
    for h, n in STRATA.items():
        pool = [r for r in rows if hops(r["id"]) == h]
        pick += rng.sample(pool, n)
    picked = {r["id"] for r in pick}
    kb_rows = rows[:KB_QUESTIONS] + [r for r in rows[KB_QUESTIONS:] if r["id"] in picked]

    seen, corpus = {}, []
    for r in kb_rows:
        for p in r["paragraphs"]:
            key = (p["title"], p["paragraph_text"])
            if key not in seen:
                seen[key] = len(corpus)
                corpus.append({"id": len(corpus), "title": p["title"], "text": p["paragraph_text"]})
    with (DATA / "corpus.jsonl").open("w") as f:
        for d in corpus:
            f.write(json.dumps(d) + "\n")
    with (DATA / "eval.jsonl").open("w") as f:
        for r in pick:
            gold = [seen[(p["title"], p["paragraph_text"])] for p in r["paragraphs"] if p["is_supporting"]]
            f.write(json.dumps({"id": r["id"], "hops": hops(r["id"]), "question": r["question"],
                                "answers": [r["answer"]] + list(r["answer_aliases"]),
                                "gold_paragraphs": gold}) + "\n")
    print(f"eval {len(pick)}  corpus {len(corpus)} paragraphs from {len(kb_rows)} questions")


if __name__ == "__main__":
    main()
