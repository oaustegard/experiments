#!/usr/bin/env python3
"""Are the generated intents faithful to the page they were written from?

The handbook is worth something only if its "use when you want to" lines actually
describe the utility above them. Checking that against the *enriched* corpus is
circular — the intents are in the document, so they retrieve it by construction.

So the query is a card's intents and the index is the **plain** corpus, with no
generated text in it at all. A faithful intent line is a goal-level restatement
of what the page documents, and the page is then reachable from it through the
original wording; a generic or invented one is not. This measures the artifact
rather than the pipeline, and it needs no eval, no labels and no held-out set —
the label is the page the card was written from.

Reported per intent line and per card (any of its lines retrieving the page),
against the same BM25 and dense arms the retrieval work uses, so a weak number
can be read as "the intents are vague" rather than "the retriever is weak".

    python3 check_cards.py --model leaf-mt-int8 --sample 600
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import dense_index as D  # noqa: E402
import retrieve as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    ap.add_argument("--cards", type=Path, default=HERE / "data" / "cards.jsonl")
    ap.add_argument("--model", default="leaf-mt-int8")
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--out", type=Path, default=HERE / "results_card_fidelity.json")
    a = ap.parse_args()

    pages = D.page_chunks(R.load_chunks(a.chunks))
    by_id = {p.id: p for p in pages}
    cards = []
    for line in a.cards.open():
        rec = json.loads(line)
        if rec.get("card") and rec["card"].get("intents") and rec["id"] in by_id:
            cards.append(rec)
    rng = random.Random(a.seed)
    rng.shuffle(cards)
    cards = cards[: a.sample]
    print(f"{len(cards)} cards sampled", file=sys.stderr)

    index = R.Index(pages)
    utilities = np.array([p.utility for p in pages], dtype=object)
    _, _, dense = D.load(a.model, a.chunks, granularity="page")

    per = []
    for rec in cards:
        gold = rec["utility"]
        rows = []
        for intent in rec["card"]["intents"]:
            bm = [u for u, _ in D.rank_utilities(index.scores(intent), utilities,
                                                 a.pool, positive_only=True)]
            dn = [u for u, _ in D.rank_utilities(dense.scores(intent), utilities,
                                                 a.pool)]
            rows.append({"intent": intent,
                         "bm25_top1": bool(bm) and bm[0] == gold,
                         "bm25_top3": gold in bm[:3],
                         "dense_top1": bool(dn) and dn[0] == gold,
                         "dense_top3": gold in dn[:3]})
        per.append({"utility": gold, "intents": rows})

    flat = [r for c in per for r in c["intents"]]
    def rate(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None
    summary = {
        "n_cards": len(per), "n_intents": len(flat),
        "per_intent": {k: rate(flat, k) for k in
                       ("bm25_top1", "bm25_top3", "dense_top1", "dense_top3")},
        "per_card_any_intent": {
            k: round(sum(any(r[k] for r in c["intents"]) for c in per) / len(per), 3)
            for k in ("bm25_top1", "bm25_top3", "dense_top1", "dense_top3")},
    }
    a.out.write_text(json.dumps({"summary": summary, "cards": per}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
