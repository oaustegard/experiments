"""Every Jev call the six steps need, in one resumable job (cheap sets first).

Run as a background job; rerunning skips everything already cached.
"""
import json
import sys
import time

import jev
from data import DATA, MAX_CHARS

CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def jsonl(p):
    return [json.loads(line) for line in open(p)]


def main():
    mixed = [(r["id"], r["text"]) for r in jsonl(DATA / "mixed.jsonl")]
    arxiv = jsonl(DATA / "arxiv.jsonl")
    ax_test = [(r["id"], r["text"]) for r in arxiv if r["split"] == "test"]
    ax_train = [(r["id"], r["text"]) for r in arxiv if r["split"] == "train"]
    sf = DATA / "scifact"
    qids = sorted({line.split("\t")[0] for line in (sf / "test.tsv").read_text().splitlines()[1:]}, key=int)
    queries = {r["_id"]: r["text"] for r in jsonl(sf / "queries.jsonl")}
    sf_q = [(q, queries[q]) for q in qids]
    sf_docs = [(r["_id"], f"{r['title']}. {r['text']}"[:MAX_CHARS]) for r in jsonl(sf / "corpus.jsonl")]

    plan = [
        ("mixed", "about", mixed),                     # step 2
        ("arxiv", "about", ax_test),                   # step 3 zero-shot
        ("arxiv", "mentions", ax_test[:50]),           # step 5
        ("arxiv", "substantially", ax_test[:50]),      # step 5
        ("scifact_q", "about", sf_q),                  # step 4 query side
        ("scifact_q", "query", sf_q),                  # step 4 query side, query phrasing
        ("arxiv", "about", ax_train),                  # step 3 Jev-vector probe arm
        ("scifact", "about", sf_docs),                 # step 4 doc side
        ("mixed_rerun", "about", mixed[:30]),          # step 6, last so it is hours after the first pass
    ]
    t0 = time.time()
    for set_name, variant, items in plan:
        recs = jev.encode(set_name, variant, items, concurrency=CONC)
        ok = sum("p" in recs.get(i, {}) for i, _ in items)
        bad = [r for i, _ in items if "p" not in (r := recs.get(i, {}))]
        print(f"DONE {set_name}/{variant}: {ok}/{len(items)} ok, {len(bad)} failed "
              f"({sum(r.get('error') == 'blocked' for r in bad)} WAF-blocked) t={time.time() - t0:.0f}s", flush=True)
        jev.to_parquet(set_name, variant)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
