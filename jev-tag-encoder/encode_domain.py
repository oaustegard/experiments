"""Round 3: encode SciFact queries and corpus with the corpus-fitted taxonomy (tags_scifact.txt)."""
import json
import time

import jev
from data import DATA, MAX_CHARS

TAGS = jev.load_tags(jev.HERE / "tags_scifact.txt")


def jsonl(p):
    return [json.loads(line) for line in open(p)]


def main():
    sf = DATA / "scifact"
    qids = sorted({line.split("\t")[0] for line in (sf / "test.tsv").read_text().splitlines()[1:]}, key=int)
    queries = {r["_id"]: r["text"] for r in jsonl(sf / "queries.jsonl")}
    docs = [(r["_id"], f"{r['title']}. {r['text']}"[:MAX_CHARS]) for r in jsonl(sf / "corpus.jsonl")]
    t0 = time.time()
    for set_name, items in (("scifact_q_dom", [(q, queries[q]) for q in qids]), ("scifact_dom", docs)):
        recs = jev.encode(set_name, "about", items, concurrency=8, tags=TAGS)  # pacing sets the rate
        ok = sum("p" in recs.get(i, {}) for i, _ in items)
        print(f"DONE {set_name}/about: {ok}/{len(items)} ok t={time.time() - t0:.0f}s", flush=True)
        jev.to_parquet(set_name, "about")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
