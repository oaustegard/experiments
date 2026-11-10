"""Candidate sections per query, for the relevance question and for the embedder's hard negatives.

    python3 run/pairs.py build      # data/queries_{train,eval}.jsonl + sections -> out/pairs_{train,eval}.jsonl,
                                    #   out/jev_pairs_in.jsonl, data/goldrel_pairs.jsonl
For every query: its source section, one other section of the same article, and the top BM25 sections
from other articles (2 for training queries, 2 for eval). BM25 runs over the query's own split pool, so
no eval article is a training negative. The 800 gold pairs are 200 eval queries with their 4 candidates.
"""
import json
import random
import sys
from pathlib import Path

import bm25s

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks import ROOT, sections


def main():
    rows = sections()
    by_sid = {r["sid"]: r for r in rows}
    by_art = {}
    for r in rows:
        by_art.setdefault(r["pmcid"], []).append(r["sid"])
    rng = random.Random(0)
    jev_in, gold = [], []
    for which in ("train", "eval"):
        with open(ROOT / "data" / f"queries_{which}.jsonl") as f:
            qs = [json.loads(line) for line in f]
        pool = [r for r in rows if (r["split"] == "train") == (which == "train")]
        psid = [r["sid"] for r in pool]
        ret = bm25s.BM25()
        ret.index(bm25s.tokenize([r["text"] for r in pool], stopwords="en", show_progress=False), show_progress=False)
        res, _ = ret.retrieve(bm25s.tokenize([q["query"] for q in qs], stopwords="en", show_progress=False), k=12,
                              show_progress=False)
        out = []
        for q, top in zip(qs, res):
            src = by_sid[q["id"]]
            sib = [s for s in by_art[src["pmcid"]] if s != q["id"]]
            same = rng.choice(sib) if sib else None
            ranked = [psid[i] for i in top]
            bm = [s for s in ranked if by_sid[s]["pmcid"] != src["pmcid"]][:2]
            rank_src = ranked.index(q["id"]) + 1 if q["id"] in ranked else None
            cands = [("pos", q["id"])] + ([("same", same)] if same else []) + [("bm25", s) for s in bm]
            out.append({"qid": q["id"], "query": q["query"], "bm25_rank_of_source": rank_src,
                        "cands": [{"kind": k, "sid": s} for k, s in cands]})
            for k, s in cands:
                jev_in.append({"id": f"{which}|{q['id']}|{s}", "state": {"query": q["query"], "section": by_sid[s]["text"]},
                               "questions": ["relevance"]})
        (ROOT / "out" / f"pairs_{which}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
        hit = sum(r["bm25_rank_of_source"] == 1 for r in out) / len(out)
        print(f"{which}: {len(out)} queries, BM25 ranks the source first for {hit:.3f}, in the top 10 for "
              f"{sum((r['bm25_rank_of_source'] or 99) <= 10 for r in out) / len(out):.3f}")
        if which == "eval":
            for r in rng.sample(out, 200):
                for c in r["cands"]:
                    gold.append({"id": f"eval|{r['qid']}|{c['sid']}", "query": r["query"], "kind": c["kind"],
                                 "text": by_sid[c["sid"]]["text"]})
    (ROOT / "out" / "jev_pairs_in.jsonl").write_text("".join(json.dumps(r) + "\n" for r in jev_in))
    (ROOT / "data" / "goldrel_pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gold))
    print(f"jev pairs {len(jev_in)}, gold pairs {len(gold)}")


if __name__ == "__main__":
    main()
