"""BM25 on the embedder's held-out section retrieval set (same queries and pool as run/embed.py eval).

    python3 run/bm25_eval.py [_para]      # -> results/emb_base-bm25[-para].json
"""
import json
import math
import sys
from pathlib import Path

import bm25s
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks import ROOT, sections

TAG = sys.argv[1] if len(sys.argv) > 1 else ""   # "_para" scores the paraphrased queries
d = json.loads((ROOT / "out" / "upload" / f"emb_data{TAG}.json").read_text())
sec = {r["sid"]: r["text"] for r in sections()}
pool = d["eval_pool"]
ret = bm25s.BM25()
ret.index(bm25s.tokenize([sec[s] for s in pool], stopwords="en", show_progress=False), show_progress=False)
res, _ = ret.retrieve(bm25s.tokenize([q["query"] for q in d["eval_queries"]], stopwords="en", show_progress=False),
                      k=100, show_progress=False)
nd, r1, r10, within, by = [], [], [], [], {}
for q, top in zip(d["eval_queries"], res):
    ranked = [pool[i] for i in top]
    rank = ranked.index(q["pos"]) + 1 if q["pos"] in ranked else 10 ** 6
    nd.append(1 / math.log2(rank + 1) if rank <= 10 else 0.0)
    r1.append(rank == 1)
    r10.append(rank <= 10)
    art = q["pos"].split(":")[0]
    same = [s for s in ranked if s.split(":")[0] == art]
    within.append(bool(same) and same[0] == q["pos"])
    by.setdefault(q["style"], []).append(nd[-1])
out = {"name": "base-bm25" + TAG.replace("_", "-"), "sections": {"ndcg@10": float(np.mean(nd)), "recall@1": float(np.mean(r1)),
                                         "recall@10": float(np.mean(r10)), "within_article_top1": float(np.mean(within)),
                                         "n_queries": len(nd), "pool": len(pool)},
       "sections_perquery_ndcg": [round(x, 4) for x in nd],
       "sections_by_style": {k: float(np.mean(v)) for k, v in by.items()}}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / f"emb_base-bm25{TAG.replace('_', '-')}.json").write_text(json.dumps(out) + "\n")
print(json.dumps(out["sections"]), out["sections_by_style"])
