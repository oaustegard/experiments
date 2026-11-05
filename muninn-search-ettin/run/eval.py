"""Score the site search on data/queries.json: does the query's source post appear in the top results?

    python3 -I run/eval.py KBI_PREFIX NAME

Arms, all through remax_kb's own reader (the same search the Worker runs): bm25 (lexical leg only),
dense (the embedding leg alone, SimHash Hamming as shipped), hybrid (RRF of both, the Worker's default).
Post-level: a hit is any returned chunk from the query's post. Writes results/<NAME>.json.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib.paths import spoke   # EXPERIMENTS_SPOKES_ROOT holds remax, remax_kb, muninn.austegard.com
for p in (spoke("remax_kb"), spoke("remax") / "src", spoke("muninn.austegard.com") / "scripts"):
    sys.path.insert(0, str(p))
from remax_kb.read_v2 import KB
from ettin_embedder import EttinEmbedder
ROOT = Path(__file__).resolve().parents[1]
kb = KB.open(sys.argv[1] + ".kbi"); name = sys.argv[2]
emb = EttinEmbedder()
Q = json.loads((ROOT / "data" / (sys.argv[3] if len(sys.argv) > 3 else "queries.json")).read_text())
post = lambda cid: cid.rsplit("#", 1)[0]

def posts(hits):
    seen = []
    for h in hits:
        p = post(kb._chunk_id_at(h.row))
        if p not in seen: seen.append(p)
    return seen

res = {}
for arm in ("bm25", "dense", "hybrid"):
    h5 = h1 = rr = 0
    for q in Q:
        if arm == "bm25": hits = kb._bm25_search(q["query"])[:40]
        elif arm == "dense": hits = kb._dense_search(q["query"], emb)[:40]
        else: hits = kb.search(q["query"], embedder=emb, k=40)
        ps = posts(hits); target = post(q["chunk_id"])
        rank = ps.index(target) + 1 if target in ps else None
        h1 += rank == 1; h5 += bool(rank and rank <= 5); rr += 1 / rank if rank and rank <= 10 else 0
    n = len(Q)
    res[arm] = {"post_hit@1": round(h1 / n, 4), "post_hit@5": round(h5 / n, 4), "post_mrr@10": round(rr / n, 4), "n": n}
    print(arm, res[arm], flush=True)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1))
