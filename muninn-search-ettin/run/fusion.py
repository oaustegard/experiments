"""Fusion sweep: post hit@5 / MRR@10 for RRF and weighted alpha on both query sets. Writes results/fusion.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib.paths import spoke   # EXPERIMENTS_SPOKES_ROOT holds remax, remax_kb, muninn.austegard.com
for p in (spoke("remax_kb"), spoke("remax") / "src", spoke("muninn.austegard.com") / "scripts"):
    sys.path.insert(0, str(p))
from remax_kb.read_v2 import KB
from ettin_embedder import EttinEmbedder
ROOT = Path(__file__).resolve().parents[1]
kb = KB.open(sys.argv[1] + ".kbi"); emb = EttinEmbedder()
post = lambda cid: cid.rsplit("#", 1)[0]
out = {}
for qs in ("queries.json", "queries_hard.json"):
    Q = json.loads((ROOT / "data" / qs).read_text())
    for cfg in [("rrf", None, 60), ("rrf_c10", None, 10), ("a0.3", 0.3, 60), ("a0.5", 0.5, 60), ("a0.7", 0.7, 60), ("a0.85", 0.85, 60)]:
        name, alpha, c = cfg; h5 = rr = 0
        for q in Q:
            hits = kb.search(q["query"], embedder=emb, k=40, alpha=alpha, rrf_c=c)
            ps = []
            for h in hits:
                p = post(kb._chunk_id_at(h.row))
                if p not in ps: ps.append(p)
            t = post(q["chunk_id"]); rank = ps.index(t) + 1 if t in ps else None
            h5 += bool(rank and rank <= 5); rr += 1 / rank if rank and rank <= 10 else 0
        out[f"{qs}:{name}"] = {"post_hit@5": round(h5 / len(Q), 4), "post_mrr@10": round(rr / len(Q), 4)}
        print(qs, name, out[f"{qs}:{name}"], flush=True)
(ROOT / "results" / "fusion.json").write_text(json.dumps(out, indent=1))
