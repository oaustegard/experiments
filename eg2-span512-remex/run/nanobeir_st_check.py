"""Check that remex's sentence-transformers wrapper (RemexIndex) gives the round 3 NanoBEIR numbers.

Usage: python3 -I run/nanobeir_st_check.py

Feeds the stored tomaarsen/embeddinggemma-2-512d embeddings (out/nanobeir_emb/) to
RemexIndex(bits=3, center=True) and to center=False, scores NDCG@10 with run/nanobeir_eval.py's
function, and compares the mean of the 13 sets with svd512_b3 in results/nanobeir.json (which used
Quantizer(512, bits=3, seed=42, mean=corpus mean) directly). Writes results/nanobeir_st_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from remex import RemexIndex

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nanobeir_eval import EMB, SETS, ndcg10, part, unit  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def main() -> int:
    out = {}
    for center in (True, False):
        per = {}
        for name in SETS:
            corpus, queries, qrels = part("corpus", name), part("queries", name), part("qrels", name)
            row = {cid: i for i, cid in enumerate(corpus["_id"].astype(str))}
            rel = {}
            for qid, cid in zip(qrels["query-id"].astype(str), qrels["corpus-id"].astype(str)):
                if cid in row:
                    rel.setdefault(qid, set()).add(row[cid])
            qids = queries["_id"].astype(str).tolist()
            keep = [i for i, q in enumerate(qids) if q in rel]
            z = np.load(EMB / f"{name}_svd512.npz")
            index = RemexIndex(bits=3, center=center, dim=512)
            index.add(unit(z["d"]))
            hits = index.search(unit(z["q"][keep]), top_k=10)
            scores = np.full((len(keep), len(corpus)), -np.inf, np.float32)
            for i, h in enumerate(hits):          # only the top 10 matter for NDCG@10
                for r in h:
                    scores[i, r["corpus_id"]] = r["score"]
            per[name] = round(100 * float(ndcg10(scores, [rel[qids[i]] for i in keep]).mean()), 3)
        out[f"center_{center}"] = {"per_set": per, "mean_of_sets": round(float(np.mean(list(per.values()))), 3)}
        print(center, out[f"center_{center}"]["mean_of_sets"], flush=True)
    ref = json.loads((HERE / "results" / "nanobeir.json").read_text())["arms"]["svd512_b3"]
    out["round3_svd512_b3"] = ref["mean_of_sets"]
    out["max_abs_set_diff_center_true"] = round(max(abs(out["center_True"]["per_set"][n] - ref["per_set"][n]) for n in SETS), 3)
    (HERE / "results" / "nanobeir_st_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
