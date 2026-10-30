"""NDCG@10 on the 13 English NanoBEIR sets: fp32 EG2, its 512 prefix, tomaarsen's SVD 512d model, and
remex codes of each.

Usage: python3 -I run/nanobeir_eval.py      (after run/nanobeir_encode.py; reads out/nanobeir_emb/)

NDCG@10 as sentence-transformers' InformationRetrievalEvaluator computes it (binary relevance from
the qrels, queries without qrels skipped, DCG over the top 10 with 1/log2(rank + 1), ideal DCG over
min(relevant, 10)), so the fp32 arms should reproduce the model card's three columns:
  eg2_768      google/embeddinggemma-2, cosine
  prefix512    its first 512 dims, renormalized
  svd512       tomaarsen/embeddinggemma-2-512d (projection replaced by diag(S) Vh), cosine
remex arms (Quantizer(d, bits, seed=42, mean=corpus mean), documents encoded, queries left float,
scored as search_adc scores, the same matmul as eval.py):
  svd512 at 2, 3, 4 bits (128, 192, 256 B), eg2_768 at 2 and 3 bits (192, 288 B), prefix512 at 3 bits (192 B).
Paired bootstrap 95% CIs over all queries for the differences that carry the comparison, and
svd512 3-bit and eg2_768 2-bit at five more remex seeds. Writes results/nanobeir.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import remex
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent.parent
EMB = HERE / "out" / "nanobeir_emb"
SETS = ["NanoArguAna", "NanoClimateFEVER", "NanoDBPedia", "NanoFEVER", "NanoFiQA2018", "NanoHotpotQA",
        "NanoMSMARCO", "NanoNFCorpus", "NanoNQ", "NanoQuoraRetrieval", "NanoSCIDOCS", "NanoSciFact", "NanoTouche2020"]
CARD = {  # tomaarsen/embeddinggemma-2-512d model card, NDCG@10 x 100
    "eg2_768": [65.377, 27.469, 65.312, 92.432, 44.736, 89.544, 56.633, 38.354, 78.487, 95.463, 37.591, 88.080, 44.226],
    "prefix512": [63.423, 28.177, 64.131, 93.170, 43.188, 88.525, 55.126, 37.229, 76.674, 95.272, 37.753, 88.735, 44.293],
    "svd512": [65.377, 27.469, 65.175, 92.432, 44.563, 89.598, 57.371, 38.290, 78.487, 95.455, 37.595, 88.080, 44.109],
}
REMEX = [("svd512", 2), ("svd512", 3), ("svd512", 4), ("eg2_768", 2), ("eg2_768", 3), ("prefix512", 3)]
K = 10


def part(kind, name):
    return pd.read_parquet(hf_hub_download("sentence-transformers/NanoBEIR-en", f"{kind}/{name}-00000-of-00001.parquet",
                                           repo_type="dataset"))


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def ndcg10(scores, rel_rows):
    """Per-query NDCG@10; scores (nq, n), rel_rows a list of sets of corpus rows."""
    top = np.argsort(-scores, axis=1, kind="stable")[:, :K]
    out = np.empty(len(top))
    disc = 1 / np.log2(np.arange(2, K + 2))
    for i, (t, rel) in enumerate(zip(top, rel_rows)):
        dcg = sum(disc[r] for r, d in enumerate(t) if d in rel)
        out[i] = dcg / disc[: min(len(rel), K)].sum()
    return out


def remex_scores(X, Q, bits, seed=42):
    qz = remex.Quantizer(X.shape[1], bits=bits, seed=seed, mean=X.mean(0).astype(np.float32))
    cv = qz.encode(X)
    dirs = qz.centroids.astype(np.float32)[cv.indices]
    rq = qz._rotate_rows(Q.astype(np.float32)).astype(np.float32)
    s = (rq @ dirs.T) * qz._score_norms(cv).astype(np.float32) + np.asarray(qz._row_bias(cv), np.float32)
    return s + (Q @ X.mean(0)).astype(np.float32)[:, None]   # the query offset; constant per query


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def main() -> int:
    per = {}        # arm -> set -> per-query ndcg
    seeds = {}
    for si, name in enumerate(SETS):
        corpus, queries, qrels = part("corpus", name), part("queries", name), part("qrels", name)
        row = {cid: i for i, cid in enumerate(corpus["_id"].astype(str))}
        rel = {}
        for qid, cid in zip(qrels["query-id"].astype(str), qrels["corpus-id"].astype(str)):
            if cid in row:
                rel.setdefault(qid, set()).add(row[cid])
        qids = queries["_id"].astype(str).tolist()
        keep = [i for i, q in enumerate(qids) if q in rel]
        rel_rows = [rel[qids[i]] for i in keep]
        e = np.load(EMB / f"{name}_eg2.npz")
        s = np.load(EMB / f"{name}_svd512.npz")
        spaces = {"eg2_768": (unit(e["d"]), unit(e["q"][keep])),
                  "prefix512": (unit(e["d"][:, :512]), unit(e["q"][keep][:, :512])),
                  "svd512": (unit(s["d"]), unit(s["q"][keep]))}
        for arm, (X, Q) in spaces.items():
            per.setdefault(arm, {})[name] = ndcg10(Q @ X.T, rel_rows)
        for arm, bits in REMEX:
            X, Q = spaces[arm]
            per.setdefault(f"{arm}_b{bits}", {})[name] = ndcg10(remex_scores(X, Q, bits), rel_rows)
        for arm, bits in (("svd512", 3), ("eg2_768", 2)):
            X, Q = spaces[arm]
            for sd in range(5):
                seeds.setdefault(f"{arm}_b{bits}", {}).setdefault(sd, []).append(ndcg10(remex_scores(X, Q, bits, sd), rel_rows))
        print(name, {a: round(100 * v[name].mean(), 2) for a, v in per.items()}, flush=True)

    res = {"sets": SETS, "metric": "NDCG@10 x 100", "n_queries": int(sum(len(v) for v in per["eg2_768"].values())),
           "arms": {}, "card": CARD, "diffs": {}, "seeds": {}}
    for arm, d in per.items():
        pq = np.concatenate([d[n] for n in SETS])
        res["arms"][arm] = {"per_set": {n: round(100 * float(d[n].mean()), 3) for n in SETS},
                            "mean_of_sets": round(100 * float(np.mean([d[n].mean() for n in SETS])), 3)}
        res["arms"][arm]["pooled"] = round(100 * float(pq.mean()), 3)
    for arm in CARD:
        mine = [res["arms"][arm]["per_set"][n] for n in SETS]
        res["arms"][arm]["card_mean"] = round(float(np.mean(CARD[arm])), 3)
        res["arms"][arm]["max_abs_diff_vs_card"] = round(float(np.max(np.abs(np.array(mine) - CARD[arm]))), 3)
    pairs = [("svd512_b3", "eg2_768_b2"), ("svd512_b3", "svd512"), ("svd512_b3", "eg2_768"), ("svd512_b2", "eg2_768_b2"),
             ("svd512_b3", "prefix512_b3"), ("eg2_768_b2", "eg2_768"), ("svd512", "eg2_768"), ("prefix512", "eg2_768")]
    for a, b in pairs:
        x = np.concatenate([per[a][n] - per[b][n] for n in SETS]) * 100
        res["diffs"][f"{a}_minus_{b}"] = {"mean_of_sets": round(float(np.mean([100 * (per[a][n] - per[b][n]).mean() for n in SETS])), 3),
                                          "pooled": round(float(x.mean()), 3), "ci95_pooled": [round(v, 3) for v in boot(x)]}
    for arm, by in seeds.items():
        means = [100 * float(np.mean([v.mean() for v in by[sd]])) for sd in sorted(by)]
        res["seeds"][arm] = {"mean_of_sets_per_seed": [round(m, 3) for m in means], "range": round(max(means) - min(means), 3)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "nanobeir.json").write_text(json.dumps(res, indent=1))
    for arm in res["arms"]:
        print(f"{arm:14s} mean {res['arms'][arm]['mean_of_sets']:.3f}" + (f"  card {res['arms'][arm]['card_mean']:.3f}  max|diff| {res['arms'][arm]['max_abs_diff_vs_card']}" if arm in CARD else ""))
    print(json.dumps(res["diffs"], indent=1))
    print(json.dumps(res["seeds"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
