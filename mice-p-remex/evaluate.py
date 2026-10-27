"""nDCG@10 for score files, and paired deltas against a baseline arm.

    python3 evaluate.py DATASET SCORES.json [BASELINE.json]

Gains are the qrels grades (linear, as pytrec_eval); queries with no positive
judgement are dropped. Ties in score break by SPLADE rank (candidate order).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def qrels(ds):
    r = pd.read_parquet(HERE / "cache" / "data" / ds / "qrels_test.parquet")
    out = {}
    for q, d, s in zip(r["query-id"].astype(str), r["corpus-id"].astype(str), r.score):
        if s > 0:
            out.setdefault(q, {})[d] = int(s)
    return out


def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def per_query(ds, scores):
    rel = qrels(ds)
    out = {}
    for q, s in scores.items():
        if q not in rel:
            continue
        docs = list(s)  # candidate (SPLADE) order
        ranked = sorted(docs, key=lambda d: (-s[d], docs.index(d)))
        out[q] = ndcg10(ranked, rel[q])
    return out


def paired(a, b, n_boot=10_000, seed=0):
    """Mean of a - b over shared queries, with a 95% paired bootstrap CI."""
    qs = sorted(set(a) & set(b))
    d = np.array([a[q] - b[q] for q in qs])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), len(qs)


def fidelity(arm, base):
    """Agreement with the uncompressed ranking, per query: Spearman rho over the
    candidates and |top-10 overlap|/10. Needs no qrels, so it is far less noisy
    than an nDCG delta on a 41-query set."""
    from scipy.stats import spearmanr
    def top10(scores, docs):
        return set(sorted(docs, key=lambda d: -scores[d])[:10])

    rho, ov = [], []
    for q in base:
        docs = list(base[q])
        a, b = [arm[q][d] for d in docs], [base[q][d] for d in docs]
        rho.append(spearmanr(a, b).statistic)
        ov.append(len(top10(arm[q], docs) & top10(base[q], docs)) / 10)
    return float(np.mean(rho)), float(np.mean(ov))


if __name__ == "__main__":
    ds, path, *base = sys.argv[1:]
    pq = per_query(ds, json.load(open(path)))
    print(f"{ds} {Path(path).name}: nDCG@10 {np.mean(list(pq.values())):.4f} over {len(pq)} queries")
    if base:
        m, lo, hi, n = paired(pq, per_query(ds, json.load(open(base[0]))))
        print(f"  delta vs {Path(base[0]).name}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] n={n}")
        rho, ov = fidelity(json.load(open(path)), json.load(open(base[0])))
        print(f"  fidelity: spearman {rho:.4f}  top10 overlap {ov:.3f}")
