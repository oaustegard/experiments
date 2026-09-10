"""Fusion (arm C), rescue rates, and the tables RESULTS.md quotes.

Rescue rate is the primary metric in PLAN.md: among the queries one arm misses
at rank 10, the fraction another arm retrieves. It compares directly to the
paper's Appendix Table 7 without depending on reproducing their absolute scores.
"""
import argparse, gzip, itertools, json, os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
RES = f"{OUT}/results"
TOPK = 10
RRF_C = 60


def load(arm):
    p = f"{RES}/{arm}_runs.jsonl.gz"
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        return {r["qid"]: r for r in map(json.loads, f)}


def metrics(ranked, gold):
    gs = set(gold)
    hits = [1 if d in gs else 0 for d in ranked[:TOPK]]
    mrr = next((1.0 / r for r, h in enumerate(hits, 1) if h), 0.0)
    dcg = sum(h / np.log2(r + 1) for r, h in enumerate(hits, 1))
    idcg = sum(1 / np.log2(r + 1) for r in range(1, min(len(gs), TOPK) + 1))
    return dict(hit10=float(any(hits)), hit1=float(hits[:1] == [1]),
                recall10=sum(hits) / len(gs), mrr10=mrr,
                ndcg10=dcg / idcg if idcg else 0.0)


def agg(rows):
    ks = ["hit10", "hit1", "recall10", "mrr10", "ndcg10"]
    return {k: round(100 * float(np.mean([r[k] for r in rows])), 2) for k in ks}


def rrf(arms, qid, gold):
    """Reciprocal rank fusion over each arm's document ranking."""
    score = {}
    for a in arms:
        for rank, d in enumerate(a[qid]["ranked"], 1):
            score[d] = score.get(d, 0.0) + 1.0 / (RRF_C + rank)
    ranked = sorted(score, key=lambda d: -score[d])[:TOPK]
    return dict(qid=qid, qtype=a[qid]["qtype"], ranked=ranked, gold=gold,
                **metrics(ranked, gold))


def rescue(miss_arm, help_arm, qids):
    """Fraction of miss_arm's HIT@10=0 queries that help_arm retrieves."""
    missed = [q for q in qids if miss_arm[q]["hit10"] == 0.0]
    if not missed:
        return dict(miss_n=0, rescued=0, rate=None)
    got = sum(1 for q in missed if help_arm[q]["hit10"] == 1.0)
    return dict(miss_n=len(missed), rescued=got, rate=round(100 * got / len(missed), 1))


def bootstrap_delta(a, b, qids, key="hit10", n=2000, seed=42):
    """Paired bootstrap on the per-query difference, matching the paper's test."""
    rng = np.random.default_rng(seed)
    d = np.array([a[q][key] - b[q][key] for q in qids], dtype=float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return dict(delta=round(100 * d.mean(), 2), ci=[round(100 * lo, 2), round(100 * hi, 2)],
                p=round(float(min(p, 1.0)), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="run-file stems, e.g. A1_chunked A2_whole_doc B_dense")
    ap.add_argument("--fuse", nargs="+", action="append", default=[],
                    help="repeatable: names of arms to RRF together")
    a = ap.parse_args()

    loaded = {}
    for name in a.arms:
        r = load(name)
        if r is None:
            print(f"skip {name}: no run file")
            continue
        loaded[name] = r
    if not loaded:
        raise SystemExit("no arms loaded")

    qids = sorted(set.intersection(*[set(r) for r in loaded.values()]))
    gold = {q: loaded[a.arms[0]][q]["gold"] for q in qids} if a.arms[0] in loaded \
        else {q: next(iter(loaded.values()))[q]["gold"] for q in qids}
    print(f"arms: {list(loaded)} | shared queries: {len(qids)}")

    # fusion arms
    for combo in a.fuse:
        missing = [c for c in combo if c not in loaded]
        if missing:
            print(f"skip fusion {combo}: missing {missing}")
            continue
        name = "C_rrf_" + "_".join(c.split("_")[0] for c in combo)
        arms = [loaded[c] for c in combo]
        fused = {q: rrf(arms, q, gold[q]) for q in qids}
        loaded[name] = fused
        with gzip.open(f"{RES}/{name}_runs.jsonl.gz", "wt") as f:
            for q in qids:
                f.write(json.dumps(fused[q]) + "\n")

    out = {"n_queries": len(qids), "rrf_c": RRF_C, "arms": {}, "rescue": {},
           "paired_bootstrap_vs": {}, "by_type": {}}

    for name, r in loaded.items():
        rows = [r[q] for q in qids]
        out["arms"][name] = agg(rows)
        out["by_type"][name] = {
            t: dict(n=sum(1 for x in rows if x["qtype"] == t),
                    **agg([x for x in rows if x["qtype"] == t]))
            for t in sorted({x["qtype"] for x in rows})}

    for x, y in itertools.permutations(loaded, 2):
        out["rescue"][f"{y}_rescues_{x}"] = rescue(loaded[x], loaded[y], qids)

    base = a.arms[0]
    for name in loaded:
        if name != base:
            out["paired_bootstrap_vs"][f"{name}_vs_{base}"] = \
                bootstrap_delta(loaded[name], loaded[base], qids)

    json.dump(out, open(f"{RES}/analysis.json", "w"), indent=2)
    print(json.dumps({k: out["arms"][k] for k in out["arms"]}, indent=2))
    print("\nrescue rates:")
    for k, v in out["rescue"].items():
        if v["rate"] is not None:
            print(f"  {k:44} {v['rescued']:4}/{v['miss_n']:<4} = {v['rate']}%")
