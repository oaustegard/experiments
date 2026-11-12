#!/usr/bin/env python3
"""Round-3 tables (RemexIndex search scoring) and summary3.json.

Each arm's score is the per-query mean over seeds 42, 1, 2, 3; paired differences
are taken on those means, with a 5,000-resample bootstrap over queries, and the
number of seeds in which a beat b is listed beside them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = [("st pf4 4-bit", "none 1-bit"), ("st pf2 2-bit", "none 1-bit"), ("st pf4 4-bit", "st pf2 2-bit"),
         ("st pf4 2-bit", "none 1-bit"), ("st pf2 4-bit", "none 2-bit"), ("none 2-bit", "none 1-bit"),
         ("st pf4 4-bit", "none 8-bit"), ("st pf4 8-bit", "none 8-bit"), ("none 4-bit", "none 8-bit"),
         ("adapt c1@2 2-bit", "match c1@2 2-bit"), ("adapt c4@3 3-bit", "match c4@3 3-bit"),
         ("adapt c2@3 3-bit", "match c2@3 3-bit"), ("adapt c1@3 3-bit", "match c1@3 3-bit"),
         ("adapt c1@2 2-bit", "st pf4 4-bit"), ("adapt c1@2 2-bit", "none 1-bit"),
         ("adapt c1@3 3-bit", "none 4-bit")]


def boot(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return [round(float(d.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    summary, out = {}, []
    for corpus in ("scifact", "nfcorpus"):
        f = HERE / f"results3_{corpus}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f)]
        by = {}
        for r in rows:
            by.setdefault(r["arm"], {})[r["seed"]] = r
        S = summary[corpus] = {"rows": {}, "pairs": {}}
        out.append(f"### {corpus}, scored by RemexIndex search (mean over seeds)\n")
        out.append("| arm | tok/doc | B/doc | nDCG@10 | seeds | range |")
        out.append("|---|---:|---:|---:|---:|---|")
        for arm, seeds in by.items():
            pq = np.mean([s["perquery"] for s in seeds.values()], axis=0)
            ms = [s["ndcg10"] for s in seeds.values()]
            any_ = next(iter(seeds.values()))
            S["rows"][arm] = dict(ndcg=round(float(pq.mean()), 4), n=len(ms), range=[min(ms), max(ms)],
                                  tok=any_["tokens_per_doc"], bytes=any_["bytes_per_doc"])
            out.append(f"| {arm} | {any_['tokens_per_doc']:.1f} | {any_['bytes_per_doc']:,.0f} | {pq.mean():.4f} | "
                       f"{len(ms)} | {min(ms):.4f}–{max(ms):.4f} |")
        out.append("")
        out.append("| a | b | B/doc a / b | Δ a − b [95% CI] | seeds with a > b |")
        out.append("|---|---|---|---|---|")
        for a, b in PAIRS:
            if a not in by or b not in by:
                continue
            shared = sorted(set(by[a]) & set(by[b]))
            d = np.mean([np.array(by[a][s]["perquery"]) - np.array(by[b][s]["perquery"]) for s in shared], axis=0)
            per = [round(by[a][s]["ndcg10"] - by[b][s]["ndcg10"], 4) for s in shared]
            ci = boot(d)
            S["pairs"][f"{a} vs {b}"] = dict(delta=ci, per_seed=per)
            out.append(f"| {a} | {b} | {by[a][shared[0]]['bytes_per_doc']:,.0f} / {by[b][shared[0]]['bytes_per_doc']:,.0f} | "
                       f"{ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}] | {sum(p > 0 for p in per)} of {len(per)} |")
        out.append("")
    json.dump(summary, open(HERE / "summary3.json", "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
