#!/usr/bin/env python3
"""Null control demanded by the adversarial pass: is best-of-12 an order statistic?

Three nulls, no new LLM calls:
  1. RANDOM axes  - 20 draws of 12 random pool titles used as axis phrases.
  2. SWAPPED axes - the other case's blind axes.
  3. Reported against the real blind axes.
If random axes routinely land the target inside the top ~20, the named-axis
claim collapses into min-over-k of noisy cosines.
"""
import json
import numpy as np
import embed, arms

rng = np.random.default_rng(0)
out = {}
cases = {}
for pn, tid in [("P1", "DOI:10.1007/s00493-004-0007-x"), ("P2", "arXiv:2412.05182")]:
    pool = json.load(open(f"cache/pool_{pn}.json"))["pool"]
    feats = json.load(open(f"signatures/{pn}.json"))["features_round1"]
    tidx = next(i for i, p in enumerate(pool) if p["id"] == tid)
    D = embed.encode([arms.doc_text(p, "title") for p in pool])
    cases[pn] = (pool, feats, tidx, D)

def ranks(D, feats, tidx):
    F = embed.encode(feats, is_query=True)
    Phi = D @ F.T
    per = [arms.rank_of(Phi[:, f], tidx) for f in range(Phi.shape[1])]
    return arms.rank_of(Phi.mean(1), tidx), int(min(per))

for pn in ["P1", "P2"]:
    pool, feats, tidx, D = cases[pn]
    other = "P2" if pn == "P1" else "P1"
    real_mean, real_best = ranks(D, feats, tidx)
    swap_mean, swap_best = ranks(D, cases[other][1], tidx)
    k = len(feats)
    rnd_mean, rnd_best = [], []
    for _ in range(20):
        idx = rng.choice(len(pool), size=k, replace=False)
        ax = [pool[int(i)]["title"] for i in idx if int(i) != tidx][:k]
        m, b = ranks(D, ax, tidx)
        rnd_mean.append(m); rnd_best.append(b)
    out[pn] = {"n": len(pool), "real_mean": real_mean, "real_best": real_best,
               "swapped_mean": swap_mean, "swapped_best": swap_best,
               "random_mean_median": int(np.median(rnd_mean)),
               "random_mean_min": int(min(rnd_mean)),
               "random_best_median": int(np.median(rnd_best)),
               "random_best_min": int(min(rnd_best)),
               "random_best_p05": int(np.percentile(rnd_best, 5)),
               "random_best_all": sorted(rnd_best)}
    print(pn, json.dumps({k2: v for k2, v in out[pn].items() if k2 != "random_best_all"}), flush=True)
    print("   random best-of-k draws:", sorted(rnd_best), flush=True)
json.dump(out, open("results/null_control.json", "w"), indent=1)
