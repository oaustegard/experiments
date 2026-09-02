#!/usr/bin/env python3
"""Arm C: frozen coordinates. Surrogate + UCB over the round-1 blind axes only."""
import json
import numpy as np
import arms

for case, pool_name, tid in [("P1_PRE","P1","DOI:10.1007/s00493-004-0007-x"),
                             ("P2","P2","arXiv:2412.05182")]:
    pool = json.load(open(f"cache/pool_{pool_name}.json"))["pool"]
    feats = json.load(open(f"signatures/{pool_name}.json"))["features_round1"]
    tidx = next(i for i,p in enumerate(pool) if p["id"]==tid)
    Phi,_ = arms.build_phi(pool, feats, "title")
    reads, trace = arms.run_sequential([Phi], tidx, budget=200, batch=5, grow=False)
    # also record the pure static rank under mean-of-axes, for reference
    static = arms.rank_of(Phi.mean(axis=1), tidx)
    print(f"{case}: arm C reads_to_hit={reads}  (mean-of-axes static rank={static}, n={len(pool)})", flush=True)
    json.dump({"case":case,"reads":reads,"static_mean_axes_rank":static,"n":len(pool),
               "trace":trace[:12]}, open(f"results/armC_{case}.json","w"), indent=1)
