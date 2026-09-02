#!/usr/bin/env python3
"""Arms A (flat dense, raw query) and N (flat dense, blind-extracted signature)."""
import json, sys
import numpy as np
import embed, arms
from queries import CASES

TARGET = {"P1_PRE": "DOI:10.1007/s00493-004-0007-x",
          "P1_POST": "DOI:10.1007/s00493-004-0007-x",
          "P2": "arXiv:2412.05182", "N3": None}
POOL = {"P1_PRE": "P1", "P1_POST": "P1", "P2": "P2", "N3": "N3"}

out = {}
for case in ["P1_PRE", "P1_POST", "P2", "N3"]:
    pool = json.load(open(f"cache/pool_{POOL[case]}.json"))["pool"]
    sig = json.load(open(f"signatures/{POOL[case]}.json"))["signature"]
    tid = TARGET[case]
    tidx = next((i for i, p in enumerate(pool) if p["id"] == tid), None) if tid else None
    row = {"n_pool": len(pool), "target_id": tid}
    for mode in ["title", "title_abstract"]:
        D = embed.encode([arms.doc_text(p, mode) for p in pool])
        qA = embed.encode([CASES[case]["text"]], is_query=True)[0]
        qN = embed.encode([sig], is_query=True)[0]
        for arm, q in (("A", qA), ("N", qN)):
            sim = D @ q
            k = f"{arm}_{mode}"
            if tidx is None:
                top = [(pool[i]["title"][:70], round(float(sim[i]), 4)) for i in np.argsort(-sim)[:3]]
                row[k] = {"rank": None, "top3": top}
            else:
                row[k] = {"rank": arms.rank_of(sim, tidx),
                          "target_cos": round(float(sim[tidx]), 4),
                          "best_cos": round(float(sim.max()), 4)}
    out[case] = row
    print(case, json.dumps(row, indent=1)[:900], flush=True)
json.dump(out, open("results/arms_AN.json", "w"), indent=1)
