#!/usr/bin/env python3
"""Token-id fixtures and expected token vectors for the WGSL engine (web/engine.js).

From NanoBEIR SciFact: 16 queries ([CLS] [Q] ... [SEP] + [MASK] to 32) and 64 documents
([CLS] [D] ... [SEP], up to 512 tokens). Expected vectors are run/mcb.py's per-token output before
the skiplist is applied, so they line up row for row with the engine's.

Writes web/fixtures.json ({queries, docs}: lists of id lists) and .work/expect_<model>.npy
(all query tokens then all document tokens, float32 [rows, 128]).
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode import load_set
from mcb import MASK_ID, MoEdernColBERT

ROOT = Path(__file__).resolve().parents[1]


def main():
    _, docs, _, queries, _ = load_set("SciFact")
    m0 = MoEdernColBERT("1.5M")
    q = [s + [MASK_ID] * (32 - len(s)) for s in m0.ids(queries[:16], "query", True)]
    d = m0.ids(docs[:64], "document", True)
    json.dump({"queries": q, "docs": d}, open(ROOT / "web" / "fixtures.json", "w"))
    for name in ["1.5M", "27M"]:
        m = MoEdernColBERT(name)
        rows = []
        for s in q + d:
            ids = torch.tensor([s])
            rows.append(m.forward(ids, torch.ones(1, len(s)))[0].numpy())
        np.save(ROOT / ".work" / f"expect_{name}.npy", np.concatenate(rows).astype(np.float32))
        print(name, sum(len(s) for s in q + d), "rows")


if __name__ == "__main__":
    main()
