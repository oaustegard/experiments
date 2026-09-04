"""Score static arms on mini-CTXBench (n=59) against rg, with RRF fusion.

    python3 scripts/run_bench.py potion-code=/home/user/models/potion-code-16M-v2 ...

An arm is NAME=PATH where PATH is a model2vec dir, one of our saved dirs, or
`bekko:<f32 path>` for a precomputed corpus matrix (bekko-a25m reference).
Appends rows to results.json keyed by arm; rg is recomputed once per run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from common import BENCH, HERE, MODELS, StaticTable, evaluate_arm, load_chunks, load_instances, rg_all


def main() -> None:
    inst = load_instances()
    chunks = load_chunks("ast")
    texts = [c["text"] for c in chunks]
    res_p = HERE / "results.json"
    results = json.load(open(res_p)) if res_p.exists() else {}
    t0 = time.time()
    rg_cache = rg_all(inst)
    print(f"rg: {time.time()-t0:.0f}s", flush=True)

    for spec in sys.argv[1:]:
        name, path = spec.split("=", 1)
        t0 = time.time()
        if path.startswith("bekko:"):
            from bekko import BekkoEncoder
            mat = np.fromfile(path[6:], dtype=np.float32).reshape(len(chunks), 384)
            enc = BekkoEncoder("a25m", threads=4)
            q = lambda s: enc.encode([s])[0]
            enc_s = 0.0
        else:
            p = Path(path)
            st = StaticTable.from_dir(p) if (p / "table.npy").exists() else StaticTable.from_model2vec(p)
            mat = st.encode(texts)
            enc_s = time.time() - t0
            q = lambda s, st=st: st.encode([s])[0]
        rows = evaluate_arm(name, q, mat, chunks, rg_cache, inst)
        results[name] = {"rows": rows, "corpus_encode_s": round(enc_s, 1),
                         "dense_r5": float(np.mean([r["dense_r5"] for r in rows])),
                         "dense_r10": float(np.mean([r["dense_r10"] for r in rows])),
                         "rrf_r5": float(np.mean([r["rrf_r5"] for r in rows])),
                         "rrf_r10": float(np.mean([r["rrf_r10"] for r in rows])),
                         "rg_r5": float(np.mean([r["rg_r5"] for r in rows])),
                         "rg_r10": float(np.mean([r["rg_r10"] for r in rows]))}
        r = results[name]
        print(f"{name:28s} dense {r['dense_r5']:.3f}/{r['dense_r10']:.3f}  "
              f"rrf {r['rrf_r5']:.3f}/{r['rrf_r10']:.3f}  rg {r['rg_r5']:.3f}/{r['rg_r10']:.3f}  "
              f"encode {enc_s:.1f}s", flush=True)
        json.dump(results, open(res_p, "w"))


if __name__ == "__main__":
    main()
