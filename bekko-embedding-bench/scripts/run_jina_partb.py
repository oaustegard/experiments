"""Jina v5 nano official-q4 arm of Part B, on the identical splits as bekko."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jina import JinaQ4Encoder  # noqa: E402
from run_partb import load_kb, recall_at_k, split_chunks  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    chunks = load_kb()
    qs, ds = split_chunks([c["text"] for c in chunks])
    code = json.load(open(HERE / "chunks_ast.json"))
    step = max(1, len(code) // len(qs))
    cq, cd = split_chunks([c["text"] for c in code[::step]][: len(qs)])

    enc = JinaQ4Encoder(threads=2)
    rows = []
    for dist, (Q, D) in (("blog", (qs, ds)), ("code", (cq, cd))):
        qv = enc.encode(Q, prompt="query", batch_size=8)
        dv = enc.encode(D, prompt="document", batch_size=8)
        for d in (768, 384, 256, 128, 64):
            def trunc(x):
                v = x[:, :d]
                return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
            sims = trunc(qv) @ trunc(dv).T
            rows.append({
                "model": "jina-v5-nano-q4", "dist": dist, "dim": d,
                "r@1": recall_at_k(sims, 1), "r@5": recall_at_k(sims, 5),
                "r@10": recall_at_k(sims, 10), "bytes_per_vec_fp32": d * 4,
            })
            print(f"jina/{dist} d={d}: R@1={rows[-1]['r@1']:.3f} "
                  f"R@10={rows[-1]['r@10']:.3f}", flush=True)
        np.save(HERE / f"jina_{dist}_q.npy", qv)
        np.save(HERE / f"jina_{dist}_d.npy", dv)
    json.dump({"retrieval": rows, "model_mb": round(enc.model_bytes() / 2**20, 1)},
              open(HERE / "results_jina.json", "w"), indent=1)
    print("wrote results_jina.json", flush=True)


if __name__ == "__main__":
    main()
