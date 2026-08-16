"""Encode compute: mdbr-leaf-mt (fp32 / int8 / q4 exports) vs jina v5 nano q4.

Same protocol as ``bekko-embedding-bench/scripts/bench_latency.py``: median of
5 for a single short query (batch=1), median of 3 for a 64-doc batch, at 1 and
4 threads, each model's own tokenizer and required prefixes. leaf is
6 layers x 384 hidden x 1536 FFN (~11.2M non-embedding params) vs jina's
12 x 768 x 3072 — roughly 8x the per-token FLOPs — so leaf should land between
bekko-a25m and jina.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
BEKKO_BENCH = HERE.parent / "bekko-embedding-bench"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BEKKO_BENCH / "scripts"))

from jina import JinaQ4Encoder  # noqa: E402
from leaf import LeafMTEncoder  # noqa: E402
from run_partb_leaf import load_blog_chunks, split_chunks  # noqa: E402


def timeit(fn, n_warm: int = 2, n_rep: int = 5) -> float:
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    qs, ds = split_chunks(load_blog_chunks())
    queries = qs[:32]
    docs = ds[:64]
    rows = []

    for threads in (1, 4):
        models = {
            "leaf-mt-fp32": LeafMTEncoder("onnx/model.onnx", threads=threads),
            "leaf-mt-int8": LeafMTEncoder("onnx/model_quantized.onnx", threads=threads),
            "leaf-mt-q4": LeafMTEncoder("onnx/model_q4.onnx", threads=threads),
            "jina-v5-nano-q4": JinaQ4Encoder(threads=threads),
        }
        for name, enc in models.items():
            is_jina = name.startswith("jina")
            qpre = "Query: " if is_jina else \
                "Represent this sentence for searching relevant passages: "
            dpre = "Document: " if is_jina else ""

            def enc_one(t=queries[0], e=enc):
                return e.encode([t], prompt="query")

            def enc_batch(e=enc):
                return e.encode(docs, prompt="document", batch_size=8)

            q_tok = len(enc.tok.encode(qpre + queries[0]).ids)
            d_tok = sum(len(x.ids) for x in enc.tok.encode_batch([dpre + d for d in docs]))

            t_q = timeit(enc_one)
            t_b = timeit(enc_batch, n_warm=1, n_rep=3)
            rows.append({
                "model": name, "threads": threads,
                "query_ms": t_q * 1e3, "query_tokens": q_tok,
                "batch_s": t_b, "docs_per_s": len(docs) / t_b,
                "doc_tokens": d_tok, "tokens_per_s": d_tok / t_b,
                "model_mb": round(enc.model_bytes() / 2**20, 1),
            })
            print(f"{threads}t {name:16s} query {t_q * 1e3:7.1f} ms ({q_tok:3d} tok)   "
                  f"batch {len(docs) / t_b:6.1f} docs/s {d_tok / t_b:8.0f} tok/s",
                  flush=True)
            del enc
        del models

    json.dump(rows, open(HERE / "results_latency_leaf.json", "w"), indent=1)

    print("\n=== jina query latency over each leaf export ===")
    for threads in (1, 4):
        j = next(r for r in rows if r["threads"] == threads
                 and r["model"] == "jina-v5-nano-q4")
        for name in ("leaf-mt-fp32", "leaf-mt-int8", "leaf-mt-q4"):
            a = next(r for r in rows if r["threads"] == threads and r["model"] == name)
            print(f"  {threads}t {name}: {j['query_ms'] / a['query_ms']:.1f}x faster query, "
                  f"{a['tokens_per_s'] / j['tokens_per_s']:.1f}x tokens/s")


if __name__ == "__main__":
    main()
