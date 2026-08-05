"""Encode compute: bekko a8m/a25m vs jina v5 nano q4.

The point of a 7.7M-*active*-parameter encoder is per-token compute, which the
iso-byte retrieval comparison in Part B does not measure at all. bekko-a8m is
4 layers x 384 hidden x 1152 FFN; jina v5 nano is 12 x 768 x 3072 — roughly 12x
the per-token FLOPs — so a quality-per-byte verdict that ignores latency is
answering only half the deployment question.

Two regimes, because remax_kb has two:
  * **query path** — batch=1, single short query, 1 thread. This is what a
    reader pays per query in a constrained container (the claude.ai container is
    1 vCPU / 3 GB).
  * **index path** — batched documents, all cores. This is what building a
    `.kb` costs once.

Fairness notes:
  * The tokenizers differ (bekko 256k multilingual vocab, jina 128k), so the
    same text yields different token counts. Both texts/s and tokens/s are
    reported; tokens/s is the compute-normalized figure.
  * jina requires "Query: " / "Document: " prefixes, bekko none. Prefixes are
    applied, because that is the cost of actually using each model.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bekko import BekkoEncoder  # noqa: E402
from jina import JinaQ4Encoder  # noqa: E402
from run_partb import load_kb, split_chunks  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


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
    chunks = load_kb()
    qs, ds = split_chunks([c["text"] for c in chunks])
    queries = qs[:32]
    docs = ds[:64]
    rows = []

    for threads in (1, 4):
        models = {
            "bekko-a8m": BekkoEncoder("a8m", threads=threads),
            "bekko-a25m": BekkoEncoder("a25m", threads=threads),
            "jina-v5-nano-q4": JinaQ4Encoder(threads=threads),
        }
        for name, enc in models.items():
            is_jina = name.startswith("jina")

            def enc_one(t=queries[0], e=enc, j=is_jina):
                return e.encode([t], prompt="query") if j else e.encode([t])

            def enc_batch(e=enc, j=is_jina):
                return (e.encode(docs, prompt="document", batch_size=8) if j
                        else e.encode(docs, batch_size=8))

            # token counts under each model's own tokenizer
            if is_jina:
                q_tok = len(enc.tok.encode("Query: " + queries[0]).ids)
                d_tok = sum(len(x.ids) for x in
                            enc.tok.encode_batch(["Document: " + d for d in docs]))
            else:
                q_tok = len(enc.tok.encode(queries[0]).ids)
                d_tok = sum(len(x.ids) for x in enc.tok.encode_batch(docs))

            t_q = timeit(enc_one)
            t_b = timeit(enc_batch, n_warm=1, n_rep=3)
            rows.append({
                "model": name, "threads": threads,
                "query_ms": t_q * 1e3,
                "query_tokens": q_tok,
                "batch_s": t_b,
                "docs_per_s": len(docs) / t_b,
                "doc_tokens": d_tok,
                "tokens_per_s": d_tok / t_b,
                "model_mb": round(enc.model_bytes() / 2**20, 1),
            })
            print(f"{threads}t {name:18s} query {t_q * 1e3:7.1f} ms "
                  f"({q_tok:3d} tok)   batch {len(docs) / t_b:6.1f} docs/s "
                  f"{d_tok / t_b:8.0f} tok/s", flush=True)
            del enc
        del models

    json.dump(rows, open(HERE / "results_latency.json", "w"), indent=1)

    print("\n=== speedup of bekko-a8m over jina ===")
    for threads in (1, 4):
        a = next(r for r in rows if r["threads"] == threads and r["model"] == "bekko-a8m")
        j = next(r for r in rows if r["threads"] == threads
                 and r["model"] == "jina-v5-nano-q4")
        print(f"  {threads} thread(s): query latency {j['query_ms'] / a['query_ms']:.1f}x "
              f"faster, throughput {a['tokens_per_s'] / j['tokens_per_s']:.1f}x "
              f"more tokens/s")


if __name__ == "__main__":
    main()
