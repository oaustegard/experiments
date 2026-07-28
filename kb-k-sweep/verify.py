#!/usr/bin/env python3
"""Verify dim=512/k=4 beats shipped dim=256/k=8 on REAL queries.

The grid in dim_sweep.py was self-retrieval (doc-vs-doc). This closes the loop on
the side that actually matters for live search: the RETRIEVAL_QUERY adapter.

For a set of realistic queries:
  - gold   = float-768 query·doc cosine top-10 (the ideal dense ranking the
             binarizer is trying to reconstruct)
  - each config's DENSE ranking comes from the real v2 reader (_dense_search:
    center by mean -> truncate to dim -> StackedSimHash -> Hamming)
  - each config's HYBRID ranking comes from KB.search (dense + BM25, RRF)

Reports recall@10 of each config's dense and hybrid ranking vs the float gold,
plus a couple of qualitative side-by-sides.
"""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITE = ROOT / ".spokes" / "muninn.austegard.com"
sys.path[:0] = [str(ROOT / ".spokes" / "remax_kb"), str(ROOT / ".spokes" / "remax" / "src")]

BUILD = HERE / "build"
CONFIGS = {"256/k8 (shipped)": BUILD / "d256_k8" / "muninn_d256_k8.kbi",
           "512/k4 (candidate)": BUILD / "d512_k4" / "muninn_d512_k4.kbi",
           "768/k2 (candidate)": BUILD / "d768_k2" / "muninn_d768_k2.kbi"}
TOPK = 10

QUERIES = [
    "how does centered SimHash differ from random projection",
    "why does one bit beat two for embedding compression",
    "search with no search server architecture",
    "Matryoshka embeddings and sign-bit compression",
    "BM25 versus dense retrieval tradeoffs",
    "how to build a portable knowledge base file",
    "Cloudflare worker for semantic search",
    "binary quantization recall tradeoff",
    "stacked sign bit quantizer hamming distance",
    "AST parsing for code navigation",
    "tree-sitter symbol lookup",
    "Bluesky firehose sampling",
    "RRF reciprocal rank fusion",
    "jina embeddings v5 nano",
    "Gemini embedding API output dimensionality",
    "how Muninn boots on Claude Code on the web",
    "spoke and hub repository architecture",
    "Turso libsql memory store",
    "SPECTER2 scientific paper embeddings",
    "bridge discovery between math and CS theory",
    "Lloyd-Max quantization codebook",
    "cosine similarity LSH Charikar",
    "vector database alternatives for small teams",
    "chunking documents for retrieval",
    "ONNX runtime embedder in a constrained container",
    "what is remax",
    "deterministic bit-identical knowledge base",
    "popcount hamming distance fast scan",
    "blog post about sign bit compression",
    "centering embeddings by corpus mean",
    "recall at k evaluation protocol",
    "matryoshka truncation dimension reduction",
    "uploading files to Claude Code on the web",
    "session start hook for web sessions",
    "static CDN range fetch chunk store",
    "tombstone based mutation in an index",
    "merkle verification per chunk sha256",
    "embedding quantization for retrieval quality",
    "how many bits per dimension for good recall",
    "query encoder versus document encoder asymmetry",
]


def load_bm():
    spec = importlib.util.spec_from_file_location("bm", SITE / "scripts" / "build_muninn_kb.py")
    bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
    return bm


def main():
    d = np.load(HERE / "embeddings.npz", allow_pickle=True)
    docs = d["vecs"].astype(np.float32)
    docs = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)

    bm = load_bm()
    real = bm.GeminiGatewayEmbedder(
        account_id=os.environ["CF_ACCOUNT_ID"],
        gateway_id=os.environ["CF_GATEWAY_ID"],
        gateway_token=os.environ["CF_API_TOKEN"])
    # embed every query ONCE (single batch), then serve readers from cache so
    # the per-query reader calls hit no network (avoids gateway 429s).
    qv = real.encode(QUERIES, prompt="query").astype(np.float32)
    qn = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

    bb = importlib.util.spec_from_file_location("bb", HERE / "build_both.py")
    bbm = importlib.util.module_from_spec(bb); bb.loader.exec_module(bbm)
    embedder = bbm.CachedEmbedder(real, {q: qv[i] for i, q in enumerate(QUERIES)})

    # gold: float-768 query·doc cosine top-10 (rows == corpus order)
    sims = qn @ docs.T
    gold = [set(np.argpartition(-sims[i], TOPK)[:TOPK]) for i in range(len(QUERIES))]

    from remax_kb.read_v2 import KB

    def recall(rows_per_q):
        return np.mean([len(set(rows_per_q[i]) & gold[i]) / TOPK for i in range(len(QUERIES))])

    results = {}
    readers = {}
    for label, path in CONFIGS.items():
        kb = KB.open(path); readers[label] = kb
        dense_rows, hybrid_rows = [], []
        for q in QUERIES:
            dh = kb._dense_search(q, embedder)[:TOPK]
            dense_rows.append([h.row for h in dh])
            hh = kb.search(q, embedder=embedder, k=TOPK)
            hybrid_rows.append([h.row for h in hh])
        results[label] = {
            "dense_R@10_vs_float": recall(dense_rows),
            "hybrid_R@10_vs_float": recall(hybrid_rows),
        }
        print(f"{label:22s}  dense R@10={results[label]['dense_R@10_vs_float']:.4f}"
              f"   hybrid R@10={results[label]['hybrid_R@10_vs_float']:.4f}")

    base = "256/k8 (shipped)"; cand = "512/k4 (candidate)"
    dd = results[cand]["dense_R@10_vs_float"] - results[base]["dense_R@10_vs_float"]
    print(f"\nDENSE delta (candidate - shipped): {dd:+.4f}")

    # qualitative side-by-side for a few queries (top-3 hybrid, fetched text)
    print("\n=== qualitative (top-3 hybrid) ===")
    for q in QUERIES[:3]:
        print(f"\nQ: {q}")
        for label in CONFIGS:
            kb = readers[label]
            hits = kb.search_and_fetch(q, embedder=embedder, k=3)
            print(f"  [{label}]")
            for h in hits:
                snippet = (h.text or "").replace("\n", " ")[:90]
                print(f"    - {h.chunk_id}  {snippet}")

    import json
    (HERE / "verify.json").write_text(json.dumps(
        {"n_queries": len(QUERIES), "topk": TOPK, "results": results,
         "dense_delta": dd}, indent=2))
    print("\nwrote verify.json")


if __name__ == "__main__":
    main()
