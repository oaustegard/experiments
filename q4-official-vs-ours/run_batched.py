"""Batched runner for the q4 head-to-head (remax_kb#23).

The repo's bench/bench_q4_official_vs_ours.py is faithful in its metric math but
calls emb.encode(all_docs) in a SINGLE onnxruntime batch — at 1500-2000 docs the
attention-mask Expand tries to allocate ~26 GB and OOMs on a 15 GB box.

This driver imports the bench module verbatim and reuses its corpus loader,
subsampler, and every metric (ndcg, retrieval_scores, fidelity_to_fp32). The ONLY
change is encode_all: mini-batch the forward pass. Per-row output is identical to
the one-shot batch — last-token pooling indexes true token lengths and L2-norm is
per-row, so batch boundaries don't change any vector (attention is masked; padding
length is irrelevant to the pooled row).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".spokes" / "remax_kb"))

import bench.bench_q4_official_vs_ours as B  # noqa: E402
from remax_kb.embedders import JinaONNXEmbedder  # noqa: E402


def encode_batched(model_path, tok, docs, queries, max_length=512, batch=16):
    emb = JinaONNXEmbedder(model_path=model_path, tokenizer_path=tok, max_length=max_length)
    emb._load()

    def run(texts, prompt):
        out = []
        n = len(texts)
        t = time.time()
        for i in range(0, n, batch):
            out.append(emb.encode(texts[i : i + batch], prompt=prompt))
            if (i // batch) % 10 == 0:
                done = min(i + batch, n)
                rate = done / max(1e-6, time.time() - t)
                print(f"      [{prompt}] {done}/{n}  {rate:.1f} txt/s", file=sys.stderr, flush=True)
        return np.vstack(out) if out else np.zeros((0, emb.full_dim), dtype=np.float32)

    t0 = time.time()
    dvec = run(list(docs.values()), "document")
    qvec = run(list(queries.values()), "query")
    return dvec, qvec, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--ours-q4", required=True)
    ap.add_argument("--official-q4", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--n-docs", type=int, default=1500)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    docs, queries, qrels = B.load_nfcorpus(a.corpus)
    docs, queries, qrels = B.subsample(docs, queries, qrels, a.n_docs, a.n_queries)
    doc_ids, query_ids = list(docs), list(queries)
    print(f"corpus: {len(docs)} docs, {len(queries)} queries", flush=True)

    out = {}
    for name, path in [("fp32", a.fp32), ("ours-q4", a.ours_q4), ("official-q4", a.official_q4)]:
        dvec, qvec, dt = encode_batched(path, a.tokenizer, docs, queries, batch=a.batch)
        ndcg, _, _ = B.retrieval_scores(dvec, qvec, doc_ids, query_ids, qrels)
        sib = Path(str(path) + "_data")
        mb = (Path(path).stat().st_size + (sib.stat().st_size if sib.exists() else 0)) / 1e6
        out[name] = dict(dvec=dvec, qvec=qvec, ndcg=ndcg, mb=mb, secs=dt)
        print(f"  {name:12s} nDCG@10={ndcg:.4f}  size={mb:6.1f}MB  encode={dt:5.1f}s", flush=True)

    ref = out["fp32"]
    print("\nfidelity to fp32 (per-doc cosine / recall@10-vs-fp32kNN / Spearman-rho):")
    for name in ("ours-q4", "official-q4"):
        m = out[name]
        cos, rec, rho = B.fidelity_to_fp32(m["qvec"], m["dvec"], ref["qvec"], ref["dvec"])
        print(f"  {name:12s} cos={cos:.4f}  recall@10={rec:.4f}  rho={rho:.4f}")

    print("\nverdict table:")
    print(f"  {'model':12s} {'nDCG@10':>8s} {'dNDCG_vs_fp32':>14s} {'MB':>7s} {'enc_s':>6s}")
    for name in ("fp32", "ours-q4", "official-q4"):
        m = out[name]
        print(f"  {name:12s} {m['ndcg']:8.4f} {m['ndcg']-ref['ndcg']:14.4f} {m['mb']:7.1f} {m['secs']:6.1f}")


if __name__ == "__main__":
    main()
