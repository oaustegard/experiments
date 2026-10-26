#!/usr/bin/env python3
"""MoEdernColBERT on the eg2-native-small Bluesky task, so it lines up with EG2, Ettin and leaf-ir.

Task: the 61,686 AI-list posts (../eg2-query-student/out/posts.jsonl, from its hydrate_posts.py)
and the 1,498 held-out synthetic queries (synth_test.jsonl in the embstore set
bsky-ai-list/embeddinggemma-2-q4/query-targets-v1); hit@10 = the query's source post in the top 10.
Documents truncated to 128 tokens like EG2's index, exact MaxSim over fp32 token vectors (no
quantization). Stock checkpoints, no in-domain tuning.

Usage: bsky.py [model ...]  (default 1.5M). Writes results/bsky.json (merged per model).
"""
from __future__ import annotations

import json, os, sys, time, urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcb import MoEdernColBERT

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
POSTS = EXP / "eg2-query-student" / "out" / "posts.jsonl"
WORK = HERE.parent / ".work" / "bsky"
RES = HERE.parent / "results" / "bsky.json"
STORE = ("https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/embeddings/"
         "bsky-ai-list/embeddinggemma-2-q4/query-targets-v1/synth_test.jsonl")


def queries():
    p = WORK / "synth_test.jsonl"
    if not p.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(STORE, headers={"User-Agent": "embstore/1"})
        p.write_bytes(urllib.request.urlopen(req).read())
    return [json.loads(l) for l in p.read_text().splitlines()]


def hit_at_10(qE, dE, gold, chunk_tokens=200_000, qb=16):
    """Exact MaxSim top-10 over all docs; qE (nq, 32, 128), dE list of (n_i, 128)."""
    D = torch.from_numpy(np.concatenate(dE))
    seg = torch.from_numpy(np.repeat(np.arange(len(dE)), [len(d) for d in dE]))
    nd = len(dE)
    Q = torch.from_numpy(np.stack(qE))
    top = []
    for a in range(0, len(Q), qb):
        q = Q[a:a + qb].reshape(-1, 128)
        best = torch.full((q.shape[0], nd), -1e9)
        for c in range(0, len(D), chunk_tokens):
            s = q @ D[c:c + chunk_tokens].T
            best.scatter_reduce_(1, seg[c:c + chunk_tokens].expand(q.shape[0], -1), s, "amax")
        sc = best.view(-1, 32, nd).sum(1)
        top.append(sc.topk(10, dim=1).indices)
    top = torch.cat(top).numpy()
    hits = np.array([g in t for g, t in zip(gold, top)], dtype=float)
    top1 = np.array([t[0] == g for g, t in zip(gold, top)], dtype=float)
    return hits, top1


def boot(x, n=2000, seed=0):
    r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(models):
    torch.set_num_threads(4)
    posts = [json.loads(l) for l in POSTS.read_text().splitlines()]
    posts.sort(key=lambda r: r["i"])
    assert [r["i"] for r in posts] == list(range(len(posts)))
    texts = [r["text"] or "" for r in posts]
    qs = queries()
    gold = [q["post"] for q in qs]
    out = json.load(open(RES)) if RES.exists() else {}
    for name in models:
        m = MoEdernColBERT(name, document_length=128)
        t0 = time.time()
        dE = m.encode(texts, "document", batch_size=64)
        td = time.time() - t0
        qE = m.encode([q["q"] for q in qs], "query", batch_size=64)
        ntok = sum(len(d) for d in dE)
        t1 = time.time()
        hits, top1 = hit_at_10(qE, dE, gold)
        out[name] = {"hit@10": boot(hits), "hit@1": float(top1.mean()), "n_queries": len(qs), "n_docs": len(texts),
                     "missing_post_text": sum(1 for t in texts if not t),
                     "doc_tokens_mean": ntok / len(texts), "bytes_per_doc_fp16": 2 * 128 * ntok / len(texts),
                     "cpu_docs_per_s": len(texts) / td, "score_s": time.time() - t1}
        print(name, json.dumps(out[name]), flush=True)
        RES.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(RES, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["1.5M"])
