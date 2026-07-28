#!/usr/bin/env python3
"""LFM2.5-Embedding-350M (local CPU) on the real muninn corpus, scored the
SAME way as lexical-kb-phase0/stage_b.py's Jina embedding ceiling:
full-float cosine over the embedding KB's own chunks, chunk hits collapsed to
distinct posts, R@5/R@10 against the identical topical gold from sweep.py.

Produces a number directly comparable to:
    lexical (agent-expand + BM25):  R@5 1.00 / R@10 1.00
    Jina v5-nano full-float cosine: R@5 0.90 / R@10 1.00
"""
from __future__ import annotations

import json, sys, time, zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment, spoke
KB_PATH = spoke("muninn.austegard.com") / "knowledge/muninn.kb"
sys.path.insert(0, str(experiment("lexical-kb-phase0")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep import QUERIES, stem  # noqa: E402
from lfm25_embedder import LFM25Embedder  # noqa: E402


def load_corpus_chunks():
    z = zipfile.ZipFile(KB_PATH)
    chunks = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    return [c["text"] for c in chunks], [stem(c["meta"]["source_path"]) for c in chunks]


CACHE = Path(__file__).resolve().parent / ".corpus_vecs.npz"


def embed_corpus(emb, texts, batch=16):
    if CACHE.exists():
        d = np.load(CACHE)
        if d["mat"].shape[0] == len(texts):
            print(f"  loaded cached corpus vectors {d['mat'].shape}", flush=True)
            return d["mat"]
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.append(emb.encode(texts[i:i + batch], prompt="document"))
        done = min(i + batch, len(texts))
        print(f"  embedded {done}/{len(texts)}", flush=True)
    mat = np.vstack(vecs)
    np.savez(CACHE, mat=mat)
    return mat


def emb_ranked_posts(emb, mat, posts, query, cap=15):
    q = emb.encode([query], prompt="query")[0]
    order = np.argsort(-(mat @ q))
    seen, out = set(), []
    for i in order:
        p = posts[i]
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) >= cap:
            break
    return out


def score(posts, gold):
    pos = {p: i + 1 for i, p in enumerate(posts)}
    ranks = [pos[g] for g in gold if g in pos]
    first = min(ranks) if ranks else None
    r5 = sum(1 for g in gold if pos.get(g, 999) <= 5) / len(gold)
    r10 = sum(1 for g in gold if pos.get(g, 999) <= 10) / len(gold)
    return first, r5, r10


def main():
    emb = LFM25Embedder()
    texts, posts = load_corpus_chunks()
    print(f"corpus: {len(texts)} chunks / {len(set(posts))} posts", flush=True)
    t0 = time.time()
    mat = embed_corpus(emb, texts)
    print(f"embedded full corpus in {time.time()-t0:.1f}s "
          f"({len(texts)/(time.time()-t0):.1f} chunks/s, CPU)\n", flush=True)

    print(f"{'query':<28}{'rank':>6}{'R@5':>7}{'R@10':>7}")
    print("-" * 48)
    a5 = a10 = 0.0
    for q in QUERIES:
        gold = set(q["gold"])
        f, r5, r10 = score(emb_ranked_posts(emb, mat, posts, q["query"]), gold)
        a5 += r5; a10 += r10
        print(f"{q['label']:<28}{(f if f else '—'):>6}{r5:>7.2f}{r10:>7.2f}")
    n = len(QUERIES)
    print("-" * 48)
    print(f"{'MEAN':<28}{'':>6}{a5/n:>7.2f}{a10/n:>7.2f}")
    print(f"\ncompare:  lexical 1.00/1.00   Jina full-float 0.90/1.00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
