#!/usr/bin/env python3
"""Can a Workers-AI-deployable embedder match Gemini/Jina retrieval on the muninn
corpus? (Gates a possible Gemini->CF-native swap for muninn search.)

Embedders (all runnable locally; BGE family is on Cloudflare Workers AI):
  * jina      — Jina v5-nano retrieval ONNX (768-d)        [reference, not CF-AI]
  * bge-base  — BAAI/bge-base-en-v1.5  (768-d, on Workers AI, matches Gemini dim)
  * bge-large — BAAI/bge-large-en-v1.5 (1024-d, on Workers AI)

Metric: full-float cosine query->chunk, collapse to distinct posts, R@5/R@10 vs
the model-independent topical gold from lexical-kb-phase0/sweep.py (5 acceptance
queries, 14 gold posts, all present in the 73-post corpus).

Baselines (same gold/corpus, from lexical-kb-phase0): lexical 1.00/1.00,
Jina-fp32 0.90/1.00. Production muninn search currently uses Gemini
(gemini-embedding-001) — no key this session, so Jina is the embedding reference.

n=5 queries: directional, not significant.
"""
from __future__ import annotations

import json, sys, time, zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment, spoke

HERE = Path(__file__).resolve().parent
KB = spoke("muninn.austegard.com") / "knowledge/muninn.kb"
JINA = experiment("jina-int8-remax_kb")
sys.path.insert(0, str(spoke("remax_kb")))
sys.path.insert(0, str(experiment("lexical-kb-phase0")))
from sweep import QUERIES, stem  # noqa: E402

KS = (5, 10)


def load_chunks():
    z = zipfile.ZipFile(KB)
    c = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    return [x["text"] for x in c], [stem(x["meta"]["source_path"]) for x in c]


def cache_or_embed(tag, enc_fn, texts, prompt):
    cache = HERE / f".vec_{tag}_{prompt}.npz"
    if cache.exists():
        m = np.load(cache)["m"]
        if m.shape[0] == len(texts):
            return m
    t0 = time.time()
    m = enc_fn(texts, prompt)
    np.savez(cache, m=m)
    print(f"    {tag}/{prompt}: {m.shape} in {time.time()-t0:.0f}s", flush=True)
    return m


# ---- encoders ---- #
def jina_encoder():
    from remax_kb.embedders import JinaONNXEmbedder
    emb = JinaONNXEmbedder(model_path=JINA / "model.onnx", tokenizer_path=JINA / "tokenizer.json")
    return lambda texts, prompt: emb.encode(texts, prompt=prompt)


def bge_encoder(model_id):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id)
    QINSTR = "Represent this sentence for searching relevant passages: "
    def enc(texts, prompt):
        pre = QINSTR if prompt == "query" else ""
        return np.asarray(m.encode([pre + t for t in texts], normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False,
                                   batch_size=32), dtype=np.float32)
    return enc


def score(dvecs, posts, qvec, gold, ks):
    order = np.argsort(-(dvecs @ qvec))
    seen, ranked = set(), []
    for i in order:
        if posts[i] not in seen:
            seen.add(posts[i]); ranked.append(posts[i])
        if len(ranked) >= max(ks): break
    pos = {p: i + 1 for i, p in enumerate(ranked)}
    return {k: sum(1 for g in gold if pos.get(g, 999) <= k) / len(gold) for k in ks}


def main():
    docs, posts = load_chunks()
    print(f"corpus: {len(docs)} chunks / {len(set(posts))} posts, {len(QUERIES)} queries\n", flush=True)
    models = [
        ("jina", jina_encoder),
        ("bge-base", lambda: bge_encoder("BAAI/bge-base-en-v1.5")),
        ("bge-large", lambda: bge_encoder("BAAI/bge-large-en-v1.5")),
    ]
    rows = []
    for tag, mk in models:
        print(f"=== {tag} ===", flush=True)
        enc = mk()
        D = cache_or_embed(tag, enc, docs, "document")
        Q = cache_or_embed(tag, enc, [q["query"] for q in QUERIES], "query")
        a5 = a10 = 0.0
        for j, q in enumerate(QUERIES):
            r = score(D, posts, Q[j], set(q["gold"]), KS)
            a5 += r[5]; a10 += r[10]
        n = len(QUERIES)
        rows.append((tag, D.shape[1], a5 / n, a10 / n))
        print(f"  {tag} ({D.shape[1]}-d): R@5={a5/n:.3f} R@10={a10/n:.3f}\n", flush=True)

    print("=" * 52)
    print(f"{'embedder':<12}{'dim':>5}{'R@5':>8}{'R@10':>8}")
    for tag, dim, r5, r10 in rows:
        print(f"{tag:<12}{dim:>5}{r5:>8.3f}{r10:>8.3f}")
    print(f"{'lexical*':<12}{'-':>5}{1.000:>8.3f}{1.000:>8.3f}")
    print(f"{'jina-fp32*':<12}{768:>5}{0.900:>8.3f}{1.000:>8.3f}")
    print("* baselines from lexical-kb-phase0 (same gold/corpus). n=5 — directional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
