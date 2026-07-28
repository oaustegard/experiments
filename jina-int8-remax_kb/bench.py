#!/usr/bin/env python3
"""Jina v5-nano: fp32 ONNX vs int8-quantized ONNX, on the real muninn corpus.

Same methodology as lexical-kb-phase0/stage_b.py: full-float* cosine over the
embedding KB's own 1238 chunks (73 posts), chunk hits -> distinct posts,
R@5/R@10 vs the topical gold from sweep.py. (*the int8 model still emits fp32
vectors; only its weights are quantized.)

Reports, per variant: model size, corpus-embed wall time + chunks/s, R@5/R@10.

Baselines for reference:
    lexical (agent-expand + BM25):  R@5 1.00 / R@10 1.00
    Jina v5-nano fp32 (prior run):  R@5 0.90 / R@10 1.00
    LFM2.5-Embedding-350M:          R@5 0.73 / R@10 0.83
"""
from __future__ import annotations

import json, sys, time, zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment, spoke
KB_PATH = spoke("muninn.austegard.com") / "knowledge/muninn.kb"
TOKENIZER = HERE / "tokenizer.json"
sys.path.insert(0, str(spoke("remax_kb")))
sys.path.insert(0, str(experiment("lexical-kb-phase0")))

from sweep import QUERIES, stem  # noqa: E402
from remax_kb.embedders import JinaONNXEmbedder  # noqa: E402

VARIANTS = [
    ("fp32", HERE / "model.onnx"),
    ("int8", HERE / "model.int8.onnx"),
]


def load_corpus_chunks():
    z = zipfile.ZipFile(KB_PATH)
    chunks = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    return [c["text"] for c in chunks], [stem(c["meta"]["source_path"]) for c in chunks]


def embed_corpus(emb, texts, cache: Path, batch=32):
    if cache.exists():
        d = np.load(cache)
        if d["mat"].shape[0] == len(texts):
            print(f"  cached {d['mat'].shape}", flush=True)
            return d["mat"], 0.0
    t0 = time.time()
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.append(emb.encode(texts[i:i + batch], prompt="document"))
        print(f"  embedded {min(i+batch,len(texts))}/{len(texts)}", flush=True)
    mat = np.vstack(vecs)
    dt = time.time() - t0
    np.savez(cache, mat=mat)
    return mat, dt


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
    r5 = sum(1 for g in gold if pos.get(g, 999) <= 5) / len(gold)
    r10 = sum(1 for g in gold if pos.get(g, 999) <= 10) / len(gold)
    return r5, r10


def main():
    texts, posts = load_corpus_chunks()
    print(f"corpus: {len(texts)} chunks / {len(set(posts))} posts\n", flush=True)
    summary = []
    for tag, model_path in VARIANTS:
        if not model_path.exists():
            print(f"[{tag}] missing {model_path.name}, skipping"); continue
        size_mb = model_path.stat().st_size / 1e6
        print(f"=== {tag} ({size_mb:.0f} MB) ===", flush=True)
        emb = JinaONNXEmbedder(model_path=model_path, tokenizer_path=TOKENIZER)
        mat, dt = embed_corpus(emb, texts, HERE / f".corpus_vecs_{tag}.npz")
        a5 = a10 = 0.0
        rows = []
        for q in QUERIES:
            gold = set(q["gold"])
            r5, r10 = score(emb_ranked_posts(emb, mat, posts, q["query"]), gold)
            a5 += r5; a10 += r10
            rows.append((q["label"], r5, r10))
        n = len(QUERIES)
        rate = f"{len(texts)/dt:.1f} ch/s" if dt else "cached"
        summary.append((tag, size_mb, dt, rate, a5/n, a10/n))
        for label, r5, r10 in rows:
            print(f"  {label:<28}{r5:>6.2f}{r10:>7.2f}")
        print(f"  {'MEAN':<28}{a5/n:>6.2f}{a10/n:>7.2f}   embed {dt:.0f}s ({rate})\n", flush=True)

    print("=" * 60)
    print(f"{'variant':<8}{'size MB':>9}{'embed s':>9}{'rate':>11}{'R@5':>7}{'R@10':>7}")
    for tag, size_mb, dt, rate, m5, m10 in summary:
        print(f"{tag:<8}{size_mb:>9.0f}{dt:>9.0f}{rate:>11}{m5:>7.2f}{m10:>7.2f}")
    print("ref: lexical 1.00/1.00 · Jina fp32 prior 0.90/1.00 · LFM2.5 0.73/0.83")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
