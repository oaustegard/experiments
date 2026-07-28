#!/usr/bin/env python3
"""Stage B — lexical (agent-expansion + BM25) vs embedding, head-to-head.

Embedding side here is the *full-float* Jina v5-nano retrieval (cosine over the
768-d L2-normalized vectors), i.e. the embedding ceiling — strictly stronger than
the deployed 1-bit muninn.kb codes, so this is a conservative test for lexical.
It uses only the ONNX embedder (no `remax` quantizer). Corpus chunks are the
embedding KB's own 500-char chunks; lexical uses the creating-kb whole-doc bundle.

Both map chunk hits -> distinct posts, scored R@5 / R@10 / rank-of-first-gold
against the identical topical gold from sweep.py.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke  # noqa: E402

HERE = Path(__file__).resolve().parent
MIRROR = spoke("jina-v5-nano-mirror")
KB_PATH = spoke("muninn.austegard.com") / "knowledge/muninn.kb"
os.environ.setdefault("REMAX_KB_TOKENIZER_PATH", str(MIRROR / "model" / "tokenizer.json"))

sys.path.insert(0, str(spoke("remax_kb")))
sys.path.insert(0, str(HERE))
from sweep import QUERIES, stem, build, ranked_posts  # noqa: E402
from remax_kb.embedders import JinaONNXEmbedder  # noqa: E402


def load_corpus_chunks() -> tuple[list[str], list[str]]:
    z = zipfile.ZipFile(KB_PATH)
    chunks = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    texts = [c["text"] for c in chunks]
    posts = [stem(c["meta"]["source_path"]) for c in chunks]
    return texts, posts


def embed_corpus(emb, texts: list[str], batch: int = 16) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch):
        vecs.append(emb.encode(texts[i:i + batch], prompt="document"))
        if (i // batch) % 10 == 0:
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)} chunks", flush=True)
    return np.vstack(vecs)


def emb_ranked_posts(emb, mat: np.ndarray, posts: list[str], query: str) -> list[str]:
    q = emb.encode([query], prompt="query")[0]
    order = np.argsort(-(mat @ q))
    seen, out = set(), []
    for i in order:
        p = posts[i]
        if p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= 15:
            break
    return out


def score(posts: list[str], gold: set[str]) -> tuple:
    pos = {p: i + 1 for i, p in enumerate(posts)}
    ranks = [pos[g] for g in gold if g in pos]
    first = min(ranks) if ranks else None
    r5 = sum(1 for g in gold if pos.get(g, 999) <= 5) / len(gold)
    r10 = sum(1 for g in gold if pos.get(g, 999) <= 10) / len(gold)
    return first, r5, r10


def main() -> int:
    print("loading Jina v5-nano ONNX embedder …", flush=True)
    emb = JinaONNXEmbedder()
    texts, posts = load_corpus_chunks()
    print(f"embedding {len(texts)} corpus chunks (full-float, document prompt) …", flush=True)
    mat = embed_corpus(emb, texts)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "kb0"
        build(0, bundle)  # lexical whole-doc

        print(f"\n{'query':<28}{'LEX rank/R@5/R@10':>22}{'EMB rank/R@5/R@10':>22}")
        print("-" * 74)
        agg = dict(l5=0.0, l10=0.0, e5=0.0, e10=0.0)
        for q in QUERIES:
            gold = set(q["gold"])
            lf, l5, l10 = score(ranked_posts(bundle, q), gold)
            ef, e5, e10 = score(emb_ranked_posts(emb, mat, posts, q["query"]), gold)
            agg["l5"] += l5; agg["l10"] += l10; agg["e5"] += e5; agg["e10"] += e10
            print(f"{q['label']:<28}{f'{lf or chr(8212)}/{l5:.2f}/{l10:.2f}':>22}"
                  f"{f'{ef or chr(8212)}/{e5:.2f}/{e10:.2f}':>22}")
        n = len(QUERIES)
        print("-" * 74)
        lex_mean = f"{agg['l5'] / n:.2f}/{agg['l10'] / n:.2f}"
        emb_mean = f"{agg['e5'] / n:.2f}/{agg['e10'] / n:.2f}"
        print(f"{'MEAN (R@5/R@10)':<28}{lex_mean:>22}{emb_mean:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
