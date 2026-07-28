#!/usr/bin/env python3
"""Paraphrase frontier — where lexical leans entirely on the agent's expansion.

The Stage-A/B queries carry domain vocabulary, favourable to BM25. This probe
asks the harder question: lay-phrased conceptual queries with little/no corpus
vocabulary, targeting known gold posts. Embedding gets the raw query; lexical
gets the raw query PLUS agent-crafted expansion (the consuming agent's job). If
lexical holds here, the "agent is the semantic layer" thesis survives the case
embeddings are supposed to win.

Caches the full-float corpus matrix to disk so reruns are fast.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/claude-workspace")
MIRROR = ROOT / ".spokes/jina-v5-nano-mirror"
KB_PATH = ROOT / ".spokes/muninn.austegard.com/knowledge/muninn.kb"
CACHE = ROOT / "experiments/lexical-kb-phase0/.corpus_vecs.npz"
os.environ.setdefault("REMAX_KB_TOKENIZER_PATH", str(MIRROR / "model" / "tokenizer.json"))

sys.path.insert(0, str(ROOT / ".spokes/remax_kb"))
sys.path.insert(0, str(ROOT / "experiments/lexical-kb-phase0"))
from sweep import stem, build, ranked_posts  # noqa: E402
from remax_kb.embedders import JinaONNXEmbedder  # noqa: E402

# Lay-phrased queries (avoid corpus jargon) + agent expansion + topical gold.
PARA = [
    {
        "label": "P1 remember-across-sessions",
        "query": "How can an AI assistant keep what it learns so it remembers next time?",
        "core": ["memory", "persistent"],
        "expand": ["Turso", "libsql", "database", "storage", "across sessions",
                   "structured memory", "recall", "remember"],
        "gold": ["introducing-muninn", "muninn-at-100-days"],
    },
    {
        "label": "P2 demo-to-production",
        "query": "Why do AI agents struggle to go from a working demo to real production use?",
        "core": ["agents", "production"],
        "expand": ["operationalization", "pilots fail", "deployment", "enterprise",
                   "bottleneck", "scaling", "reliability"],
        "gold": ["the-operationalization-gap-why-78-of-agentic-ai-pilots-fail",
                 "from-selective-consolidation-to-bounded-cognitive-state-the-agent-memory-frontie"],
    },
    {
        "label": "P3 weights-as-program",
        "query": "Can a neural network do arithmetic by treating its weights as a tiny program?",
        "core": ["weights", "arithmetic"],
        "expand": ["compiled transformer", "matmul", "polynomial", "executor",
                   "operators", "branchless", "computer"],
        "gold": ["the-matmul-is-the-polynomial", "where-the-computer-meets-the-calculator",
                 "126-million-steps-per-second"],
    },
    {
        "label": "P4 decentralized-twitter",
        "query": "Is the new decentralized alternative to Twitter actually gaining traction?",
        "core": ["decentralized", "social"],
        "expand": ["ATProto", "Bluesky", "protocol", "federation", "network",
                   "adoption", "year two"],
        "gold": ["atproto-year-two-architecture-working-network-waiting",
                 "atproto-the-protocol-thats-quietly-building-a-new-internet"],
    },
    {
        "label": "P5 forgetting-old-info",
        "query": "How should a system decide which old information to forget over time?",
        "core": ["forget", "old information"],
        "expand": ["forgetting", "retention", "decay", "memory", "consolidation",
                   "selection", "pruning", "clocks"],
        "gold": ["three-clocks-for-forgetting",
                 "from-selective-consolidation-to-bounded-cognitive-state-the-agent-memory-frontie"],
    },
]


def load_corpus():
    z = zipfile.ZipFile(KB_PATH)
    chunks = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    return [c["text"] for c in chunks], [stem(c["meta"]["source_path"]) for c in chunks]


def get_matrix(emb, texts):
    if CACHE.exists():
        d = np.load(CACHE)
        if d["mat"].shape[0] == len(texts):
            return d["mat"]
    vecs = []
    for i in range(0, len(texts), 16):
        vecs.append(emb.encode(texts[i:i + 16], prompt="document"))
        if (i // 16) % 15 == 0:
            print(f"  embedded {min(i + 16, len(texts))}/{len(texts)}", flush=True)
    mat = np.vstack(vecs)
    np.savez(CACHE, mat=mat)
    return mat


def score(posts, gold):
    pos = {p: i + 1 for i, p in enumerate(posts)}
    ranks = [pos[g] for g in gold if g in pos]
    first = min(ranks) if ranks else None
    r5 = sum(1 for g in gold if pos.get(g, 999) <= 5) / len(gold)
    r10 = sum(1 for g in gold if pos.get(g, 999) <= 10) / len(gold)
    return first, r5, r10


def emb_posts(emb, mat, posts, query):
    q = emb.encode([query], prompt="query")[0]
    seen, out = set(), []
    for i in np.argsort(-(mat @ q)):
        p = posts[i]
        if p not in seen:
            seen.add(p); out.append(p)
        if len(out) >= 15:
            break
    return out


def lex_posts_rawonly(bundle, q):
    """Lexical WITHOUT expansion — raw query only, to isolate expansion's effect."""
    qq = dict(q, core=[], expand=[])
    return ranked_posts(bundle, qq)


def main() -> int:
    emb = JinaONNXEmbedder()
    texts, posts = load_corpus()
    print(f"corpus matrix ({len(texts)} chunks) …", flush=True)
    mat = get_matrix(emb, texts)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "kb0"
        build(0, bundle)
        print(f"\n{'paraphrase query':<28}{'LEX-raw':>10}{'LEX+exp':>12}{'EMBED':>10}  (rank / R@10)")
        print("-" * 74)
        for q in PARA:
            gold = set(q["gold"])
            rf, _, r10_raw = score(lex_posts_rawonly(bundle, q), gold)
            ef2, _, l10 = score(ranked_posts(bundle, q), gold)
            mf, _, e10 = score(emb_posts(emb, mat, posts, q["query"]), gold)
            print(f"{q['label']:<28}{f'{rf or chr(8212)}/{r10_raw:.2f}':>10}"
                  f"{f'{ef2 or chr(8212)}/{l10:.2f}':>12}{f'{mf or chr(8212)}/{e10:.2f}':>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
