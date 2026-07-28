#!/usr/bin/env python3
"""Stage A — lexical chunk-size sweep on the real muninn corpus.

Builds a lexical KB at several chunk sizes (via creating-kb's JS builder) and, for
each #76 acceptance query with agent-crafted expansion, measures retrieval of the
*gold posts* (defined topically from post descriptions, independent of body term
frequency, so gold doesn't pre-favor lexical matching).

Metric per (size, query):
  rank   — position of the first gold post in the distinct-post ranking
  R@5    — fraction of gold posts whose chunks appear in the top-5 distinct posts
  R@10   — same at top-10

Tests the null hypothesis: lexical BM25 tolerates bigger chunks than embeddings.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/home/user/claude-workspace")
CORPUS = ROOT / "experiments/lexical-kb-phase0/corpus"
SCRIPTS = ROOT / ".spokes/claude-skills/creating-kb/scripts"
SIZES = [500, 1500, 4000, 0]  # 0 = whole document

# Gold = source_path stems judged on-topic from the post descriptions.
QUERIES = [
    {
        "label": "Q1 centered-simhash",
        "query": "How does centered SimHash differ from random projection?",
        "core": ["centered", "simhash", "random projection"],
        "expand": ["sign bit", "hyperplane", "binary quantization", "hamming", "LSH",
                   "hashing trick", "sign-bit extraction", "centering", "mean subtraction",
                   "isotropic", "coarse index", "one bit"],
        "gold": ["when-matryoshka-does-buy-you-sign-bit-compression",
                 "your-embedding-has-a-free-coarse-index-in-it",
                 "three-gigs-to-search-a-hundred-million-papers"],
    },
    {
        "label": "Q2 memory-storage",
        "query": "What does Muninn use as memory storage?",
        "core": ["memory", "storage"],
        "expand": ["Turso", "libsql", "sqlite", "database", "persistent memory",
                   "store", "four-layer architecture", "database wrapper", "schema"],
        "gold": ["introducing-muninn", "muninn-at-100-days"],
    },
    {
        "label": "Q3 failure-modes",
        "query": "What are the failure modes of agentic AI memory systems?",
        "core": ["failure modes", "agentic", "memory"],
        "expand": ["fail", "pilots", "operationalization", "drift", "forgetting",
                   "consolidation", "amnesia", "null", "bottleneck", "production"],
        "gold": ["the-operationalization-gap-why-78-of-agentic-ai-pilots-fail",
                 "from-selective-consolidation-to-bounded-cognitive-state-the-agent-memory-frontie",
                 "null-induced-amnesia"],
    },
    {
        "label": "Q4 compiled-transformer-mojo",
        "query": "Compiled transformer executor and Mojo speed",
        "core": ["compiled transformer", "executor", "Mojo"],
        "expand": ["steps per second", "matmul", "polynomial", "speed", "throughput",
                   "branchless", "opcode", "kernel", "CPU", "executes programs"],
        "gold": ["126-million-steps-per-second", "the-matmul-is-the-polynomial",
                 "where-the-computer-meets-the-calculator"],
    },
    {
        "label": "Q5 atproto-bluesky",
        "query": "ATProto and Bluesky architecture",
        "core": ["ATProto", "Bluesky", "architecture"],
        "expand": ["AT protocol", "PDS", "did:plc", "firehose", "lexicon",
                   "federation", "protocol", "network", "publishing", "feed"],
        "gold": ["atproto-the-protocol-thats-quietly-building-a-new-internet",
                 "atproto-year-two-architecture-working-network-waiting",
                 "building-atproto-publishing-utilities-from-scratch-no-sdk-required"],
    },
]


def stem(source_path: str) -> str:
    return source_path.split("/")[-1].rsplit(".", 1)[0]


def build(size: int, out: Path) -> dict:
    r = subprocess.run(
        ["node", str(SCRIPTS / "build_lexkb.js"), str(CORPUS),
         "--ext", "txt", "--out", str(out), "--target-chars", str(size)],
        capture_output=True, text=True, check=True)
    idx = json.loads((out / "index.json").read_text())
    return {"chunks": idx["N"], "avgdl": idx["avgdl"]}


def ranked_posts(bundle: Path, q: dict, k_chunks: int = 40) -> list[str]:
    cmd = ["python3", str(bundle / "search.py"), "--index", str(bundle),
           "--query", q["query"], "--k", str(k_chunks)]
    for c in q["core"]:
        cmd += ["--core", c]
    for e in q["expand"]:
        cmd += ["--expand", e]
    hits = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)["hits"]
    seen, posts = set(), []
    for h in hits:
        sp = stem(h["meta"]["source_path"])
        if sp not in seen:
            seen.add(sp)
            posts.append(sp)
    return posts


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        for size in SIZES:
            bundle = Path(tmp) / f"kb{size}"
            info = build(size, bundle)
            label = "whole-doc" if size == 0 else str(size)
            print(f"\n=== target_chars={label}  ({info['chunks']} chunks, avgdl={info['avgdl']:.0f} tok) ===")
            print(f"{'query':<28}{'rank':>5}{'R@5':>7}{'R@10':>7}   gold(found@10/total)")
            print("-" * 78)
            agg5 = agg10 = 0.0
            for q in QUERIES:
                posts = ranked_posts(bundle, q)
                gold = set(q["gold"])
                pos = {p: i + 1 for i, p in enumerate(posts)}
                ranks = [pos[g] for g in gold if g in pos]
                first = min(ranks) if ranks else None
                r5 = sum(1 for g in gold if pos.get(g, 999) <= 5) / len(gold)
                r10 = sum(1 for g in gold if pos.get(g, 999) <= 10) / len(gold)
                agg5 += r5; agg10 += r10
                found10 = sum(1 for g in gold if pos.get(g, 999) <= 10)
                print(f"{q['label']:<28}{(first if first else '—'):>5}{r5:>7.2f}{r10:>7.2f}   {found10}/{len(gold)}")
            n = len(QUERIES)
            print("-" * 78)
            print(f"{'MEAN':<28}{'':>5}{agg5/n:>7.2f}{agg10/n:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
