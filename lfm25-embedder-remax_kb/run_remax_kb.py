"""Pack remax_kb's tiny_corpus with the local LFM2.5-Embedding-350M and run
the same 3 retrieval queries the Jina-torch test uses, end to end through the
1-bit binary .kb pipeline.

Proves: a small in-process CPU embedder produces correct topical retrieval
through remax_kb's pack -> SRHT rotate -> 1-bit -> Hamming-search path.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

WS = Path("/home/user/claude-workspace")
sys.path.insert(0, str(WS / ".spokes" / "remax" / "src"))
sys.path.insert(0, str(WS / ".spokes" / "remax_kb"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from remax_kb import pack, KB
from lfm25_embedder import LFM25Embedder

CORPUS = WS / ".spokes" / "remax_kb" / "examples" / "tiny_corpus"
OUT = Path(__file__).resolve().parent / "tiny_lfm25.kb"

# Same (query, expected source file) triples as tests/test_retrieval.py
CASES = [
    ("What is a faction?", "federalist_10_factions.txt"),
    ("Why must government be controlled by checks and balances?", "federalist_51_checks.txt"),
    ("What was said at the dedication of the battlefield?", "gettysburg.txt"),
]

emb = LFM25Embedder()

t0 = time.time()
pack(CORPUS, OUT, embedder=emb, dim=256, k=8, seed=0,
     source_description="tiny_corpus packed with LFM2.5-Embedding-350M")
pack_s = time.time() - t0
print(f"packed {OUT.name}: {OUT.stat().st_size/1024:.1f} KB in {pack_s:.1f}s "
      f"(includes one-time model load)")

kb = KB.open(OUT)

passed = 0
for q, expected in CASES:
    t0 = time.time()
    hits = kb.search(q, embedder=emb, k=3)
    dt = time.time() - t0
    srcs = []
    for dist, chunk in hits:
        sp = chunk.get("meta", {}).get("source_path", chunk.get("id"))
        srcs.append((Path(str(sp)).name, dist))
    ok = any(expected in s for s, _ in srcs)
    passed += ok
    print(f"\nQ: {q}  ({dt*1000:.0f} ms)")
    print(f"   expect: {expected}  -> {'PASS' if ok else 'FAIL'}")
    for name, dist in srcs:
        mark = "*" if expected in name else " "
        print(f"   {mark} [{dist:>4}] {name}")

print(f"\n=== {passed}/{len(CASES)} topical top-3 retrievals correct ===")
sys.exit(0 if passed == len(CASES) else 1)
