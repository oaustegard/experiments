"""Run the issue-#76 acceptance queries against the freshly built muninn.kb."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "JINA_V5_NANO_MIRROR_PATH",
    "/home/user/claude-workspace/.spokes/jina-v5-nano-mirror",
)

from remax_kb.embedders import JinaTorchEmbedder  # noqa: E402
from remax_kb.read import KB  # noqa: E402


KB_PATH = "/home/user/claude-workspace/.spokes/muninn.austegard.com/knowledge/muninn.kb"

QUERIES = [
    "How does centered SimHash differ from random projection?",
    "What does Muninn use as memory storage?",
    "What are the failure modes of agentic AI memory systems?",
    # Two extras for sanity:
    "Compiled transformer executor and Mojo speed",
    "ATProto and Bluesky architecture",
]


def main() -> int:
    kb = KB.open(KB_PATH)
    emb = JinaTorchEmbedder(task_adapter="retrieval")
    print(f"opened {KB_PATH}; chunk_count={kb.manifest.corpus.chunk_count}")
    print()

    for q in QUERIES:
        print(f"=== query: {q}")
        hits = kb.search(q, embedder=emb, k=3)
        for rank, (dist, chunk) in enumerate(hits, 1):
            meta = chunk.get("meta", {})
            title = meta.get("title", "")[:80]
            src = meta.get("source_path", "")
            text = chunk["text"].replace("\n", " ")
            preview = text[:160] + ("…" if len(text) > 160 else "")
            print(f"  #{rank} hamming={dist}  [{src}]  {title}")
            print(f"      {preview}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
