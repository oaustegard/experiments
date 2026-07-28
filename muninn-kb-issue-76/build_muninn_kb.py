"""Build muninn.kb — full corpus pack for muninn.austegard.com.

Spec (from issue #76):
  dim=256, k=8, task_adapter=retrieval, pooling=last-token, seed=0
  source_description="muninn.austegard.com — blog + perch + scratch"
  Use JinaTorchEmbedder (torch + peft + safetensors via jina-v5-nano-mirror).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Import sibling extractor.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_chunks import extract_chunks  # noqa: E402

from remax_kb.embedders import JinaTorchEmbedder  # noqa: E402
from remax_kb.pack import pack  # noqa: E402
from remax_kb.read import KB  # noqa: E402


SITE_ROOT_DEFAULT = "/home/user/claude-workspace/.spokes/muninn.austegard.com"
JINA_MIRROR_DEFAULT = "/home/user/claude-workspace/.spokes/jina-v5-nano-mirror"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-root", default=SITE_ROOT_DEFAULT)
    ap.add_argument("--jina-mirror", default=JINA_MIRROR_DEFAULT)
    ap.add_argument("--out", default=str(Path(SITE_ROOT_DEFAULT) / "knowledge" / "muninn.kb"))
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    # JinaTorchEmbedder resolves the loader via $JINA_V5_NANO_MIRROR_PATH.
    os.environ["JINA_V5_NANO_MIRROR_PATH"] = args.jina_mirror

    print(f"[1/4] Extracting chunks from {args.site_root} …", flush=True)
    t0 = time.time()
    chunks = extract_chunks(args.site_root)
    n_posts = len({c.meta["source_path"] for c in chunks})
    print(
        f"      {len(chunks)} chunks across {n_posts} posts "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    print(f"[2/4] Loading JinaTorchEmbedder (task=retrieval) …", flush=True)
    t0 = time.time()
    embedder = JinaTorchEmbedder(task_adapter="retrieval")
    # Force a tiny encode to materialize weights so the next stage's first
    # batch is uncluttered.
    _ = embedder.encode(["warm-up"], prompt="document")
    print(f"      ready ({time.time() - t0:.1f}s)", flush=True)

    print(
        f"[3/4] Packing → {args.out} "
        f"(dim={args.dim}, k={args.k}, seed={args.seed}, batch={args.batch_size}) …",
        flush=True,
    )
    t0 = time.time()
    out_path = pack(
        chunks,
        args.out,
        embedder=embedder,
        dim=args.dim,
        k=args.k,
        seed=args.seed,
        source_description="muninn.austegard.com — blog + perch + scratch",
        batch_size=args.batch_size,
    )
    dt = time.time() - t0
    rate = len(chunks) / dt if dt else 0.0
    print(
        f"      wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB) "
        f"in {dt:.1f}s ({rate:.1f} chunks/sec)",
        flush=True,
    )

    print(f"[4/4] Reopening to verify …", flush=True)
    kb = KB.open(out_path)
    m = kb.manifest
    print(f"      chunk_count: {m.corpus.chunk_count}")
    print(f"      build_hash:  {m.corpus.build_hash[:16]}…")
    print(f"      source:      {m.corpus.source}")
    print(
        f"      embedder:    {m.embedder.model_id} / "
        f"{m.embedder.task_adapter} / pooling={m.embedder.pooling} / "
        f"full_dim={m.embedder.full_dim}"
    )
    print(
        f"      binarizer:   dim={m.binarizer.dim}, k={m.binarizer.k}, "
        f"seed={m.binarizer.seed}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
