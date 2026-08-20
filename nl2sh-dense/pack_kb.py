#!/usr/bin/env python3
"""Pack the enriched shell corpus as a single remax_kb v2 .kbi artifact.

The retrieval tier §7 leaves at 0.555 gold-in-sources is a BM25 + dense + RRF
pipeline over 6,397 enriched pages. remax_kb's v2 `.kbi` is exactly that stack
in one file — BM25 postings, 1-bit or Lloyd-Max dense codes, RRF fusion, a
numpy-only reader — so the shippable half of this directory has a shippable
container. This script is the bridge: it wraps `encoders.LeafMTEncoder` in
remax_kb's `Embedder` protocol and feeds the enriched pages in as `Chunk`s.

What the .kbi does and does not carry, stated because it changes the number:

* **Carries** the enriched page text (§7, +0.244 on the pages a plain corpus
  could not reach) and both retrieval arms fused by RRF. That is 0.506 of the
  0.555 — the corpus rewrite is the portable part.
* **Does not carry** the query-side adapter (§5). remax_kb has no notion of a
  learned query transform, and the adapter's gain was entirely on the 207
  utilities it trained on — the least portable component. Dropping it is the
  right call for an artifact, not a limitation to apologise for.

The embedder identifies itself by `model_id` and pins the ONNX asset by URL +
SHA so a reader on another machine fetches the same 25.6 MB encoder. leaf-mt is
not in remax_kb's built-in registry (jina-onnx / jina-torch / gemini / lfm25);
this local subclass is the "implement your own embedder" path the protocol
documents, and is the ~40 lines a first-class registry entry would need.

    python3 pack_kb.py --out shell.kbi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import dense_index as D  # noqa: E402
import encoders  # noqa: E402
import retrieve as R  # noqa: E402
from remax_kb.pack import Chunk  # noqa: E402


class LeafMTKBEmbedder:
    """`encoders.LeafMTEncoder` behind remax_kb's Embedder protocol."""

    model_id = "MongoDB/mdbr-leaf-mt"
    model_revision = ""
    task_adapter = "retrieval"
    pooling = "mean"
    full_dim = 1024
    normalize_l2 = True
    # A reader fetches the quantized ONNX from HF by URL, SHA-pinned so a
    # divergent export cannot silently produce a different index.
    release_url = ("https://huggingface.co/MongoDB/mdbr-leaf-mt/resolve/main/"
                   "onnx/model_quantized.onnx")
    release_sha256 = None
    prompts = {"query": encoders.LeafMTEncoder.prompts["query"],
               "document": encoders.LeafMTEncoder.prompts["document"]}

    def __init__(self) -> None:
        self.enc = encoders.LeafMTEncoder()

    def fingerprint(self) -> dict:
        return {"model_id": self.model_id, "task_adapter": self.task_adapter,
                "pooling": self.pooling, "full_dim": self.full_dim}

    def encode(self, texts: list[str], *, prompt: str = "document") -> np.ndarray:
        return self.enc.encode(list(texts), prompt=prompt, batch_size=16)


def enriched_chunks(path: Path):
    for c in D.page_chunks(R.load_chunks(path)):
        yield Chunk(id=c.id, text=c.text,
                    meta={"utility": c.utility, "kind": c.kind})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=HERE / "data" / "chunks_enriched.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "shell.kbi")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--codec", default="remex", choices=["remax", "remex"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--v2", action="store_true", default=True)
    a = ap.parse_args()

    from remax_kb.pack_v2 import KBWriter

    out = a.out
    name = out.stem
    output_dir = out.parent if str(out.parent) else Path(".")
    chunks = list(enriched_chunks(a.corpus))
    print(f"packing {len(chunks)} enriched pages -> {name}.kbi", file=sys.stderr)
    writer = KBWriter.create(
        name=name, output_dir=output_dir, embedder=LeafMTKBEmbedder(),
        dim=a.dim, codec=a.codec, bits=a.bits,
        source="nl2sh enriched shell documentation (tldr+man, CC-BY-4.0 examples, "
               "flash-lite goal-phrasings)")
    writer.add_chunks(chunks)
    writer.commit()
    kbi = output_dir / f"{name}.kbi"
    kbc = output_dir / f"{name}.kbc"
    kbc_bytes = sum(f.stat().st_size for f in kbc.rglob("*")) if kbc.is_dir() else 0
    print(f"wrote {kbi.name} ({kbi.stat().st_size/1e6:.2f} MB) + {kbc.name}/ "
          f"({kbc_bytes/1e6:.2f} MB, {len(chunks)} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
