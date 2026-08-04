"""End-to-end latency of remax_kb's ACTUAL v1 query path, per encoder.

The previous benchmark timed encoders in isolation. This one drives
``remax_kb.read.KB.search`` — the code a reader really runs — and decomposes it,
because an encoder advantage only survives if encode actually dominates the path:

    KB.open  ->  embedder.encode  ->  center + truncate  ->  stacked-SimHash
             ->  hamming_scan     ->  top_k              ->  chunk fetch

Both `.kb` files are packed here with identical binarizer params (dim/k/seed) and
identical chunks, so the only difference is the embedder. Timed at 1 thread,
which is what a constrained reader container has.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("REMAX_KB_ROOT", "/home/user/remax_kb"))
sys.path.insert(0, os.environ.get("REMAX_ROOT", "/home/user/remax/src"))
from bekko import BekkoEncoder  # noqa: E402
from jina import JinaQ4Encoder  # noqa: E402
from run_partb import load_kb  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
KB_DIM, KB_K, KB_SEED = 256, 8, 0


class BekkoKBEmbedder:
    """bekko wrapped in the Embedder protocol remax_kb's packer/reader expect."""

    model_revision = ""
    task_adapter = "retrieval"
    pooling = "mean"
    full_dim = 384
    normalize_l2 = True
    release_url = None
    release_sha256 = None
    prompts = {"query": "", "document": ""}  # bekko uses no prefixes

    def __init__(self, variant: str = "a8m", threads: int | None = None) -> None:
        self.variant = variant
        self.model_id = f"hotchpotch/bekko-embedding-v1-{variant}"
        self._enc = BekkoEncoder(variant, threads=threads)

    def fingerprint(self) -> dict:
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def encode(self, texts: list[str], *, prompt: str = "document") -> np.ndarray:
        return self._enc.encode(texts, batch_size=8)


class JinaKBEmbedder:
    """The incumbent, same protocol, so both paths are byte-for-byte comparable."""

    model_id = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    model_revision = "ac5d898c8d382b17167c33e5c8af644a3519b47d"
    task_adapter = "retrieval"
    pooling = "last-token"
    full_dim = 768
    normalize_l2 = True
    release_url = None
    release_sha256 = None
    prompts = {"query": "Query: ", "document": "Document: "}

    def __init__(self, threads: int | None = None) -> None:
        self._enc = JinaQ4Encoder(threads=threads)

    def fingerprint(self) -> dict:
        return {
            "model_id": self.model_id,
            "task_adapter": self.task_adapter,
            "pooling": self.pooling,
            "full_dim": self.full_dim,
        }

    def encode(self, texts: list[str], *, prompt: str = "document") -> np.ndarray:
        return self._enc.encode(texts, prompt=prompt, batch_size=8)


def median(fn, warm: int = 2, rep: int = 7) -> float:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    from remax_kb.pack import Chunk, pack
    from remax_kb.read import KB
    from remax_kb._hamming import hamming_scan, top_k

    chunks = load_kb()
    corpus = [
        Chunk(id=c["id"], text=c["text"], meta=c.get("meta", {})) for c in chunks
    ]
    QUERIES = [
        "How does centered SimHash differ from random projection?",
        "Why does one-bit beat two-bit on retrieval?",
        "What is persistent memory for an assistant?",
    ]

    embedders = {
        "bekko-a8m": lambda: BekkoKBEmbedder("a8m", threads=1),
        "bekko-a25m": lambda: BekkoKBEmbedder("a25m", threads=1),
        "jina-v5-nano-q4": lambda: JinaKBEmbedder(threads=1),
    }

    rows = []
    for name, make in embedders.items():
        emb = make()
        kb_path = HERE / f"bench_{name}.kb"
        if not kb_path.exists():
            t0 = time.perf_counter()
            pack(corpus, kb_path, embedder=emb, dim=KB_DIM, k=KB_K, seed=KB_SEED,
                 source_description=f"latency bench ({name})", batch_size=8)
            build_s = time.perf_counter() - t0
            print(f"{name}: packed {kb_path.name} in {build_s:.1f}s "
                  f"({kb_path.stat().st_size / 1024:.0f} KB)", flush=True)
        else:
            build_s = float("nan")

        t_open = median(lambda p=kb_path: KB.open(p), warm=1, rep=3)
        kb = KB.open(kb_path)
        q = QUERIES[0]

        # whole path, as a reader calls it
        t_search = median(lambda: kb.search(q, embedder=emb, k=5))

        # decomposition
        t_encode = median(lambda: emb.encode([q], prompt="query"))
        vec = emb.encode([q], prompt="query")
        m = kb.manifest

        def _center():
            c = vec[0].astype(np.float32) - m.binarizer.mean_vector
            return c[: m.binarizer.dim]

        t_center = median(_center, rep=15)
        trunc = _center()[None, :]
        t_simhash = median(lambda: kb._stacked_simhash_encode(trunc))
        qcode = kb._stacked_simhash_encode(trunc)[0]
        t_scan = median(lambda: hamming_scan(kb.codes, qcode), rep=15)
        d = hamming_scan(kb.codes, qcode)
        t_topk = median(lambda: top_k(d, k=5), rep=15)

        rows.append({
            "model": name, "n_chunks": len(kb), "kb_kb": kb_path.stat().st_size / 1024,
            "build_s": build_s,
            "open_ms": t_open * 1e3, "search_ms": t_search * 1e3,
            "encode_ms": t_encode * 1e3, "center_ms": t_center * 1e3,
            "simhash_ms": t_simhash * 1e3, "scan_ms": t_scan * 1e3,
            "topk_ms": t_topk * 1e3,
        })
        r = rows[-1]
        print(f"{name:18s} search {r['search_ms']:7.1f} ms  = encode {r['encode_ms']:6.1f} "
              f"+ simhash {r['simhash_ms']:5.1f} + scan {r['scan_ms']:.2f} "
              f"+ topk {r['topk_ms']:.2f} (open {r['open_ms']:.1f})", flush=True)

        # sanity: the path returns sensible hits
        hits = kb.search(QUERIES[0], embedder=emb, k=3)
        print(f"   top hit: {hits[0][1]['id'][:64]} (d={hits[0][0]})", flush=True)
        del emb, kb

    json.dump(rows, open(HERE / "results_kbpath.json", "w"), indent=1)

    a = next(r for r in rows if r["model"] == "bekko-a8m")
    j = next(r for r in rows if r["model"] == "jina-v5-nano-q4")
    print(f"\nend-to-end search speedup, bekko-a8m over jina: "
          f"{j['search_ms'] / a['search_ms']:.1f}x "
          f"({a['search_ms']:.1f} vs {j['search_ms']:.1f} ms)")
    print(f"encoder-only speedup was 12.9x; non-encode overhead is "
          f"{a['search_ms'] - a['encode_ms']:.1f} ms (a8m) / "
          f"{j['search_ms'] - j['encode_ms']:.1f} ms (jina)")


if __name__ == "__main__":
    main()
