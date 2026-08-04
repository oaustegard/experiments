"""Is the ~57 ms stacked-SimHash step per query actually necessary?

``remax_kb.read.KB._stacked_simhash_encode`` constructs a
``StackedSignBitQuantizer(d, k, seed)`` on **every call**, and that constructor
builds k Haar rotations by QR of a d x d Gaussian. All three parameters come
from the manifest and never change for an opened `.kb`, so the rotations are
identical every query — the work is pure repetition.

This measures the shipped path against a one-line cached variant, and checks the
cached variant returns **identical codes** (so it is a speedup, not a behaviour
change). It also varies query length, because encode scales with tokens while
the binarizer tax does not.
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
from bench_kb_path import BekkoKBEmbedder, JinaKBEmbedder, median  # noqa: E402

HERE = Path(__file__).resolve().parents[1]

QUERIES = {
    "short (9 tok)": "What is centered SimHash?",
    "typical (14 tok)": "How does centered SimHash differ from random projection?",
    "long (52 tok)": (
        "I am trying to understand why one-bit sign-bit codes outperform two-bit "
        "Lloyd-Max quantization on retrieval tasks, particularly for SPECTER2 "
        "embeddings, and whether that result transfers to other encoders or is "
        "specific to the geometry of citation-trained representation spaces."
    ),
}


def main() -> None:
    from remax_kb.read import KB

    rows = []
    for name, make in (("bekko-a8m", lambda: BekkoKBEmbedder("a8m", threads=1)),
                       ("jina-v5-nano-q4", lambda: JinaKBEmbedder(threads=1))):
        emb = make()
        kb = KB.open(HERE / f"bench_{name}.kb")

        # ── the one-line fix: build the quantizer once per opened KB ─────────
        from remax import StackedSignBitQuantizer
        b = kb.manifest.binarizer
        cached_q = StackedSignBitQuantizer(d=b.dim, k=b.k, seed=b.seed)

        def patched(X, _q=cached_q):
            return _q.encode(X)

        # correctness before speed: cached codes must equal shipped codes
        probe = np.random.default_rng(0).normal(size=(4, b.dim)).astype(np.float32)
        same = np.array_equal(kb._stacked_simhash_encode(probe), patched(probe))

        for qlabel, q in QUERIES.items():
            t_shipped = median(lambda: kb.search(q, embedder=emb, k=5))
            t_enc = median(lambda: emb.encode([q], prompt="query"))

            orig = kb._stacked_simhash_encode
            kb._stacked_simhash_encode = patched
            t_fixed = median(lambda: kb.search(q, embedder=emb, k=5))
            hits_fixed = kb.search(q, embedder=emb, k=5)
            kb._stacked_simhash_encode = orig
            hits_shipped = kb.search(q, embedder=emb, k=5)

            identical = [h[1]["id"] for h in hits_fixed] == [
                h[1]["id"] for h in hits_shipped]
            rows.append({
                "model": name, "query": qlabel,
                "shipped_ms": t_shipped * 1e3, "fixed_ms": t_fixed * 1e3,
                "encode_ms": t_enc * 1e3,
                "codes_identical": bool(same), "hits_identical": identical,
            })
            r = rows[-1]
            print(f"{name:16s} {qlabel:17s} shipped {r['shipped_ms']:7.1f} ms -> "
                  f"cached {r['fixed_ms']:6.1f} ms  (encode {r['encode_ms']:6.1f}) "
                  f"codes={'=' if same else 'DIFFER'} hits={'=' if identical else 'DIFFER'}",
                  flush=True)
        del emb, kb

    json.dump(rows, open(HERE / "results_kbfix.json", "w"), indent=1)

    print("\n=== bekko-a8m over jina, per query length ===")
    for qlabel in QUERIES:
        a = next(r for r in rows if r["model"] == "bekko-a8m" and r["query"] == qlabel)
        j = next(r for r in rows if r["model"] != "bekko-a8m" and r["query"] == qlabel)
        print(f"  {qlabel:17s} shipped {j['shipped_ms'] / a['shipped_ms']:5.1f}x   "
              f"cached {j['fixed_ms'] / a['fixed_ms']:5.1f}x")


if __name__ == "__main__":
    main()
