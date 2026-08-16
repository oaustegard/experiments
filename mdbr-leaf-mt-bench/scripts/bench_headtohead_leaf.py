"""Byte-budget head-to-head on mdbr-leaf-mt: the vendor's own compression
menu (MRL truncation, int8, binary, binary+asymmetric rescore) vs remex
Lloyd-Max quantization and remax stacked sign bits — including compositions
(truncate *and* quantize), so the output is a Pareto frontier rather than
three disjoint ladders.

Follows ``bekko-embedding-bench/scripts/bench_headtohead.py`` but on the
model's own recommended menu:

  matryoshka   fp32, cut coordinates            bytes = d * 4
  vendor int8  per-dim int8 in [-1, +1]         bytes = d      (card's recipe)
  vendor bin   sign bits, symmetric Hamming     bytes = d / 8  (card's recipe)
  bin-asym     sign-bit docs, fp32 query        bytes = d / 8  (the card's
               rescore trick collapsed to exhaustive asymmetric scoring —
               an upper bound on what binary+rescore can recover)
  remex        Lloyd-Max b bits/coord, rotated  bytes = CompressedVectors.nbytes
  remax        k stacked sign bits, rotated,    bytes = d * k / 8
               corpus-centered, Hamming

Every arm is also run at truncated dims (1024/512/256/128/64), so
quantization composes with MRL in both families. Embeddings come from the
int8 ONNX export — the deployment artifact this experiment recommends —
encoded once and reused across all arms. Vectors are L2-normalized per dim
(matryoshka()), so vendor int8 ranges [-1, +1] are the card's suggested
defaults. Paired McNemar + bootstrap CI on the claims the chart invites.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
BEKKO_BENCH = HERE.parent / "bekko-embedding-bench"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BEKKO_BENCH / "scripts"))
sys.path.insert(0, "/home/user/remax/src")

import remex  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

from leaf import LeafMTEncoder, matryoshka  # noqa: E402
from run_partb_leaf import (  # noqa: E402
    boot_ci, hits_at, load_blog_chunks, load_code_chunks, mcnemar_exact,
    split_chunks,
)

DIMS = [1024, 512, 256, 128, 64]
FULL = 1024


def recall_rows(sims: np.ndarray) -> dict:
    n = sims.shape[0]
    order = np.argsort(-sims, axis=1)
    return {f"r@{k}": float(np.mean([i in order[i, :k] for i in range(n)]))
            for k in (1, 10, 50)}


def hamming_sims(qc: np.ndarray, dc: np.ndarray) -> np.ndarray:
    """Negative Hamming distance as a similarity matrix (small n, exhaustive)."""
    out = np.empty((qc.shape[0], dc.shape[0]), dtype=np.int32)
    for i in range(qc.shape[0]):
        out[i] = -np.bitwise_count(np.bitwise_xor(qc[i], dc)).sum(axis=1)
    return out


def pack_signs(v: np.ndarray) -> np.ndarray:
    """(N, d) floats -> (N, d/8) uint8 sign bits."""
    return np.packbits(v > 0, axis=1)


def main() -> None:
    blog = load_blog_chunks()
    qs, ds = split_chunks(blog)
    cq, cd = split_chunks(load_code_chunks(len(blog)))

    enc = LeafMTEncoder("onnx/model_quantized.onnx", threads=4)
    E = {}
    for dist, (Q, D) in (("blog", (qs, ds)), ("code", (cq, cd))):
        E[dist] = (enc.encode(Q, prompt="query", batch_size=8),
                   enc.encode(D, prompt="document", batch_size=8))
        print(f"encoded {dist}", flush=True)
    del enc

    rows, hits = [], {}

    def add(dist: str, arm: str, d: int, param, nbytes: float, sims: np.ndarray) -> None:
        r = recall_rows(sims)
        rows.append({"dist": dist, "arm": arm, "dim": d, "param": param,
                     "bytes": nbytes, **r})
        hits[(dist, arm, d, param)] = hits_at(sims, 10)

    for dist, (qf, df) in E.items():
        for d in DIMS:
            q = matryoshka(qf, d if d < FULL else None)
            v = matryoshka(df, d if d < FULL else None)

            # fp32 matryoshka
            add(dist, "matryoshka-fp32", d, None, d * 4, q @ v.T)

            # vendor int8, card's [-1, +1] ranges on both sides
            step = 2.0 / 255.0
            qi = np.clip(np.round((q + 1.0) / step) - 128, -128, 127)
            vi = np.clip(np.round((v + 1.0) / step) - 128, -128, 127)
            add(dist, "vendor-int8", d, None, d, qi @ vi.T)

            # vendor binary, symmetric Hamming
            add(dist, "vendor-binary", d, None, d / 8,
                hamming_sims(pack_signs(q), pack_signs(v)))

            # binary docs, fp32 query (asymmetric — rescore upper bound)
            add(dist, "binary-asym", d, None, d / 8, q @ np.sign(v).T)

            # remex Lloyd-Max at this width
            for bits in (1, 2, 4):
                qz = remex.Quantizer(d=d, bits=bits, seed=0)
                cv = qz.encode(v)
                xh = qz.decode(cv)
                xh = xh / np.clip(np.linalg.norm(xh, axis=1, keepdims=True), 1e-9, None)
                add(dist, "remex", d, bits, cv.nbytes / v.shape[0], q @ xh.T)

            # remax stacked sign bits at this width
            for k in (1, 2, 4):
                sq = StackedSignBitQuantizer(d=d, k=k, seed=0).fit(v)
                add(dist, "remax", d, k, d * k / 8,
                    hamming_sims(sq.encode(q), sq.encode(v)))
        print(f"{dist}: arms done", flush=True)

    # paired claims on the blog distribution (the parent bench's Pareto corpus)
    claims = []

    def claim(label: str, a_key, b_key, dist="blog"):
        a, b = hits[(dist, *a_key)], hits[(dist, *b_key)]
        diff = float(a.mean() - b.mean())
        lo, hi = boot_ci(a, b)
        n01, n10, p = mcnemar_exact(a, b)
        claims.append({"claim": label, "dist": dist, "delta": diff, "ci_lo": lo,
                       "ci_hi": hi, "wins": n01, "losses": n10, "p": p,
                       "significant": bool(p < 0.05)})
        print(f"{label:<58}{diff:+7.3f}  [{lo:+.3f},{hi:+.3f}]  "
              f"{n01:>3}/{n10:<4}{p:8.4f}", flush=True)

    for dist in ("blog", "code"):
        claim(f"[{dist}] remax k=1 vs vendor binary @128B (d=1024)",
              ("remax", 1024, 1), ("vendor-binary", 1024, None), dist)
        claim(f"[{dist}] remex 1-bit vs vendor binary (d=1024)",
              ("remex", 1024, 1), ("vendor-binary", 1024, None), dist)
        claim(f"[{dist}] remex 1-bit vs remax k=1 (d=1024)",
              ("remex", 1024, 1), ("remax", 1024, 1), dist)
        claim(f"[{dist}] binary-asym vs vendor binary (d=1024)",
              ("binary-asym", 1024, None), ("vendor-binary", 1024, None), dist)
        claim(f"[{dist}] remex 2-bit @1024 (~260B) vs MRL d=64 fp32 (256B)",
              ("remex", 1024, 2), ("matryoshka-fp32", 64, None), dist)
        claim(f"[{dist}] remex 2-bit @1024 (~260B) vs fp32 @1024 (4096B)",
              ("remex", 1024, 2), ("matryoshka-fp32", 1024, None), dist)

    json.dump({"n": len(qs), "rows": rows, "claims": claims},
              open(HERE / "results_headtohead_leaf.json", "w"), indent=1)
    print("wrote results_headtohead_leaf.json", flush=True)

    # console Pareto table, blog
    print("\nblog R@10 by payload bytes/vector:")
    sub = sorted([r for r in rows if r["dist"] == "blog"], key=lambda r: r["bytes"])
    for r in sub:
        tag = f"{r['arm']} d={r['dim']}" + (f" b/k={r['param']}" if r["param"] else "")
        print(f"{r['bytes']:>8.1f}  {r['r@10']:.3f}  {tag}")


if __name__ == "__main__":
    main()
