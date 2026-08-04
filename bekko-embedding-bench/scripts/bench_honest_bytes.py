"""Honest bytes: does "quantization dominates truncation" survive real accounting?

Two errors in the first pass, both flattering the codecs:

1. Per-vector payload was hand-computed as ``dim*bits/8``, which omits the
   **float32 norms remex stores separately** (4 B/vec). Negligible at d=384
   (48 -> 52 B), but +50% at d=64 @ 1-bit (8 -> 12 B). Fixed here by asking
   remex for ``CompressedVectors.nbytes``.

2. Shared structures were counted as free. **Matryoshka truncation ships
   nothing** — it is literally a slice — while remex needs a d x d rotation and
   remax needs k of them. At small n that dominates: this repo already found the
   same trap once (`remex-vs-higgs-ablation`: "counting the shared codebook the
   vector arm costs 52.5 B/vector at 4 bits against a 50 B payload... needs
   ~350k vectors to amortize"). Ignoring it biases every comparison toward the
   codec and against the baseline that needs no side data.

The wrinkle that makes this interesting rather than just an erratum: those
rotations are **seed-derived**. You either ship them (bytes) or regenerate them
(the 53 ms/query constant measured in section 6). They are the same object priced
two ways, so the honest frontier depends on which you pay in.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("REMEX_ROOT", "/home/user/remex"))
sys.path.insert(0, os.environ.get("REMAX_ROOT", "/home/user/remax/src"))
from bekko import BekkoEncoder, matryoshka  # noqa: E402
from run_partb import load_kb, recall_at_k, split_chunks  # noqa: E402

import remex  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
CORPUS_SIZES = [179, 1238, 10_000, 100_000, 1_000_000]
F32 = 4


def side_bytes(codec: str, dim: int, k_or_bits: int) -> int:
    """Bytes of shared structure a reader needs, if materialized rather than
    regenerated from the seed."""
    if codec == "fp32":
        return 0  # Matryoshka truncation is a slice: nothing to ship
    if codec == "remex":
        # one d x d rotation + a 2^bits-level codebook per dimension tier
        return dim * dim * F32 + (2 ** k_or_bits) * F32
    if codec == "remax":
        return k_or_bits * dim * dim * F32  # k stacked rotations
    raise ValueError(codec)


def main() -> None:
    chunks = load_kb()
    qs, ds = split_chunks([c["text"] for c in chunks])
    rows = []

    for variant in ("a8m", "a25m"):
        enc = BekkoEncoder(variant, threads=4)
        qv_full = enc.encode(qs, batch_size=8)
        dv_full = enc.encode(ds, batch_size=8)
        n_probe = len(ds)
        for dim in (384, 256, 128, 64):
            q = matryoshka(qv_full, dim if dim < 384 else None)
            d = matryoshka(dv_full, dim if dim < 384 else None)

            rows.append({
                "variant": variant, "dim": dim, "codec": "fp32", "param": 32,
                "r@10": recall_at_k(q @ d.T, 10),
                "payload_b": dim * F32, "side_b": 0,
            })
            for bits in (1, 2, 3, 4, 8):
                qz = remex.Quantizer(d=dim, bits=bits, seed=0)
                cv = qz.encode(d)
                xhat = qz.decode(cv)
                nrm = np.linalg.norm(xhat, axis=1, keepdims=True)
                sims = q @ (xhat / np.clip(nrm, 1e-9, None)).T
                rows.append({
                    "variant": variant, "dim": dim, "codec": "remex", "param": bits,
                    "r@10": recall_at_k(sims, 10),
                    # remex's OWN accounting: includes the separately-stored norms
                    "payload_b": cv.nbytes / n_probe,
                    "naive_payload_b": dim * bits / 8,
                    "side_b": side_bytes("remex", dim, bits),
                })
            for k in (1, 2, 4, 8):
                sq = StackedSignBitQuantizer(d=dim, k=k, seed=0).fit(d)
                qc, dc = sq.encode(q), sq.encode(d)
                hits = 0
                for i in range(len(qs)):
                    dist = np.bitwise_count(np.bitwise_xor(qc[i], dc)).sum(axis=1)
                    if i in np.argsort(dist, kind="stable")[:10]:
                        hits += 1
                rows.append({
                    "variant": variant, "dim": dim, "codec": "remax", "param": k,
                    "r@10": hits / len(qs),
                    "payload_b": dc.shape[1], "side_b": side_bytes("remax", dim, k),
                })
        del enc
    json.dump(rows, open(HERE / "results_honest_bytes.json", "w"), indent=1)

    # ── frontiers under three accounting regimes ────────────────────────────
    def frontier(rs, total_fn):
        best = {}
        for r in rs:
            b = round(total_fn(r))
            if b not in best or r["r@10"] > best[b]["r@10"]:
                best[b] = r
        out, top = [], -1.0
        for b in sorted(best):
            if best[b]["r@10"] > top:
                top = best[b]["r@10"]
                out.append((b, best[b]))
        return out

    sub = [r for r in rows if r["variant"] == "a25m"]
    print("=== a25m frontier: PAYLOAD ONLY (what the first pass reported) ===")
    for b, r in frontier(sub, lambda r: r["payload_b"]):
        print(f"  {b:8d} B  R@10 {r['r@10']:.3f}  {r['codec']}(d={r['dim']},p={r['param']})")

    for n in CORPUS_SIZES:
        print(f"\n=== a25m frontier: TRUE bytes/vec, side data MATERIALIZED, n={n:,} ===")
        f = frontier(sub, lambda r, n=n: r["payload_b"] + r["side_b"] / n)
        for b, r in f[:8]:
            print(f"  {b:8d} B  R@10 {r['r@10']:.3f}  "
                  f"{r['codec']}(d={r['dim']},p={r['param']})"
                  f"   [payload {r['payload_b']:.0f} + side {r['side_b'] / n:.0f}]")

    json.dump({"corpus_sizes": CORPUS_SIZES}, open(HERE / "_hb_meta.json", "w"))


if __name__ == "__main__":
    main()
