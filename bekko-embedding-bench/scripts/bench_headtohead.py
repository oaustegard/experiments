"""Head-to-head: the vendor's Matryoshka trimming vs remex/remax quantization.

One question, no combinations: **at a given bytes/vector, which single approach
gives better retrieval?**

  Matryoshka arm  keep fp32, cut coordinates   -> bytes = d * 4
  remex arm       keep all 384 coords, cut bits/coord -> bytes = 384 * bits / 8
  remax arm       keep all 384 coords, k stacked sign bits -> bytes = 384 * k / 8

The two families barely overlap on the vendor's supported ladder (Matryoshka's
cheapest supported tier, d=64, is 256 B; remex at 384d only reaches 388 B at
8 bits), so the Matryoshka arm is extended below d=64 to make the curves
comparable. Sub-64 tiers are **off the vendor's spec** (the card lists 256/128/64)
and are marked as such — whether they hold up is itself part of the answer.

Bytes are payload per vector. Both families need the encoder; remex/remax
additionally need a seed-derived rotation, which remax_kb regenerates rather
than ships, so it costs latency rather than bytes (measured separately).

Also answers the follow-up: is a wide-N Matryoshka tier a usable coarse filter
in front of remax?
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
from run_partb import load_kb, split_chunks  # noqa: E402

import remex  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
FULL = 384
VENDOR_TIERS = {384, 256, 128, 64}


def recall_at(sims: np.ndarray, k: int) -> float:
    n = sims.shape[0]
    order = np.argsort(-sims, axis=1)[:, :k]
    return float(np.mean([i in order[i] for i in range(n)]))


def hamming_recall(qc: np.ndarray, dc: np.ndarray, k: int) -> float:
    hits = 0
    for i in range(qc.shape[0]):
        d = np.bitwise_count(np.bitwise_xor(qc[i], dc)).sum(axis=1)
        if i in np.argsort(d, kind="stable")[:k]:
            hits += 1
    return hits / qc.shape[0]


def main() -> None:
    chunks = load_kb()
    qs, ds = split_chunks([c["text"] for c in chunks])
    rows = []

    for variant in ("a8m", "a25m"):
        enc = BekkoEncoder(variant, threads=4)
        qf = enc.encode(qs, batch_size=8)
        df = enc.encode(ds, batch_size=8)

        # ── arm 1: Matryoshka trimming, fp32 ────────────────────────────────
        for d in (384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8):
            q, v = matryoshka(qf, d), matryoshka(df, d)
            sims = q @ v.T
            rows.append({
                "variant": variant, "arm": "matryoshka", "param": d,
                "bytes": d * 4, "r@1": recall_at(sims, 1),
                "r@10": recall_at(sims, 10), "r@50": recall_at(sims, 50),
                "vendor_supported": d in VENDOR_TIERS,
            })

        # ── arm 2: remex quantization at FULL width ─────────────────────────
        qfull, dfull = matryoshka(qf, None), matryoshka(df, None)
        for bits in (1, 2, 3, 4, 8):
            qz = remex.Quantizer(d=FULL, bits=bits, seed=0)
            xhat = qz.decode(qz.encode(dfull))
            xhat = xhat / np.clip(np.linalg.norm(xhat, axis=1, keepdims=True), 1e-9, None)
            sims = qfull @ xhat.T
            rows.append({
                "variant": variant, "arm": "remex@384", "param": bits,
                "bytes": FULL * bits // 8, "r@1": recall_at(sims, 1),
                "r@10": recall_at(sims, 10), "r@50": recall_at(sims, 50),
                "vendor_supported": True,
            })

        # ── arm 3: remax stacked sign bits at FULL width ────────────────────
        for k in (1, 2, 3, 4, 6, 8):
            sq = StackedSignBitQuantizer(d=FULL, k=k, seed=0).fit(dfull)
            qc, dc = sq.encode(qfull), sq.encode(dfull)
            rows.append({
                "variant": variant, "arm": "remax@384", "param": k,
                "bytes": FULL * k // 8,
                "r@1": hamming_recall(qc, dc, 1), "r@10": hamming_recall(qc, dc, 10),
                "r@50": hamming_recall(qc, dc, 50), "vendor_supported": True,
            })
        del enc

    json.dump(rows, open(HERE / "results_headtohead.json", "w"), indent=1)

    for variant in ("a8m", "a25m"):
        sub = [r for r in rows if r["variant"] == variant]
        print(f"\n{'=' * 74}\nbekko-{variant}: R@10 by payload bytes/vector\n{'=' * 74}")
        print(f"{'bytes':>7}  {'Matryoshka (fp32)':>26}  {'remex@384':>18}  {'remax@384':>18}")
        allb = sorted({r["bytes"] for r in sub})
        for b in allb:
            cells = {}
            for arm in ("matryoshka", "remex@384", "remax@384"):
                m = [r for r in sub if r["arm"] == arm and r["bytes"] == b]
                if m:
                    r = m[0]
                    tag = "d=%d%s" % (r["param"], "" if r["vendor_supported"] else "*")
                    cells[arm] = f"{r['r@10']:.3f} ({tag})" if arm == "matryoshka" \
                        else f"{r['r@10']:.3f} ({'b' if 'remex' in arm else 'k'}={r['param']})"
            print(f"{b:>7}  {cells.get('matryoshka', ''):>26}  "
                  f"{cells.get('remex@384', ''):>18}  {cells.get('remax@384', ''):>18}")
        print("  * = below the vendor's supported Matryoshka tiers (256/128/64)")


if __name__ == "__main__":
    main()
