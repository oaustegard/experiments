"""Iso-byte comparison of three ways to spend a byte budget on bekko vectors.

The axes are orthogonal (2eba5b5b): Matryoshka cuts *coordinates*, remex cuts
*bits per coordinate*, remax stacks *sign-bit signatures*. The question this
answers is not "which is best" in the abstract but: **at a fixed bytes/vector,
which spend wins?** Truncating to 128 fp32 dims and quantizing 384 dims to
1 bit both cost ~48-512 B; only a Pareto frontier says which to ship.

Also settles the embedder-specific bit-depth finding from one-bit-beats-two:
1-bit beat 2-bit on SPECTER2 and inverted on Jina. Where does bekko land?

METHODS.md, non-negotiable: score cosine as cosine — divide by ||x_hat||.
Ranking by bare inner product q.x_hat rewards codecs whose reconstruction norm
is constant by construction (a 1-bit scalar quantizer emits +/-c everywhere, so
||x_hat|| = c*sqrt(d) always) and penalises those with norm spread. That
artefact once flipped a verdict on its own.
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


def cosine_scores(q: np.ndarray, xhat: np.ndarray) -> np.ndarray:
    """q @ xhat.T with every reconstruction renormalized to unit length."""
    n = np.linalg.norm(xhat, axis=1, keepdims=True)
    return q @ (xhat / np.clip(n, 1e-9, None)).T


def hamming_recall(qcodes: np.ndarray, dcodes: np.ndarray, k: int) -> float:
    """Recall@k under packed-bit Hamming, gold on the diagonal."""
    n = qcodes.shape[0]
    hits = 0
    for i in range(n):
        d = np.bitwise_count(np.bitwise_xor(qcodes[i], dcodes)).sum(axis=1)
        if i in np.argsort(d, kind="stable")[:k]:
            hits += 1
    return hits / n


def main() -> None:
    chunks = load_kb()
    qs, ds = split_chunks([c["text"] for c in chunks])
    rows = []
    for variant in ("a8m", "a25m"):
        enc = BekkoEncoder(variant, threads=4)
        qv_full = enc.encode(qs, batch_size=8)
        dv_full = enc.encode(ds, batch_size=8)
        for dim in (384, 256, 128, 64):
            q = matryoshka(qv_full, dim if dim < 384 else None)
            d = matryoshka(dv_full, dim if dim < 384 else None)

            rows.append({"variant": variant, "dim": dim, "codec": "fp32", "param": 32,
                         "r@10": recall_at_k(q @ d.T, 10), "bytes": dim * 4})

            for bits in (1, 2, 3, 4, 8):
                qz = remex.Quantizer(d=dim, bits=bits, seed=0)
                # decode() returns reconstructions in the ORIGINAL space
                # (rotation already inverted), so the query needs no rotation.
                xhat = qz.decode(qz.encode(d))
                rows.append({"variant": variant, "dim": dim, "codec": "remex",
                             "param": bits, "r@10": recall_at_k(cosine_scores(q, xhat), 10),
                             "bytes": round(dim * bits / 8)})

            for k in (1, 2, 4, 8):
                sq = StackedSignBitQuantizer(d=dim, k=k, seed=0).fit(d)
                rows.append({"variant": variant, "dim": dim, "codec": "remax",
                             "param": k,
                             "r@10": hamming_recall(sq.encode(q), sq.encode(d), 10),
                             "bytes": round(dim * k / 8)})
            print(f"{variant} d={dim} done", flush=True)
    json.dump(rows, open(HERE / "results_compose.json", "w"), indent=1)

    # ── iso-byte Pareto frontier ────────────────────────────────────────────
    print("\n=== Pareto frontier (best R@10 at each byte budget) ===")
    for variant in ("a8m", "a25m"):
        sub = [r for r in rows if r["variant"] == variant]
        best_at: dict[int, dict] = {}
        for r in sub:
            b = r["bytes"]
            if b not in best_at or r["r@10"] > best_at[b]["r@10"]:
                best_at[b] = r
        print(f"\n{variant}:")
        frontier, top = [], -1.0
        for b in sorted(best_at):
            r = best_at[b]
            mark = ""
            if r["r@10"] > top:
                top = r["r@10"]; frontier.append(r); mark = " <- frontier"
            print(f"  {b:5d} B  R@10={r['r@10']:.3f}  "
                  f"{r['codec']}(d={r['dim']},p={r['param']}){mark}")


if __name__ == "__main__":
    main()
