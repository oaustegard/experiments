"""The RHT option: does it remove the per-query binarizer constant, and at what
cost to retrieval?

Three ways remax_kb can produce the k projection matrices:

  haar        QR of a dim x dim Gaussian, per stack (what v1 rebuilds every query)
  rademacher  +/-1 planes from a splitmix64 stream (v2)
  srht        subsampled randomized Hadamard, rounds=3 (v2 DEFAULT)

v2 already defaults to ``projection="srht"`` and keeps ``"haar"`` only for
back-compat, so "run with the RHT option" is partly a question about whether v1
should follow. Measured here: construction cost, per-query encode cost, and
whether retrieval survives the substitution — the last one matters because an
RHT is **not** Haar-distributed (O(d log d) bits of randomness vs O(d^2),
structured rather than independent directions), so the Charikar collision bound
has to be re-measured rather than inherited. remax floors RHT rounds at 2 for
exactly this reason.
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
from bekko import BekkoEncoder, matryoshka  # noqa: E402
from bench_kb_path import median  # noqa: E402
from run_partb import load_kb, recall_at_k, split_chunks  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DIM, K, SEED = 256, 8, 0


def build_projections() -> dict:
    from remax import StackedSignBitQuantizer
    from remax_kb.projection import rademacher_planes, srht_matrix

    def haar():
        q = StackedSignBitQuantizer(d=DIM, k=K, seed=SEED)
        q.fit(None)
        return q.rotations_

    return {
        "haar (v1 default)": haar,
        "rademacher (v2)": lambda: rademacher_planes(DIM, K, SEED),
        "srht (v2 default)": lambda: srht_matrix(DIM, K, SEED, rounds=3),
    }


def sign_encode(X: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Pack sign(X @ Q_j) over the k stacks into uint8, matching remax layout."""
    bits = np.concatenate([(X @ rot[j]) >= 0 for j in range(rot.shape[0])], axis=1)
    return np.packbits(bits, axis=1)


def hamming_topk(qc: np.ndarray, dc: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((qc.shape[0], k), dtype=np.int64)
    for i in range(qc.shape[0]):
        d = np.bitwise_count(np.bitwise_xor(qc[i], dc)).sum(axis=1)
        out[i] = np.argsort(d, kind="stable")[:k]
    return out


def main() -> None:
    chunks = load_kb()
    qs, ds = split_chunks([c["text"] for c in chunks])

    rows = []
    builders = build_projections()

    # ── 1. construction + encode cost, embedder-independent ─────────────────
    print("=== projection construction (dim=256, k=8) ===")
    probe = np.random.default_rng(1).normal(size=(1, DIM)).astype(np.float32)
    built = {}
    for name, fn in builders.items():
        t_build = median(fn, warm=1, rep=5)
        R = fn()
        built[name] = R
        t_enc = median(lambda r=R: sign_encode(probe, r), warm=2, rep=15)
        rows.append({"stage": "construct", "projection": name,
                     "build_ms": t_build * 1e3, "apply_ms": t_enc * 1e3,
                     "shape": list(R.shape)})
        print(f"  {name:20s} build {t_build * 1e3:8.2f} ms   apply {t_enc * 1e3:6.3f} ms"
              f"   {R.shape}", flush=True)

    # ── 2. retrieval quality under each projection, per encoder ─────────────
    print("\n=== retrieval (self-retrieval on muninn-subset.kb chunks) ===")
    for variant in ("a8m", "a25m"):
        enc = BekkoEncoder(variant, threads=4)
        qv = matryoshka(enc.encode(qs, batch_size=8), DIM)
        dv = matryoshka(enc.encode(ds, batch_size=8), DIM)
        mean_v = dv.mean(axis=0)
        qc_f, dc_f = qv - mean_v, dv - mean_v
        base = recall_at_k(qv @ dv.T, 10)
        print(f"  bekko-{variant}: fp32 R@10 = {base:.3f}")
        for name, R in built.items():
            idx = hamming_topk(sign_encode(qc_f, R), sign_encode(dc_f, R), 10)
            r10 = float(np.mean([i in idx[i] for i in range(len(qs))]))
            idx1 = idx[:, :1]
            r1 = float(np.mean([i == idx1[i, 0] for i in range(len(qs))]))
            rows.append({"stage": "retrieval", "projection": name,
                         "model": f"bekko-{variant}", "r@1": r1, "r@10": r10,
                         "fp32_r@10": base})
            print(f"    {name:20s} R@1 {r1:.3f}  R@10 {r10:.3f}"
                  f"  (vs fp32 {base:.3f})", flush=True)
        del enc

    json.dump(rows, open(HERE / "results_rht.json", "w"), indent=1)

    h = next(r for r in rows if r["stage"] == "construct" and r["projection"].startswith("haar"))
    s = next(r for r in rows if r["stage"] == "construct" and r["projection"].startswith("srht"))
    rd = next(r for r in rows if r["stage"] == "construct" and r["projection"].startswith("rade"))
    print(f"\nbuild speedup vs haar: srht {h['build_ms'] / s['build_ms']:.1f}x, "
          f"rademacher {h['build_ms'] / rd['build_ms']:.1f}x")


if __name__ == "__main__":
    main()
