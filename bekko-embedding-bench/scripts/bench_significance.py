"""Paired significance for the headline claims, at n=179.

Every embedding-quality result in this experiment rides on **179 chunks from 11
blog posts**, so one query is 1/179 = 0.56 points of R@10 and the differences
being reported are 2-8 queries wide. This runs the test that should have
accompanied them from the start: the arms are evaluated on the *same* queries,
so the comparison is paired — exact McNemar on the discordant pairs, plus a
paired bootstrap CI on the difference.

The output is a verdict per claim, not a p-value to admire: which of the
headline statements survive n=179, and which were noise reported as findings.
"""
from __future__ import annotations

import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("REMEX_ROOT", "/home/user/remex"))
sys.path.insert(0, os.environ.get("REMAX_ROOT", "/home/user/remax/src"))
from bekko import BekkoEncoder, matryoshka  # noqa: E402
from jina import JinaQ4Encoder  # noqa: E402
from run_partb import load_kb, split_chunks  # noqa: E402

import remex  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)


def hits_at(sims: np.ndarray, k: int) -> np.ndarray:
    """Per-query 0/1 hit vector — gold is the diagonal."""
    order = np.argsort(-sims, axis=1)[:, :k]
    return np.array([i in order[i] for i in range(sims.shape[0])], dtype=np.int8)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact McNemar. Returns (a-wins, b-wins, p)."""
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return n01, n10, p


def boot_ci(a: np.ndarray, b: np.ndarray, reps: int = 20000) -> tuple[float, float]:
    """Paired bootstrap 95% CI on mean(a) - mean(b)."""
    n = len(a)
    idx = RNG.integers(0, n, size=(reps, n))
    d = (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    chunks = load_kb()
    qs, ds = split_chunks([c["text"] for c in chunks])
    n = len(qs)
    print(f"corpus: {n} chunks from 11 posts; 1 query = {100 / n:.2f} pp of R@10\n")

    enc = BekkoEncoder("a25m", threads=4)
    qf, df = enc.encode(qs, batch_size=8), enc.encode(ds, batch_size=8)
    A = {}

    def add(name, sims, k=10):
        A[name] = hits_at(sims, k)

    for d in (384, 256, 128, 64):
        q, v = matryoshka(qf, d if d < 384 else None), matryoshka(df, d if d < 384 else None)
        add(f"matryoshka d={d}", q @ v.T)

    qfull, dfull = matryoshka(qf, None), matryoshka(df, None)
    for bits in (1, 2, 4):
        qz = remex.Quantizer(d=384, bits=bits, seed=0)
        xh = qz.decode(qz.encode(dfull))
        xh = xh / np.clip(np.linalg.norm(xh, axis=1, keepdims=True), 1e-9, None)
        add(f"remex {bits}-bit", qfull @ xh.T)
    for k_ in (1, 2):
        sq = StackedSignBitQuantizer(d=384, k=k_, seed=0).fit(dfull)
        qc, dc = sq.encode(qfull), sq.encode(dfull)
        h = np.zeros(n, dtype=np.int8)
        for i in range(n):
            dist = np.bitwise_count(np.bitwise_xor(qc[i], dc)).sum(axis=1)
            h[i] = i in np.argsort(dist, kind="stable")[:10]
        A[f"remax k={k_}"] = h

    j = JinaQ4Encoder(threads=4)
    jq = j.encode(qs, prompt="query", batch_size=8)
    jd = j.encode(ds, prompt="document", batch_size=8)
    add("jina q4 d=768", jq @ jd.T)

    CLAIMS = [
        ("remex 2-bit @96B beats UNCOMPRESSED fp32 @1536B", "remex 2-bit", "matryoshka d=384"),
        ("remex 1-bit @48B beats vendor floor d=64 @256B", "remex 1-bit", "matryoshka d=64"),
        ("remex 1-bit @48B equals Matryoshka d=128 @512B", "remex 1-bit", "matryoshka d=128"),
        ("remex 2-bit beats remax k=2 (same 96 B)", "remex 2-bit", "remax k=2"),
        ("remex 1-bit beats remax k=1 (same 48 B)", "remex 1-bit", "remax k=1"),
        ("jina q4 beats bekko-a25m (full width)", "jina q4 d=768", "matryoshka d=384"),
        ("Matryoshka d=384 beats d=64", "matryoshka d=384", "matryoshka d=64"),
        ("Matryoshka d=384 beats d=256", "matryoshka d=384", "matryoshka d=256"),
    ]
    out = []
    print(f"{'claim':<50}{'Δ R@10':>8}{'95% CI':>18}{'disc':>9}{'p':>8}  verdict")
    for label, x, y in CLAIMS:
        a, b = A[x], A[y]
        diff = a.mean() - b.mean()
        lo, hi = boot_ci(a, b)
        n01, n10, p = mcnemar_exact(a, b)
        sig = p < 0.05
        verdict = "SUPPORTED" if sig else ("noise" if abs(diff) < 0.05 else "unresolved")
        print(f"{label:<50}{diff:+8.3f}  [{lo:+.3f},{hi:+.3f}]  {n01:>3}/{n10:<4}{p:>8.3f}  {verdict}")
        out.append({"claim": label, "delta": float(diff), "ci_lo": lo, "ci_hi": hi,
                    "wins": n01, "losses": n10, "p": p, "significant": bool(sig)})

    json.dump({"n": n, "claims": out}, open(HERE / "results_significance.json", "w"), indent=1)

    print(f"\nn={n}: a 2-query difference is {2 / n:+.3f} R@10. To resolve a true "
          f"0.01 gap at 80% power you would need roughly n>2000.")


if __name__ == "__main__":
    main()
