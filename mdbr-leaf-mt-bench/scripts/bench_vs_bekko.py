"""Paired leaf-mt vs bekko-a8m on the same splits — the compute-bound rung.

The incumbent-quality question is settled by ``run_partb_leaf.py`` (jina wins).
The remaining question is the *cheap* rung of the iso-quality ladder, which
bekko-a8m currently holds: does leaf-mt-int8 (23.7 MB, ~7 ms/query) match
bekko-a8m (124 MB, ~11 ms/query) on retrieval? Cross-run aggregates say
"close"; this makes it paired. Also re-times bekko-a8m on this box so the
latency comparison is same-session, not read off the 2026-08-04 run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
BEKKO_BENCH = HERE.parent / "bekko-embedding-bench"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BEKKO_BENCH / "scripts"))

from bekko import BekkoEncoder  # noqa: E402
from leaf import LeafMTEncoder, matryoshka  # noqa: E402
from run_partb_leaf import (  # noqa: E402
    boot_ci, hits_at, load_blog_chunks, load_code_chunks, mcnemar_exact,
    recall_at_k, split_chunks,
)


def main() -> None:
    blog = load_blog_chunks()
    qs, ds = split_chunks(blog)
    cq, cd = split_chunks(load_code_chunks(len(blog)))
    dists = {"blog": (qs, ds), "code": (cq, cd)}

    out = {"retrieval": [], "claims": [], "latency": []}
    V = {}

    bek = BekkoEncoder("a8m", threads=4)
    leaf = LeafMTEncoder("onnx/model_quantized.onnx", threads=4)
    for dist, (Q, D) in dists.items():
        V[("bekko-a8m", dist)] = (bek.encode(Q, batch_size=8), bek.encode(D, batch_size=8))
        V[("leaf-mt-int8", dist)] = (
            leaf.encode(Q, prompt="query", batch_size=8),
            leaf.encode(D, prompt="document", batch_size=8),
        )
        for name in ("bekko-a8m", "leaf-mt-int8"):
            qv, dv = V[(name, dist)]
            sims = qv @ dv.T
            out["retrieval"].append({
                "model": name, "dist": dist, "dim": qv.shape[1],
                "r@1": recall_at_k(sims, 1), "r@10": recall_at_k(sims, 10),
            })
            print(f"{name}/{dist}: r@1 {out['retrieval'][-1]['r@1']:.3f} "
                  f"r@10 {out['retrieval'][-1]['r@10']:.3f}", flush=True)

    def H(model, dist, k):
        qv, dv = V[(model, dist)]
        return hits_at(qv @ dv.T, k)

    CLAIMS = [
        ("leaf int8 vs bekko-a8m, blog R@10", H("leaf-mt-int8", "blog", 10), H("bekko-a8m", "blog", 10)),
        ("leaf int8 vs bekko-a8m, code R@10", H("leaf-mt-int8", "code", 10), H("bekko-a8m", "code", 10)),
        ("leaf int8 vs bekko-a8m, blog R@1", H("leaf-mt-int8", "blog", 1), H("bekko-a8m", "blog", 1)),
        ("leaf int8 vs bekko-a8m, code R@1", H("leaf-mt-int8", "code", 1), H("bekko-a8m", "code", 1)),
    ]
    print(f"\n{'claim':<40}{'Δ':>8}{'95% CI':>18}{'w/l':>9}{'p':>8}")
    for label, a, b in CLAIMS:
        diff = float(a.mean() - b.mean())
        lo, hi = boot_ci(a, b)
        n01, n10, p = mcnemar_exact(a, b)
        print(f"{label:<40}{diff:+8.3f}  [{lo:+.3f},{hi:+.3f}]  {n01:>3}/{n10:<4}{p:>8.4f}")
        out["claims"].append({"claim": label, "delta": diff, "ci_lo": lo, "ci_hi": hi,
                              "wins": n01, "losses": n10, "p": p,
                              "significant": bool(p < 0.05)})

    # same-session 1-thread query latency for both
    del bek, leaf
    for name, mk in (
        ("bekko-a8m", lambda: BekkoEncoder("a8m", threads=1)),
        ("leaf-mt-int8", lambda: LeafMTEncoder("onnx/model_quantized.onnx", threads=1)),
    ):
        e = mk()
        q = qs[0]
        def one(e=e, q=q):
            return e.encode([q]) if name == "bekko-a8m" else e.encode([q], prompt="query")
        for _ in range(2):
            one()
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); one(); ts.append(time.perf_counter() - t0)
        ms = float(np.median(ts)) * 1e3
        out["latency"].append({"model": name, "threads": 1, "query_ms": ms})
        print(f"1t {name}: {ms:.1f} ms")
        del e

    json.dump(out, open(HERE / "results_vs_bekko.json", "w"), indent=1)
    print("wrote results_vs_bekko.json")


if __name__ == "__main__":
    main()
