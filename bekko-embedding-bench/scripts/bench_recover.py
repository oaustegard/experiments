"""The ~14% of queries whose gold never enters the top 50 — what are they, and
what recovers them?

An important distinction first. The vendor's recommended **rescore** step
(binary −12.93% → −0.44% on HAKARI) recovers *quantization* loss: it reranks a
candidate set with full-precision vectors, which works because the gold is
already in the set. The 14% here is a **recall** ceiling — gold is absent from
the top 50 at *every* precision including uncompressed fp32 — so rescoring
cannot touch it by construction. Different failure, different remedy.

Candidate remedies, and what this repo already knows about them:

  query expansion   METHODS.md records a negative: RM3 pseudo-relevance feedback
                    "does not help on a small corpus" (muninn-rm3, R@10
                    1.000 -> 0.900). Tested anyway, on the failures specifically.
  lexical (BM25)    METHODS.md: plain BM25 matches the dense ceiling for
                    in-vocabulary queries. remax_kb v2 already ships BM25 + RRF.
  hybrid RRF        Part A of this experiment: rg and dense fail on *disjoint*
                    instances, so fusion beat either alone. Same shape of claim.
  a better encoder  Part B: jina beats bekko. Is the ceiling bekko's or the task's?
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bekko import BekkoEncoder, matryoshka  # noqa: E402
from jina import JinaQ4Encoder  # noqa: E402
from run_partb import load_kb, split_chunks  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RRF_K = 60


def topk_idx(sims: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-sims, axis=1)[:, :k]


def recall_from_idx(idx: np.ndarray, k: int) -> float:
    return float(np.mean([i in idx[i, :k] for i in range(idx.shape[0])]))


def bm25_rank(queries: list[str], docs: list[str]) -> np.ndarray:
    """Dense (nq, nd) BM25 score matrix, stdlib-only tokenizer."""
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
    dtok = [tok(d) for d in docs]
    n, avgdl = len(dtok), sum(len(d) for d in dtok) / max(1, len(dtok))
    df: dict[str, int] = {}
    for d in dtok:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    idf = {w: np.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
    k1, b = 1.5, 0.75
    tf = [{} for _ in dtok]
    for i, d in enumerate(dtok):
        for w in d:
            tf[i][w] = tf[i].get(w, 0) + 1
    out = np.zeros((len(queries), n), dtype=np.float32)
    for qi, q in enumerate(queries):
        for w in tok(q):
            if w not in idf:
                continue
            iw = idf[w]
            for di, d in enumerate(dtok):
                f = tf[di].get(w, 0)
                if f:
                    out[qi, di] += iw * f * (k1 + 1) / (
                        f + k1 * (1 - b + b * len(d) / avgdl))
    return out


def rrf_fuse(*score_mats: np.ndarray) -> np.ndarray:
    fused = np.zeros_like(score_mats[0])
    for m in score_mats:
        order = np.argsort(-m, axis=1)
        ranks = np.empty_like(order)
        for i in range(m.shape[0]):
            ranks[i, order[i]] = np.arange(m.shape[1])
        fused += 1.0 / (RRF_K + ranks + 1)
    return fused


def expand_queries(queries: list[str], docs: list[str], sims: np.ndarray,
                   n_fb: int = 3, n_terms: int = 10) -> list[str]:
    """Crude RM3-style expansion: append top terms from the top-n_fb docs."""
    tok = lambda t: re.findall(r"[a-z0-9]{3,}", t.lower())
    stop = set("the and for with that this from are was you your има not but".split())
    out = []
    for qi, q in enumerate(queries):
        top = np.argsort(-sims[qi])[:n_fb]
        counts: dict[str, int] = {}
        for di in top:
            for w in tok(docs[di]):
                if w not in stop:
                    counts[w] = counts.get(w, 0) + 1
        terms = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n_terms]]
        out.append(q + " " + " ".join(terms))
    return out


def main() -> None:
    chunks = load_kb()
    qs, ds = split_chunks([c["text"] for c in chunks])
    n = len(qs)

    enc = BekkoEncoder("a25m", threads=4)
    qv, dv = matryoshka(enc.encode(qs, batch_size=8), None), \
        matryoshka(enc.encode(ds, batch_size=8), None)
    dense = qv @ dv.T
    idx50 = topk_idx(dense, 50)
    fail = [i for i in range(n) if i not in idx50[i]]
    print(f"fp32 384-d: R@50 = {recall_from_idx(idx50, 50):.3f}; "
          f"{len(fail)}/{n} queries never surface gold in the top 50\n")

    # ── 1. what do the failures look like? ──────────────────────────────────
    print("=== the failures (query head -> gold body), first 6 ===")
    for i in fail[:6]:
        rank = int(np.where(np.argsort(-dense[i]) == i)[0][0])
        print(f"  [{i}] gold rank {rank}/{n} | doc {len(ds[i])} chars")
        print(f"      Q: {qs[i][:110]!r}")
        print(f"      D: {ds[i][:110]!r}")
    short = [i for i in fail if len(ds[i]) < 400]
    print(f"\n  {len(short)}/{len(fail)} failures have a gold body under 400 chars "
          f"(vs {sum(1 for i in range(n) if len(ds[i]) < 400)}/{n} overall)")

    # ── 2. is it bekko, or the task? ────────────────────────────────────────
    jenc = JinaQ4Encoder(threads=4)
    jq = jenc.encode(qs, prompt="query", batch_size=8)
    jd = jenc.encode(ds, prompt="document", batch_size=8)
    jdense = jq @ jd.T
    jidx = topk_idx(jdense, 50)
    jfail = [i for i in range(n) if i not in jidx[i]]
    print(f"\n=== encoder ceiling or task ceiling? ===")
    print(f"  bekko-a25m R@50 {recall_from_idx(idx50, 50):.3f}  ({len(fail)} fail)")
    print(f"  jina q4    R@50 {recall_from_idx(jidx, 50):.3f}  ({len(jfail)} fail)")
    both = set(fail) & set(jfail)
    print(f"  failed by BOTH: {len(both)}  |  bekko-only: {len(set(fail) - set(jfail))}"
          f"  |  jina-only: {len(set(jfail) - set(fail))}")

    # ── 3. remedies, measured on the whole set and on the failures ──────────
    bm25 = bm25_rank(qs, ds)
    bidx = topk_idx(bm25, 50)
    fused = rrf_fuse(dense, bm25)
    fidx = topk_idx(fused, 50)
    exp = expand_queries(qs, ds, dense)
    ev = matryoshka(enc.encode(exp, batch_size=8), None)
    eidx = topk_idx(ev @ dv.T, 50)

    print(f"\n=== remedies ===")
    print(f"{'method':<28} {'R@10':>7} {'R@50':>7}   {'recovers of the ' + str(len(fail)):>22}")
    rows = [("dense only (bekko-a25m)", idx50), ("BM25 only", bidx),
            ("dense + BM25, RRF", fidx), ("query expansion (RM3-ish)", eidx)]
    res = []
    for name, ix in rows:
        rec = sum(1 for i in fail if i in ix[i, :50])
        print(f"{name:<28} {recall_from_idx(ix, 10):>7.3f} {recall_from_idx(ix, 50):>7.3f}"
              f"   {rec:>10}/{len(fail)}")
        res.append({"method": name, "r@10": recall_from_idx(ix, 10),
                    "r@50": recall_from_idx(ix, 50), "recovered": rec,
                    "n_fail": len(fail)})

    json.dump({"n": n, "fail": fail, "both_fail": sorted(both), "results": res},
              open(HERE / "results_recover.json", "w"), indent=1)


if __name__ == "__main__":
    main()
