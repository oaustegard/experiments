#!/usr/bin/env python3
"""Independent re-verification of the committed artifact.

Loads `data/{Dm.npy,Q.npy,meta.json}` cold and recomputes the sanity metrics
through a deliberately *different* code path from `encode.py`:

  * ranking by `sorted()` over Python floats, not `np.argpartition`
  * DCG accumulated in an explicit loop with `math.log2`, no vectorised
    discount array
  * ideal DCG from the qrels counts directly

then diffs against `sanity.json`. Also runs two negative controls -- shuffled
qrels and shuffled query rows -- because a scorer that cannot go red is not
evidence of anything.

Exits non-zero on any mismatch. No network, no model, seconds to run.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TOL = 1e-9


def ndcg10_independent(Q, Dm, doc_ids, q_ids, qrels, k=10):
    total = 0.0
    for qi, q in enumerate(q_ids):
        rel = qrels[q]
        scores = (Q[qi] @ Dm.T).tolist()  # plain Python floats
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        dcg = 0.0
        for rank, i in enumerate(ranked):
            g = rel.get(doc_ids[i], 0)
            if g:
                dcg += g / math.log2(rank + 2)
        idcg = 0.0
        for rank, g in enumerate(sorted(rel.values(), reverse=True)[:k]):
            idcg += g / math.log2(rank + 2)
        total += (dcg / idcg) if idcg else 0.0
    return total / len(q_ids)


def main() -> int:
    Dm = np.load(DATA / "Dm.npy")
    Q = np.load(DATA / "Q.npy")
    meta = json.loads((DATA / "meta.json").read_text())
    doc_ids, q_ids, qrels = meta["doc_ids"], meta["q_ids"], meta["qrels"]
    claimed = json.loads((HERE / "sanity.json").read_text())

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
        ok = ok and cond

    # --- shape / dtype / normalization contract --------------------------
    check("Dm shape (5183, 256) fp32", Dm.shape == (5183, 256) and Dm.dtype == np.float32,
          f"{Dm.shape} {Dm.dtype}")
    check("Q shape (300, 256) fp32", Q.shape == (300, 256) and Q.dtype == np.float32,
          f"{Q.shape} {Q.dtype}")
    check("ids aligned to row order",
          len(doc_ids) == Dm.shape[0] and len(q_ids) == Q.shape[0])
    check("ids are strings",
          all(isinstance(d, str) for d in doc_ids) and all(isinstance(q, str) for q in q_ids))
    check("no duplicate doc ids", len(set(doc_ids)) == len(doc_ids))
    for name, X in (("Dm", Dm), ("Q", Q)):
        n = np.linalg.norm(X, axis=1)
        check(f"{name} L2-normalized", float(np.abs(n - 1).max()) < 1e-5,
              f"max |‖v‖-1| = {float(np.abs(n - 1).max()):.2e}")
        check(f"{name} finite", bool(np.isfinite(X).all()))
        check(f"{name} no duplicate rows (collapse check)",
              len(np.unique(X, axis=0)) == X.shape[0])
    check("every qrels query present", set(qrels) == set(q_ids))
    check("every qrels doc present in corpus",
          set().union(*(set(v) for v in qrels.values())) <= set(doc_ids))

    # --- independent metric ---------------------------------------------
    got = ndcg10_independent(Q, Dm, doc_ids, q_ids, qrels)
    check("nDCG@10 reproduces sanity.json", abs(got - claimed["ndcg@10"]) < TOL,
          f"independent {got:.6f} vs claimed {claimed['ndcg@10']:.6f}")
    check("nDCG@10 in the issue's expected band 0.60-0.72", 0.60 <= got <= 0.72,
          f"{got:.4f}")

    # --- negative controls: the scorer must be able to go red ------------
    rng = random.Random(0)
    shuffled_docs = list(doc_ids)
    rng.shuffle(shuffled_docs)
    broken = ndcg10_independent(Q, Dm, shuffled_docs, q_ids, qrels)
    check("control: shuffled doc-id mapping collapses the score", broken < 0.05,
          f"{broken:.4f} (vs {got:.4f})")

    perm = np.random.default_rng(0).permutation(len(q_ids))
    broken_q = ndcg10_independent(Q[perm], Dm, doc_ids, q_ids, qrels)
    check("control: shuffled query rows collapse the score", broken_q < 0.05,
          f"{broken_q:.4f} (vs {got:.4f})")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
