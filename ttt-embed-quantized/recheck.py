#!/usr/bin/env python3
"""Check RESULTS.md against the committed artifact. Seconds, no network.

Two independent jobs, per METHODS principle 1 (verify with a disjoint path):

1. **Instrument check.** `ndcg_at_k` is scored against hand-computed values on
   synthetic score matrices before it is allowed to say anything about the real
   data — including the two cases a naive implementation gets wrong (a relevant
   doc outside k, and an ideal DCG that must cap at k rather than at |rel|).
   Without this, "the nDCG is 0.6x" only means the metric agrees with itself.
2. **Artifact check.** Shapes, dtypes, L2 normalization and id alignment of the
   committed `.npy`/`.json`, then the nDCG re-derived from them and compared to
   the number written in RESULTS.md — so prose and data cannot drift apart.

    python3 recheck.py        # exit 0 = RESULTS.md still describes data/
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_scifact import DIM, N_DOCS, N_TEST_QRELS, N_TEST_QUERIES, ndcg_at_k  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
fails: list[str] = []


def check(name: str, got, want, tol: float | None = None) -> None:
    ok = abs(got - want) <= tol if tol is not None else got == want
    print(f"  {'ok ' if ok else 'BAD'} {name}: {got}" + ("" if ok else f" (want {want})"))
    if not ok:
        fails.append(name)


def _synthetic(score_row: list[float], rel: list[int], k: int = 10) -> float:
    """nDCG for one query whose score vector is exactly `score_row`."""
    n = len(score_row)
    return ndcg_at_k(
        np.array([score_row], dtype=np.float32),
        np.eye(n, dtype=np.float32),
        [str(i) for i in range(n)],
        ["q"],
        {"q": [str(i) for i in rel]},
        k=k,
    )


def instrument_checks() -> None:
    print("instrument — ndcg_at_k against hand computation:")
    d = 1.0 / np.log2(np.arange(2, 12))
    desc = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1]
    check("one relevant doc at rank 1", _synthetic(desc, [0]), 1.0, 1e-12)
    check("one relevant doc at rank 3", _synthetic(desc, [2]), float(d[2]), 1e-12)
    check("one relevant doc at rank 11 (outside k)", _synthetic(desc, [10]), 0.0, 1e-12)
    check("two relevant, ranks 1 and 3",
          _synthetic(desc, [0, 2]), float((d[0] + d[2]) / (d[0] + d[1])), 1e-12)
    check("12 relevant, ideal DCG caps at k",
          _synthetic(list(range(12, 0, -1)), list(range(12))), 1.0, 1e-12)


def artifact_checks() -> None:
    print("artifact — data/ shapes, norms, alignment:")
    Dm = np.load(DATA / "Dm.npy")
    Q = np.load(DATA / "Q.npy")
    meta = json.loads((DATA / "meta.json").read_text())

    check("Dm shape", Dm.shape, (N_DOCS, DIM))
    check("Q shape", Q.shape, (N_TEST_QUERIES, DIM))
    check("Dm dtype", Dm.dtype, np.dtype("float32"))
    check("Q dtype", Q.dtype, np.dtype("float32"))
    check("Dm rows unit-norm (max dev)", float(np.abs(np.linalg.norm(Dm, axis=1) - 1).max()), 0.0, 1e-5)
    check("Q rows unit-norm (max dev)", float(np.abs(np.linalg.norm(Q, axis=1) - 1).max()), 0.0, 1e-5)
    check("no non-finite values", int(np.isfinite(Dm).all() and np.isfinite(Q).all()), 1)

    doc_ids, q_ids, qrels = meta["doc_ids"], meta["q_ids"], meta["qrels"]
    check("doc_ids aligned to Dm rows", len(doc_ids), N_DOCS)
    check("q_ids aligned to Q rows", len(q_ids), N_TEST_QUERIES)
    check("doc_ids unique", len(set(doc_ids)), N_DOCS)
    check("q_ids unique", len(set(q_ids)), N_TEST_QUERIES)
    check("ids are strings", int(all(isinstance(i, str) for i in doc_ids + q_ids)), 1)
    check("qrels pairs", sum(len(v) for v in qrels.values()), N_TEST_QRELS)
    check("every qrels doc is in the corpus",
          len({d for v in qrels.values() for d in v} - set(doc_ids)), 0)
    check("every query is judged", sum(1 for q in q_ids if qrels.get(q)), N_TEST_QUERIES)

    print("prose — RESULTS.md agrees with data/:")
    ndcg = ndcg_at_k(Q, Dm, doc_ids, q_ids, qrels)
    text = (HERE / "RESULTS.md").read_text()
    m = re.search(r"against the qrels = \*{0,2}(\d+\.\d+)", text)
    if not m:
        print("  BAD could not find the nDCG number in RESULTS.md")
        fails.append("RESULTS.md nDCG")
        return
    check("nDCG@10 recomputed vs RESULTS.md", round(ndcg, 4), float(m.group(1)), 1e-4)
    # The pre-registered band is the only check on document text; assert it too.
    check("nDCG@10 inside the 0.60-0.72 band", int(0.60 <= ndcg <= 0.72), 1)


def main() -> int:
    instrument_checks()
    if fails:
        print(f"\nFAIL — the metric itself is wrong ({len(fails)}); artifact checks skipped")
        return 1
    artifact_checks()
    print("\nFAIL: " + ", ".join(fails) if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
