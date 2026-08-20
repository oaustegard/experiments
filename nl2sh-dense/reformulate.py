#!/usr/bin/env python3
"""One bounded query-rewrite pass before retrieval, and whether it earns its cost.

Issue #48's item 3: the biggest lever on retrieval is usually a better *query*.
*"Recover the password for backup.zip"* retrieves `funzip` and `bzip2recover`;
*"crack zip password dictionary attack"* retrieves `fcrackzip`. A full agentic
interpret-retrieve-assess loop needs a model a 270M cannot drive, but one
expansion pass is cheap and bounded.

Two expansions, both unsupervised — nothing here is authored against the eval,
which a hand-written synonym table for `fcrackzip` would be:

* **RM3** — classic pseudo-relevance feedback. Take the top `fb_docs` BM25 hits,
  score their terms by relevance-model weight, append the top `fb_terms` to the
  query. `muninn-rm3` already measured this as *no help* on a small in-vocab
  corpus (R@5 unchanged, R@10 0.900 from 1.000), and explicitly predicted that
  the case where it cannot help is the vocabulary-divergent query — which is
  exactly this corpus's dominant failure. So this arm is run expecting a
  negative, and it is here to confirm the prediction rather than to hope.

* **Dense-PRF** — the same feedback loop with the *dense* arm supplying the
  pseudo-relevant set. This is the one that has a reason to work: the lexical
  gap that stops BM25 from reaching `fcrackzip` is the gap the encoder crosses,
  so the encoder can hand BM25 the vocabulary it was missing and let BM25 do the
  precise matching. The generator still sees documents, not the rewritten query,
  so a bad expansion costs a retrieval, not a wrong command.

Both reuse `retrieve.tokens`, so the expansion is emitted in the same term space
the index was built in — an expansion tokenized differently would silently miss.

    python3 reformulate.py "recover the password for backup.zip"
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _lib.paths import experiment  # noqa: E402

RETRIEVAL = experiment("nl2sh-retrieval")
sys.path.insert(0, str(RETRIEVAL))
import retrieve as R  # noqa: E402
import dense_index as D  # noqa: E402


def _feedback_terms(index: R.Index, doc_ids: list[int], weights: list[float],
                    fb_terms: int, exclude: set[str]) -> list[tuple[str, float]]:
    """Relevance-model term weights over a pseudo-relevant set.

    p(t|R) = sum_d P(d) * tf(t,d)/|d|, with P(d) the normalized retrieval score,
    then damped by idf so a term common to the whole corpus does not win on
    frequency alone. Terms already in the query are dropped: re-adding them
    only reweights what BM25 has already scored.
    """
    total = sum(weights) or 1.0
    acc: dict[str, float] = defaultdict(float)
    for i, w in zip(doc_ids, weights):
        c = index.chunks[i]
        tf = Counter(R.tokens(f"{c.utility} {c.text}"))
        n = sum(tf.values()) or 1
        for t, f in tf.items():
            if t in exclude:
                continue
            acc[t] += (w / total) * (f / n)
    out = []
    for t, p in acc.items():
        post = index.postings.get(t)
        if post is None:
            continue
        out.append((t, p * post[2]))  # p(t|R) * idf
    out.sort(key=lambda x: -x[1])
    return out[:fb_terms]


def rm3(index: R.Index, query: str, fb_docs: int = 10, fb_terms: int = 10) -> str:
    scores = index.scores(query)
    ids = [i for i in index.topk(query, fb_docs)]
    if not ids:
        return query
    terms = _feedback_terms(index, ids, [float(scores[i]) for i in ids], fb_terms,
                            set(R.tokens(query)))
    return query + " " + " ".join(t for t, _ in terms)


def dense_prf(index: R.Index, dense, query: str, fb_docs: int = 10,
              fb_terms: int = 10) -> str:
    """Feedback set from the dense arm, expansion consumed by BM25."""
    s = dense.scores(query)
    k = min(fb_docs, len(s))
    ids = np.argpartition(-s, k - 1)[:k]
    ids = [int(i) for i in ids[np.argsort(-s[ids], kind="stable")]]
    # Cosine can be negative; shift to a positive weight so P(d) is a distribution.
    w = [max(float(s[i]), 0.0) + 1e-6 for i in ids]
    terms = _feedback_terms(index, ids, w, fb_terms, set(R.tokens(query)))
    return query + " " + " ".join(t for t, _ in terms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--model", default="leaf-mt-int8")
    ap.add_argument("--fb-docs", type=int, default=10)
    ap.add_argument("--fb-terms", type=int, default=10)
    a = ap.parse_args()
    q = " ".join(a.query)
    chunks, index, dense = D.load(a.model)
    print(f"query      : {q}")
    print(f"rm3        : {rm3(index, q, a.fb_docs, a.fb_terms)}")
    print(f"dense-prf  : {dense_prf(index, dense, q, a.fb_docs, a.fb_terms)}")
    for label, text in (("raw", q), ("rm3", rm3(index, q, a.fb_docs, a.fb_terms)),
                        ("dense-prf", dense_prf(index, dense, q, a.fb_docs, a.fb_terms))):
        utils = [u for u, _ in D.rank_utilities(index.scores(text),
                                                dense.utilities, 400, positive_only=True)][:5]
        print(f"  {label:<10} -> {utils}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
