#!/usr/bin/env python3
"""Dense and hybrid retrieval over the shell-documentation chunk corpus.

The corpus and the BM25 arm are `nl2sh-retrieval`'s: 31,169 tldr/man chunks over
4,698 utilities, scored by `retrieve.Index`. This module adds the second arm and
the fusion, and nothing else — the eval scripts import from here.

Three decisions worth stating, because each could have gone the other way:

* **The document text is `f"{utility} {text}"`, matching BM25 exactly.**
  `retrieve.Index` folds the utility name into every document because a man
  option entry ("-r  recurse into directories") never repeats the utility it
  belongs to. Encoding a different string than BM25 indexes would make the two
  arms answer different questions, and any fusion result would be confounded by
  the field choice rather than by the retrieval method. `--no-utility-prefix`
  runs the ablation.
* **Fusion is offered as both RRF and a weighted score sum.** `hybrid-code-index`
  won 24/24 with RRF, but `gh-mcp-regex-fit` measured RRF *losing* to a weighted
  sum (0.554 vs 0.622) when the two arms were of unequal quality, because RRF is
  unweighted and a weak arm votes as loudly as a strong one. BM25 here puts the
  gold utility at rank 1 on 11.8% of queries, which is weak enough that the
  unequal-arm case is the one to expect. Both are measured.
* **Ranking is at utility level, not chunk level.** The consumer is a generator
  that gets k utilities' examples in its prompt, so a utility's score is its
  best chunk's score. Fusion happens after that aggregation for the weighted
  sum, and on the utility ranks for RRF.

    python3 dense_index.py build --model leaf-mt-int8      # encode + cache
    python3 dense_index.py query leaf-mt-int8 "crack a zip password"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root, for _lib
from _lib.paths import experiment  # noqa: E402

RETRIEVAL = experiment("nl2sh-retrieval")
sys.path.insert(0, str(RETRIEVAL))
import retrieve as R  # noqa: E402

sys.path.insert(0, str(HERE))
import encoders  # noqa: E402

CACHE = HERE / "cache"
DEFAULT_CHUNKS = RETRIEVAL / "data" / "chunks.jsonl"


def doc_text(chunk, utility_prefix: bool = True) -> str:
    return f"{chunk.utility} {chunk.text}" if utility_prefix else chunk.text


def page_chunks(chunks: list) -> list:
    """Collapse the chunk corpus to one document per source page.

    Chunking existed for Pleias' 4k context window; Gemma 3 270M has 32k, so a
    whole tldr or man page fits in the prompt and retrieval granularity becomes a
    free knob (issue #48 item 4). Page identity is the chunk id up to its `#`
    (`tldr:common/tar#3` -> `tldr:common/tar`, `man:clusterdb#opt0` ->
    `man:clusterdb`), which is how `build_corpus.py` emitted them.

    A page inherits the kind of its first chunk, so `kind` still separates tldr
    from man at page level.
    """
    order: list[str] = []
    parts: dict[str, list[str]] = {}
    meta: dict[str, tuple[str, str, bool]] = {}
    for c in chunks:
        page = c.id.split("#", 1)[0]
        if page not in parts:
            parts[page] = []
            order.append(page)
            meta[page] = (c.utility, c.kind, c.runnable)
        parts[page].append(c.text)
    out = []
    for page in order:
        utility, kind, runnable = meta[page]
        out.append(R.Chunk(page, utility, kind, "\n".join(parts[page]), runnable))
    return out


GRANULARITIES = {"chunk": lambda cs: cs, "page": page_chunks}


def cache_path(model: str, utility_prefix: bool, granularity: str = "chunk") -> Path:
    tag = "" if utility_prefix else "_noutil"
    grain = "" if granularity == "chunk" else f"_{granularity}"
    return CACHE / f"chunks_{model}{tag}{grain}.npy"


def build_vectors(model: str, chunks: list, utility_prefix: bool = True,
                  batch_size: int = 16, threads: int | None = None,
                  granularity: str = "chunk") -> np.ndarray:
    """Encode every document once and cache. Regenerable; the cache is gitignored."""
    path = cache_path(model, utility_prefix, granularity)
    if path.exists():
        v = np.load(path)
        if len(v) == len(chunks):
            return v
        print(f"cache {path.name} has {len(v)} rows for {len(chunks)} chunks; rebuilding",
              file=sys.stderr)
    enc = encoders.build(model, threads=threads)
    texts = [doc_text(c, utility_prefix) for c in chunks]
    v = enc.encode(texts, prompt="document", batch_size=batch_size, progress=True)
    CACHE.mkdir(exist_ok=True)
    np.save(path, v)
    print(f"encoded {len(texts)} chunks in {enc.last_wall:.0f}s -> {path.name}",
          file=sys.stderr)
    return v


class DenseArm:
    """Cosine over cached chunk vectors, aggregated to utilities.

    `adapter` is an optional (d, d) matrix applied to the query vector only
    (see `adapter.py`). Document vectors are untouched by design, so an adapter
    can be added to or removed from a live index without re-encoding anything.
    """

    def __init__(self, model: str, chunks: list, vectors: np.ndarray,
                 threads: int | None = None, adapter: np.ndarray | None = None) -> None:
        self.model = model
        self.chunks = chunks
        self.vectors = vectors
        self.enc = encoders.build(model, threads=threads)
        self.utilities = np.array([c.utility for c in chunks], dtype=object)
        self.adapter = adapter

    def scores(self, query: str) -> np.ndarray:
        q = self.enc.encode([query], prompt="query")[0]
        if self.adapter is not None:
            q = self.adapter @ q
            q = q / max(float(np.linalg.norm(q)), 1e-9)
        return self.vectors @ q


def rank_utilities(scores: np.ndarray, utilities: np.ndarray,
                   pool: int = 400, positive_only: bool = False) -> list[tuple[str, float]]:
    """Utilities ranked by their best chunk's score, from the top `pool` chunks.

    `positive_only` drops non-scoring chunks; BM25 needs it (a zero score means
    no query term matched at all and the chunk is not a hit), cosine does not
    (every chunk has a real similarity, and negatives are meaningful).
    """
    k = min(pool, len(scores))
    cand = np.argpartition(-scores, k - 1)[:k]
    cand = cand[np.argsort(-scores[cand], kind="stable")]
    out: dict[str, float] = {}
    for i in cand:
        s = float(scores[i])
        if positive_only and s <= 0:
            break
        out.setdefault(str(utilities[i]), s)
    return list(out.items())


def rrf(*rankings: list[tuple[str, float]], k: int = 60,
        weights: tuple[float, ...] | None = None) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion over utility rankings."""
    w = weights or (1.0,) * len(rankings)
    acc: dict[str, float] = {}
    for wi, ranking in zip(w, rankings):
        for rank, (u, _) in enumerate(ranking, start=1):
            acc[u] = acc.get(u, 0.0) + wi / (k + rank)
    return sorted(acc.items(), key=lambda x: -x[1])


def _minmax(ranking: list[tuple[str, float]]) -> dict[str, float]:
    if not ranking:
        return {}
    vals = [s for _, s in ranking]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return {u: 1.0 for u, _ in ranking}
    return {u: (s - lo) / span for u, s in ranking}


def wsum(bm25: list[tuple[str, float]], dense: list[tuple[str, float]],
         alpha: float = 0.5) -> list[tuple[str, float]]:
    """Min-max-normalized weighted sum. alpha weights the dense arm.

    Normalization is per query over the candidate pool, which is the only way to
    put a BM25 score (unbounded, query-length dependent) and a cosine on one
    scale. A utility missing from one arm's pool scores 0 there.
    """
    a, b = _minmax(bm25), _minmax(dense)
    keys = set(a) | set(b)
    return sorted(((u, (1 - alpha) * a.get(u, 0.0) + alpha * b.get(u, 0.0))
                   for u in keys), key=lambda x: -x[1])


def load(model: str, chunks_path: Path = DEFAULT_CHUNKS, utility_prefix: bool = True,
         threads: int | None = None, granularity: str = "chunk",
         adapter: Path | None = None):
    chunks = GRANULARITIES[granularity](R.load_chunks(chunks_path))
    index = R.Index(chunks)
    vectors = build_vectors(model, chunks, utility_prefix, threads=threads,
                            granularity=granularity)
    W = np.load(adapter)["W"] if adapter else None
    return chunks, index, DenseArm(model, chunks, vectors, threads=threads, adapter=W)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="leaf-mt-int8", choices=sorted(encoders.ENCODERS))
    b.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    b.add_argument("--batch-size", type=int, default=16)
    b.add_argument("--threads", type=int, default=None)
    b.add_argument("--no-utility-prefix", action="store_true")
    b.add_argument("--granularity", default="chunk", choices=sorted(GRANULARITIES))
    q = sub.add_parser("query")
    q.add_argument("model", choices=sorted(encoders.ENCODERS))
    q.add_argument("text", nargs="+")
    q.add_argument("-k", type=int, default=10)
    a = ap.parse_args()

    if a.cmd == "build":
        chunks = GRANULARITIES[a.granularity](R.load_chunks(a.chunks))
        build_vectors(a.model, chunks, not a.no_utility_prefix,
                      batch_size=a.batch_size, threads=a.threads,
                      granularity=a.granularity)
        return 0

    chunks, index, dense = load(a.model)
    text = " ".join(a.text)
    bm = rank_utilities(index.scores(text), dense.utilities, positive_only=True)
    dn = rank_utilities(dense.scores(text), dense.utilities)
    print(f"{'bm25':<22}{'dense':<22}{'rrf':<22}{'wsum(a=0.5)':<22}")
    fused_r, fused_w = rrf(bm, dn), wsum(bm, dn)
    for i in range(a.k):
        row = [f"{r[i][0]} {r[i][1]:.3f}" if i < len(r) else ""
               for r in (bm, dn, fused_r, fused_w)]
        print("".join(f"{c:<22}" for c in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
