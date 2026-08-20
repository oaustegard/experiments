#!/usr/bin/env python3
"""Code-aware BM25 over the shell-documentation chunk corpus.

WHY this exists rather than a stock BM25: the retrieval tier's job is to turn a
plain-English request into a *shortlist of utilities*, and the discriminating
tokens in shell documentation are compounds — `--max-depth`, `-print0`,
`git-checkout`, `no_wrap`. A whitespace/word tokenizer either keeps the compound
whole (so "limit how deep it recurses" can never reach `--max-depth`) or splits
it (so an exact `--max-depth` query no longer scores the option entry highest
than any doc merely containing "max").

`hybrid-code-index` resolved that by emitting BOTH: the whole token and its
parts. This module carries the same rule over to shell docs, with one change —
its `_WORD` pattern is `[A-Za-z_][A-Za-z0-9_]*`, which never sees a hyphen, so
`--max-depth` there yields only `max`/`depth` and loses the compound entirely.
Here hyphens are part of a word, so `--max-depth` emits `max-depth`, `max`,
`depth`. Leading dashes are stripped, which makes a query's `--max-depth` and a
SYNOPSIS's `--max-depth` the same term without a flag-specific code path.

Design notes:
  * BM25 (k1=1.2, b=0.75) with idf derived at query time from the postings —
    nothing precomputed, so the index is trivially rebuildable (METHODS.md:
    fitting BM25 is <1% of build time for a hybrid index).
  * Postings are stored as parallel numpy arrays and scored with `np.add.at`,
    which is what keeps a 31k-chunk query in the sub-millisecond range.
  * `--stem` applies a *light suffix stripper*, not Porter. It exists only as an
    ablation, because gh-mcp-regex-fit measured stemming helping recall@10 while
    hurting top-1 ("stem if you are shortlisting, not if you are deciding").

Usage as a library:

    idx = Index.load(Path("data/chunks.jsonl"))
    for chunk, score in idx.search("count lines in a file", k=10):
        print(chunk.utility, chunk.text)

Usage as a CLI (smoke test / manual inspection):

    python3 retrieve.py "delete files older than 30 days" -k 10
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# A "word" may contain hyphens and underscores internally. Bare integers are
# kept (man pages are full of `-1`, `4096`, `755`).
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*|\d+")
# camelCase / ALLCAPS splitter, applied inside each hyphen/underscore part.
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")
_SPLIT = re.compile(r"[-_]+")

# Terms this common carry no ranking signal on shell docs and appear in nearly
# every chunk of prose. Kept deliberately short: BM25's idf already suppresses
# frequent terms, so this list is about query-side noise, not index size.
STOPWORDS = frozenset("""
a an the of to in on for with and or is are be as by from at it its this that
these those how do does i you your my me we us all any some what when which
""".split())

MAX_TOKEN_LEN = 40  # p99 of identifier length; longer runs are hashes/base64


def _stem(t: str) -> str:
    """Light, conservative suffix stripping. NOT Porter — see module docstring.

    Only the three morphological endings that actually separate a shell request
    from its documentation ("removes"/"remove", "listing"/"list",
    "compressed"/"compress"). Anything shorter than 4 characters is left alone,
    and stems are never applied to compounds containing a hyphen (those are
    flags, where morphology is meaningless).
    """
    if "-" in t or len(t) < 5:
        return t
    if t.endswith("ies"):
        return t[:-3] + "y"
    # "-es" is only a plural marker after a sibilant; stripping it blindly turns
    # "files" into "fil".
    for suf in ("sses", "shes", "ches", "xes", "zes"):
        if t.endswith(suf) and len(t) - 2 >= 4:
            return t[:-2]
    for suf in ("ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    if t.endswith("s") and not t.endswith("ss") and len(t) - 1 >= 4:
        return t[:-1]
    return t


def tokens(s: str, stem: bool = False) -> list[str]:
    """Code-aware tokens: the whole compound *and* its parts.

    `--max-depth`  -> max-depth, max, depth
    `-print0`      -> print0
    `git-checkout` -> git-checkout, git, checkout
    `NO_COLOR`     -> no_color, no, color
    `asciiFold`    -> asciifold, ascii, fold

    Emitting both means an exact `--max-depth` query still ranks the option
    entry above a doc that merely says "max", while a prose query reaches it
    through "depth". Dropping the whole token trades exact lookup away; keeping
    only it is a plain text tokenizer.
    """
    out: list[str] = []
    for raw in _WORD.findall(s):
        if len(raw) > MAX_TOKEN_LEN:
            continue
        bare = raw.strip("-_")
        if not bare:
            continue
        out.append(bare.lower())
        # Parts are split from the *unlowered* form so camelCase survives.
        parts = [p for sub in _SPLIT.split(bare) if sub for p in _CAMEL.findall(sub)]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    out = [t for t in out if t and t not in STOPWORDS]
    return [_stem(t) for t in out] if stem else out


@dataclass(frozen=True)
class Chunk:
    id: str
    utility: str
    kind: str
    text: str
    runnable: bool


def load_chunks(path: Path, kinds: set[str] | None = None) -> list[Chunk]:
    """Read chunks.jsonl (built by build_corpus.py). `kinds` filters by chunk kind."""
    out: list[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if kinds is not None and d["kind"] not in kinds:
                continue
            out.append(Chunk(d["id"], d["utility"], d["kind"], d["text"],
                             bool(d.get("runnable", False))))
    return out


class Index:
    """BM25 over chunk text, with the utility name folded into the document.

    The utility name is prepended to each document because a chunk's own text
    often does not repeat it (`man` option entries say "-r  recurse into
    directories", never "grep"), yet a query naming the utility outright — "tar
    extract" — must reach it. This is a one-token boost, not a duplication:
    idf handles the rest.
    """

    k1 = 1.2
    b = 0.75

    def __init__(self, chunks: list[Chunk], stem: bool = False):
        self.chunks = chunks
        self.stem = stem
        self.n = len(chunks)
        self.utilities = np.array([c.utility for c in chunks], dtype=object)
        lens = np.zeros(self.n, dtype=np.float32)
        raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(chunks):
            tf = Counter(tokens(f"{c.utility} {c.text}", stem=self.stem))
            lens[i] = max(1, sum(tf.values()))
            for t, f in tf.items():
                raw[t].append((i, f))
        self.lens = lens
        self.avgdl = float(lens.mean()) if self.n else 1.0
        # Parallel arrays per term: (doc ids, precomputed tf saturation, idf).
        self.postings: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for t, post in raw.items():
            ids = np.fromiter((i for i, _ in post), dtype=np.int32, count=len(post))
            fs = np.fromiter((f for _, f in post), dtype=np.float32, count=len(post))
            df = len(post)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            dl = lens[ids]
            sat = fs * (self.k1 + 1) / (fs + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            self.postings[t] = (ids, (idf * sat).astype(np.float32), idf)

    @classmethod
    def load(cls, path: Path, kinds: set[str] | None = None, stem: bool = False) -> "Index":
        return cls(load_chunks(path, kinds), stem=stem)

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for t in set(tokens(query, stem=self.stem)):
            p = self.postings.get(t)
            if p is None:
                continue
            ids, contrib, _ = p
            np.add.at(s, ids, contrib)
        return s

    def topk(self, query: str, k: int = 10) -> list[int]:
        """Chunk indices of the k highest-scoring chunks, best first.

        Chunks with a zero score are dropped — an empty list is a real outcome
        (no query term is in the vocabulary) and must not be reported as k
        arbitrary hits.
        """
        s = self.scores(query)
        k = min(k, self.n)
        if k <= 0:
            return []
        cand = np.argpartition(-s, k - 1)[:k]
        cand = cand[np.argsort(-s[cand], kind="stable")]
        return [int(i) for i in cand if s[i] > 0]

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        s = self.scores(query)
        return [(self.chunks[i], float(s[i])) for i in self.topk(query, k)]

    def topk_utilities(self, query: str, k: int = 10) -> list[str]:
        """Distinct utilities from the top-k *chunks*, in rank order."""
        seen: list[str] = []
        for i in self.topk(query, k):
            u = self.chunks[i].utility
            if u not in seen:
                seen.append(u)
        return seen

    def rank_utilities(self, query: str, k: int = 20, pool: int = 200) -> list[str]:
        """Distinct utilities, best-chunk-first, from a deeper chunk pool.

        `topk_utilities` answers "which utilities are in the top-k chunks";
        this answers "what are the top-k utilities", which needs a deeper pool
        because one utility can own many of the leading chunks.
        """
        out: list[str] = []
        for i in self.topk(query, pool):
            u = self.chunks[i].utility
            if u not in out:
                out.append(u)
                if len(out) >= k:
                    break
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query", nargs="*", help="natural-language request")
    ap.add_argument("--chunks", type=Path, default=HERE / "data" / "chunks.jsonl")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--stem", action="store_true")
    a = ap.parse_args()
    if not a.query:
        print("no query given; use: retrieve.py 'delete files older than 30 days'",
              file=sys.stderr)
        return 2
    idx = Index.load(a.chunks, stem=a.stem)
    print(f"[{idx.n} chunks, {len(idx.postings)} terms]", file=sys.stderr)
    for c, s in idx.search(" ".join(a.query), a.k):
        head = c.text.replace("\n", " | ")[:110]
        print(f"{s:7.2f}  {c.utility:<16} {c.kind:<12} {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
