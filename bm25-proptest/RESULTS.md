# bm25-proptest — a model-based property test of the bm25 skill

A pilot of the method in Inanna Malick's
[Property Testing with Agent Swarms](https://recursion.wtf/posts/agents-and-property-tests/)
and her `proptest-praxis` skill, scaled down to one target and one pass. The
target is `bm25/scripts/bm25.py` from `oaustegard/claude-skills` (skill v0.2.1,
bm25s 0.3.16). Budget: one harness, one mutation-calibration pass, then stop.
Run 2026-11-16.

## Harness

`test_bm25_model.py` uses Hypothesis `RuleBasedStateMachine`. Each history is
up to 25 steps of `write(name, text)`, `delete(name)` and `query(q, k)` against
a temp directory. Every query calls the skill's own `get_or_build_index` and
compares it with an independent Okapi BM25 (Lucene IDF, k1=1.5, b=0.75) written
in plain loops in the test file. It checks:

- the index holds exactly the files currently on disk;
- `len(hits) == min(k, N)`, with no duplicate hits;
- each returned score is within 1e-3 (relative) of the model's;
- results come back ranked, and no document left out scores above the weakest
  one returned.

Generators use small pools so operations interact: 4 file names, 8 words
(including a one-letter word that the tokenizer drops, a case variant,
a hyphenated word and a non-ASCII word), texts of 0–8 words and queries of
1–3 words. One machine runs with the session cache off and one with it on.

## Findings

Two bugs. Each has a deterministic regression in `test_regressions.py`, and
all five regressions fail on the unmodified skill.

**1. The session cache serves a stale index after the corpus changes.**
`cache_key` hashes the corpus path, filters and size cap, but nothing about
the files. Its docstring says "same key → identical index". The second run on
the same directory loads the first run's index, so an edited file returns its
old text, a deleted file is still returned, and a new file is never seen.
Shrunk case: write `c.md`, query, delete `c.md`, query — the deleted file comes
back. The cache lives for the whole session, so editing a repo and searching it
again hits this case directly.

**2. A corpus with no indexable terms crashes the CLI with a traceback.**
An empty directory, or files holding only one-character tokens (`x y z`), gives
bm25s an empty vocabulary, and `BM25.index` raises
`ValueError: max() iterable argument is empty`.

With both findings excluded, 1,500 histories with the cache off found no
scoring or ranking mismatch against the model.

## Calibration and coverage

Two planted bugs, applied by monkeypatch so the skill file stays untouched:

| mutant | defect | caught |
|---|---|---|
| `drop_last` | the corpus walk drops its last file | yes, first example |
| `query_case` | the query is not lowercased (the corpus is) | yes: `Alpha` scored 0.0 vs 0.115 |

Hypothesis event counts over 500 histories (shares of query steps): k < N on
34%, which is where top-k selection matters; two or more documents with
nonzero scores on 47%; no matching document on 28%. The corpus never exceeded
4 files. Not covered: include/exclude globs, the size cap, the GitHub tarball
path, snippets and the interactive mode.

## Fix

`bm25-fix.diff` (34 changed lines) applies to the skill.

- **Cache key.** It now also hashes each discovered file's path and the SHA-256
  of its text. Files are read once and passed to `build`, so a cache hit still
  reads the corpus but skips tokenizing and indexing, which is the expensive
  part. Hashing content rather than `(size, mtime)` matters here: the
  regression edits `alpha` to `gamma` (same size) within a few milliseconds,
  which can fall inside one coarse mtime tick.
- **Empty vocabulary.** The index is built with no retriever, `query` returns
  zero-score hits up to `min(k, N)`, and that index is not cached. My first
  version returned `[]`, and the property suite flagged it: a normal corpus pads
  to k with zero-score documents, so the empty case now does the same.

Against the patched copy, all 7 regression and edge tests pass. Both machines
pass 2,000 histories each with the empty-vocabulary skip turned off
(`SKIP_KNOWN=0`), both mutants are still caught, and a CLI smoke query ranks
correctly.

## Reproduce

```bash
uv pip install --system --break-system-packages bm25s hypothesis pytest
python3 -m pytest -q -p no:logging test_regressions.py test_bm25_model.py   # fails: the findings
patch -o /tmp/bm25_fixed.py /mnt/skills/user/bm25/scripts/bm25.py bm25-fix.diff
BM25_PY=/tmp/bm25_fixed.py SKIP_KNOWN=0 N=2000 python3 -m pytest -q -p no:logging .
BM25_MUTANT=query_case BM25_PY=/tmp/bm25_fixed.py python3 -m pytest -q -k NoCache test_bm25_model.py  # must fail
```

## Assessment of the method

On this target the method found both bugs within the first second of
generation. Both are shallow, and finding 1 can also be seen by reading
`cache_key`. The harness still did more than reading would have: it produced
minimal reproductions, confirmed that the scoring path agrees with an
independent model over 1,500 histories, and caught a wrong first fix (the
`[]` return) that a hand-written regression would have accepted. Writing the
harness took one session and roughly 150 lines. Whether the full swarm
approach pays off on a larger target, such as Muninn's supersede chains, is
not tested here.
