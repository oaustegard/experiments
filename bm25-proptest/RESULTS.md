# bm25-proptest — a model-based property test of the bm25 skill

The `bm25` skill in `oaustegard/claude-skills` searches a directory of text files
with BM25 ranking and keeps a per-session cache of each index. This experiment
checks it with a property test: random sequences of file writes, deletes and
queries are run against the skill and against a separate, deliberately simple
BM25 implementation, and any disagreement is a lead. It is a one-target pilot of
the method in Inanna Malick's
[Property Testing with Agent Swarms](https://recursion.wtf/posts/agents-and-property-tests/)
and her `proptest-praxis` skill. Run 2026-11-16 on skill v0.2.1 and bm25s 0.3.16.

## Answer

Does a cheap property test find anything in a small, already-used skill? Yes,
two bugs, both within the first second of generation. The session cache served
a stale index after any file in the corpus was edited, added or deleted, and a
corpus with no indexable words crashed the CLI. The 5 regression tests all
fail on the original skill <!-- @evidence[[results/summary.json#regressions_old.failed]] -->
and all 5 pass with the fix. <!-- @evidence[[results/summary.json#regressions_fixed.passed]] -->
Outside those two bugs, scoring and ranking agreed with the independent model
over 1,500 generated histories. The fix shipped as bm25 0.2.2
(`claude-skills` 23e33f6). The cache bug is also visible by reading
`cache_key`, so the harness's added value was minimal reproductions, a check on
scoring, and catching a wrong first version of the fix. A deeper target, where
reading the code would not show the bug, would test the method harder.

## Findings

1. **The session cache returned a stale index after the corpus changed.** The key
   hashed the corpus path, filters and size cap, but nothing about the files, so
   an edited file returned its old text, a deleted file was still returned and a
   new file was never seen. Shrunk case: write `c.md`, query, delete `c.md`,
   query. Fixed by adding each file's path and content hash to the key.
   (`test_regressions.py::test_cache_*`)
2. **A corpus with no indexable terms crashed the CLI.** An empty directory, or
   files holding only one-character tokens, gave bm25s an empty vocabulary and a
   `ValueError` traceback. Fixed by returning zero-score hits up to `min(k, N)`.
   (`test_regressions.py::test_cli_no_indexable_terms_exits_cleanly`)
3. **The suite rejected a plausible wrong fix.** The first version of the
   empty-vocabulary fix returned `[]`. The property test flagged it, because a
   normal corpus pads to k with zero-score documents. A hand-written regression
   would have accepted it.
4. **Both planted defects were caught**: an off-by-one in the corpus walk, and a
   query that was not lowercased like the corpus. (`results/summary.json#mutants`)
5. **A content hash is needed for the cache key; `(size, mtime)` is not enough.**
   The regression edits `alpha` to `gamma` (same size) within milliseconds,
   which can fall inside one coarse mtime tick. With the fix, a cache hit reads
   and hashes the corpus but skips tokenizing: 0.35 s against 0.29 s before on
   1,649 files, measured once with no repeats.

## Method

`test_bm25_model.py` is a Hypothesis `RuleBasedStateMachine`. Each history is up
to 25 steps of `write(name, text)`, `delete(name)` and `query(q, k)` against a
temporary directory. Every query calls the skill's own `get_or_build_index` and
compares the result with an Okapi BM25 (Lucene IDF, k1=1.5, b=0.75) written as
plain loops in the test file. The checks:

- the index holds exactly the files currently on disk;
- `len(hits) == min(k, N)`, with no duplicate hits;
- each returned score is within 1e-3 (relative) of the model's;
- results are ranked, and no omitted document scores above the weakest one returned.

Generators draw from small pools so operations interact. There are 4 file names
and 8 words: one that the tokenizer drops, a case variant, a hyphenated word and
a non-ASCII word. Texts are 0–8 words and queries 1–3. One machine runs with the
session cache off and one with it on. `SKIP_KNOWN=1` (the default) skips the
empty-vocabulary case so the search can continue past finding 2.
`BM25_MUTANT=drop_last|query_case` plants a defect by monkeypatch.

`bm25_v0.2.1.py` is the skill as tested, and `bm25-fix.diff` is the fix as
shipped. `python3 record.py` reruns every check (about 2 minutes) and writes
`results/summary.json`, which the numbers here cite. It needs `bm25s`,
`hypothesis` and `pytest`.

Not covered: include/exclude globs, the size cap, the GitHub tarball path,
snippets and interactive mode. The corpus never exceeds 4 files.

## Log

### Round 1 — 2026-11-16, pilot

Asked: "run the pilot on the BM25 index", after a review of `proptest-praxis`,
with a budget of one harness and one mutation pass.

The first run failed within a second on both machines. The empty-vocabulary
crash came first; with it skipped, the cache machine shrank to the stale-index
case. The no-cache machine passed 1,500 histories. Both mutants were caught.
Coverage, as the share of generated histories with at least one such query
step, came out as k < N 17%, two or more nonzero-scored documents 27%, and no
match 16% (`results/summary.json#coverage_pct`). An earlier run of the same
command reported 34/47/28 with the same counting, so these shares vary a lot
between runs.

The first fix returned `[]` for an empty vocabulary. Running the suite with
`SKIP_KNOWN=0` failed on `len(hits) == min(k, N)`, and the fix was changed to
pad with zero-score hits. With the final fix, the regressions and both machines
pass at 2,000 histories each: 4 of 4 test functions in the property file.
<!-- @evidence[[results/summary.json#all_fixed_2000.passed]] -->

### Round 2 — 2026-11-16, fix shipped

Oskar: "Fix the found bugs, yes". The diff went into `claude-skills` as bm25
0.2.2 (23e33f6) with `bm25/tests/test_regressions.py`. Against the shipped file,
the five regressions pass, the property suite passes 1,000 histories per
machine, and a CLI edit-then-search returns a cache miss.
