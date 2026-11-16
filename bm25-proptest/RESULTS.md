# bm25-proptest — a model-based property test of the bm25 skill

The `bm25` skill in `oaustegard/claude-skills` ranks the files in a directory
against a text query with BM25, a standard keyword-relevance score. It keeps an
on-disk cache of each index for the rest of the container's life, so a repeat
search can skip rebuilding it. This experiment checks the skill with a
*property test*. The Hypothesis library generates random sequences of file
writes, deletes and queries. Each sequence runs against the skill and against a
separate, deliberately simple BM25 written for the test, and any disagreement is
a lead. When a sequence fails, Hypothesis *shrinks* it to the shortest one that
still fails. To check that the test can catch bugs at all, defects are planted
on purpose (*mutants*), and the test should fail on each.

This is a one-target pilot of the method in Inanna Malick's
[Property Testing with Agent Swarms](https://recursion.wtf/posts/agents-and-property-tests/)
and her `proptest-praxis` skill. Run 2026-11-16 on skill v0.2.1 and bm25s 0.3.16.

## Answer

Does a property test find anything in a small skill that is already in use? It
found two bugs on the first run. The cache returned a stale index after any file
in the directory was edited, added or deleted. A directory with no indexable
words crashed the command-line tool. The 5 regression tests all
fail on the original skill <!-- @evidence[[results/summary.json#regressions_old.failed]] -->
and all 5 pass with the fix. <!-- @evidence[[results/summary.json#regressions_fixed.passed]] -->
With the cache off and those two cases set aside, scores and rankings agreed
with the simple BM25 over 1,500 generated sequences, on directories of at most
4 files. The fix shipped as bm25 0.2.2 (`claude-skills` 23e33f6).

Reading `cache_key` also shows the cache bug. The test added three things
reading would not: minimal reproductions, a check of the scoring, and the
rejection of a wrong first version of the fix. A target where reading the code
would not reveal the bug would test the method harder.

## Findings

1. **The cache returned a stale index after the directory changed.** The cache
   key hashed the directory path, the include/exclude filters and the file-size
   cap, but nothing about the files. An edited file returned its old text, a
   deleted file was still returned, and a new file was never seen. Shrunk case:
   write `a.txt`, query, rewrite `a.txt`, query. Fixed by adding each file's
   path and a hash of its content to the key.
   (`test_regressions.py::test_cache_serves_edited_file_stale`, `_deleted_file`, `_misses_new_file`)
2. **A directory with no indexable terms crashed the tool.** An empty directory,
   or files containing only one-character tokens (the tokenizer keeps words of
   two or more characters), gave bm25s an empty vocabulary and a `ValueError`
   traceback. The fix returns zero-score hits up to `min(k, N)`, the same
   padding a normal directory gets.
   (`test_regressions.py::test_cli_no_indexable_terms_exits_cleanly`)
3. **The property test rejected a wrong first fix.** The first version of the
   empty-vocabulary fix returned no hits at all. The property
   `len(hits) == min(k, N)` failed on it. The CLI regression test as first
   written checked only for the absence of a traceback, so it would have passed;
   it now also checks for one zero-score hit per file. The wrong version is not
   kept.
4. **Both planted defects were caught by assertion failures, not crashes**: an
   off-by-one in the directory walk, and a query that is not lowercased the way
   the corpus is. (`results/summary.json#mutants`)
5. **A content hash in the cache key is the safe choice.** Keying on file size
   and modification time would miss a same-size edit made within one coarse
   timestamp tick, as the regression's `alpha` → `gamma` edit is. This is argued
   from how filesystem timestamps work and was not tested here.

## Method

`test_bm25_model.py` holds two Hypothesis `RuleBasedStateMachine` classes (cache
off, cache on) and two plain tests for the empty-vocabulary case. Each generated
sequence is up to 25 steps of `write(name, text)`, `delete(name)` and
`query(q, k)` against a temporary directory. Every query calls the skill's own
`get_or_build_index` and compares the result with an Okapi BM25 (Lucene IDF,
k1=1.5, b=0.75) written as plain loops in the test file. The checks:

- the index holds exactly the files currently on disk;
- `len(hits) == min(k, N)`, with no duplicate hits;
- each returned score is within 1e-3 of the model's (absolute below a score of
  1, relative above);
- results are ranked, and no omitted document scores above the weakest one returned.

Generators draw from small pools so operations interact. There are 4 file names
and 8 words: one that the tokenizer drops, a case variant, a hyphenated word and
a non-ASCII word. Texts are 0–8 words and queries 1–3. `SKIP_KNOWN=1`, the
default, skips the empty-vocabulary case so the search can continue past
finding 2. `BM25_MUTANT=drop_last|query_case` plants a defect by monkeypatch.

`test_regressions.py` holds the deterministic tests for findings 1 and 2: three
tests, one of them run for two inputs, so 5 cases. `bm25_v0.2.1.py` is the skill
as tested and `bm25-fix.diff` is the fix as shipped (the shipped file differs
only in quote style). `python3 record.py` reruns every check (about 2 minutes)
and writes `results/summary.json`, which the numbers here cite. It needs
`bm25s`, `hypothesis` and `pytest`.

Not covered: include/exclude globs, the size cap, the GitHub tarball path,
snippets and interactive mode.

## Log

### Round 1 — 2026-11-16, pilot

Asked: "run the pilot on the BM25 index", after a review of `proptest-praxis`,
with a budget of one harness and one mutation pass.

Both machines failed on the first run. The empty-vocabulary crash came first;
with it skipped, the cache machine shrank to the stale-index case. The cache-off
machine passed 1,500 sequences, and both mutants were caught.

Coverage, measured as the share of generated sequences containing at least one
query step of each kind, varies a lot between runs. One recorded run gave k < N
34%, two or more documents with nonzero scores 38%, and no matching document
21% (`results/summary.json#coverage_pct`). Two other runs of the same command
gave 17/27/16 and 34/47/28.

The first fix returned no hits for an empty vocabulary. Running the suite with
`SKIP_KNOWN=0` failed on `len(hits) == min(k, N)`, and the fix was changed to
pad with zero-score hits. With the final fix, the regressions and both machines
pass at 2,000 sequences each: 4 of 4 test functions in the property file.
<!-- @evidence[[results/summary.json#all_fixed_2000.passed]] -->

### Round 2 — 2026-11-16, fix shipped

Oskar: "Fix the found bugs, yes". The diff went into `claude-skills` as bm25
0.2.2 (23e33f6) with `bm25/tests/test_regressions.py`. These checks were run by
hand and are not in `record.py`. Against the shipped file, the regressions
passed, and the property suite passed 1,000 sequences per machine. A CLI search,
an edit, then a second search reported a cache miss on the second search. On
the `claude-skills` checkout (1,649 files), one timing each gave a cache hit of
0.29 s before the fix and 0.35 s after, because a hit now reads and hashes every
file.

A fresh-reader review then found that the `query_case` mutant also crashed by
itself once `SKIP_KNOWN=0` let empty-vocabulary corpora through, and that the
CLI regression test only checked for no traceback. The mutant now has the same
guard as the fix, and `record.py` stores which error types each mutant raised;
both show only `AssertionError`. The CLI test now checks the hits as well.
