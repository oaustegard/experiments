# Errors

What went wrong, how it was caught, and which way it pushed the conclusion. The
base rate matters more than any single entry.

## 1. Thesis stated stronger than the evidence

**Direction: would have overstated the finding.**

The review that started this experiment said the paper's +12.5 HIT@10 enterprise
gain was "substantially a missing-BM25 artifact." Their Appendix Table 7 refutes
that on its own numbers: per-method rescue rates of 14.8 / 14.5 / 13.2 / 10.1 /
10.1 percent on a 283-query miss set, overlapping partially, convert about 20%
of the misses, which is what +12.5 on a 39.22 base needs. No missing arm is
required to explain their result.

Caught by running `thesis-discipline-check` item 4 — take the paper's own
appendix numbers and check whether they already account for the effect — before
building anything.
The surviving claim is narrower and is what `PLAN.md` pre-registers: the paper
does not report what fraction of its baseline's misses a free lexical arm
rescues.

## 2. `bm25s` OOM-killed on the 512K-document corpus

**Direction: none — a crash, not a wrong number.**

`bm25s.tokenize` materialises `corpus_token_ids` as `List[List[int]]`, about
230M Python ints for this corpus at roughly 36 bytes each, and
`build_index_from_ids` needs all of it resident to compute document
frequencies. The first attempt also built a second full copy of the corpus in a
list comprehension. It died at rc=137 during indexing.

Two fixes. Joining title and content in place rather than into a new list
removed 2.6 GB. `bm25_sparse.py` then replaced the library path entirely for the
large arms, building the same Lucene-variant index in two streaming passes into
a scipy CSC matrix, about 1 GB where `bm25s` needed 8.

## 3. `streaming_tokenize` indexed zero documents

**Direction: would have produced a garbage result that looked like a run.**

The first fix attempt used `bm25s`'s own `Tokenizer.streaming_tokenize()`, which
is a generator, fed to `BM25.index()`. `index()` calls `_infer_corpus_object()`,
which consumes the generator, so the build then saw an empty corpus. The only
signal at index time was a `RuntimeWarning: Mean of empty slice`; the failure
surfaced later as `k of 10 is larger than the number of available scores, which
is 0`.

Caught by smoke-testing both arms on a 3,723-document gold-enriched subset
before the full run. That cost three seconds and caught a zero-document index
that would otherwise have looked like a forty-minute run.

## 4. Encoder cost

**Direction: none — the estimate and the measurement agreed.**

`bge-base-en-v1.5` is the paper's own retriever and was arm B's first choice.
A FLOP estimate said it was infeasible on four CPU cores; the measurement said
3.7 chunks/s, or 83 hours for this corpus, which agreed. The Hub carries no
quantised variant (`model_O2`, `model_quantized` and `model_int8` all return a
15-byte error body), so there was no cheaper path to the same encoder.

Recorded because the deviation it forced is material: arm B runs
`all-MiniLM-L6-v2` at 45.3 chunks/s on a 20% subcorpus, the anchor to their
39.22 BGE baseline is not claimed, and `PLAN.md` pre-registered that outcome as
invalidating the external comparison only.

## 5. Two arrows rewritten as "to" in one rewrite batch

**Direction: negligible, and disclosed rather than found later.**

The subagent generating S4 rewrites for questions `qst_0377`-`qst_0423` replaced
the arrow in `SLO→customer impact`, `eu-west→us-east` and `Usage → Ad-hoc Query`
with the word "to", for JSON cleanliness. The analyzer strips non-word
characters anyway, so the arrow never reached either index, but the rewrite file
is the arm's input and the substitution is recorded here rather than left to be
noticed in a diff.
