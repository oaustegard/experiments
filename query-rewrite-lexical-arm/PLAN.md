# Pre-registration: the lexical arm arXiv:2609.05637 never ran

## The paper

Written 2026-09-10, before any measurement on this corpus.

Shanian, Yi, Ruban and MacDonald (ServiceNow), *Better Together: Complementary
Query Rewriting Under a Strong RAG Baseline*, arXiv:2609.05637v2, EMNLP 2026.

They fix one pipeline (`bge-base-en-v1.5` dense retrieval, `bge-reranker-base`
cross-encoder, MMR at lambda 0.85, GPT-4.1 for every rewrite) and vary the
query-rewriting strategy across six methods and three datasets. On
EnterpriseRAG-Bench (512K documents, 470 answerable questions) the baseline S1
scores 39.22 HIT@10, the best single method 42.91, a four-way ensemble 51.70,
a five-way 52.98, and an all-method oracle 55.74. Two budget-matched controls, a
TopK-100 rerank pool and a four-seed multi-pass HNSW, reach 44.33 and 44.19,
which is about 40% of the ensemble's gain. They read the remaining 60% as
complementarity between rewriting strategies.

## The axis nothing varies

Every configuration in the paper retrieves densely. The strings `bm25`,
`lexical`, `sparse`, `hybrid`, `keyword`, `tf-idf` and `splade` appear zero
times in the full extracted text; `Rocchio` appears twice, both in Related Work
and the bibliography, and pseudo-relevance feedback is never run. The
budget-matched controls vary retrieval depth and index seeds, both inside the
dense modality.

Their stated mechanism is that rewriting surfaces documents "no amount of
deeper indexing or reranking of a single query can find." A BM25 arm is a second
coverage source over that same single query and costs no LLM call.

The corpus is Slack 285,605 / Gmail 121,390 / Linear 35,308 / Drive 25,108 /
HubSpot 15,017 / Fireflies 10,173 / GitHub 8,052 / Jira 6,120 / Confluence
5,189. Design principle 4 of the benchmark is "internal terminology: project
codenames, product-specific acronyms, and organizational jargon." Of 500
questions, 175 are `basic` (one gold document) and 125 are `semantic`
("roundabout phrasing with less keyword overlap"). We verified the 470 count
from the parquet: 500 minus 20 `info_not_found` minus 10 `high_level`, both of
which carry no `expected_doc_ids`. The paper says "30 unanswerable questions"
and does not name the `high_level` half.

## Date of the thesis

The conclusion — that a zero-LLM lexical arm may rescue a comparable fraction of
the baseline's misses — formed on 2026-09-10 on first read, before any
measurement. Evidence at that moment: the grep above, the corpus composition,
the benchmark's terminology design principle, and three in-house results that
favour lexical retrieval (`hybrid-code-index`, the ReFind review, the 2026-07-04
`searching-codebases` replication). Everything measured here is downstream of a
conclusion already fixed, and this document exists so that is visible.

## The paper's rescue-rate arithmetic

The first draft of the thesis was "the +12.5 is a missing-BM25 artifact."
Working back from the numbers in their own appendix kills that. Appendix Table 7 gives
per-method rescue rates on Enterprise of 14.8% (Query2Doc), 14.5% (HyDE), 13.2%
(S3), 10.1% (S4), 10.1% (S2), against an S1 miss set of 283 queries. Four
methods each recovering 10-15% of 283, overlapping partially, converts roughly
20% of the misses, which is what a +12.5 point gain on a 39.22 base requires.
Their arithmetic is internally consistent and needs no missing arm to explain
it.

What survives is narrower and is what this measures: **the paper does not report
what fraction of its baseline's misses a free lexical arm rescues.** If BM25
rescues 10-15%, it belongs in the ensemble on cost grounds alone. If it rescues
substantially more, the practical recommendation in Section 6 changes.

## Pre-registered null

If the paper is entirely correct, and its gains genuinely come from rewriting
diversity rather than from an absent modality, then on this corpus:

- BM25 alone scores at or below the dense baseline on HIT@10.
- BM25's rescue rate on the dense arm's miss set is at or below the 10-15% the
  LLM methods achieve.
- `rrf(BM25, dense)` lands near the best single method, around 43, and well
  short of the 51.70 four-way ensemble.

If instead the missing arm matters, BM25 rescues materially more than 15% and
fusion lands in the 50s with no LLM call. The two readings predict different
numbers on the same scale, so the measurement discriminates.

## Decision rule

The comparison that counts is **internal**: BM25 against dense against
`rrf(BM25, dense)`, all on this chunking, this corpus build, the same qrels and
the same metric. That needs no anchor to the paper.

The **external** comparison to 39.22 / 42.91 / 51.70 is only valid if our
dense-only arm reproduces their S1 within +/-3 HIT@10 points. Outside that
tolerance we report the internal comparison and state that the external one does
not hold. Our pipeline has no cross-encoder and no MMR, and our chunker is not
theirs, so this check can fail legitimately.

Primary metric is **rescue rate on the dense arm's miss set**, because it is
directly comparable to their Appendix Table 7 and does not depend on reproducing
their absolute scores. HIT@10 and Recall@10 at document granularity are
secondary.

## Adversarial arms

Three ways this comes out against the thesis. Each is reported whatever it says:

1. **BM25 is simply bad here.** Design principle 3 of the benchmark is "realistic
   noise ... near-duplicates with updated or conflicting facts." Near-duplicate
   flooding is a known BM25 weakness and this corpus was built to contain it.
2. **The `semantic` slice.** 125 of 500 questions are written for low keyword
   overlap. Per-category results ship in every table. If BM25 wins overall and
   loses that slice, the paper's dense-first framing is partly right and the
   headline is a mix effect.
3. **No reranker on our side.** Their S1 includes a cross-encoder over the
   retrieved pool. BM25 beating a reranked dense pipeline is a stronger claim
   than BM25 beating our unreranked one, and only the first is interesting.

## The sentence to distrust

"BM25 beat their whole LLM ensemble for free." If the numbers come out that way,
re-check chunk-to-document mapping, near-duplicate collapse, and whether we are
scoring documents where they scored chunks, before writing it down.

## Arms

| arm | retrieval | LLM calls |
|---|---|---|
| A1 | BM25 over chunks | 0 |
| A2 | BM25 over whole documents | 0 |
| B | dense, `bge-base-en-v1.5`, no reranker | 0 |
| C | `rrf(A1, B)` | 0 |
| D | C plus one S4-style rewrite leg | 1 per query |

A1, A2 and B are the experiment. C tests fusion. D is the question the paper
leaves open and runs only if A-C leave it open.

`bge-base-en-v1.5` is the paper's own retriever rather than the `bekko` encoder
`xr` and `remax_kb` use, chosen so the anchor check in the decision rule can
mean something.

## Prediction on arm D

`hybrid-code-index` (2026-08-05) found `rrf(dense, bm25, rg)` losing to
`rrf(dense, bm25)`, 24 of 24 down to 22 of 24, because unweighted RRF lets a weak
arm vote as loudly as a strong one. If that transfers, D scores below C. That is
also a competing mechanism for the paper's own AmbigNQ failure, which they
attribute to document overlap.

## Relevance judgments

`expected_doc_ids` are benchmark-supplied relevance judgments, so every metric
here is computed against labels. The leaderboard's GPT-5.4 answer judge is not
used and no arm requires one.
