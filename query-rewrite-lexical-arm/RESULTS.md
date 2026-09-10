# The lexical arm arXiv:2609.05637 never ran

BM25 alone scores 73.62 HIT@10 on EnterpriseRAG-Bench. The paper reports 39.22
for the dense baseline it calls deliberately strong, 52.98 for its best
five-method LLM ensemble, and 55.74 for an all-method oracle, on the same 470
questions over the same corpus. BM25 makes no LLM call, uses no reranker and no
MMR.

That ordering is not a discovery of this experiment. EnterpriseRAG-Bench's own
paper (Sun et al., arXiv:2605.05253) reports BM25 at 68.4% document recall
against 46.0% for OpenAI `text-embedding-3-large`, and writes that "vector
search underperforms expectations even on semantic questions ... the category
explicitly designed to favor embedding-based retrieval." arXiv:2609.05637 cites
that paper as its benchmark source. The finding here is the omission: a study
whose thesis is that rewriting supplies coverage a single query cannot reach,
run entirely without the coverage source its own benchmark paper documents.

Four results follow from the arms below. BM25 rescues 63.2% of the dense arm's
misses while dense rescues 9.0% of BM25's. Fusing dense into BM25 changes
nothing measurable, −0.43 HIT@10 at p=0.639. Adding an S4 comparative rewrite on
top costs 2.13 points at p=0.048. And the rewrite does rescue 12.1% of the
hybrid's misses, inside the 10.1% to 14.8% band the paper reports, so its
complementarity claim reproduces against a baseline that already has both
modalities.

## This run against the benchmark authors' own BM25

This run reproduces the benchmark authors' BM25 with a different implementation
— a streaming scipy CSC index built here, against their OpenSearch — and lands
0.73 below it overall.

| question type | n | this run, Recall@10 | Sun et al., BM25 | difference |
|---|---:|---:|---:|---:|
| basic | 175 | 75.43 | 77.7 | −2.27 |
| semantic | 125 | 43.20 | 43.2 | +0.00 |
| intra_document_reasoning | 40 | 95.00 | 90.0 | +5.00 |
| project_related | 40 | 58.60 | 65.5 | −6.90 |
| constrained | 30 | 85.00 | 85.0 | +0.00 |
| conflicting_info | 20 | 95.00 | 82.5 | +12.50 |
| completeness | 20 | 40.64 | 46.5 | −5.86 |
| miscellaneous | 20 | 90.00 | 90.0 | +0.00 |
| **overall** | **470** | **67.67** | **68.4** | **−0.73** |

The 20-question categories move 5 points per question, so the two large
categories carry the agreement: `basic` within 2.3 points and `semantic`
matching to the decimal. This is the pre-registered check on the sentence
`PLAN.md` said to distrust, and it passes.

The arms above stem with Snowball and drop 33 stopwords; Sun et al. index
through OpenSearch's standard analyzer, which does neither. Running the
whole-document arm their way closes the remaining gap: Recall@10 68.74 against
their 68.4, a difference of 0.34.

## Arms

<!-- TABLE:MAIN -->
| arm | HIT@10 | HIT@1 | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| A2   BM25, whole document, full corpus | 73.62 | 53.83 | 67.67 | 60.47 | 59.42 |
| A2p  A2 with no stemming and no stopword list | 73.83 | 53.40 | 68.74 | 60.05 | 59.52 |
| A1   BM25, 380-word chunks, full corpus | 72.55 | 51.49 | 67.18 | 58.74 | 58.11 |
| A1g  A1 scoring the top 10 chunks, not 10 documents | 71.91 | 51.49 | 66.29 | 58.67 | 57.81 |
| A2s  BM25, whole document, 20% subcorpus | 84.89 | 64.68 | 81.99 | 71.50 | 71.80 |
| A1s  BM25, 180-word chunks, 20% subcorpus | 81.06 | 62.13 | 78.53 | 68.92 | 69.32 |
| A1sg A1s scoring the top 10 chunks, not 10 documents | 80.85 | 62.13 | 77.18 | 68.89 | 68.71 |
| B    MiniLM-L6-v2 dense, 180-word chunks, 20% subcorpus | 53.19 | 31.70 | 47.91 | 38.68 | 38.63 |
| A1s' A1s queried with the S4 comparative rewrite | 77.87 | 57.87 | 74.72 | 64.36 | 64.75 |
| B'   B queried with the S4 comparative rewrite | 50.85 | 27.45 | 45.37 | 34.63 | 35.19 |
| C    rrf(A1s, B) | 80.64 | 54.68 | 77.07 | 64.39 | 65.39 |
| Dr   rrf(A1s', B') — the rewrite's own hybrid | 75.74 | 52.55 | 71.86 | 61.06 | 61.52 |
| D    rrf(A1s, B, A1s', B') | 78.94 | 52.77 | 75.24 | 62.45 | 63.43 |
<!-- /TABLE:MAIN -->

## Question-to-document term overlap

`overlap.py` measures the fraction of each question's stemmed content terms that
appear verbatim in its gold document. Across 470 questions the mean is 72.7%.

| question type | n | mean term coverage | median |
|---|---:|---:|---:|
| conflicting_info | 20 | 93.7 | 100.0 |
| project_related | 40 | 86.3 | 86.8 |
| intra_document_reasoning | 40 | 78.7 | 77.8 |
| constrained | 30 | 75.3 | 75.8 |
| basic | 175 | 75.1 | 76.5 |
| completeness | 20 | 73.8 | 75.1 |
| miscellaneous | 20 | 67.3 | 73.0 |
| semantic | 125 | 59.7 | 59.1 |
| **all** | **470** | **72.7** | **75.0** |

That ordering tracks BM25's per-category HIT@10 closely. The questions were
generated from the documents, so three quarters of a question's content terms
are already in the text it is meant to find. The benchmark's authors built the
`semantic` category as their control for this, and BM25 drops to 43.20 there
against 73.62 overall.

So the margin is partly a property of the benchmark. A corpus of questions a
person actually typed would narrow it, and nothing here measures that. What the
benchmark property does not excuse is running only the modality it disadvantages
and calling the result a strong baseline.

## Per-question-type results

<!-- TABLE:BYTYPE -->
| question type | n | A1s_chunked_sub | B_dense_sub |
|---|---:|---:|---:|
| basic | 175 | 86.29 | 52.00 |
| completeness | 20 | 85.00 | 75.00 |
| conflicting_info | 20 | 95.00 | 70.00 |
| constrained | 30 | 100.00 | 93.33 |
| intra_document_reasoning | 40 | 97.50 | 50.00 |
| miscellaneous | 20 | 100.00 | 90.00 |
| project_related | 40 | 97.50 | 92.50 |
| semantic | 125 | 52.80 | 21.60 |
<!-- /TABLE:BYTYPE -->

## Rescue rates

`PLAN.md` names this the primary metric: among the queries one arm misses at
rank 10, the fraction another arm retrieves. It compares to the paper's Appendix
Table 7, where each LLM method rescues 10.1% to 14.8% of the dense baseline's
283-query miss set.

BM25 rescues **63.2%** of the dense arm's misses, 139 of 220. Dense rescues
**9.0%** of BM25's, 8 of 89. The asymmetry is the finding: on this corpus the
lexical arm recovers most of what dense loses, and dense recovers almost nothing
that lexical loses.

The 63.2% overstates what BM25 would recover from a stronger dense retriever.
Arm B is MiniLM-L6-v2 with no reranker, weaker than the paper's
`bge-base-en-v1.5` pipeline, and a weaker arm misses more rescuable queries. The
9.0% in the other direction does not have that problem, and it is the number
that says dense is close to redundant here.

One rate lands inside the paper's band. The S4 rewrite's lexical leg rescues
**12.1%** of the hybrid's misses, 11 of 91, against their 10.1% to 14.8%. Their
complementarity claim about rewriting reproduces; what changes is the baseline
it should be measured against.

<!-- TABLE:RESCUE -->
| miss set | rescued by | misses | rescued | rate |
|---|---|---:|---:|---:|
| A1s_chunked_sub | B_dense_sub | 89 | 8 | 9.0% |
| A1s_chunked_sub | A1s_chunked_sub_rw | 89 | 8 | 9.0% |
| A1s_chunked_sub | B_dense_sub_rw | 89 | 8 | 9.0% |
| A1s_chunked_sub | C_rrf_A1s_B | 89 | 6 | 6.7% |
| A1s_chunked_sub | D_rewrite_hybrid | 89 | 8 | 9.0% |
| A1s_chunked_sub | D_rrf_all | 89 | 7 | 7.9% |
| B_dense_sub | A1s_chunked_sub | 220 | 139 | 63.2% |
| B_dense_sub | A1s_chunked_sub_rw | 220 | 128 | 58.2% |
| B_dense_sub | B_dense_sub_rw | 220 | 12 | 5.5% |
| B_dense_sub | C_rrf_A1s_B | 220 | 131 | 59.5% |
| B_dense_sub | D_rewrite_hybrid | 220 | 112 | 50.9% |
| B_dense_sub | D_rrf_all | 220 | 124 | 56.4% |
| A1s_chunked_sub_rw | A1s_chunked_sub | 104 | 23 | 22.1% |
| A1s_chunked_sub_rw | B_dense_sub | 104 | 12 | 11.5% |
| A1s_chunked_sub_rw | B_dense_sub_rw | 104 | 11 | 10.6% |
| A1s_chunked_sub_rw | C_rrf_A1s_B | 104 | 24 | 23.1% |
| A1s_chunked_sub_rw | D_rewrite_hybrid | 104 | 7 | 6.7% |
| A1s_chunked_sub_rw | D_rrf_all | 104 | 11 | 10.6% |
| B_dense_sub_rw | A1s_chunked_sub | 231 | 150 | 64.9% |
| B_dense_sub_rw | B_dense_sub | 231 | 23 | 10.0% |
| B_dense_sub_rw | A1s_chunked_sub_rw | 231 | 138 | 59.7% |
| B_dense_sub_rw | C_rrf_A1s_B | 231 | 143 | 61.9% |
| B_dense_sub_rw | D_rewrite_hybrid | 231 | 121 | 52.4% |
| B_dense_sub_rw | D_rrf_all | 231 | 134 | 58.0% |
| C_rrf_A1s_B | A1s_chunked_sub | 91 | 8 | 8.8% |
| C_rrf_A1s_B | B_dense_sub | 91 | 2 | 2.2% |
| C_rrf_A1s_B | A1s_chunked_sub_rw | 91 | 11 | 12.1% |
| C_rrf_A1s_B | B_dense_sub_rw | 91 | 3 | 3.3% |
| C_rrf_A1s_B | D_rewrite_hybrid | 91 | 6 | 6.6% |
| C_rrf_A1s_B | D_rrf_all | 91 | 5 | 5.5% |
| D_rewrite_hybrid | A1s_chunked_sub | 114 | 33 | 28.9% |
| D_rewrite_hybrid | B_dense_sub | 114 | 6 | 5.3% |
| D_rewrite_hybrid | A1s_chunked_sub_rw | 114 | 17 | 14.9% |
| D_rewrite_hybrid | B_dense_sub_rw | 114 | 4 | 3.5% |
| D_rewrite_hybrid | C_rrf_A1s_B | 114 | 29 | 25.4% |
| D_rewrite_hybrid | D_rrf_all | 114 | 17 | 14.9% |
| D_rrf_all | A1s_chunked_sub | 99 | 17 | 17.2% |
| D_rrf_all | B_dense_sub | 99 | 3 | 3.0% |
| D_rrf_all | A1s_chunked_sub_rw | 99 | 6 | 6.1% |
| D_rrf_all | B_dense_sub_rw | 99 | 2 | 2.0% |
| D_rrf_all | C_rrf_A1s_B | 99 | 13 | 13.1% |
| D_rrf_all | D_rewrite_hybrid | 99 | 2 | 2.0% |
<!-- /TABLE:RESCUE -->

## Fusion and the rewrite leg

Fusing dense into BM25 does not help. `rrf(A1s, B)` scores 80.64 against BM25's
81.06, a difference of −0.43 with a 95% interval of [−2.13, +1.06] and p=0.639
over 470 questions. Adding both S4 rewrite legs takes it to 78.94, −2.13 with
[−4.05, −0.21] and p=0.048.

`PLAN.md` predicted that sign before the run, from `hybrid-code-index`
(2026-08-05), where `rrf(dense, bm25, rg)` lost to `rrf(dense, bm25)`, 24 of 24
down to 22 of 24, because unweighted RRF lets a weak arm vote as loudly as a
strong one. Here the same shape holds twice over: dense is the weak arm that
fusion cannot profit from, and the rewrite legs are weaker still.

The rewrite also loses on its own. Queried with the S4 comparative rewrite
instead of the original, BM25 drops from 81.06 to 77.87 (−3.19, p=0.005) and
dense from 53.19 to 50.85. That direction agrees with the paper, which reports
rewriting as competitive at best in isolation.

## Paired bootstrap

<!-- TABLE:BOOTSTRAP -->
| contrast | delta HIT@10 | 95% CI | p |
|---|---:|---|---:|
| B_dense_sub vs A1s_chunked_sub | -27.87 | [-32.13, -23.62] | 0.0 |
| A1s_chunked_sub_rw vs A1s_chunked_sub | -3.19 | [-5.53, -1.06] | 0.005 |
| B_dense_sub_rw vs A1s_chunked_sub | -30.21 | [-34.89, -25.74] | 0.0 |
| C_rrf_A1s_B vs A1s_chunked_sub | -0.43 | [-2.13, +1.06] | 0.639 |
| D_rewrite_hybrid vs A1s_chunked_sub | -5.32 | [-8.09, -2.77] | 0.0 |
| D_rrf_all vs A1s_chunked_sub | -2.13 | [-4.05, -0.21] | 0.048 |
<!-- /TABLE:BOOTSTRAP -->

## Deviations from the pre-registration

`PLAN.md` fixed the decision rule before any measurement. Three things departed
from it, all recorded in `ERRORS.md` and none decided after seeing a result.

**The dense arm is not their retriever.** Arm B was specified as
`bge-base-en-v1.5`, the paper's own encoder, so that the pre-registered ±3-point
anchor to their 39.22 baseline could mean something. It measures 3.7 chunks/s on
this container's four cores, which is 83 hours for this corpus, and the Hub
has no quantised build of it. Arm B runs `all-MiniLM-L6-v2` at 45.3 chunks/s,
the encoder the paper itself falls back to in its own Appendix J. **The external
comparison to 39.22 / 42.91 / 51.70 is therefore not claimed.** `PLAN.md`
pre-registered that outcome as invalidating the external comparison and leaving
the internal one intact.

**Arms B, C and D run on a 20% subcorpus.** All 722 gold documents plus 103,000
sampled non-gold, seed 42. Retaining every gold document while dropping four
fifths of the distractors inflates absolute scores for every arm on it. The
whole-document lexical arm is run on both the full corpus and the subcorpus so
the size of that inflation is visible rather than assumed.

**The corpus is four documents larger than the paper's.** Their Table 2 reports
511,958; the Hugging Face release carries 511,962. Version drift, noted rather
than reconciled.

## Limits of the comparison

The paper's arithmetic is not wrong. Its Appendix Table 7 rescue rates account
for its own +12.5 without any missing arm, which is why the thesis was weakened
before measurement rather than after.

The paper never states its retrieval unit. It says "standard IR metrics",
indexes `sentence chunking (512/50)`, sets per-strategy retrieval depth
K=10/7/5/8, and describes merging "top-20 lists" and "max 20 merged documents",
but nowhere says whether HIT@10 counts ten chunks or ten distinct documents. If
ten chunks, several land in the same document, so their top-10 covers fewer than
ten documents and the figure is depressed against the collapse used here.

Measured, it does not matter. Scoring the top ten chunks directly instead of
pooling 200 and taking ten distinct documents costs 0.64 HIT@10 on the full
corpus (72.55 to 71.91) and 0.21 on the subcorpus (81.06 to 80.85). Both
readings of their metric sit far above 39.22, so the gap is modality, not
granularity.

A second gap sits inside the dense modality and this experiment does not
explain it: Sun et al. get 46.0% document recall from `text-embedding-3-large`
with no reranker, while arXiv:2609.05637 reports 34.33% Recall@10 from a
pipeline that adds a cross-encoder and MMR on top of `bge-base-en-v1.5`. A
reranked pipeline landing 12 points below an unreranked one points at the
encoder, at the chunking, or at something in the pipeline neither paper
describes in enough detail to check.

Nothing here measures a reranker. The paper's baseline includes
`bge-reranker-base` and MMR; these arms include neither. A cross-encoder over a
BM25 pool would very likely score higher than the BM25 numbers reported here,
which makes the omission of the lexical arm more consequential, not less.
