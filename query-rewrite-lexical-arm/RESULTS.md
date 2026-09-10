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

## Arms

<!-- TABLE:MAIN -->
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
<!-- /TABLE:BYTYPE -->

## Rescue rates

`PLAN.md` names this the primary metric: among the queries one arm misses at
rank 10, the fraction another arm retrieves. It compares to the paper's Appendix
Table 7, where each LLM method rescues 10.1% to 14.8% of the dense baseline's
283-query miss set.

<!-- TABLE:RESCUE -->
<!-- /TABLE:RESCUE -->

## Paired bootstrap

<!-- TABLE:BOOTSTRAP -->
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

The comparison to their absolute numbers carries an unresolved confound: the
paper never states its retrieval unit. It says "standard IR metrics", indexes
`sentence chunking (512/50)`, sets per-strategy retrieval depth K=10/7/5/8, and
describes merging "top-20 lists" and "max 20 merged documents". Nowhere does it
say whether HIT@10 counts ten chunks or ten distinct documents. If ten chunks,
several land in the same document, so their top-10 covers fewer than ten
documents and the figure is depressed against the collapse used here.

The `_chunkgran` arms bracket both readings by scoring the top ten chunks
directly instead of pooling 200 and taking ten distinct documents. The gap
between the two is how much of the difference is metric rather than modality.

Nothing here measures a reranker. The paper's baseline includes
`bge-reranker-base` and MMR; these arms include neither. A cross-encoder over a
BM25 pool would very likely score higher than the BM25 numbers reported here,
which makes the omission of the lexical arm more consequential, not less.
