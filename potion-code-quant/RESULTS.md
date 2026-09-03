# Static embedding tables against bekko-a8m and quantized SPECTER2

Run 2026-09-03 on CCotw, 4 vCPU / 15 GiB. Oskar's two questions, in order:
does the static Model2Vec code model hold up against our embedders, and does its
output quantize; then, can a static table be made compatible with an index a
transformer built, including a remax/remex-quantized SPECTER2 set. Four passes.
A Sonnet delegate ran passes 1 through 4 from `BRIEF.md` and `BRIEF_SPECTER2.md`;
the parent session verified the pass-2 collapse and the pass-4 cosine
(`check_centered.py`) and wrote this file.

**Headline.** potion-code-16M-v2 trails bekko-a8m by 0.16 r@5 on the n=59 code
task (p<0.001), and bekko's 96-byte remex sidecar still beats potion's
uncompressed 512-byte vector. potion's own output quantizes about as well as
bekko's: remex 2-bit costs 0.015 r@5 (not significant), remax 1-bit costs 0.075
(p=0.10). Model2Vec's int8 table is free. A token-level Model2Vec distillation of
bekko is not compatible with bekko's space (cosine 0.26; r@5 0.02 against
bekko's index). A table fitted by ridge regression against bekko's own sentence
vectors gets to cosine 0.76 on held-out code and recovers r@5 0.36 of bekko's
0.60 against bekko's index. On SPECTER2 the same fit reads as cosine 0.96 raw
but 0.71 after centering, and recovers r@10 0.29 of the teacher's 0.65 against
the remax 1-bit index. None of the static tables is a drop-in query encoder for
an index a transformer built.

## Pass 1: potion vs bekko-a8m on the code task with a quantization ladder

Corpus: scikit-learn `sklearn/` at `7cb1868a` (the merge commit of the newest
instance, PR #34645), AST-chunked by the bekko-bench harness into 11,380 chunks
over 674 files. All 143 gold files of the 59 instances exist in that checkout.
Task: NL issue to gold file set, file-level r@5 / r@10, plus RRF fusion with the
shared ripgrep-identifier arm. Same rg baseline and corpus on every row.

| encoder | codec | dim | B/vec | r@5 | r@10 | RRF r@10 |
|---|---|---|---|---|---|---|
| rg | | | | 0.596 | 0.682 | 0.682 |
| bekko-a8m | fp32 | 384 | 1536 | 0.595 | 0.667 | 0.728 |
| bekko-a8m | remex 2-bit | 384 | 96 | 0.590 | 0.652 | 0.713 |
| bekko-a8m | remax k=1 | 384 | 48 | 0.544 | 0.637 | 0.719 |
| potion | fp16 as shipped | 256 | 512 | 0.436 | 0.533 | 0.668 |
| potion | int8 table | 256 | 256 | 0.436 | 0.532 | 0.668 |
| potion | truncate | 128 | 256 | 0.423 | 0.506 | 0.627 |
| potion | truncate | 64 | 128 | 0.393 | 0.467 | 0.606 |
| potion | remex 1-bit | 256 | 32 | 0.376 | 0.440 | 0.603 |
| potion | remex 2-bit | 256 | 64 | 0.422 | 0.531 | 0.645 |
| potion | remex 4-bit | 256 | 128 | 0.434 | 0.524 | 0.647 |
| potion | remax k=1 | 256 | 32 | 0.361 | 0.451 | 0.626 |
| potion | remax k=2 | 256 | 64 | 0.432 | 0.513 | 0.596 |
| potion | remax k=4 | 256 | 128 | 0.482 | 0.547 | 0.637 |
| potion | remax k=8 | 256 | 256 | 0.474 | 0.536 | 0.659 |

The bekko-a8m fp32 row reproduces the harness's recorded 0.595 exactly. rg alone
still beats every dense arm on this task, as in both prior runs.

Paired tests on per-instance r@5, sign test plus paired bootstrap:

| comparison | Δ r@5 | 95% CI | wins/losses | p |
|---|---|---|---|---|
| potion fp16 vs bekko fp32 | -0.159 | [-0.236, -0.088] | 2/20 | <0.001 |
| potion remex 2-bit vs potion fp16 | -0.015 | [-0.057, +0.026] | 5/6 | 1.00 |
| potion remax k=1 vs potion fp16 | -0.075 | [-0.155, +0.004] | 5/13 | 0.10 |

Two native levers, both checked. `dimensionality=` at load time and slicing the
256-d output then renormalizing give the same scores at d=128 and d=64, so the
table is PCA-ordered as the card says. `quantize_to="int8"` on the table gives
cosine 0.998 mean / 0.930 min to the fp16 output over the corpus and the same
r@5 to three decimals.

Timing on this box. The a8m corpus encode took 282 s on 4 threads; potion took
3.3 s. Per query on one pinned thread, a8m 104.5 ms median, potion 1.30 ms.

Stratum split is uninformative here: one of 59 instances is identifier-poor by
the harness's measured definition. On that one, rg scores 0, bekko 0.667, potion
0.333. The question in cabf6b3d (does a static model hurt the identifier-poor
stratum) stays open at this n.

## Pass 2: Model2Vec distillation of bekko-a8m

`distill("hotchpotch/bekko-embedding-v1-a8m", pca_dims=None)` keeps the
teacher's 384-d basis; the Zipf toggle in model2vec 0.9.0 is `sif_coefficient`
(1e-4 on, None off). Both variants: 255,753 x 384 fp16 = 187 MB.

| student | cos to teacher, corpus mean / min | cos, queries mean / min | student/student r@5 | student query / teacher index r@5 |
|---|---|---|---|---|
| Zipf on | 0.262 / 0.098 | 0.281 / 0.192 | 0.080 | 0.017 |
| Zipf off | 0.208 / 0.080 | 0.237 / 0.076 | 0.050 | 0.018 |

Teacher/teacher is 0.595 on the same task. The parent checked why the student
is this far off: mean pairwise cosine among student vectors is 0.957 (teacher:
0.364), so a single common direction dominates every distilled vector. Centering
the student removes it but only lifts cosine to teacher to 0.38 and top-10
neighbour overlap with the teacher to 0.17. The bag of single-token teacher
outputs does not point where the teacher's contextual output points; Model2Vec's
default PCA step is what makes its own models usable, and PCA leaves the
teacher's space.

## Pass 3: a table fitted by regression against bekko's sentence vectors

Instead of distilling weights, fit the table to the teacher's outputs. X is the
bag-of-tokens matrix over chunks under bekko's own tokenizer, each row divided by
the chunk's full token count so `X @ W` is exactly the mean-pool a deployed
encoder computes; Y is bekko's sentence vectors. Ridge with no intercept, so
`W` is the table. Split 80/20 by file (539/135 files, 8,962/2,418 chunks), vocab
restricted to train frequency >= 2 (12,509 tokens). Alpha 0.1 won over 1 and 10
on held-out cosine (0.763 vs 0.672 vs 0.627). Variant B adds 1,116 docstrings
from train-split chunks, embedded by the teacher, as NL-shaped rows.

| cell | A: chunks | B: +docstrings |
|---|---|---|
| cos to teacher, held-out chunks mean / min | 0.763 / 0.417 | 0.767 / 0.398 |
| cos to teacher, 59 queries mean / min | 0.585 / 0.402 | 0.601 / 0.446 |
| student/student r@5 | 0.230 | 0.235 |
| student query / teacher float index r@5 | 0.322 | 0.360 |
| student query / teacher remex 2-bit index r@5 | 0.364 | 0.385 |
| student query / teacher remax k=1 index r@5 | 0.233 | 0.235 |
| table, fp16 | 9.2 MB | 9.3 MB |

Teacher/teacher r@5 is 0.595; RRF with rg brings the student-query/teacher-index
cell to 0.63 against the teacher's 0.73. The queries are bug reports, not code,
and their cosine to the teacher is 0.17 below the held-out chunks'. The remex
2-bit index scored 0.02 to 0.04 above the float index for student queries; at
n=59 that is inside noise. remax 1-bit costs the student 0.09 to 0.13 r@5 where
it cost the teacher 0.05.

## Pass 4: a fitted table against a remax/remex-quantized SPECTER2 index

Data: the remax bench's 10,000 SPECTER2 vectors (768-d, unnormalized, norm mean
21.7) with their title-plus-abstract texts, from the `specter2-nlp-broad-10k`
release. Protocol follows `remax/bench/sketch_matryoshka.py`: seed 99, 100 query
papers, 9,900-paper corpus, truth = top-10 by raw float inner product. The
table is fitted as in pass 3 against the raw teacher vectors, under SPECTER2's
own WordPiece tokenizer, vocab 18,028 at frequency >= 2, alpha 0.1 by 5-fold
CV. Table 18,028 x 768 fp16 = 26.4 MB. Quantizers are fitted on the teacher
corpus and applied unchanged to student queries.

| cell | index | r@10 | r@100 | top-1 |
|---|---|---|---|---|
| teacher query | float | 1.000 | 1.000 | 1.000 |
| teacher query | remax 1-bit | 0.645 | 0.990 | 0.490 |
| teacher query | remex 2-bit | 0.478 | 0.953 | 0.320 |
| teacher query | remex 4-bit | 0.731 | 0.999 | 0.530 |
| student query | float | 0.264 | 0.581 | 0.140 |
| student query | remax 1-bit | 0.288 | 0.701 | 0.110 |
| student query | remex 2-bit | 0.117 | 0.394 | 0.030 |
| student query | remex 4-bit | 0.195 | 0.532 | 0.050 |
| student query | student float index | 0.002 | 0.056 | 0.000 |

The remax 1-bit row is the one Oskar asked about: a 26 MB table querying the
teacher's 1-bit index gets r@10 0.288 where the teacher gets 0.645, and r@100
0.70 where the teacher gets 0.99. The teacher's remax row lands near the bench's
own 0.620 for full-width centered sign bits.

The delegate reported student/teacher cosine 0.96 on held-out papers and on the
queries. That number is mostly the mean. SPECTER2 vectors share a large common
component: raw pairwise cosine among teacher vectors is 0.847, and 0.001 after
subtracting the corpus mean. After the same centering, student/teacher cosine
on the queries is 0.706, and on out-of-fold corpus rows 0.636
(`check_centered.py`). That is the number the retrieval cells reflect. Sign
agreement between student and teacher query codes after the index's centering
and rotation is 0.753 mean (min 0.62), so about a quarter of the 768 bits flip,
which is what r@10 0.29 on a Hamming index looks like.

The student/student cell of 0.002 is a raw-inner-product artifact on top of a
weak encoder: student vectors have near-constant norm, so an inner-product
ranking among them is a ranking on the poorly fitted residual. Under centered
cosine the same cell scores 0.134. Either way the fitted table is not a usable
encoder on its own for this corpus.

## Findings

- A static code model (potion) sits well below a 124 MB transformer on the
  file-discovery task, and its speed (80x) buys nothing here that bekko's own
  2-bit sidecar does not already give at 96 bytes per vector.
- potion's output quantizes about like bekko's: remex 2-bit is near free, remax
  1-bit costs more, and the ordering matches METHODS' rule to re-measure the bit
  ladder per encoder.
- Token-level Model2Vec distillation does not stay in the teacher's space.
  Fitting a table to the teacher's sentence outputs does, partially: cosine
  0.6 to 0.76 on bekko, 0.71 centered on SPECTER2, and half the teacher's recall
  against the teacher's index in both cases. The remaining gap is the part of
  the teacher's output that no bag of tokens predicts.
- A quantized index does not change that picture. Against a mismatched query,
  remax 1-bit and the float index score about the same on SPECTER2 (0.29 vs
  0.26); remex 2-bit is worse (0.12) because its Lloyd-Max reconstruction is
  tuned to the teacher's coordinate distribution, which the student's residual
  does not follow.

Unrun: fitting the table against sign agreement after rotation rather than
against float vectors, which targets the 1-bit cell directly; and a query-side
adapter (a small learned rotation on the student output), which the pass-3
docstring variant hints would help by 0.02 to 0.04.

## Files

`run.py` (pass 1), `distill.py` (pass 2), `fit_table.py` (pass 3),
`specter2_fit.py` (pass 4), `check_centered.py` (parent verification), each
with a `results_*.json` and a `*_output.log`. `BRIEF.md` and `BRIEF_SPECTER2.md`
are the delegate briefs as sent. Inputs not committed: sklearn checkout at
`/home/user/sklearn-bench`, bekko-a8m ONNX under `/home/user/models`, potion
and the two distilled tables under `../.cache/`, the SPECTER2 cache under the
remax checkout.

One bug the delegate caught before reporting: the first draft of the pass-4
R@100 helper re-sliced predictions to width 10, so R@100 equalled R@10 on every
row including the float reference. Fixed and re-run.
