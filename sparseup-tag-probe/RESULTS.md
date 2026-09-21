# RESULTS: SPARSEUP sparse vectors as predictors of Muninn's human tags

Measured 2026-09-21 on CPU (4 vCPU). Issue:
[muninn-utilities#137](https://github.com/oaustegard/muninn-utilities/issues/137).
Plan and predictions: [`PLAN.md`](PLAN.md), committed before the first encode.
Errors: [`ERRORS.md`](ERRORS.md), five entries, two of which changed the reading.

## Headline

A linear map from SPARSEUP's 50,370-dimension sparse vector to the 325-tag label
set exists (micro-AP 0.612 against 0.043 for the label-frequency prior and 0.018
for shuffled labels), and TF-IDF over the same text carries more of the same
information: word + char n-gram TF-IDF reaches micro-AP 0.664, and the paired
difference SPARSEUP - TF-IDF is -0.051 [-0.059, -0.044] over documents. Giving
TF-IDF only the 512-token window the encoder saw moves that to -0.044
[-0.051, -0.038], so truncation accounts for a seventh of the gap. On P@5 the
two tie (-0.002 [-0.005, +0.002]). gte-small, a 384-dimension dense encoder, is
below both on micro-AP (0.576) and P@5 (0.389). Dropping SPARSEUP's weights to
0/1 costs 0.021 [0.017, 0.025] micro-AP and 0.008 P@5.

Untrained, the tags are mostly not among a memory's expansion terms: at k = 32,
16.4% of tag instances appear as an exact token, 18.9% with every BPE piece
present, 23.6% with every hyphen-separated part present. Only 98 of the 325 tags
are a single token in SPARSEUP's vocabulary.

The pre-registered metric, micro-F1 at a 0.5 probability threshold, orders the
arms the other way (SPARSEUP 0.532, gte-small 0.516, TF-IDF 0.452). Under 1% of
probability cells cross 0.5 for the sparse and TF-IDF arms, so that ordering
reports calibration. At each arm's best single threshold the F1 ordering matches
AP: TF-IDF 0.627, SPARSEUP 0.612, gte-small 0.565.

## Fixture

3,457 memories from the 3,494 live ones, after excluding 37 tagged
`confidential` or a registered private scope. Tag cleanup dropped 1,098 date
tags, 558 bare numbers and 2 hex ids. 325 tags have >= 10 uses; 3,315 memories
carry at least one and are the probe set. Memory text is not in git;
`fetch_text.py` re-fetches by id and checks the per-memory sha256
(`fixture.json`, corpus sha256 `b778fdb3…`).

Encoding: `Linkup-Platform/linkup-sparseup-embed-v1` via `encode_document`,
512-token maximum; 994 memories (28.8%) exceed it. Mean 522 active dimensions
per memory, 26,254 distinct dimensions used across the corpus. `thenlper/gte-small`
at the same 512 tokens. 44.6 minutes for both.

## Arms

| arm | representation | micro-AP | macro-AP | F1 at best thr. | P@5 |
|---|---|---|---|---|---|
| SPARSEUP | log1p(relu) weights, 50,370-d | 0.612 | 0.511 | 0.612 | 0.430 |
| SPARSEUP binary | same, nonzero → 1 | 0.591 | 0.478 | 0.595 | 0.422 |
| TF-IDF word | uni+bigrams, sublinear, fit per fold | 0.652 | 0.577 | 0.618 | 0.426 |
| TF-IDF char | char_wb 3–5 | 0.655 | 0.580 | 0.621 | 0.428 |
| TF-IDF word+char | hstack of the two | **0.664** | **0.586** | **0.627** | **0.432** |
| TF-IDF word+char, 512-token view | same, on the encoder's token window | 0.656 | 0.580 | 0.621 | 0.428 |
| gte-small | 384-d dense, L2-normalised | 0.576 | 0.512 | 0.565 | 0.389 |
| prior | label frequency in the training fold | 0.043 | 0.008 | 0.115 | 0.091 |
| shuffled | SPARSEUP features, permuted labels | 0.018 | 0.011 | 0.051 | 0.052 |

One-vs-rest logistic regression (liblinear), one stratified 5-fold split shared
by every arm, pooled out-of-fold scores. C swept over {1, 10, 100, 1000, 10000}
and selected per arm by micro-AP; every arm except gte-small picked 10000, the
top of the grid, and gte-small picked 100 (its micro-AP falls from 0.576 at
C = 100 to 0.506 at C = 10000). The 512-token TF-IDF arm ran at 1000 and 10000
only. The full grid is in `results/selected.json`; `make_tables.py` prints it.

Paired bootstrap over documents, 1,000 resamples, each arm at its selected C:

| difference | micro-AP | P@5 |
|---|---|---|
| SPARSEUP - TF-IDF word+char | -0.051 [-0.059, -0.044] | -0.002 [-0.005, +0.002] |
| SPARSEUP - TF-IDF word+char, 512-token view | -0.044 [-0.051, -0.038] | +0.002 [-0.002, +0.005] |
| SPARSEUP - SPARSEUP binary | +0.021 [+0.017, +0.025] | +0.008 [+0.005, +0.011] |
| SPARSEUP - gte-small | +0.036 [+0.027, +0.044] | +0.041 [+0.036, +0.045] |

## Predictions against measurements

| quantity | predicted | measured |
|---|---|---|
| memories truncated at 512 tokens | ~25% | 28.8% |
| arm 1 exact-token recall @16 / @32 / @64 | 0.15 / 0.20 / 0.30 | 0.127 / 0.164 / 0.203 |
| arm 1 all-parts recall @16 / @32 / @64 | 0.22 / 0.30 / 0.42 | 0.174 / 0.236 / 0.305 |
| SPARSEUP micro-F1@0.5 / macro-F1@0.5 / P@5 | 0.55 / 0.35 / 0.45 | 0.532 / 0.355 / 0.430 |
| TF-IDF ties or beats SPARSEUP on micro-F1 | yes, by 0–3 points | no at 0.5 (-0.079); yes at best threshold (+0.016) and on micro-AP (+0.051) |
| gte-small micro-F1 below both | 0.50 | 0.516 at 0.5, above TF-IDF; below both on AP and P@5 |
| binarization gap, micro-F1 | <= 0.02 | 0.056 at 0.5; 0.017 at best threshold; 0.021 micro-AP |
| where SPARSEUP beats TF-IDF | lexical labels only | nowhere: TF-IDF is ahead by 0.078 mean per-label AP on the 200 labels with a literal-mention rate >= 0.5 and by 0.071 on the other 125 |

Six of eight readings were wrong on the pre-registered F1@0.5 metric and four of
them come out as predicted once the threshold is removed. The prediction that
held everywhere was the direction of the untrained recall: below the guess, and
capped by tokenization. `remind-nag`, `zeitgeist-digest` and `perch-time` are
three and four BPE pieces; 227 of 325 tags are two pieces or more.

## Arm 1 in detail

Three hit definitions on the top-k expansion terms, over 3,315 memories:

| k | exact token | every BPE piece | every hyphen part |
|---|---|---|---|
| 16 | 0.127 | 0.135 | 0.174 |
| 32 | 0.164 | 0.189 | 0.236 |
| 64 | 0.203 | 0.249 | 0.305 |

Tags that are one common word recall well: `sleep` and `sage` at 1.00,
`consolidation`, `anthropic` and `boot` at 0.92 (subword, k = 32, n >= 20).
Compound tags that name a Muninn concept recall at zero: `add_repo`,
`agent-memory`, `between-the-spokes`, `blog-publish`, `bike-coach`. Their pieces
are ordinary words and the encoder spends its per-position expansion budget
elsewhere. The literal-mention rate (the tag string appears in the memory text)
averages 0.586 across labels, so most of what arm 1 misses is present in the
input and not promoted to the top of the expansion.

## Bearing on the 2024 tag-vector idea

The question was whether a learned sparse encoder's dimensions land on a human
tag vocabulary closely enough to stand in for a curated one. On this corpus they
land on it weakly untrained and no better than n-grams trained: the linear map
SPARSEUP supports is worth 0.05 micro-AP less than the one TF-IDF supports over
the same words, and a fifth of that is the 512-token window. What the expansion
adds over surface forms (`player → footballer`) is not what separates
`perch-time` from `session-log`. The binary version of the SPARSEUP vector loses
0.02 micro-AP and 0.008 P@5 to the weighted one, so the "bits, not weights" half
of the idea costs little; the "dimensions are concepts" half is where the gap
is. Taxonomy induction and prefix ordering stay out of scope: the mapping exists,
but not in a form that beats the lexical baseline it would replace.

## Costs and what broke

Encode 44.6 min; five C sweeps 1,406 + 1,914 + 2,018 + 1,069 + 1,034 s plus 25
min for the 512-token TF-IDF arm; three `select.py` runs at ~24 min each, most
of it the eight bootstrap comparisons. Two Sonnet workers wrote the arm, selection
and check scripts (323k tokens) and one ran the adversarial read (123k tokens);
Fable wrote the fixture, encoder and this file.

Five errors logged. The two that changed the reading: one C across arms whose
feature scales differ by 30× (ERRORS #3, caught reading the C = 1 table against
the prediction), and F1 at a fixed 0.5 threshold ordering arms by calibration
(ERRORS #4, caught by the scheduled adversarial read). The 512-token TF-IDF arm
(ERRORS #5) came out of the same read. `recheck.py` recomputes the fixture hash,
the fold sizes, every selected arm's micro-AP from its cached probabilities, and
greps this file for every number in `results/headline.json`.
