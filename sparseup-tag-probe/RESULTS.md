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

## Round 2: the 2024 map step as the first layer

Oskar, on the round-1 result: "is it feasible to do a two step — a multi tag
assignment by BPE pieces and from there actual tags?" Two quick measurements
on the fixture said the piece layer is the wrong intermediate: a probe on only
the 522 dims that are pieces of some tag reaches micro-AP 0.328 against 0.612
on the full vector, and the untrained rule "every piece active → tag" fires
12.5 tags per memory at 6% precision. Oskar: "First try the LR model", so this
round puts the same probe on the 2024 idea's own first layer.

Candidate phrases per memory are spaCy `en_core_web_sm` noun chunks and
entities, one to four words, edge stop-words stripped (median 48 per memory,
7,655 with df >= 3). A vocabulary of K dims is selected inside each training
fold by document frequency, with or without a dissimilarity filter (skip a
candidate whose gte-small cosine to a chosen dim exceeds 0.9), and the memory
becomes a binary presence vector. Controls at each K: frequency-only phrases and
document-frequency word unigrams. Same folds, same LR, C ∈ {1 … 10000} selected
by micro-AP. Predictions are in `PLAN.md`, round 2. `phrase_vocab.py`, 17.5 min.

| K | phrases + dissimilarity | phrases, frequency only | word unigrams |
|---|---|---|---|
| 128 | 0.242 | 0.240 | 0.274 |
| 256 | 0.309 | 0.302 | 0.332 |
| 512 | 0.358 | 0.369 | 0.395 |
| 1024 | 0.395 | 0.416 | 0.457 |
| 2048 | 0.418 | 0.453 | 0.506 |
| all (7,655 / 16,316) | — | 0.481 | 0.573 |

micro-AP, best C per cell; C = 1 wins everywhere below K = 2048, C = 10000 only
for the full unigram vocabulary. For comparison, round 1: SPARSEUP 0.612,
TF-IDF word+char 0.664, gte-small 0.576.

| quantity | predicted | measured |
|---|---|---|
| phrases, K = 512 | 0.50 | 0.369 |
| phrases, K = 2048 | 0.60 | 0.453 |
| phrases, all dims | 0.63 | 0.481 |
| dissimilarity filter over frequency-only, K = 512 | +0.01 to +0.02 | −0.011 (and −0.035 at K = 2048) |
| unigrams vs phrases, K = 512 | within 0.02 | unigrams +0.026 at K = 512 and +0.053 at K = 2048 |
| K at which phrases pass SPARSEUP's 0.612 | ~2048 | never; the full 7,655-phrase vocabulary reaches 0.481 |

Every prediction was too optimistic about the phrase layer, and by a margin
that grows with K. Three readings of the table:

- **A curated few-hundred-dim vocabulary loses most of the signal.** 512
  phrase dims carry 0.37 of the 0.66 a full TF-IDF matrix carries; 2,048 carry
  0.45. The tags depend on a long tail of words and on phrase-internal words the
  chunker keeps inside a longer span (`human-assigned tags` is one dim, `tags`
  is not).
- **Noun chunks are a worse dictionary than words.** At the same K, single
  words beat phrases by 0.03 to 0.05, and the full unigram vocabulary (0.573)
  beats the full phrase vocabulary (0.481) by 0.09. Chunking merges the units
  that predict tags into rarer compounds.
- **The dissimilarity filter costs, not buys.** Removing near-duplicates by
  gte-small cosine takes out phrases whose surface variants are separately
  predictive, and at K = 2048 the loss is 0.035. Selection by frequency alone is
  the better of the two.

Feasible, then, in the sense that the map step builds and trains in minutes.
As a replacement for the encoder or the n-gram matrix it is not: at every
vocabulary size the binary phrase vector is the weakest representation
measured, and the gap to SPARSEUP (0.612) is not closed by the whole phrase
inventory. What the 2024 design would need is the part this round did not
build — snapping unseen phrases onto selected dims — and the unigram column
puts an upper bound on what that could recover at each K.

## The tagger

`tag_model.py` is the model round 1 pointed at: word + char TF-IDF and one
logistic regression per tag at C = 10000, trained on the 3,315 labelled
memories (344,968 features, 665 s single-threaded — the char n-grams and the
high C are what the last 0.05 AP costs; word unigrams at C = 100 fit in
seconds). Per-label thresholds come from this arm's out-of-fold scores, so the
out-of-fold numbers at those thresholds are optimistic by construction: micro
precision 0.667, recall 0.651, F1 0.659, 2.9 tags predicted per memory against
2.96 assigned. The fitted model holds n-grams of memory text and stays in
`data/`.

On the three memories written after the fixture was pinned, the only unseen
inputs available, it predicted `delegation` for the Muse note (its one tag in
the label set) and `experiments-repo` and `muninn-utilities` for the two
experiment write-ups, and missed `experiments`, `measured`, `negative-result`
and `ccotw` on them. Most of their tags (`sparseup`, `tag-vectors`, `muse`)
have fewer than ten uses and are outside the label set. Three long, atypical
memories are not an evaluation; the out-of-fold numbers are.

## Round 3: retrieval in the tag space

Oskar, after round 2: the tag vocabulary is whatever I assign at write time,
5,827 distinct tags of which 3,603 are used once, and a sparse vector has no
reason to prune it. The SPARSEUP-shaped object is a binary vector over the whole
inventory, expanded with co-occurring tags at PMI weight. The test is retrieval,
whether that space finds the memories a text embedding finds.

Relevance is the 466 `refs` links inside the fixture, from 235 memories to the
memories they cite, written at remember() time. Each query retrieves from the
other 3,456. Two caveats stated before the run: refs are chosen by me, often
after a recall() that ranks by tags and full-text search, so they favour what
lexical and tag retrieval already surface; and refs cite recent memories (median
age gap 0.1 days, 13% over 30 days), so a nearest-in-time ranking is a serious
control. `retrieval.py`, predictions in `PLAN.md` round 3.

| arm | R@10 | R@50 | MRR | R@10 vs gte-small |
|---|---|---|---|---|
| tag-binary (5,827 dims, 5.7 active) | 0.667 | 0.832 | 0.566 | +0.019 [−0.033, +0.069] |
| tag-expanded, α = 0.25 | 0.672 | 0.821 | 0.543 | +0.024 [−0.029, +0.079] |
| tag-expanded, α = 0.5 | 0.610 | 0.803 | 0.477 | −0.038 [−0.097, +0.021] |
| tag-expanded, α = 1.0 | 0.465 | 0.730 | 0.365 | −0.183 [−0.245, −0.117] |
| gte-small | 0.648 | 0.824 | 0.513 | — |
| TF-IDF word+char | **0.808** | 0.928 | **0.619** | +0.160 [+0.112, +0.207] |
| SPARSEUP doc–doc | 0.711 | 0.872 | 0.552 | +0.063 [+0.021, +0.104] |
| SPARSEUP query–doc | 0.657 | 0.828 | 0.509 | +0.009 [−0.035, +0.054] |
| RRF(tag-expanded α = 0.5, gte) | 0.707 | 0.903 | 0.577 | +0.059 [+0.015, +0.101] |
| RRF(TF-IDF, gte) | 0.753 | **0.937** | 0.595 | +0.105 [+0.072, +0.139] |
| nearest in time | 0.593 | 0.751 | 0.421 | −0.055 [−0.122, +0.013] |
| random | 0.005 | 0.015 | 0.004 | |

Recency check, pre-registered: on the half of the pairs whose age gap is above
the median (121 queries, 233 pairs), nearest-in-time falls to R@10 0.122 while
tag-binary holds at 0.534 against gte-small's 0.566 (−0.033 [−0.113, +0.045]),
TF-IDF 0.678 (+0.112 [+0.037, +0.183]), and RRF(tag-expanded α = 0.5, gte) 0.617
(+0.051 [−0.016, +0.116]). The tag-space result is not recency.

Of the 211 relevant memories gte-small misses at k = 10, tag-expanded (α = 0.5)
finds 79 and TF-IDF 111; the mean top-10 Jaccard between the tag space and
gte-small is 0.18, so the two rank different neighbourhoods.

| quantity | predicted | measured |
|---|---|---|
| gte-small R@10 / MRR | 0.45 / 0.30 | 0.648 / 0.513 |
| tag-binary R@10 | 0.30 | 0.667 |
| expansion (α = 0.5) over binary, R@10 | +0.05 | −0.057; α = 0.25 +0.005 |
| TF-IDF R@10 | within 0.03 of gte | +0.160 |
| SPARSEUP doc–doc | within 0.03 of gte; query mode no better | +0.063; query mode −0.054 below doc–doc |
| nearest in time R@10 | 0.15 | 0.593 (0.122 on the distant half) |
| RRF(tag-expanded, gte) over gte | +0.05 | +0.059 |
| gte misses rescued by the tag space at 10 | 15% | 37% |

Three readings:

- **Five or six human tags per memory retrieve as well as a 384-dim text
  embedding.** tag-binary ties gte-small on the full set and on the distant
  half, and fusing the two beats either. The tags carry a view of the memory
  the text embedding does not (top-10 Jaccard 0.18), and 37% of what gte misses
  at 10 the tag space has.
- **PMI expansion does not help this space.** At α = 0.25 it is within noise of
  the plain binary vector and every larger weight costs; three sparser
  exploratory variants (top 3–5 per tag, pair count >= 3–5, not pre-registered)
  land at 0.651–0.673, the same as binary. With 5.7 tags per memory and
  co-occurrence learned on 3,457 memories, the expansion adds neighbours faster
  than it adds relevant ones. The SPARSEUP analogy holds for the vector shape,
  not for the expansion step.
- **TF-IDF is the strongest single representation here, as in round 1.** Part of that is the ground truth: refs written after a lexical
  recall favour lexical neighbours. Part is that I cite in the same words I
  wrote. SPARSEUP in document mode is next; its query mode, meant for short
  queries against long documents, is worse when the query is itself a memory.

What this says for the store: the tags are already a retrieval channel worth
fusing with the embedding, unexpanded. Canonicalizing the 70 surface-variant
families is the one cleanup that would tighten it. Expansion, if wanted, should
be learned from text, not from co-occurrence over this few documents.

### Round 3 addendum: the map step without the LLM

Oskar: "this seems promising for document tagging in general, my theory?" The
2024 page's map step was spaCy phrases, not a model assigning tags. The same
retrieval test with a binary vector over machine-extracted phrases (df >= 3,
selected by frequency) and over word unigrams:

| binary vector over | dims | R@10 | R@50 | MRR | R@10 vs gte-small |
|---|---|---|---|---|---|
| spaCy phrases, K = 512 | 512 | 0.223 | 0.401 | 0.195 | −0.425 |
| spaCy phrases, K = 2048 | 2048 | 0.377 | 0.589 | 0.315 | −0.271 |
| spaCy phrases, all | 9,865 | 0.517 | 0.724 | 0.448 | −0.131 [−0.193, −0.073] |
| word unigrams, K = 512 | 512 | 0.371 | 0.536 | 0.300 | −0.277 |
| word unigrams, K = 2048 | 2048 | 0.555 | 0.716 | 0.452 | −0.093 |
| word unigrams, all | 19,563 | 0.701 | 0.839 | 0.549 | +0.054 [−0.000, +0.103] |
| human tags (round 3) | 5,827 | 0.667 | 0.832 | 0.566 | +0.019 |

The six model-assigned tags per memory do what about 20,000 unigram dims do and
what 9,865 phrase dims do not. The compression is in the tagging, not in the
vocabulary selection.
