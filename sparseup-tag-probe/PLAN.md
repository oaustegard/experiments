# PLAN — can SPARSEUP's sparse vectors predict Muninn's human tags?

Pre-registered 2026-09-21 before the first encode. Issue:
[muninn-utilities#137](https://github.com/oaustegard/muninn-utilities/issues/137).
Background: memory `5312cc75` (Oskar's 2024 binary tag-vector idea vs SPARSEUP).

## Question

A learned sparse encoder's dimensions are vocabulary tokens. Do they line up with
a human tag vocabulary well enough that (a) the tags show up among a document's
expansion terms without any training, and (b) a linear map from the sparse vector
to the tag set beats the lexical and dense representations it would replace?

## What each arm varies (thesis-discipline item 7)

| contrast | what differs | what stays fixed |
|---|---|---|
| arm 1 | nothing trained; asks whether the tag *string* is an active dim | the document, k |
| arm 2 vs arm 4 | the representation: SPARSEUP weights vs TF-IDF (word, char) vs gte-small | classifier (one-vs-rest LR), folds, label set |
| arm 2 vs arm 3 | weights kept vs dropped to 0/1 | everything else |

Oskar's request was whether the encoder's output *maps onto* the tag vocabulary.
Arm 1 answers the untrained version; arm 2 vs 4 answers whether a linear map exists
and whether it needs the learned expansion at all.

## Fixture (pinned, `fixture.json`)

3,457 memories after excluding 37 tagged `confidential`/registered private scopes,
from 3,494 live. Tag cleanup dropped 1,098 date tags, 558 bare numbers, 2 hex ids.
325 tags have >= 10 uses; 3,315 memories carry at least one. Memory text is not in
git; `fetch_text.py` re-fetches by id and checks the per-memory sha256.

Arms 2–4 run on the 3,315 labelled memories, stratified 5-fold on the most frequent
label of each memory, one seed. Arm 1 runs on the same 3,315.

## Null readings (thesis-discipline item 1)

- If SPARSEUP carries no tag information at all, arm 2 reads like a
  frequency-prior baseline (always predict the 5 commonest labels; P@5 around
  0.10) and a shuffled-label control. Both are run.
- If it carries only *lexical* information, arm 2 reads within noise of the
  word TF-IDF probe. That is the outcome to distinguish from "the expansion adds
  something": the per-label split below is what separates them.

## Predictions

| quantity | prediction |
|---|---|
| memories truncated at 512 tokens | ~25% (median 1,223 chars, p90 3,112) |
| arm 1 exact-token tag recall @16 / @32 / @64 | 0.15 / 0.20 / 0.30 |
| arm 1 all-parts recall @16 / @32 / @64 | 0.22 / 0.30 / 0.42 |
| arm 2 SPARSEUP + LR micro-F1 / macro-F1 / P@5 | 0.55 / 0.35 / 0.45 |
| TF-IDF (word+char) + LR micro-F1 | 0.57 — **TF-IDF ties or beats arm 2** by 0–3 points |
| gte-small + LR micro-F1 | 0.50, below both |
| binarization gap (arm 2 − arm 3, micro-F1) | <= 0.02 |
| labels where SPARSEUP beats TF-IDF | the lexical ones (tag word appears in the text), not process tags like `correction`, `session-log`, `preference` |

Reasoning for the TF-IDF call: `hypothetical-classification` finding 7 (METHODS.md)
measured char-ngram TF-IDF beating an encoder when documents carry their label
words literally, which is the common case for this tag vocabulary. Half the labels
(148/325) are hyphenated jargon (`remax_kb`, `small-reasoner-big-kb`) that BPE
fragments, which caps arm 1 and gives TF-IDF char n-grams an edge. The binarization
gap is predicted small because LR re-weights active dims itself; what it cannot
recover is the *ranking* within a document, which SPLADE's top-k gating already
compresses.

## Refutation to search for (item 3)

The result I would like is "the mapping exists and needs the learned expansion".
The adversary: any arm 2 win over TF-IDF could be subword tokenization of jargon,
not semantic expansion. Checks scheduled: char-ngram TF-IDF as the lexical arm;
a per-label split by literal-mention rate; the frequency-prior and shuffled-label
controls; a second independent reader of the numbers before the write-up.

## Out of scope

Taxonomy induction and prefix ordering (only worth it if arm 2 shows a mapping).

## Round 2 (2026-09-21, after the round-1 write-up): the 2024 map step as the first layer

Oskar, on the round-1 result: "is it feasible to do a two step — a multi tag
assignment by BPE pieces and from there actual tags?" Measured on the fixture:
a probe restricted to the 522 dims that are pieces of some tag reaches micro-AP
0.328 against 0.612 on the full vector, and the untrained all-pieces rule fires
12.5 tags per memory at 6% precision. Pieces are the wrong intermediate. Oskar:
"First try the LR model" — the same probe on the 2024 idea's own first layer.

Arm: candidate phrases per memory (spaCy noun chunks and entities, lowercased,
stop-words stripped from the edges, 1–4 words) → a vocabulary of K dims selected
inside each training fold by document frequency, with a dissimilarity filter
(skip a candidate whose gte-small cosine to an already-selected dim is > 0.9)
→ a binary presence vector → the same one-vs-rest LR, same folds, C swept and
selected by micro-AP. K ∈ {128, 256, 512, 1024, 2048, all}. Controls at the same
K: frequency-only selection (no dissimilarity filter), and binary word unigrams
by document frequency. No LSH snapping of unseen phrases in this round.

Predictions, written before the first run:

| quantity | prediction |
|---|---|
| phrase vocabulary, K = 512, micro-AP | 0.50 |
| phrase vocabulary, K = 2048, micro-AP | 0.60 |
| phrase vocabulary, all dims | 0.63, still below TF-IDF word+char (0.664) |
| dissimilarity filter over frequency-only at K = 512 | +0.01 to +0.02 |
| binary unigrams by df at K = 512 vs phrases at K = 512 | unigrams within 0.02, either side |
| K at which phrases pass SPARSEUP's 0.612 | ~2048 |

Null reading: if the selection criterion carries nothing, phrases and unigrams
tie at every K and the dissimilarity filter is within noise of frequency-only.
The refutation to search for: a win for phrases at small K that comes from
a few tags that are literally noun chunks (`perch-time`, `session-log`); the
per-label split by literal-mention rate from round 1 is reused.
