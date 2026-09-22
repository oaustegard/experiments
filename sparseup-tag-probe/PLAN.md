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

## Round 3 (2026-09-22): retrieval in the expanded tag space

Oskar, after round 2: the tag vocabulary is not 325 labels, it is whatever I
assign at write time (5,827 distinct tags, 3,603 used once), and with sparse
storage there is no reason to prune it. The SPARSEUP-shaped object is a binary
vector over the whole tag inventory, expanded with co-occurring tags at PMI
weight — the expansion Muninn's recall already computes. The question that
decides whether it is worth anything is retrieval: does similarity in that
space find the memories a text embedding finds, and what does it find that the
embedding does not?

Relevance: the 467 `refs` links inside the fixture (`elaborates`, `extends`,
citations written at remember() time), from 235 query memories to the memories
they cite. Each query retrieves from the other 3,456. Caveat stated up front:
refs were chosen by me, often after a recall() that ranks by tags and FTS, so
they are biased toward what lexical and tag retrieval already surfaces.

Arms, all over the same corpus and queries:
- tag-binary: cosine over the 5,827-dim binary tag vector
- tag-expanded: the same plus co-occurring tags at weight α·PMI/PMI_max, top 20
  per tag, PMI computed leave-one-out for the query; α ∈ {0.25, 0.5, 1.0},
  all three reported, none selected
- gte-small cosine; TF-IDF word+char cosine; SPARSEUP document–document dot
  and query-mode–document dot (`encode_query` on the 235 queries)
- RRF fusion of tag-expanded (α = 0.5) and gte-small
- controls: random; nearest in time (|Δt|), since refs cite recent memories

Metrics: recall@10, recall@50, MRR over the 235 queries; paired bootstrap
CIs against gte-small; top-10 overlap and the share of relevant hits found by
the tag space and missed by gte-small at k = 10.

Predictions, written before the run:

| quantity | prediction |
|---|---|
| gte-small recall@10 / MRR | 0.45 / 0.30 |
| tag-binary recall@10 | 0.30 |
| tag-expanded (α = 0.5) over tag-binary, recall@10 | +0.05 |
| TF-IDF recall@10 | within 0.03 of gte-small |
| SPARSEUP doc–doc recall@10 | within 0.03 of gte-small; query-mode no better |
| nearest-in-time recall@10 | 0.15 |
| RRF(tag-expanded, gte) over gte, recall@10 | +0.05 |
| relevant hits found by tag-expanded@10 and missed by gte@10 | 15% of gte's misses |

Null reading: if the tag space carries nothing beyond the text, tag-expanded
sits at or below TF-IDF and the RRF fusion is within noise of gte alone. The
refutation to search for: a tag-space win that is really the time baseline
(tags such as `perch-time` and dated project tags encode recency).

## Round 4 (2026-09-22): can gemini-3.5-flash-lite write the tags?

The round-3 vector that ties gte-small was built from tags a frontier model
assigned at write time. Oskar: "Run the flash-lite test on 300 memories." The
question is write-time cost: do tags from `gemini-3.5-flash-lite` (through the
Cloudflare gateway, no training clause) retrieve as well as mine?

Design: pick query memories from the round-3 set and their cited memories,
greedily, until the union is 300 memories. flash-lite writes four to seven tags
per memory from a register prompt that shows the 150 most frequent tags as
examples of the vocabulary's register (it may reuse them or write new ones in
the same shape). Each written tag is snapped to the nearest existing tag by
gte-small cosine when that cosine is >= 0.85, otherwise kept as new. The
round-3 retrieval test then runs on the selected queries three times: my tags
on every memory; flash-lite's snapped tags substituted on the 300; flash-lite's
raw tags substituted on the 300. Same corpus otherwise, same refs relevance.
Also reported: Jaccard between flash-lite's snapped tags and mine per memory,
tags per memory, share of written tags that were new after snapping.

Predictions, written before the first call:

| quantity | prediction |
|---|---|
| my tags, R@10 on the selected queries | 0.65 |
| flash-lite snapped tags, R@10 | 0.55 |
| flash-lite raw tags, R@10 | 0.50 |
| Jaccard(flash-lite snapped, mine) | 0.30 |
| tags per memory written | 5.5 |
| share of written tags new after snapping | 35% |

Null reading: if the tagging step carries nothing model-specific, flash-lite
ties my tags. The refutation to search for: a flash-lite win or tie that comes
from tags copied out of the 150-tag hint (process tags such as `correction`)
rather than from reading the memory; the per-memory Jaccard and the share of
hint tags used are reported for that.

## Round 5 (2026-09-22): the tagger with recall context

Oskar: "Run the tagger with recall context on the same 300." Round 4's untested
cause for the gap: my tags were written with related memories in context. Here
flash-lite gets what a write-time recall would give it: the five most similar
OLDER memories by gte-small cosine (older only, so a cited memory never sees
the memory that cites it), each as its tags and its first 150 characters, with
the specific-names prompt and an instruction to reuse a context tag whenever it
fits. Same 300 memories, same snap, same retrieval test. Control with no model:
each memory tagged with the union of its five older neighbours' tags
(inheritance by embedding alone).

Predictions, written before the first call:

| quantity | prediction |
|---|---|
| flash-lite + context, snapped, R@10 | 0.58 |
| neighbour-tag inheritance, no model, R@10 | 0.50 |
| Jaccard(flash-lite + context, mine) | 0.30 |
| share of written tags taken from the context | 40% |

Null reading: if context adds nothing, the context run ties the specific-names
run at 0.461. The refutation to search for: a context win that is only the
inheritance control, meaning the model contributes nothing over copying the
neighbours' tags.
