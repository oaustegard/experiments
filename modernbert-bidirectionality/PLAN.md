# modernbert-bidirectionality — plan and pre-registered predictions

Written 2026-09-20 before any arm ran. Handoff from the claude.ai session
(Muninn memory `879c3575`): *"ModernBERT bidirectionality probe — does
non-causal dependence concentrate in few layers and few positions?"* Only that
title survived in the memory store (the body was lost at write time, memory
`1c9ab421`), so the design below is reconstructed from the title, the tags
(`anchor-tokens`, `encoder-architecture`, `dqwen3.5`) and the same day's
encoder work (`encoder-platform-survey`, `msd-context-classifier`). Where the
original spec and this one disagree, this one is what ran.

## Question

A bidirectional encoder lets every position read every other. How much of that
right-to-left (future-context) information flow is actually used, and where?
Two axes:

1. **Layers.** If a masked token's prediction can be recovered with only a few
   layers left bidirectional and the rest causal, bidirectionality is a
   property of a few layers, and an AR→bidirectional adaptation (LLM2Vec-style,
   or the hybrid Gated-DeltaNet/attention dQwen3.5 line where only the attention
   layers can cheaply go bidirectional) needs to touch few layers.
2. **Positions.** If future context reaches a token through a few *anchor*
   positions (special tokens, punctuation, attention sinks), a decoder could
   be made "bidirectional enough" by letting it read a handful of summary
   positions rather than every future token.

The two axes get separate answers. The position axis has a locality confound:
"few positions" can mean *anchors* (a fixed small set) or *near* positions
(the next few tokens). Both are tested against each other.

## Model and data

- **Primary:** `answerdotai/ModernBERT-base` — 22 layers, 12 heads, layers
  0,3,6,…,21 (8 layers) are global attention, the other 14 are sliding-window
  with a ±64-token window (`config.sliding_window = 64`). Loaded with the
  `eager` attention path (1.1 s per batch of 8×165 tokens on 4 vCPU, same as
  sdpa), so attention weights can be read.
- **Replication:** `jhu-clsp/ettin-encoder-32m` — same architecture recipe and
  tokenizer (10 layers, 6 heads, global at 0,3,6,9, same ±64 window),
  independent training run, ~6× faster. Layer-axis arms are defined relative to
  the layer count; the `*8` keep-arms are skipped on it. If the layer and position
  findings hold on both, they are properties of the recipe, not of one
  checkpoint.
- **Corpus:** wikitext-103 raw validation split (`Salesforce/wikitext`),
  headings dropped, consecutive paragraphs of an article joined until ≥ 256
  tokens, truncated to exactly 256 tokens including `[CLS]`/`[SEP]`. 256
  sequences (32 batches of 8), fixed order, saved to `data/`. No padding
  anywhere, so the masks are clean.
- **Masking:** 15% of non-special positions per sequence, seed 20260920, one
  draw, the *same* masked inputs for every arm. ~9,700 masked tokens. All
  metrics are computed at masked positions only.

## Mechanics

Every arm is a per-layer boolean mask `allowed[l, i, j]` (query `i` may read
key `j` at layer `l`), AND-ed with the model's own base mask (all-true for
global layers, `|i−j| ≤ 64` for local layers), converted to the additive float
mask and injected with a forward pre-hook on each encoder layer. Two tests
guard the injection: (a) an all-true arm reproduces the stock model's logits to
1e-4; (b) under the fully causal arm, changing tokens after position `i` leaves
the layer-22 hidden state at `i` unchanged to 1e-4.

Metrics per arm: mean cross-entropy at masked positions (`CE`), top-1 accuracy,
and **fraction of the bidirectional benefit retained**

    retained = (CE_causal − CE_arm) / (CE_causal − CE_bidir)

where `CE_bidir` is the stock model and `CE_causal` is every layer causal. A
retained of 1 means the arm loses nothing; 0 means it is as bad as a fully
causal model. Wall-clock and forward count per arm are recorded.

## Arms

**Reference (2):** `bidir` (stock), `causal` (all 22 layers causal).

**Layer axis (70):**

| family | arms | reads |
|---|---|---|
| `single:l` | layer `l` causal, all others bidirectional, l = 0..21 | which single layer's right-context reading matters; if the 22 ΔCEs sum to far less than the causal Δ, bidirectionality is redundant across layers |
| `prefix:k` | layers 0..k−1 causal, k = 1..21 | can right context be picked up late? |
| `suffix:k` | layers k..21 causal, k = 1..21 | can right context be picked up early and then processed causally? |
| `keep:global` | only the 8 global layers bidirectional | the dQwen-shaped question: one bidirectional layer in three |
| `keep:local` | only the 14 local layers bidirectional | the complement |
| `keep:every4` | layers 0,4,8,…,20 bidirectional (6 layers) | a 1-in-4 pattern that does not coincide with the global layers |
| `keep:top4`, `keep:top8` | the 4 / 8 layers with the largest `single:l` ΔCE bidirectional | is the single-layer ranking additive? |
| `keep:first4`, `keep:last4`, `keep:first8`, `keep:last8` | | early vs late |

**Position axis (~24), applied at every layer** — a query at `i` reads `j ≤ i`
freely and `j > i` only if `j` is in the arm's allowed set:

| family | arms | reads |
|---|---|---|
| `look:m` | `j ≤ i + m`, m ∈ {1, 2, 4, 8, 16, 32, 64, 128} | the locality control: how much of the benefit is the next few tokens |
| `anchor:special` | `[CLS]`, `[SEP]` | sinks as routers |
| `anchor:punct` | special + tokens whose text is punctuation (`.`, `,`, `;`, `:`, `(`, `)`, `"`, `@-@`, `@,@`, `@.@` in wikitext) | the Clark et al. 2019 targets |
| `anchor:attn:p` | the top p ∈ {5, 10, 25}% of positions per sequence by future-attention mass received in the stock model (sum over layers and heads of attention from `i` to `j > i`) | data-derived anchors |
| `anchor:rand:p` | p ∈ {5, 10, 25}% random positions, seed-fixed | the control for `anchor:attn` at equal budget |
| `anchor:masked` | the other masked positions | a negative control: positions that carry no lexical information |
| `look:8+punct`, `look:8+attn:10` | union | do anchors add anything on top of locality? |

**Descriptive (1 pass):** from the stock model's attention weights, per layer:
share of attention mass on `j > i`; of that future mass, the share landing on
the top 5% of key positions, on special tokens, and on punctuation; and the
mean distance `j − i` weighted by future mass. This is what "concentration"
would look like if measured by attention alone; the arms above measure whether
the concentration carries information.

## Predictions (confidence)

- **P1 (70%)** No single layer matters much: the largest `single:l` ΔCE is
  under 10% of the causal Δ, and the 22 single-layer ΔCEs sum to under 50% of
  it. Right-context reading is redundant across layers because any
  bidirectional layer can re-gather what a causal one dropped.
- **P2 (75%)** `keep:global` (8 of 22 layers bidirectional) retains ≥ 70% of
  the benefit. Local layers only reach 64 tokens ahead anyway, and the global
  layers are spaced every third layer, so nothing is more than two layers from
  a bidirectional read. This is the prediction that bears on dQwen3.5-style
  hybrids.
- **P3 (55%)** `keep:last8` retains more than `keep:first8`. Near coin-flip:
  late bidirectionality reads right context from representations that never
  saw it; early bidirectionality reads it raw and then processes it causally.
  Stated so the direction is on record.
- **P4 (80%)** Anchors do not route information: `anchor:punct` retains
  under 30% of the benefit, and `anchor:special` under 10%. A masked token
  needs the *specific* next words, which sinks do not carry; attention mass on
  sinks is the head's idle state, not a channel.
- **P5 (80%)** Locality dominates: `look:8` retains ≥ 60% and `look:32`
  ≥ 85%. "Few positions" will be true in the sense of *near*, not *anchor*.
- **P6 (70%)** Descriptive concentration is real but not causal: the top 5% of
  key positions receive over half of the future attention mass in most layers,
  yet `anchor:attn:10` retains less than `look:8` — and less than
  `anchor:rand:10` plus 15 points at most. Measuring attention mass would have
  said "concentrated"; the ablation says otherwise.
- **P7 (60%)** ettin-32m reproduces the ordering in P2, P4 and P5 within 10
  points of retained.

If P2 holds and P4 holds, the answer to the handoff question is *layers yes,
positions no (except locally)*. If P5 fails and P4 holds, right context is
diffuse in both axes and there is no cheap adaptation on this axis.

## Prior art (searched before the run; full notes in `PRIOR-ART.md`)

- **dQwen3.5** (arXiv 2609.20751) bidirectionalises only the softmax-attention
  layers of Qwen3.5 (6 of 24 or 8 of 32, every fourth layer) and leaves the
  Gated DeltaNet layers causal; it never asks whether fewer attention layers
  would do. `keep:global` (8 of 22) and `keep:every4` (6 of 22) are the
  encoder-side version of that ratio.
- **DecBERT** (Findings of NAACL 2022) causally masks the first two layers of
  BERT as a stand-in for position embeddings and loses nothing on GLUE. That is
  one point on the `prefix:k` curve; nobody has swept `single:l`.
- **Clark et al. 2019** measured over half of BERT's attention in layers 6–10
  landing on `[SEP]`, with `[CLS]` early and punctuation late. That is the
  descriptive pass here; the anchor arms test whether the mass carries
  information.
- **Attention sinks in diffusion LMs** (arXiv 2510.15731): masking 5 sinks costs
  1–3% in LLaDA/Dream/MMaDA, so sinks are cheap to *remove*. Whether they are
  sufficient to *route through* is the untested direction, and P4 bets no.
- **StreamingLLM** sinks are positional (the first ~4 tokens); BERT's are
  type-defined. Added arm before the run: `anchor:first4` (S = positions 0–3),
  same budget as `anchor:special` plus two.
- **Confound (unfixable here):** ModernBERT's global layers also use RoPE
  theta 160k against 10k in local layers, so `single:l` on a global layer
  changes direction and frequency regime together. Comparisons within a layer
  type are clean; global-vs-local contrasts carry this caveat.

## Amendment after the 16-sequence smoke, before the full run

The smoke (8 arms, 16 sequences) showed the all-causal reference at CE 23.7,
above the 10.8 nats of a uniform guess over the vocabulary: a bidirectional
model with every layer causal is off its training distribution, not merely
deprived of right context. Three additions, made before any full arm ran:

- **`truncate`** — an in-distribution left-only reference: for each masked
  position `i` the model sees tokens `0..i` plus `[SEP]` under its own
  bidirectional attention. Smoke CE 6.19. **`retained_trunc`**, computed
  against it, replaces `retained` as the metric the predictions are judged on;
  `retained` (against `causal`) is still reported. Negative `retained_trunc`
  means worse than having no right context at all, i.e. off-distribution
  damage rather than information loss.
- **`+special` variants** of `causal`, `single:l` and `keep:global`: the causal
  layers may still read `[CLS]` and `[SEP]`. Smoke: `single:0` CE 11.1 but
  `single+special:0` CE 4.0, so most of the layer-0 damage is losing the sink,
  not losing the right context. The sweep separates the two.
- **`anchor:first4`** (positional sinks, from the prior-art pass).

The prediction thresholds are unchanged and now apply to `retained_trunc`.
Smoke values that bear on them (n=16, not the result): `keep:global` 0.29,
`look:8` 0.77, `anchor:punct` 0.24, `anchor:attn:10` −0.04. P2 looks likely to
fail; it stands as written.

## Not measured here

Training anything; ModernBERT-large; downstream tasks (the MLM objective is
the model's own, so it is the fairest single probe, but a classifier head may
need less of the right context than token prediction does); decoders (the
dQwen3.5 question is *about* a decoder, and this is the encoder-side half of
it: what a model trained bidirectionally actually uses).
