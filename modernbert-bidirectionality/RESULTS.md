# modernbert-bidirectionality — results

**Question (handoff 879c3575):** does a bidirectional encoder's dependence on
right context concentrate in a few layers and a few positions?

**Answer.** Layers: every layer but one is individually redundant, and no small
set of them is sufficient. On ModernBERT-base, making any single layer 1–21
causal costs at most 0.14 nats of masked-LM cross-entropy against a 5.06-nat
gap between the bidirectional model and a left-context-only reference; making
layer 0 causal costs 9.4 nats and drops the model below a uniform guess. Keeping
the 8 global-attention layers bidirectional and the other 14 causal retains 33%
of the benefit; the 8 layers with the largest single-layer damage retain 58%.
Positions: near, not anchors. Letting every query read 4 tokens ahead in every
layer retains 59% of the benefit, 8 tokens 79%, 32 tokens 88%. Letting queries
read only `[CLS]`/`[SEP]` beyond the causal boundary is worse than reading
nothing; punctuation (38 positions per 256-token sequence) retains 20%, the same
as reading the 38 masked positions, which carry no lexical content. Attention
mass does concentrate on sinks in the global layers (57–65% of future-directed
attention in layers 6, 9, 12 and 15 lands on the two special tokens), and
routing right context through those positions does not work.

Two models, same pipeline: `answerdotai/ModernBERT-base` (22 layers) and
`jhu-clsp/ettin-encoder-32m` (10 layers, same recipe, independent training).
256 wikitext-103 validation sequences of exactly 256 tokens, 15% masked once
with a fixed seed, 9,728 masked tokens, identical across every arm. 120 arms on
ModernBERT-base in 113 minutes and 69 on ettin-32m in 17, on 4 vCPU, eager
attention, per-layer masks injected through forward pre-hooks (`masks.py`,
14 tests in `tests/`). Pre-registered predictions in [`PLAN.md`](PLAN.md); the
prior-art pass is [`PRIOR-ART.md`](PRIOR-ART.md).

## The reference the predictions are scored against

`retained_trunc = (CE_truncate − CE_arm) / (CE_truncate − CE_bidir)`, where
`truncate` feeds the model tokens `0..i` plus `[SEP]` for each masked position
`i` under its own bidirectional attention. It is what the model can do with no
right context and nothing off its training distribution. The all-causal arm is
not that: on ModernBERT-base it scores 23.5 nats, above the 10.8 of a uniform
guess over the vocabulary, and negative `retained_trunc` values below are that
collapse. The smoke run exposed this and the reference was added before the
full run (PLAN.md, amendment section). `retained` against the all-causal arm is
in `results/*.json` and is not used here.

| reference | ModernBERT-base CE / top-1 | ettin-32m CE / top-1 |
|---|---|---|
| bidirectional (stock) | 1.414 / 0.699 | 2.128 / 0.594 |
| truncate (left context only, in distribution) | 6.471 / 0.132 | 7.351 / 0.096 |
| all layers causal | 23.477 / 0.000 | 8.301 / 0.067 |
| all layers causal, `[CLS]`/`[SEP]` readable | 15.110 / 0.045 | 5.689 / 0.170 |

ettin-32m does not collapse under a causal mask (8.3 against 7.4 for
truncate). The collapse is a property of the ModernBERT-base checkpoint, not
of the architecture.

## Layer axis

### One layer causal, the rest bidirectional

ΔCE against the stock model, in nats. `+special` is the same arm with the two
special tokens readable from the causal layer.

| layer | type | ModernBERT-base | +special | ettin-32m | +special |
|---|---|---|---|---|---|
| 0 | global | **9.442** | **2.820** | 0.257 | 0.085 |
| 1 | local | 0.019 | 0.019 | 0.130 | 0.130 |
| 2 | local | 0.017 | 0.016 | 0.285 | 0.293 |
| 3 | global | 0.067 | 0.034 | 0.420 | 0.061 |
| 4 | local | 0.009 | 0.016 | 0.277 | 0.269 |
| 5 | local | 0.038 | 0.037 | 0.359 | 0.362 |
| 6 | global | 0.031 | 0.014 | 0.215 | 0.079 |
| 7 | local | 0.076 | 0.075 | 0.152 | 0.149 |
| 8 | local | 0.022 | 0.022 | 0.172 | 0.169 |
| 9 | global | 0.142 | 0.015 | 0.407 | 0.265 |
| 10–21 | | 0.030–0.092 | 0.022–0.092 | | |
| sum, layers ≥ 1 | | 1.10 (22% of the 5.06 gap) | 0.85 | 2.42 (46% of 5.22) | 1.78 |

On ModernBERT-base the global layers 3, 6, 9, 12 and 15 lose most of their
single-layer damage when the sinks stay readable (layer 9: 0.142 → 0.015),
which says their contribution as bidirectional layers is mostly the sink read.
Layer 0 does not: 2.8 nats remain with the sinks readable. Two post-hoc arms
(`diag.py`, not pre-registered) opened the `[CLS]` query row at layer 0 so the
sink token could still be built from the whole sequence: 10.82 CE without the
sinks readable, 4.22 with, against 10.86 and 4.23 for the arms without the open
row. The sink's own construction is not what layer 0 is for. Every token's
layer-0 representation needs its right neighbours, and the 21 bidirectional
layers above cannot rebuild what layer 0 did not read. ettin-32m has no such
layer: its layer 0 costs 0.26 nats causal, and its per-layer damage is spread
across the stack (0.13–0.42, largest at the global layers 3 and 9).

### Prefix and suffix

`retained_trunc` with the first k layers causal (`prefix:k`) and with the last
layers causal from k up (`suffix:k`, so layers `0..k−1` stay bidirectional).

| k | 1 | 2 | 3 | 4 | 6 | 8 | 11 | 14 | 16 | 19 | 21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MB prefix | −0.87 | −1.50 | −2.30 | −3.18 | −3.29 | −3.30 | −3.27 | −3.28 | −3.30 | −3.32 | −3.34 |
| MB suffix | +0.07 | 0.00 | −0.06 | −0.07 | +0.06 | +0.24 | +0.56 | +0.77 | +0.84 | +0.97 | +0.99 |

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| ettin prefix | +0.95 | +0.87 | +0.75 | +0.55 | +0.49 | +0.38 | +0.10 | −0.02 | −0.11 |
| ettin suffix | −0.18 | −0.01 | +0.17 | +0.42 | +0.56 | +0.71 | +0.79 | +0.87 | +0.92 |

On ModernBERT-base the prefix curve is layer 0 and nothing else. The suffix
curve is smooth: the top 3 layers causal costs 3%, the top 6 costs 16%, the
top 11 costs 44%. Right context is read throughout the stack, and the layers
that read it late are worth less than the ones that read it early. On ettin the
two curves cross near the middle (5 bidirectional layers retain about half
whichever end they are at).

### Keeping a fixed set bidirectional

| set | layers (MB) | MB | layers (ettin) | ettin |
|---|---|---|---|---|
| global layers | 0,3,…,21 (8) | 0.333 | 0,3,6,9 (4) | 0.426 |
| global layers, sinks readable elsewhere | | 0.337 | | 0.428 |
| every 4th | 0,4,…,20 (6) | 0.272 | 0,4,8 (3) | 0.043 |
| top 4 by single-layer damage | 0,9,13,17 | 0.351 | 3,9,5,2 | 0.536 |
| top 8 by single-layer damage | +7,16,18,3 | 0.582 | | |
| first 4 | | −0.073 | | 0.416 |
| first 8 | | 0.242 | | |
| last 4 | | −3.314 | | 0.376 |
| last 8 | | −3.283 | | |
| local layers only | 14 layers, layer 0 causal | −2.302 | 6 layers | 0.555 |

The dQwen3.5-shaped arm (bidirectional attention in the global layers, one in
three here, one in four there) retains a third of the benefit on ModernBERT-base
and 43% on ettin-32m. Choosing the 8 layers by measured damage rather than by
type gets to 58%. On ettin the 6 local layers (0.555) beat the 4 global ones
(0.426) and every-4th (0.043) is the worst set: with 10 layers, which three you
keep matters more than how many.

## Position axis

Every layer, a query at `i` reads `j ≤ i` freely and `j > i` only inside the
arm's set. `retained_trunc`; set sizes are per 256-token sequence.

| set | positions | ModernBERT-base | ettin-32m |
|---|---|---|---|
| next 1 token | | −0.992 | 0.441 |
| next 2 | | −0.058 | 0.529 |
| next 4 | | 0.592 | 0.595 |
| next 8 | | **0.793** | **0.650** |
| next 16 | | 0.851 | 0.696 |
| next 32 | | 0.877 | 0.743 |
| next 64 | | 0.914 | 0.803 |
| next 128 | | 0.957 | 0.900 |
| `[CLS]`, `[SEP]` | 2 | −1.709 | 0.318 |
| positions 0–3 | 4 | −3.371 | −0.142 |
| special + punctuation | 38 | 0.202 | 0.397 |
| the other masked positions | 38 | 0.182 | 0.004 |
| top 5% by received future attention | 13 | −0.362 | 0.358 |
| top 10% | 26 | 0.001 | 0.411 |
| top 25% | 64 | 0.384 | 0.558 |
| random 5% | 13 | −1.072 | −0.091 |
| random 10% | 26 | −0.247 | −0.025 |
| random 25% | 64 | 0.376 | 0.237 |
| next 8 + punctuation | | 0.933 | 0.932 |
| next 8 + top 10% | | 0.911 | 0.911 |

Half the benefit sits within 4 tokens to the right and 80% within 8 on
ModernBERT-base; ettin-32m is more spread (65% at 8, 90% at 128) but the same
shape. The anchor sets lose to an 8-token lookahead in every row, and on
ModernBERT-base punctuation (0.20) is indistinguishable from the masked
positions (0.18) at the same budget, so what those 38 positions contribute is
not content. The attention-derived sets beat random sets of equal size at 5%
and 10% (they contain the two special tokens, and on ModernBERT-base reading
the sinks is the difference between −1.07 and −0.36) and tie at 25%.
Adding punctuation on top of an 8-token lookahead adds 14 points on
ModernBERT-base and 28 on ettin; the lookahead does most of the work on both.

## Attention mass in the stock model

Per layer over the 256 sequences: share of attention on `j > i`; of that, the
share landing on the top 5% of key positions, on the special tokens, and on
punctuation; and the mass-weighted mean distance `j − i`.

| ModernBERT-base layer | future share | top-5% share | special | punct | mean distance |
|---|---|---|---|---|---|
| 0 (global) | 0.46 | 0.23 | 0.12 | 0.29 | 60 |
| 1–5 (local) | 0.39–0.48 | 0.11–0.29 | 0.03–0.16 | 0.18–0.36 | 6–20 |
| 6, 9, 12, 15 (global) | 0.64–0.68 | 0.64–0.70 | **0.57–0.65** | 0.65–0.70 | 100–110 |
| 16, 17, 19, 20 (local) | 0.42–0.49 | **0.68–0.82** | 0.01–0.03 | 0.09–0.10 | 19–22 |
| 18, 21 (global) | 0.49–0.55 | 0.55–0.62 | 0.26–0.30 | 0.38 | 74–78 |

ettin-32m: the global layers 0, 3 and 6 put 60–77% of future attention on the
special tokens; the local layers 7 and 8 concentrate on a top 5% (0.53, 0.74)
that is not special or punctuation. Attention mass concentrates on 5% of the keys in 10 of 22 ModernBERT-base
layers and 6 of 10 ettin layers, and the ablations above say the concentrated
mass is not where the right-context information travels. The late local layers of ModernBERT-base (16–20) concentrate 70–80%
of their future attention on 5% of positions that are neither special nor
punctuation; what those positions are was not characterised.

## Prediction scorecard

| | prediction | ModernBERT-base | ettin-32m | verdict |
|---|---|---|---|---|
| P1 (70%) | no single layer over 10% of the gap; sum under 50% | layer 0 is 187% of the truncate gap; layers 1–21 each ≤ 3%, sum 22% | max 8%, sum 46% | half wrong: the redundancy claim holds for 21 of 22 layers and the exception is a 9-nat collapse |
| P2 (75%) | global layers alone retain ≥ 70% | 0.33 | 0.43 | wrong |
| P3 (55%) | last 8 retains more than first 8 | first 8: 0.24, last 8: −3.28 | first 4: 0.42, last 4: 0.38 | wrong on both |
| P4 (80%) | punctuation < 30%, special < 10% | 0.20, −1.71 | 0.40, 0.32 | right on the primary model, wrong on ettin's thresholds; on both, every anchor set loses to an 8-token lookahead |
| P5 (80%) | 8 tokens ≥ 60%, 32 ≥ 85% | 0.79, 0.88 | 0.65, 0.74 | right on the primary model, half on ettin |
| P6 (70%) | top-5% keys take over half the future mass in most layers; attention anchors lose to lookahead-8 and beat random by ≤ 15 points | 10 of 22 layers; 0.00 vs 0.79; +25 over random | 6 of 10; 0.41 vs 0.65; +44 | the concentration is there in half the layers, the lookahead comparison holds, the random margin does not |
| P7 (60%) | ettin reproduces P2/P4/P5 within 10 points | | global 10 points apart, punctuation 20, lookahead-8 14 | wrong on the tolerance, the ordering holds |

One right, three wrong, three partial. The two calls that missed by the most
were the layer axis ones: I expected redundancy to make a spaced subset of
layers sufficient, and it does not. Each layer's right-context read is
replaceable by its neighbours; a third of the layers together are not
replaceable by the rest.

## What this says for adaptation

dQwen3.5 keeps three quarters of its layers causal (the Gated DeltaNet ones)
and makes the attention quarter bidirectional. The encoder-side analogue of
that pattern retains 27–43% of the bidirectional benefit here without any
training. An 8-token lookahead in every layer retains 65–79%, and a 32-token
one 74–88%. A bidirectional model trained from a causal one with a small
block-local lookahead in every layer is a cheaper target than full
bidirectionality in a quarter of them, on this evidence. Two caveats carry:
these are untrained ablations of an encoder (an adapted decoder gets gradient
steps to repair what a mask breaks, and these numbers are a floor), and the
task is the encoder's own masked-LM objective (a classification head may need
less of the right context than token prediction does).

## What broke

- The handoff body was lost between the claude.ai session and the memory
  store; only the title survived (memory `1c9ab421`). The design was
  reconstructed and pre-registered before any arm ran.
- The all-causal arm was pre-registered as the "no right context" reference
  and turned out to be a collapse (23.5 nats). The `truncate` reference,
  the `+special` variants and `anchor:first4` were added after the
  16-sequence smoke and before the full run; the thresholds were not changed.
- The code subagent refused a mid-task amendment (layer-count-relative arm
  lists, since ettin has 10 layers) as a possible injection and built to the
  original spec. The amendment was applied by hand afterwards.
- A push to `main` bounced on the repo-index bot's commit; merged, not
  rebased.

## Files

`PLAN.md` predictions and amendments · `PRIOR-ART.md` · `data.py` corpus
builder · `masks.py` per-layer masks and hook injection · `probe.py` the
experiment (checkpointed per arm, resumable) · `diag.py` the two post-hoc
`[CLS]`-row arms · `tests/test_masks.py` (14 tests) · `results/mb.json`,
`results/ettin32.json`, `results/diag_*.json`, `results/*.log`.
