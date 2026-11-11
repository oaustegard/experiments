# gliner-browser: plan

Written 2026-11-10, before any training. Predictions are committed here so the
results can be scored against them.

## Question

GLiNER (urchade/GLiNER) does zero-shot named-entity recognition: the entity
types are given as text at run time. The smallest published checkpoint with a
generalist benchmark number is 60M parameters and 243 MB (65 MB as third-party
int8). Can the same task run from a download under 25 MB, in a browser tab, on
our hand-written WebGPU engine, and how much accuracy does that cost?

## Prior art (survey of 2026-11-10, sources in RESULTS.md when written)

- GLiNER on Ettin-32m and 68m exists (knowledgator/gliner-bi-edge-v2.0, a
  bi-encoder, 54.0 on the 7-set zero-shot benchmark at threshold 0.4, trained on
  10M GPT-4o-labelled samples that are not public). Did not find a GLiNER on
  Ettin-17m, a generalist GLiNER at or under 32M parameters with a benchmark
  number, GLiNER weights under 25 MB, or a hand-written WGSL GLiNER.
- GLiNER.js runs ONNX through onnxruntime-web; its smallest model is 183 MB.
  kleinhirn has hand-written WGSL encoders including ModernBERT, with GLiNER
  entity extraction listed as not implemented.
- Paper numbers do not reproduce with released weights (GLiNER issue 54), so
  every baseline is re-run in our harness.
- Weight-only quantization keeps GLiNER working; quantized activations break it
  (Dakera, Tazner model cards). Our packer is weight-only already.
- Our own negative result to beat: a 17M specialist did not transfer to an
  unseen schema in browser-decider (leave-one-workflow-out 0.353, below the
  prior). GLiNER's premise is that transfer, so the held-out benchmark is
  measured first, before any engine work.

## Method

Backbone: stock `jhu-clsp/ettin-encoder-17m` (7 layers, hidden 256), with a 32m
arm for the size curve. Model code: the upstream `gliner` library, uni-encoder,
driven by our own train and eval scripts (upstream `train.py` drops config keys
and `eval.py` cannot run a zero-shot-only directory).

Training data: `urchade/pile-mistral-v0.1` (19,724 passages, Apache-2.0), the
set the v2.1 checkpoints list. Round 2 adds `numind/NuNER` (about 1M sentences,
MIT). Train splits of the benchmark sets are never used.

Benchmark: the 7-set zero-shot benchmark from the GLiNER paper (CrossNER AI,
literature, music, politics, science; MIT movie, MIT restaurant), test splits
from InstructUIE's `IE_INSTRUCTIONS.zip`, entity-level exact-match micro F1 per
set, unweighted mean of the 7. Reported at threshold 0.5 (upstream default) and
at the threshold chosen on dev. Model selection (checkpoint, threshold, which
arm goes forward) uses the CrossNER dev splits only; the MIT dev files are
byte-identical to test, so MIT is never used for selection.

Baselines re-run in the same harness: `urchade/gliner_small-v2.1` (611 MB fp32,
183 MB int8 ONNX), `gliner-community/gliner_small-v2.5`,
`knowledgator/gliner-bi-edge-v2.0`, `urchade/gliner_large-v2.1`.

### Round 1 arms (all on pile-mistral, seed 0)

| arm | backbone | head | BiLSTM | lr enc / other | batch x steps | other |
|---|---|---|---|---|---|---|
| A0 upstream | 17m | span markerV0 | yes | 1e-5 / 5e-5 | 8 x 30k | dropout 0.4, upstream recipe unchanged |
| A1 | 17m | span | yes | 5e-5 / 1e-4 | 32 x 12k | |
| A2 | 17m | span | yes | 1.5e-4 / 3e-4 | 32 x 12k | |
| A3 | 17m | span | no | 5e-5 / 1e-4 | 32 x 12k | |
| A4 | 17m | span | no | 1.5e-4 / 3e-4 | 32 x 12k | |
| A5 | 17m | token-level | no | 1.5e-4 / 3e-4 | 32 x 12k | |
| A6 | 17m | span | no | 1.5e-4 / 3e-4 | 32 x 12k | every word tokenized with a leading space |
| A7 | 17m | span | no | 1.5e-4 / 3e-4 | 32 x 12k | dropout 0.1 |
| A8 | 32m | span | no | 5e-5 / 1e-4 | 32 x 12k | hidden 384 |

Why these: the upstream learning rate was tuned for DeBERTa and our 17M runs
have wanted about 8e-5. The BiLSTM is sequential, so it would run on the CPU
(JavaScript or WASM) between the GPU encoder and the head; its accuracy value
against its measured milliseconds decides whether the browser carries it. The span head is otherwise gathers, two MLPs and a dot product.
Upstream tokenizes each word without the leading-space marker, which hands a
BPE model word forms it rarely saw in pre-training; A6 tests that.

Amendment, 2026-11-11 00:08 UTC, after A3, A4 and A6 reported (test mean at
0.5: 25.1, 29.8, 36.7): leading-space tokenization is worth about 7 points, far
more than P5 predicted, so every remaining comparison is re-based on A6 instead
of A4. A1, A2, A5, A7 and A8 are dropped unrun and replaced by A9 (A6 plus
BiLSTM), A10 (A6 with the token-level head), A11 (A6 with dropout 0.1) and A12
(A6 on 32m, hidden 384, same learning rate). P3, P4 and P6 are scored on these
against A6. A0 runs as planned.

Later rounds, in order, each gated on the one before:
2. Data scale (NuNER) and distillation from a large GLiNER teacher on
   unlabelled text with sampled label sets.
3. Pack (4-bit token table, int8 layers), NumPy reference, F1 of the packed
   model on the full benchmark.
4. Browser runtime, hybrid by design (Oskar, 2026-11-10: not WGSL only; WASM
   plus WGSL or plain JavaScript are all fine, the GPU is used where it pays).
   The encoder runs on the existing WGSL engine with a hidden-row readback.
   Everything after it (BiLSTM if kept, span head, label scoring, decoding) is
   built first in JavaScript, timed per stage on a real GPU through the GPU
   Bridge, and moved to WGSL or WASM only where its share of the time earns it.
   Parity is checked headless first.
5. Page: text box, label chips, highlighted entities.
6. In-browser LoRA trainer (Oskar, 2026-11-10: a LoRA tool that runs in a
   browser on a capable laptop is worth having, as with the Spiky Ball
   trainer). Extend `ettin-wgsl-train`'s `Trainer` with the GLiNER head's
   forward and backward (gathers, span MLPs, span x label scores, focal loss)
   and a JavaScript port of GLiNER's batch construction, so a user can adapt
   the shipped packed model to their own entity types from a few labelled
   examples. Checked against PyTorch with the existing gradient-check harness,
   run on a real GPU through the GPU Bridge. Depends on rounds 3 and 4.

## Predictions

- P1. Baselines in our harness at threshold 0.5: gliner_small-v2.1 between 50
  and 56; gliner-bi-edge-v2.0 between 49 and 55.
- P2. A0 (upstream recipe on 17m) lands between 25 and 38. The best 17m arm of
  round 1 lands between 38 and 46, so 8 to 15 points under gliner_small-v2.1.
- P3. Removing the BiLSTM costs at most 1.5 points (A4 vs A2, A3 vs A1).
- P4. The token-level head is within 2 points of the span head (A5 vs A4).
- P5. Leading-space tokenization gains 1 to 3 points (A6 vs A4).
- P6. 32m beats the best 17m arm by 4 to 7 points.
- P7. Round 2: NuNER adds 2 to 5 points on 17m; distillation adds 1 to 4 more.
- P8. Packing at 4-bit table and int8 layers costs at most 1.0 point; the
  download is at most 12.5 MB.
- P9. Browser, real GPU: a 256-token text with 10 labels, tokenize to entities,
  in at most 15 ms; max logit difference from PyTorch under 1e-3 on the fp32
  path.

Decision rule: if the best 17m model after round 2 is under 35, ship 32m
(about 23 MB packed) and report 17m as the negative result. Success is a
download under 25 MB within 10 points of gliner_small-v2.1 re-run.

## Compute

Colab L4, one session, arms run serially and resume-safe. Budget 20 units for
round 1.

## Change of direction, 2026-11-11 01:20 UTC

Oskar: the Ettin shrink-and-pack pattern was meant to be applied to an existing
GLiNER, not to a GLiNER retrained from stock Ettin. Round 1 is closed early:
A0, A3, A4 and A6 reported (21.1, 25.1, 29.8, 36.7 on the 7-set test mean at
0.5, against 54.6 for gliner_small-v2.1); A9 and A10 each stopped on an
out-of-memory batch and A11 and A12 were not finished. With 19,724 training
passages a 17M backbone is 18 points short, and the published Ettin-32m GLiNER
needed 10M labelled samples to reach 54.

New base: `knowledgator/gliner-bi-edge-v2.0` (Apache-2.0), 54.2 in our harness.
Its text encoder is Ettin-32m, which our packer and WGSL engine already run;
its head is the markerV0 span head plus a BiLSTM, which the NumPy reference
already reproduces; its label encoder is all-MiniLM-L6-v2 (a BERT), which is
new. Labels are short and can be cached per label set, so the label encoder is
a candidate for plain JavaScript or WASM.

Order of work:
1. Pack it: 4-bit token tables, int8 layers and heads; NumPy reference with
   parity against the library; F1 of each packed variant on the full benchmark.
2. Shrink further only where the measurement says it pays (label encoder
   depth or width, code range, table bits).
3. Browser runtime, page and LoRA trainer as in rounds 4 to 6 above.

Predictions: P10, the packed download is between 30 and 40 MB (text encoder
about 23 MB, label encoder about 12 MB). P11, packing at 4-bit tables and int8
weights costs at most 1.5 points of the 54.2. P12, the label encoder at int8
contributes less than a third of that loss.
