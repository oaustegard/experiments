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
have wanted about 8e-5. The BiLSTM is sequential, which is the one part of the
head a GPU cannot batch, so its accuracy value decides whether the browser
carries it. The span head is otherwise gathers, two MLPs and a dot product.
Upstream tokenizes each word without the leading-space marker, which hands a
BPE model word forms it rarely saw in pre-training; A6 tests that.

Later rounds, in order, each gated on the one before:
2. Data scale (NuNER) and distillation from a large GLiNER teacher on
   unlabelled text with sampled label sets.
3. Pack (4-bit token table, int8 layers), NumPy reference, F1 of the packed
   model on the full benchmark.
4. WGSL: hidden-row readback, span kernel, label scoring; parity headless, then
   timing on a real GPU through the GPU Bridge.
5. Page: text box, label chips, highlighted entities.

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
