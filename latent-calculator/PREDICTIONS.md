# Predictions, registered before the first training run

The question. A harness-external tool costs a round trip and a token stream
in each direction. RETRO and Memorizing Transformers show a tool can sit
between two layers instead, latent in and latent out, when the tool is a
retriever. This experiment asks whether a frozen small language model can
consume an *exact arithmetic result* delivered the same way: injected into
its hidden state at one layer, with no tokens emitted to call the tool and
none consumed to read the answer. And if it cannot, whether the failure is
in forming the query or in using the answer.

Design space and precedent: memory 88477132 (2026-09-06). This is the depth-3
port from that note, with a calculator in place of RETRO's database.

## Setup

Two frozen models: Pleias Monad (56.7M, 64 layers x 256 hidden, multi-digit
BPE digit tokens) and SmolLM2-135M (30 layers x 576, single-digit tokens).
Every model parameter stays frozen. Three trained pieces, under 2M parameters
together:

* **Query head.** Reads the residual stream at the output of layer `k` at the
  last prompt token and emits an operator (add, sub, mul, cmp) and two
  operands as six right-aligned digit slots each, eleven classes per slot
  (0-9 and blank). Trained with supervised cross-entropy on the synthetic
  labels. That routes around the missing gradient through the tool.
* **Calculator.** Plain Python on the argmaxed query. `cmp` returns greater,
  less or equal.
* **Result encoder.** Sign, twelve digit slots and a kind slot, through a
  shared digit embedding and a two-layer MLP, to one vector of size hidden.
  Trained by the frozen model's own next-token loss on the answer tokens,
  gradient flowing back through the frozen upper layers into the encoder.

Injection arms, one encoder each: `residual` adds the vector to the residual
stream at the output of layer `k` at the query position; `kv` gives layer
`k+1`'s attention one extra key/value slot built from the vector, visible only
from the query position onward; `delayed` is `residual` shifted one position
later, onto the first answer token, the async-by-one form that hides tool
latency behind a decode step. Two baselines: `none`, the frozen model alone;
`text`, the same calculator result appended as text tokens after the prompt.
Two query modes at evaluation: `oracle` feeds the true operands to the
calculator; `learned` feeds the query head's argmax. The encoder always
trains on oracle results.

Choosing `k`: a linear probe at every layer for the query labels, fitted
before any encoder is trained. The port goes at the shallowest layer whose
probe recovers the full query (operator and all twelve slots) on more than
95% of in-distribution validation rows. The whole curve is reported.

## Data and the length split

About twelve prompt templates, four operators, operands with up to six
digits, answer as a digit string (with a leading minus where `sub` goes
negative) or one of three words for `cmp`. 20,000 training rows, 2,000
validation, and two 2,000-row test sets, all under 48 tokens on both
tokenizers.

**Amendment to the handoff spec.** The spec trained on operand lengths 1-4 and
held out 5-6. A slot-wise head cannot extrapolate into a slot that was blank on
every training row, so that split would measure a construction artifact and
call it query failure. Training lengths are `{1, 2, 3, 4, 6}` and the held-out
length is **5**: every slot is active on some training row and blank on
others, but the 5-digit pattern (slot 4 active, slot 5 blank) never occurs in
training. Held-out rows have at least one 5-digit operand. `test_in` holds
unseen operand values at training lengths.

## Measured quantities

* **Exact-match accuracy** of greedy generation against the answer string, by
  arm, query mode, operator, and maximum operand length.
* **Query extraction accuracy**: fraction of rows where the query head (and,
  per layer, the linear probe) gets operator and all twelve slots right.
* **Tokens per answer**: prompt tokens, plus inserted tool-result tokens for
  `text`, plus generated tokens. Latent arms insert zero.
* **CPU milliseconds per answer**, batch 1, wall clock, over at least 200 rows.
* **Teacher-forced answer-token accuracy** per epoch, in the training journal.

## Predictions

**D1 — the frozen models cannot do this.** `none` scores below 20% exact match
on `test_len5` for both models, and below 20% on `test_in` rows with a
maximum operand length above 2.

**D2 — the port works where the query is extractable.** With oracle queries,
the best latent arm exceeds 90% exact match on `test_len5` for at least one
model. Graded only if the probe at `k` exceeds 95% on validation; if no layer
reaches 95%, D2 is ungraded and the probe curve is the result.

**D3 — latent costs nothing over text.** The best latent arm with oracle
queries lands within 5 points of `text` on `test_len5`, with zero inserted
tokens and lower CPU milliseconds per answer.

**D4 — query formation is where it fails.** On `test_len5`, `oracle` beats
`learned` by at least 15 points for the best latent arm. That is the NALU and
Tracr-Injection pattern again: the model can use an answer it did not have to
ask for. Refuted if the two modes sit within 5 points, which would put the
bottleneck in consumption instead.

**D5 — the delay is free.** `delayed` loses fewer than 5 points against
`residual` on `test_len5` with oracle queries.

**The result that matters is D4 against D2**: whether a local model can be
taught to ask.

## What I expect to be wrong about

The digit-slot output of the encoder has to be read back out of one vector by
frozen layers that were never trained to serialize a number from a residual
direction. `cmp` results are categorical and should be the easy case; `mul`
results of up to twelve digits are the hard one. If the port passes `cmp` and
fails `mul`, consumption capacity is the bottleneck and D2 fails for a reason
D4 cannot see. Per-operator accuracy is reported for that reason.

## Prior art and the delta

Re-checked 2026-09-06 (arXiv search, four queries). Nothing found that
injects a tool's exact result into a frozen decoder's hidden state and reads
the answer out as tokens.

* Toolformer (2302.04761) and ToolkenGPT (2305.11554): tools in the decode
  loop; the result comes back as text. `text` here is that shape.
* RETRO (2112.04426), Memorizing Transformers (2203.08913): a tool between
  layers, latent in and out, but the tool is a retriever and the model is
  trained with it. Here the model is frozen and the tool is exact.
* Abacus embeddings (2405.17399): teaches the model arithmetic. This does not;
  the calculator is exact and the model only has to ask and read.
* Tool Calling is Linearly Readable and Steerable (2605.07990): the
  *decision* to call a tool is one direction in activation space. That is the
  query side and supports the linear query head; the result side is not
  addressed.
* Trained Persistent Memory for Frozen Decoder-Only LLMs (2603.22329):
  six ways to inject a learned memory into frozen GPT-2 (prefix, KV extension,
  parallel cross-attention, Hebbian, gated branch, slot write). KV extension
  and prefix injection failed at standard capacity; cross-attention and
  slot-write worked. Closest consumption-side precedent; the content there is
  a learned embedding, not a symbolic result that must be read out digit by
  digit. Their KV-extension failure is a warning for the `kv` arm here.
* Prometheus Mind (2601.15324): memory adapters on frozen Qwen3-4B; semantic
  embeddings, staged proxy training because end-to-end failed.
* Latent Reasoning with Supervised Thinking States (2602.08332), and the
  looped and recurrent-depth line (2502.05171, 2507.06203, 2605.26797): the
  model's own latents fed back in; no external exact computation.
* NALU (1808.00508) and the Tracr-Injection line: query-formation failure is
  the known shape when a network must route to an exact module.

The delta: exact, symbolic, externally computed result; frozen consumer;
zero tokens in either direction; and the oracle-query arm that separates
asking from reading.
