# Latent calculator: a depth-3 tool port on a frozen small model

A frozen Pleias Monad (56.7M) can read an exact arithmetic result injected
into its residual stream at one layer, with no tokens in either direction,
when the result is categorical: the `cmp` operator scores 1.00 exact match
with the result injected at layer 29 and oracle operands. It reads a numeric
result poorly: add 0.23, sub 0.24, mul 0.10, falling from 0.65 on two-digit
operands to 0.27 on six-digit ones. The frozen model alone scores 0.001. The
same result delivered as text tokens is contained in the model's output 0.39
of the time and never in the exact answer format, so the latent port matches
the text route on accuracy (0.387 exact against 0.395 contains) at zero
inserted tokens and 155 ms per answer against 513.

Asking is a separate failure. No layer of either model carries the two
operands in a linearly decodable form at the query token: exact recovery of
operator plus twelve digit slots peaks at 2% (Monad, layer 29) and 3.6%
(SmolLM2, layer 16), flat across depth, while the operator alone is 100%
readable from layer 1. The learned query head reaches 5.7% (Monad) and 12.3%
(SmolLM2) exact on validation and 0% on the held-out length. With the learned
query the port still answers `cmp` 0.91 of the time, because the order of two
numbers survives digit errors that the calculator does not.

The delayed arm, which injects at the position of the first answer token,
scores 0.01: the first answer token is predicted from the query position,
before the injection exists, and for `cmp` that token is the whole answer.

Predictions are in [`PREDICTIONS.md`](PREDICTIONS.md), registered before
the first training run. SmolLM2-135M results are pending (run in progress at
the time of this draft).

## Setup

Both models frozen, fp32, CPU, 4 threads. Trained parts: a two-layer MLP
query head on the residual stream at layer `k` at the last prompt token
(operator plus two operands as six right-aligned digit slots each, 11 classes
per slot), a Python calculator on the argmaxed query, and a result encoder
(sign, 12 digit slots, kind slot, shared digit embedding, MLP) producing one
vector of size hidden. 793k trainable parameters on Monad. The encoder trains
on the frozen model's next-token loss over the answer tokens, gradient
flowing through the frozen upper layers; layers below `k` are cached, so
Monad trains at 0.41 s per batch of 32.

Data: 20,000 training rows over twelve templates and four operators, operand
lengths {1, 2, 3, 4, 6}; `test_in` is 2,000 unseen values at training
lengths, `test_len5` 2,000 rows with at least one 5-digit operand (the
interpolation holdout; see PREDICTIONS.md for why the spec's 5-6 extrapolation
split was changed). Exact match is greedy generation against the answer
string, stopping at the first non-empty line; `contains` scores the answer
string appearing anywhere in the generation, added because the text baseline
answers in prose (`∴ Final answer: **5505**`).

`k` was chosen as the shallowest layer whose linear probe recovers the full
query above 95% on validation. No layer did, so the fallback picked the best:
29 of 64 for Monad, 16 of 30 for SmolLM2.

## Monad: choosing k

Linear probe at the last prompt token, validation exact query recovery and
mean slot accuracy, selected layers:

| layer | exact query | slot acc | operator | len-5 slot acc |
|---|---|---|---|---|
| emb | 0.000 | 0.46 | 0.40 | 0.31 |
| 1 | 0.002 | 0.55 | 1.00 | 0.39 |
| 13 | 0.004 | 0.58 | 1.00 | 0.41 |
| 29 | 0.020 | 0.64 | 1.00 | 0.45 |
| 45 | 0.006 | 0.59 | 1.00 | 0.41 |
| 63 | 0.011 | 0.61 | 1.00 | 0.40 |

The operator is a single linearly readable feature from the first layer on.
The operand digits are not assembled at the query token at any depth. Units
digits are the worst slots (0.41-0.46) and the leading digits the best
(0.84-0.92); the model keeps the size of a number at the query position and
loses its low-order digits. SmolLM2, which tokenizes one digit per token,
shows the same flat curve at a slightly higher level (slot 0.69 at layer 16),
so Monad's multi-digit BPE is not the cause.

The MLP query head at layer 29: operator 1.00, slot accuracy 0.72, exact
query 0.057 on validation and 0.058 on `test_in`; 0.000 on `test_len5`. The
calculator run on its argmax gives the right result on 0.27 of `test_in` and
0.22 of `test_len5`.

## Monad: results

Exact match on `test_in` / `test_len5`, tokens per answer (prompt + inserted
tool tokens + generated), CPU ms per answer at batch 1:

| arm | query | test_in | test_len5 | contains (in / len5) | tokens | tool tok | ms |
|---|---|---|---|---|---|---|---|
| none | – | 0.001 | 0.000 | 0.03 / 0.01 | 19.7 | 0 | 438 |
| text | oracle | 0.000 | 0.000 | 0.39 / 0.29 | 26.7 | 4.8 | 513 |
| text | learned | 0.000 | 0.000 | 0.08 / 0.02 | 26.6 | 4.8 | 632 |
| residual | oracle | **0.387** | **0.262** | 0.39 / 0.26 | 12.9 | 0 | 155 |
| residual | learned | 0.255 | 0.222 | 0.26 / 0.22 | 12.9 | 0 | 222 |
| kv | oracle | 0.252 | 0.238 | 0.26 / 0.24 | 14.2 | 0 | 214 |
| kv | learned | 0.228 | 0.211 | 0.24 / 0.21 | 14.2 | 0 | 282 |
| delayed | oracle | 0.013 | 0.004 | 0.05 / 0.01 | 19.6 | 0 | 421 |
| delayed | learned | 0.006 | 0.001 | 0.04 / 0.01 | 19.5 | 0 | 499 |

By operator and by maximum operand length, `test_in`:

| arm | query | add | sub | mul | cmp | len 2 | len 3 | len 4 | len 6 |
|---|---|---|---|---|---|---|---|---|---|
| residual | oracle | 0.23 | 0.24 | 0.10 | 1.00 | 0.65 | 0.54 | 0.34 | 0.27 |
| residual | learned | 0.05 | 0.04 | 0.04 | 0.91 | 0.40 | 0.25 | 0.23 | 0.23 |
| kv | oracle | 0.03 | 0.04 | 0.03 | 0.93 | 0.30 | 0.28 | 0.26 | 0.22 |
| kv | learned | 0.02 | 0.02 | 0.03 | 0.86 | 0.28 | 0.23 | 0.23 | 0.21 |

Every non-`cmp` score above sits on the `cmp` rows' share of the split (about
a quarter). On `test_len5` the residual arm with oracle operands answers `cmp`
1.00 and the numeric operators 0.00-0.02.

Reading the failures. With a numeric result injected, the residual arm's
wrong answers are numbers of the right sign and length with wrong digits
(`-881714` for `-821918`). The kv arm's wrong answers are copies of an
operand from the prompt (`5496` for `5496 + 9`, `-828708` for
`6790 - 828708`): given a slot it cannot decode, layer 30's attention reads
a number it can. The delayed arm generates the frozen model's own prose
(`Let me try: 1653 - 2064 = -123`) because its first token is produced before
the injection exists.

Training signal per arm (teacher-forced answer-token accuracy on validation
after 3 epochs): residual 0.59, delayed 0.46, kv 0.44. Residual was still
improving at epoch 3; kv plateaued at epoch 2.

Latency. The latent arms are faster than `none` mostly because they generate
3.7 tokens against 10.5: the injected vector also steers the model into
answering tersely. Per-token cost is unchanged.

## Predictions, graded on Monad

* **D1 confirmed.** Frozen Monad: 0.001 on `test_in`, 0.000 on `test_len5`.
* **D2 ungraded by its own rule** (no layer reached 95% extraction), and the
  oracle arm shows it would have failed anyway: consumption of a numeric
  result caps at 0.39 on `test_in` and 0.26 on `test_len5`, not 0.90.
* **D3 confirmed on the generous metric.** Residual-oracle exact match sits
  within 5 points of the text arm's contains-match on both splits (0.387 vs
  0.395; 0.262 vs 0.293), at zero inserted tokens and a third of the
  milliseconds. On exact match the text arm scores zero, which says the
  frozen model does not copy a tool result into the answer slot without prose.
* **D4 refuted as stated, and the reason is in the per-operator table.** Oracle
  beats learned by 4.0 points on `test_len5`, not 15. On numeric operators
  both modes are at the floor because consumption fails first; on `cmp`, where
  consumption works, the gap is 9 points on `test_in` and 15 on `test_len5`
  (1.00 vs 0.85). The query-formation failure is real (calculator right 0.22
  of the time from the learned query) and is masked on three of four operators
  by the consumption failure the prediction assumed away.
* **D5 refuted.** Delayed loses 25 points, to 0.004. The async-by-one form
  blinds the first answer token.

## What this says

A single injected vector carries one categorical fact into a frozen 56M model
reliably and a multi-digit number unreliably, with accuracy falling with digit
count. The frozen upper layers were never trained to serialize digits from a
residual direction, and 20,000 examples of next-token loss through them do not
teach an encoder to find one. The text route is not better at this size: the
model surrounds the number with prose and never lands the bare answer.

The KV-slot arm underperforming the residual add matches the persistent-memory
paper on frozen GPT-2 (2603.22329), where KV extension failed and direct
injection routes worked.

On the query side, both models keep the operator and the magnitude of the
operands at the query token and drop the low-order digits. A query head that
attends over the prompt positions, where the digits are, is the obvious next
arm; so is injecting the result at every answer position rather than one.

## SmolLM2-135M

Pending: run in progress.

## Files

`data.py`, `model_utils.py`, `probe.py`, `query_head.py`, `train_port.py`,
`eval.py`, `run_all.sh`, `test_latent_calculator.py` (21 tests), `results/`
(one JSON per probe, query head, and eval configuration), `journal.jsonl`
(per-epoch training records). `data/` and `ckpt/` are regenerable and
gitignored.
