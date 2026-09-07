# Latent calculator: a depth-3 tool port on two frozen small models

A frozen SmolLM2-135M answers arithmetic prompts at 0.90 exact match with a
calculator wired in between layer 16 and layer 17 and no tokens in either
direction. Two trained parts, 1.3M parameters together and every model weight
frozen: a cross-attention head that reads the operator and operands out of the
layer-16 activations of the prompt tokens (0.993 exact query recovery in
distribution), and an encoder that writes the calculator's result back into
the residual stream after layer 16, one vector per answer step with the digits
stored most-significant first. Addition 0.99, subtraction 0.98, comparison
1.00, twelve-digit multiplication 0.66. The frozen model alone scores 0.001.
The same result pasted into the prompt as text tokens never lands in the
answer format at this size (exact 0.000, present somewhere in the prose 0.19),
so the latent port beats the text route with six fewer tokens per answer and
about half the milliseconds.

With true operands the reading side generalizes to a held-out operand
length at 0.85. The query head does not: it recovers the query on 0.45 of
5-digit prompts because it locates digits by counting positions back from
the end of the prompt, and an unseen operand length shifts the count. End to
end on the held-out length: 0.41.

This took three designs. The phase-1 port read the query from one vector at
the last prompt token (0.10 exact recovery; the digits are not assembled
there at any layer of either model) and injected the result once (0.60 exact
with true operands). Attending over the prompt positions solved asking.
Supplying the result at every answer step lifted reading to 0.75, and storing
the digits most-significant first, so that answer step `j` reads slot `j`, to
0.91. Monad (56.7M, 64 layers of width 256) follows the same curve at a lower
level: 0.94 query recovery, 0.50 end to end on the streamed arm. The phase-1
results are kept below because the predictions were registered against them.

Predictions are in [`PREDICTIONS.md`](PREDICTIONS.md), registered before
the first training run; grades are after the phase-1 results. A demo page
built from the result files is `demo.html`.

## Setup

Both models frozen, fp32, CPU, 4 threads. Trained parts: a two-layer MLP
query head on the residual stream at layer `k` at the last prompt token
(operator plus two operands as six right-aligned digit slots each, 11 classes
per slot), a Python calculator on the argmaxed query, and a result encoder
(sign, 12 digit slots, kind slot, shared digit embedding, MLP) producing one
vector of size hidden. 793k trainable parameters on Monad, 1.06M on SmolLM2.
The encoder trains on the frozen model's next-token loss over the answer
tokens, gradient flowing through the frozen upper layers; layers below `k`
are cached, so Monad trains at 0.41 s and SmolLM2 at 1.1 s per batch of 32,
three epochs over 20,000 rows.

Data: twelve templates, four operators, operand lengths {1, 2, 3, 4, 6};
`test_in` is 2,000 unseen values at training lengths, `test_len5` 2,000 rows
with at least one 5-digit operand, an interpolation holdout (PREDICTIONS.md
says why the spec's 5-6 extrapolation split was changed). Exact match is
greedy generation against the answer string, stopping at the first non-empty
line. `contains` scores the answer string appearing anywhere in the
generation; it exists because the text baseline answers in prose
(`∴ Final answer: **5505**`, `5505 5505 5505`).

Injection arms: `residual` adds the vector to the residual stream after
layer `k` at the query position; `kv` gives layer `k+1`'s attention one extra
key/value slot built from the vector, visible from the query position on;
`delayed` is `residual` one position later. Baselines: `none`, the frozen
model; `text`, the calculator result inserted as tokens after the prompt.
Query modes: `oracle` (true operands to the calculator) and `learned` (the
query head's argmax). The encoder always trains on oracle results.

`k` was to be the shallowest layer whose linear probe recovers the full query
above 0.95 on validation. No layer did on either model, so the fallback took
the best: 29 of 64 for Monad, 16 of 30 for SmolLM2.

## Choosing k: the probe curves

Linear probe at the last prompt token; validation exact query recovery, mean
digit-slot accuracy, operator accuracy, and slot accuracy on `test_len5`.

Monad:

| layer | exact query | slot | operator | len-5 slot |
|---|---|---|---|---|
| emb | 0.000 | 0.46 | 0.40 | 0.31 |
| 1 | 0.002 | 0.55 | 1.00 | 0.39 |
| 13 | 0.004 | 0.58 | 1.00 | 0.41 |
| 29 | 0.020 | 0.64 | 1.00 | 0.45 |
| 45 | 0.006 | 0.59 | 1.00 | 0.41 |
| 63 | 0.011 | 0.61 | 1.00 | 0.40 |

SmolLM2:

| layer | exact query | slot | operator | len-5 slot |
|---|---|---|---|---|
| emb | 0.000 | 0.46 | 0.40 | 0.31 |
| 1 | 0.012 | 0.64 | 1.00 | 0.47 |
| 7 | 0.014 | 0.66 | 1.00 | 0.44 |
| 16 | 0.036 | 0.69 | 1.00 | 0.46 |
| 22 | 0.007 | 0.63 | 1.00 | 0.40 |
| 29 | 0.012 | 0.63 | 1.00 | 0.39 |

The operator is a single linearly readable feature from the first layer on in
both models. The operand digits are not assembled at the query token at any
depth. Units digits are the worst slots (0.41-0.50) and leading digits the
best (0.84-0.93): both models keep the size of a number at the query position
and lose its low-order digits. SmolLM2 tokenizes one digit per token and
Monad chunks digits variably (` 12`, `3`, `45`); the curves have the same
shape, so tokenization is not the cause. Monad's upper 35 layers add nothing
to this readout.

The MLP query heads: Monad exact 0.057 on validation, 0.058 on `test_in`,
0.000 on `test_len5`; SmolLM2 0.123, 0.098, 0.001. Operator 1.00 on both. The
calculator on the learned query is right 0.27 / 0.22 (Monad, `test_in` /
`test_len5`) and 0.31 / 0.22 (SmolLM2).

## Results

Exact match; `contains` on `test_in` / `test_len5`; tokens per answer (prompt
+ inserted tool tokens + generated); CPU ms per answer at batch 1.

Monad, k = 29:

| arm | query | test_in | test_len5 | contains | tokens | tool tok | ms |
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

SmolLM2, k = 16:

| arm | query | test_in | test_len5 | contains | tokens | tool tok | ms |
|---|---|---|---|---|---|---|---|
| none | – | 0.001 | 0.000 | 0.03 / 0.02 | 25.9 | 0 | 489 |
| text | oracle | 0.000 | 0.000 | 0.19 / 0.18 | 32.7 | 6.1 | 554 |
| text | learned | 0.000 | 0.000 | 0.05 / 0.02 | 32.7 | 6.1 | 622 |
| residual | oracle | **0.604** | **0.341** | 0.61 / 0.34 | 18.0 | 0 | 235 |
| residual | learned | 0.298 | 0.220 | 0.30 / 0.22 | 18.0 | 0 | 309 |
| kv | oracle | 0.337 | 0.277 | 0.34 / 0.28 | 18.0 | 0 | 238 |
| kv | learned | 0.262 | 0.221 | 0.26 / 0.22 | 18.0 | 0 | 318 |
| delayed | oracle | 0.268 | 0.107 | 0.30 / 0.13 | 18.8 | 0 | 263 |
| delayed | learned | 0.052 | 0.002 | 0.06 / 0.00 | 18.8 | 0 | 352 |

By operator and by maximum operand length on `test_in`, oracle query:

| model | arm | add | sub | mul | cmp | len 2 | len 3 | len 4 | len 6 |
|---|---|---|---|---|---|---|---|---|---|
| Monad | residual | 0.23 | 0.24 | 0.10 | 1.00 | 0.65 | 0.54 | 0.34 | 0.27 |
| Monad | kv | 0.03 | 0.04 | 0.03 | 0.93 | 0.30 | 0.28 | 0.26 | 0.22 |
| SmolLM2 | residual | 0.60 | 0.62 | 0.22 | 1.00 | 0.93 | 0.82 | 0.61 | 0.39 |
| SmolLM2 | kv | 0.14 | 0.17 | 0.07 | 1.00 | 0.60 | 0.36 | 0.31 | 0.27 |
| SmolLM2 | delayed | 0.53 | 0.31 | 0.23 | 0.00 | 0.52 | 0.42 | 0.31 | 0.09 |

Same, learned query:

| model | arm | add | sub | mul | cmp |
|---|---|---|---|---|---|
| Monad | residual | 0.05 | 0.04 | 0.04 | 0.91 |
| SmolLM2 | residual | 0.12 | 0.08 | 0.08 | 0.94 |
| SmolLM2 | kv | 0.05 | 0.04 | 0.05 | 0.94 |

On `test_len5` with oracle operands the SmolLM2 residual arm scores add
0.14, sub 0.19, mul 0.02, cmp 1.00; Monad's scores 0.00-0.02 on the numeric
operators. `cmp` rows are about a quarter of each split, so every aggregate
above 0.25 is carrying numeric answers.

Reading the failures. The residual arm's wrong numeric answers are numbers of
the right sign and length with wrong digits (`-819118` for `-821918` on
SmolLM2, `-881714` on Monad). The kv arm's wrong answers are often copies of
an operand from the prompt (`5496` for `5496 + 9`, `-828708` for
`6790 - 828708`): given a slot it cannot decode, layer k+1's attention reads
a number it can. Monad's delayed arm generates the frozen model's own prose
(`Let me try: 1653 - 2064 = -123`); SmolLM2's delayed arm answers `a` on
every `cmp` row and gets the number right when the digits come after the
blind first token.

Training signal (teacher-forced answer-token accuracy on validation after
three epochs): Monad residual 0.59, delayed 0.46, kv 0.44; SmolLM2 residual
0.83, delayed 0.75, kv 0.66. SmolLM2 residual went 0.60, 0.76, 0.83 across
epochs and was still climbing.

Latency. The latent arms are faster than `none` mostly because they generate
fewer tokens (Monad 3.7 against 10.5, SmolLM2 5.9 against 13.7): the
injected vector also steers the model into a terse answer. Per-token cost is
unchanged.

## Predictions, graded

* **D1 confirmed** on both models: frozen, 0.001 on `test_in` and 0.000 on
  `test_len5`.
* **D2 ungraded by its own rule** (no layer reached 0.95 extraction), and the
  oracle arms show it would have failed: consumption caps at 0.60 / 0.34
  (SmolLM2) and 0.39 / 0.26 (Monad), not 0.90.
* **D3 confirmed.** Residual-oracle exact match against the text arm's
  `contains` (its exact match is zero on both models): Monad 0.387 vs 0.395
  and 0.262 vs 0.293, within 5 points; SmolLM2 0.604 vs 0.185 and 0.341 vs
  0.182, ahead by 42 and 16. Zero inserted tokens, 155 vs 513 ms and 235 vs
  554 ms per answer.
* **D4 refuted as stated on `test_len5`, confirmed in distribution on
  SmolLM2.** Oracle minus learned on `test_len5`: 4 points (Monad), 12
  (SmolLM2), under the 15 the prediction named. On `test_in` for SmolLM2 the
  gap is 31 points (0.604 vs 0.298), and on `cmp`, the one operator where
  consumption is solved, it is 6-15 points on both models. Query formation
  fails (calculator right 0.22-0.31 of the time from the learned query), and
  on the held-out length the consumption failure the prediction assumed away
  hides it on the numeric operators.
* **D5 refuted on the aggregate, with the mechanism visible.** Delayed loses
  25 points on Monad and 23 on SmolLM2 (`test_len5`, oracle). Where the first
  answer token is a space (SmolLM2 numbers) the delayed arm sits within 7
  points of synchronous on add and mul in distribution; where it carries the
  answer (`cmp`, Monad's two-digit first token) the arm scores zero.

## Interpretation

One injected vector carries one categorical fact into a frozen 56M or 135M
model reliably, and a multi-digit number at a rate that falls with digit
count. The wider model (576 against 256 hidden) reads numbers better at every
length, and its encoder was still improving at three epochs, so the ceiling
on SmolLM2 is not established here. The text route is worse at this size on
both models: the frozen model surrounds the number with prose or repeats it
and never lands the bare answer.

The KV-slot arm underperforms the residual add on both models, in line with
the persistent-memory paper on frozen GPT-2 (2603.22329), where KV extension
failed and direct injection routes worked.

The async-by-one form loses nothing when the first answer token is a
format token and loses the answer when that token is the answer. A fixed
lead-in token before every answer would remove the loss.

On the query side both models keep the operator and the magnitude of the
operands at the query token and drop the low-order digits. A query head that
attends over the prompt positions, where the digits are, is the next arm; so
is injecting the result at every answer position rather than one, and
SmolLM2-135M-Instruct as a third model for a fairer `text` baseline.

## Phase 2: reading the digits where they are, and streaming the result

Two changes, both trained parts, both models frozen as before.

**Attention query head.** Thirteen learned query vectors (operator plus
twelve digit slots) cross-attend over the layer-`k` activations of every
prompt token, with absolute and relative-to-end position embeddings on the
keys; 232k parameters on SmolLM2, 191k on Monad. It replaces the one-vector
MLP head.

**Stream arm.** The result encoder takes the answer-step index as an extra
input and its vector is added to the residual stream after layer `k` at the
query position and again at every answer position, one vector per step.
Still zero tokens in either direction; the encoder is the same size plus a
step embedding.

Exact query recovery (operator and all twelve slots):

| model | head | val | test_in | test_len5 |
|---|---|---|---|---|
| SmolLM2 | one vector at the last token (phase 1) | 0.123 | 0.098 | 0.001 |
| SmolLM2 | attention over the prompt | **0.996** | **0.993** | 0.312 |
| Monad | one vector (phase 1) | 0.057 | 0.058 | 0.000 |
| Monad | attention over the prompt | 0.928 | **0.941** | 0.010 |

In distribution the digits are all there: SmolLM2's head gets every slot at
1.00, Monad's at 0.99 despite its multi-digit tokens. On the held-out
length the same heads fail, and the per-slot pattern says why: on SmolLM2
operand A's slots 1-5 sit at 0.52-0.69 while operand B's slots are at
0.90-0.98. The head finds operand B by counting back from the end of the
prompt and operand A by counting back past B; a B of unseen length shifts
the count. Monad shows the same shape with both operands' middle slots
near 0.5. Position-by-counting does not generalize across lengths; a head
that finds digit runs by content would.

End-to-end exact match, attention head for the learned query:

| model | arm | query | test_in | test_len5 | add | sub | mul | cmp | ms |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2 | residual (phase 1) | oracle | 0.604 | 0.341 | 0.60 | 0.62 | 0.22 | 1.00 | 235 |
| SmolLM2 | residual | learned | 0.603 | 0.254 | 0.60 | 0.61 | 0.22 | 1.00 | 432 |
| SmolLM2 | stream | oracle | **0.753** | **0.573** | 0.84 | 0.81 | 0.38 | 1.00 | 347 |
| SmolLM2 | stream | learned | **0.750** | 0.310 | 0.83 | 0.80 | 0.38 | 1.00 | 2901\* |
| Monad | residual (phase 1) | oracle | 0.387 | 0.262 | 0.23 | 0.24 | 0.10 | 1.00 | 155 |
| Monad | residual | learned | 0.381 | 0.226 | 0.22 | 0.23 | 0.10 | 1.00 | 314 |
| Monad | stream | oracle | 0.502 | 0.300 | 0.43 | 0.45 | 0.15 | 1.00 | 247 |
| Monad | stream | learned | 0.490 | 0.231 | 0.42 | 0.42 | 0.15 | 1.00 | 319 |

By longest operand, SmolLM2 stream with the learned query, `test_in`: 1
digit 1.00, 2 digits 0.99, 3 digits 0.92, 4 digits 0.80, 6 digits 0.55.
Teacher-forced answer-token accuracy after three epochs: SmolLM2 stream
0.91 (0.77, 0.88, 0.91 by epoch, still rising), Monad stream 0.68.

\* The learned-query timing in this eval runs a second prompt pass to feed
the head and is an artifact of the eval harness, not of the mechanism; the
head itself costs under a millisecond per row (0.017 s per batch of 64).
The oracle row is the port's cost.

What changed between the phases. In distribution, asking is solved on
SmolLM2 (learned within 0.3 points of oracle) and nearly on Monad (1.2
points). Reading improved by 15 points on SmolLM2 and 12 on Monad from
supplying the vector at every answer step instead of once, and the gain is
on the numeric operators (SmolLM2 add 0.60 to 0.84, mul 0.22 to 0.38);
`cmp` was already at 1.00. The remaining numeric errors are digit errors in
long results, including transpositions (`174` generated as `147`), which
points at the encoder's layout: results are stored with the units digit in
slot 0, so at answer step `j` the encoder has to infer the result's length
before it can pick the digit to emit. The `stream-left` variant stores the
most significant digit first so step `j` reads slot `j`.

**`stream-left`, SmolLM2.** Same encoder and training, digits stored
most-significant first. Teacher-forced answer-token accuracy 0.83, 0.96, 0.97
by epoch.

| query | test_in | test_len5 | add | sub | mul | cmp | 2 dig | 3 dig | 4 dig | 6 dig | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | **0.911** | **0.846** | 0.99 | 0.99 | 0.67 | 1.00 | 1.00 | 1.00 | 0.96 | 0.80 | 357 |
| learned | **0.905** | 0.411 | 0.99 | 0.98 | 0.66 | 1.00 | 1.00 | 1.00 | 0.95 | 0.79 | 449 |

The layout was worth 16 points in distribution and 27 on the held-out length
with true operands. Addition and subtraction, whose results stay under eight
digits, are at 0.99; the remaining misses are twelve-digit products with the
leading digits right and the last few off (`884262791923` generated as
`884267296266`). On the held-out length the reading side now holds (0.846
with true operands, per-operator add 0.99, sub 0.99, mul 0.42) and the
learned query is the whole gap: the head's calculator input is right 0.45 of
the time there, and the pipeline lands at 0.41.

**Regex query: the null model for asking.** The operands are literals the
user typed, so `regex_query.py` lifts them from the prompt by span-bound
regex (operator from cue words, the first two integer runs, one reversal rule
for "subtract X from Y"), the pattern `nl2sh-retrieval/extract_params.py` and
`monad-bsky` settled on: a model must never retype an identifier it was given.
It is exact on every split, including the held-out length and phrasings the
templates never used, and `eval.py --query regex` on the stream-left arm
reproduces the oracle numbers, 0.911 and 0.846. Anything shipped should use
it. The learned head stays as the measurement the handoff asked for, whether
the frozen model's own activations carry the operands: in distribution yes,
across lengths no.

On the twelve demo rows (`results/demo_smol_left.json`): frozen 0, text 0,
latent 10, at 21 tokens against 28 and 42 and 384 ms against 762 and 834.

## Demo

`demo.py --model smol --n 12 --head attn` runs twelve random test prompts
through the frozen model, the text route and the latent port and prints the
three answers, the decoded query, tokens and milliseconds per arm.
`make_demo_page.py` renders the same into `demo.html` from `results/`.
`demo.py` defaults to the `stream-left` checkpoint; `results/demo_smol.json`
holds the earlier right-aligned run (7 of 12) and `results/demo_smol_left.json`
the final one (10 of 12).

## Files

`data.py`, `model_utils.py`, `probe.py`, `query_head.py`, `train_port.py`,
`eval.py`, `demo.py`, `make_demo_page.py`, `run_all.sh`, `run_phase2.sh`,
`test_latent_calculator.py` (29 tests), `results/`
(one JSON per probe, query head, and eval configuration), `journal.jsonl`
(per-epoch training records). `data/` and `ckpt/` are regenerable and
gitignored. Whole run: 5.4 h on 4 CPU cores.
