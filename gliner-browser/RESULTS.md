# RESULTS: zero-shot entity recognition in a browser tab

Named-entity recognition marks the spans of a text that name things: people, companies, places, drugs, songs.
GLiNER does it zero-shot. The entity types are given as plain words when the model runs ("person", "song title",
anything), and the model scores every span of up to 12 words against every type. The question here was whether a
GLiNER can run inside a web page from a small download, and how much accuracy that costs. The page runs the model on
a hand-written WebGPU engine, the browser interface for running programs on the graphics card, which the other
experiments in this repo share. Those experiments use Ettin, a family of small text encoders (17M and 32M parameters
here) that turn text into vectors. Oskar, who asked for the work, wanted the download small.

Accuracy is measured on the 7-set zero-shot benchmark from the GLiNER paper (CrossNER AI, literature, music,
politics and science; MIT Movie; MIT Restaurant). The score is entity-level exact-match F1 per set, averaged over
the seven sets, in points out of 100. Each span gets a score between 0 and 1 and counts as an entity when the score reaches the
threshold, which is 0.5 by default in the GLiNER library. The score is reported at that default and at a threshold
picked on the dev split, the part of each set used for choosing settings.

Two directions were tried, in this order. Round 1 trained a new GLiNER on the 17M-parameter Ettin encoder and was
closed early, after four of its training runs had reported. The later work packed an existing model,
`knowledgator/gliner-bi-edge-v2.0`, and built a browser runtime and a page for it. Packing means storing the model's
weights in fewer bits so the download is smaller: the token tables (one vector per word piece) at 4 bits per number
and the encoder layers as 8-bit integers (int8) in place of 32-bit floats (fp32), while the arithmetic still runs in
fp32. The plan numbered the rounds 1 (recipe), 2 (more data and distillation, never run), 3 (packing), 4 (browser
runtime), 5 (page) and 6 (an in-browser LoRA trainer, not built); the Log names the later stages instead of
numbering them. Predictions P1 to P12 were written into [`PLAN.md`](PLAN.md) before each stage, and the Log gives
each one's wording and verdict in the stage where it was scored. The page is live at
https://austegard.com/experiments/gliner-browser/web/.

## Answer

Yes. `knowledgator/gliner-bi-edge-v2.0`, a 60.8M-parameter GLiNER, runs in a browser tab with no loss in the benchmark's mean score. <!-- @evidence[[results/summary.json#baselines.B3.params_M]] -->
Packed with 4-bit token tables and int8 weights it scores 54.28 against 54.23 unpacked, and the browser returns the same entities as the packed reference. <!-- @evidence[[results/summary.json#pack_cost.all_t4-w8L63.test_at_05]] --> <!-- @evidence[[results/summary.json#pack_cost.fp32.test_at_05]] -->
The packed files are 49.5 MB, or 39.9 MB with brotli compression, down from 241.4 MB. <!-- @evidence[[results/summary.json#sizes.shipped_pack_q11.download_raw_MB]] --> <!-- @evidence[[results/summary.json#sizes.shipped_pack_q11.download_brotli_MB]] --> <!-- @evidence[[results/summary.json#sizes.fp32.download_raw_MB]] -->
On one Mac's GPU, a 256-token text with 10 encoded entity types takes 5.3 ms. <!-- @evidence[[results/summary.json#mac.gpu_head_256tok_10lab_total_ms]] -->
Each new entity type costs about 62 ms once, to encode it in JavaScript; the page then caches it. <!-- @evidence[[results/summary.json#mac.label_encoder_ms_per_label_10]] -->

A GLiNER trained on Ettin-17m from 19,724 open passages fell far short. The best run (A6) scored 36.7, 17.9 points under `gliner_small-v2.1`. <!-- @evidence[[results/summary.json#round1.best_arm_gap_to_B1]] -->

Two measurements would change this answer. One is accuracy on Bluesky posts, which was not measured. The other is a 17M text encoder trained on the 10M GPT-4o-labelled samples behind the published Ettin-32m model, which are not public.

## Findings

1. **Packing costs no measurable accuracy.** The full pack scores 54.28 against 54.23 in fp32. <!-- @evidence[[results/summary.json#pack_cost.all_t4-w8L63.test_at_05]] -->
   Quantizing one part at a time (storing just that part at lower precision) moved the score by at most 0.19 points, with one exception. <!-- @evidence[[results/summary.json#pack_cost.text_table_t4.delta_test_at_05]] -->
   The exception is the text encoder's layers with int8 codes held to ±31, which cost 0.48. <!-- @evidence[[results/summary.json#pack_cost.text_layers_L31.cost_test_at_05]] -->
   The fp32 pack matches the library on 48 of 48 benchmark cells, which are the 7 test and 5 dev sets at 4 thresholds each. (`results/pack_cost_biedge.json`) <!-- @evidence[[results/summary.json#pack_cost_fp32_check.cells]] -->
   Individual predictions do change, since 19% of entities at 0.5 differ from fp32, and the mean score does not move. (Log, packing section)
2. **The browser gives the same entities as the reference.** Headless Chromium matched the NumPy reference on 48 of 48 entity checks per pack, on both packs and both span heads. These are not the 48 benchmark cells of finding 1. <!-- @evidence[[results/summary.json#/web_parity/gliner-bi-edge-v2.0-t4-w8L63-h8/entity_matches]] -->
   Logits, the raw scores the model gives each span and entity type before the threshold, differ from the reference by at most 2.5e-5 in headless Chromium and by at most 2.2e-5 on the Mac. <!-- @evidence[[results/summary.json#/web_parity/gliner-bi-edge-v2.0-fp32/logit_max_abs_diff]] --> <!-- @evidence[[results/summary.json#mac.parity_logit_max_abs_diff]] -->
   The NumPy reference itself is within 1.1e-4 of the library on the 55 inputs of up to 1,024 tokens. <!-- @evidence[[results/summary.json#reference_parity.max_abs_logit_diff_up_to_1024_tokens]] --> <!-- @evidence[[results/summary.json#reference_parity.inputs_up_to_1024_tokens]] -->
   (`results/web_parity.json`, `results/bridge_gliner-web-2.json`, `results/pack_parity_biedge.json`)
3. **The span head's per-word layers belong on the GPU, beside the encoder.** The span head takes the text encoder's word vectors and the entity-type vectors and scores each span of words against each type; its per-word layers are dense layers applied to every word's vector. On the Mac, a 30-token text took 139.9 ms with the span head in JavaScript. <!-- @evidence[[results/summary.json#mac.js_head_30tok_10lab_total_ms]] -->
   Of that, 126.9 ms went to the per-word dense layers. <!-- @evidence[[results/summary.json#mac.js_head_30tok_10lab_span_words_ms]] -->
   With the head in the encoder's GPU submit, the single batch of work sent to the graphics card, the whole text took 1.9 ms. <!-- @evidence[[results/summary.json#mac.gpu_head_30tok_10lab_total_ms]] -->
   (`results/bridge_gliner-web-2.json`)
4. **Batches of short texts run at 2,732 texts a second on the Mac.** That is 100 texts of 4,388 tokens in all, with 20 entity types, pipelined. <!-- @evidence[[results/summary.json#mac.batch100_texts_per_second_pipelined]] --> <!-- @evidence[[results/summary.json#mac.batch100_tokens]] -->
   Run serially it is 2,445 texts a second, and 8.07 with the JavaScript head. <!-- @evidence[[results/summary.json#mac.batch100_texts_per_second_serial]] --> <!-- @evidence[[results/summary.json#mac.batch100_js_head_texts_per_second_serial]] -->
   (`results/bridge_gliner-web-2.json`)
5. **Encoding new entity types is the slowest step a user meets.** The label encoder turns each type into a vector, in JavaScript, at 62 ms per type on the Mac for 10 types. <!-- @evidence[[results/summary.json#mac.label_encoder_ms_per_label_10]] -->
   A fresh set of 10 types costs 619.6 ms before the first text, against 5.3 ms for a 256-token text once they are cached. <!-- @evidence[[results/summary.json#mac.label_set_10_cold_ms]] --> <!-- @evidence[[results/summary.json#mac.gpu_head_256tok_10lab_total_ms]] -->
   (`results/bridge_gliner-web-2.json`)
6. **Retraining GLiNER on stock Ettin-17m fell 17.9 points short.** <!-- @evidence[[results/summary.json#round1.best_arm_gap_to_B1]] -->
   The best arm, A6, scored 36.7 against 54.6 for `gliner_small-v2.1` in the same harness. That model is the bar because the round-1 goal was to come within 10 points of it. <!-- @evidence[[results/summary.json#arms.A6.test_at_05]] --> <!-- @evidence[[results/summary.json#baselines.B1.test_at_05]] -->
   The upstream recipe unchanged (A0) scored 21.1. <!-- @evidence[[results/summary.json#arms.A0.test_at_05]] -->
   Oskar had asked for an existing GLiNER to be packed, so this direction was a misreading and was closed early. Four arms (A0, A3, A4, A6) had reported, two more (A9, A10) stopped on out-of-memory batches and two (A11, A12) were never finished. (Round 1, `results/A0.json` to `results/A6.json`)
7. **Tokenizing each word with its leading space was worth 6.9 points on Ettin-17m.** <!-- @evidence[[results/summary.json#round1.leading_space_gain]] -->
   The upstream library tokenizes words without the leading-space form a BPE tokenizer uses mid-sentence, so "Mountain" became "M" + "ountain". At each arm's dev-chosen threshold the gain was 8.1. <!-- @evidence[[results/summary.json#round1.leading_space_gain_at_dev_best]] -->
   (Round 1, A6 against A4, `results/A6.json` `tokenization_check`)
8. **Packing shrinks the files 4.88 times.** <!-- @evidence[[results/summary.json#sizes.fp32_over_pack_raw]] -->
   With brotli the text encoder is 19.5 MB, the label encoder 14.8 MB and the span head 5.0 MB. <!-- @evidence[[results/summary.json#sizes.shipped_pack_q11.text_encoder_brotli_MB]] --> <!-- @evidence[[results/summary.json#sizes.shipped_pack_q11.label_encoder_brotli_MB]] --> <!-- @evidence[[results/summary.json#sizes.shipped_pack_q11.span_head_brotli_MB]] -->
   The library builds a BiLSTM for this model and never calls it, so the pack leaves out its 887,808 parameters. <!-- @evidence[[web/models/gliner-bi-edge-v2.0-t4-w8L63-h8/manifest.json#dropped.2]] -->
   (`results/pack_cost_biedge.json` `sizes`)
9. **Of the four baselines re-run in this harness, `gliner_small-v2.5` is the strongest.** It scored 58.5 at threshold 0.5, ahead of `gliner_large-v2.1` at 57.5. <!-- @evidence[[results/summary.json#baselines.B2.test_at_05]] --> <!-- @evidence[[results/summary.json#baselines.B4.test_at_05]] -->
   `gliner-bi-edge-v2.0` scored 54.2, close to the 54.0 on its card, which uses threshold 0.4. <!-- @evidence[[results/summary.json#baselines.B3.test_at_05]] -->
   The packed model's 54.28 is therefore below both the 58.5 and the 57.5 baselines, which is the accuracy given up by choosing this model.
   (Round 1, `results/B1_gliner_small-v2.1.json` to `results/B4_gliner_large-v2.1.json`)

## Method

**Benchmark.** The seven test sets come from InstructUIE's `IE_INSTRUCTIONS.zip`: 6,467 sentences. The score is
entity-level exact-match micro F1 per set with greedy flat decoding, as in the GLiNER library's evaluator, and the
unweighted mean of the seven. It is reported at threshold 0.5 and at the threshold chosen on dev, the development
split held apart from the test sentences for choosing settings. Selection of
checkpoints, thresholds and arms used only the five CrossNER dev sets, because the MIT dev files are
byte-identical to their test files. Every baseline was re-run in this harness, since the paper's numbers do not
reproduce with the released weights (GLiNER issue 54). A 60-sentence check matched the library's own evaluator on
`gliner_small-v2.1` and `gliner-bi-edge-v2.0` (commit 08d1dc59).

**Round 1 arms.** Stock `jhu-clsp/ettin-encoder-17m` (7 layers, hidden 256) under the upstream `gliner` library,
uni-encoder, span head (markerV0), trained on `urchade/pile-mistral-v0.1` (19,724 passages, Apache-2.0), seed 0,
on one Colab L4. An arm is one training run. Arms varied learning rate, the BiLSTM (a layer that reads the words in order, in both
directions, between the encoder and the span head), the head type and leading-space tokenization; the table is in the
log and in `run/arms.json`. Of the arms that reported, A0 is the upstream recipe, A3 and A4 drop the BiLSTM at two
learning rates, and A6 is A4 with leading-space tokenization. The span head is the markerV0 type, the library's name
for its span-scoring head.

**Packing.** `run/gpack.py` keeps every weight but stores it in fewer bits, and drops only the unused BiLSTM.
It writes 4-bit token tables, the vocabulary embeddings with one vector per word piece (`t4`), int8 per-column codes
for the encoder layers held to ±63 (`w8L63`) or ±31 (`w8L31`), and optionally int8 span-head and projection matrices
(`h8`). Rounding to fewer bits is what it loses. Activations, the values computed while the model runs, stay in
fp32. `run/gref.py` is a NumPy reference that runs the packed file; its fp32 output is checked against the library
on 56 inputs, and each packed variant is benchmarked on all 8,587 dev and test sentences. One part (text table,
text layers, label table, label layers, heads) was quantized at a time, then all together.

**Browser.** The text encoder runs on the repo's WGSL engine (`web/engine.js`, from `_lib/wgsl/`). The span head
runs in the same GPU submit, or in JavaScript (`web/jshead.js`) as a fallback and as a second arm for parity. The
label encoder (all-MiniLM-L6-v2) and decoding run in JavaScript. Parity is the browser against the NumPy reference
on 12 texts and 42 labels, in headless Chromium on SwiftShader (`run/check_web.mjs`), then on a real GPU through
the GPU Bridge in a Chrome tab on Oskar's Mac (`web/bridge_job.js`, Chrome 157, Apple metal-3). Timings are
medians of 3 repetitions.

**Prior art**, from a survey of 2026-11-10. GLiNER.js (https://github.com/Ingvarstep/GLiNER.js) runs GLiNER through
onnxruntime-web, and the smallest pre-converted model it can load, `onnx-community/gliner_small-v2.1` int8, is 183
MB. kleinhirn (https://github.com/smashburger-dev/kleinhirn) has hand-written WGSL encoders, including ModernBERT,
and lists GLiNER entity extraction as not implemented. Dakera's int8 build of the same bi-encoder
(https://huggingface.co/DakeraAI/gliner-bi-edge-v2.0-int8) is 64.9 MB on onnxruntime's CPU path and also keeps
activations in fp32. The survey did not find, in its searches, a GLiNER on Ettin-17m, a generalist GLiNER at or
under 32M parameters with a benchmark number, GLiNER weights under 25 MB, a hand-written WGSL GLiNER span head, or
a browser GLiNER with published latency. The composition here (a packed bi-encoder GLiNER on hand-written WGSL with
measured accuracy and latency) was not found in those searches. Its components all exist separately.

**Cost and wall-clock.** Round 1 used about 12 Colab L4 compute units, 2.8 hours of it training four arms. Packing
ran on the container's CPU, about 5 minutes per full benchmark pass of the NumPy reference. Headless browser parity
took 26 minutes and the Mac run 94 seconds. From the plan commit to the Mac timings was 20:37 to 04:19 UTC, 7 hours
42 minutes, of which round 1 took the first 4 hours 41 minutes.

**Limits of the page.** A single text is capped at 1,024 tokens, the window of the engine's attention kernel.
The GPU head takes up to 128 entity types; more fall back to the JavaScript head. Accuracy on Bluesky posts, which
the page's second mode reads, is not measured. The in-browser LoRA trainer planned as round 6 was not built.

## Log

### Round 1: retraining GLiNER on stock Ettin-17m (2026-11-10 20:37 to 2026-11-11 01:18 UTC)

Asked: can a GLiNER on Ettin-17m, under 25 MB, come within 10 points of `gliner_small-v2.1`? The plan was a size
and accuracy curve: round 1 for the recipe, round 2 for more data and distillation, round 3 for packing.

Ran: CPU smoke tests of the training and bench drivers, the four baselines, then arms A4, A3, A6 and A0 on a Colab
L4, about 12 compute units. After A3, A4 and A6 reported, the amendment of 00:08 UTC re-based the remaining arms on
leading-space tokenization: A1, A2, A5, A7 and A8 were dropped unrun and replaced by A9 (A6 plus BiLSTM), A10 (A6
with the token-level head), A11 (A6 with dropout 0.1) and A12 (A6 on Ettin-32m). A9 and A10 each stopped on one
out-of-memory batch; tolerance for a few skipped batches was added (commits a493d3fd, c1f1715b), and A11 and A12
were never finished.

Baselines, 7-set test mean:

| | model | params | at 0.5 | at 0.4 |
|---|---|---|---|---|
| B1 | `urchade/gliner_small-v2.1` | 152.6M | 54.6 | 54.7 |
| B2 | `gliner-community/gliner_small-v2.5` | 166.0M | 58.5 | 57.6 |
| B3 | `knowledgator/gliner-bi-edge-v2.0` | 60.8M | 54.2 | 54.0 |
| B4 | `urchade/gliner_large-v2.1` | 445.5M | 57.5 | 57.0 |

Arms, all on 17m, pile-mistral, seed 0:

| arm | change | test at 0.5 | dev best (threshold) | test at dev threshold | best step | train minutes |
|---|---|---|---|---|---|---|
| A0 | upstream recipe: BiLSTM, lr 1e-5 / 5e-5, batch 8 x 30k | 21.1 | 27.4 (0.3) | 23.9 | 30,000 | 37.0 |
| A3 | no BiLSTM, lr 5e-5 / 1e-4, batch 32 x 12k | 25.1 | 35.7 (0.3) | 28.3 | 6,000 | 44.4 |
| A4 | as A3, lr 1.5e-4 / 3e-4 | 29.8 | 36.9 (0.4) | 30.2 | 6,000 | 44.2 |
| A6 | as A4, every word tokenized with a leading space | 36.7 | 46.4 (0.3) | 38.4 | 4,000 | 42.1 |

Training time for the four arms was 2.8 hours of L4; each full bench pass took about 20 seconds.

The upstream tokenization check in `results/A4.json` shows the cause of A6's gain. The library feeds each word to
the tokenizer alone, without the leading space, so `Mountain` becomes `M` + `ountain`, and every word gets the form a BPE
model mostly sees at the start of a text. A6 tokenizes `ĠMountain` as one token.

Predictions:

- P1, baselines at 0.5 (small-v2.1 50 to 56, bi-edge 49 to 55): hit, 54.6 and 54.2.
- P2, A0 between 25 and 38 and the best 17m arm between 38 and 46: miss on both, 21.1 and 36.7. The gap to
  small-v2.1 was 17.9 points against the predicted 8 to 15.
- P3, the BiLSTM costs at most 1.5 points: unmeasured. A1 and A2 were dropped, and A9 stopped on an out-of-memory batch.
- P4, the token-level head within 2 points of the span head: unmeasured. A5 was dropped, and A10 stopped on an out-of-memory batch.
- P5, leading-space tokenization gains 1 to 3 points: miss, 6.9.
- P6, 32m beats the best 17m arm by 4 to 7 points: unmeasured. A8 was dropped and A12 never finished.
- P7, NuNER and distillation: unmeasured. Round 2 never ran.
- P8, packing the 17m model costs at most 1.0 point and fits 12.5 MB: unmeasured on the model it was written for.
  The packer was only checked for logit parity on smoke checkpoints (3.7e-5 from the library). P10 and P11 replaced it.

Reading at the time: with 19,724 training passages a 17M backbone ends about 18 points short, and the published
Ettin-32m GLiNER needed 10M labelled samples to reach 54. Oskar, who had asked for the work, then said he had meant the Ettin shrink-and-pack pattern to be applied to an
existing GLiNER. Round 1 had retrained one from stock Ettin instead, which was a misreading of the request. The plan
dates his message 01:20 UTC and this log starts the packing stage at 01:18 UTC, so the change of direction came about
four and a half hours after the plan commit (20:37 UTC).
Its lasting results are the leading-space effect and the four baselines.

Operations: the restart script (`run/colab_restart.py`, commit e9dc3ed7) brackets its `pkill -f` patterns
(`'run/[t]rain.py'`) so that they do not match the shell that runs them.

### Packing gliner-bi-edge-v2.0 (2026-11-11 01:18 to 02:08 UTC)

Asked: pack the existing bi-encoder (Ettin-32m text encoder, all-MiniLM-L6-v2 label encoder, markerV0 span head)
and measure what each part of the packing costs on the full benchmark.

Ran: `run/gpack.py` and the bi-encoder path of `run/gref.py`. The fp32 reference matched the library on 56 inputs
with identical token ids, label ids and entities; its largest logit difference was 1.5e-3, on a 2,597-token input,
and 1.1e-4 on the other 55. The fp32 pack scored exactly the library's numbers on all 48 benchmark cells (7 test and
5 dev sets at 4 thresholds). One full NumPy bench pass on the CPU took about 5 minutes (`results/gbench_biedge-fp32_verify.json`).

The library builds a BiLSTM (`rnn.lstm.*`, 887,808 parameters) for this checkpoint, and its bi-encoder forward never
calls it in `gliner` 0.2.30 or 0.2.22. Zeroing its weights changes the logits by 0.0, so the pack leaves it out.

Cost of each part, 7-set test mean at 0.5, against fp32 54.23 (`results/pack_cost_biedge.json`):

| quantized part | variant | test at 0.5 | change |
|---|---|---|---|
| text token table | 4-bit | 54.43 | +0.19 |
| text token table | 8-bit | 54.29 | +0.05 |
| text layers | int8, ±63 | 54.08 | −0.15 |
| text layers | int8, ±31 | 53.75 | −0.48 |
| label token table | 4-bit | 54.20 | −0.04 |
| label token table | 8-bit | 54.24 | +0.01 |
| label layers | int8, ±63 | 54.39 | +0.15 |
| label layers | int8, ±127 | 54.15 | −0.08 |
| span head | int8 | 54.22 | −0.01 |
| everything (shipped) | t4, ±63, h8 | 54.28 | +0.04 |
| everything but the label encoder | t4, ±63, h8 | 54.26 | +0.03 |

Individual predictions do change. Over the 56 parity inputs the mean logit shift is 0.36 from the 4-bit text
table alone, against an fp32 logit spread of 4.32, and 19% of entities at 0.5 differ from fp32. The aggregate F1
does not move.

Sizes (brotli quality 11): the shipped pack is 39.9 MB, of which the text encoder is 19.5 MB, the label encoder
14.8 MB and the span head 5.0 MB. Uncompressed it is 49.5 MB, against 241.4 MB in fp32. The files committed under
`web/models/` measure 40.3 MB at brotli quality 9. On a single CPU thread the NumPy reference takes 24 ms for the text
encoder on a 27-token text and 7.7 ms per label for the label encoder.

Predictions:

- P10, the download between 30 and 40 MB (text encoder about 23 MB, label encoder about 12 MB): hit at brotli
  quality 11 (39.9 MB), with the parts off in opposite directions (19.5 and 14.8 MB). The files committed under
  `web/models/` measure 40.3 MB at quality 9, just above the predicted range, and a server that sends them
  uncompressed sends 49.5 MB.
- P11, packing costs at most 1.5 points: hit, +0.04.
- P12, the label encoder contributes less than a third of the loss: not decidable. There was no loss to divide; the
  label-encoder-only arms moved −0.04 and +0.15.

### Browser runtime (2026-11-11 02:08 to 02:58 UTC)

Asked: run the packed model in the browser. The text encoder goes on the WGSL engine with a readback of the word
rows, and everything after it starts in JavaScript, to be moved to the GPU where timing says it pays.

Ran: `web/engine.js` (the text encoder on the shared WGSL kernels, which already took hidden 384), `web/jshead.js`
(label encoder, span head, decoding) and `web/ner.js` (tokenizing into words with Unicode code-point offsets,
batching many short texts into one encoder pass). Two work-in-progress checkpoints were committed around a container
restart (a3dbf2b4, 15d76ded). Parity in headless Chromium on SwiftShader, against the NumPy reference:

| pack | token and label ids | logits, max difference | entity checks |
|---|---|---|---|
| fp32 | 12 of 12 | 2.5e-5 | 48 of 48 |
| t4-w8L63-h8 (shipped) | 12 of 12 | 2.5e-5 | 48 of 48 |

Splitting the 12 texts into 6 batches changed no logit. The headless run took 26 minutes.

P9's parity half (under 1e-3 from PyTorch on the fp32 path): hit when the two links are added, 2.5e-5 from the
browser to the reference plus 1.1e-4 from the reference to the library on inputs the browser accepts. No run
compared the browser to PyTorch directly.

### GPU head and page (2026-11-11 02:58 to 04:13 UTC)

Asked: move the span head to the GPU where it pays, and build the page.

Ran: the span head's per-word layers and the span by label scores now run in the same GPU submit as the encoder,
and only the scores are read back. Headless, the JavaScript head had cost 88 ms per text for 36-word texts, almost
all of it in the per-word layers. The GPU head brought that to 0.02 ms (commit cfab0a94). Both heads still pass
parity on both packs, and the GPU head agrees with the JavaScript head within 3.8e-6. The GPU head takes up to 128
entity types and falls back to the JavaScript head above that.

The page has a text box, entity-type chips with presets, a threshold and highlighted entities, plus a Bluesky mode
that reads an account's posts and groups them by the entities in them. Its smoke test passes 19 checks. It is
listed in `pages.json` as "Browser NER" with the shipped pack (commit 771e53bb).

The first Mac bridge job (gliner-web-1: JavaScript head, 500 texts x 7 repetitions) hit the bridge's 15-minute
timeout with no logs. The second job was cut to 100 texts and 3 repetitions.

### Mac timings (2026-11-11 04:19 UTC)

Ran: `gliner-web-2` in a Chrome 157 tab on Oskar's Mac (Apple metal-3), with the shipped pack, in 94 seconds.
Parity on 12 cases passed for both heads: logits within 2.2e-5 of the reference, 24 of 24 entity checks.

Medians of 3 runs, milliseconds:

| case | head on | encoder (GPU) | span head | total |
|---|---|---|---|---|
| 256 tokens, 10 labels | GPU | 4.5 (includes head) | in the encoder submit | 5.3 |
| 256 tokens, 10 labels | JavaScript | 11.0 | 1,071.7 | 1,083.5 |
| 30 tokens, 10 labels | GPU | 1.7 (includes head) | in the encoder submit | 1.9 |
| 30 tokens, 50 labels | GPU | 1.5 (includes head) | in the encoder submit | 1.7 |
| 30 tokens, 10 labels | JavaScript | 3.8 | 135.5 (126.9 in per-word layers) | 139.9 |

A batch of 100 short texts (4,388 tokens, 20 labels) ran at 2,732 texts a second with batches pipelined and 2,445
run serially with the GPU head. With the JavaScript head it ran at 8.07 texts a second. A new label set costs 62 ms per label for
10 labels and 55 ms per label for 50 (the label encoder in JavaScript, plus a projection), so 10 new labels cost
619.6 ms. Headless on the container's CPU the same label encoder took 21 ms per label; why the Mac tab was
slower was not investigated.

P9's timing half (a 256-token text with 10 labels, tokenize to entities, in at most 15 ms): hit, 5.3 ms, with the
labels already encoded. With 10 new labels the first call costs about 620 ms more.

Reading: the encoder and head on the GPU are fast enough that the label encoder in JavaScript is now the slowest
step a user sees, and only when they change the entity types.
