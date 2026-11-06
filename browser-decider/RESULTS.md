# RESULTS: a Jev-shaped decision model in a 10 MB browser download

A decision model answers typed questions about a piece of state: `choice` (pick one label), `score`
(a level on a rubric) or `noul` (is this statement true), with a probability for every option and no
generated text. TypeSafe's Jev is the hosted example: a large model behind an API, answering any question
schema zero-shot in about a second. The question here was whether one can run in a web page instead, built
on a small model already shipped on austegard.com: the 8-bit Monad (56M parameters, 65 MB) or the 10 MB
Ettin-17m build from `ettin-native`. The demo is live at https://austegard.com/experiments/browser-decider/web/.

The benchmark is Typed Decisions: four synthetic workflows with five questions per case. Its gold answers
are the mean of three samples from an unnamed ~4B-class teacher model, so accuracy means agreement with
that teacher's top option. Its card lists Jev at 0.727 zero-shot, the teacher agreeing with itself at
0.735, and 0.470 for predicting each question's most frequent label without reading the input. Those three
numbers come from the card; every other number here was measured in this experiment.

## Answer

Yes for a fixed set of workflows, no for arbitrary questions. Ettin-17m, fine-tuned as a cross-encoder
on the four workflows of the public Typed Decisions benchmark, agrees with the gold label 0.740 of the time over three seeds. <!-- @evidence[[results/summary.json#spec.stock-e10.accuracy.mean]] -->
That matches Jev and the teacher's self-agreement. The page downloads 9.74 MB. <!-- @evidence[[results/summary.json#page_build.downloadMB]] -->
Since 2026-11-06 it ships a seed-0 checkpoint retrained on Colab, with matmul codes held to ±31 and a 4-bit token table; that file scores 0.745. <!-- @evidence[[results/summary.json#packed_spec-stock-e10q-s0-t4-w8L31.accuracy]] -->
The first build (±63 codes, 10.2 MB) scored 0.730. <!-- @evidence[[results/summary.json#packed_spec-stock-e10-s0-t4-w8L63.accuracy]] -->
It answered a five-question case in 19 to 41 ms on a MacBook's GPU in Chrome, depending on GPU load; the ±31 file has the same shapes and kernels and was not timed again.

It does not generalise. Trained on three workflows and tested on the fourth, it scores 0.353, below the 0.470 label prior. <!-- @evidence[[results/summary.json#lowo.stock.mean_accuracy]] -->
On 104 questions from unrelated schemas it scores 0.279, where Jev scores 0.894. <!-- @evidence[[results/summary.json#ood_spec-stock-e10-s0.summary.accuracy]] -->
It is a specialist: retrain it for each set of workflows, from a few hundred labelled cases.
Monad (56M, causal) scores 0.752 at five epochs, above Ettin, but it would take about 65 MB at 8 bits and has the same failure on unseen workflows (0.333); Ettin ships for its size. <!-- @evidence[[results/summary.json#spec.monad.accuracy.mean]] -->

## Findings

1. **Specialist accuracy matches Jev's zero-shot score at 17M parameters.** Stock Ettin-17m, 10 epochs:
   0.740 ± 0.010 over seeds 0 to 2 (0.7375, 0.7515, 0.732). KL from gold 0.142, Brier 0.079. <!-- @evidence[[results/summary.json#spec.stock-e10.kl_from_gold.mean]] -->
   By type: noul 0.814, choice 0.712, score 0.706 (Jev: 0.775, 0.720, 0.696 on the dataset card).
   Score answers land within one level of gold 0.965 of the time. <!-- @evidence[[results/score_pm1.json#mean]] -->
   nanoGentzen-Jev-v3's own card, on the same 2,000 test decisions: 0.731 overall, noul 0.828, choice 0.685,
   score 0.693, score ±1 0.973, KL 0.150, Brier 0.081, ECE 0.087. Ettin is ahead on overall, choice, score,
   KL and Brier, behind on noul, score ±1 and ECE. (`results/score_pm1.json`)
   On the card's table of models fitted on `train`, this sits between SafeDecide-140M (0.754) and
   ProKope-421M (0.7185), and far above the other 17M entry, Bekko 17M (0.483). (`results/spec-stock-e10-s*.json`)
2. **The browser build runs the model exactly and fast.** The hand-written WGSL engine matches the
   NumPy reference of the packed file to 7e-6 on every logit, with the same top option on 40 of 40
   questions, on an Apple GPU (Chrome 156), in two runs on two builds. A case of five questions (about
   1,700 tokens) took 19.3 ms median and 25.5 ms p90 in the first run, one question 5.2 ms. <!-- @evidence[[results/bridge_decider-final-1.json#caseMsMedian]] -->
   The second run, on the shipped weights and the same shapes, took 41.2 ms median and 144 ms p90 while the Mac's GPU was
   busy with other work (41.2 ms; the whole job ran 6x longer). <!-- @evidence[[results/bridge_decider-final-2.json#caseMsMedian]] -->
   Start-up is about 0.2 s once the files are cached.
   (`results/bridge_decider-final-*.json`)
3. **Quantization costs 0.0075 at 10 epochs and 0.001 at 5.** Weight-only int8 matmuls (codes held to
   ±63) and a 4-bit token table: 0.7375 fp32 against 0.730 packed for the shipped seed; 0.7145 against
   0.7135 for the five-epoch seed 0. (`results/packed_*.json`)
4. **No transfer to an unseen workflow.** Leave-one-workflow-out, three seeds each: stock init 0.353,
   the 10 MB build's init 0.378, both below the 0.470 prior. <!-- @evidence[[results/summary.json#lowo.e17c.mean_accuracy]] -->
   The 10-epoch specialists, which trained on every workflow, score 0.70 to 0.77 on each one.
   (`results/lowo-*.json`)
5. **Longer training buys accuracy and costs breadth.** Ten epochs instead of five: +0.026 on the
   benchmark, and 0.394 to 0.279 on the out-of-schema set. <!-- @evidence[[results/summary.json#ood_spec-stock-s0.summary.accuracy]] -->
   At ten epochs: 64 banking intent questions with 8 options each 0.19, yes/no predicates 0.50 on a 50/50 set, rubric scores 0.31.
   (`results/ood_*.json`)
6. **Retrieval tuning hurts a cross-encoder start.** At five epochs the 10 MB build's checkpoint
   reaches 0.636 ± 0.008 against stock Ettin-17m's 0.714 ± 0.005. <!-- @evidence[[results/summary.json#spec.e17c.accuracy.mean]] -->
   It loses most on score questions (0.545 against 0.679).
   Its contrastive tuning for mean-pooled sentence vectors appears to have worn down the per-token
   features the option positions read.
7. **Monad buys about 0.01 to 0.04 of accuracy for about six times the bytes.** The Monad decider was fine-tuned
   from Pleias' original weights and scored at full precision; it was not quantized or built for the browser,
   and 65 MB is the size of the chat page's 8-bit Monad, used here as the estimate. Same layout reordered for a
   causal model (state first, each option scored at its line's last token), five epochs, three seeds:
   0.752 ± 0.009, KL 0.122, Brier 0.066; noul 0.832, choice 0.702, score 0.728. Against Ettin at five
   epochs (0.714) that is +0.037; against Ettin at ten (0.740), +0.011. Left out of one workflow
   (seed 0 only) it scores 0.333, the same failure as Ettin. Monad was not run at ten epochs.
   (`results/spec-monad-s*.json`, `results/lowo-monad-*.json`)
8. **Calibration is the weak metric.** ECE 0.144 on the fp32 runs and 0.156 packed, after a per-type
   temperature fitted to KL on validation. The fitted entries on the card sit at 0.12 to 0.18. Fitting
   the temperature to KL against soft gold labels does not optimise argmax-confidence calibration.
9. **The token table is the size, and it cannot ship partial.** It holds 6.39 MB of the roughly 9.8 MB
   compressed download. <!-- @evidence[[results/vocab_fallback.json#full_table_brotli]] -->
   A core of common rows with exact fallback (keep the full tokenizer, read missing rows from the shipped
   file by HTTP range) keeps answers identical, but on 626 held-out Bluesky accounts even a 16K-row core
   (2.03 MB) leaves 99.8% of accounts needing fetched rows, median 27. Every row seen in 675 other accounts
   (26K rows, 3.34 MB) still leaves 99.4%, median 11; the benchmark's test cases need one 29% to 42% of the time.
   A fallback page would make network requests on almost every answer. Lossy pruning was ruled out earlier
   (`ettin-native` finding 5), as were table bits below 4 (finding 4). (`results/vocab_fallback.json`)
10. **No layer or head is free to drop, and one head carries the task.** On 400 test questions (80
   agent-trace cases), removing the last layer changes 5% of top answers and layer 5 13.5%; layers 0, 2
   and 3 are essential (61% to 75% of answers change). Head 1 of layer 3, a global-attention layer, alone
   changes 70% of answers; seven of the other 27 heads change 2% or fewer. Dropping a whole layer saves about 0.5 MB
   raw, a twentieth of the download, before any retraining to recover. Channels 11, 166 and 29 of the
   residual stream reach about 1,100 in every layer against a median channel peak of 6 to 10; activations
   stay in 32-bit floats here, so they cost the weight-only int8 nothing. The int8 columns are near-Gaussian
   (peak about 3.1 to 4.1 times the column RMS). (`results/weight_probe.json`)
11. **Codes held to ±31 save 0.50 MB for about 0.002.** Three fresh 10-epoch checkpoints (seeds 0 to 2,
   retrained on Colab), each packed at ±63 and at ±31 from its own fp32 weights and scored from the packed
   bytes: ±63 averages 0.7343, ±31 0.7322 (per seed +0.001, −0.004, −0.0035), with KL from gold 0.150
   for both. <!-- @evidence[[results/quant_levels-summary.json#delta_L31_minus_L63]] -->
   The weights file goes from 9.76 to 9.26 MB brotli, the matmul codes from 3.05 to 2.55 MB.
   The retrained fp32 checkpoints score 0.7315 against the first round's 0.740 with the same seeds and
   settings (GPU nondeterminism), a larger spread than the ±31 cost. Halving the shipped codes instead of
   requantizing gave an upper bound of −0.0085 (`results/codes31.json`).
   (`results/quant_levels-*.json`, `run/quant_levels.py`)

## Method

**Fixture.** `LocalLLaMA/typed-decisions` (apache-2.0, `data/td/`): four synthetic workflows (agent
trace review, customer service, invoice processing, security alerts), 300 train and 100 test cases
each, five typed questions per case, every option with a written description. Gold is the mean of three
teacher samples. Scoring is the dataset's own `score_one` (accuracy = argmax match, KL from gold, Brier,
10-bin ECE over 2,000 test decisions), copied into `run/common.py`. Out-of-schema check:
`data/decider_eval.jsonl`, copied from the hub's `tests/data/` (64 banking77 card intents, 24 agent
guardrail predicates, 16 rubric scores), where Jev scores 0.894 and a Haiku decider 0.846.

**Terms.** `noul` is the benchmark's name for a yes/no question. A cross-encoder reads the question, the
options and the state together in one pass, as opposed to embedding each separately. KL from gold measures
how far the predicted distribution is from the gold one (0 is identical); Brier is the squared error of the
probabilities; ECE (expected calibration error) is the gap between stated confidence and accuracy. `e17c`
is the 10 MB embedder build's checkpoint. The GPU Bridge is a page on Oskar's Mac that runs WebGPU jobs a
session sends it; SwiftShader is Chromium's CPU stand-in for a GPU.

**Model.** One sequence per question:
`[CLS] type: instructions [SEP] [MASK] option: description [MASK] option: description … [SEP] state [SEP]`,
state as the JSON string. A two-layer MLP scores the final hidden state at each `[MASK]`, softmax
over the question's options, soft cross-entropy against the gold distribution. AdamW, lr 8e-5
(head 8e-4), batch 16, warm-up 6% then linear decay, best epoch by validation KL on 10% of train
cases, one temperature per question type fitted on that slice. Monad (causal) reads state, then
question, then one line per option, scored at each line's last token. `run/train_enc.py`.

**Arms.** Inits: stock `jhu-clsp/ettin-encoder-17m`, the 10 MB build's checkpoint (`e17c`, release
`ettin-native-20261021`) and `PleIAs/Monad` (Pleias' original weights, not the loop-edited build on the chat page,
whose edit removes a direction that drives repetition during generation; nothing here generates text). Modes: specialist (all four workflows) and leave one
workflow out. Seeds 0 to 2 everywhere, 5 epochs, plus a 10-epoch round for the stock specialist, added
after the 5-epoch round because every seed's best validation epoch was its last (test scores were in hand
by then, so read the +0.026 as a second look, not a pre-registered arm). The
shipped seed (0) was fixed before results; seeds draw different validation splits, so validation
cannot rank them.

**Browser build.** `run/pack.py` writes one weights file and a manifest in ettin-native's layout
(int8 per-column matmuls with codes in ±63, 4-bit token table, f32 head), plus `Ref`, a NumPy forward
pass from the file alone (matches PyTorch fp32 to 1.6e-6). `web/engine.js` is ettin-native's WGSL
engine extended to 1,024-token sequences (an attention kernel that loops keys beyond one workgroup) and
a gather of the option positions; the head runs in JS. `web/decider.js` builds the same token layout
(the JS tokenizer reproduced the Python ids on 40 of 40 questions) and returns System One shaped
answers. Timing ran on Oskar's Mac through the GPU Bridge (`web/bridge_job.js`); SwiftShader in the
container is used only to check the page renders.

**Compute.** Colab L4: 1.5 min per Ettin run at five epochs, 4.4 at ten, about 18 min per Monad run; the
Monad runs that were kept ran on an A100 at about 10 min each. About 3 compute units in all.

## Log

### 2026-11-05: Ettin arms, browser engine, demo

Registered `PLAN.md` first. Predictions against results:

| prediction | result |
|---|---|
| P1 specialist 0.72 (0.68 to 0.76) | 0.714 at 5 epochs, 0.740 at 10: held |
| P2 stock init beats `e17c` by ≥ 0.02 | +0.078: held |
| P3 leave-one-out ≤ 0.55 | 0.353 (stock), 0.378 (e17c): held, and below the prior |
| P4 Monad within 0.03 of Ettin | 0.752 against 0.714 at five epochs, +0.037: missed (+0.011 against Ettin at ten epochs) |
| P5 out-of-schema ≤ 0.60 | 0.394 (5 ep), 0.279 (10 ep): held |
| P6 ≤ 30 ms per case, download within 0.5 MB of 10.2 | 19.3 ms quiet, 41.2 ms with the GPU shared; 10.24 MB: held on a quiet GPU |

What broke: the first engine run read garbage because the gather kernel never read the `dn` uniform,
so the auto bind-group layout dropped binding 1 and every dispatch was skipped; the SwiftShader check
caught it (6 ms "timings" for 1,700 tokens were the tell). The first CPU training run died in a
container restart and moved to Colab. Running Monad beside the Ettin arms on one L4 put nine stock
leave-one-out runs out of GPU memory; they were rerun on a second VM. The Colab runner had reported
those crashes as exit 0 because the trainer was piped through `grep`; it now runs under `pipefail`.

The first Monad runs were lost twice: the specialist seeds crashed at start because only the `stock` init
fetched its Hub files (fixed for any Hub id), and the rerun's L4 VM was reclaimed by Colab mid-run, most
likely as idle, since the runner was detached and the notebook kernel did nothing. The kept Monad runs
ran on an A100 with a keepalive loop holding the kernel. An L4 seed-0 run seen in the log before the VM
vanished scored 0.7475; the kept A100 seed 0 scored 0.755 (GPU nondeterminism).

### 2026-11-05: score ±1 and the nanoGentzen-Jev-v3 card

Oskar asked for ours to be run on the benchmark's 400 test cases and 2,000 decisions. It already had been:
every spec number here uses the dataset's own scorer on exactly that split. The one column missing against
nanoGentzen-Jev-v3's card was score ±1 (rubric answer within one level of gold), which the dataset scorer does
not compute. `run/score_pm1.py` takes it from the saved predictions of the three 10-epoch stock seeds:
0.9563, 0.9663, 0.9712, mean 0.965, against the card's 0.973. No model was retrained. That card reports 0.731
overall, below the 0.7585 the leaderboard lists for nanoGentzen-v3, a separate repo.

### 2026-11-05: weight probe, token-table fallback, ±31 codes

Oskar asked whether Ettin's weights hold improvements like the loop edit found in Monad (a generation
fix, so not applicable to an encoder that writes nothing). Quantization levels and lossy vocabulary
pruning were already measured in `ettin-native` (findings 4 and 5 there) and were not rerun. Three new
checks ran on the shipped file through the NumPy reference, on 4 vCPUs: `run/weight_probe.py`
(reproduced 0.730 on all 2,000 decisions in 11 min, then 49 ablations on 400 questions in about an hour),
`run/vocab_fallback.py` (two Bluesky pools collected 3 min apart, 675 and 626 accounts, disjoint, the
second used only for measuring) and `run/codes31.py`. GitHub Pages serves single byte ranges (206) but
answers a multi-range request with the whole 11.5 MB file (200), so a fallback loader must fetch one
range per row; a 144-byte range took 0.12 to 0.34 s from the container through its proxy. The ablation
subset is the first 80 agent-trace cases only, where the file scores 0.7925, so the per-layer numbers
describe one workflow.

### 2026-11-06: fresh ±31 build on Colab

Oskar asked for the ±31 build from fp32 rather than from the shipped codes. The first round's checkpoints
were gone with their Colab VM, so `run/colab_arms.py q31` retrained the stock 10-epoch specialist for
seeds 0 to 2 (`spec-stock-e10q-s*`, Colab L4, about 25 min with scoring, under 1 compute unit) and
`run/quant_levels.py` packed each checkpoint at fp32, ±63 and ±31 with `pack.pack()`, loaded the
dequantized weights from those bytes into PyTorch and scored all 2,000 test decisions on the GPU
(logits within 5e-6 of `Ref` on six questions per variant).

| variant | accuracy s0 / s1 / s2 | mean | KL | weights file, brotli | top option same as fp32 |
|---|---|---|---|---|---|
| fp32 | 0.7335 / 0.743 / 0.718 | 0.7315 | 0.146 | 61.9 MB | — |
| ±63 (shipped format) | 0.744 / 0.741 / 0.718 | 0.7343 | 0.150 | 9.76 MB | 0.931 / 0.947 / 0.941 |
| ±31 | 0.745 / 0.737 / 0.7145 | 0.7322 | 0.150 | 9.26 MB | 0.932 / 0.938 / 0.935 |

Reading: ±31 costs about 0.002 against ±63 from the same weights, inside the seed-to-seed and
rerun-to-rerun spread. Oskar then asked for the page to ship it: `run/pack.py spec-stock-e10q-s0 --variant
t4-w8L31 --score` (NumPy reference, 0.745, same as the GPU score) and `run/build_page.py` replaced
`web/models/decider/`. Seed 0 is the seed fixed as shipped before the first round's results. The WGSL
engine matches the reference on the new file in headless Chromium (max logit difference 4.3e-6, 40 of 40
top options, 40 of 40 tokenizations; `results/check_engine_spec-stock-e10q-s0-t4-w8L31.json`), and the
page loads it and answers a test case. Download 9.74 MB, from 10.24.

