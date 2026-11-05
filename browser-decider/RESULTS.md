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
That matches Jev and the teacher's self-agreement. The page downloads 10.2 MB. <!-- @evidence[[results/summary.json#page_build.downloadMB]] -->
The shipped int8/int4 build scores 0.730 and answers a five-question case in 19 to 41 ms on a MacBook's GPU in Chrome, depending on GPU load. <!-- @evidence[[results/summary.json#packed_spec-stock-e10-s0-t4-w8L63.accuracy]] -->

It does not generalise. Trained on three workflows and tested on the fourth, it scores 0.353, below the 0.470 label prior. <!-- @evidence[[results/summary.json#lowo.stock.mean_accuracy]] -->
On 104 questions from unrelated schemas it scores 0.279, where Jev scores 0.894. <!-- @evidence[[results/summary.json#ood_spec-stock-e10-s0.summary.accuracy]] -->
It is a specialist: retrain it for each set of workflows, from a few hundred labelled cases.
Ettin is the shipped model because it is 10 MB against Monad's 65 MB; Monad's own result is pending.

## Findings

1. **Specialist accuracy matches Jev's zero-shot score at 17M parameters.** Stock Ettin-17m, 10 epochs:
   0.740 ± 0.010 over seeds 0 to 2 (0.7375, 0.7515, 0.732). KL from gold 0.142, Brier 0.079. <!-- @evidence[[results/summary.json#spec.stock-e10.kl_from_gold.mean]] -->
   By type: noul 0.814, choice 0.712, score 0.706 (Jev: 0.775, 0.720, 0.696 on the dataset card).
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
7. **Calibration is the weak metric.** ECE 0.144 on the fp32 runs and 0.156 packed, after a per-type
   temperature fitted to KL on validation. The fitted entries on the card sit at 0.12 to 0.18. Fitting
   the temperature to KL against soft gold labels does not optimise argmax-confidence calibration.

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
`ettin-native-20261021`) and `PleIAs/Monad`. Modes: specialist (all four workflows) and leave one
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

**Compute.** Colab L4: 1.5 min per Ettin run at five epochs, 4.4 at ten, about 18 min per Monad run.

## Log

### 2026-11-05: Ettin arms, browser engine, demo

Registered `PLAN.md` first. Predictions against results:

| prediction | result |
|---|---|
| P1 specialist 0.72 (0.68 to 0.76) | 0.714 at 5 epochs, 0.740 at 10: held |
| P2 stock init beats `e17c` by ≥ 0.02 | +0.078: held |
| P3 leave-one-out ≤ 0.55 | 0.353 (stock), 0.378 (e17c): held, and below the prior |
| P4 Monad within 0.03 of Ettin | seed 0: 0.7475 against 0.7145, +0.033: missed narrowly (other seeds pending) |
| P5 out-of-schema ≤ 0.60 | 0.394 (5 ep), 0.279 (10 ep): held |
| P6 ≤ 30 ms per case, download within 0.5 MB of 10.2 | 19.3 ms quiet, 41.2 ms with the GPU shared; 10.24 MB: held on a quiet GPU |

What broke: the first engine run read garbage because the gather kernel never read the `dn` uniform,
so the auto bind-group layout dropped binding 1 and every dispatch was skipped; the SwiftShader check
caught it (6 ms "timings" for 1,700 tokens were the tell). The first CPU training run died in a
container restart and moved to Colab. Running Monad beside the Ettin arms on one L4 put nine stock
leave-one-out runs out of GPU memory; they were rerun on a second VM. The Colab runner had reported
those crashes as exit 0 because the trainer was piped through `grep`; it now runs under `pipefail`.
