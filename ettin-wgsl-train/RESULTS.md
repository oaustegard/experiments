# RESULTS: training a LoRA on Ettin-17m in the browser, in WGSL

`bsky-account-decider` is a page that reads a Bluesky account and says, among other things, which of its posts
are about AI. A 10 MB Ettin-17m model answers on the visitor's GPU, but the small adapter (a LoRA) that taught
it to read a reply together with its thread was trained in PyTorch on a CPU. This experiment is a trainer for
that model written as WebGPU compute shaders (WGSL): forward pass, backward pass and optimizer. With it, a page
could fine-tune the model on the GPU it already uses, on data that never leaves the browser. The base weights
are the page's own compressed file (int8 matrices, a 4-bit token table), expanded to 32-bit floats on the GPU
exactly as inference reads them, so the adapter learns against the model the page runs. The code is `web/train.js`.

Two test environments appear below: SwiftShader, a software GPU in headless Chromium in the container
(correct, very slow), and the GPU Bridge, a page open on Oskar's Mac that runs WebGPU jobs sent from this
session and commits the results back.

## Answer

Can hand-written WGSL train a LoRA on the packed Ettin-17m page model, and does it learn the same thing PyTorch
does? Yes. On Oskar's Mac (Apple GPU, Chrome) it ran bsky-account-decider's training job (6,194 posts labelled
"about AI" or not by Sonnet, 4 epochs of batches of 16, 1,552 steps) in 65 seconds, including evaluation after each epoch. <!-- @evidence[[results/trajectory.json#timing.gpu_train_wall_s]] -->
Both runs started from the same randomly initialized adapter, saved from PyTorch, and saw the same batches.
The training loss of each step differed from PyTorch's by 4.6e-7 relative, on average. <!-- @evidence[[results/trajectory.json#loss_rel_diff.all_mean]] -->
After training, each held-out post's probability of "about AI" differed by at most 6.7e-6, and the yes/no answer
(probability above 0.5) agreed on 2,096 of 2,096 held-out posts (validation, test and a separate reply set). <!-- @evidence[[results/trajectory.json#answers_agree]] -->
For practical purposes it is the same model. Against a float64 reference, its gradients are closer to exact than
PyTorch's own float32 ones. A training step on a typical 16-post batch took 51.7 ms on the Mac against 315.8 ms
in PyTorch on the container's 4 CPUs, 6.1 times faster. <!-- @evidence[[results/trajectory.json#timing.step_speedup]] -->
One run, one seed. Untested: weaker GPUs, the iPhone in particular, where the longest batch's 550 MB of working
memory may not fit.

## Findings

1. **The WGSL run reproduces the PyTorch run.** With the same initial adapter, batch order and schedule, the
   per-step loss differs by 4.6e-7 relative on average and 5.6e-6 at most over 1,552 steps, from 3.2e-7 in <!-- @evidence[[results/trajectory.json#loss_rel_diff.all_mean]] -->
   epoch 1 to 4.8e-7 in epoch 4. Final probabilities differ by at most 6.7e-6 on the 1,154 test posts, and test <!-- @evidence[[results/trajectory.json#final_probs.test.max_abs_prob_diff]] -->
   accuracy is 0.9437 for both (recall of "about AI" 0.8956, false positives 3.4% of the other posts, also
   equal). Accuracy, recall and false-positive rate are the same to four places after every epoch on all three
   held-out sets. (`results/trajectory.json`)
2. **Its gradients are closer to exact than PyTorch's float32 ones.** On one batch of 8 sequences (90 to 627
   tokens, so the local attention window and the padding both matter), with every LoRA B set to random values so
   that every tensor gets a gradient: against PyTorch in float64, the Mac's 60 nonzero-gradient tensors are all within 5.3e-6 relative <!-- @evidence[[results/gradcheck_f64.json#max_gpu_vs_64]] -->
   L2 error, and PyTorch's own float32 run within 1.2e-4. The Mac's loss is within 1.6e-7 of float64, PyTorch's <!-- @evidence[[results/gradcheck_f64.json#max_torch32_vs_64]] -->
   float32 loss 9.2e-6. Measured against PyTorch float32, the WGSL numbers therefore show PyTorch's rounding: <!-- @evidence[[results/gradcheck_f64.json#torch32_loss_rel_err_vs64]] -->
   gradients within 1.2e-4 and the AdamW update within 5.1e-3, with 7 of 245,504 elements moved the other way. <!-- @evidence[[results/gradcheck_metal.json#maxGradRelErr]] -->
   Every element whose gradient sign differs between the two has a gradient under 2.1e-7 (the median is 3.2e-4). SwiftShader gives the same numbers as the Mac to two digits.
   (`results/gradcheck_f64.json`, `results/gradcheck_metal.json`, `results/gradcheck_swiftshader.json`)
3. **It is 6.1 times faster than the CPU per step, and the run 7.4 times faster.** One forward and backward pass on a <!-- @evidence[[results/trajectory.json#timing.step_speedup]] -->
   16 x 222 batch: 51.7 ms on the Mac, 315.8 ms in PyTorch on the 4-CPU container. The whole run, with evaluation <!-- @evidence[[results/trajectory.json#timing.cpu_step_ms]] -->
   on 2,096 held-out posts after each epoch: 65 seconds, against 478 seconds for the seeded PyTorch run with the <!-- @evidence[[results/trajectory.json#timing.cpu_train_wall_s]] -->
   same evaluation. Training takes 14.3 to 15.0 seconds per epoch, about 38 ms a step: batches are sorted by
   length, and the timed 16 x 222 batch is longer than most. (`results/trajectory.json`, `results/cpu_step_time.json`)
4. **The LoRA recipes had a fourth target that nobody named.** bsky-account-decider's `thread_lora.py` and
   adapter-library's `lora.py` add a LoRA to every linear layer whose name is `Wqkv`, `Wo` or `Wi`. ModernBERT names
   the MLP's output layer `mlp.Wo`, so it gets one too: 245,761 trainable parameters, not 209,921. The first
   round of this experiment trained three targets against a reference that also had three; the trainer now
   takes all four by default (`targets` option).
5. **On adapter-library's growth protocol the browser pipeline scores higher than PyTorch's, by about 0.02.**
   With 10 seeds a side, paired by seed (same labelled invoices), the creator page training through this
   trainer averages 0.727 at 64 labels and 0.837 at 300, against 0.703 and 0.816 for adapter-library's PyTorch <!-- @evidence[[results/growth_seeds.json#by_n.64.wgsl_mean]] -->
   arms: +0.024 (95% interval 0.003 to 0.045) and +0.021 (0.009 to 0.033), ahead on 9 of 10 seeds at each size. <!-- @evidence[[results/growth_seeds.json#by_n.300.paired_diff_mean]] -->
   The two-seed gap at 64 that prompted this (PyTorch 0.748) came from PyTorch seeds that do not reproduce:
   with the round-1 library adapters rebuilt, seeds 0 and 1 score 0.608 and 0.680. The browser pipeline
   differs in its start (the merge of the shipped, quantization-tuned spikes), its base (packed) and its
   training loop, and this round does not separate them. The trainer is not the cause: it is exact (findings 1
   and 2). (`results/growth_seeds.json`)
6. **Two head biases have a true gradient of zero, in PyTorch as well.** The last linear layer's bias and the
   head LayerNorm's bias add the same constant to every option's logit, and the segment softmax cancels it.
   Their reference gradients (about 1e-8) are float noise, so a gradient check has to leave them out
   of relative errors. AdamW still moves them by about the learning rate each step, in both implementations,
   because it divides a gradient by its own size.
7. **The one real-GPU failure was in JavaScript.** The first Mac run gave a finite first loss and NaN after it:
   `web/traj.js` read the schedule's step count from a key the fixture did not have, so the learning rate was
   NaN. The kernels were already right on SwiftShader, and the Mac matched them.

## Method

**Model.** bsky-account-decider's page model (`web/models/account/`): Ettin-17m fine-tuned as a cross-encoder
decision model, 7 layers, hidden 256, 4 heads of 64, GeGLU 384, sliding-window attention (128) on layers 1, 2,
4 and 5, RoPE, LayerNorm without bias; int8 matmuls with a scale per output column and a 4-bit token table.
Trainable: rank-8 LoRA (alpha 16) on every Wqkv, attention Wo, Wi and MLP Wo, and the decision head (Linear,
GELU, LayerNorm, Linear), 245,761 parameters in one flat vector. Loss: cross-entropy of the gold option
probabilities against the softmax of the logits over each question's options, divided by the calibration
temperature. AdamW (lr 5e-4, head 8e-4, weight decay 0.01), global-norm clipping at 1.0, 6% linear warmup then
linear decay, batch 16 sorted by length, 4 epochs.

**Trainer** (`web/train.js`). Frozen weights are dequantized to f32 once on upload, and activations are f32.
The forward pass keeps only each layer's input, and the backward pass recomputes a layer before differentiating
it. Attention backward runs in two passes without atomics: one per query row (dQ, and D = dO·O), one per key
row (dK, dV). Both recompute the probabilities from the saved log-sum-exp. LoRA A and B gradients sum over every
token, through a split-K matmul and a reduction. One tiled matmul kernel (64 x 64 tiles, arbitrary strides)
serves every product, forward and backward. Gradient clipping and AdamW also run on the GPU, so a step uploads
only the batch. About 38 KB of buffers per token: 550 MB for the longest batch (16 x 902).

**Checks.** `run/reference.py` rebuilds the model in PyTorch from the packed weights (logits match the NumPy
reference to 3.6e-6, `bsky-account-decider/run/thread_lora.py`) and writes the fixtures. The gradcheck
(`web/gradcheck.js`) compares one batch's loss, gradients and AdamW update with PyTorch's, and `reference.py f64`
recomputes that batch in float64. It runs in headless Chromium on SwiftShader (`run/headless.mjs`) and on a real
GPU through the GPU Bridge (`web/bridge_gradcheck.js`). The trajectory run (`web/traj.js`, `web/bridge_job.js`)
repeats a seeded PyTorch run step for step, and `run/compare.py` compares the two. Data: bsky-account-decider's
thread-aware "about AI" labels (6,194 train, 566 val, 1,154 test, split by account) and its 376-reply
whole-thread set.

**Prior art** (three web searches). webgpu-llm trains small decoders from scratch with hand-written WGSL and
AdamW. rullama has WGSL LoRA backward kernels over a frozen Q4_K base. ferrotensor has a two-pass,
atomics-free attention backward in WGSL. I did not find LoRA training of a pretrained, quantized
ModernBERT-style encoder in the browser, checked step for step against PyTorch.

## Log

### 2026-11-05: trainer, gradcheck, full run on the Mac

Oskar, after bsky-account-decider's LoRA trained on the CPU: "build the WGSL backward pass as its own
experiment". `PLAN.md`, with the predictions below, was committed before any kernel ran.

| prediction | result |
|---|---|
| P1 gradcheck: loss within 1e-5, every gradient tensor within 1e-4, parameters after one AdamW step within 1e-5 | loss 1.8e-7 and gradients 3.9e-6: met. The parameters themselves were not compared; the update they took (new minus old) was, a stricter test because it removes the shared starting values: 1.5e-4 relative, no element moved the wrong way |
| P2 trajectory: per-step loss within 1% over epoch 1; final test accuracy within 0.005 | 2.8e-7 over epoch 1; test accuracy 0.9445 on both |
| P3 Mac: a batch-16 step under 150 ms, 4 epochs under 5 minutes | 51.5 ms; 62 s with evaluation |

Three LoRA targets (Wqkv, Wo, Wi) in this round; its results are the `results/*_3targets.json` files.

What ran: the gradcheck on SwiftShader (202 s for the one step) and on the Mac (74 ms); two smoke jobs and the
full run through the GPU Bridge (`ewt-smoke-1`, `ewt-smoke-2`, `ewt-traj-1`, Mac tab, page v10); the seeded
PyTorch run. Labels came from bsky-account-decider ($7.17, already spent); this experiment added no cost.

What broke: a uniform struct named `P` clashed with a storage binding named `P` in two kernels (renamed `Uni`).
The first Mac run's learning rate was NaN, from the key name in Finding 5. The seeded PyTorch run shared the
container's 4 CPUs with a SwiftShader job for its first 300 steps and took 1,307 s, so its time is not used. The
CPU numbers came from an uncontended timing of the same batch and from bsky-account-decider's unseeded run of the
same recipe (452 s).

### 2026-11-06: the fourth LoRA target, Wo_mlp

The decider-lora-creator session, which trains through this trainer in its page, reported that every
adapter-library spike carries a LoRA on the MLP's output matrix, which this trainer lacked, so its library
starts lost that part. Oskar: "Yes". Finding 4 is why the recipes had it: their target names match `mlp.Wo` as
well as `attn.Wo`. Added `Wo_mlp` as a fourth target (the default), regenerated the reference with all four
(245,761 parameters, the count bsky-account-decider's run reports), and reran everything.

| check | three targets (round 1) | four targets |
|---|---|---|
| gradcheck vs PyTorch float32, loss / worst tensor | 1.8e-7 / 3.9e-6 | 9.1e-6 / 1.2e-4 |
| gradcheck vs PyTorch float64, loss / worst tensor | not run | 1.6e-7 / 5.3e-6 (PyTorch float32: 9.2e-6 / 1.2e-4) |
| trajectory, per-step loss, mean relative | 5.6e-7 | 4.6e-7 |
| final probabilities, largest difference | 6.2e-6 | 6.7e-6 |
| answers agreeing | 2,096 / 2,096 | 2,096 / 2,096 |
| Mac run / CPU run | 62 s / 452 s (other run) | 65 s / 478 s (this run, uncontended) |

The gradcheck against PyTorch float32 got 50 times looser with the fourth target. With random B on every
target, the perturbed network rounds worse in PyTorch's CPU float32, while the WGSL numbers stayed within
5.3e-6 of float64 (`results/gradcheck_f64.json`). The trajectory, where B starts at zero as in training,
matches as closely as before. Bridge jobs `ewt-gradcheck-2` (Mac), `ewt-gradcheck-3` (Mac, raw gradients
out), `ewt-traj-2` (Mac); the SwiftShader gradcheck took 270 s.

### 2026-11-06: the growth-protocol gap with 10 seeds a side

decider-lora-creator's two-seed comparison on adapter-library's growth protocol (invoices held out, the
library merge as the start, 100 test cases, 500 decisions) had the page's LoRA 0.055 behind PyTorch at 64
labels and 0.020 ahead at 300. Oskar: "run more seeds to settle the 64 and 300 gaps". Predictions in
`PLAN.md` (round 3), committed before any arm reported. WGSL: the creator's own `bridge_lora.js`, unchanged,
seeds 0 to 9 (`ewt-growth-n64`, 3.8 min; `ewt-growth-n300`, 16.8 min; Mac). PyTorch: `run/growth_python.py` on a
Colab L4 (`run/colab_growth.py`), lora.py's grow branch for seeds 0 to 9, after rebuilding the three round-1
library spikes, whose saved adapters were gone. `run/growth_compare.py` pairs the arms by seed.

| labels | WGSL mean (sd) | PyTorch mean (sd) | paired difference | 95% interval | WGSL ahead |
|---|---|---|---|---|---|
| 64 | 0.727 (0.027) | 0.703 (0.048) | +0.024 | 0.003 to 0.045 | 9 of 10 |
| 300 | 0.837 (0.016) | 0.816 (0.012) | +0.021 | 0.009 to 0.033 | 9 of 10 |

| prediction | result |
|---|---|
| P4 both intervals include 0 | missed: both exclude it, with WGSL ahead |
| P5 the two published seeds reproduce within 0.01 on each side | WGSL: all four exactly. PyTorch: no, 0.746 / 0.750 at 64 became 0.608 / 0.680, and 0.802 / 0.824 at 300 became 0.824 / 0.796, from the rebuilt start |

Reading: the gap is real, smaller than the two-seed numbers suggested, and the other way round at 64. Since the
trainer is exact, it is a setup effect. The likeliest cause is the start: the shipped spikes the page merges
are the v0.1.1 ones, tuned on the quantized base, while lora.py merges the round-1 fp32 spikes. The next arm
would be PyTorch on the page's start and base, which leaves only the training loop different. Colab cost:
about 15 minutes of an L4.

