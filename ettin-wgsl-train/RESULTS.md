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
"about AI" or not by Sonnet, 4 epochs of batches of 16, 1,552 steps) in 62 seconds, including evaluation after each epoch. <!-- @evidence[[results/trajectory.json#timing.gpu_train_wall_s]] -->
Both runs started from the same randomly initialized adapter, saved from PyTorch, and saw the same batches.
The training loss of each step differed from PyTorch's by 5.6e-7 relative, on average. <!-- @evidence[[results/trajectory.json#loss_rel_diff.all_mean]] -->
After training, each held-out post's probability of "about AI" differed by at most 6.2e-6, and the yes/no answer
(probability above 0.5) agreed on 2,096 of 2,096 held-out posts (validation, test and a separate reply set). <!-- @evidence[[results/trajectory.json#answers_agree]] -->
For practical purposes it is the same model. A training step on a typical 16-post batch took 51.5 ms on the Mac
against 336.7 ms in PyTorch on the container's 4 CPUs, 6.5 times faster. <!-- @evidence[[results/trajectory.json#timing.step_speedup]] -->
One run, one seed. Untested: weaker GPUs, the iPhone in particular, where the longest batch's 550 MB of working
memory may not fit.

## Findings

1. **The WGSL run reproduces the PyTorch run.** With the same initial adapter, batch order and schedule, the
   per-step loss differs by 5.6e-7 relative on average and 6.8e-6 at most over 1,552 steps; the gap grows <!-- @evidence[[results/trajectory.json#loss_rel_diff.all_mean]] -->
   slowly with training, from 2.8e-7 in epoch 1 to 7.3e-7 in epoch 4. Final probabilities differ by at most <!-- @evidence[[results/trajectory.json#loss_rel_diff.epoch1_mean]] -->
   6.2e-6 on the 1,154 test posts, and test accuracy is 0.9445 for both (recall of "about AI" 0.8956, false <!-- @evidence[[results/trajectory.json#final_probs.test.max_abs_prob_diff]] -->
   positives 3.3% of the other posts, also equal). (`results/trajectory.json`)
2. **Every gradient matches PyTorch's.** On one batch of 8 sequences (90 to 627 tokens, so the local attention
   window and the padding both matter), the loss is within 1.8e-7 relative on the Mac and 2.7e-7 on SwiftShader. <!-- @evidence[[results/gradcheck_metal.json#lossRelErr]] -->
   Every one of the 46 parameter tensors with a nonzero gradient is within 3.9e-6 relative L2 error on the Mac. <!-- @evidence[[results/gradcheck_metal.json#maxGradRelErr]] -->
   One AdamW update is within 1.5e-4, with no parameter moved the other way. (`results/gradcheck_metal.json`, <!-- @evidence[[results/gradcheck_metal.json#updateRelErr]] -->
   `results/gradcheck_swiftshader.json`)
3. **It is 6.5 times faster than the CPU per step, and the run 7.3 times faster.** One forward and backward pass on a <!-- @evidence[[results/trajectory.json#timing.step_speedup]] -->
   16 x 222 batch: 51.5 ms on the Mac, 336.7 ms in PyTorch on the 4-CPU container. The whole run, with evaluation <!-- @evidence[[results/trajectory.json#timing.cpu_step_ms]] -->
   on 2,096 held-out examples after each epoch: 62 seconds, against 452 seconds for the same recipe on the <!-- @evidence[[results/trajectory.json#timing.cpu_uncontended_4epoch_s]] -->
   container with only validation, so the 7.3 times understates the gap a little. Training takes 13.5 to 14.4
   seconds per epoch, about 36 ms a step: batches are sorted by length, and the timed 16 x 222 batch is longer than
   most. (`results/trajectory.json`,
   `results/cpu_step_time.json`)
4. **Two head biases have a true gradient of zero, in PyTorch as well.** The last linear layer's bias and the
   head LayerNorm's bias add the same constant to every option's logit, and the segment softmax cancels it.
   Their reference gradients (5e-9 and 1e-8) are float noise, so a gradient check has to leave them out
   of relative errors. AdamW still moves them by about the learning rate each step, in both implementations,
   because it divides a gradient by its own size.
5. **The one real-GPU failure was in JavaScript.** The first Mac run gave a finite first loss and NaN after it:
   `web/traj.js` read the schedule's step count from a key the fixture did not have, so the learning rate was
   NaN. The kernels were already right on SwiftShader, and the Mac matched them.

## Method

**Model.** bsky-account-decider's page model (`web/models/account/`): Ettin-17m fine-tuned as a cross-encoder
decision model, 7 layers, hidden 256, 4 heads of 64, GeGLU 384, sliding-window attention (128) on layers 1, 2,
4 and 5, RoPE, LayerNorm without bias; int8 matmuls with a scale per output column and a 4-bit token table.
Trainable: rank-8 LoRA (alpha 16) on every Wqkv, Wo and Wi, and the decision head (Linear, GELU, LayerNorm,
Linear), 209,921 parameters in one flat vector. Loss: cross-entropy of the gold option probabilities against
the softmax of the logits over each question's options, divided by the calibration temperature. AdamW (lr 5e-4,
head 8e-4, weight decay 0.01), global-norm clipping at 1.0, 6% linear warmup then linear decay, batch 16 sorted
by length, 4 epochs.

**Trainer** (`web/train.js`). Frozen weights are dequantized to f32 once on upload, and activations are f32.
The forward pass keeps only each layer's input, and the backward pass recomputes a layer before differentiating
it. Attention backward runs in two passes without atomics: one per query row (dQ, and D = dO·O), one per key
row (dK, dV). Both recompute the probabilities from the saved log-sum-exp. LoRA A and B gradients sum over every
token, through a split-K matmul and a reduction. One tiled matmul kernel (64 x 64 tiles, arbitrary strides)
serves every product, forward and backward. Gradient clipping and AdamW also run on the GPU, so a step uploads
only the batch. About 38 KB of buffers per token: 550 MB for the longest batch (16 x 902).

**Checks.** `run/reference.py` rebuilds the model in PyTorch from the packed weights (logits match the NumPy
reference to 3.6e-6, `bsky-account-decider/run/thread_lora.py`) and writes the fixtures. The gradcheck
(`web/gradcheck.js`) compares one batch's loss, gradients and AdamW update with PyTorch's. It runs in headless
Chromium on SwiftShader (`run/headless.mjs`) and on a real GPU through the GPU Bridge (`web/bridge_gradcheck.js`).
The trajectory run (`web/traj.js`, `web/bridge_job.js`) repeats a seeded PyTorch run step for step, and
`run/compare.py` compares the two. Data: bsky-account-decider's thread-aware "about AI" labels (6,194 train,
566 val, 1,154 test, split by account) and its 376-reply whole-thread set.

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

What ran: the gradcheck on SwiftShader (202 s for the one step) and on the Mac (74 ms); two smoke jobs and the
full run through the GPU Bridge (`ewt-smoke-1`, `ewt-smoke-2`, `ewt-traj-1`, Mac tab, page v10); the seeded
PyTorch run. Labels came from bsky-account-decider ($7.17, already spent); this experiment added no cost.

What broke: a uniform struct named `P` clashed with a storage binding named `P` in two kernels (renamed `Uni`).
The first Mac run's learning rate was NaN, from the key name in Finding 5. The seeded PyTorch run shared the
container's 4 CPUs with a SwiftShader job for its first 300 steps and took 1,307 s, so its time is not used. The
CPU numbers come from an uncontended timing of the same batch (`results/cpu_step_time.json`) and from
bsky-account-decider's unseeded run of the same recipe (452 s).
