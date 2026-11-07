# ettin-kernel-profile: where the Ettin-17m decider's GPU time goes

The Ettin-17m decider (a 17M-parameter encoder that answers typed questions such as "how should this invoice be
dispositioned?") runs on hand-written WebGPU kernels: adapter-library's `SpikeEngine` scores decisions, and
ettin-wgsl-train's `Trainer` trains LoRA adapters ("spikes") in the decider-lora-creator page. This experiment
timed every kernel of both on a Mac's Apple GPU to answer two questions. Would compiling the graph into fewer,
fused kernels, as tinygrad's WebGPU exporter does for jik-wtf/tinygpt-webgpu, make it faster? And can the LoRA
spike be merged into the int8 base weights to remove its side path without losing accuracy?

Terms used below. A LoRA adapter adds a low-rank update to a frozen weight matrix: y = Wx + B(Ax), with A and B
rank 8 here. The engine runs that update as a "side path" of two extra matmuls per weight matrix. The base
weights ship as int8 codes held to ±31 with one scale per output column ("base31"). "Merging" folds B·A into W
and stores the sum as new int8 codes. "Agreement" is the share of questions whose top answer is unchanged by
the merge. Two timing arms recur: "drop" times the whole graph with one group of kernels left out, and "iso"
times each kernel alone; both are per cent of the whole graph's GPU time.

## Answer

Compiling or fusing the graph would buy almost nothing on the GPU. A dispatch costs 1.47 µs on the Mac <!-- @evidence[[results/summary.json#null_us.rwBinding_us]] -->,
so the 122 dispatches of one decision take 3.0% of its GPU time <!-- @evidence[[results/summary.json#S.one.dispatch_floor_pct]] -->,
and dropping every small elementwise kernel saves only 2.4% <!-- @evidence[[results/summary.json#S.one.elementwise_drop_pct]] -->.
Attention and the base matmuls take 78.9% of the 6.0 ms GPU time of one decision <!-- @evidence[[results/summary.json#S.one.base_plus_attn_pct]] -->,
and attention alone is 59.9% of a training step <!-- @evidence[[results/summary.json#T.step.iso_pct.attn]] -->.

The LoRA side path costs 17.8% of scoring <!-- @evidence[[results/summary.json#S.one.drop_saves_pct.lora]] -->, and merging removes it with almost no change.
Merged at an int8 scale aligned to the ±31 grid, the model scores 0.834 on 500 test questions <!-- @evidence[[results/summary.json#merge.aligned.accuracy]] -->,
the same as unmerged, with 99.2% of answers unchanged <!-- @evidence[[results/summary.json#merge.aligned.argmaxAgree]] -->.
Requantizing it to ±31 instead gives agreement 0.93 <!-- @evidence[[results/summary.json#merge.fresh31.argmaxAgree]] -->.

The page sees 14 ms per decision <!-- @evidence[[results/summary.json#S.one.score_wall_ms]] -->; the 8.0 ms spent outside the kernels <!-- @evidence[[results/summary.json#S.one.outside_gpu_ms]] -->
(readback, command encoding, the JavaScript head) is the largest slice, not yet broken down. If encoding the 122
dispatches dominates it, fewer kernels would help after all.

## Findings

1. Per-dispatch cost is not the bottleneck: 1.47 µs a dispatch with a read-write binding (0.90 µs without) <!-- @evidence[[results/summary.json#null_us.rwBinding_us]] -->,
   and the per-kernel times measured alone sum to the whole graph's GPU time (5.97 vs 6.0 ms for one decision <!-- @evidence[[results/summary.json#S.one.iso_sum_ms]] -->),
   so there are no stalls between kernels for fusion to remove. This covers GPU time only; CPU-side encoding is
   not measured.
   (Round 1, `ettin-prof-1`)
2. Attention is the largest kernel group: 40.5% of scoring <!-- @evidence[[results/summary.json#S.one.iso_pct.attn]] --> and 59.9% of training <!-- @evidence[[results/summary.json#T.step.iso_pct.attn]] -->.
   The three global-attention layers cost 540.7 µs each <!-- @evidence[[results/summary.json#S.one.attn_global_us]] --> against 199.7 µs <!-- @evidence[[results/summary.json#S.one.attn_local_us]] -->
   for a sliding-window layer. The kernel runs one workgroup per query row and head (grid 1,723 x 4), each reading
   the keys and values from memory on its own; a tiled (flash-style) kernel that shares them is the next lever. (Round 1, `ettin-prof-1`)
3. The LoRA side path costs 17.8% of one decision's GPU time <!-- @evidence[[results/summary.json#S.one.drop_saves_pct.lora]] -->, and most of that is its second
   matmul, Y += (XA)B, at 13.2% <!-- @evidence[[results/summary.json#S.one.lora_B_pct]] -->. Its inner dimension is only 32 (rank 8 padded), so its cost is reading
   and rewriting the full activation Y. Fusing it into the base matmul's epilogue would remove most of it without
   merging (estimated, not built). (Round 1, `ettin-prof-1`)
4. An aligned merge leaves accuracy unchanged on this spike and test set. Each column's int8 scale is set to the base scale / 4 (or / 3, / 2 where
   the delta overflows), so the ±31 base codes stay exact and only the LoRA delta rounds: accuracy 0.834 = 0.834,
   agreement 99.2%. A fresh ±127 requantization is about as good (0.990 agreement <!-- @evidence[[results/summary.json#merge.fresh127.argmaxAgree]] -->). The LoRA
   delta reaches 33.4 base quantization steps <!-- @evidence[[results/summary.json#merge.aligned.maxDeltaSteps]] -->; at scale / 4 a column overflows int8 once
   base plus delta passes 31.75 base steps, which happened in 1,388 of 14,336 columns (1,386 took / 3, 2 took / 2).
   Measured on one spike over 500 questions from 100 cases. The merge takes 482 ms of JavaScript per spike <!-- @evidence[[results/summary.json#merge.aligned.mergeMs]] -->. (Round 1, `ettin-prof-1`)
5. A merge requantized to ±31 changes 7% of answers (agreement 0.93 <!-- @evidence[[results/summary.json#merge.fresh31.argmaxAgree]] -->) with logit errors up to 3.25 <!-- @evidence[[results/summary.json#merge.fresh31.maxAbsDlogit]] -->;
   its accuracy (0.838 <!-- @evidence[[results/summary.json#merge.fresh31.accuracy]] -->) moved within noise. The ±31 download format stays; the merge must go
   to the wider int8 range on the device. (Round 1, `ettin-prof-1`)
6. In training, the frozen-base matmuls are 22.5% of a step <!-- @evidence[[results/summary.json#T.step.iso_pct.base_mm]] --> and the LoRA matmuls and gradients 11.9% <!-- @evidence[[results/summary.json#T.step.lora_pct]] -->;
   backward attention alone is 48.7% <!-- @evidence[[results/summary.json#T.step.iso_pct_dir.bwd:attn]] -->. (Round 1, `ettin-prof-1`)

## Method

- Device: Oskar's Mac (Apple GPU, Metal 3, Chrome 156) through the GPU Bridge, job `ettin-prof-1` (76.5 s).
  GPU times come from timestamp queries around a pass holding many copies of the graph; wall times from
  `onSubmittedWorkDone`.
- Model: adapter-library `base31` (int8 matmul codes held to ±31, 4-bit token table) with the
  `invoice_processing` spike (rank 8 on Wqkv, Wo, Wi, Wo_mlp), code at experiments `715e92af`.
- Workloads: `one` = the 5 questions of one invoice case (1,723 tokens); `b16` = 16 test questions (5,618 tokens);
  trainer batch = 16 training questions padded to 384 tokens. Merge arms score all 500 test questions of 100 cases.
- Arms: per-dispatch floor (an empty kernel 4,000 times); the whole graph; the graph with one kernel group
  removed (outputs wrong, timing valid; "no LoRA" is exactly the merged model's graph); every dispatch alone,
  repeated until its pass takes 3 ms. Merge arms: `aligned`, `fresh127`, `fresh31`, and `noDelta` (LoRA zeroed,
  accuracy 0.46, to show the spike matters).
- PLAN.md's 19.3 ms per decision came from browser-decider's engine (no LoRA side path) on its own data; it was
  not reproduced here, so the two page timings are not comparable.
- Trainer drop arms are wall times of single submissions and vary by ±10% (some "savings" are negative); the
  per-kernel sums are the reliable trainer numbers.
- `run/summarize.py` derives `results/summary.json` from the raw `results/ettin-prof-1/result.json`.
- Prior art: arXiv 2604.02344 found kernel fusion gave no gain on Metal at batch 1 (1.4x on Vulkan); QA-LoRA merges
  into quantized weights exactly only through group-wise zero points its training sets up.

## Log

### Round 1, 2026-11-07: profile and merge arms

Asked: is a compiled, fused graph (tinygrad-exporter style) worth building for Ettin, and can the LoRA spike be
merged into the int8 base? Registered predictions in PLAN.md (commit 53c217cc), then ran `ettin-prof-1` on the Mac.
A SwiftShader smoke run first exposed two harness problems: `gpu-bridge/run/local.js` resolves on the first
progress message, and full-size sequences take over 25 minutes on 4 CPUs, so smoke mode uses the shortest
examples and cuts every loop.

| | one decision | b16 |
|---|---|---|
| GPU time, whole graph | 6.0 ms | 19.0 ms |
| page `score()` wall | 14 ms | 40.5 ms |
| attention | 40.5% | 41.8% |
| base int8 matmuls | 38.4% | 38.5% |
| LoRA side path (iso) | 18.1% | 16.6% |
| norm + RoPE + GLU (drop) | 2.4% | -2.8% (noise) |
| dispatch floor x 122 | 3.0% | 0.9% |

| merge arm | accuracy | agreement with unmerged | max |Δ logit| |
|---|---|---|---|
| unmerged | 0.834 | 1 | 0 |
| aligned | 0.834 | 0.992 | 0.56 |
| fresh ±127 | 0.832 | 0.990 | 0.53 |
| fresh ±31 | 0.838 | 0.930 | 3.25 |
| LoRA zeroed | 0.460 | 0.442 | 5.61 |

Training step (511 dispatches, 83.1 ms wall): attention 59.9% (backward 48.7%), base matmuls 22.5%, LoRA
matmuls and gradients 11.9%, everything else under 6%.

Predictions against measurements:

- P1 (dispatch floor <= 10 µs, <= 10% of one decision): held, 1.47 µs and 3.0%.
- P2 (base matmuls + attention >= 70%): held, 78.9%.
- P3 (dropping LoRA saves 8-20%): held, 17.8%.
- P4 (norm + RoPE + GLU <= 10%): held, 2.4%.
- P5 (aligned merge agreement >= 0.99, accuracy within 0.01; ±31 agrees less): held, 0.992, 0.834 = 0.834, ±31 0.93.
- P6 (training: base matmuls >= 40%, LoRA <= 30%): split. LoRA 11.9% held; base matmuls 22.5% failed, because
  attention backward takes 48.7% on its own.

Reading at the time: by the decision rule in PLAN.md, a fused or compiled graph is not worth building (dispatch
floor plus elementwise kernels are about 5%, under the 20% bar), and the merge passes (lossless, saves 17.8%).
The merge costs 482 ms per spike, so it suits a page that loads one spike, not adapter-library's router, which
switches spikes per decision; fusing the LoRA B matmul into the base matmul's epilogue would serve both. The
two levers with more in them are a tiled attention kernel and the 8 ms of `score()` outside the GPU.
