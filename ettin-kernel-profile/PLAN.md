# ettin-kernel-profile: plan

## Question

The Ettin-17m decider runs on hand-written WGSL (adapter-library `SpikeEngine` for scoring, ettin-wgsl-train
`Trainer` for the creator page's LoRA training). Would compiling the graph into fewer, fused kernels (what
tinygrad's WebGPU exporter does for jik-wtf/tinygpt-webgpu) make it faster? Separately, would merging the LoRA
spike into the int8 base (W' = W + BA, requantized) remove the LoRA side path without losing accuracy?

Both depend on where the GPU time goes, which nobody has measured per kernel. End-to-end numbers so far:
19.3 ms per 5-question decision on the Mac (browser-decider), 51.5 ms per training step (ettin-wgsl-train).

## Prior art (4 searches, 2026-11-06)

- arXiv 2604.02344 (WebGPU dispatch overhead for LLM inference, batch 1): per-dispatch API cost 32-71 µs on
  Metal; fusing Qwen2.5-0.5B from 876 to 564 dispatches gave 1.4x on Vulkan and **no gain on Metal**.
- A Zenodo preprint fuses a whole decode step into one shader (6.6-13.5x on an M2 Pro). Single source.
- QA-LoRA merges its adapter into quantized weights exactly, but only through group-wise zero points that its
  own training sets up. For plain LoRA into int8, the guides say "dequantize, merge, requantize, and measure";
  no int8 numbers found.
- This repo: METHODS.md "one dispatch per step beats one workgroup that loops" measured ~3.5 µs a dispatch on
  the Mac's GPU; "Ettin-17m matmul codes can be held to ±31" found that requantizing already-quantized codes
  overstated the cost fourfold against requantizing from fp32. A fresh ±31 merge does exactly that.

## Fixture

- Model: adapter-library `base31` (t4-w8L31: int8 codes held to ±31, 4-bit token table), invoice_processing
  spike (rank 8 on Wqkv, Wo, Wi, Wo_mlp; refit on base31).
- Data: decider-lora-creator `demo/invoice_processing.json`, 100 test cases (one sequence per labelled
  question), 300 train cases.
- Workloads: `one` = every question of the first test case (one live decision); `b16` = the first 16 test
  sequences; trainer batch = first 16 train sequences, padded as the creator page pads them.
- Device: Oskar's Mac (Apple GPU, Chrome 156) through the GPU Bridge. SwiftShader only for a smoke run.

## Arms

1. `null`: empty kernel x4000 in one pass, with and without a read-write binding: the per-dispatch floor.
2. `S e2e`: SpikeEngine's recorded graph, many copies per pass, GPU timestamps and wall.
3. `S drop <group>`: the same graph without one group (lora, base_mm, attn, norm, rope, glu, embed). Outputs
   are wrong, timing is valid. `drop lora` is the merged model's graph exactly.
4. `S iso`: every dispatch alone, repeated in its own pass.
5. `merge`: the spike folded into the base and requantized per column, scored on the 100 test cases against the
   unmerged engine: `aligned` (scale = base scale / 4, so the ±31 codes become ±124 exactly and only the delta
   rounds; falls back to /3, /2, /1 when a column overflows ±127), `fresh127`, `fresh31`. Control: `noDelta`
   (LoRA path zeroed).
6. `T`: Trainer forward and forward+backward+AdamW: e2e wall, drop per role, iso per dispatch.

## Predictions (registered before the Mac run)

- P1. Per-dispatch floor on the Mac is <= 10 µs, so SpikeEngine's ~116 dispatches cost <= 10% of `one`.
- P2. Base int8 matmuls plus attention take >= 70% of `one` (the model is compute-bound at ~3k tokens).
- P3. Dropping the LoRA path (= merged model) saves 8-20% of `one`.
- P4. Dropping rope + glu + norm together saves <= 10% of `one`: the upper bound on what fusing them buys.
- P5. `aligned` merge: argmax agreement >= 0.99 with unmerged and accuracy within 0.01; `fresh31` agrees less
  than `aligned`.
- P6. Trainer step: frozen-base matmuls (forward, recompute and backward) >= 40% of the step; LoRA matmuls
  plus LoRA gradients <= 30%.

## Decision rule

- Pursue a fused or compiled graph only if P1/P4 fail in the direction of more overhead: dispatch floor x count
  plus the elementwise groups >= 20% of `one`. Otherwise the lever is the matmul and attention kernels.
- Ship the merge (client-side, at spike load) if `aligned` passes P5 and `drop lora` saves >= 10%.
