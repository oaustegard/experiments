# RESULTS: the 4-layer EG2 query encoder on hand-written WGSL

Built 2026-10-21 for the "Embedded Embedder" project on board
[#104](https://github.com/oaustegard/experiments/issues/104). The model is
[`eg2-layer-student`](../eg2-layer-student/RESULTS.md)'s shipped k = 4 cut (`model_t3w.onnx`: q4
layers, 3-bit token table, 32K vocabulary, 25.35 MB), which the
[Bluesky image search](https://austegard.com/bsky/image-search/) runs in the page.

## Answer

Can a custom WebGPU engine in hand-written WGSL replace onnxruntime-web for this query encoder,
with the same vectors? Yes, and it is much faster. On Oskar's Mac (Metal 3, Chrome 155), a typical
query of 10 to 20 tokens takes 0.7 ms <!-- @evidence[[results/bench_apple_metal3.json#summary.wgsl_ms]] -->
with readback included. On the same file and the same queries, onnxruntime-web takes 4.875 ms on
WebGPU <!-- @evidence[[results/bench_apple_metal3.json#summary.ort_webgpu_ms]] -->
and 38.825 ms on wasm. <!-- @evidence[[results/bench_apple_metal3.json#summary.ort_wasm_ms]] -->
That makes the engine 7.0× <!-- @evidence[[results/bench_apple_metal3.json#derived.speedup_vs_ort_webgpu]] -->
faster than onnxruntime-web on WebGPU. Its vectors agree with onnxruntime CPU at cosine 1 on all 12
test queries, run singly and in a batch. <!-- @evidence[[results/bench_apple_metal3.json#wgsl.cos_single_min]] -->
The engine reads the same `model_t3w.onnx_data` that onnxruntime-web uses, so a page that keeps
onnxruntime-web as a fallback downloads the weights only once.

The same engine also runs the full 24-layer EmbeddingGemma 2 (`model_q4.onnx`, 174 MB), again at
cosine 1 to onnxruntime CPU in headless Chromium. <!-- @evidence[[results/check_swiftshader_full.json#wgsl.cos_single_min]] -->
Its speed on a real GPU has not been measured yet.

Two things could change this: a GPU that is not Apple's, or Safari. Neither has been measured.

## Findings

1. **Faster on a real GPU at every length tested.** The per-query medians on Metal are 0.7 to 0.8 ms
   up to 40 tokens and 1.7 ms at 118 tokens. Onnxruntime-web on WebGPU takes 4.8 to 5.45 ms, which
   is 3.2× slower at 118 tokens. <!-- @evidence[[results/bench_apple_metal3.json#derived.speedup_118_tokens_vs_ort_webgpu]] -->
   Twelve queries in one batch take 3.3 ms. <!-- @evidence[[results/bench_apple_metal3.json#wgsl.ms_batch12.median]] -->
   Engine init takes 37.5 ms <!-- @evidence[[results/bench_apple_metal3.json#engine_init_ms]] -->
   and an onnxruntime-web WebGPU session 163.5 ms. <!-- @evidence[[results/bench_apple_metal3.json#ort_webgpu.init_ms]] -->
   (`web/index.html`, Oskar's run)
2. **Exact.** Every check came out at cosine 1, on Metal and in headless Chromium, for single
   queries and for packed batches. The NumPy reference (`run/ref.py`) also matches onnxruntime CPU
   at 1.0000000 on 10 to 118 tokens.
3. **SwiftShader ranks the arms in reverse.** In headless Chromium the engine took 1678.625 ms <!-- @evidence[[results/check_swiftshader.json#summary.wgsl_ms]] -->
   against onnxruntime-web WebGPU's 460.15 ms, <!-- @evidence[[results/check_swiftshader.json#summary.ort_webgpu_ms]] -->
   because SwiftShader charges a few ms for each of the engine's ~79 dispatches. This is the
   reversal METHODS.md already warns about: decide on a real-GPU run only.
4. **A loop-local WGSL accumulator has to be initialized explicitly.** The first build came out at
   cosine 0.94, because `var part: array<f32, 8>;` inside the block loop kept the previous block's
   sums on Tint/SwiftShader. Probing each buffer against the NumPy reference found the bug in the
   first matmul.

5. **The full model needs no new kernel design, only generality.** The full model fuses q/k/v and
   gate/up. Its layers 5, 11, 17 and 23 use full attention with head_dim 512, one KV head and their
   own RoPE table (θ 1e6). Its token table is 4-bit, with 262K rows. Its 174 MB weight file is more
   than the 128 MiB a storage binding allows by default. The manifest turns each fused matmul into
   row slices. Each dispatch binds only the window of the weight buffer that holds its tensors. The
   exporter folds each layer scalar into the next norm's epsilon (rms(s·x, ε) = rms(x, ε/s²)), so the
   manifest converts every epsilon back to the engine's scaled input. In headless Chromium, every
   test query, run singly and in a batch, comes out at cosine 1. <!-- @evidence[[results/check_swiftshader_full.json#wgsl.cos_batch_min]] -->
   (Round 2)

## Method

- **Graph.** EmbeddingGemma 2's text tower cut to 4 layers. The 3-bit token table is scaled by
  √512. PLE is rms(x·W_ple·512^-½) per layer. Each layer runs: RMSNorm, then q/k/v (q4); per-head
  RMSNorm of q and k with weights and of v without; RoPE (θ 1e4, head dim 256); bidirectional GQA
  attention (4 query heads, 2 KV heads, scale 1); o; a post-norm residual; a GELU-tanh gated MLP; a
  post-norm residual; and the PLE gate, h = (h + rms(W_p(gelu(W_g h)·ple_i))·w)·s. After the
  layers come the final RMSNorm, the mean over tokens, the 512→768 q4 projection and L2. The engine
  pools before the projection, which is linear, so the projection runs once per sequence.
- **Kernels** (`web/engine.js`, 8 WGSL shaders, 79 dispatches in one compute pass):
  - the table gather;
  - a q4 matmul in onnxruntime's MatMulNBits layout (32 columns × 8 k-splits × 8 rows per
    workgroup, each weight dequantized once and applied to 8 rows);
  - RMSNorm, with the residual and the layer scalar fused in;
  - q/k/v norm plus RoPE, from a host-computed cos/sin table;
  - attention, one workgroup per token and head;
  - the GELU gate, pooling and L2.

  Sequences are packed back to back, and attention stays inside each one. Each dispatch has its own
  static uniform, and only the token count changes per run.
- **Weights.** `run/manifest.py` walks the ONNX graph and writes `web/manifest.json`, which holds
  each tensor's role and its byte offset in `model_t3w.onnx_data`, plus the constants the graph
  keeps inline. The whole file is one storage buffer.
- **Checks.** Twelve queries, tokenized as the page tokenizes them, are in `web/queries.json` with
  their onnxruntime CPU vectors. `run/check.py` runs `web/index.html` headless. The page
  `austegard.com/experiments/eg2-wgsl-engine/web/` times the three arms: 20 reps after 3 warm-ups,
  median per query, readback included.

## Log

### Round 1, 2026-10-21

The coordinator asked whether the image-search page used a custom WGSL kernel. It did not: only
the scan was custom, and the query model ran on onnxruntime-web. Oskar said "Yes build it". I wrote
the NumPy reference from the ONNX graph (cosine 1.0000000 to onnxruntime CPU on the first try), then
the engine. The first headless run gave cosine 0.94. Probing showed the embedding was exact and the
first matmul was wrong; it was the loop-local accumulator (finding 4), and fixing it gave 1.0000000
everywhere. Headless with onnxruntime-web: every arm at cosine 1, and SwiftShader timings reversed
(finding 3). Oskar's Mac run gave finding 1.

| tokens | WGSL ms | ORT WebGPU ms | ORT wasm ms |
|---:|---:|---:|---:|
| 10 | 0.7 | 4.85 | 35.0 |
| 13–19 (9 queries) | 0.7–0.8 | 4.8–4.95 | 37.3–42.5 |
| 40 | 0.8 | 5.0 | 58.1 |
| 118 | 1.7 | 5.45 | 121.2 |

The site change is on branch `wgsl-engine` of oaustegard.github.io (55f49c3). The engine runs on the
index's own device and writes the query vector straight into the scan's buffer. Onnxruntime-web
loads only if the engine fails, and `?engine=ort` forces it. Headless, it returned the same top 36
as the onnxruntime-web path.

### Round 2, 2026-10-23: the full 24-layer model

Oskar asked whether the engine works for the full model as well, then said to try it. The source
is onnx-community/embeddinggemma-2-ONNX `onnx/model_q4.onnx`, text inputs only (the image, video
and audio features are empty).

- `run/manifest_full.py` reads that graph into the same manifest schema. q/k/v and gate/up become
  row slices of the fused MatMulNBits weights; that works because each output column's blocks,
  scales and zero points are contiguous. The q and k norm weights are tiled over heads, and v's are
  all ones; the script checks both. Layer 23's scalar has no tensor, because it is folded into the
  final norm's epsilon (7.6e-5), so the engine uses 1 and that epsilon. Every other norm comes out
  at ε = 1e-6 on the engine's scaled input.
- `run/ref.py --full`: NumPy against onnxruntime CPU, cosine 1.0000000 on all 12 queries (10 to 118
  tokens).
- `web/engine.js` changes:
  - an `embed_q4` gather;
  - prep and attention take head_dim 256 or 512 per layer;
  - one RoPE table per attention kind;
  - norm epsilons per tensor;
  - weight windows, each aligned to `minStorageBufferOffsetAlignment` and kept under
    `maxStorageBufferBindingSize`.

  The largest window is the token table's 86 MB.
- `run/check.py --full` (parity only; SwiftShader takes minutes per timed pass): cosine 1 single
  and batched. The 4-layer cut still passes at cosine 1.
- Bench: `web/index.html?full=1` fetches the 174 MB weight file from Hugging Face, which sends CORS
  headers, and compares the engine with onnxruntime-web on WebGPU and wasm. A Metal run is pending.


### Round 3, 2026-10-26: the shared runtime

Oskar picked the block-spec plan for a generic engine (`_lib/wgsl/README.md`). Phase 1 moved what this
engine and `ettin-native`'s had copied from each other into `_lib/wgsl/runtime.js` (`SeqEngine`):
batching, dispatch recording, weight upload with binding windows, and `embed` / `embedTo` / `probe`.
The shaders and the forward pass did not change.

- `run/check.py` and `run/check.py --full`: cosine 1, single and batched, the same values as the
  pre-move engine in the same Chromium. `eg2-vocab-codebook/run/engine_check.py` on the site's rhq
  build passes as well.
- `web/queries.json` was regenerated. The committed one predated the weights the site now serves at
  `bsky/image-search/model/ort/` (same shapes, retrained), against which the engine and onnxruntime
  CPU both gave 0.90 to 0.98 to the old fixture. `bench.js` and `side.html` point at that directory,
  since the site's `model/` now holds the rhq build.
