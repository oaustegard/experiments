# lfm-wgsl

LFM2.5-350M (Liquid AI) as a chat model on a hand-written WebGPU engine, built for
[`voice-loop`](../voice-loop/README.md) after transformers.js 3.8.1 crashed an iPhone 16 Pro tab
while loading SmolLM2-360M.

## Why a custom engine

The crash was the page running out of memory. In headless Chromium, transformers.js 3.8.1 loading
SmolLM2-360M (388 MB of q4 weights) on WebGPU took the browser's summed RSS from 364 MB to 3.9 GB,
and to 4.8 GB while generating. That RSS sum counts shared pages more than once, so treat it as an
upper bound. Either way, the library fetches the whole file into an ArrayBuffer and hands it to
onnxruntime, which copies it again.

This engine streams `model_q4.onnx_data` and writes each network chunk straight into GPU buffers,
one per operator. JS never holds more than one chunk, and the queue is drained every 32 MB so
writeBuffer's staging copies are released. The token table and the tied lm_head are the same bytes,
stored twice in the file, and only one copy is uploaded. Most of the 16 MB RoPE tables are skipped
too, leaving 228.4 MB on the GPU out of a 293.6 MB file.

## Pieces

- `run/manifest.py` reads onnx-community's `model_q4.onnx` graph and writes `web/manifest.json`. The
  manifest maps every byte range of the data file to a buffer and a destination offset, and gives
  each layer's word offsets.
- `web/lfm.js` holds the engine. It reuses the shared q4 MatMulNBits matmul from `_lib/wgsl` for
  prefill, and adds its own kernels:
  - a single-row q4 GEMV for decode
  - the 4-bit embedding gather
  - RMS norm for 1024-wide rows and 64-wide heads
  - the short convolution and its 3-step state
  - RoPE with the KV cache write
  - causal grouped-query attention over a 2048-position cache
  - SwiGLU
- `run/ref.py` is the onnxruntime CPU reference. It prefills a 35-token chat prompt, then
  greedy-decodes 12 tokens. `test/check.mjs` runs the engine on the same inputs in headless
  Chromium on SwiftShader.

Two export details matter for parity:
- **The short convolution:** the graph keeps the last T of the T + 1 conv outputs.
- **The final norm:** it is exported as `SkipSimplifiedLayerNormalization(x, skip=x)`, which
  normalizes 2x. That equals normalizing x with eps / 4.

## Results

**Parity.** The engine matches onnxruntime CPU, in headless Chromium on SwiftShader:

| | |
|---|---|
| Logits cosine, prefill and all 12 decode steps | 1.000000 (minimum 0.9999999999974) |
| Largest absolute logit difference | 0.0001 |
| Greedy next token agreeing with onnxruntime | 13 of 13 steps |

**Speed.** On SwiftShader, prefill took 67 s and a decode step 4.6 s. SwiftShader is a software
rasterizer whose timings say nothing about a real GPU (METHODS.md), so speed is still unmeasured.

**Memory.** The check measured a peak PSS of 1412 MB against a 899 MB baseline. That baseline
included another Chromium instance running a measurement at the same time, so the number needs a
clean rerun.

Pending:
- Decode speed and memory on the iPhone 16 Pro, through `voice-loop`.
- A clean memory comparison against transformers.js 4.3.1 and WebLLM.

Run the check:

```bash
python3 lfm-wgsl/run/ref.py           # needs model/ from onnx-community/LFM2.5-350M-ONNX (onnx/model_q4.onnx + _data, config, tokenizer)
python3 lfm-wgsl/run/manifest.py
node lfm-wgsl/test/check.mjs
```
