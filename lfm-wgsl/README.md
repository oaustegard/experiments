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

**Real GPUs.** Bridge job `lfm-bench-2-iphone` ran on the iPhone 16 Pro in Safari on iOS 27.0.1,
and `lfm-bench-1` ran on the Mac in Chrome 156 on Metal. Each ran the same 35-token prompt and 64
decode steps three times. Next-token picks agreed with onnxruntime at 13 of 13 steps in every run on
both devices.

| | iPhone 16 Pro, Safari | Mac, Chrome on Metal |
|---|---|---|
| Load: stream 228 MB from Hugging Face to the GPU | 91.3 s (phone network) | 6.2 s |
| Prefill, 35 tokens | 107–181 ms | 15–18 ms |
| Decode, median per token | 14–15 ms (about 71 tokens/s) | 3.8–8.1 ms |
| Decode, 90th percentile | 17–21 ms | 5.0–8.6 ms |

The iPhone tab loaded the model and ran without being killed. The tab running transformers.js 3.8.1
had died on SmolLM2-360M, a model of similar size.

On SwiftShader in the container, prefill took 67 s and a decode step 4.6 s. As METHODS.md warns,
those timings say nothing about real hardware.

**Memory.** Peak PSS summed over all Chromium processes, in headless Chromium on SwiftShader. Each
row loads a model and generates once. With SwiftShader, GPU buffers count as process memory, much as
they do on Apple's unified memory.

| Loader and model | Peak, load and generate | After |
|---|---|---|
| This engine, LFM2.5-350M q4 (228 MB uploaded) | 649 MB | 571 MB |
| transformers.js 4.3.1, LFM2.5-350M q4 on WebGPU | 1726 MB | 866 MB |
| transformers.js 3.8.1, SmolLM2-360M q4 on WebGPU (the page that crashed the iPhone) | 4054 MB | 3094 MB |
| transformers.js 4.3.1, SmolLM2-360M q4 on WebGPU | 3729 MB | 2737 MB |
| transformers.js 3.8.1, SmolLM2-360M q8 on wasm | 1722 MB | 1553 MB |
| Supertonic 2 TTS on wasm, transformers.js 4.3.1 | 1195 MB | 989 MB |

Every row starts from a baseline of 155 to 158 MB. The engine's own row ran with no other browser
open. The transformers.js rows came from a serial batch, and a few of them overlapped with other
headless runs in this session, so read them as rough. The WebLLM runs from the same batch are left
out: one started from a baseline of 1382 MB because another browser was running, and one lost its
GPU instance. The Qwen3.5-0.8B run under transformers.js 4.3.1 timed out after 15 minutes on
SwiftShader.

Pending: memory on the phone itself. Safari reports nothing comparable to PSS, so the iPhone test is
whether the tab survives with both the engine and Supertonic loaded.

Run the check:

```bash
python3 lfm-wgsl/run/ref.py           # needs model/ from onnx-community/LFM2.5-350M-ONNX (onnx/model_q4.onnx + _data, config, tokenizer)
python3 lfm-wgsl/run/manifest.py
node lfm-wgsl/test/check.mjs
```
