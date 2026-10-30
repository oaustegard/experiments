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

## LFM3-1.7B: how far below 4 bits?

LFM3-1.7B is the smallest LFM3 model. The question was whether it could go to 2 bits so that it
fits on a phone. It cannot, without retraining. At 4 bits, though, a rotated GPTQ quantizer is
smaller than onnx-community's q4 file and closer to the bf16 release.

**Method.** `run/lfm3.py` is LFM3 in plain PyTorch, written from the release's model source and the
ONNX graph rather than imported. On onnx-community's q4 file with the same weights dequantized, it
matches onnxruntime at cosine 1.000000 and 100% top-1 over a 320-token prompt that crosses the
256-position sliding window. Getting there took two corrections, both in `run/parity3.py`:
- The export permutes each head's q/k dimensions into onnxruntime's partial-RoPE pairing, so the
  forward pass takes `ort=True` for ONNX weights.
- The export sets `accuracy_level` 4, which quantizes activations to int8 inside every matmul
  (cosine 0.9975, top-1 0.984). The comparison uses a copy with it set to 0.

`run/quant3.py` builds each variant from the bf16 release and scores it against bf16 on two sets:
- 4,088 tokens of WikiText-2 test text
- the bf16 model's own greedy replies to 8 voice-assistant prompts (251 tokens)

The scores are KL(bf16 ‖ variant) over bf16's top 256 tokens, top-1 agreement and perplexity.

The variants:
- `rtn`: round to nearest on MatMulNBits' grid (block 32, fp32 scale and zero point).
- `gq`: GPTQ in a 256-wide randomized-Hadamard basis onto Lloyd-Max levels, with an fp16 scale per
  block of 32 (METHODS.md). The calibration set is 48 × 512 tokens of WikiText-2 train.
- `mixA_B`: A bits on the feed-forward matrices (1.13B of the 1.47B layer weights), B bits on the rest.

The token table, which is also the lm_head, is 4-bit RTN in every variant. The GPTQ took 21–29 s
per layer on a Colab A100 and 237–346 s on the container's 4 CPUs. Results are in
`results/quant3.json`, and every variant's answers to the 8 prompts are in `results/quant3_gen.json`.

| Variant | Bits per layer weight | MB | Text KL | Text top-1 | Text ppl | Chat KL | Chat top-1 | Chat ppl |
|---|---|---|---|---|---|---|---|---|
| bf16 | 16 | 3457 | 0 | 1 | 22.56 | 0 | 1 | 1.35 |
| onnx-community q4 | 5.12 | 1107 | 0.101 | 0.824 | 24.53 | 0.149 | 0.845 | 1.55 |
| rtn4 | 5.12 | 1107 | 0.095 | 0.821 | 24.32 | 0.115 | 0.873 | 1.52 |
| **gq4** | 4.50 | **993** | **0.043** | **0.872** | 22.99 | **0.084** | **0.896** | 1.43 |
| mix3_4 | 3.73 | 851 | 0.118 | 0.814 | 24.08 | 0.323 | 0.761 | 1.95 |
| gq3 | 3.50 | 809 | 0.142 | 0.796 | 24.34 | 0.489 | 0.753 | 2.41 |
| rtn3 | 4.09 | 918 | 0.385 | 0.688 | 29.79 | 0.462 | 0.773 | 2.23 |
| mix2_4 | 2.96 | 710 | 0.430 | 0.674 | 29.72 | 1.306 | 0.594 | 5.98 |
| mix2_3 | 2.73 | 668 | 0.460 | 0.660 | 30.34 | 1.351 | 0.574 | 6.35 |
| gq2 | 2.50 | 626 | 0.577 | 0.629 | 33.00 | 1.712 | 0.522 | 9.45 |
| rtn2 | 3.06 | 729 | 5.097 | 0.094 | 3525 | 6.247 | 0.135 | 917 |

What the answers show:
- **gq4 is the one to ship.** It is 10% smaller than onnx-community's q4, and its KL to bf16 is less
  than half: 0.043 against 0.101 on text, 0.084 against 0.149 on chat. Its replies read like
  bf16's ("About 384,400 kilometers from Earth right now").
- **3 bits stay fluent and get facts wrong.** gq3 put the moon 2.1 million km away and gave an
  octopus a "third eye". mix3_4 is a little better.
- **2 bits break the chat format.** gq2 and mix2_4 emit stray `</think>` tags, repeat sentences, and
  put the moon 57 million km away. Plain round-to-nearest at 2 bits is noise (perplexity 3525).
  Usable 2-bit LFM3 would need quantization-aware training, not just post-training rounding.

## LFM3-1.7B in the browser (`web/lfm3.js`)

`run/export3.py` writes gq4 as one 994.5 MB file. The file is the engine's 122 GPU buffers back to
back, each 256-byte aligned:
- the 4-bit codes and fp16 scales of every layer matrix
- q, k, v and the gate fused into one matrix per attention layer, and w1 and w3 fused per layer
- the norms and conv taps
- the token table, as 4-bit round-to-nearest in MatMulNBits layout

`run/upload3.py` puts the file on R2 (`austegard-media`, `lfm3/gq4-v1/`) in fifteen 64 MB parts.
The bucket has a GET/HEAD CORS rule for austegard.com, and the parts' URLs and hashes are in
`web/manifest3.json`.

The engine streams the parts straight into the GPU buffers, like `lfm.js`. Before each matmul it
rotates the activations with a 256-point fast Walsh-Hadamard transform. The matmul then reads 4-bit
codes through a 16-entry Lloyd-Max table and applies an fp16 scale per 32 weights. The rest of the
engine is LFM3's own:
- RMS norm per 128-wide head on q and k
- RoPE on pairs (i, i + 64), with pairs past a full-attention layer's 32 frequencies left unrotated
- grouped-query attention over a 2048-position cache, with sliding layers seeing the last 256 keys
- the per-head sigmoid gate
- the sandwich norms
- the 7-tap conv with a 6-value state

**Parity.** `run/ref3.py` runs `run/lfm3.py` on weights decoded from the exported file. The engine
matches it at cosine 1.000000 and 7/7 greedy tokens (41-token chat prompt plus 6 decode steps), both
in headless Chromium on SwiftShader (`test/check3.mjs`) and on the Mac's GPU (bridge job
`lfm3-bench-2-mac`).

**Speed and memory.** Bridge job `lfm3-bench-2-mac`, Mac (M5), Chrome on Metal:

| | |
|---|---|
| Load: stream 994.5 MB from R2 straight to the GPU | 30.4 s |
| Prefill, 41 tokens | 28 ms |
| Decode, median per token | 4.8 ms (about 208 tokens/s); 90th percentile 4.9–5.1 ms |

On SwiftShader, peak PSS was 1528 MB against a 152 MB baseline.

The token table is the largest buffer at 168 MB. That is more than WebGPU's default 128 MiB binding
limit, so the engine needs an adapter that allows larger bindings, which the Mac's does (4 GB). An
iPhone has not been tried.
