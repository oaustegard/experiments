# Nemotron Parse 2.0 WebGPU engine

`nemotron.js` runs [NVIDIA Nemotron Parse 2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0)
(C-RADIOv2-H ViT-H/16 encoder, conv neck, 10-layer mBART decoder) on hand-written WGSL kernels, from
a pack written by `run/export.py`. `postprocess.js` holds the tokenizer decode and the ports of the
model repo's `postprocessing.py` / `latex2html.py`. No dependencies.

## API

```js
import { NemotronParse } from "./engine/nemotron.js";
const m = await NemotronParse.load({ base: "models/default/", onProgress });
const r = await m.parse(imageBitmapOrCanvasOrImageData, { onToken, signal });
// r = { raw, tokens, ms: { encode, decode }, blocks: [{ cls, bbox: [x0, y0, x1, y1], text }] }
```

`load(opts)`

| option | |
|---|---|
| `base` | pack directory URL (`manifest.json`, `tokenizer.json`, `weights-NNN.bin.gz`) |
| `device` | a `GPUDevice`; otherwise one is requested with the adapter's `maxBufferSize` and `maxStorageBufferBindingSize` |
| `onProgress({phase, loaded, total})` | `phase` is `download`, `upload` or `ready`; bytes are compressed bytes |
| `size` | `{height, width}` canvas, multiples of 16, at most 2048; default the pack's 2048 x 1664. Small sizes are for tests |
| `cache` | `false` skips the Cache API (`nemotron-pack-v1`, keyed by shard URL and pack id) |
| `budget` | multiply-adds per dispatch, default 4e9; lower it if a driver's watchdog kills long dispatches |
| `parallel` | concurrent shard fetches, default 4 |
| `cacheRows` | initial self-attention cache rows, default 1024; it doubles when a page runs past it |

`parse(image, opts)`: `image` is an `ImageBitmap`, canvas, `HTMLImageElement` or `{data (RGBA), width, height}`.

| option | |
|---|---|
| `prompt` | control-token string, default `</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>` |
| `maxTokens` | default 9000 |
| `onToken(id, text)` | per generated token, `text` is the UTF-8 completed by it |
| `signal` | `AbortSignal`, checked per encoder layer and per token |
| `size` | changes the canvas for this and later calls |
| `tableFormat` | `markdown` (default), `html` or `latex` for `Table` blocks |
| `textFormat` | `markdown` (default) or `plain` |
| `onStage(done, total)` | encoder progress: patch embedding, 32 blocks, neck |

`bbox` is in pixels of the image passed in. `raw` is the decoded output with the `<x_..><y_..>` and
`<class_..>` tokens in it. Lower-level calls: `preprocess(image)`, `encode(patches)`,
`decode(promptIds, opts)`, `probe(name, floats)`.

Preprocessing follows the model's processor: shrink only (never enlarge) to fit the canvas keeping
aspect, PIL-style bilinear, centre on a white canvas, scale to [0, 1], CLIP mean and std.

## Pack format (`nemotron-pack/1`)

`manifest.json`:

- `tensors[name]`, named as in `model.safetensors`, offsets into one byte stream, every record 4-byte aligned:
  - `{kind: "f16", shape, w: [offset, length]}`
  - `{kind: "int8", shape: [N, K], q: [offset, length], s: [offset, length]}`: int8 codes row-major, one f16 scale per row
  - `{kind: "int4g", shape: [N, K], group: G, q, s}`: two codes per byte (code + 8, low nibble = even column), f16 scales `[N, K / G]`
- `shards: [{file, bytes, gzip}]`: the stream cut every 47 MiB and gzipped. A host may serve them with
  `Content-Encoding: gzip` or as plain files; the loader inflates what arrives still compressed.
- `id` (hash of the stream, the cache key), `dims`, `image` (canvas, mean, std), `generation`
  (eos, repetition penalty, prompt), `scheme`, `sizes` (raw and gzip bytes per component).

Shapes: a leading 1 is dropped (`pos_embed` is `[16384, 1280]`), other tensors with more than 2 dims
are flattened to `[out, in*kh*kw]`. `lm_head` is the embedding table, stored once. The exact
quantization math is in the docstring of `run/export.py`; `run/ref.py` `load_quantized(scheme)` gives
the PyTorch model with the same dequantized values.

`tokenizer.json` is slimmed to `{tokens: [72256 strings], special: [ids]}`; decoding is byte-level BPE.

The engine reads any kind for any 2-D tensor (K a multiple of 8; int4g groups a multiple of 8).
`pos_embed` and `cls_token` are dequantized in JS; everything else is dequantized in the kernels.

## How it runs

- Weights are written into GPU buffers shard by shard as they arrive; a tensor larger than the
  storage binding limit is split by rows over several buffers. f16 records are widened to f32 on upload.
- Encoder: patch rows are built in JS and embedded by the same linear kernel as everything else. The
  position grid is interpolated in JS once per canvas size. Each linear thread computes one output
  column for 8 rows; attention keeps a running softmax per (4 queries, head) over all keys, so there
  is no score matrix. Linears and attention are cut into dispatches of at most `budget`
  multiply-adds; the MLP runs in row chunks so its hidden buffer stays under 128 MB.
- Decoder: one token per submit. Cross-attention keys and values are computed once per page, the
  self-attention keys and values are appended to a cache, the output projection reads the embedding
  table, and the argmax with the repetition penalty runs on the GPU (a 1 KB readback per token).
- Activations are f32.

## Check

`python3 run/ref.py` writes fixtures from the PyTorch model at a 512 x 416 canvas (fp32 and the
quantized weights); `node test/check.mjs [pack]` runs the engine in headless Chromium on SwiftShader
against them.
