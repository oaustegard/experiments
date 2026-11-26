# OCRia-nano-1 on WebGPU (hand-written WGSL)

LiteParse runs its OCR nets (OCRia-nano-1: a text-line detector, a script router and four line
recognizers) in the browser through its WebAssembly build, which carries the tract inference engine
inside a 20.6 MB wasm file. This folder runs the same six nets on WebGPU with about 10 KB (gzipped) of
JavaScript and WGSL instead: `engine.js` executes a straight op program per model, exported from the
upstream ONNX files by `run/export_gpu.py`, with the weights stored as int8. It is the inference half of
a WebGPU LiteParse OCR; LiteParse's page logic (resizing, boxes, routing, decoding) is ported to JS
separately and calls this engine through the interface below. The layout model (tablenet) is not ported.

Numbers below come from `test/run.sh` (export, onnxruntime references, headless Chromium); the full
output is in `../../results/gpu_parity.json` and `../../results/gpu_sizes.json`.

## Interface

```js
import { OcriaGpu } from "./engine.js";
const eng = await OcriaGpu.load(baseUrl, { models: ["det", "router", "rec_latin"] });  // opts.device: your GPUDevice
const out = await eng.infer(model, shape, data, outputNames);   // mirrors LiteParse's SessionPool::infer
```

`baseUrl` is the folder that holds `<model>/program.json` (default: `models/` beside `engine.js`). `data` is a
`Float32Array` in NCHW order; `outputNames` is optional (default: all outputs of the model).

| model | input `shape` | output |
|---|---|---|
| `det` | `[1, 3, H, W]`, H and W multiples of 32 | `{ prob: { shape: [1, 1, H/2, W/2], data: Float32Array } }` |
| `router` | `[N, 1, 32, 512]` | `{ group: { shape: [N, 4] }, flip: { shape: [N, 2] } }` (logits, Float32Array) |
| `rec_latin`, `rec_world`, `rec_rtl`, `rec_cjk` | `[N, 1, 32, W]`, any W | `{ ids: { shape: [N, T], data: Int32Array }, conf: { shape: [N, T], data: Float32Array } }` |

T is W / 4 for latin, world and rtl and W / 8 for cjk. `ids` are argmax indexes (0 is the CTC blank,
character `c` is `charset[c - 1]`); `conf` is the softmax probability of that argmax (exp(max - logsumexp)),
as the ONNX graphs compute them. onnxruntime returns `ids` as int64; here they are int32. Each model folder
also carries the upstream side files the page logic needs: `charset.json` (recognizers), `meta.json`
(`det`: `{"thresh": 0.55}`; `rec_rtl`: `{"visual_rtl": true}`).

Calls are queued and run one at a time. A forward pass is one command buffer; activation buffers come
from a pool kept between calls (`eng.trim()` frees it); only the requested outputs are read back.
`eng.stats[model]` holds the call count and wall milliseconds (upload to readback). Setting
`eng.profile = true` submits each op on its own and leaves per-op times in `eng.lastProfile` (slow; for
tuning only).

### Input normalisation (from LiteParse 2.21.2, `src/ocr/ocria/{det,rec,img}.rs`)

Every model takes pixel / 127.5 - 1 (white = 1.0). The engine does no preprocessing.

- **det** (`Detector::detect_at`): scale s = min(long_side / max(h, w), 1); H and W = max(32,
  round_half_even(side * s / 32) * 32); area resize (`resize_area`, OpenCV INTER_AREA) of the RGB image;
  RGB planar. The net turns RGB into gray itself (0.299 R + 0.587 G + 0.114 B on the normalised values).
  The probability map is at half resolution (stride 2); LiteParse thresholds it at `meta.json` thresh 0.55.
- **router** (`Router::prep`): gray (`to_gray`: (4899 R + 9617 G + 1868 B + 8192) >> 14), height 32, width
  round_half_even(w * 32 / h), at least 8 (area resize when the crop is taller than 32, else bilinear,
  `resize_linear`); the line is tiled across the 512 columns (column x takes source column x mod width)
  and cut there. Batches of up to 64. Softmax of `group` gives P(latin, cjk, rtl, world); softmax of `flip`
  gives P(upside down) as its index 1.
- **rec_*** (`Recognizer::read`): the same gray height-32 line, width clamped to [8, 1280]. Lines are sorted
  by width and batched 32 at a time; a batch is padded on the right with 1.0 to its widest line rounded
  up to a multiple of 128. A line's output is read for t < its own width / stride (stride 4, cjk 8): the
  output depends on the padded width the line ran at, so keep that batching to match LiteParse.

## Files

| | |
|---|---|
| `engine.js` | loader, shape inference, buffer pool, dispatch of each op |
| `kernels.js` | WGSL: direct conv, 1x1 conv as a tiled GEMM, 1x1 conv for few channels, fused CTC head, gray, elementwise add / fake-quant, nearest resize, concat copy, router mean+max pool |
| `models/<model>/` | `program.json` + `weights.bin` (exported at L = 127), `charset.json` / `meta.json` |
| `../../run/export_gpu.py` | ONNX -> program + weights; `--sizes` measures every model at L = 127 and L = 63 |
| `test/run.sh` | export, `make_ref.py` (fixtures and onnxruntime references), `run_parity.mjs` (headless Chromium) |

## Program format (`ocria-gpu/program v1`)

`program.json` lists `tensors` (index = id, `c` = channels; tensor 0 is the input; `dtype: "i32"` for
`ids`), `ops` in execution order, `outputs` (`{name: {tensor, rank}}`) and `weights` (`file`, `bytes`, and per
tensor `{kind, shape, off[, soff]}`). Spatial sizes are worked out at run time from the input shape.

The exporter walks the ONNX graph once. A node folds into the op that produced its input when nothing
else reads that value (the fused op then runs where the folded node was), so most graphs reduce to one
op per convolution:

| op | from | |
|---|---|---|
| `conv` | Conv (any groups; 1-D convs as 2-D with H = 1) | direct, one output per invocation |
| `pw` | Conv 1x1, group 1, stride 1 | GEMM, 64 x 64 tile per workgroup (cout > 32) or one pixel per invocation (cout <= 32) |
| `gemm` | Gemm transB = 1 (router) | `pw` with one pixel |
| `head` | Conv 1x1 -> ArgMax, ReduceMax, ReduceLogSumExp, Sub, Exp | fused: logits never reach memory |
| `gray` | Mul(x, [0.299, 0.587, 0.114]) -> ReduceSum(axis 1) | |
| `add` | Add, when it cannot fold into a conv | |
| `fq` | QuantizeLinear -> DequantizeLinear with no producer to fold into (rec_latin's input) | |
| `resize` | Resize nearest, integer scale, asymmetric, floor | |
| `concat` | Concat on channels | |
| `meanmax` | ReduceMean + ReduceMax over W -> Concat (router) | |

Squeeze(axis 2) is no op (NCHW with H = 1 is NCT). Conv-like ops end in one epilogue, applied in this
order: `v = acc + bias; v = act1(v); v = fq1(v); v = v + res; v = act2(v); v = fq2(v)` (act: relu or
sigmoid). Relu and Sigmoid fold in as act1 (act2 after a residual), an Add as `res`, and an activation
QuantizeLinear -> DequantizeLinear pair as fq1 / fq2: `fq(v) = (clamp(round(v / s) + z, 0, 255) - z) * s`
with WGSL's round-half-to-even, the value onnxruntime's uint8 QuantizeLinear -> DequantizeLinear gives. A
repeat of the fq a tensor already went through (rec_latin re-quantizes after Squeeze with the same
scale) is dropped.

The fused CTC head: a workgroup owns 64 timesteps of one line and walks the vocabulary (324 to 6214
rows) in 64-row GEMM tiles; each thread keeps a running max, first argmax and rescaled sum of exp for its
4 timesteps over the rows it computes, and the 16 partial states per timestep are merged at the end (ties
go to the lowest index, as ArgMax's). The [N, V, T] logits (up to 200 MB for a CJK batch) are never
stored.

`weights.bin`: little-endian, 4-byte aligned, all int8 codes first, then all float32. `i8` = codes
row-major with rows = dim 0 (the output channel), then one float32 scale per row at `soff`; value =
code * scale. `f32` = float32 values. Float tensors of 1024 or more elements are i8 with codes in -L..L,
scale = max|row| / L (the scheme of `run/pack.py`); biases and smaller tensors are f32. rec_latin ships as
a QDQ graph upstream, so its own int8 codes (-127..127) and per-channel scales are used as they are, its
int32 biases are dequantized as onnxruntime does, and L does not apply to it. The engine dequantizes to
float32 at load (1x1 and head weights transposed to [cin, cout]) and computes in float32.

## Numbers

### Download

gzip -9 of `program.json` + `weights.bin`, against the upstream ONNX file gzipped
(`results/gpu_sizes.json`):

| model | upstream ONNX | upstream gzip | L = 127 gzip (shipped) | L = 63 gzip |
|---|---:|---:|---:|---:|
| det | 347,580 | 309,989 | 118,862 | 110,441 |
| router | 159,029 | 143,564 | 56,587 | 53,031 |
| rec_latin | 943,530 | 787,690 | 762,308 | 762,308 (int8 as shipped) |
| rec_world | 3,913,987 | 3,625,922 | 947,468 | 828,161 |
| rec_rtl | 2,842,451 | 2,633,226 | 693,278 | 607,508 |
| rec_cjk | 5,077,778 | 4,706,867 | 1,266,608 | 1,113,055 |
| **all six** | | **12,207,258** | **3,845,111** | **3,474,504** |
| Latin page (det + router + rec_latin) | | 1,241,243 | 937,757 | 925,780 |

The runtime: `engine.js` + `kernels.js` are 32,860 bytes, 10,020 gzipped. LiteParse's wasm build
(`liteparse_wasm_bg.wasm` 2.21.2) is 20,638,448 bytes, 7,146,770 gzipped, 4,865,248 brotli (q 11); it also
holds pdfium and LiteParse's page logic, so the fair comparison is with this engine plus the JS port of
that logic, not this engine alone. Only L = 127 has a parity run (the shipped level, the same as the
existing wasm page's packs); L = 63 sizes are listed for reference.

### Parity with onnxruntime

Fixtures (`test/make_ref.py`): 3 page images from the gray600 set at a long side of 1600 for `det`, and
320 line crops cut from 10 of its pages by the detector itself, prepared and batched as LiteParse does
(router: 5 batches of 64; recognizers: 10 width-sorted batches of 32, padded to 128 to 1152 px). The
references are onnxruntime 1.32.0 on the CPU with default session options. For the fp32 models the ONNX
weights are replaced by exactly the values the engine decodes from `weights.bin` (the "unpacked-equivalent"
weights), so the comparison isolates the engine; the column on the right shows the cost of the int8 pack
itself against the upstream weights. Run from a clean shell with `bash web/gpu/test/run.sh`:

| model | reference | criterion | result | other checks |
|---|---|---|---|---|
| det | onnxruntime, unpacked-equivalent weights, 3 pages (1600 long side) | prob max abs diff < 1e-3 | **1.7e-06** (pass) | cells on the other side of 0.55 from the reference: 0; vs upstream fp32 weights max diff 0.031 |
| router | same, 320 real line crops | logits max abs diff < 1e-3, same argmax on every line | group **4.3e-06**, flip **3.8e-06**, argmax 100% / 100% (pass) | vs upstream fp32 weights: argmax group 99.38%, flip 99.38% |
| rec_latin | onnxruntime on the QDQ graph (default optimizations), 320 crops in 10 batches | ids identical on >= 99% of timesteps | **99.915%** of 32,768 (pass); within line widths 99.881%; conf max diff 0.342, mean 1.4e-03; same text on 95.6% of lines | vs onnxruntime with optimizations off (plain fake-quant): ids 99.945%, text 97.8% |
| rec_world | onnxruntime, unpacked-equivalent weights, 320 crops in 10 batches | ids identical on >= 99.9% of timesteps | **100.000%** of 32,768 (pass); within line widths 100.000%; conf max diff 3.2e-05, mean 1.4e-06; same text on 100.0% of lines | vs upstream fp32 weights: ids 99.527%, text 82.8% |
| rec_rtl | onnxruntime, unpacked-equivalent weights, 320 crops in 10 batches | ids identical on >= 99.9% of timesteps | **100.000%** of 32,768 (pass); within line widths 100.000%; conf max diff 2.8e-05, mean 7.0e-07; same text on 100.0% of lines | vs upstream fp32 weights: ids 99.460%, text 76.9% |
| rec_cjk | onnxruntime, unpacked-equivalent weights, 320 crops in 10 batches | ids identical on >= 99.9% of timesteps | **100.000%** of 16,384 (pass); within line widths 100.000%; conf max diff 4.5e-05, mean 5.5e-06; same text on 100.0% of lines | vs upstream fp32 weights: ids 99.023%, text 75.6% |

Reading it:

- The four fp32 nets match onnxruntime to float rounding (largest difference 4.5e-5 in a confidence, 1.7e-6
  in a detector probability); every argmax and every line's text is the same.
- rec_latin is a quantized graph: every activation is rounded to one of 256 levels, including the logits,
  so a last-bit difference in a sum can move a value across a rounding boundary and change a later level,
  and tied logits are common. onnxruntime itself does not agree with itself here: with default
  optimizations it runs integer QLinearConv / QLinearAdd kernels, with optimizations off it runs the float
  fake-quant this engine runs, and on these 320 lines those two agree on 99.93% of timesteps and 96.6% of
  line texts. The engine sits inside that spread (99.915% / 95.6% against the default run, 99.945% /
  97.8% against the fake-quant run). Matching the default run more closely would mean emulating its
  integer requantization; that was not needed for the 99% bar.
- Against the upstream fp32 weights (right column), the L = 127 pack moves 2 of 320 router decisions and
  0.5 to 1% of recognizer timesteps. The text of 17 to 24% of lines changes for world, rtl and cjk, but
  these crops are Latin-script pages, so those three recognizers are reading text they were not trained
  on and their outputs are low-confidence and fragile; this is not a measure of their in-domain accuracy,
  which was not tested here. The detector's probabilities move by up to 0.031, which flips 52 to 123
  map cells (0.01 to 0.025%) across the 0.55 threshold on these pages (measured on the onnxruntime
  outputs in the fixtures).

### Speed

Wall time per `infer` call in headless Chromium on **SwiftShader, a CPU emulation of a GPU**, from the
same run. These are not GPU timings: SwiftShader runs every invocation on the 4 CPUs of this container,
which other jobs were also using (load average 3.8 to 8.5 during the run; an earlier run under load 11
was 2.5x slower). They show the engine runs end to end at the sizes LiteParse uses, nothing about speed
on a real GPU.

| model | calls | input | wall ms per call | throughput |
|---|---:|---|---:|---|
| det | 3 | 1600x1248, 1600x1248, 1024x1600 | 10,331 | 10,593 ms, 11,648 ms, 8,751 ms |
| router | 5 | [64, 1, 32, 512] | 2,433 | 26.3 lines/s |
| rec_latin | 10 | [32, 1, 32, 128..1152] | 9,397 | 3.4 lines/s |
| rec_world | 10 | [32, 1, 32, 128..1152] | 9,635 | 3.3 lines/s |
| rec_rtl | 10 | [32, 1, 32, 128..1152] | 8,999 | 3.6 lines/s |
| rec_cjk | 10 | [32, 1, 32, 128..1152] | 9,118 | 3.5 lines/s |

On SwiftShader the 1x1 convolutions take about 60% of a recognizer forward and the fused head about 11%
(per-op profile of rec_world at [32, 1, 32, 384]). First-call costs (pipeline compile and buffer
allocation, the warm-up in `test/check.js`) were 57 to 222 ms per model; loading all six models took
2.2 s. **No real-GPU timing has been taken yet.** The page should be timed on a real adapter (for
example through the GPU bridge to a laptop, or `run_parity.mjs --gpu` on a machine with a GPU) before
claiming a speed-up over LiteParse's wasm build.

## Rebuild and test

```bash
cd ocria-pack
python3 -I run/export_gpu.py            # web/gpu/models/<model>/ at L = 127 (--levels 63 for another level)
python3 -I run/export_gpu.py --sizes    # results/gpu_sizes.json
bash web/gpu/test/run.sh                # export + references + headless Chromium parity -> results/gpu_parity.json
```

`make_ref.py` needs onnx, onnxruntime, opencv-python and scipy; the page images default to
`/home/user/opb/gray600` (`$OCRIA_PAGES`). `run_parity.mjs --gpu` uses the machine's own adapter instead
of SwiftShader.

## Licence

The weights in `models/` are derived from OCRia-nano-1 by LlamaIndex and are under the OCRia-nano-1 Model
License (modified OpenRAIL-M), with its use restrictions: see `../../NOTICE.md`. The code here is the
experiments repo's.
