# web/det: PP-OCRv6 text-line detector on WebGPU, packed weights

PaddlePaddle's PP-OCRv6 det (tiny and small, Apache-2.0) in hand-written WGSL. The weights ship as the
shared packed spec (int8 codes with one fp16 scale per output channel by default) and are dequantized to
f32 once at load, so the kernels read plain f32 and are the ones the tesseract-gpu port uses.

```
export PPOCR_MODELS=<folder with tiny_det/inference.{onnx,yml}, small_det/...>
python3 -I run/export_det.py [--scheme r63|r127|f16|f32] [tiny|small]   # -> web/det/models/<name>/
python3 -I run/export_det.py --sizes                                    # byte counts -> results/det_sizes.json
python3 -I web/det/test/make_pages.py && python3 -I web/det/test/fetch_funsd.py   # test pages (git-ignored)
python3 -I run/ref_det.py                                               # NumPy executor + fixtures
node web/det/test/run_det.mjs [tiny=page,page] [small=page]             # WGSL in headless Chromium vs NumPy
python3 -I run/ref_pipeline.py && node web/det/test/check_pipeline.mjs  # web/pipeline.js vs PaddleOCR's code
node web/det/test/run_e2e.mjs [tiny|small] [page]                       # real det + rec through pipeline.js
```

## Files

| file | what |
|---|---|
| `models/<name>/manifest.json`, `weights.bin` | the op program and the packed weights (default scheme r63, committed) |
| `engine.js` | `DetNet.load(gpu, base)`, `det.run(rgbaOrGray, w, h, longSide = 1280)` returns `{prob, H, W, scale, validH, validW}`; `det.post` holds the DB thresholds from `inference.yml` |
| `kernels.js`, `gpu.js` | WGSL (conv, convT, maxpool, resize, squeeze-excite, add) and the small WebGPU helper |
| `test/check_det.js`, `test/run_det.mjs` | parity: WGSL against the NumPy executor on the same packed weights, prob map max abs 1e-3 |
| `test/check_pipeline.mjs` | `../pipeline.js` against `run/ref_pipeline.py` on the same probability maps, in Node |
| `test/run_e2e.mjs`, `test/check_e2e.js` | detector, `../pipeline.js` and the `web/rec` recognizer on a fixture page in headless Chromium (needs `run/export_rec.py`) |
| `test/make_pages.py`, `test/fetch_funsd.py` | the test pages (data only, never committed) |

`../pipeline.js` is the page pipeline (detector postprocess, crops, recognizer batches); see its header.

## Packed tensor kinds

`weights.bin` is little-endian with every blob 4-byte aligned; the manifest names each tensor's kind.

| kind | stored | value |
|---|---|---|
| `f32` | n float32 | as is |
| `f16` | n half floats | as is |
| `i8` | n int8 codes in [-L, L] (L <= 127), then one fp16 scale per output channel | code * scale |

The output channel is weight dim 0 for a conv and dim 1 for a ConvTranspose. Biases and every tensor under
1,024 elements are f32 in every scheme. The scheme `r<L>` makes the rest `i8` with scale = f16(max|row| / L)
and code = rint(w / scale), the arithmetic of `whisper-pack/run/quant.py`; `f16` and `f32` make it that kind.
Squeeze-excite fc layers are conv entries and pack the same way.

## What the export does

Single-consumer chains fold into one op: Conv + BatchNorm (none present in these graphs, the fold is in the
exporter) + bias + Relu, HardSigmoid, Sigmoid or exact GELU (`Div, Erf, Add, Mul, Mul`). Squeeze-excite blocks
(`ReduceMean` or `GlobalAveragePool`, Conv, Relu, Conv, HardSigmoid, Mul, optional residual Add) are one `se` op.
ConvTranspose absorbs its bias and activation. A `ReduceMean` that is not a squeeze-excite block is an export
error. The 94-op program per model is 57 conv, 13 se, 13 add, 6 resize, 2 concat, 2 convT, 1 maxpool.

## Numerics

Activations are f32 throughout; the squeeze-excite global mean is a Kahan-compensated sum (a plain f32 running
sum over ~80k pixels drifted about 1e-4 relative, and the gate carried that through the network). The
reference is the NumPy executor in `run/ref_det.py` on the dequantized weights; with `--scheme f32` that
executor matches fp32 onnxruntime to 1e-3 (the check that the executor itself is right). Packing changes the
map: see `results/det_parity.json` (`pack_vs_ort_*`) for how much, and `results/pipeline_parity.json`
(`pack_vs_fp32`) for what it does to the boxes.

Preprocessing: the long side is resized to `longSide` (up or down), BGR order, `(p/255 - mean[c]) / std[c]` with
mean and std in BGR order, zero-padded to multiples of 32 after normalisation.
