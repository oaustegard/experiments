# web/det: PP-OCRv6 text-line detector on WebGPU

Port of PaddlePaddle's PP-OCRv6 det (tiny and small, Apache-2.0) to hand-written WGSL, so the demo
can find text lines without Indra OCR's weights.

```
python3 -I run/export_det.py                        # ONNX -> web/det/models/{tiny,small}/
python3 -I run/ref_det.py                           # NumPy executor vs onnxruntime -> results/det_parity.json
python3 -I web/det/test/make_pages.py               # synthetic pages -> data/det_pages/ (git-ignored)
python3 -I web/det/test/fetch_funsd.py              # FUNSD test forms -> data/det_pages/ (research use only)
node web/det/test/run_det.mjs                       # WGSL in headless Chromium (SwiftShader) vs onnxruntime
```

The verify run is `python3 run/ref_det.py && node web/det/test/run_det.mjs` (exit 0). Under
SwiftShader one page takes about 45 s (tiny) and 115 s (small) on the 4-core host used here, so the
full run is about 25 minutes; `node web/det/test/run_det.mjs tiny` runs one model.

## Files

| file | what |
|---|---|
| `models/<name>/manifest.json`, `weights.bin` | the op program and packed weights (written by export; `models/small/weights.bin` is git-ignored) |
| `engine.js` | `DetNet.load(gpu, base)`, `det.run(rgbaOrGray, w, h, longSide=1280)` returns `{prob, H, W, scale, validH, validW}` |
| `kernels.js` | WGSL: conv (+bias, +activation), convT, maxpool, concat copy, nearest resize, squeeze-excite (mean, scale), add |
| `test/check_det.js` | in-page parity check against `data/fixtures/det_<page>/<model>.ort.f32` |
| `test/run_det.mjs` | Chromium runner; merges WGSL numbers into `results/det_parity.json` |
| `test/make_pages.py`, `test/fetch_funsd.py` | the test pages (data only, never committed) |

## What the export does

Single-consumer chains fold into one op: Conv + BatchNorm (none present in these graphs, the fold
is in the exporter) + bias + Relu, HardSigmoid, Sigmoid or exact GELU (`Div, Erf, Add, Mul, Mul`).
Squeeze-excite blocks (`ReduceMean` or `GlobalAveragePool`, Conv, Relu, Conv, HardSigmoid, Mul,
optional residual Add) are one `se` op. ConvTranspose absorbs its bias and activation. There is no
LayerNorm in these graphs; a `ReduceMean` that is not a squeeze-excite block is an export error.

The 94-op program per model is 57 conv, 13 se, 13 add, 6 resize, 2 concat, 2 convT, 1 maxpool.

## Numerics

Each conv weight is stored as two f16 planes, hi = f16(w) and lo = f16(w - hi). `DetNet.load`
dequantises them once to f32 (hi + lo, rounded to f32; the same value run/ref_det.py uses), so the
conv kernels read plain f32. One f16 plane was measured first: about 2e-2 max error on the
probability map, which compounds over ~90 ops and misses the 1e-3 budget. The second plane costs 2x
weight bytes on disk (tiny 1.7 MB, small 9.8 MB). Activations are f32 throughout.

The squeeze-excite global mean is a Kahan-compensated sum. A plain f32 running sum over ~80k pixels
drifted about 1e-4 relative, and the gate carried that through the network (funsd_1 at 4.1e-3 before
the fix, 4.0e-4 after).

The onnxruntime reference uses `ORT_ENABLE_BASIC`. At full optimisation onnxruntime fuses the GELU
chains into its own kernels, which moves the heading-table page by about 1.8e-3 (recorded as
`ort_all_opt_max_abs_vs_ref` in the results).

Preprocessing: the long side is resized to `longSide` (up or down), BGR order, `(p/255 - mean[c]) /
std[c]` with mean and std in BGR order, zero-padded to multiples of 32 after normalisation.
