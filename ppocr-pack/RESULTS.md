# RESULTS: PP-OCRv6 packed to int8 and run in a browser tab

PaddlePaddle's PP-OCRv6 is a pair of small neural networks that read text from an image: a detector that draws a
box around each line, and a recognizer that reads each boxed line. This experiment packs the weights of the tiny
and small versions of both to 8-bit integers (one scale per output channel), loads them in a browser tab and runs
them on hand-written WebGPU shaders, with no onnxruntime. Live page:
[austegard.com/experiments/ppocr-pack/web/](https://austegard.com/experiments/ppocr-pack/web/). The question: how
small does the pair get, what does that cost in accuracy, and does the in-browser engine reproduce the reference?

## Answer

Tiny (detector plus recognizer) ships as 1.57 MB gzipped, 3.99 times smaller than the upstream ONNX files, and small as 6.26 MB, 4.96 times smaller. <!-- @evidence[[results/packs.json#totals.tiny.upstream_over_shipped_gzip]] --><!-- @evidence[[results/packs.json#totals.small.upstream_over_shipped_gzip]] -->

The accuracy cost is below what 400 real form lines can resolve: the recognizer's character error rate rises by 0.08% (tiny) and 0.30% (small), and both 95% intervals include zero. <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.real.delta]] --><!-- @evidence[[results/sweep.json#models.small.rec.arms.r127.real.delta]] -->

It is not free: 91% of the tiny recognizer's real lines decode to exactly the fp32 text. <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.real.identical]] -->

The browser engine gives the same text as a NumPy reference on the same packed weights for all 20 test lines of the tiny recognizer. <!-- @evidence[[results/rec_parity.json#models.tiny.wgsl_packed.same_text]] -->

What would change this: a larger or more varied real-line set, since the intervals hide a cost of up to 0.4% (tiny) or 0.7% (small). Mac timing is not measured yet.

## Findings

1. **The pair shrinks about fourfold to fivefold.** Gzipped bytes shipped per model: tiny detector 454,259, tiny recognizer 1,110,974, small detector 1,528,683, small recognizer 4,733,153 (upstream 1.78, 4.46, 9.88 and 21.16 MB). The raw files total 1.7 MB (tiny) and 8.1 MB (small). (`run/packs_sizes.py`) <!-- @evidence[[results/packs.json#models.tiny_det.shipped_gzip_bytes]] --> <!-- @evidence[[results/packs.json#models.tiny_rec.shipped_gzip_bytes]] --> <!-- @evidence[[results/packs.json#models.small_det.shipped_gzip_bytes]] --> <!-- @evidence[[results/packs.json#models.small_rec.shipped_gzip_bytes]] -->
2. **8-bit weights (codes within plus or minus 127) cost no measurable recognizer accuracy.** Tiny: +0.02% absolute character error on 3,600 synthetic Latin lines and +0.08% on 400 real form lines (interval -0.25 to +0.42). Small: +0.30% on the form lines (-0.03 to +0.68). The intervals include zero, so the claim is "not measurable at this sample size", not "zero". <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.synth_latin.delta]] --> <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.real.delta]] --> <!-- @evidence[[results/sweep.json#models.small.rec.arms.r127.real.delta]] -->
3. **A tiny recognizer has a cliff between 8 and 7 bits.** Codes within plus or minus 63 add 2.13% absolute character error on the real lines (interval +1.38 to +2.92) and leave 78% of lines unchanged. <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r63.real.delta]] --> <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r63.real.identical]] -->
4. **The small recognizer is less sensitive but still fails at 63.** It adds 0.85% on the real lines (interval +0.05 to +1.60); plus or minus 31 is worse on both models. <!-- @evidence[[results/sweep.json#models.small.rec.arms.r63.real.delta]] -->
5. **The small detector tolerates 7 bits and the tiny one does not.** Box F1 against the fp32 detector's boxes: tiny 0.993 at plus or minus 127 and 0.981 at 63; small 1.000 and 0.999. The small detector ships at 63 and the tiny one at 127. <!-- @evidence[[results/sweep.json#models.tiny.det.arms.r127.box_f1]] --> <!-- @evidence[[results/sweep.json#models.tiny.det.arms.r63.box_f1]] --> <!-- @evidence[[results/sweep.json#models.small.det.arms.r127.box_f1]] --> <!-- @evidence[[results/sweep.json#models.small.det.arms.r63.box_f1]] -->
6. **A 4-bit output layer costs accuracy.** Each recognizer's final layer (80 by 6,906 in tiny) is about half of its weights and is also the table that maps features to characters. At 4 bits tiny gains +0.09% absolute error on the Latin lines (interval +0.06 to +0.12) and saves 0.25 MB gzipped (839,241 bytes against 1,086,788). It is not shipped. <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.t4r127.synth_latin.delta]] --> <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.t4r127.size.gzip9_bytes]] --> <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.size.gzip9_bytes]] -->
7. **The WGSL detector matches the NumPy reference to 3.0e-05.** Probability-map maximum absolute difference on the same packed weights, worst tiny page 3.0e-05 and worst small page tested 1.2e-05, against a 0.001 limit. <!-- @evidence[[results/det_parity.json#models.tiny.pages.funsd_1.wgsl_max_abs]] --> <!-- @evidence[[results/det_parity.json#models.small.pages.funsd_1.wgsl_max_abs]] -->
8. **The page pipeline reproduces PaddleOCR's postprocess exactly.** The JavaScript box finder, written without OpenCV, gives the same boxes as OpenCV and pyclipper on five pages per model: corners differ by 0 px. <!-- @evidence[[results/pipeline_parity.json#js_vs_python.tiny.heading_table.max_corner_px]] -->
9. **The whole chain reads a page.** Detector, pipeline and recognizer read all 30 lines of a synthetic heading-and-table page in headless Chromium on a software GPU. Timings from that run mean nothing. <!-- @evidence[[results/pipeline_e2e.json#tiny/heading_table.lines]] -->
10. **Mac timing: PENDING.** Oskar opens the page with `?log=1` on his Mac and runs tiny and small on both samples; the per-stage times land in the GPU bridge logs (`gpu_bridge.py logs --page ppocr-pack`). This item is replaced with the numbers.

## Method

Models: PaddlePaddle/PP-OCRv6 `tiny` and `small`, detector and recognizer, fp32 ONNX from the Hugging Face Hub
(`run/fetch_models.py`), Apache-2.0.

Packing (`run/export_det.py`, `run/export_rec.py`): BatchNorm folds into the preceding convolution. Each convolution
weight and each matrix weight becomes int8 codes in plus or minus L with one fp16 scale per output channel, scale
= max|w| / L. Tensors under 1,024 values, biases and norm parameters stay fp32. The engine dequantizes to f32 once at
load and the kernels read f32. Shipped: tiny detector L=127, tiny recognizer 127, small detector 63, small
recognizer 127. Per-model sizes in `results/packs.json`.

Accuracy sweep (`run/sweep.py`, `run/quant.py`): fake-quantize the ONNX initializers and run onnxruntime on CPU.
Arms: fp16, L=127, 63, 31, and for recognizers a 4-bit output layer (groups of 32, fp16 scale per group) with the
rest at 63 or 127. Recognizer metric: mean per-line character error rate and the share of lines decoding to
fp32's exact text, on 3,600 synthetic Latin lines (English and French), 400 real form lines (FUNSD crops) and 400
receipt lines (SROIE); intervals are 95% paired bootstrap over lines, 4,000 resamples. Detector metric: box F1
against fp32's boxes (IoU 0.5) and probability-map difference on 20 pages (3 synthetic, 17 FUNSD). Decision rule:
the smallest arm whose recognizer error change has an interval containing zero (or at least 99% identical text)
and whose detector F1 is at least 0.99. Synthetic Greek lines are left out of the gate because they read at 55 to
71% error even in fp32, so any perturbation reshuffles them.

Engine parity: WGSL against a NumPy executor over the same manifest, in headless Chromium on SwiftShader. Twenty
synthetic line crops per recognizer (widths 82 to 1,412 px, several scripts), 4 tiny and 2 small detector pages
(long side 1,280). Page pipeline against `run/ref_pipeline.py` (OpenCV, pyclipper) on the same probability maps.

Page: `web/index.html`, `web/app.js`. Smoke test: `web/test/smoke_page.mjs`.

## Log

### Round 1, 2026-11-24: sweep, engines, page

Asked: pack PP-OCRv6 tiny, then small, and run both in a browser with custom kernels if needed. Three agents
worked in parallel from separate git worktrees: a CPU quantization sweep, a WGSL recognizer (including small's
attention blocks and layer norms), and a packed detector plus page pipeline built on `tesseract-gpu/web/det`.
Custom WGSL rather than onnxruntime-web, because that runtime's wasm is several MB, larger than a packed tiny model.

Sweep outcome: the smallest passing arms are tiny detector 127, tiny recognizer 127, small recognizer 127, small
detector 63. The 127 recognizer arms pass only through the interval rule; text-identical shares stay under 99%
(91 to 98%). Added after the first results: a 4-bit head with the rest at 127, because the 63 arms had failed and left
the head's 4 bits untested; it fails on tiny. Over all synthetic lines including Greek, tiny has no passing arm
(fp16 itself "fails" by improving by 0.008 points) and small picks fp16, which is why Greek is excluded from the gate.

Parity caveat: the engine parity files (`rec_parity.json`, `det_parity.json`, `pipeline_parity.json`,
`pipeline_e2e.json`) were produced on the first default packs (scheme 63) before the sweep's choice was exported.
The kernels do not depend on the code range, and the shipped packs went through the same exporter, but the parity
runs were not repeated on the shipped files. The page smoke test below ran on the shipped packs.

Page smoke test (headless Chromium, SwiftShader, long side 640): tiny read both bundled samples with no console
errors (the cafe notice into 8 lines, all correct; the receipt into 17 lines with one "O" read as "0"); small
read the receipt into 17 lines, all correct, with no console errors. Not done: PDF input.

Shipped sizes were recorded by `run/packs_sizes.py` into `results/packs.json`; the first committed tiny
detector (scheme 63) was replaced by the 127 pack.
