# RESULTS: PP-OCRv6 packed to int8 and run in a browser tab

PaddlePaddle's PP-OCRv6 reads text from an image with two small neural networks: a detector that draws a box
around each line of text, and a recognizer that reads the text in each box. PaddlePaddle publishes it in three
sizes. This experiment takes the two smallest, tiny (1.5 million parameters for detector and recognizer together)
and small (7.7 million), stores their weights as 8-bit integers, and runs them in a browser tab on hand-written
WebGPU shaders instead of onnxruntime. Try it at
[austegard.com/experiments/ppocr-pack/web/](https://austegard.com/experiments/ppocr-pack/web/): drop an image or use
a bundled sample, choose tiny or small, press Run. It needs a browser with WebGPU. To use the engine in another page,
see [`web/README.md`](web/README.md).

The question: how much smaller is the download, what does that cost in accuracy, and does the browser engine
compute what the reference computes?

## Answer

The download is 3.6 (tiny) to 4.2 (small) times smaller than PaddlePaddle's ONNX files, both gzipped: 1.57 MB against 5.69 MB, and 6.26 MB against 26.0 MB. <!-- @evidence[[results/packs.json#totals.tiny.upstream_gzip_over_shipped_gzip]] --><!-- @evidence[[results/packs.json#totals.small.upstream_gzip_over_shipped_gzip]] --> No onnxruntime download is needed on top (Finding 1).

On 400 lines from real scanned forms (FUNSD), packed tiny misreads 7.29% of characters against 7.20% for the original weights, and small 5.59% against 5.29%. <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.real.cer]] --><!-- @evidence[[results/sweep.json#models.small.rec.arms.r127.real.cer]] --> Both differences have 95% intervals that include zero; the detectors find the same lines (Finding 5).

The browser engine matches a NumPy implementation running the same packed weights: the same text on all 20 test lines for both recognizers (Finding 7).

Speed on a real GPU is not measured yet. More real lines would narrow the intervals, which still allow a cost of up to 0.42% (tiny) or 0.68% (small) absolute.

## Findings

1. **Each model shrinks about fourfold.** (Round 2) Gzipped bytes shipped against the upstream ONNX file gzipped: tiny detector 454,259 against 1,589,489; tiny recognizer 1,110,974 against 4,105,137; small detector 1,528,683 against 7,530,991; small recognizer 4,733,153 against 18,481,888. <!-- @evidence[[results/packs.json#models.tiny_det.shipped_gzip_bytes]] --> <!-- @evidence[[results/packs.json#models.tiny_rec.shipped_gzip_bytes]] --> <!-- @evidence[[results/packs.json#models.small_det.shipped_gzip_bytes]] --> <!-- @evidence[[results/packs.json#models.small_rec.shipped_gzip_bytes]] --> The small detector shrinks most because it uses ±63 codes, which compress better. The engine's code adds 29,346 bytes gzipped, where running the ONNX files would add onnxruntime-web's WebGPU build, 5,171,684 bytes gzipped. <!-- @evidence[[results/packs.json#engine_js_gzip_bytes]] --> <!-- @evidence[[results/packs.json#onnxruntime_web_1_20_1.jsep_wasm_gzip_bytes]] -->
2. **8-bit weights cost no recognizer accuracy these samples can resolve.** (Round 1) Character error on 3,600 synthetic English and French lines: tiny 0.73% against 0.71% for the original, small 0.38% against 0.38%. <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.synth_latin.cer]] --> <!-- @evidence[[results/sweep.json#models.small.rec.arms.r127.synth_latin.cer]] --> On the 400 real form lines the differences are +0.08% absolute for tiny (interval −0.25 to +0.42) and +0.30% for small (−0.03 to +0.68). <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.real.delta]] --> <!-- @evidence[[results/sweep.json#models.small.rec.arms.r127.real.delta]] --> The intervals include zero, so the claim is that the cost is not measurable at this sample size, not that it is zero.
3. **The tiny recognizer breaks at ±63.** (Round 1) Codes in ±63 add 2.13% absolute character error on the real lines (interval +1.38 to +2.92) and leave 78% of lines unchanged. <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r63.real.delta]] --> <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r63.real.identical]] -->
4. **The small recognizer is less sensitive but also fails at ±63.** (Round 1) It adds 0.85% absolute on the real lines (interval +0.05 to +1.60); ±31 is worse on both models. <!-- @evidence[[results/sweep.json#models.small.rec.arms.r63.real.delta]] -->
5. **The small detector tolerates ±63 and the tiny one does not.** (Round 1) Box F1 against the original detector's boxes: tiny 0.993 at ±127 and 0.981 at ±63; small 1.000 and 0.999. So the small detector ships at ±63 and the other three models at ±127. <!-- @evidence[[results/sweep.json#models.tiny.det.arms.r127.box_f1]] --> <!-- @evidence[[results/sweep.json#models.tiny.det.arms.r63.box_f1]] --> <!-- @evidence[[results/sweep.json#models.small.det.arms.r127.box_f1]] --> <!-- @evidence[[results/sweep.json#models.small.det.arms.r63.box_f1]] -->
6. **A 4-bit output layer costs accuracy.** (Round 1) Each recognizer's final layer (80 by 6,906 in tiny) holds about half its weights and maps features to characters. At 4 bits tiny gains +0.09% absolute error on the synthetic lines (interval +0.06 to +0.12) to save 0.25 MB gzipped (839,241 bytes against 1,086,788), so it is not shipped. <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.t4r127.synth_latin.delta]] --> <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.t4r127.size.gzip9_bytes]] --> <!-- @evidence[[results/sweep.json#models.tiny.rec.arms.r127.size.gzip9_bytes]] -->
7. **The WebGPU engines compute what the NumPy references compute.** (Round 2) On the shipped packs, the recognizers give the same text as NumPy on all 20 test lines for tiny and for small, with the largest probability difference 4.4e-05 <!-- @evidence[[results/rec_parity.json#models.tiny.wgsl_packed.max_abs]] --> (tiny) and 1.1e-04 <!-- @evidence[[results/rec_parity.json#models.small.wgsl_packed.max_abs]] --> (small). The detector probability maps differ by at most 3.6e-05 (tiny) and 1.2e-05 (small), against a limit of 0.001 set before the runs. <!-- @evidence[[results/rec_parity.json#models.tiny.wgsl_packed.same_text]] --> <!-- @evidence[[results/rec_parity.json#models.small.wgsl_packed.same_text]] --> <!-- @evidence[[results/det_parity.json#models.tiny.pages.funsd_1.wgsl_max_abs]] --> <!-- @evidence[[results/det_parity.json#models.small.pages.funsd_1.wgsl_max_abs]] -->
8. **The page's box finder gives PaddleOCR's boxes to the pixel.** (Round 1) The JavaScript that turns the detector's output into line boxes and crops, written without OpenCV, gives the same boxes as PaddleOCR's OpenCV and pyclipper code on five pages per model, fed the same probability maps: corners differ by 0 px. <!-- @evidence[[results/pipeline_parity.json#js_vs_python.tiny.heading_table.max_corner_px]] -->
9. **Detector, box finder and recognizer read a whole page in the tab.** (Round 1) All 30 lines of a synthetic page with a heading and a table, in headless Chromium on a software GPU. <!-- @evidence[[results/pipeline_e2e.json#tiny/heading_table.lines]] -->

## Method

Models: PaddlePaddle/PP-OCRv6 `tiny` and `small`, detector and recognizer, fp32 ONNX from the Hugging Face Hub
(`run/fetch_models.py`), Apache-2.0. Parameters: tiny detector 0.43 M, recognizer 1.10 M; small detector 2.45 M,
recognizer 5.27 M.

Packing (`run/export_det.py`, `run/export_rec.py`): BatchNorm folds into the preceding convolution. Each convolution
and matrix weight becomes integer codes in ±L, stored one byte each, with one fp16 scale per output channel,
scale = max|w| / L. ±127 is a full 8-bit code; ±63 and ±31 use fewer distinct values and so compress better under
gzip. Tensors under 1,024 values, biases and norm parameters stay fp32. The engine turns the codes back into f32
once at load, so the shaders read f32. Shipped: ±127 for the tiny detector and both recognizers, ±63 for the small
detector (`results/packs.json`). Upstream sizes are compared gzipped against gzipped.

Accuracy sweep (`run/sweep.py`, `run/quant.py`): rewrite the ONNX weights with the packed values and run
onnxruntime on CPU. Arms: fp16, ±127, ±63, ±31, and for recognizers a 4-bit output layer (codes in ±7, groups of 32
sharing an fp16 scale) with the rest at ±63 or ±127. Recognizer metric: character error rate (edit distance over
reference length, per line, averaged) and the share of lines whose text is identical to the original model's,
on 3,600 synthetic English and French lines, 400 real form lines (FUNSD crops) and 400 receipt lines (SROIE);
intervals are 95% paired bootstrap over lines, 4,000 resamples. Detector metric: box F1 against the original
model's boxes (a match needs IoU of at least 0.5) on 20 pages (3 synthetic, 17 FUNSD). Decision rule: the smallest
arm whose recognizer error change has an interval containing zero (or at least 99% identical text) and whose
detector F1 is at least 0.99. Synthetic Greek lines were left out of the decision because the original models
read them at 55 to 71% character error, so any change in the weights reshuffles them.

Engine checks, in headless Chromium on SwiftShader (a software implementation of a GPU, so correct but slow):
each WebGPU engine against a NumPy executor that runs the same op program on the same packed weights. Recognizers:
20 synthetic line crops (widths 82 to 1,412 px; English, French, German, Spanish, Chinese). Detectors: 5 tiny pages
and 2 small pages at long side 1,280. Box finder: `web/pipeline.js` against `run/ref_pipeline.py` (OpenCV,
pyclipper) on the same probability maps, 5 pages per model, no GPU involved.

Page: `web/index.html`, `web/app.js`; smoke test `web/test/smoke_page.mjs`. Porting a server PaddleOCR app to this engine: `web/README.md`.

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

### Round 2, 2026-11-24: parity on the shipped packs, writeup pass

The Round 1 parity files came from the first default packs (±63). Rerun on the shipped files: recognizers ±127,
tiny and small, 20 of 20 lines with the same text as NumPy, max probability difference 4.4e-05 and 1.1e-04; tiny
detector ±127 on all five pages, max 3.6e-05. The small detector already shipped at ±63, so its two pages stand.
Sizes are now compared gzipped against gzipped (the Round 1 ratios, 3.99 and 4.96, compared gzipped packs with raw
ONNX). A fresh reader's pass on RESULTS.md led to defining the code ranges, giving error rates with their
baselines, and adding the onnxruntime-web size. A porting recipe and a PaddleOCR-shaped result adapter
(`toPaddleResult`) went into `web/README.md`; the packed engine also now serves the PP-OCRv6 options on
austegard.com/web-utilities/anything-to-text.html when the browser has WebGPU.


### 2026-11-26: CPU speed against OCRia-nano-1's Latin recognizer

Asked (Oskar): what makes OCRia-nano-1 faster on a CPU than PP-OCRv6 small, and can PP small be made fast without
retraining? `run/cpu_speed.py` timed the recognizers on onnxruntime 1.32, one thread, one line per call, on the 400
FUNSD lines (`results/cpu_speed.json`):

| arm | ms per line | CER |
|---|---:|---:|
| PP small as published | 26.9 | 5.29% |
| head cut to blank + 458 Latin characters | 23.3 | 5.29% |
| that, with int8 MatMuls | 22.4 | 5.30% |
| PP small, lines padded to a multiple of 32 px instead of at least 320 | 21.8 | 5.49% |
| head cut and no 320 px minimum | 17.4 | 5.49% |
| PP tiny as published | 5.5 | 7.20% |
| OCRia-nano-1 rec_latin (int8) | 1.4 | 5.80% |

The CTC head over 18,710 characters is not where PP small's time goes: cutting it to Latin reads the same text on
399 of 400 lines and saves 13%. A profile of 100 lines puts 67% of the time in convolutions, 11% in the
attention MatMuls and 8% in GELU. The no-retraining changes together give 1.5x; OCRia's recognizer is 19x faster
at 0.5 points more CER, from a smaller, depthwise-separable, all-convolution net on 32 px gray lines.
