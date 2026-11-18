# indra-browser: Indra OCR as a web page

Datalab's [Indra OCR](https://huggingface.co/datalab-to/indra-ocr) (indra-ocr 1.1.2) turns a scanned page or a PDF into
markdown with about 12M parameters: a line detector, CTC line recognizers per script group, n-gram word models for
unsure lines, an orientation classifier and a small reading-order transformer. This experiment runs all of it in a
browser tab on WebGPU, with no onnxruntime: hand-written WGSL kernels, the package's Python pipeline ported to JS, and
the weights packed to as few bytes as the benchmark allows. Page: https://austegard.com/experiments/indra-browser/web/
(`web/index.html`). Oskar Austegard, who owns this repo, asked for it.

## Answer

Can Indra run in a browser tab, how small can its download get, and what does packing cost? It runs, and
with the same weights the browser's markdown equals the Python package's on the pages compared:
15 of 15 without the word model. <!-- @evidence[[results/summary.json#parity.nolm.same_markdown]] -->
Accuracy is Omni Parse Bench's headline (the mean of its text, tables and layout pass rates, in percent) on 2,937
pages read without the PDF text layer. fp32 weights score 61.51 and fp16 61.55; 4-bit collapses to 49.56. <!-- @evidence[[results/summary.json#s4c.headline]] -->
In fp16 the detector, Latin recognizer, reading-order, orientation and script models come to
4.05 MB gzipped. <!-- @evidence[[results/summary.json#bytes_f16.core]] -->
The demo page and the anything-to-text converter ship a pack with 8-bit recognizers and
reading-order model instead, which scores 61.36 at 2.24 MB (Log, Round 3). <!-- @evidence[[results/summary.json#bytes_mix8.core]] -->
The n-gram word models pack further: 2-bit values keep their handwriting gain, and the Latin table goes from 15.7 MB
(npz) to 2.54 MB gzipped, fetched only when a line needs it. <!-- @evidence[[results/summary.json#bytes_f16.lm-base]] -->
Once the models are loaded, a page takes 54 to 97 ms on Oskar's Mac (Chrome 157, Apple Metal-3), with every network on the GPU (Log, Round 2). <!-- @evidence[[results/summary.json#/mac_indra8/olmo_old_scans-87-p1/gpu_order/warm_ms]] -->

## Findings

1. **Weights hold at fp16 and 8-bit, and break below.** Against fp32's 61.51: fp16 61.55 (+0.04, CI −0.03 to
   +0.11); fp16 for the three small models (detector, orientation, script) with 8-bit recognizers and order model 61.36 (−0.15, CI −0.37 to +0.04);
   8-bit everywhere 61.18 (−0.34, CI −0.75 to +0.04); 4-bit 49.56 (−11.95; its tables score falls from fp32's 52.78
   to 33.36); and between them 6-bit 60.40 (−1.11, CI −1.87 to −0.48). <!-- @evidence[[results/summary.json#s6c.headline]] -->
   A scale per 32 weights lifts 4-bit only to 55.43. <!-- @evidence[[results/summary.json#s4g32.headline]] --> These conv nets lose accuracy far faster per bit than the
   transformer in `gliner-browser` (a packed named-entity model in this repo), which held 4-bit token tables at no
   cost (Round 1).
2. **The word model matters for handwriting and not overall.** With the word model off, the headline moves +0.11 (CI −0.17 to +0.40)
   while the 833 handwriting text and table tests drop from 25.3 to 22.0 (no interval computed). <!-- @evidence[[results/summary.json#handwriting.lm-off]] -->
   The longest orders carry the gain: the Latin table goes to 5-grams, and keeping up to 3-grams scores 22.4,
   up to 4-grams 23.5 (Round 1).
3. **Its values survive 2 bits and its keys a trie.** All orders with 2-bit per-table k-means codebooks score 61.64
   and handwriting 25.0 (3-bit: 61.58, 25.1). <!-- @evidence[[results/summary.json#handwriting.lm-q2]] -->
   Every history of n tokens is itself a stored (n−1)-token entry, except the history of n line-starts. So the
   file stores histories as one bit per lower-order entry, and entries as a count per history plus the gaps between
   its token ids: 32% fewer key bytes than gap-coded sorted keys (Round 1).
4. **The browser reproduces the package on nearly every page.** With the same weights, every line's text and box and the
   markdown match indra-ocr on the same pixels: 1452 lines on 15 pages without the word model, <!-- @evidence[[results/summary.json#parity.nolm.lines]] -->
   10 pages with it, and 9 pages with the line crops on the GPU as the page ships them (Round 1). With the 8-bit pack
   the markdown matches on 59 of 60 pages; <!-- @evidence[[results/summary.json#parity.mix8.same_markdown]] -->
   on the 60th one line reads "N.V." where the package reads "N.Y.", and the 16-bit pack reads it the same way (Round 3). Two ports had to
   match OpenCV 5 and NumPy to the bit: `warpAffine` samples through the inverse matrix in float32, and NumPy's
   `arange` computes its step as `(start + step) − start`, which decides whether some pages cross the 0.3° deskew
   threshold.
5. **The GPU does tens of milliseconds of a page's work; JS around it was the cost.** A 30-megapixel scan now takes
   112 ms warm, its line crops 2 ms. <!-- @evidence[[results/summary.json#mac_indra7.olmo_old_scans-87-p1.warm_ms]] -->
   In an earlier build the handwritten-letter test page's kernels summed to 59.3 ms; <!-- @evidence[[results/summary.json#mac_indra5_gpu_kernel_ms.synth_copies-images_sv5a6924ae-p1]] -->
   the rest of the page's time went to JS around them (per-batch submits, line crops, the skew search). Those parts
   then moved to the GPU or into one submit (Round 1). The deskew and downscales alone took the scan from 1,820 ms
   to 416 ms. <!-- @evidence[[results/summary.json#mac_indra3.olmo_old_scans-87-p1.gpu_ms]] -->

## Method

**Benchmark.** [Omni Parse Bench](https://huggingface.co/datasets/datalab-to/omni_parse_bench) 1.6.0's own scorer on
all 2,937 pages, headline = mean of the text, tables and layout families. Pages are prerendered once to grayscale
(`run/prerender.py`: pdfium at 200 DPI as indra-ocr's Converter renders them) so every arm and the browser read the
same pixels. indra-ocr runs in OCR mode (no PDF text layer), math off, page headers and footers kept, through
`run/opb_eval.py` inside `scripts/jail.sh` (no network, no secrets) since it imports code nobody here wrote. The
benchmark's indra-ocr adapter maps the markdown and blocks to its outputs. Runs are deterministic: a repeat of fp32
gave the same markdown on all 2,937 pages. Intervals are 95% bootstrap over documents (1,000 resamples), paired
against fp32; differences are taken before rounding, so a table's two rounded scores can differ from its Δ by 0.01.

**Arms.** `run/quant.py` rounds each Conv, MatMul and Gemm weight of at least 1,024 values (fp16 for the rest) and
`run/make_arm.py` writes the rounded floats back into the ONNX files, so ONNX Runtime computes exactly what the packed
file decodes to. Schemes: fp16; symmetric per-output-channel 8-bit (absmax), 6-, 5- and 4-bit (MSE-searched clip);
4-bit with a scale per 32. `run/lmpack.py` prunes n-gram orders and replaces values with per-table k-means
codebooks. indra-ocr re-reads a line with a CTC beam search under the word model only when its confidence is between
0.5 and 0.9, so n-gram arms run only the 2,478 pages where that happened and take the other pages from fp32. The
highest orders are 5 (Latin), 4 (world) and 3 (CJK).

**Browser.** `run/export.py` turns each conv net into an op program and packs every tensor (fp16, or b-bit codes in a
bit stream with fp16 scales; bit-packing gzips 8 to 15% smaller than a byte per code). `web/` holds the WebGPU runner
(`gpu.js`, `nets.js`: NHWC conv, tiled 1x1 matmul, CTC top-8, the recognizer's row attention), the deskew, downscale
and line crops on the GPU (`deskew.js`, `crops.js`), the pipeline (`indra.js`, `layout.js`, `text.js`, `lm.js`,
`imgproc.js`) and the page. `run/ref.py` is a NumPy forward pass that matches ONNX Runtime to 1e-5;
`web/test/check_engine.mjs` checks the WGSL nets against it, `web/test/check_pages.mjs` runs the pipeline on pages
for `run/compare_pages.py`. Mac timings come from the GPU Bridge, this repo's job runner on a browser tab of Oskar's Mac
(Chrome 157, Apple Metal-3): `results/bridge_indra-*_mac.json`. Each bridge job has an id (`indra-1` to `indra-7`, one per build
timed). Accuracy is measured in Python with the rounded weights; the browser's own score follows from the parity checks, not from a separate run.

Arm names in `results/`: `f16`, `i8c` (8-bit per channel), `s6c`/`s5c`/`s4c` (6/5/4-bit per channel), `s4g32`
(4-bit, a scale per 32), `mix8` (fp16 detector, orientation and script models, 8-bit recognizers and order model);
`lm-off`, `lm-oN` (orders up to N), `lm-top1` (longest order dropped), `lm-qB` (all orders, B-bit values).

**Running it.** Scoring an arm: `sh run/sweep.sh ARM` (needs the jail's dependencies, see `run/opb_eval.py`).
Browser checks: `node web/test/check_engine.mjs PACK FIXTURES` (fixtures from `run/make_fixtures.py`) and
`node web/test/check_pages.mjs PACK GRAY_DIR PAGES.txt OUT`, then `python3 run/compare_pages.py OUT RUN_DIR`.
Packing: `python3 run/export.py --src MODEL_DIR --out web/models/f16 --scheme f16 --lm base=0:2,world=0:2,cjk=0:2`.

## Log

### Round 1 (2026-11-18): port, sweep, pack

Asked (Oskar): pack Indra, the CPU OCR model inside Datalab's `marker` converter, as a web page like this repo's
Ettin and GLiNER conversions, at maximum Pareto byte efficiency, with the
n-gram tables heavily quantized and models loaded on demand.

| arm | headline | Δ vs fp32 | 95% CI | text | tables | layout | core gzip |
|---|---|---|---|---|---|---|---|
| fp32 | 61.51 | | | 40.32 | 52.78 | 91.43 | 8.9 MB of ONNX, uncompressed |
| fp16 | 61.55 | +0.04 | −0.03 to +0.11 | 40.36 | 52.78 | 91.51 | 4.05 MB |
| fp16 small models, 8-bit recognizers and order (mix8) | 61.36 | −0.15 | −0.37 to +0.04 | 40.33 | 52.28 | 91.47 | ~2.2 MB |
| 8-bit | 61.18 | −0.34 | −0.75 to +0.04 | 40.10 | 51.88 | 91.55 | 2.06 MB |
| 6-bit | 60.40 | −1.11 | −1.87 to −0.48 | 39.31 | 50.84 | 91.05 | |
| 5-bit | 58.51 | −3.00 | | | | | |
| 4-bit, a scale per 32 | 55.43 | −6.08 | | 32.16 | 43.67 | 90.47 | |
| 4-bit | 49.56 | −11.95 | −13.10 to −10.71 | 28.02 | 33.36 | 87.31 | 1.02 MB |

| n-gram arm (fp32 weights) | headline | Δ | handwriting |
|---|---|---|---|
| full (fp32 values) | 61.51 | | 25.3 |
| off | 61.63 | +0.11 | 22.0 |
| orders ≤ 2 | 61.34 | −0.17 | 22.2 |
| orders ≤ 3 | 61.41 | −0.10 | 22.4 |
| longest order dropped | 61.51 | −0.01 | 23.5 |
| all orders, 4-bit values | 61.48 | −0.03 | 25.5 |
| all orders, 3-bit values | 61.58 | +0.06 | 25.1 |
| all orders, 2-bit values | 61.64 | +0.13 | 25.0 |

Shipped pack `web/models/f16`, gzipped: core 4.05 MB, Latin word model 2.54 MB, right-to-left group 2.18 MB, world
group 4.49 MB, CJK group 9.24 MB (each group: its recognizer, vocabulary and word model), page code 0.05 MB. The 5-bit and grouped 4-bit arms were still running at write
time; 5-bit then scored 58.51 and 4-bit with a scale per 32 weights 55.43. Prediction before the sweep: 4-bit would cost under a point, as for gliner-browser; it cost 12.

Mac timings (GPU Bridge, Chrome 157, Apple Metal-3), warm, GPU deskew and downscale against the CPU versions:

| page | lines | GPU path | CPU path |
|---|---|---|---|
| Amharic page | 69 | 272 ms | 633 ms |
| table page | 141 | 405 ms | 792 ms |
| 30 MP handwritten scan | 18 | 416 ms | 1,820 ms |
| handwritten letter | 27 | 171 ms | 559 ms |

Two later bridge runs were 4 to 6 times slower across every stage, including plain-JS reading order, so their
absolute numbers are not comparable; their GPU-vs-CPU ratios and per-kernel profiles are. The last run, with
recognizer batches in one submit and line crops on the GPU (`results/bridge_indra-7_mac.json`), warm:

| page | lines | total | orientation | detection | recognition | of which crops | reading order |
|---|---|---|---|---|---|---|---|
| Amharic page | 71 | 159 ms | 39 | 20 | 59 | 1 | 42 |
| table page | 141 | 180 ms | 18 | 19 | 52 | 1 | 90 |
| 30 MP handwritten scan | 19 | 112 ms | 38 | 17 | 47 | 2 | 10 |
| handwritten letter | 28 | 71 ms | 16 | 12 | 28 | 1 | 15 |

What broke along the way: a
`warpAffine` that mirrored every rotation (sampled through the forward matrix), a skew threshold missed by `np.arange`
rounding, and a two-hour background limit that killed the first sweep chain partway through the weight arms.

### Round 2 (2026-11-18): reading order on the GPU

Asked (Oskar): move reading order to the GPU. The page's `order_ms` timer covers the reading-order transformer and
the decode after it, which clusters lines into blocks and orders the blocks (`decodeBlocksOrder` in `web/layout.js`).
Timed separately in headless Chromium in the container (software GPU, CPU-bound JS), the transformer took 252 ms and
the decode 7 ms of the 263 ms stage on the 141-line table page; the rest is line splitting before them.

`web/order.js` now runs the transformer on the GPU in one submission: input projection, four layers of layer norm,
4-head attention and a GELU feed-forward (GELU is a new activation in `gpu.js`'s tiled matmul kernel), then its five
output projections (order query and key, same-block query and key, block class) as one matmul, and a kernel that
forms the pairwise order and same-block scores from them. Only those two N x N score matrices and the class logits are
read back. The decode stays in JS. `ocr.cpuOrder = true` keeps the JS transformer.

Against the NumPy reference (`run/ref.py`, fixtures rebuilt from `web/models/f16`, the shipped 16-bit pack) the GPU
transformer's outputs differ by at most 2.3e-6 and the JS one's by 3.2e-6, both inside `check_engine.mjs`'s 1e-4
relative tolerance. On the 59 parity pages in headless Chromium, the first 30 finished before a 30-minute limit
stopped the run, and on all 30 (1,817 lines) CPU and GPU order gave identical markdown, lines and blocks.

On the Mac (GPU Bridge job `indra-8`, Chrome 157, Apple Metal-3) each page was read with the JS transformer three times
and then with the GPU one three times; the table gives the last full read of each. The job used the same pack files as
`indra-7`; every pack is dequantized to float32 at load, so the timings do not depend on its precision. "Same output"
means the two reads' markdown, lines and blocks were identical.

| page | lines | order, JS | order, GPU | page, JS order | page, GPU order | same output |
|---|---|---|---|---|---|---|
| Amharic page | 71 | 62.2 ms | 2.9 ms | 154 ms | 80 ms | yes |
| table page | 141 | 77.4 ms | 3.5 ms | 157 ms | 74 ms | yes |
| 30 MP handwritten scan | 19 | 9.2 ms | 1.4 ms | 104 ms | 97 ms | yes |
| handwritten letter | 28 | 13.5 ms | 1.3 ms | 71 ms | 54 ms | yes |

Reading order takes 3.5 ms on the table page, where the JS transformer and decode took 77.4 ms. <!-- @evidence[[results/summary.json#/mac_indra8/tables_derived-PMC10627924.1_doc-p4/gpu_order/order_ms]] -->
Page totals also move with the other stages from read to read, so the page columns differ by a few milliseconds from
the order columns' difference. With GPU order, recognition (26 to 52 ms) is the largest stage on every page and
orientation (12 to 34 ms) the next (`results/summary.json`, `mac_indra8`).

### Round 3 (2026-11-18): a smaller pack for anything-to-text

Asked (Oskar): ship the 8-bit option, `mix8` (8-bit recognizers and reading-order model, fp16 detector, orientation
and script models), plus a byte shuffle of the fp16 data, after checking parity; first for the anything-to-text
converter, then, at 0.15 points below fp32, for the demo page too.

The byte shuffle stores each fp16 array as its low bytes, then its high bytes (`"planes": 1` on a tensor's record), so
the sign-and-exponent bytes sit together and gzip better. On the fp16 pack it would cut the core files by 5 to 8%
(`rec.bin` 2.26 to 2.09 MB). On `mix8` it saves 9 KB (2.248 to 2.239 MB), because the recognizers' and order model's weights are
8-bit codes and only the three small models and the per-row scales remain fp16. It stays in the format: older packs
load unchanged, and the two layouts decode to identical weights (652 tensors, in `run/ref.py` and `web/pack.js`).

| pack, gzipped | core | Latin word model | English page (core, Latin word model, 0.05 MB page code) | world group | CJK group | right-to-left group |
|---|---|---|---|---|---|---|
| fp16 (`web/models/f16`) | 4.05 MB | 2.54 MB | 6.64 MB | 4.49 MB | 9.24 MB | 2.18 MB |
| mix8 (`web/models/mix8`) | 2.24 MB | 2.54 MB | 4.83 MB | 3.18 MB | 6.71 MB | 1.09 MB |

Accuracy, from the Round 1 sweep: headline 61.36 against fp16's 61.55 and fp32's 61.51 (difference from fp32 −0.15,
95% interval −0.37 to +0.04); the 833 handwriting text and table checks 25.6 against fp16's 25.8. `check_engine.mjs` on
the mix8 pack: every net within 1.1e-5 of the NumPy reference.

Parity, with a mix8 pack whose word-model values stay float32 so it matches the Python mix8 run: the browser's markdown
and line texts equal the package's on 59 of the 60 parity pages (5,209 lines). On the other page one line reads
"With N.V. Co. Clerk and Register" where the package reads "N.Y."; its box differs by 0.1 px. The same page with the
fp16 pack also reads "N.V." (the package: "N.Y."), with the GPU or the CPU for crops, deskew and order, so the
difference predates mix8 and was outside the Round 1 parity pages. Python's run with every model at 8 bits (`i8c`) reads "N.V." as well.
anything-to-text now loads `models/mix8/`; its smoke test (headless Chromium, a synthetic page) gives the same markdown
as with fp16.
The demo page then switched to `models/mix8/` as well (build `2026-11-18-c`); its smoke test reads the same page to the
same markdown. `web/models/f16` stays in the repo for the parity and benchmark checks.
