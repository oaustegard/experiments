# indra-browser: Indra OCR as a web page

Datalab's [Indra OCR](https://huggingface.co/datalab-to/indra-ocr) (indra-ocr 1.1.2) turns a scanned page or a PDF into
markdown with about 12M parameters: a line detector, CTC line recognizers per script group, n-gram word models for
unsure lines, an orientation classifier and a small reading-order transformer. This experiment runs all of it in a
browser tab on WebGPU, with no onnxruntime: hand-written WGSL kernels, the package's Python pipeline ported to JS, and
the weights packed to as few bytes as the benchmark allows. Page: https://austegard.com/experiments/indra-browser/web/
(`web/index.html`). Oskar Austegard, who owns this repo, asked for it.

## Answer

Can Indra run in a browser tab, how small can its download get, and what does the packing cost in accuracy? It runs:
given the same weights, the browser's markdown equals the Python package's character for character on every page
compared, 15 of 15 without the word model. <!-- @evidence[[results/summary.json#parity.nolm.same_markdown]] -->
Accuracy is Omni Parse Bench's headline (the mean of its text, tables and layout pass rates, in percent) on all
2,937 pages, read without the PDF text layer. fp32 weights score 61.51 and fp16 61.55; 4-bit weights collapse to 49.56. <!-- @evidence[[results/summary.json#s4c.headline]] -->
8-bit (61.18) halves the download but leans negative (interval −0.75 to +0.04), and Oskar chose accuracy over two
megabytes, so the page ships fp16: the detector, Latin recognizer, reading-order, orientation and script models come
to 4.05 MB gzipped. <!-- @evidence[[results/summary.json#bytes_f16.core]] -->
The n-gram word models pack much further: 2-bit values keep their handwriting gain, and a trie layout takes the
Latin table from 15.7 MB (npz) to 2.54 MB gzipped, fetched only when a line needs it. <!-- @evidence[[results/summary.json#bytes_f16.lm-base]] -->
A warm page takes 71 to 180 ms on Oskar's Mac, all of the pipeline but reading order on the GPU. <!-- @evidence[[results/summary.json#/mac_indra7/tables_derived-PMC10627924.1_doc-p4/warm_ms]] -->

## Findings

1. **Weights hold at fp16 and 8-bit, and break below.** Against fp32's 61.51: fp16 61.55 (+0.04, CI −0.03 to
   +0.11); fp16 for the three small models (detector, orientation, script) with 8-bit recognizers and order model 61.36 (−0.15, CI −0.37 to +0.04);
   8-bit everywhere 61.18 (−0.34, CI −0.75 to +0.04); 4-bit 49.56 (−11.95; its tables score falls from fp32's 52.78
   to 33.36); and between them 6-bit 60.40 (−1.11, CI −1.87 to −0.48). <!-- @evidence[[results/summary.json#s6c.headline]] --> These conv nets lose accuracy far faster per bit than the
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
4. **The browser reproduces the package exactly.** With the same weights, every line's text and box and the
   markdown match indra-ocr on the same pixels: 1452 lines on 15 pages without the word model, <!-- @evidence[[results/summary.json#parity.nolm.lines]] -->
   10 pages with it, and 9 pages with the line crops on the GPU as the page ships them (Round 1). Two ports had to
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
time; 5-bit then scored 58.51. Prediction before the sweep: 4-bit would cost under a point, as for gliner-browser; it cost 12.

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
| handwritten letter | 28 | 71 ms | 16 | 12 | 28 | 1 | 15 | What broke along the way: a
`warpAffine` that mirrored every rotation (sampled through the forward matrix), a skew threshold missed by `np.arange`
rounding, and a two-hour background limit that killed the first sweep chain partway through the weight arms.
