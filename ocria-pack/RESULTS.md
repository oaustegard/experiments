# ocria-pack: OCRia-nano-1 in a browser tab, packed

LlamaIndex's [OCRia-nano-1](https://huggingface.co/llamaindex/OCRia-nano-1) (8.1M parameters) reads a page into
Markdown with tables, headings and lists. It runs inside LiteParse, whose WebAssembly build runs it on the CPU with
no WebGPU. This experiment asked whether its weights pack smaller, and whether it reads well enough next to Indra
OCR to join [anything-to-text](https://austegard.com/web-utilities/anything-to-text.html). Try it at
[austegard.com/experiments/ocria-pack/web/](https://austegard.com/experiments/ocria-pack/web/).

## Answer

On Omni Parse Bench, OCRia-nano-1 in LiteParse's WebAssembly build scores 67.66 against the published Indra OCR
row's 71.70 on the same 2,544 pages (gap −4.04, interval −6.02 to −2.03). <!-- @evidence[[results/summary.json#vs_indra.gap_points]] -->
It reads tables better (+13.9 points) and loses on text (−3.5) and layout labels (−22.5). It went into
anything-to-text as an option beside Indra, which stays the default.

8-bit weights change the score by −0.33 points (interval −0.98 to +0.06) on 206 pages. <!-- @evidence[[results/summary.json#int8_vs_fp32.gap_points]] -->
The model files shrink from 28.0 MB to 7.9 MB gzipped, but the LiteParse runtime adds 5.7 MB from jsdelivr, so a
first run downloads about 14 MB, against about 6.6 MB for Indra on an English page.

A full run on all 2,937 pages, or Indra rerun without its LaTeX math reader, could move the gap by a point or two.

## Findings

1. **OCRia trails Indra by 4 points overall and leads on tables.** Text 60.0 against 63.4, tables 73.8 against
   59.9, layout 69.3 against 91.8, on 2,544 of the 2,937 pages (`results/opb_base.json`). The layout figure is
   partly the adapter: LiteParse's blocks carry no page-header, caption or footnote kinds, so those layout tests
   cannot pass.
2. **8-bit packing is free within the noise.** −0.33 points on 206 OCR-only pages (`results/opb_L127.json`).
3. **The runtime is the download.** LiteParse 2.21.2's wasm is 20.6 MB (5.7 MB as jsdelivr serves it); the
   packed models are 0.97 MB for Latin OCR, 4.01 MB for LayoutNet and 2.9 MB for the other three scripts.
4. **The page size of a wrapped image changes LiteParse's Markdown.** Its layout rules measure text in points, so
   `web/ocria.js` sizes the one-page PDF for 250 DPI, the benchmark's setting. With 144 DPI the Markdown differed
   on 3 of 3 test pages; with 250 it matched the benchmark harness on 3 of 3.

## Method

Omni Parse Bench (datalab-to/omni_parse_bench, scorer 1.8.0 from source, run inside the hub's `scripts/jail.sh`).
OCRia: `run/lp_pages.mjs` runs `@llamaindex/liteparse-wasm` 2.21.2 in Node with `OcriaNano` (all four
recognizers, LayoutNet, headers and footers kept), PDFs as shipped and image pages wrapped by `run/png2pdf.py`.
`run/grade.py` turns each page into the benchmark's html and blocks and pairs it with the published Indra OCR and
Tesseract rows from `results.parquet` (bootstrap over documents, the scorer's own `uncertainty.paired`). The run
was stopped at 2,544 pages. Packing: `run/pack.py` (int8 per row, L = 127, every other byte kept), compared with
the original weights on gray 200 DPI renders of a seeded 600-page sample, of which 206 finished.

## Log

### 2026-11-26: evaluation and packing

Asked (Oskar): test OCRia-nano-1, see if it packs, add it to anything-to-text if comparable to Indra and PP-OCRv6
tiny. Ran the table above. A 7-bit arm, a no-LayoutNet arm and a PP-OCRv6 tiny run were started and stopped
unfinished when Oskar called the evaluation overkill. The anything-to-text option loads `web/ocria.js`.
