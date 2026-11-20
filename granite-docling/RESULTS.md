# RESULTS: granite-docling-258M against Indra OCR on Omni Parse Bench

IBM's [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) is a 258M-parameter
vision-language model that turns a page image into DocTags (text, layout, tables, equations). Indra OCR is
Datalab's 12M-parameter OCR model; `indra-browser/` runs it in a browser tab, and austegard.com's
anything-to-text page (any file to text, in the browser) uses that build. The question: on the benchmark Indra
was measured with, how does docling score, how big and slow is it, how does IBM's WebGPU demo ship it, and
should anything-to-text offer it as a second engine? Scores are Omni Parse Bench (OPB) percentages: the share of
a page's yes/no tests passed, the headline being the mean of the text, tables and layout families (0 to 100).

## Answer

No.
On all 2,937 pages of OPB, docling scores 52.46 against Indra fp32's 61.51. <!-- @evidence[[results/summary.json#opb.docling_fp32_docling_html.headline]] -->
That is −9.05 points (95% CI −11.10 to −6.93); Indra's shipped 8-bit pack scores 61.36. <!-- @evidence[[results/summary.json#opb.docling_fp32_docling_html.delta]] -->
The deficit sits on pages docling never finished: 790 ran to the 8,192-token cap, <!-- @evidence[[results/summary.json#loops.capped_pages]] -->
and on those docling scores 19.39 where Indra scores 61.71. <!-- @evidence[[results/summary.json#loops.capped.docling]] -->
Those pages are not harder for Indra than the rest. On the other 2,147 the two do not differ detectably: 60.03 vs 59.41. <!-- @evidence[[results/summary.json#loops.ended.docling]] -->
docling's 515 MB bf16 weights (IBM's WebGPU demo fetches 1,148 MB) compare with Indra's 2.24 MB shipped pack, <!-- @evidence[[results/summary.json#download_mb.docling_webgpu_space]] -->
and it takes 9.94 s per page on a T4 GPU against about 1 s for Indra on one CPU thread. <!-- @evidence[[results/summary.json#speed.docling_s_per_page_per_t4]] -->
That is 100 to 500 times the download, even packed, for a lower score.

What would change this: a loop stop (a repetition check, or a token cap with fallback to Indra) that turns the
capped pages into Indra-level pages. That is a prediction, not measured; docling would still be over 100 times
the download, and an independent benchmark could still move the gap.

## Findings

1. **docling scores 52.46 against Indra fp32's 61.51 on 2,937 pages, 16,288 tests**; through Markdown instead
   of docling-core's HTML it scores 51.17. Indra f16 61.55, mix8 61.36. (`results/opb_full.json`) <!-- @evidence[[results/summary.json#opb.docling_fp32_markdown.headline]] -->
2. **27% of pages run to the token cap.** 790 pages hit 8,192 tokens and score 19.39 (Indra 61.71 on them).
   A capped page is a repetition loop: in 38 of 40 sampled, the last 40 DocTags elements hold 8 or fewer distinct ones.
   The mean page generates 2,919 tokens, the median 1,575. (`results/breakdown.json`) <!-- @evidence[[results/summary.json#speed.docling_gen_tokens_mean]] -->
3. **Where docling stops on its own, it and Indra do not differ detectably:** docling 60.03, Indra 59.41 on 2,147 pages. <!-- @evidence[[results/summary.json#loops.ended.indra_fp32]] -->
   The difference's CI is −1.23 to +2.38; docling leads text (38.6 vs 31.0) and trails tables (48.9 vs 54.2).
   The page set is chosen by docling's own behaviour, so it is not a neutral sample. (`results/breakdown.json`)
4. **The capped pages follow page type.** Per OPB suite: docling wins arXiv math from olmOCR-bench (64.3 vs 49.0; Indra runs
   with math off, reading equations as plain text)
   and nearly ties PubMed Central tables (70.0 vs 71.5). It collapses on large drawings (large_format 0.0, 25 of
   27 pages looped; tiny_plans 11.0, 55 of 60), rotated pages (46.7 vs 84.7, 87 of 142 looped; Indra turns pages
   upright first) and multilingual PDFs (40.8 vs 83.6, 91 of 158). (`results/breakdown.json`)
5. **IBM's WebGPU Space downloads 1,148 MB.** <!-- @evidence[[results/summary.json#download_mb.docling_webgpu_space]] -->
   It runs transformers.js 3.7.5 (ONNX Runtime Web, WebGPU) on
   onnx-community/granite-docling-258M-ONNX with the token table in fp16 (116 MB), the vision encoder in fp32
   (374 MB) and the decoder in fp32 (658 MB, with a second copy of the token table as its output head). Its code
   comments that the 4-bit decoder "sometimes runs into repetition issues". (`results/summary.json`)
6. **Our packing recipe would store docling in 237.31 MB**, 4.8 times smaller than the Space. <!-- @evidence[[results/summary.json#download_mb.docling_pack_recipe]] -->
   That is a 4-bit token table in groups of 32 (32.5 MB), int8 matrices (198.7 MB) and the rest in fp32
   (6.1 MB). Its accuracy was not measured: the run was cut. (`run/pack_quant.py`)
7. **docling's HF generation config disables the KV cache** (`use_cache: false`), so plain `model.generate`
   recomputes the whole sequence each step; a 24-page probe ran an hour before it was stopped. Pass
   `use_cache=True`. (Log, Round 1)

## Method

**Benchmark.** [Omni Parse Bench](https://huggingface.co/datasets/datalab-to/omni_parse_bench) (Datalab) at
commit `d70182b9`: 2,937 pages, 2,343 documents, 22 suites, 16,288 yes/no tests; scorer `omni-parse-bench` 1.6.0
([source](https://github.com/datalab-to/omni_parse_bench)), the version `indra-browser/` used. Headline = mean of
the text, tables and layout family scores. Intervals are 95% bootstrap over documents, paired against Indra fp32
(`run/compare.py`, the same computation as `indra-browser/run/analyze.py` restricted to shared pages).

**Pages.** Every page is the grayscale prerender Indra reads (`indra-browser/run/prerender.py`: pdfium at 200 DPI,
lowered past 100 MP). The Kaggle job renders the same way and records each page's pixel md5; 294 sampled pages
matched the local renders exactly. docling gets the gray image as RGB.

**docling.** `ibm-granite/granite-docling-258M` (revision `untied`, the model card's vLLM revision) on vLLM
0.34.0, prompt "Convert this page to docling.", greedy, up to 8,192 new tokens, fp32 compute (a T4 has no bf16;
fp32 holds the bf16 weights exactly). Two Kaggle jobs on 2x Tesla T4, one vLLM engine per GPU
(`run/kaggle_gen.py`, `run/launch.py`).

**Adapter.** OPB 1.6.0 has no docling adapter. `run/opb_docling.py` turns DocTags into a docling-core 2.107.1
`DoclingDocument`; html is either its `export_to_html()` (the headline route: tables keep spans) or
`export_to_markdown()` through the benchmark's own `markdown.to_html` (the indra-ocr adapter's path); blocks are
every item with a box, its label mapped onto the benchmark's labels (`LABELS` in the script). It runs inside
`scripts/jail.sh`, as do the Indra runs.

**Indra.** Re-scored on the same renders with `indra-browser/run/opb_eval.py` (indra-ocr 1.1.2, models
v1.1.0, OCR mode, math off, furniture on); fp32 61.51, f16 61.55 and mix8 61.36 reproduce the earlier run to the
hundredth. Speed is the package on one CPU thread per page (four workers), median 917 ms per page; the browser
build takes 54 to 97 ms on a Mac (`indra-browser/`).

**Speed.** docling's 9.94 s is the mean GPU busy time per page per T4; Indra's figure is its mean of 1,028 ms
(median 917 ms), so the two are on different hardware and only show scale.

**Cost.** Kaggle GPU: 8.0 h of the weekly 30 h, of which about 1.8 h were probes and a deleted runaway probe.

## Log

### Round 1 (2026-11-19): docling on OPB, the WebGPU Space, the cut

Asked (Oskar, via the parent session): measure granite-docling against the earlier OCR experiments, assess IBM's
WebGPU Space, and say whether docling belongs in anything-to-text; later, also granite-vision-4.1-4b on the
same pages if it fit the budget. Oskar then cut the experiment: no new jobs, finish and report what was running.

- A first Kaggle probe (24 pages) showed vLLM 0.34 runs docling at fp32 on T4 (24 pages in 297 s, load
  included). Its HF transformers arm ran for over 45 minutes: the model's `generation_config.json` sets
  `use_cache: false`. The Kaggle API has no way to cancel a session, only to delete the kernel, so the probe kernel
  was deleted.
- The full run's first job lost one of its two workers in the first 128-page vLLM call: preprocessing a call
  holds every page's tiles in fp32 (up to 17 tiles of 512 px, about 50 MB per page), and the engine was killed,
  most likely for host memory. The other worker finished its 1,469 pages alone; a second job with 32-page calls
  and the multimodal cache off ran the other 1,468 on both GPUs.
- `kaggle_job.py pull` read only the first 500 files of a job's output listing; it now follows every page
  (claude-workspace `0471f07`).
- granite-vision-4.1-4b got a 16-page fp16 probe only (the 300-page run was cut): text 37.5, tables 70.0, no
  layout output, 4 of 16 pages at the 4,096-token cap, about 33 s per page per T4. It answers most pages as
  one-column `<md>` pipe tables. Too few pages to compare.
- The packing-recipe arms (`PACK_ARMS` in `run/kaggle_gen.py`: 4-bit table with the tied head, or an int8 head)
  were built and checked on CPU but not run.

| system | pages | headline | text | tables | layout | download | speed |
|---|---|---|---|---|---|---|---|
| Indra fp32 | 2,937 | 61.51 | 40.32 | 52.78 | 91.43 | 8.25 MB core gzip | 917 ms/page, 1 CPU thread |
| Indra f16 | 2,937 | 61.55 | 40.36 | 52.78 | 91.51 | 4.05 MB core gzip | 54–97 ms/page, Mac WebGPU |
| Indra mix8 (shipped) | 2,937 | 61.36 | 40.33 | 52.28 | 91.47 | 2.24 MB core gzip | same |
| docling fp32, docling HTML | 2,937 | 52.46 | 30.45 | 45.32 | 81.61 | 515 MB bf16; Space 1,148 MB | 9.94 s/page, T4 fp32 |
| docling fp32, Markdown | 2,937 | 51.17 | 28.89 | 43.02 | 81.61 | same | same |
| Surya OCR 2.5 (Datalab's report) | 2,937 | 85.9 | 83.8 | 78.3 | 95.7 | 1.38 GB bf16 | not measured |

Datalab reports Indra at 69.5; that run reads PDF text layers and math, which this one turns off for both
systems. Surya's row is from the benchmark's README, not measured here.

Reading at the time: docling reads ordinary pages about as well as Indra and better on math, and fails by looping
on a quarter of pages, mostly drawings, rotated scans and non-Latin PDFs. A looped page costs 8,192 tokens, which
is also why its mean time per page is high.
