# OCRia-nano-1 page OCR in JavaScript

`ocr.js` is LiteParse's ocria-nano page logic (LiteParse 2.21.2, `src/ocr/ocria/{mod,page,det,img,rec}.rs`,
Apache-2.0) ported to one dependency-free ES module (72 KB, 24 KB gzipped). It runs everything around the
nets: page preprocessing, detector post-processing, line crops, the script router's page logic, the
recognizers' batching and CTC decoding, the rescue passes and the reading filters. The nets themselves
are called through an injected `infer`, so a browser can run OCRia-nano-1 on any backend (WebGPU, or
onnxruntime-web) without LiteParse's 20.6 MB WebAssembly runtime.

On the 40 reference pages, its results match LiteParse's native build (same ONNX Runtime 1.31.1 nets
as the test harness here) on 99.20% of text items verbatim. Counting the items LiteParse's projection
merges from side-by-side OCR boxes, the match is 100%. Against LiteParse's WebAssembly build the
figure is 93.86%, under the 95% target. Every miss there that is not such a merge is also missed by
LiteParse's own native build. The cause is the backend that runs the nets (tract in WebAssembly, ONNX
Runtime natively), not this port: see [Comparison](#comparison).

## Use

```js
import { ocrPage } from "./ocr.js";

const results = await ocrPage(
  { data, w, h, ch },            // Uint8Array, row-major; ch 1 (grey) or 3 (RGB); RGBA is also accepted
  infer,                         // see below
  { latin, cjk, rtl, world },    // parsed rec_<group>.charset.json arrays (rtl may be {chars, visual_rtl})
  { detMeta },                   // optional: parsed det.meta.json, or any option below
);
// [{ text, bbox: [x1, y1, x2, y2], confidence, polygon: [[x, y] x 4], words: [], rotation: null, faint }]
```

The results are LiteParse's `OcrResult` in page pixels, as `OcriaEngine::run` returns them. `bbox` holds
the truncated extents of `polygon`, which is the line quad with its corners in reading order (top-left,
top-right, bottom-right, bottom-left of the upright text). `faint` marks a short token from the faint
pass. `words` is always empty and `rotation` is always null, as in the Rust engine.

`infer` has the contract of LiteParse's `SessionPool::infer`:

```js
const out = await infer(model, shape, data, outputNames);
// "det":    [1,3,H,W] Float32Array          -> { prob: {shape: [1,1,h,w], data: Float32Array} }
// "router": [N,1,32,512]                    -> { group: {shape: [N,4], data}, flip: {shape: [N,2], data} }
// "rec_latin" | "rec_world" | "rec_rtl" | "rec_cjk": [N,1,32,W]
//                                           -> { ids: {shape: [N,T], data: Int32Array|BigInt64Array},
//                                                conf: {shape: [N,T], data: Float32Array} }
```

Calls are sequential, at most 64 crops per router batch and 32 per recognizer batch, with widths padded
to a multiple of 128 px. Each recognizer is also probed once, per `infer` function, with a [1,1,32,64]
input of ones; the ratio of input width to output timesteps gives its stride, as `Recognizer::new` does.
If the charsets object has no entry for a script group, that group's lines are read by `rec_latin`, as
LiteParse's WebAssembly build does when a recognizer is left out.

Options (`DEFAULT_OPTIONS` is the benchmarked `OcriaOptions::default()`):

| option | default | |
|---|---|---|
| `detLongSide` | 1600 | detector input long side (never enlarged) |
| `detThresh` | `detMeta.thresh`, else 0.55 | probability threshold; 0.55 is what det.meta.json ships (LiteParse falls back to 0.5 only when the file is missing) |
| `detMerge` | `detMeta.merge`, else 0.6 | collinear merge gap, in line heights |
| `detAdaptive` | null | `{minH, targetH, maxSide}`: second pass for small print (off by default) |
| `gutter` | true | cut lines at column gutters |
| `orient` | true | sideways-page retry and reading-order quads |
| `rescue` | true | low-confidence script rescue, per page and per line |
| `junk` | true | drop readings with no letter or digit, or one or two unsure characters |
| `faint` | `{thresh: 0.1, max: 24}` | faint short tokens next to detected lines (null = off) |

## What was ported

Everything `OcriaEngine::run` does for a page, type for type:

- **mod.rs**: `run_page` (grey to RGB, detection with faint candidates, sideways retry, gutter split,
  reading, the faint pass, the junk filter, reading-frame quads, truncated boxes) and `read` (router,
  upside-down pages, per-group recognition, the page-level and per-line script rescue, re-reading
  off-script lines with the page's recognizer, homoglyph cleaning). Every threshold keeps its Rust type:
  f32 constants go through `Math.fround`, so comparisons such as `p > 0.55f32` behave identically.
- **det.rs**: `detect_at` (area resize to a multiple of 32, the input normalization, threshold mask,
  4-connected components, mean-score filter, per-row hull, `minAreaRect`, unshrinking, `boxPoints`,
  quad ordering), `merge_collinear`, the row-band sort, `faint_boxes`, and the optional adaptive pass.
- **img.rs**: `to_gray`, `mean_gray`, `rot90`, `resize_linear` and `resize_area` with OpenCV's
  fixed-point taps and rounding, `perspective_transform` (with `max_by`'s last-maximum pivot),
  `warp_perspective_cubic` with the 32x32 fixed-point bicubic table, `components4` in the same visit
  order, `min_area_rect`, `box_points` and `otsu`.
- **page.rs**: `crop`, `long_lines`, `looks_sideways`, `reading_frame`, `split_gutters` with
  `rule_columns`, `majority`, `script_chars`, `latin_letters`, `clean_latin` (homoglyph folding both
  ways, other scripts' digits), `is_short_token`, `is_junk`. Rust's `char` predicates map to the Unicode
  property escapes they are defined by (`\p{Alphabetic}`, `\p{White_Space}`, ...).
- **rec.rs**: recognizer preprocessing, width-sorted batching, CTC decoding against the charset, the
  mean confidence, the router's tiling, softmax and `route`, and `visual_to_logical`. That last one
  needs the Unicode Bidirectional Algorithm (unicode-bidi 0.3.18 in Rust). ocr.js carries its own
  implementation: rules W1-W7, N0 (bracket pairs), N1-N2, I1-I2 and L1-L2 for one RTL paragraph, without
  mirroring, as `reorder_line` does. It uses a 3.5 KB Bidi_Class table generated from Unicode 15.1
  (`test/gen_bidi.py`).

## What was left out, and why

- **pool.rs**: replaced by the injected `infer`. Sessions, thread pools and timings are the backend's job.
- **serve.rs** (HTTP server) and **package.rs** (model download and checksums): these are not page
  logic. The caller loads the model files and passes in the charsets and det.meta.json.
- **`OcriaEngine::stats`** and the `threads` option are meaningless without the pool.
- **Explicit bidi formatting characters** (LRE, RLO, isolates and so on) are treated as removed (X9)
  rather than given embedding levels. No recognizer charset contains one, so a reading never does.
- **LiteParse stages after OCR** (`ocr_merge.rs` deskew and the confidence floor,
  `clean_ocr_table_artifacts`, projection, LayoutNet) are outside `OcriaEngine` and outside this port.
  The test harness reimplements the steps that change item text, so that it can compare at item level.

Known numeric deviations from the Rust code, none of which showed up on the 40 pages:

- `f32::exp` in the router softmax is `Math.fround(Math.exp(x))`, which can differ from libm's `expf`
  in the last bit.
- `hypot` in `minAreaRect` is computed as `sqrt(dx*dx + dy*dy)`. This is exact for the integer
  cell-corner coordinates it receives.

## Verification

- `node test/unit.mjs`: the Rust unit tests of img.rs, page.rs and det.rs (rot90, the bicubic table,
  minAreaRect, short tokens, `clean_latin`, the reading frame, the faint candidates), plus a bidi
  fixture. 4000 random strings over rec_rtl's charset went through python-bidi 0.6.12 (the Rust
  unicode-bidi crate under a Python wrapper, run in the jail) and `visualToLogical`; all 4000 matched,
  and 400 of them are kept as `test/bidi_cases.json`.
- **Image ops**: on 60 random rotated quads on a 1700x2200 page, the crops match OpenCV 4.10's
  `getPerspectiveTransform` + `warpPerspective(INTER_CUBIC, BORDER_REPLICATE)` bit for bit, and the
  detector input matches `resize(INTER_AREA)` bit for bit. OpenCV 5.1 (installed here) differs by up
  to 6 levels: newer OpenCV replaced the fixed-point warp, which LiteParse follows. The linear upscale
  path matches a NumPy transliteration of img.rs exactly; OpenCV's x86 SIMD path differs from it by 1
  level on some pixels (img.rs notes that it follows OpenCV's portable scalar code, not the SIMD paths).
- **Same pixels**: the raster LiteParse hands its OCR engine for each reference PDF, captured through a
  JS `ocrEngine` in the jail (`test/lp_raster.mjs`), equals the source PNG pixel for pixel on all 40 pages.

## Comparison

`test/run.mjs` backs `infer` with onnxruntime-node 1.31.1 on the upstream models in `../../models`.
It runs `ocrPage` once on each of the 40 reference pages and compares the results with two LiteParse
2.21.2 readings of the same pixels: gray PNGs from `/home/user/opb/gray600`, wrapped as PDFs of
px*72/250 pt, rendered at 250 DPI, layout off.

- **wasm**: the WebAssembly build, whose nets run on tract (`run/lp_pages.mjs --dump-items`,
  `/home/user/opb/ref40/ref40-nolayout`). This is the reference the task named.
- **native**: the Python wheel's native build, whose nets run on ONNX Runtime 1.31.1, the same runtime
  as the harness (`test/lp_native.py`, run in the jail; `/home/user/opb/ref40/ref40-native`).

Before comparing, each result goes through the parts of LiteParse's merge that touch item text: the
0.1 confidence floor, `clean_ocr_table_artifacts`, and the upright frame for whole-page rotated scans.

- **found**: share of reference item strings found verbatim in our page text.
- **+merged**: also counts a reference string that equals 2 to 6 of our same-row items joined by " " or
  "". LiteParse's projection merges neighbouring OCR boxes this way: gutter-split pieces of one line,
  a list marker and its text, a superscript.
- **ours found**: the reverse of found, our item strings searched in the reference text.
- **item CER**: an order-free character error rate. Each of our items is assigned to the reference item
  that covers it.
- **box**: share of reference items that have one of ours at IoU >= 0.5.
- **native vs wasm**: LiteParse's two builds compared with each other in the same way (`test/refdiff.mjs`).

| page | items (ref/ours) | wasm: found | +merged | ours found | item CER | box | native: found | +merged | native vs wasm: found |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| docparsing_images-images_5ed0e922_images_finance_resear... | 88/93 | 100.0 | 100.0 | 96.8 | 1.03 | 100.0 | 100.0 | 100.0 | 100.0 |
| docparsing_images-images_d26e7708_images_education_engl... | 61/61 | 98.4 | 98.4 | 98.4 | 0.06 | 100.0 | 100.0 | 100.0 | 98.4 |
| docparsing_images-images_dc807ef3_images_finance_resear... | 39/39 | 100.0 | 100.0 | 100.0 | 0.00 | 100.0 | 100.0 | 100.0 | 100.0 |
| forms_documentcloud-5683958-PA-Right-to-Know-Request-Fo... | 33/33 | 93.9 | 93.9 | 93.9 | 0.13 | 100.0 | 100.0 | 100.0 | 93.9 |
| forms_federal-filled_ntsb_6120_1-p1 | 228/231 | 95.6 | 96.9 | 94.8 | 0.40 | 100.0 | 98.7 | 100.0 | 96.9 |
| forms_federal-scan_osha_300a-p1 | 90/90 | 93.3 | 93.3 | 93.3 | 0.39 | 100.0 | 100.0 | 100.0 | 93.3 |
| forms_filled-filled_irs_1040-p1 | 218/224 | 96.8 | 98.6 | 99.1 | 0.13 | 99.1 | 97.7 | 100.0 | 98.6 |
| frbench_handwriting-Scan_11122025150333_p5 | 82/82 | 86.6 | 86.6 | 85.4 | 0.73 | 100.0 | 100.0 | 100.0 | 86.6 |
| multiling_finepdfs-bel_Cyrl_29588650-p1 | 58/61 | 69.0 | 74.1 | 78.7 | 0.72 | 100.0 | 94.8 | 100.0 | 74.1 |
| multiling_finepdfs-hau_Latn_87392203-p1 | 13/13 | 92.3 | 92.3 | 92.3 | 0.21 | 100.0 | 100.0 | 100.0 | 92.3 |
| multiling_finepdfs-images_ita_Latn_81612474-p1 | 10/10 | 100.0 | 100.0 | 100.0 | 0.00 | 100.0 | 100.0 | 100.0 | 100.0 |
| multilingual-source_fr_cerfa_10072-p10 | 33/33 | 97.0 | 97.0 | 97.0 | 0.04 | 100.0 | 100.0 | 100.0 | 97.0 |
| multilingual-source_fr_cerfa_15186-p4 | 86/86 | 100.0 | 100.0 | 100.0 | 0.00 | 100.0 | 100.0 | 100.0 | 100.0 |
| olmo_arxiv_math-2503.05844_pg8-p1 | 139/140 | 88.5 | 89.2 | 90.0 | 0.20 | 99.3 | 99.3 | 100.0 | 89.2 |
| olmo_arxiv_math-2503.08419_pg6-p1 | 38/38 | 57.9 | 57.9 | 57.9 | 1.17 | 100.0 | 100.0 | 100.0 | 57.9 |
| olmo_arxiv_math-2503.08554_pg1-p1 | 125/130 | 87.2 | 88.8 | 89.2 | 0.72 | 100.0 | 98.4 | 100.0 | 88.8 |
| olmo_long_tiny_text-11_pg161_pg1-p1 | 152/155 | 96.1 | 96.7 | 96.1 | 0.16 | 100.0 | 98.0 | 100.0 | 96.7 |
| olmo_multi_column-06c4bdabd51644320829c59e425e2350493b_... | 135/138 | 96.3 | 97.8 | 97.8 | 0.11 | 100.0 | 97.8 | 100.0 | 97.8 |
| olmo_multi_column-081875a1035e34dee439b9a2a3a55e319405_... | 93/94 | 95.7 | 96.8 | 95.7 | 0.18 | 98.9 | 98.9 | 100.0 | 96.8 |
| olmo_multi_column-0a7e813e2010ecd8239725c32c859fe00c92_... | 121/121 | 95.9 | 95.9 | 95.9 | 0.11 | 100.0 | 100.0 | 100.0 | 95.9 |
| olmo_table_tests-426df27cf990a3426b9e0731ad346f28184f_p... | 161/164 | 99.4 | 99.4 | 99.4 | 0.66 | 99.4 | 100.0 | 100.0 | 99.4 |
| olmo_table_tests-66aa22ac70dd3480f89737d77b79bc31ec46_p... | 67/66 | 100.0 | 100.0 | 100.0 | 0.06 | 98.5 | 100.0 | 100.0 | 100.0 |
| olmo_table_tests-b0d7c744416e5536f7cb0d7b894df96d7f9f_p... | 55/55 | 100.0 | 100.0 | 100.0 | 0.00 | 100.0 | 100.0 | 100.0 | 100.0 |
| parsebench_docs-5844c91b_docs_text_text_ocr__blurry2-p1 | 34/34 | 50.0 | 50.0 | 47.1 | 2.22 | 100.0 | 100.0 | 100.0 | 50.0 |
| parsebench_docs-b4ce8088_docs_chart_natural-catastrophe... | 72/72 | 100.0 | 100.0 | 100.0 | 0.00 | 100.0 | 100.0 | 100.0 | 100.0 |
| parsebench_docs-e6f85565_docs_layout_pdf_0a9304022e2d_p... | 122/122 | 98.4 | 98.4 | 98.4 | 0.15 | 100.0 | 100.0 | 100.0 | 98.4 |
| realdoc_pages-images_4aa5ef65_images_06b65d46-b66a-4a4c... | 85/86 | 81.2 | 82.4 | 82.6 | 0.53 | 100.0 | 98.8 | 100.0 | 82.4 |
| realdoc_pages-images_8f0133ee_images_1e387d04-ae4c-4afe... | 101/101 | 94.1 | 95.0 | 96.0 | 0.32 | 99.0 | 99.0 | 100.0 | 95.0 |
| realdoc_pages-images_d25eda87_images_3ae8a58c-a865-4504... | 46/46 | 95.7 | 95.7 | 95.7 | 0.07 | 100.0 | 100.0 | 100.0 | 95.7 |
| realdoc_pdfs-16975792_docs_supply_chain_67-p1 | 88/88 | 97.7 | 97.7 | 97.7 | 0.09 | 100.0 | 100.0 | 100.0 | 97.7 |
| realdoc_pdfs-5b38c959_docs_medical_healthcare_12-p1 | 102/102 | 97.1 | 97.1 | 97.1 | 0.23 | 100.0 | 100.0 | 100.0 | 97.1 |
| realdoc_pdfs-9003253c_docs_supply_chain_57-p1 | 101/101 | 84.2 | 87.1 | 90.1 | 0.82 | 97.0 | 96.9 | 100.0 | 87.1 |
| rotation_skew-images_rsb61ecb4a_skew2-p1 | 55/55 | 100.0 | 100.0 | 100.0 | 0.00 | 100.0 | 100.0 | 100.0 | 100.0 |
| rotation_skew-rs0700454e_rot270-p1 | 49/51 | 95.9 | 95.9 | 88.2 | 26.51 | 87.8 | 100.0 | 100.0 | 95.9 |
| synth_copies-images_sv165e4b2d-p1 | 89/89 | 97.8 | 97.8 | 97.8 | 0.11 | 100.0 | 100.0 | 100.0 | 97.8 |
| synth_copies-images_sv84d95810-p1 | 33/33 | 97.0 | 97.0 | 97.0 | 0.19 | 100.0 | 100.0 | 100.0 | 97.0 |
| synth_copies-svb1b97b0e-p1 | 115/116 | 96.5 | 96.5 | 96.6 | 0.45 | 100.0 | 100.0 | 100.0 | 96.5 |
| tables_derived-PMC10158643.1_doc-p6 | 91/92 | 91.2 | 92.3 | 91.3 | 0.30 | 100.0 | 98.9 | 100.0 | 92.3 |
| tables_derived-PMC12399693.1_doc-p4 | 176/177 | 98.9 | 99.4 | 98.9 | 0.10 | 100.0 | 99.4 | 100.0 | 99.4 |
| tables_derived-PMC7655504.1_doc-p4 | 147/148 | 90.5 | 90.5 | 89.9 | 0.93 | 100.0 | 99.3 | 100.0 | 90.5 |
| **all 40 pages** | 3631/3670 | **93.86** | 94.52 | 94.28 | 0.350 | 99.56 | **99.20** | 100.00 | 94.52 |

Overall, against **native** LiteParse: 99.20% found verbatim and 100.00% counting merges. 99.92% of our
item strings are in its text, 99.94% of boxes agree, and the item CER is 0.020%. Every page matches
completely once merges are counted. Three of our items are missing from native's output: the chart-axis
numbers "80", "60" and "30" on the finance page. LiteParse's projection drops them as gutter line numbers
(`form_lines`, `is_gutter_number_candidate`), which happens after OCR.

Against **wasm** LiteParse: 93.86% found verbatim (3408/3631), 94.52% counting merges, 94.28% the
other way, box agreement 99.56%, item CER 0.350%. The reading-order CER is 0.745%; it is order-sensitive,
which inflates it on the rotated junk page and the wide OSHA form. In the "+merged" column, every page
has exactly the figure that LiteParse's own native build gets against wasm. Of the 223 wasm items not
found verbatim in our text, 24 are projection merges. The other 199 are also missing from native
LiteParse's output. The nets' outputs differ between tract and ONNX Runtime, and close CTC decisions
flip on hard pages: "Unip(x)" vs "Unip(X)", "Hs" vs "H's", "КіраỸнікі" vs "Кіраунікі". The int8 Latin recognizer
is the most sensitive, as LiteParse's own README warns for its WebAssembly build. The 95% target cannot
be reached against the wasm reference while the nets run on another backend. LiteParse's native build
reaches 94.52%.

Pages under 90% against wasm, with the cause:

| page | found (+merged) | native LiteParse vs wasm | cause |
|---|---:|---:|---|
| parsebench_docs-...blurry2 | 50.0 (50.0) | 50.0 | blurred scan: almost every line has a borderline character, so tract and ONNX Runtime part ways |
| olmo_arxiv_math-2503.08419_pg6 | 57.9 (57.9) | 57.9 | dense inline math read as text (`izC`/`tzC`, `ωx`/`ωX`): many near-ties per line |
| multiling_finepdfs-bel_Cyrl | 69.0 (74.1) | 74.1 | Belarusian text: the dot-pattern logo reads as digit strings (`000000001,` vs `000000081,`), and ў is a near-tie between Ỹ and у; 3 merges |
| realdoc_pages-...4aa5ef65 | 81.2 (82.4) | 82.4 | Chilean census form with a FACSÍMIL watermark across the text, grey bars and small italic print; 1 merge |
| realdoc_pdfs-...supply_chain_57 | 84.2 (87.1) | 87.1 | scanned railcar checklist with handwritten entries and a large "34" written across the lines; 3 merges ("2 No damaged...") |
| frbench_handwriting-...p5 | 86.6 (86.6) | 86.6 | handwriting |
| olmo_arxiv_math-2503.08554_pg1 | 87.2 (88.8) | 88.8 | math symbols in running text; 2 merges |
| olmo_arxiv_math-2503.05844_pg8 | 88.5 (89.2) | 89.2 | math; 1 merge |

The rotated page `rotation_skew-...rot270` reaches 95.9% found, but its item CER and box agreement
against wasm are poor (26.5%, 87.8%). LiteParse itself reads this page as junk ("ででで", "O O O", "גاי"):
the page looks sideways (533 boxes, 39 long lines), but the retry a quarter turn counter-clockwise finds
29 long lines where it needs more than 83, so the page is read unturned. Which junk survives the junk filter differs between tract
and ONNX Runtime; native LiteParse and this port agree on all of it.

Time for the 40 pages in Node on this shared 4-core container: 23 to 29 s in ocr.js (0.6 to 0.7 s a
page) and 21 to 35 s in onnxruntime-node (2 threads), depending on the load from other jobs.

## Files

- `ocr.js`: the port.
- `test/run.mjs`: the harness (wasm and native comparison, per-page JSON in `test/out/pages/`, totals
  in `test/out/summary.json`).
- `test/compare.mjs`: the merge steps and metrics.
- `test/refdiff.mjs`: LiteParse native vs wasm.
- `test/report.mjs`: the table above.
- `test/unit.mjs`: the ported Rust tests and the bidi fixture `test/bidi_cases.json`.
- `test/gen_bidi.py`: regenerates the Bidi_Class table.
- `test/lp_native.py` and `test/lp_raster.mjs`: the jailed LiteParse runs that made the native reference
  and checked the rasters (they run LiteParse code, so only inside `scripts/jail.sh` of the hub).

```sh
cd web/page/test && npm install          # onnxruntime-node 1.31.1, pngjs 7.0.0 into test/node_modules
cd .. && node test/unit.mjs && node test/run.mjs && node test/refdiff.mjs --json test/out/refdiff.json && node test/report.mjs
```

## Licences

`ocr.js` is a translation of LiteParse source files (Copyright LlamaIndex, Apache License 2.0) into
JavaScript, changed as described above. The OCRia-nano-1 weights, and the charset and metadata files
that ship with them, fall under the OCRia-nano-1 Model License (see `../../NOTICE.md`). ocr.js embeds
none of them. A caller loads them, and their use restrictions apply to that use.
