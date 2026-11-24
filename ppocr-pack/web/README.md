# ppocr-pack/web: PP-OCRv6 text reading in a browser tab, and how to port a server PaddleOCR app to it

For an agent converting a server-side PaddleOCR app into a page that reads text on the user's device.
Everything below was run on 2026-11-24 unless a line says it was not. Live demo:
https://austegard.com/experiments/ppocr-pack/web/ (`index.html`, `app.js`). Results and method: `../RESULTS.md`.

## 1. What the engine is, and what it cannot do

PaddlePaddle's PP-OCRv6 **text detector and text recognizer** (tiny and small, Apache-2.0) with the weights packed
to int8 and run by hand-written WebGPU shaders. No onnxruntime, no wasm, nothing uploaded.
`pipeline.js` adds PaddleOCR's own geometry in JavaScript: DB postprocess, box ordering, perspective crops,
recognizer batching, CTC decode.

| | tiny | small |
|---|---|---|
| shipped, gzip (det + rec) | 1,565,233 B | 6,261,836 B |
| shipped, raw | 1,728,604 B | 8,109,511 B |
| upstream ONNX | 6.24 MB | 31.04 MB |
| recognizer character error, 400 real form-line crops (fp32, then packed) | 7.20%, 7.29% | 5.29%, 5.59% |
| lines whose text equals fp32's exactly | 91% | 96.25% |
| detector box F1 against the fp32 detector's boxes | 0.993 | 0.999 |
| recognizer dictionary | 6,904 characters | 18,708 characters |

(`../results/packs.json`, `../results/sweep.json`; the form-line CER differences are inside the 95% intervals.
Receipt-line crops read at 22.8% (tiny) and 21.1% (small) error even in fp32.) Dictionary coverage, counted from
`rec/models/<size>/dict.txt` by Unicode block: tiny holds 6,174 Han characters and no kana, Hangul, Cyrillic or
Arabic; small holds 15,565 Han and 180 kana, and no Hangul, Cyrillic or Arabic. Latin with accents, digits and
Chinese lines were read correctly in the parity fixtures. A Korean, Cyrillic or Arabic app needs a different
recognizer (section 6).

Limits:

- **Detection and recognition only.** No layout, tables, formulas, seals, document orientation or unwarping, textline
  orientation, word boxes. Section 4 has the options.
- **WebGPU is required.** `Gpu.create()` throws `WebGPU unavailable` or `no WebGPU adapter` otherwise. Chrome, Edge and
  Safari on a computer with a GPU work; check `navigator.gpu` first. For other browsers use the onnxruntime-web
  fallback below.
- **Quadrilateral boxes only** (DB `box` mode). A curved-text polygon mode does not exist here.
- **One page at a time.** Timings on a real GPU are not measured yet (`../RESULTS.md` finding 10). SwiftShader
  timings (a software GPU, 20 s detect and 50 s recognize for one 900 x 560 page) mean nothing.
- Large scans: the detector resizes the long side to `longSide` (default 1280). A 200 dpi A4 page is 2339 px tall, so
  small text is read at about half resolution. Raise `longSide` or cut the page into overlapping tiles and offset the
  boxes back; `experiments/anything-to-text/src/parseimages.aspx` (`recognizeTiled`, tiles of 1024 px with 192 px
  overlap, duplicates dropped at IoU 0.45) does this.

### Fallback for browsers without WebGPU

`experiments/anything-to-text/src/parseimages.aspx` runs the same PP-OCRv6 models through `ppu-paddle-ocr`
(onnxruntime-web, wasm) from the upstream fp32 ONNX files on Hugging Face (6 MB tiny, 31 MB small, 138 MB medium). It
needs the dictionary from `inference.yml`, with a space appended so the package prepends the CTC blank. Run on
2026-11-24 in headless Chromium; it read the cafe sample into the same 8 lines:

```js
import { PaddleOcrService } from "https://esm.sh/ppu-paddle-ocr@5.8.3/web";

const HF = "https://huggingface.co/PaddlePaddle";
const tier = "tiny";                                              // tiny | small | medium
const url = (kind, f) => `${HF}/PP-OCRv6_${tier}_${kind}_onnx/resolve/main/${f}`;

async function loadDict() {   // character_dict from inference.yml, plus " " so that ppu-paddle-ocr prepends the CTC blank
  const unq = (s) => (s.length > 1 && s[0] === "'" && s.at(-1) === "'") ? s.slice(1, -1).replace(/''/g, "'")
    : (s.length > 1 && s[0] === '"' && s.at(-1) === '"') ? s.slice(1, -1).replace(/\\(.)/g, "$1") : s;
  const lines = (await (await fetch(url("rec", "inference.yml"))).text()).split(/\r?\n/);
  let i = lines.findIndex((l) => /^\s*character_dict\s*:/.test(l));
  const chars = [];
  for (i++; i < lines.length; i++) { const m = lines[i].match(/^\s*-\s?(.*)$/); if (!m) break; chars.push(unq(m[1])); }
  chars.push(" ");
  return new TextEncoder().encode(chars.join("\n")).buffer;      // an ArrayBuffer is inline content, a string is a URL
}

const paddle = new PaddleOcrService({ model: { detection: url("det", "inference.onnx"),
  recognition: url("rec", "inference.onnx"), charactersDictionary: await loadDict() } });
await paddle.initialize();
const res = await paddle.recognize(canvas, { flatten: true, noCache: true });   // noCache: tiles of one size collide on its cache key
const lines = res.results.map(({ text, confidence: score, box: { x, y, width: w, height: h } }) =>
  ({ text, score, box: [[x, y], [x + w, y], [x + w, y + h], [x, y + h]] }));     // same shape as TextPipeline lines
```

With `lines` in that shape, `toPaddleResult(lines, {width, height})` (below) works on either engine.

## 2. Quick start

Imports from the published site (CORS is open, `access-control-allow-origin: *`). Run on 2026-11-24 in headless Chromium
on SwiftShader, with the URLs served from this folder; it read the cafe sample into 8 correct lines.

```html
<script type="module">
import { Gpu } from "https://austegard.com/experiments/ppocr-pack/web/det/gpu.js";
import { DetNet } from "https://austegard.com/experiments/ppocr-pack/web/det/engine.js";
import { RecNet } from "https://austegard.com/experiments/ppocr-pack/web/rec/engine.js";
import { TextPipeline, toPaddleResult } from "https://austegard.com/experiments/ppocr-pack/web/pipeline.js";

const BASE = "https://austegard.com/experiments/ppocr-pack/web/";
const size = "tiny"; // or "small"
const gpu = await Gpu.create();                                   // throws "WebGPU unavailable" without WebGPU
const det = await DetNet.load(gpu, `${BASE}det/models/${size}/`);
const rec = await RecNet.load(gpu, `${BASE}rec/models/${size}/`);
const pipe = new TextPipeline({ det, rec, longSide: 960 });

async function toCanvas(src) {                                    // a File, Blob or anything createImageBitmap takes
  const bmp = await createImageBitmap(src);
  const cv = document.createElement("canvas");
  cv.width = bmp.width; cv.height = bmp.height;
  cv.getContext("2d").drawImage(bmp, 0, 0);
  return cv;
}

const canvas = await toCanvas(await (await fetch(BASE + "samples/letter.png")).blob());
const lines = await pipe.run(canvas);                             // [{ box, text, score, detScore }], reading order
const result = toPaddleResult(lines, { width: canvas.width, height: canvas.height });
console.log(result.rec_texts, lines.timing);                      // timing: det_ms, post_ms, crop_ms, rec_ms
</script>
```

- `pipe.run(x)` takes a canvas, `ImageData`, `ImageBitmap` or `<img>` (`toRGBA` in `pipeline.js`).
  `box` is four `[x, y]` page pixels, clockwise from the top-left corner. `score` is the recognizer's
  confidence (PaddleOCR's `rec_score`), `detScore` the detector's box score.
- Constructor options: `longSide` (1280), `recBatch` (6), `minScore` (0, drops lines under it), `post` (overrides of
  `det.post`: `thresh`, `box_thresh`, `unclip_ratio`, `max_candidates`, `min_size`; model defaults are in
  `det/models/<size>/manifest.json`). `pipe.detect(image)` returns boxes without recognizing.
- Create the `Gpu`, `DetNet`, `RecNet` and `TextPipeline` once and reuse them; loading is the slow part. Load only the
  size the user picked.
- `app.js` is the full page: size picker, file input, drawing the boxes, timings, `relay.js` logging, `build.json`
  self-reload. Copy from it rather than rewriting.
- Smoke test of the whole page: `node web/test/smoke_page.mjs [tiny|small] [samples/receipt.png] [long=640]`.

### The PaddleOCR-shaped result

`toPaddleResult(lines, { width, height, minScore = 0 })` returns the five fields a PaddleOCR 3.x
`PaddleOCR().predict(img)[0]` carries for text (`res["rec_texts"]`, or `res.json["res"]`). Names, dtypes and
ordering were read from the `paddlex` 3.7.2 and `paddleocr` 3.7.0 source (`paddlex/inference/pipelines/ocr/pipeline.py`,
`result.py`, `components/common/convert_points_and_boxes.py`, `models/text_detection/processors.py`); no PaddleOCR
run was made. `web/test/check_paddle_result.mjs` checks the shape on a fixed input (`node web/test/check_paddle_result.mjs`).

| field | Python | here |
|---|---|---|
| `dt_polys` | list of int16 arrays `[4, 2]`, every detected box, sorted top to bottom then left to right (10 px row tolerance), clipped to `[0, W]`, `[0, H]` | array of `[[x, y] x 4]` integers, same corner order and sort |
| `rec_texts` | `list[str]`, same order as `dt_polys`, lines under `text_rec_score_thresh` removed | same, `minScore` is the threshold |
| `rec_scores` | `np.float64` array, parallel to `rec_texts` | plain numbers |
| `rec_polys` | `dt_polys` minus the removed lines | same |
| `rec_boxes` | int16 `[N, 4]`, `[left, top, right, bottom]` = min and max of each polygon's x and y; empty `np.array([])` when none | `[x0, y0, x1, y1]` per line, `[]` when none |

Plain arrays serialize with `JSON.stringify`, which is what `save_to_json` writes. Convert in the consuming code if it
needs typed arrays. If `TextPipeline` ran with `minScore > 0` the dropped lines never reach the adapter, so
`dt_polys` equals `rec_polys`; leave the pipeline's `minScore` at 0 and pass it to the adapter to match PaddleOCR.

**Not produced** (a consumer that reads one gets `undefined`; set it yourself where the value is known):

| PaddleOCR field | why |
|---|---|
| `input_path`, `page_index` | the caller owns them; add `page_index` per PDF page |
| `model_settings` (`use_doc_preprocessor`, `use_textline_orientation`) | both stages are absent; the values are `false` |
| `doc_preprocessor_res` | no document orientation classifier, no UVDoc unwarping |
| `textline_orientation_angles` | no textline orientation classifier (PaddleOCR fills `-1` per line when it is off). `cropQuad` does rotate a crop 90 degrees when height / width >= 1.5, as PaddleOCR does for vertical text; 180-degree flips are not detected |
| `text_det_params` | the adapter does not see the settings; build `{ limit_side_len: pipe.longSide, ...pipe.det.post, ...pipe.post }` if needed (the key names differ: Python uses `limit_side_len`, `thresh`, `box_thresh`, `unclip_ratio`, `max_side_limit`, `limit_type`) |
| `text_type` (`"general"`), `text_rec_score_thresh`, `return_word_box` | constants and echoes of arguments |
| `text_word`, `text_word_region`, `text_word_boxes`, `vis_fonts` | `return_word_box` is not implemented |

PaddleOCR 2.x's `ocr.ocr(img)` shape, `[[box, (text, score)], ...]`, is `lines.map((l) => [l.box, [l.text, l.score]])`.

## 3. Vendoring: copy the files instead of hot-linking

A page that imports from `austegard.com/experiments/...` follows whatever is published there. Copy for anything
that ships. The engine needs these files and nothing else (no npm package, no build step; relative imports only):

```
pipeline.js                      36,779 B
det/engine.js gpu.js kernels.js  29,673 B
rec/engine.js kernels.js         36,004 B                          code total 102,456 B
det/models/<size>/manifest.json weights.bin                        tiny 503,760 B   small 2,555,492 B
rec/models/<size>/manifest.json weights.bin dict.txt               tiny 1,224,844 B small 5,554,019 B
```

Models per size: tiny 1,728,604 B, small 8,109,511 B, both 9,837,115 B. With code: tiny 1,831,060 B, small 8,211,967 B.
Serve the `.bin` files gzipped (GitHub Pages does); gzip totals are in section 1.

```bash
SRC=ppocr-pack/web; DST=<other repo>/ocr; SIZE=tiny
mkdir -p $DST/det/models $DST/rec/models
cp $SRC/pipeline.js $DST/
cp $SRC/det/{engine,gpu,kernels}.js $DST/det/
cp $SRC/rec/{engine,kernels}.js $DST/rec/
cp -r $SRC/det/models/$SIZE $DST/det/models/ && cp -r $SRC/rec/models/$SIZE $DST/rec/models/
```

Import from `./ocr/det/gpu.js` and so on, and pass `./ocr/det/models/tiny/` to `DetNet.load`. A tiny-only copy of exactly
those files read the receipt sample into 17 lines in headless Chromium. Add `NOTICE.md` from this folder's parent
(Apache-2.0 attribution for the PaddlePaddle weights). Re-copy when `build.json` changes upstream.

## 4. Porting a server PaddleOCR app, step by step

**Step 0. Inventory what the app uses.** Run this in the server repo and sort each hit into the table:

```bash
grep -rnoE --include='*.py' "paddleocr|paddlex|PaddleOCR(VL)?\(|PPStructure\w*|TableRecognition\w*|LayoutDetection|FormulaRecognition|SealRecognition|DocPreprocessor|\.predict(_iter)?\(|\.ocr\(|use_[a-z_]+=|text_(det|rec)_[a-z_]+=|return_word_box|lang=|ocr_version|save_to_[a-z]+|\.json\b|rec_(texts|scores|polys|boxes)|dt_polys|(text|doc)_[a-z_]*model_(name|dir)" .
```

| server call or argument | client equivalent |
|---|---|
| `PaddleOCR(lang="en")` / `ocr_version="PP-OCRv6"` | `size = "tiny"` or `"small"`; the dictionary covers Latin and Chinese (small adds kana). Other `lang` values need section 6 |
| `text_detection_model_name`, `text_recognition_model_name` | `DetNet.load` and `RecNet.load` with a model folder |
| `ocr.predict(path_or_array)`, `ocr.ocr(img)` | `pipe.run(canvas)`; load a File with `toCanvas` above |
| input is a PDF, a directory, a URL | PDF pages through pdf.js (below); a directory is a file input with `multiple`; a URL is `fetch` then `createImageBitmap` (the host needs CORS) |
| `text_det_thresh`, `text_det_box_thresh`, `text_det_unclip_ratio` | `post: { thresh, box_thresh, unclip_ratio }` in the `TextPipeline` options |
| `text_det_limit_side_len` | `longSide`. The engine always resizes the long side to it (up or down); `text_det_limit_type`, `text_det_max_side_limit`, `text_det_input_shape` have no counterpart |
| `text_rec_score_thresh` | `toPaddleResult(lines, { minScore })` |
| `text_recognition_batch_size` | `recBatch` (6) |
| `use_doc_orientation_classify`, `use_doc_unwarping`, `use_textline_orientation` | no browser equivalent here; see "Stages without an equivalent" |
| `return_word_box` | none |
| `res["rec_texts"]`, `rec_scores`, `rec_polys`, `rec_boxes`, `dt_polys` | the same keys on `toPaddleResult(...)` |
| `res.save_to_json(dir)` | `JSON.stringify(result)` and a download |
| `res.save_to_img(dir)` | draw `rec_polys` on the canvas: `draw()` in `app.js` |
| `res.print()` | `console.log` |
| `device="gpu"`, `enable_mkldnn`, `cpu_threads`, `precision` | drop them; the GPU is the user's |
| `PPStructureV3`, `PaddleOCRVL`, `TableRecognition*`, `LayoutDetection`, `FormulaRecognition`, `SealRecognition` | no equivalent here; see below |

**Step 1. Replace the server call with the client pipeline** (section 2) and keep the rest of the app reading
`toPaddleResult` output, so the downstream code does not change. Keep the old endpoint running until the page is
checked against it on real documents; expect about 9% of tiny's lines to differ from fp32 text by a character
or two (section 1) and the box corners to match OpenCV's to the pixel on the test pages.

**Step 2. PDF pages.** Render each page to a canvas with pdf.js and treat it as an image. Run on 2026-11-24 (pdf.js
6.0.227 from jsDelivr, a two-page image PDF, 100 dpi): page 1 gave 8 lines and page 2 gave 17, the same as the
PNGs. The worker URL must be set explicitly.

```js
import * as pdfjs from "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.0.227/build/pdf.min.mjs";
pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.0.227/build/pdf.worker.min.mjs";

async function* pdfPages(file, dpi = 200) {                       // file: File or Blob
  const doc = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
  for (let n = 1; n <= doc.numPages; n++) {
    const page = await doc.getPage(n);
    const viewport = page.getViewport({ scale: dpi / 72 });       // PDF points are 1/72 inch
    const cv = document.createElement("canvas");
    cv.width = Math.round(viewport.width); cv.height = Math.round(viewport.height);
    await page.render({ canvasContext: cv.getContext("2d"), viewport }).promise;
    page.cleanup();
    yield { page_index: n - 1, canvas: cv };
  }
}

const results = [];
for await (const { page_index, canvas } of pdfPages(file)) {
  const lines = await pipe.run(canvas);
  results.push({ page_index, ...toPaddleResult(lines, { width: canvas.width, height: canvas.height }) });
}
```

A born-digital PDF has a text layer: `page.getTextContent()` gives exact text without OCR, and
`parseimages.aspx` (`extractPdfText`) tries it first and falls back to OCR for pages with no usable text.

**Step 3. Batching pages.** Loop over pages one after another with one shared `pipe`, as above. Line batching
inside a page is `recBatch`. Do not start several `pipe.run` calls at once on one pipeline: concurrent calls were
not tested. For a long document, show progress after each page, release the canvas (`cv.width = 0`) when its result
is stored, and keep results as JSON, not canvases. A page at 200 dpi A4 is a 15 MB RGBA buffer.

**Step 4. Stages without an equivalent.** Ask which of these the app really needs before choosing.

| server stage | status here | options |
|---|---|---|
| PP-Structure layout, reading order across blocks, tables | none | `experiments/indra-browser` (Indra OCR, also WebGPU, no onnxruntime): `Indra.load("models/s4c/")`, `ocr.read({w, h, data})` returns `markdown`, `lines [{box, text, conf}]`, `blocks`. Different engine, different output shape (markdown and blocks, not PaddleOCR fields); see its `RESULTS.md` |
| table structure (SLANet) | none | Indra's markdown tables, or a server endpoint |
| document orientation, UVDoc unwarping | none | ask the user to upload upright scans; rotate the canvas by 90 degrees steps and keep the orientation with the higher mean `score`; or a server endpoint |
| textline orientation | none | `cropQuad` already handles vertical lines; upside-down lines are not handled |
| PaddleOCR-VL, formula, seal | none | a server endpoint |

Upstream publishes ONNX files for layout, table, orientation and unwarping models (`PaddlePaddle/PP-DocLayout_plus-L_onnx`,
`SLANet_plus_onnx`, `UVDoc_onnx`, `PP-LCNet_x1_0_doc_ori_onnx`, `PP-LCNet_x1_0_textline_ori_onnx`). The exporters here were
not tried on them. The hybrid that needs the least work: read every page in the browser, and send a page to the old
server endpoint only when the app needs a stage above.

**Step 5. Log on the user's device** so you can read what happened without asking. `relay.js` sends `console.*`,
uncaught errors and `Relay.log` calls to the GPU bridge repo, only in a browser that has the GPU Bridge token, and
only after `?log=1`. Elsewhere it does nothing (run on 2026-11-24: `start`, `log`, `warn` and `error` returned without
errors with no token). Load it first, start it early, log each step, result and error:

```html
<script src="https://austegard.com/experiments/gpu-bridge/web/relay.js"></script>
<script type="module">
const relay = window.Relay.start({ page: "my-ocr", build: "2026-11-24-a" });   // page names the log folder; build names the code
relay.log("models ready", { size, loadMs });
relay.log("result", { page_index, lines: lines.length, timing: lines.timing, meanScore });
relay.error(e);                                                                // in every catch
</script>
```

Ship a `build.json` and compare it with the running build as `app.js` does, so a cached page reloads itself. Oskar
opens the page with `?log=1` on his device; read the result with `python3 scripts/gpu_bridge.py logs --page my-ocr`
from the `claude-workspace` hub (`docs/gpu-bridge.md`; needs `add_repo oaustegard/gpu-bridge-comms`). Log the user
agent, `navigator.gpu`, image size and per-stage times; they decide most failures.

**Step 6. Check the port.** Run the same documents through the server and the page and diff `rec_texts`
(a character error rate, not equality). Check at least one page with a rotated or vertical line, one with small text
at your real `dpi`/`longSide`, and one blank page (`rec_texts` is `[]`, `rec_boxes` is `[]`).

## 5. Tests in this folder

| command | what |
|---|---|
| `node web/test/check_paddle_result.mjs` | `toPaddleResult` shape, clipping, score threshold; Node, no GPU |
| `node web/test/smoke_page.mjs [tiny\|small] [samples/receipt.png] [long=640]` | the demo page end to end in headless Chromium on SwiftShader |
| `python3 -I run/ref_pipeline.py && node web/det/test/check_pipeline.mjs` | `pipeline.js` against PaddleOCR's postprocess (OpenCV, pyclipper) |
| `node web/det/test/run_e2e.mjs [tiny\|small] [page]`, `node web/det/test/run_det.mjs` | detector and full chain parity |
| `node web/rec/test/run_rec.mjs [tiny] [small] [--lines a,b]` | recognizer WGSL against the NumPy executor |

Headless Chromium flags (from `web/det/README.md` and `smoke_page.mjs`): `--no-sandbox --enable-unsafe-webgpu
--enable-features=Vulkan --use-vulkan=swiftshader --use-angle=swiftshader --use-webgpu-adapter=swiftshader`, and
`executablePath: "/opt/pw-browsers/chromium"` if present.

## 6. Packing a different PaddleOCR model

Pipeline: fetch the ONNX, export to an op program plus int8 weights, check parity against the NumPy executor and
onnxruntime, measure accuracy, copy the folder under `web/{det,rec}/models/<name>/`.

```bash
cd ppocr-pack
python3 -I run/fetch_models.py tiny_rec small_det               # PP-OCRv6 tiny/small only -> models/<size>_<kind>/inference.{onnx,yml}
# any other model: put it where the exporters look, <src>/<name>_rec/inference.{onnx,yml}
mkdir -p models/medium_rec && for f in inference.onnx inference.yml; do
  curl -sL -o models/medium_rec/$f https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/$f; done
python3 -I run/export_rec.py medium --scheme r127               # -> web/rec/models/medium-r127/{manifest.json,weights.bin,dict.txt}
python3 -I run/export_rec.py medium --scheme f32                # -> web/rec/models/medium-f32/ (the unquantized reference)
python3 -I run/export_det.py [tiny|small] --scheme r127 --src DIR --out DIR2   # -> DIR2/<name>/
```

Schemes: `f32`, `f16`, `r<L>` (int8 codes in plus or minus L, L <= 127, one fp16 scale per output channel); the
recognizer also takes `r63,head=t4` (4-bit CTC head, which lost accuracy on tiny). The default is `r63`; shipped are
tiny det and tiny rec r127, small det r63, small rec r127 (the 63 arms cost accuracy, `../RESULTS.md` findings 3 to 5).
Weights under 1,024 values and every bias stay f32. The output folder is `<name>` for the default scheme and
`<name>-<scheme>` otherwise; `--out` sets it for one model.

Ran on 2026-11-24:

- `PP-OCRv6_medium_rec_onnx` exported to 96 ops and 19.38 MB of weights in 2 s. `python3 -I run/ref_rec.py medium --pack r127`
  (after `python3 -I web/rec/test/make_lines.py` and the f32 export): NumPy f32 against onnxruntime max difference
  6.9e-05, packed text equal to onnxruntime's on 20 of 20 lines, character error 0 on those lines. The WGSL run
  (`run_rec.mjs`) was not run for medium: its model list is hard-coded to tiny and small (line 29).
- `PP-OCRv6_medium_det_onnx` exports to 172 ops and 15.6 MB of weights, but `export_det.py` accepts only the names
  `tiny` and `small`; put the files in `<src>/tiny_det/` and use `--out`, or add the name to `NAMES`.
- `latin_PP-OCRv5_mobile_rec_onnx` does **not** export: `unsupported op Constant at node 0`. Upstream PP-OCRv5 files
  are ONNX IR 3 with 330 `Constant` nodes where the v6 files have initializers. Rewriting the constants into
  initializers (and setting `ir_version` to 8) gets the exporter further, to `KeyError: 'conv2d_136.b_0'` in
  `op_alias`, a graph pattern it does not handle. Packing the language-specific v5 recognizers needs exporter work;
  nothing here has produced one. The Korean, Cyrillic, Arabic, Greek, Thai, Devanagari, Tamil and Telugu v5 files in the
  `PaddlePaddle/*_PP-OCRv5_mobile_rec_onnx` family were not tried and probably share the structure.

Checks, in order:

```bash
python3 -I web/rec/test/make_lines.py                          # 20 synthetic line crops + a 4-line batch -> web/rec/test/fixtures/
python3 -I run/ref_rec.py <name> [--pack <scheme>]             # NumPy executor vs onnxruntime; exit 0 when f32 matches ort within 1e-4 and text agrees
taskset -c 0,1 node web/rec/test/run_rec.mjs tiny --no-f32 --lines l00_en,l05_en   # WGSL vs NumPy: max abs 4.4e-05, same text 2/2 (tiny, run)
python3 -I run/ref_det.py && node web/det/test/run_det.mjs     # detector probability map, limit 1e-3
```

`ref_rec.py` and `run_rec.mjs` rewrite `results/rec_parity.json`; `git checkout results/rec_parity.json` after a trial
run on a model that is not shipped. The line fixtures hold Latin, Chinese, digits and mixed text; add lines in the
new model's script to `make_lines.py` before trusting a pass.

**Accuracy.** `python3 -I run/sweep.py tiny|small [--part rec|det]` is the sweep behind `../RESULTS.md`. It takes only
`tiny` and `small`, and reads line and page sets from `experiments/tesseract-gpu/data/` (made by that experiment's
`run/make_lines.py` and `run/make_real.py`, with FUNSD and SROIE downloads); that folder is absent in a fresh clone and
the sweep was not rerun for this recipe. For another model, copy its arm loop, or compare `cer_ort` with `cer_packed`
that `ref_rec.py` prints per line on your own labelled crops. The decision rule used: the smallest `r<L>` whose character
error change has a 95% interval that contains zero, or at least 99% identical text, and detector box F1 >= 0.99.

**Ops the engines support.** The exporters walk the ONNX graph once and fold chains into one op; any node they have no
handler for stops the export with `unsupported op <Type> at node <k>`, or with a more specific message such as
`ReduceMean at node k is neither squeeze-excite nor LayerNorm`. There is no silent skip.

| program op (what the browser runs) | ONNX it comes from |
|---|---|
| `conv` (det and rec) | `Conv` [+ constant `Add` bias] [+ `BatchNormalization`] [+ `Squeeze`/`Unsqueeze`] [+ `Relu`, `HardSigmoid`, `Sigmoid`, exact GELU (`Div, Erf, Add, Mul, Mul`), hard-swish, swish] |
| `se` (det and rec) | `ReduceMean(axes 2,3)` or `GlobalAveragePool`, `Conv`, `Relu`, `Conv`, `HardSigmoid`, `Mul` [, residual `Add`] |
| `convt`, `resize`, det only | `ConvTranspose` [+ bias, `Relu`, `Sigmoid`]; `Resize` nearest, integer scale |
| `maxpool`, `concat` (axis 1), `add` (tensor + tensor) | `MaxPool`, `Concat`, `Add` |
| `avgpool`, `softmax` (last axis), rec only | `AveragePool`, `Softmax` |
| `img2seq`, `seq2img`, rec only | a `Transpose` that moves the channel axis; `Squeeze`, `Unsqueeze`, `Reshape`, `Flatten` alias their input |
| `linear` (rec) | `MatMul` with a constant `[K, N]` weight [+ bias, activation]; the one before the final `Softmax` is the CTC head |
| `ln` (rec) | `ReduceMean, Sub, Pow, ReduceMean, Add(eps), Sqrt, Div, Mul(gamma), Add(beta)` |
| `attn` (rec) | `Reshape [T,3,nh,hd]`, `Transpose`, 3 `Slice`, `Mul(scale)`, `MatMul`, `Softmax`, `MatMul`, `Transpose`, `Reshape`; the engine refuses head dimension above 32 |

Also handled: `Identity` (dropped) and any node that only computes a shape (`Shape` and what consumes only its output).
Not handled, so they stop the export with `unsupported op`: `Constant`, `Gather`, `LSTM`, `GRU`, `Gemm`, `Pad` and any
other type with no handler in `run/export_det.py` or `run/export_rec.py` (`op_<Type>` methods). `Resize` must be nearest,
asymmetric, floor with an integer scale (an `assert`). A `MatMul` with no constant weight outside the attention pattern
stops the rec export. At run time a recognizer line narrower than the network's downsampling stops with `line too narrow`.

Further reading: `det/README.md` (detector program, tensor kinds, numerics), `../run/export_rec.py` header (recognizer
layouts and scheme grammar), `../RESULTS.md`.
