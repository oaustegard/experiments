# anything-to-text — compose two in-browser apps into one "Anything → Text" SPA

**Request:** "Create an *Anything to Text* SPA, to be hosted under
austegard.com/web-utilities, based on a composite of these two apps. Remove the
MSD branding, replace with that of my own site."

**Inputs:** two single-file `.aspx` apps (uploaded):
- `parseimages.aspx` — in-browser OCR (images + PDF) via PaddleOCR PP-OCRv6 (ONNX) + pdf.js
- `transcriber.aspx` — in-browser speech-to-text (audio + video) via Whisper (Transformers.js) + ffmpeg.wasm

**Output:** `web-utilities/anything-to-text.html` (+ `_README.md`) in
`oaustegard/oaustegard.github.io` — PR
[#247](https://github.com/oaustegard/oaustegard.github.io/pull/247).

## Approach

The two source apps are a built-as-a-family pair: identical chrome (drop zone,
panels, buttons, badges), same class names, the same removed-vendor theme. The
real engineering value is in their JS, whose inline comments document hard-won
fixes:
- OCR: the CTC dictionary `+1` shift (append a space sentinel so the decoder
  prepends the blank), and the `noCache` flag that defeats ppu-paddle-ocr's
  1 KB-prefix cache key colliding across same-DPI PDF pages.
- Transcriber: the same-origin blob worker that lets `@ffmpeg/ffmpeg` spawn its
  worker from a CDN (cross-origin `new Worker()` is forbidden), and VAD bundling
  into ≤30 s full-context windows.

So the engines are kept **verbatim**. Each lives in its own
`<script type="module">` — separate top-level lexical scope, so both can declare
`setStatus`, `$`, `escapeHtml`, `download`, etc. without colliding. A
deterministic Python builder (`build.py`) extracts each engine's module and
applies only surgical transforms:

1. **Strip the favicon/brand**: remove the animated-favicon block (giant
   vendor base64 + the M/S/D cycling functions + the `visibilitychange`
   handler), the leftover anim calls, the header logo marks, and the
   `<link rel=icon>` data-URI. Notification icon → `/favicon.ico`.
2. **De-collide ids**: only three transcriber ids clash with the OCR engine
   once the shared drop zone owns `drop`/`file` — `run`/`status`/`copy` →
   `tr-run`/`tr-status`/`tr-copy`.
3. **Shared router**: replace each engine's own drop-zone wiring with a single
   router that classifies dropped files (by MIME, with extension fallback) and
   dispatches `intake:ocr` / `intake:transcribe` CustomEvents; each engine
   reveals its own (initially `hidden`) section on receipt. Mixed drops light up
   both.

Branding is the site's **Grouch's Workshop** system: link `styles/style.css` for
the canonical tokens/fonts/dark-mode, then component CSS written against those
variables (`--grouch`, `--tongue`, `--code-bg`, `--rule`, `--font-display`…),
plus `ai-disclosure` meta and a back-link — matching the existing
`web-utilities/` tools. The `web-utilities/index.html` auto-lists via
`github-toc`, so no index edit was needed.

## Verification (in-sandbox, no GPU/network)

- `node --check` passes on all three modules (router, OCR, transcriber).
- HTML tag balance OK; every `$("#…")`/`getElementById`/`el("…")` lookup in both
  engines resolves to a markup id; no dangling `fileInput`/`drop.on`/
  `makeDropZone(dom.drop)` refs.
- No surviving branding strings (only `/favicon.ico` matches a case-insensitive
  `favicon` grep).
- Router classification unit-tested: image/PDF → OCR, audio/video → transcript,
  type-less drops resolved by extension, `txt`/`csv` → skipped.

Live model-download + inference was **not** run here (no GPU/network in the build
sandbox) — that validation belongs on the Cloudflare branch preview the site's
CI deploys for the PR. The engines themselves are unchanged from working apps.

## Iteration 2 — OCR robustness (2026-06-22)

Feedback: a born-digital arXiv PDF OCR'd at **37 % confidence and was useless**,
with no way back to text. Root cause: the auto path extracts the text layer
first and offers "Re-run as OCR", but had **no inverse** — once a page was OCR'd
you were stuck; and PP-OCR's detector downscales its input to a fixed long side,
so a full page at 200 DPI is recognized at ~half resolution, which destroys dense
multi-column body text.

Sources are now **vendored** under `src/` (canonical, editable) and `build.py`
reads from there instead of the ephemeral session upload. OCR-engine changes:

- **`Extract text instead`** per-page button (inverse of `Re-run as OCR`):
  `reprocessPage(id, mode)` with `mode="text"` sets `forceText` and pulls the
  exact pdf.js layer. A forced-text page with no embedded layer stops and reports
  it rather than bouncing back into OCR.
- **Tiled recognition** (`recognizeTiled`/`tileGrid`/`dedupeBoxes`): renders above
  `SINGLE_MAX` are cut into ~1024 px tiles overlapping by 192 px, each recognized
  at near-native resolution; boxes are offset back to page pixels and overlap
  seams deduped by IoU (highest confidence wins). Small pages keep the single
  pass. Unit-tested: full grid coverage, single-tile small pages, correct dedup.
- Larger click-to-zoom previews so the parsed render is legible.

## Iteration 3 — column-aware reading order (2026-06-22)

Feedback: multi-column PDFs read as nonsense — `readingOrder()` grouped boxes
into lines by vertical proximity alone, zip-merging two columns at equal height.
Fix: column-aware ordering for both the OCR and text-layer paths, auto by default
with manual overrides ("do both").

- **`detectColumnCuts`** — 2-D occupancy projection. A flat height-weighted
  histogram is fooled by full-width title/byline/section-header boxes crossing
  the gutter (first attempt literally picked the *margins*). Instead, tile the
  content box into rows and count how many rows each x-bin occupies; a gutter bin
  occupies far fewer rows than its flanking columns. Bounded to the content span
  (margins excluded), and a cut is accepted only when columns reaching >=50% of
  peak flank it on both sides — rejecting ragged right edges and single-column
  pages.
- **`orderByColumns`** — full-width boxes act as band breaks slicing the page
  into horizontal stripes; within each stripe columns emit left-to-right, each
  line-grouped top-to-bottom. Title -> left -> right -> header -> left, never
  interleaved.
- **`pdfOrderLines`** + `PDF text order` toggle — preserve pdf.js native
  content-stream order for born-digital pages, chunking runs at baseline shifts.
- UI: `Columns` (auto/1/2/3) and `PDF text order` (geometric/preserve) selects.

Node-tested: realistic 2-col detects one gutter and orders correctly across two
stripes split by a full-width header; manual "2" matches auto; single-column and
ragged-right pages stay unsplit; 3-column yields 2 gutters. No-cuts path is the
original line grouping unchanged. Ships as oaustegard.github.io PR #249.

## Reusable

- `build.py` is idempotent: re-run after editing either source to regenerate.
- Pattern worth keeping: **two heavy single-file apps merged by giving each its
  own ES module + a CustomEvent router**, instead of namespacing/renaming every
  symbol. Module scope does the isolation for free; only document-unique ids and
  the shared intake need touching.
