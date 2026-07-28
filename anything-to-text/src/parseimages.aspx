<%@ Page Language="C#" %>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parse Images - in-browser OCR</title>
<style>
  /* ---- MSD theme tokens ---- */
  :root {
    --bg:        #ffffff;
    --panel:     #f2f2f2;          /* MSD light gray */
    --line:      rgba(0,0,0,0.12);
    --ink:       #000000;          /* MSD black body text */
    --ink-dim:   rgba(0,0,0,0.62);
    --ink-faint: rgba(0,0,0,0.45);
    --accent:    #990000;          /* MSD Red - the only red in the palette */
    --danger:    #990000;
    --radius:    6px;
    --ui:      "Arial Narrow", Arial, "Helvetica Neue", sans-serif;   /* MSD body / UI font */
    --display: "Arial Narrow", Arial, "Helvetica Neue", sans-serif;   /* MSD headings (bold) */
    --mono:    ui-monospace, "Cascadia Mono", Consolas, monospace;    /* OCR output, kept monospaced for legibility */
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--ink); font-family: var(--ui);
    font-size: 14px; line-height: 1.5; min-height: 100vh; -webkit-font-smoothing: antialiased;
  }
  /* MSD brand: thin dark-red rule across the very top */
  body::before { content: ""; position: fixed; top: 0; left: 0; right: 0; height: 4px; background: var(--accent); z-index: 5; }
  .wrap { position: relative; z-index: 1; max-width: 920px; margin: 0 auto; padding: 40px 24px 80px; }

  header { margin-bottom: 24px; }
  .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
  .brand-mark { height: 30px; width: auto; display: block; }
  .brand-name { font-family: var(--display); font-weight: 700; font-size: 15px; letter-spacing: 0.02em; text-transform: uppercase; color: var(--ink); }
  .eyebrow { font-size: 11px; letter-spacing: 0.28em; text-transform: uppercase; color: var(--accent); margin: 0 0 10px; }
  h1 { font-family: var(--display); font-weight: 700; color: var(--accent); font-size: clamp(30px, 6vw, 50px); line-height: 1.0; text-transform: uppercase; margin: 0 0 12px; }
  .sub { color: var(--ink-dim); margin: 0; }
  .sub b { color: var(--ink); font-weight: 600; }

  [hidden] { display: none !important; }

  .panel { border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel); padding: 20px; margin-bottom: 18px; }

  /* drop zone */
  .dropzone { border: 1.5px dashed var(--line); border-radius: var(--radius); padding: 34px 20px; text-align: center; cursor: pointer; transition: border-color .2s, background .2s; background: #ffffff; }
  .dropzone:hover, .dropzone.over { border-color: var(--accent); background: rgba(153,0,0,0.04); }
  .dropzone .big { font-family: var(--display); font-size: 18px; font-weight: 600; color: var(--ink); }
  .dropzone .hint { color: var(--ink-faint); font-size: 12px; margin-top: 6px; }

  /* controls */
  .controls { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 18px; }
  label.field { display: block; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); margin-bottom: 6px; }
  input[type=number] { width: 96px; font-family: var(--ui); font-size: 13px; color: var(--ink); background: #ffffff; border: 1px solid var(--line); border-radius: 8px; padding: 9px 10px; }
  input[type=number]:focus { outline: none; border-color: var(--accent); }

  .runbar { display: flex; align-items: center; gap: 12px; margin-top: 18px; flex-wrap: wrap; }
  button.primary { font-family: var(--display); font-weight: 700; font-size: 14px; letter-spacing: 0.04em; text-transform: uppercase; color: #ffffff; background: var(--accent); border: 0; border-radius: var(--radius); padding: 11px 22px; cursor: pointer; transition: transform .08s, filter .2s; }
  button.primary:hover:not(:disabled) { filter: brightness(1.18); }
  button.primary:active:not(:disabled) { transform: translateY(1px); }
  button.primary:disabled { opacity: 0.4; cursor: not-allowed; }
  button.ghost { font-family: var(--ui); font-size: 12px; color: var(--ink); background: transparent; border: 1px solid var(--line); border-radius: 8px; padding: 8px 14px; cursor: pointer; transition: color .15s, border-color .15s; }
  button.ghost:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); }
  button.ghost:disabled { color: var(--ink-faint); border-color: var(--line); opacity: 0.6; cursor: not-allowed; }

  #status { color: var(--ink-dim); font-size: 12px; min-height: 18px; white-space: pre-wrap; flex: 1 1 220px; }
  #status .err { color: var(--danger); }
  #eta { color: var(--ink-faint); font-size: 12px; margin: 10px 0 0; min-height: 16px; }

  /* exports */
  .result-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .result-actions .label { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); margin-right: 4px; }

  /* per-page result cards */
  .page { border: 1px solid var(--line); border-radius: var(--radius); background: #ffffff; padding: 12px 14px; margin: 12px 0; }
  .page > h2 { font-size: 13px; font-weight: 600; margin: 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .page .pg-name { color: var(--ink); word-break: break-all; }
  .page .meta { font-size: 11px; color: var(--ink-faint); font-variant-numeric: tabular-nums; }
  .page .copy-page { padding: 5px 11px; font-size: 11px; }
  .page .reocr { padding: 5px 11px; font-size: 11px; }
  .page .hdr-actions { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
  .src { font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--ink-dim); }
  .src.ocr { color: var(--accent); border-color: rgba(153,0,0,0.45); }
  .page .cols { display: flex; gap: 14px; margin-top: 8px; align-items: flex-start; }
  .page .preview { width: 240px; flex: 0 0 240px; height: auto; border: 1px solid var(--line); border-radius: 4px; background: #fff; }
  .page .col-text { flex: 1 1 auto; min-width: 0; }
  .page .col-text .out { margin-top: 0; }
  .out.muted { color: var(--ink-faint); }
  @media (max-width: 680px) { .page .cols { flex-direction: column; } .page .preview { width: 100%; flex-basis: auto; max-width: 360px; } }
  .badge { font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--ink-dim); }
  .badge.pending { color: var(--ink-faint); }
  .badge.done { color: #ffffff; background: var(--accent); border-color: var(--accent); }
  .badge.error { color: var(--accent); border-color: var(--accent); }
  .out { white-space: pre-wrap; font-family: var(--mono); font-size: 13px; line-height: 1.5; background: var(--panel); border-radius: 6px; padding: 10px; margin-top: 8px; max-height: 280px; overflow: auto; color: var(--ink); }
  .out.err { color: var(--danger); background: rgba(153,0,0,0.05); }

  a { color: var(--accent); }

  #overlay { position: fixed; inset: 0; z-index: 50; background: rgba(0,0,0,0.55); display: flex; flex-direction: column; }
  #overlay[hidden] { display: none; }
  .ov-bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 8px 12px; background: var(--panel); border-bottom: 1px solid var(--line); font: 13px var(--ui); color: var(--ink); }
  #ov-title { font-weight: bold; }
  #ov-body { flex: 1; min-height: 0; overflow: auto; background: var(--bg); }
  #ov-body pre { margin: 0; padding: 12px; font: 12px/1.5 var(--mono); white-space: pre-wrap; word-break: break-word; color: var(--ink); }
  #ov-body iframe { display: block; width: 100%; height: 100%; border: 0; background: #fff; }

  @media (max-width: 680px) { .controls { gap: 12px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <img id="brandMark" class="brand-mark" alt="MSD">
      <span class="brand-name">Meso Scale Diagnostics</span>
    </div>
    <p class="eyebrow">Offline OCR</p>
    <h1>Parse Images</h1>
    <p class="sub">Extract text from images and PDFs with <b>PP-OCRv6 medium</b>, running entirely in your browser. Document bytes are never uploaded - only the model weights load on first run. Returns text lines in reading order with confidence, not chart or diagram interpretation.</p>
  </header>

  <div class="panel">
    <div id="drop" class="dropzone">
      <div class="big">Drop images or a PDF</div>
      <div class="hint">paste an image, or click to choose files</div>
    </div>
    <input id="file" type="file" accept="image/*,application/pdf" multiple hidden>

    <div class="controls">
      <div>
        <label class="field" for="tier">Model tier</label>
        <select id="tier">
          <option value="tiny">Tiny (~3M, fastest)</option>
          <option value="small">Small (~7M, balanced)</option>
          <option value="medium" selected>Medium (~35M, best)</option>
        </select>
      </div>
      <div><label class="field" for="dpi">PDF DPI</label><input id="dpi" type="number" value="200" step="50" min="72" max="400"></div>
      <div><label class="field" for="batch">Batch size</label><input id="batch" type="number" value="10" step="1" min="1" max="50"></div>
    </div>

    <div class="runbar">
      <button id="run" class="primary" disabled>Run batch</button>
      <button id="continue" class="ghost" disabled>Continue</button>
      <button id="estimate" class="ghost" disabled>Estimate</button>
      <button id="clear" class="ghost" disabled>Clear</button>
      <span id="status">Loading libraries...</span>
    </div>
    <div id="eta"></div>
  </div>

  <div class="panel">
    <div class="result-actions">
      <span class="label">Export</span>
      <button id="copy" class="ghost" disabled>Copy all text</button>
      <button id="txt" class="ghost" disabled>Download .txt</button>
      <button id="json" class="ghost" disabled>Download .json</button>
      <button id="html" class="ghost" disabled>Download .html</button>
      <button id="show-json" class="ghost" disabled>Show .json</button>
      <button id="show-html" class="ghost" disabled>Show .html</button>
    </div>
  </div>

  <div id="pages"></div>

  <div id="overlay" hidden>
    <div class="ov-bar">
      <span id="ov-title"></span>
      <button id="ov-close" class="ghost">Close</button>
    </div>
    <div id="ov-body"></div>
  </div>
</div>

<script type="module">
import { PaddleOcrService } from "https://esm.sh/ppu-paddle-ocr@5.8.3/web";
import * as pdfjsLib from "https://esm.sh/pdfjs-dist@6.0.227";

/* pdf.js needs an explicit worker URL; module worker is fine over http(s) (SharePoint), not file://. */
pdfjsLib.GlobalWorkerOptions.workerSrc = "https://esm.sh/pdfjs-dist@6.0.227/build/pdf.worker.min.mjs";

/* =========================================================================
   Animated favicon: cycle the MSD mark M -> S -> D by swapping the
   <link rel="icon"> node. Paused (rests on the full MSD mark) while the tab is hidden so it
   never flickers in a background tab. Static head icon is the S frame. */
const FAVICON_M = "data:image/png;base64,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";
const FAVICON_D = "data:image/png;base64,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";
/* PLACEHOLDER: the real MSD mark base64 arrived truncated in the source snippet; replace this value with the full string from transcriber.aspx */
const FAVICON_MSD = "data:image/png;base64,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";
/* PLACEHOLDER: the real S mark base64 arrived truncated in the source snippet; replace this value with the full string from transcriber.aspx */
const FAVICON_S = "data:image/png;base64,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";
const FAVICON_FRAMES = [FAVICON_M, FAVICON_S, FAVICON_D, FAVICON_MSD, FAVICON_MSD];   /* M, S, D, then the full mark held an extra tick (2s) */
let favTimer = null, favIdx = FAVICON_FRAMES.length - 1;   /* MSD: the resting frame, shown statically */
let ocrRunning = false;   /* favicon animates only while OCR is working */
function setFavicon(uri) {
  const cur = document.querySelector('link[rel="icon"]');
  const next = document.createElement("link");
  next.rel = "icon"; next.type = "image/png"; next.href = uri;
  if (cur && cur.parentNode) cur.parentNode.replaceChild(next, cur);
  else document.head.appendChild(next);
}
function startFaviconAnim() {
  if (favTimer) return;
  favTimer = setInterval(() => {
    favIdx = (favIdx + 1) % FAVICON_FRAMES.length;
    setFavicon(FAVICON_FRAMES[favIdx]);
  }, 1000);
}
function stopFaviconAnim() {
  if (favTimer) { clearInterval(favTimer); favTimer = null; }
  favIdx = FAVICON_FRAMES.length - 1; setFavicon(FAVICON_MSD);   /* rest on the full mark */
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopFaviconAnim(); else if (ocrRunning) startFaviconAnim();
});

const $ = s => document.querySelector(s);
const statusEl = $("#status"), etaEl = $("#eta"), pagesEl = $("#pages");
const estimateBtn = $("#estimate"), runBtn = $("#run"), continueBtn = $("#continue"), clearBtn = $("#clear");
const copyBtn = $("#copy"), txtBtn = $("#txt"), jsonBtn = $("#json"), htmlBtn = $("#html");
const showJsonBtn = $("#show-json"), showHtmlBtn = $("#show-html");
const overlay = $("#overlay"), ovBody = $("#ov-body"), ovTitle = $("#ov-title"), ovClose = $("#ov-close");
const setStatus = m => { statusEl.textContent = m; console.log("[status]", m); };

const SECONDS_PER_PAGE = 10;   /* skill estimate: ~10s/page on CPU */
const LOAD_SECONDS = 6;        /* one-time model load */

let paddle = null;             /* lazy PaddleOcrService */
let pages = [];                /* { id, image, kind, render, status, text, lines, meanConf, ms, error } */
let nextIndex = 0;             /* pointer to the next unprocessed page */
const pdfDocs = [];            /* keep pdf documents alive so lazy page render works */

/* ---------- OCR model selection ----------
   PP-OCRv6 in three tiers; weights stream from Hugging Face on first run.
   The 50-language recognition dictionary is fetched from inference.yml at init time
   because it contains non-ASCII characters that cannot live in this ASCII-only page.
   Set USE_V6 = false to fall back to the bundled PP-OCRv5 mobile (English only). */
const USE_V6 = true;
const HF_BASE = "https://huggingface.co/PaddlePaddle";
const V6_TIERS = {
  tiny:   { det: HF_BASE + "/PP-OCRv6_tiny_det_onnx/resolve/main/inference.onnx",
            rec: HF_BASE + "/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.onnx",
            yml: HF_BASE + "/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.yml" },
  small:  { det: HF_BASE + "/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx",
            rec: HF_BASE + "/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx",
            yml: HF_BASE + "/PP-OCRv6_small_rec_onnx/resolve/main/inference.yml" },
  medium: { det: HF_BASE + "/PP-OCRv6_medium_det_onnx/resolve/main/inference.onnx",
            rec: HF_BASE + "/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx",
            yml: HF_BASE + "/PP-OCRv6_medium_rec_onnx/resolve/main/inference.yml" },
};

function unquoteYaml(s) {
  if (s.length >= 2 && s[0] === "'" && s[s.length - 1] === "'") return s.slice(1, -1).replace(/''/g, "'");
  if (s.length >= 2 && s[0] === '"' && s[s.length - 1] === '"') return s.slice(1, -1).replace(/\\(.)/g, "$1");
  return s;
}
/* Pull PostProcess.character_dict (a YAML list) out of a PaddleX inference.yml. */
function parseCharacterDict(yml) {
  const lines = yml.split(/\r?\n/);
  let i = lines.findIndex(l => /^\s*character_dict\s*:/.test(l));
  if (i < 0) throw new Error("character_dict not found in inference.yml");
  const out = [];
  for (i = i + 1; i < lines.length; i++) {
    const m = lines[i].match(/^\s*-\s?(.*)$/);
    if (!m) break;
    out.push(unquoteYaml(m[1]));
  }
  return out;
}
async function loadV6Dict(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("dictionary fetch HTTP " + res.status);
  const chars = parseCharacterDict(await res.text());
  if (chars.length < 100) throw new Error("dictionary parse produced only " + chars.length + " entries");
  /* PaddleOCR's CTC label layout for a use_space_char model (every PP-OCRv6 multilingual
     rec head is one) is  [blank] + character_dict + [space]  so numClasses === glyphs + 2.
     The inference.yml character_dict lists ONLY the real glyphs: neither the leading blank
     nor the trailing space is in it.

     ppu-paddle-ocr's decodeResults (core/recognition/ctc.js) auto-prepends the blank ONLY
     when (dict.length === numClasses - 1). Handing it the raw glyph list (length
     numClasses - 2) fails that test, so no blank is prepended and every class index
     resolves one slot early -> the uniform +1 shift. Appending the space sentinel makes
     dict.length === numClasses - 1, the decoder prepends the blank, and the layout lines
     up: index 0 = blank, 1..glyphs = real chars, last index = space (the decoder emits a
     literal space for its lastDictIndex class regardless of the stored value).

     Dict is returned as UTF-8 bytes because _loadResource treats a STRING as a URL to
     fetch and only accepts an ArrayBuffer as inline content. */
  chars.push(" ");
  console.log("[dict] v6 glyphs:", chars.length - 1, "| passing entries:", chars.length, "| expected model numClasses:", chars.length + 1);
  return new TextEncoder().encode(chars.join("\n")).buffer;
}

/* ---------- model ---------- */
function selectedTier() {
  const el = document.getElementById("tier");
  return (el && V6_TIERS[el.value]) ? el.value : "medium";
}

async function ensurePaddle() {
  if (paddle) return paddle;
  try {
    let opts = {};
    if (USE_V6) {
      const tier = selectedTier();
      const urls = V6_TIERS[tier];
      setStatus("Loading PP-OCRv6 " + tier + " dictionary...");
      const dict = await loadV6Dict(urls.yml);
      setStatus("Loading PP-OCRv6 " + tier + " weights (first run, may take a moment)...");
      opts = { model: { detection: urls.det, recognition: urls.rec, charactersDictionary: dict } };
    } else {
      setStatus("Loading PP-OCRv5 mobile weights (first run, ~10s)...");
    }
    paddle = new PaddleOcrService(opts);
    await paddle.initialize();
    console.log("[ocr] PaddleOcrService initialized; model:", USE_V6 ? "PP-OCRv6 " + selectedTier() : "PP-OCRv5 mobile");
  } catch (e) {
    paddle = null;
    console.error("[ocr] init failed", e);
    throw e;
  }
  return paddle;
}

/* ---------- intake ---------- */
function addImage(file) {
  const id = pages.length;
  pages.push({
    id, image: file.name || ("image-" + (id + 1)), kind: "image",
    status: "pending", text: "", lines: [], segments: [], pageW: 0, pageH: 0,
    meanConf: 0, ms: 0, error: null, source: null, preview: null, forceOcr: false, forceText: false,
    render: async () => {
      const bmp = await createImageBitmap(file);
      const cv = document.createElement("canvas");
      cv.width = bmp.width; cv.height = bmp.height;
      cv.getContext("2d").drawImage(bmp, 0, 0);
      bmp.close && bmp.close();
      return cv;
    },
  });
}

async function addPdf(file) {
  const buf = await file.arrayBuffer();
  const doc = await pdfjsLib.getDocument({ data: buf }).promise;
  pdfDocs.push(doc);
  const label = file.name || "document.pdf";
  console.log("[pdf]", label, "pages:", doc.numPages);
  for (let p = 1; p <= doc.numPages; p++) {
    const id = pages.length;
    const pageNo = p;
    pages.push({
      id, image: label + " p." + pageNo, kind: "pdf",
      status: "pending", text: "", lines: [], segments: [], pageW: 0, pageH: 0,
      meanConf: 0, ms: 0, error: null, source: null, preview: null, forceOcr: false, forceText: false,
      getPdfPage: () => doc.getPage(pageNo),
      render: async (scale) => {
        const pdfPage = await doc.getPage(pageNo);
        const viewport = pdfPage.getViewport({ scale });
        return renderPdfCanvas(pdfPage, viewport);
      },
    });
  }
}

function renderPdfCanvas(pdfPage, viewport) {
  const cv = document.createElement("canvas");
  cv.width = Math.round(viewport.width); cv.height = Math.round(viewport.height);
  return pdfPage.render({ canvasContext: cv.getContext("2d"), viewport }).promise.then(() => cv);
}

async function accept(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  setStatus("Reading input...");
  for (const f of files) {
    try {
      if (f.type === "application/pdf" || /\.pdf$/i.test(f.name || "")) await addPdf(f);
      else if (f.type.startsWith("image/")) addImage(f);
      else console.warn("[skip] unsupported type", f.name, f.type);
    } catch (e) {
      console.error("[intake] failed for", f.name, e);
      setStatus("Could not read " + (f.name || "a file") + ": " + (e && e.message || e));
    }
  }
  renderPages();
  refreshControls();
  estimate();
}

/* ---------- estimate (skill step 1) ---------- */
function estimate() {
  const remaining = pages.length - nextIndex;
  if (!pages.length) { etaEl.textContent = ""; return; }
  const secs = remaining * SECONDS_PER_PAGE + (paddle ? 0 : LOAD_SECONDS);
  const mins = secs / 60;
  const eta = mins >= 1 ? (mins.toFixed(1) + " min") : (secs + " s");
  let msg = pages.length + " page(s); " + remaining + " pending; about " + eta + " of processing.";
  if (remaining > 10) msg += " Larger than 10 pages - process in batches: press Run, then Continue.";
  etaEl.textContent = msg;
  console.log("[estimate]", msg);
}

/* ---------- OCR ---------- */
function clampDpi() {
  const v = parseInt($("#dpi").value, 10);
  return Math.min(400, Math.max(72, isNaN(v) ? 200 : v));
}
function batchSize() {
  const v = parseInt($("#batch").value, 10);
  return Math.min(50, Math.max(1, isNaN(v) ? 10 : v));
}

/* Group boxes into lines by vertical proximity, then sort top-to-bottom, left-to-right.
   Returns an array of line strings. This is the single-column primitive that the
   column-aware orderer composes per column band. */
function lineify(group) {
  if (!group.length) return [];
  const heights = group.map(r => r.box.height).filter(h => h > 0).sort((a, b) => a - b);
  const medH = heights.length ? heights[Math.floor(heights.length / 2)] : 12;
  const tol = Math.max(4, medH * 0.6);
  const byY = group.slice().sort((a, b) => a.box.y - b.box.y);
  const lines = [];
  for (const r of byY) {
    let line = null;
    for (const L of lines) { if (Math.abs(L.y - r.box.y) <= tol) { line = L; break; } }
    if (!line) { line = { y: r.box.y, items: [] }; lines.push(line); }
    line.items.push(r);
  }
  lines.sort((a, b) => a.y - b.y);
  return lines.map(L => L.items.sort((a, b) => a.box.x - b.box.x).map(r => r.text).join(" "));
}

/* Auto-detect column gutters with a 2-D occupancy projection. A multi-column page has a
   vertical channel that is empty over most of the page HEIGHT; a height-weighted 1-D
   histogram is fooled by the handful of full-width elements (title, byline, section
   headers) that cross that channel at a few y's. So instead we tile the content box into
   rows and, per x-bin, count how many distinct rows hold any text: a gutter bin occupies
   far fewer rows than the column bins flanking it. The search is bounded to the content
   span [minX,maxX] (margins excluded by construction) and a candidate gutter is accepted
   only if tall columns flank it on BOTH sides (rejecting ragged right edges). Returns []
   (single column) when sparse or channel-free - the safe default the manual Columns
   control overrides. */
function detectColumnCuts(boxed, pageW) {
  if (boxed.length < 12) return [];
  const minX = Math.min.apply(null, boxed.map(r => r.box.x));
  const maxX = Math.max.apply(null, boxed.map(r => r.box.x + r.box.width));
  const minY = Math.min.apply(null, boxed.map(r => r.box.y));
  const maxY = Math.max.apply(null, boxed.map(r => r.box.y + r.box.height));
  const spanX = maxX - minX, spanY = maxY - minY;
  if (spanX <= 0 || spanY <= 0) return [];
  const BINS = 120, binW = spanX / BINS;
  const heights = boxed.map(r => r.box.height).filter(h => h > 0).sort((a, b) => a - b);
  const medH = heights.length ? heights[Math.floor(heights.length / 2)] : 12;
  const ROWS = Math.max(8, Math.min(240, Math.round(spanY / Math.max(4, medH))));
  const rowH = spanY / ROWS;
  const occ = Array.from({ length: BINS }, () => new Uint8Array(ROWS));
  for (const r of boxed) {
    const x0 = Math.max(0, Math.floor((r.box.x - minX) / binW));
    const x1 = Math.min(BINS - 1, Math.floor((r.box.x + r.box.width - minX) / binW));
    const y0 = Math.max(0, Math.floor((r.box.y - minY) / rowH));
    const y1 = Math.min(ROWS - 1, Math.floor((r.box.y + r.box.height - minY) / rowH));
    for (let b = x0; b <= x1; b++) { const col = occ[b]; for (let rr = y0; rr <= y1; rr++) col[rr] = 1; }
  }
  const rowCount = occ.map(col => { let s = 0; for (let i = 0; i < col.length; i++) s += col[i]; return s; });
  const peak = rowCount.slice().sort((a, b) => b - a)[Math.floor(BINS * 0.1)] || 1;
  const thresh = Math.max(1, peak * 0.25);    /* gutter bin: <25% of a column's row count
                                                 (loose enough to survive several full-width
                                                 headers crossing the channel) */
  const hi = peak * 0.5;                       /* a real column flank reaches >=50% of peak */
  const minRun = Math.max(2, Math.round(BINS * 0.015));
  const runs = [];
  let run = null;
  const close = () => { if (run) { runs.push(run); run = null; } };
  for (let i = 0; i < BINS; i++) { if (rowCount[i] <= thresh) { run ? (run.e = i) : (run = { s: i, e: i }); } else close(); }
  close();
  const cuts = [];
  for (const rn of runs) {
    if ((rn.e - rn.s + 1) < minRun) continue;
    if (!rowCount.slice(0, rn.s).some(v => v >= hi)) continue;
    if (!rowCount.slice(rn.e + 1).some(v => v >= hi)) continue;
    cuts.push(minX + ((rn.s + rn.e + 1) / 2) * binW);
  }
  return cuts.slice(0, 3);   /* up to 4 columns; more than that is almost always noise */
}

/* Column-aware reading order. With no cuts it is exactly the old single-column behaviour.
   With cuts, boxes that straddle a gutter (titles, full-width section headers, wide
   figures) are treated as band breaks: the page is sliced into horizontal stripes at each
   full-width element, and within every stripe the columns are emitted left-to-right, each
   line-grouped top-to-bottom. So a paper reads title -> left col -> right col -> next
   full-width header -> left col ..., instead of zig-zagging across the gutter. */
function orderByColumns(boxed, cuts) {
  if (!cuts.length) return lineify(boxed);
  const cx = r => r.box.x + r.box.width / 2;
  const cy = r => r.box.y + r.box.height / 2;
  const spans = r => cuts.some(c => r.box.x < c - 1 && r.box.x + r.box.width > c + 1);
  const colOf = r => { let i = 0; while (i < cuts.length && cx(r) >= cuts[i]) i++; return i; };
  const full = boxed.filter(spans).sort((a, b) => cy(a) - cy(b));
  const colBoxes = boxed.filter(r => !spans(r));
  const out = [];
  const emitStripe = (yTop, yBot) => {
    for (let ci = 0; ci <= cuts.length; ci++) {
      const g = colBoxes.filter(r => colOf(r) === ci && cy(r) >= yTop && cy(r) < yBot);
      for (const ln of lineify(g)) out.push(ln);
    }
  };
  let prevY = -Infinity;
  for (const f of full) {
    const fy = cy(f);
    emitStripe(prevY, fy);
    for (const ln of lineify([f])) out.push(ln);
    prevY = fy;
  }
  emitStripe(prevY, Infinity);
  return out;
}

/* Reading order over flattened boxes, honoring the Columns control (auto-detect, or a
   fixed 1/2/3). Used for OCR output and for geometric text-layer ordering. */
function readingOrder(items, opts) {
  opts = opts || {};
  const boxed = (items || []).filter(r => r && r.box && typeof r.text === "string" && r.text.length);
  if (!boxed.length) return (items || []).map(r => (r && r.text) || "").filter(Boolean);
  const pageW = opts.pageW || (boxed.reduce((m, r) => Math.max(m, r.box.x + r.box.width), 0) || 1);
  const mode = opts.columns || "auto";
  let cuts;
  if (mode === "auto") cuts = detectColumnCuts(boxed, pageW);
  else { const n = Math.max(1, parseInt(mode, 10) || 1); cuts = []; for (let i = 1; i < n; i++) cuts.push((pageW * i) / n); }
  return orderByColumns(boxed, cuts);
}

/* Preserve pdf.js's own emission order (content-stream order, which for born-digital
   documents is usually correct column-by-column reading order), only chunking the run
   sequence into lines where the baseline shifts. The escape hatch for pages where the
   geometric reorder does worse than the source's native order. */
function pdfOrderLines(items) {
  const seq = (items || []).filter(r => r && typeof r.text === "string" && r.text.length);
  if (!seq.length) return [];
  const heights = seq.map(r => (r.box ? r.box.height : 0)).filter(h => h > 0).sort((a, b) => a - b);
  const medH = heights.length ? heights[Math.floor(heights.length / 2)] : 12;
  const tol = Math.max(4, medH * 0.6);
  const lines = [];
  let cur = null, lastY = null;
  for (const r of seq) {
    const y = r.box ? r.box.y : (lastY == null ? 0 : lastY);
    if (cur && lastY != null && Math.abs(y - lastY) <= tol) cur.push(r.text);
    else { if (cur) lines.push(cur.join(" ")); cur = [r.text]; }
    lastY = y;
  }
  if (cur) lines.push(cur.join(" "));
  return lines;
}

/* current values of the PDF reading controls (absent elements -> safe defaults) */
function columnMode() { const el = document.getElementById("columns"); return el ? el.value : "auto"; }
function textOrderMode() { const el = document.getElementById("textorder"); return el ? el.value : "geometric"; }


function meanConfidence(res, items) {
  if (res && typeof res.confidence === "number") return res.confidence;
  if (!items || !items.length) return 0;
  return items.reduce((a, r) => a + (r.confidence || 0), 0) / items.length;
}

/* Downscaled preview data URL so each card can show the rendered page next to its text
   without holding full-resolution canvases in memory. */
function makePreview(canvas, maxW) {
  const cap = maxW || 720;
  const scale = Math.min(1, cap / (canvas.width || 1));
  const pv = document.createElement("canvas");
  pv.width = Math.max(1, Math.round(canvas.width * scale));
  pv.height = Math.max(1, Math.round(canvas.height * scale));
  pv.getContext("2d").drawImage(canvas, 0, 0, pv.width, pv.height);
  return pv.toDataURL("image/jpeg", 0.85);
}

/* Extract pdf.js text-layer runs as uniform {text, box, confidence} items in canvas
   pixels for the given viewport, using pdf.js's own TextLayer geometry (build/pdf.mjs):
   tx = viewport.transform composed with the run transform; glyph height = hypot(tx[2],
   tx[3]); width = run width * scale; box top = baseline tx[5] minus height. Rotated and
   vertical runs are uncommon in documents and are placed at their origin without
   rotation. Whitespace-only runs are dropped. confidence is 1 - the text is exact. */
async function extractPdfText(pdfPage, viewport) {
  const tc = await pdfPage.getTextContent();
  const scale = viewport.scale;
  const items = [];
  for (const it of (tc.items || [])) {
    if (typeof it.str !== "string" || !it.str.trim()) continue;
    const tx = pdfjsLib.Util.transform(viewport.transform, it.transform);
    const h = Math.hypot(tx[2], tx[3]) || (it.height * scale) || 1;
    const w = (it.width * scale) || (it.str.length * h * 0.5);
    items.push({ text: it.str, confidence: 1, box: { x: tx[4], y: tx[5] - h, width: w, height: h } });
  }
  return items;
}
function hasUsableText(items) {
  return items.some(r => r.text && r.text.trim().length);
}

/* Shared assembly for both sources: reading-order lines, normalized segments stamped
   with their source, and mean confidence (exact text is 1). */
function buildResult(pg, items, source, res) {
  const W = pg.pageW || 1, H = pg.pageH || 1;
  pg.source = source;
  /* born-digital text can keep pdf.js's native order when asked; everything else (and
     geometric text) flows through the column-aware orderer. */
  pg.lines = (source === "text" && textOrderMode() === "pdf")
    ? pdfOrderLines(items)
    : readingOrder(items, { pageW: W, pageH: H, columns: columnMode() });
  pg.text = pg.lines.join("\n");
  pg.segments = items
    .filter(r => r && r.box && typeof r.text === "string" && r.text.length)
    .map(r => ({
      text: r.text, source,
      conf: typeof r.confidence === "number" ? r.confidence : 0,
      x: r.box.x / W, y: r.box.y / H, w: r.box.width / W, h: r.box.height / H,
    }));
  pg.meanConf = source === "text" ? 1 : meanConfidence(res, items);
}

async function processPage(pg) {
  const t0 = performance.now();
  const scale = clampDpi() / 72;

  /* Hybrid PDF path: prefer pdf.js's embedded text layer - exact, instant, and needs no
     model. Auto pages fall back to OCR when no usable text is found; a page the user has
     explicitly asked back to text (forceText) stops here even if empty, rather than
     bouncing straight back into the OCR it was trying to escape. */
  if (pg.kind === "pdf" && !pg.forceOcr) {
    const pdfPage = await pg.getPdfPage();
    const viewport = pdfPage.getViewport({ scale });
    const items = await extractPdfText(pdfPage, viewport);
    if (hasUsableText(items)) {
      const canvas = await renderPdfCanvas(pdfPage, viewport);
      pg.pageW = canvas.width; pg.pageH = canvas.height;
      pg.preview = makePreview(canvas);
      buildResult(pg, items, "text", null);
      pg.ms = Math.round(performance.now() - t0);
      pg.status = "done";
      console.log("[text]", pg.image, pg.ms + "ms", pg.lines.length + " lines (pdf.js text layer)");
      return;
    }
    if (pg.forceText) {
      const canvas = await renderPdfCanvas(pdfPage, viewport);
      pg.pageW = canvas.width; pg.pageH = canvas.height;
      pg.preview = makePreview(canvas);
      buildResult(pg, [], "text", null);
      pg.ms = Math.round(performance.now() - t0);
      pg.status = "done";
      console.log("[text]", pg.image, "no embedded text layer on this page (forced text)");
      return;
    }
    console.log("[text]", pg.image, "no usable text layer - falling back to OCR");
  }

  /* OCR path: images, forced-OCR PDF pages, and text-less PDF pages. The model loads
     lazily here, so a PDF that is entirely born-digital never touches the weights. */
  await ensurePaddle();
  const canvas = await pg.render(scale);
  pg.pageW = canvas.width; pg.pageH = canvas.height;
  pg.preview = makePreview(canvas);
  /* Tile large renders before recognition. PP-OCR's detector resizes its input down to a
     fixed long side (~960 px), so a full A4/Letter page rendered at 200 DPI (~1700x2200)
     is shrunk to roughly half resolution before any box is found - which is why dense,
     small, multi-column body text scores badly. recognizeTiled feeds the model
     near-native-resolution tiles, offsets each tile's boxes back into page pixels, and
     dedupes the overlaps. */
  const { items, meanConf } = await recognizeTiled(canvas, pg.image);
  buildResult(pg, items, "ocr", { confidence: meanConf });
  pg.ms = Math.round(performance.now() - t0);
  pg.status = "done";
  console.log("[ocr]", pg.image, pg.ms + "ms", pg.lines.length + " lines", (pg.meanConf * 100).toFixed(1) + "%");
}

/* ---------- tiled recognition ----------
   Below SINGLE_MAX (long side) a page is small enough that the detector's downscale is
   harmless, so it goes through in one pass (and keeps the old behaviour exactly). Larger
   pages are cut into TILE-sized cells overlapping by OVERLAP so a line split by one tile's
   hard edge is whole inside a neighbour. Each cell is recognized independently; boxes are
   translated from cell pixels to page pixels, then near-duplicate detections from the
   overlap seams are removed (highest confidence wins). noCache as in processPage:
   same-size tiles would otherwise collide on ppu-paddle-ocr's 1 KB-prefix cache key. */
const TILE = 1024;        /* target tile size in page pixels */
const OVERLAP = 192;      /* seam overlap; > a few text lines so most lines fall whole in a tile */
const SINGLE_MAX = 1280;  /* long side at/under which we skip tiling */

function tileGrid(w, h, tile, overlap) {
  const step = Math.max(1, tile - overlap);
  const origins = (extent) => {
    if (extent <= tile) return [0];
    const out = [];
    for (let p = 0; ; p += step) {
      if (p + tile >= extent) { out.push(extent - tile); break; }
      out.push(p);
    }
    return [...new Set(out)];
  };
  const xs = origins(w), ys = origins(h);
  const cells = [];
  for (const y of ys) for (const x of xs) cells.push({ x, y, w: Math.min(tile, w - x), h: Math.min(tile, h - y) });
  return cells;
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0 ? inter / uni : 0;
}

/* keep the highest-confidence box from each cluster of overlapping detections */
function dedupeBoxes(items) {
  const kept = [];
  for (const it of items.slice().sort((p, q) => (q.confidence || 0) - (p.confidence || 0))) {
    if (!it.box) continue;
    if (kept.some(k => iou(it.box, k.box) > 0.45)) continue;
    kept.push(it);
  }
  return kept;
}

async function recognizeTiled(canvas, label) {
  const W = canvas.width, H = canvas.height;
  if (Math.max(W, H) <= SINGLE_MAX) {
    const res = await paddle.recognize(canvas, { flatten: true, noCache: true });
    const items = (res && res.results) || [];
    return { items, meanConf: meanConfidence(res, items) };
  }
  const cells = tileGrid(W, H, TILE, OVERLAP);
  const all = [];
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    setStatus("OCR " + (label || "page") + " - tile " + (i + 1) + "/" + cells.length + "...");
    const sub = document.createElement("canvas");
    sub.width = c.w; sub.height = c.h;
    sub.getContext("2d").drawImage(canvas, c.x, c.y, c.w, c.h, 0, 0, c.w, c.h);
    const res = await paddle.recognize(sub, { flatten: true, noCache: true });
    for (const r of (res && res.results) || []) {
      if (!r || !r.box) continue;
      all.push({
        text: r.text, confidence: r.confidence,
        box: { x: r.box.x + c.x, y: r.box.y + c.y, width: r.box.width, height: r.box.height },
      });
    }
  }
  const items = dedupeBoxes(all);
  console.log("[ocr]", label, cells.length + " tiles ->", all.length, "boxes,", items.length, "after dedupe");
  return { items, meanConf: meanConfidence(null, items) };
}

async function runBatch() {
  if (nextIndex >= pages.length) return;
  setBusy(true);
  ocrRunning = true; startFaviconAnim();
  const end = Math.min(pages.length, nextIndex + batchSize());
  setStatus("Processing pages " + (nextIndex + 1) + "-" + end + " of " + pages.length + "...");
  for (; nextIndex < end; nextIndex++) {
    const pg = pages[nextIndex];
    try {
      await processPage(pg);
    } catch (e) {
      /* one bad page records its error and we keep going - never abort the batch */
      pg.status = "error";
      pg.error = (e && e.message) || String(e);
      console.error("[process] page failed", pg.image, e);
    }
    updatePage(pg);
  }
  const remaining = pages.length - nextIndex;
  setStatus(remaining > 0 ? ("Batch done. " + remaining + " page(s) remaining - press Continue.") : "Done.");
  setBusy(false);
  ocrRunning = false; stopFaviconAnim();
  estimate();
}

/* Force a single page through the OCR path - used when the pdf.js text layer was present
   but partial. Re-renders that page only and updates its card in place. */
/* Re-run a single PDF page through the other source. mode "ocr" forces recognition
   (used when the embedded text layer is present but partial); mode "text" forces the
   pdf.js text layer (the escape hatch from a bad OCR pass back to exact text). Re-renders
   that page only and updates its card in place. */
async function reprocessPage(id, mode) {
  const pg = pages[id];
  if (!pg) return;
  if (mode === "text") { pg.forceOcr = false; pg.forceText = true; }
  else { pg.forceOcr = true; pg.forceText = false; }
  pg.status = "pending";
  pg.error = null;
  updatePage(pg);
  setBusy(true);
  setStatus((mode === "text" ? "Extracting embedded text on " : "Re-running OCR on ") + pg.image + "...");
  try {
    await processPage(pg);
  } catch (e) {
    pg.status = "error";
    pg.error = (e && e.message) || String(e);
    console.error("[reprocess] page failed", pg.image, e);
  }
  updatePage(pg);
  setBusy(false);
  setStatus("Done.");
  refreshControls();
}

/* ---------- rendering (additive: each page block updates in place) ---------- */
function renderPages() {
  pagesEl.innerHTML = "";
  for (const pg of pages) {
    const div = document.createElement("div");
    div.className = "page";
    div.id = "pg-" + pg.id;
    div.innerHTML = pageInner(pg);
    pagesEl.appendChild(div);
  }
}
function updatePage(pg) {
  const div = $("#pg-" + pg.id);
  if (div) div.innerHTML = pageInner(pg);
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function pageInner(pg) {
  const done = pg.status === "done";
  const srcBadge = done
    ? '<span class="src ' + (pg.source || "") + '">' + (pg.source === "text" ? "text layer" : "OCR") + '</span>'
    : "";
  const meta = done
    ? '<span class="meta">' + (pg.source === "text"
        ? pg.lines.length + ' lines | exact'
        : pg.ms + 'ms | ' + pg.lines.length + ' lines | conf ' + (pg.meanConf * 100).toFixed(1) + '%') + '</span>'
    : (pg.status === "error" ? '<span class="meta">failed</span>' : "");
  /* PDF pages can swap source either way: text->OCR when the layer is partial, and
     OCR->text to escape a bad recognition back to the exact embedded text. */
  let convBtn = "";
  if (pg.kind === "pdf" && done) {
    if (pg.source === "text") convBtn = '<button class="ghost reocr" data-pg="' + pg.id + '">Re-run as OCR</button>';
    else if (pg.source === "ocr") convBtn = '<button class="ghost retext" data-pg="' + pg.id + '">Extract text instead</button>';
  }
  const copyBtn = done ? '<button class="ghost copy-page" data-pg="' + pg.id + '">Copy</button>' : "";
  const preview = pg.preview ? '<img class="preview" src="' + pg.preview + '" alt="page preview">' : "";
  const textOut = pg.status === "error"
    ? '<div class="out err">' + escapeHtml(pg.error || "error") + '</div>'
    : (done
        ? '<div class="out">' + (escapeHtml(pg.text) || "(no text)") + '</div>'
        : '<div class="out muted">working&#x2026;</div>');
  const body = '<div class="cols">' + preview + '<div class="col-text">' + textOut + '</div></div>';
  return '<h2><span class="badge ' + pg.status + '">' + pg.status + '</span> '
    + '<span class="pg-name">' + escapeHtml(pg.image) + '</span> ' + srcBadge + meta
    + '<span class="hdr-actions">' + convBtn + copyBtn + '</span></h2>' + body;
}

/* ---------- exports ---------- */
function allText() {
  return pages.filter(p => p.status === "done").map(p => "=== " + p.image + " ===\n" + (p.text || "(no text)")).join("\n\n");
}
function asJson() {
  return JSON.stringify(pages.map((p, i) => ({
    page: i + 1, image: p.image, source: p.source, text: p.text, lines: p.lines,
    mean_conf: p.meanConf,
    page_px: { w: p.pageW || 0, h: p.pageH || 0 },
    /* bbox is [x, y, w, h] normalized 0..1 against page_px; conf and source are per-segment. */
    segments: (p.segments || []).map(s => ({
      text: s.text, conf: s.conf, source: s.source,
      bbox: [+s.x.toFixed(5), +s.y.toFixed(5), +s.w.toFixed(5), +s.h.toFixed(5)],
    })),
    error: p.error,
  })), null, 2);
}
/* Absolutely-positioned text layer: one container per page, one div per detected
   segment placed by percentage so the markup carries the original geometry. A reader,
   the browser, and a downstream model all recover columns and tables from position
   alone - no layout heuristic. Font size tracks box height via container-query height
   units; segments below 60% confidence get a faint tint for triage. */
function asHtml() {
  const done = pages.filter(p => p.status === "done" && p.segments && p.segments.length);
  const head =
    '<!doctype html>\n<html>\n<head>\n<meta charset="utf-8">\n' +
    '<title>OCR layout export</title>\n<style>\n' +
    'body{margin:0;background:#525659;font-family:Arial,Helvetica,sans-serif;}\n' +
    'h2{color:#e8e8e8;font:13px Arial;font-weight:normal;margin:16px auto 6px;max-width:900px;}\n' +
    '.page{position:relative;width:100%;max-width:900px;margin:0 auto 24px;background:#fff;\n' +
    '  border:1px solid #222;box-shadow:0 1px 6px rgba(0,0,0,0.4);container-type:size;}\n' +
    '.seg{position:absolute;white-space:nowrap;overflow:visible;line-height:1;color:#111;}\n' +
    '.seg.low{background:rgba(214,40,40,0.13);}\n' +
    '</style>\n</head>\n<body>\n';
  const body = done.map(p => {
    const ar = (p.pageW && p.pageH) ? (p.pageW + " / " + p.pageH) : "1 / 1.4142";
    const segs = p.segments.map(s => {
      const left = (s.x * 100).toFixed(3), top = (s.y * 100).toFixed(3);
      const w = (s.w * 100).toFixed(3), h = (s.h * 100).toFixed(3);
      const fs = (s.h * 100 * 0.78).toFixed(3);
      const cls = (s.conf > 0 && s.conf < 0.6) ? "seg low" : "seg";
      const title = "conf " + (s.conf * 100).toFixed(0) + "%";
      return '<div class="' + cls + '" title="' + title + '" style="left:' + left +
        '%;top:' + top + '%;width:' + w + '%;height:' + h + '%;font-size:' + fs +
        'cqh;">' + escapeHtml(s.text) + '</div>';
    }).join("\n");
    return '<h2>' + escapeHtml(p.image) + (p.source ? ' (' + p.source + ')' : '') +
      '</h2>\n<div class="page" style="aspect-ratio:' + ar + ';">\n' + segs + '\n</div>';
  }).join("\n");
  return head + body + '\n</body>\n</html>\n';
}
function download(name, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------- control state ---------- */
function setBusy(b) {
  estimateBtn.disabled = b || !pages.length;
  runBtn.disabled = b || nextIndex >= pages.length;
  continueBtn.disabled = b || nextIndex >= pages.length || nextIndex === 0;
  clearBtn.disabled = b;
}
function refreshControls() {
  const has = pages.length > 0;
  const more = nextIndex < pages.length;
  const someDone = pages.some(p => p.status === "done");
  estimateBtn.disabled = !has;
  runBtn.disabled = !more;
  continueBtn.disabled = !more || nextIndex === 0;
  clearBtn.disabled = !has;
  copyBtn.disabled = !someDone;
  txtBtn.disabled = !someDone;
  jsonBtn.disabled = !someDone;
  htmlBtn.disabled = !someDone;
  showJsonBtn.disabled = !someDone;
  showHtmlBtn.disabled = !someDone;
}

function clearAll() {
  pages = []; nextIndex = 0; pdfDocs.length = 0;
  pagesEl.innerHTML = ""; etaEl.textContent = "";
  refreshControls();
  setStatus("Cleared. Choose images or a PDF.");
}

/* ---------- wiring ---------- */
const drop = $("#drop"), fileInput = $("#file");
drop.onclick = () => fileInput.click();
fileInput.onchange = () => { if (fileInput.files.length) accept(fileInput.files); fileInput.value = ""; };
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files.length) accept(e.dataTransfer.files); };
window.addEventListener("paste", e => {
  const it = [...(e.clipboardData && e.clipboardData.items || [])].find(i => i.type.startsWith("image/"));
  if (it) { const f = it.getAsFile(); if (f) accept([f]); }
});

estimateBtn.onclick = estimate;
document.getElementById("tier").onchange = () => {
  if (paddle) { paddle = null; setStatus("Model tier changed - will reload on next run."); }
};
runBtn.onclick = () => runBatch().then(refreshControls);
continueBtn.onclick = () => runBatch().then(refreshControls);
clearBtn.onclick = clearAll;
copyBtn.onclick = async () => {
  try { await navigator.clipboard.writeText(allText()); setStatus("Copied all text to clipboard."); }
  catch (e) { console.error("[copy] failed", e); setStatus("Copy failed: " + (e && e.message || e)); }
};
txtBtn.onclick = () => download("ocr.txt", allText(), "text/plain");
jsonBtn.onclick = () => download("ocr.json", asJson(), "application/json");
htmlBtn.onclick = () => download("ocr.html", asHtml(), "text/html;charset=utf-8");

/* In-page preview overlay: JSON as text, HTML rendered in an iframe so the positioned
   layout is shown as-is. srcdoc is set via property to avoid attribute escaping, and
   the iframe is sandboxed (no allow-* tokens) since the export is static markup only. */
function showOverlay(title, node) {
  ovTitle.textContent = title;
  ovBody.innerHTML = "";
  ovBody.appendChild(node);
  overlay.hidden = false;
}
function hideOverlay() { overlay.hidden = true; ovBody.innerHTML = ""; }
ovClose.onclick = hideOverlay;
overlay.addEventListener("click", e => { if (e.target === overlay) hideOverlay(); });
document.addEventListener("keydown", e => { if (e.key === "Escape" && !overlay.hidden) hideOverlay(); });
showJsonBtn.onclick = () => {
  const pre = document.createElement("pre");
  pre.textContent = asJson();
  showOverlay("ocr.json", pre);
};
showHtmlBtn.onclick = () => {
  const iframe = document.createElement("iframe");
  iframe.setAttribute("sandbox", "");
  iframe.srcdoc = asHtml();
  showOverlay("ocr.html", iframe);
};

pagesEl.addEventListener("click", async e => {
  const reBtn = e.target.closest(".reocr");
  if (reBtn) { await reprocessPage(Number(reBtn.dataset.pg), "ocr"); return; }
  const txBtn = e.target.closest(".retext");
  if (txBtn) { await reprocessPage(Number(txBtn.dataset.pg), "text"); return; }
  const prev = e.target.closest(".preview");
  if (prev) { prev.classList.toggle("zoomed"); return; }
  const btn = e.target.closest(".copy-page");
  if (!btn) return;
  const pg = pages[Number(btn.dataset.pg)];
  if (!pg) return;
  try {
    await navigator.clipboard.writeText(pg.text || "");
    const prev = btn.textContent; btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = prev; }, 1200);
  } catch (err) { console.error("[copy-page] failed", err); setStatus("Copy failed: " + (err && err.message || err)); }
});

const brandMark = $("#brandMark");
if (brandMark) { brandMark.src = FAVICON_MSD; brandMark.onerror = () => brandMark.remove(); }
setFavicon(FAVICON_MSD);   /* rest on the full mark until a batch runs */

setStatus("Ready - choose images or a PDF.");
estimateBtn.disabled = true; runBtn.disabled = true; continueBtn.disabled = true; clearBtn.disabled = true;
</script>
</body>
</html>
