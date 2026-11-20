// The page: an image in, its text lines found by Tesseract's own layout (tesslayout.js), each line cleaned up
// (lineprep.js: flat-field, tight crop) and read by Tesseract's LSTM recognizer on WebGPU in one batch, and the text
// in Tesseract's reading order or by column bands (layout.js). Logs go through relay.js (?log=1 on Oskar's devices).
import { Gpu, recognize } from "./engine.js";
import { loadTraineddata } from "./traineddata.js";
import { tessLayout } from "./tesslayout.js";
import { cleanLine } from "./lineprep.js";
import { pageTextLayer, coverage, looseCer } from "./pdftext.js";

// layout.js is a classic (UMD) script, loaded before this module; it sets window.TessLayout.
const orderBoxes = (boxes) => window.TessLayout.orderBoxes(boxes);

const BUILD = "2026-11-20-tess11"; // keep equal to build.json
const TESS_CDN = (variant) => `https://cdn.jsdelivr.net/gh/tesseract-ocr/tessdata_${variant}@main/`;
const PAD = 3; // px of page around each line's box, clamped to the page
const ORDER_LABEL = { tess: "Tesseract's reading order", columns: "column bands" };
const BOX_COLOR = "#2f5fd0";
const LAYER_COLOR = "#1f8a4c"; // lines taken from a PDF's own text layer
const SPOT_LINES = 12;        // text-layer lines OCR'd to check the layer against the page image
const SPOT_MAX_CER = 0.5;     // median disagreement above this rejects the layer
const MISSING = 0.5;          // a Tesseract line box less covered than this by text-layer lines is OCR'd

const $ = (id) => document.getElementById(id);
const relay = (window.Relay || { start() { return this; }, log() {}, warn() {}, error() {} }).start({ page: "tesseract-gpu", build: BUILD });
$("build").textContent = BUILD;

// build.json is fetched past every cache; when it names another build, reload once under a URL no cache holds.
fetch("build.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((b) => {
  if (!b || !b.build || b.build === BUILD) return;
  relay.warn("stale page", { running: BUILD, current: b.build });
  let tried = null;
  try { tried = sessionStorage.getItem("tesseract:reloaded-for"); sessionStorage.setItem("tesseract:reloaded-for", b.build); } catch (e) {}
  if (tried === b.build) return;
  const url = new URL(location.href);
  url.searchParams.set("b", b.build);
  location.replace(url.href);
}).catch(() => {});

const ms = (v) => `${Math.round(v)} ms`;

function status(text, kind = "", frac = null) {
  $("status").className = `status ${kind}`;
  $("status-text").textContent = text;
  const bar = $("status-bar");
  bar.hidden = frac === null;
  if (frac !== null) bar.value = frac;
}

// ------------------------------------------------------------------ state

let gpu = null;          // WebGPU device (engine.js)
let langs = null;        // langs.json
let byCode = new Map();
const models = new Map(); // "variant/path" -> Promise<{model, ms, bytes, variant, code, name}>
let current = null;      // the recogniser model in use
let last = null;         // {name, gray, boxes (reading order, page px), order, clean, findMs, texts, timings, counts}
let layoutCache = null;  // {gray, boxes (Tesseract's order), ms}: one layout per page, reused when only the order or clean-up changes
let runs = 0;            // completed page readings in this load
let busy = false;

// ------------------------------------------------------------------ models

function entryFor(code) { return byCode.get(code); }

function resolveVariant(entry, variant) {
  if (entry[variant]) return variant;
  return variant === "fast" ? "best" : "fast";
}

function getModel(entry, variant) {
  const v = resolveVariant(entry, variant);
  const info = entry[v];
  const key = `${v}/${info.url || info.path}`;
  if (!models.has(key)) {
    const p = (async () => {
      const t0 = performance.now();
      const model = await loadTraineddata(gpu, info.url || TESS_CDN(v) + info.path); // url: a model served with this page (key includes it)
      const took = performance.now() - t0;
      relay.log("model", { code: entry.code, variant: v, bytes: info.bytes, ms: Math.round(took) });
      return { model, ms: took, bytes: info.bytes, variant: v, code: entry.code, name: entry.name, cached: false };
    })();
    models.set(key, p);
    p.catch(() => models.delete(key)); // a failed download is retried on the next choice
  }
  return models.get(key).then((m) => m);
}

function selectedVariant() {
  return document.querySelector('input[name="variant"]:checked').value;
}

function describeModel(m) {
  return `${m.name} · ${m.variant} · ${(m.bytes / 1e6).toFixed(1)} MB download, ${m.cached ? "already loaded" : `loaded in ${ms(m.ms)}`}`;
}

async function switchModel() {
  if (!gpu || !langs) return;
  if (busy) { status("Still reading; choose again in a moment.", ""); return; }
  const entry = entryFor($("lang").value);
  // Word lists help Google's models and cost the fine-tuned one 0.6 points (results/ft_eval.json): langs.json says which.
  $("dict").checked = entry.dict !== false;
  if (last) last.dict = $("dict").checked;
  const variant = selectedVariant();
  const v = resolveVariant(entry, variant);
  const key = `${v}/${entry[v].url || entry[v].path}`;
  const fresh = !models.has(key);
  busy = true;
  try {
    status(`Loading ${entry.name} (${v})…`);
    const m = await getModel(entry, variant);
    if (!fresh) m.cached = true;
    current = m;
    $("model-info").textContent = describeModel(m);
    // A PDF page's text-layer check depends on the model, so such a page is planned again; others are only reread.
    if (last) await (last.layer && last.mode !== "ocr" ? runPage(last.gray, last.name, last.layer) : recogniseLast());
    status(`Ready: ${entry.name} (${v}). ${last ? "Text reread with this model." : "Choose a page."}`, "ready");
  } catch (e) {
    relay.error(e);
    status(`Could not load ${entry.name} (${v}): ${e.message}`, "error");
  } finally {
    busy = false;
  }
}

function populatePicker() {
  const sel = $("lang");
  sel.innerHTML = "";
  const common = new Set(langs.common);
  const group = (label, items) => {
    if (!items.length) return;
    const og = document.createElement("optgroup");
    og.label = label;
    for (const e of items) {
      const o = document.createElement("option");
      o.value = e.code;
      o.textContent = e.name;
      og.appendChild(o);
    }
    sel.appendChild(og);
  };
  const entries = langs.entries;
  const top = langs.common.map((c) => byCode.get(c)).filter(Boolean);
  group("Most common", top);
  group("Other languages", entries.filter((e) => e.group === "Languages" && !common.has(e.code)));
  group("Scripts (one alphabet each)", entries.filter((e) => e.group === "Scripts"));
  sel.value = langs.default;
  sel.disabled = false;
}

// ------------------------------------------------------------------ input

function toGray(rgba, w, h) {
  // cv2's RGB -> GRAY weights, 14-bit fixed point
  const g = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = (rgba[4 * i] * 4899 + rgba[4 * i + 1] * 9617 + rgba[4 * i + 2] * 1868 + 8192) >> 14;
  return { w, h, data: g };
}

async function imageToGray(blob) {
  const bm = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bm.width, bm.height);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, bm.width, bm.height);
  ctx.drawImage(bm, 0, 0);
  const { width: w, height: h } = bm;
  bm.close();
  return toGray(ctx.getImageData(0, 0, w, h).data, w, h);
}

// ------------------------------------------------------------------ PDF
// pdf.js (Mozilla, Apache-2.0) renders each PDF page to a canvas at about PDF_LONG_SIDE pixels on its long side (about
// 240 dpi for a letter page) and gives its text layer (pdftext.js). "PDF text" picks what is read:
//   auto  the text layer where it passes its checks; Tesseract reads only the lines it does not cover, and only on
//         pages that paint an image. A layer that fails falls back to OCR of the whole page.
//   ocr   every page OCR'd like an image, text layer ignored.
//   text  the text layer even when its checks fail, no OCR.
const PDFJS = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/";
const PDF_LONG_SIDE = 2600;
let pdfjsPromise = null;
let pdf = null; // {doc, name, n, page, pdfjs}

function loadPdfjs() {
  if (!pdfjsPromise) {
    pdfjsPromise = import(PDFJS + "pdf.min.mjs").then((m) => {
      m.GlobalWorkerOptions.workerSrc = PDFJS + "pdf.worker.min.mjs";
      return m;
    });
    pdfjsPromise.catch(() => { pdfjsPromise = null; });
  }
  return pdfjsPromise;
}

// The page as grey pixels, and its text layer in those pixels (null when the layer cannot be read).
async function pdfPage(n) {
  const page = await pdf.doc.getPage(n);
  const v1 = page.getViewport({ scale: 1 });
  const scale = PDF_LONG_SIDE / Math.max(v1.width, v1.height);
  const vp = page.getViewport({ scale });
  const c = new OffscreenCanvas(Math.round(vp.width), Math.round(vp.height));
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, c.width, c.height);
  await page.render({ canvasContext: ctx, viewport: vp }).promise;
  const gray = toGray(ctx.getImageData(0, 0, c.width, c.height).data, c.width, c.height);
  let layer = null;
  try {
    layer = await pageTextLayer(pdf.pdfjs, page, vp);
    relay.log("text layer", { page: n, runs: layer.runs, lines: layer.lines.length, chars: layer.verdict.chars, bad: layer.verdict.bad, images: layer.images, ok: layer.verdict.ok, ms: Math.round(layer.ms) });
  } catch (e) {
    relay.warn("text layer failed", { page: n, e: String(e) });
  }
  page.cleanup();
  return { gray, layer };
}

async function readPdfPage(n) {
  if (!pdf || busy) return;
  pdf.page = n;
  $("pdfpage").value = String(n);
  status(`Rendering page ${n} of ${pdf.n} of ${pdf.name}…`);
  const t0 = performance.now();
  const { gray, layer } = await pdfPage(n);
  relay.log("pdf page", { page: n, of: pdf.n, w: gray.w, h: gray.h, ms: Math.round(performance.now() - t0) });
  await runPage(gray, `${pdf.name}, page ${n} of ${pdf.n}`, layer);
}

async function openPdf(file) {
  status(`Opening ${file.name}…`);
  const pdfjs = await loadPdfjs();
  const doc = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
  pdf = { doc, name: file.name, n: doc.numPages, page: 1, pdfjs };
  relay.log("pdf", { pages: doc.numPages, bytes: file.size });
  const sel = $("pdfpage");
  sel.innerHTML = Array.from({ length: doc.numPages }, (_, i) => `<option value="${i + 1}">Page ${i + 1} of ${doc.numPages}</option>`).join("");
  $("pdfnav").hidden = doc.numPages < 2;
  $("pdfmodes").hidden = false;
  await readPdfPage(1);
}

async function handle(file) {
  if (!gpu || busy) return;
  const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
  if (!isPdf && !/^image\//.test(file.type)) {
    status(`${file.name} is neither an image nor a PDF.`, "error");
    return;
  }
  try {
    if (isPdf) return await openPdf(file);
    pdf = null;
    $("pdfnav").hidden = true;
    $("pdfmodes").hidden = true;
    const gray = await imageToGray(file);
    relay.log("image", { w: gray.w, h: gray.h, bytes: file.size, type: file.type });
    await runPage(gray, file.name);
  } catch (e) {
    relay.error(e);
    status(`Could not read ${file.name}: ${e.message}`, "error");
  }
}

// ------------------------------------------------------------------ reading

// Tesseract's layout returns some "lines" that are not one text line: table rules (a few px wide, hundreds tall),
// logos and banners, cells holding several lines. Drop boxes too narrow to hold text, and split any box taller than
// 1.8 median line heights at its empty or ruled pixel rows; pieces shorter than 0.4 median heights are dropped; then
// drop duplicates (dedupe).
function medianOf(v) { const s = v.slice().sort((a, b) => a - b); return s.length ? s[s.length >> 1] : 0; }

function refineBoxes(gray, boxes) {
  const med = medianOf(boxes.map((b) => b[3] - b[1])) || 20;
  const out = [];
  for (const b of boxes) {
    const [x0, y0, x1, y1] = b.map(Math.round);
    const w = x1 - x0, h = y1 - y0;
    if (w < 1.2 * med) continue;
    if (h <= 1.8 * med) { out.push(b); continue; }
    let lo = 255, hi = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const v = gray.data[y * gray.w + x]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const thr = (lo + hi) / 2;
    const ink = [];
    for (let y = y0; y < y1; y++) {
      let n = 0;
      for (let x = x0; x < x1; x++) if (gray.data[y * gray.w + x] < thr) n++;
      ink.push(n > 0.002 * w && n < 0.6 * w); // a row that is mostly ink is a ruling, not text
    }
    let start = -1;
    for (let i = 0; i <= ink.length; i++) {
      if (i < ink.length && ink[i]) { if (start < 0) start = i; continue; }
      if (start >= 0 && i - start >= 0.4 * med) out.push([x0, Math.max(y0, y0 + start - 2), x1, Math.min(y1, y0 + i + 2)]);
      start = -1;
    }
  }
  return dedupe(out);
}

// Table regions come back twice, as rows and again inside a table-wide block: where two boxes overlap by more than
// half of the smaller, keep the smaller (tighter) one. Reading order is kept.
function dedupe(boxes) {
  const area = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const inter = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const idx = boxes.map((_, i) => i).sort((i, j) => area(boxes[i]) - area(boxes[j]));
  const kept = [];
  for (const i of idx) {
    if (!kept.some((k) => inter(boxes[i], boxes[k]) > 0.5 * Math.min(area(boxes[i]), area(boxes[k])))) kept.push(i);
  }
  const keep = new Set(kept);
  return boxes.filter((_, i) => keep.has(i));
}

// Table rules inside a line read as runs of =, ~, —, _ or |: drop tokens made only of such characters, and drop a
// line with fewer than three letters or digits (a speck or a fragment of a ruling).
const RULE_TOKEN = /^[=~—–_|§«»‘’“”'".,:;·•\-]+$/;
function tidy(t) {
  const kept = t.split(/\s+/).filter((w) => w && !RULE_TOKEN.test(w)).join(" ");
  return (kept.match(/[\p{L}\p{N}]/gu) || []).length >= 3 ? kept : "";
}

// Line boxes in reading order, as [x0, y0, x1, y1] in page pixels.
async function findLines(gray, order) {
  if (!layoutCache || layoutCache.gray !== gray) {
    const t0 = performance.now();
    const r = await tessLayout(gray.data, gray.w, gray.h);
    const took = performance.now() - t0;
    relay.log("tess layout", { lines: r.lines.length, ms: Math.round(took), core_ms: Math.round(r.ms) });
    const raw = r.lines.map((ln) => ln.box);
    const boxes = refineBoxes(gray, raw);
    relay.log("refine", { raw: raw.length, lines: boxes.length });
    layoutCache = { gray, boxes, ms: took };
  }
  const { boxes, ms: took } = layoutCache;
  if (order !== "columns") return { boxes, ms: took };
  const t1 = performance.now();
  const seq = orderBoxes(boxes);
  const ordMs = performance.now() - t1;
  relay.log("order", { order, lines: boxes.length, ms: Math.round(ordMs) });
  return { boxes: seq.map((i) => boxes[i]), ms: took + ordMs };
}

// One crop per line: axis-aligned, padded by PAD px, clamped to the page. null for a box with no area.
function cropOf(gray, box) {
  const [x0, y0, x1, y1] = box;
  const sx = Math.max(0, Math.floor(x0) - PAD), sy = Math.max(0, Math.floor(y0) - PAD);
  const ex = Math.min(gray.w, Math.ceil(x1) + PAD), ey = Math.min(gray.h, Math.ceil(y1) + PAD);
  const w = ex - sx, h = ey - sy;
  if (w < 2 || h < 2) return null;
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = (sy + y) * gray.w + sx;
    data.set(gray.data.subarray(o, o + w), y * w);
  }
  return { data, width: w, height: h };
}

// OCR a sample of text-layer lines and compare: a layer whose glyphs map to the wrong characters (a font with a broken
// ToUnicode map, or an invisible OCR layer that was wrong) disagrees with the page image on nearly every line. The
// verdict is kept per layer and model. With a model for another script every line disagrees too, so the page then
// falls back to OCR with that model, which is what the reader asked for.
async function spotCheck(gray, layer, clean) {
  const key = `${current.code}/${current.variant}/${clean}`;
  if (layer.spot && layer.spot.key === key) return { ...layer.spot, ms: 0 };
  const t0 = performance.now();
  const cands = layer.lines.map((l, i) => i).filter((i) => layer.lines[i].text.replace(/\s/g, "").length >= 8);
  const pick = [];
  for (let k = 0; k < Math.min(SPOT_LINES, cands.length); k++) pick.push(cands[Math.floor((k * cands.length) / Math.min(SPOT_LINES, cands.length))]);
  if (!pick.length) return (layer.spot = { key, ok: true, why: "", n: 0, cer: null, ms: 0 });
  const crops = [], used = [];
  for (const i of pick) { const c = cropOf(gray, layer.lines[i].box); if (c) { crops.push(clean ? cleanLine(c) : c); used.push(i); } }
  const out = crops.length ? await recognize(gpu, current.model, crops, { dict: $("dict").checked }) : [];
  const cers = used.map((i, k) => looseCer(layer.lines[i].text, (out[k] && out[k].text) || "")).sort((a, b) => a - b);
  const med = cers.length ? cers[cers.length >> 1] : 0;
  const ok = med <= SPOT_MAX_CER;
  const took = performance.now() - t0;
  layer.spot = { key, ok, n: cers.length, cer: med, ms: took, why: ok ? "" : `the text layer disagrees with the page image (median CER ${(100 * med).toFixed(0)}% on ${cers.length} lines)` };
  relay.log("text layer check", { lines: cers.length, median_cer: Number(med.toFixed(3)), ok, model: key, ms: Math.round(took) });
  return layer.spot;
}

// Lines of a PDF page from its text layer, plus Tesseract's lines the layer does not cover (only on pages that paint
// an image, where text inside the image is not in the layer). Returns null when the page should be OCR'd instead.
async function layerLines(gray, layer, mode, order, clean) {
  if (mode === "ocr" || !layer || !layer.lines.length) return null;
  let spotMs = 0;
  if (mode === "auto") {
    if (!layer.verdict.ok) return { reject: layer.verdict.why, spotMs };
    const spot = await spotCheck(gray, layer, clean);
    spotMs = spot.ms;
    if (!spot.ok) return { reject: spot.why, spotMs };
  }
  const text = layer.lines;
  let extra = [], findMs = 0;
  if (mode === "auto" && layer.images) {
    const found = await findLines(gray, "tess");
    findMs = found.ms;
    const tb = text.map((l) => l.box);
    extra = found.boxes.filter((b) => coverage(b, tb) < MISSING);
  }
  // Order: column bands over every line, or the PDF's own order with each OCR'd line placed before the first
  // text-layer line that starts below it.
  let items = text.map((l) => ({ box: l.box, text: l.text, src: "layer" }));
  const ocr = extra.map((b) => ({ box: b, text: null, src: "ocr" }));
  if (order === "columns") {
    const all = items.concat(ocr);
    items = orderBoxes(all.map((x) => x.box)).map((i) => all[i]);
  } else {
    for (const o of ocr) {
      const at = items.findIndex((x) => x.src === "layer" && x.box[1] > o.box[1]);
      if (at < 0) items.push(o); else items.splice(at, 0, o);
    }
  }
  return { items, findMs, spotMs, ocr: ocr.length };
}

async function runPage(gray, name, layer = null) {
  busy = true;
  try {
    const order = $("order").value;
    const clean = $("clean").checked;
    const dict = $("dict").checked;
    const mode = layer ? $("pdfmode").value : "ocr";
    let plan = null, note = "", spotMs = 0;
    if (layer && mode !== "ocr" && layer.lines.length) {
      status(`Checking the text layer of ${name}…`);
      plan = await layerLines(gray, layer, mode, order, clean);
      if (plan) spotMs = plan.spotMs;
      if (plan && plan.reject) { note = `Text layer not used: ${plan.reject}.`; plan = null; }
    } else if (layer && mode !== "ocr") note = "This page has no text layer.";
    if (plan) {
      last = {
        name, gray, order, clean, dict, layer, mode, note,
        boxes: plan.items.map((x) => x.box), src: plan.items.map((x) => x.src), fixed: plan.items.map((x) => x.text),
        findMs: plan.findMs, layerMs: layer.ms + spotMs, spotMs,
      };
    } else {
      status(`Finding the lines of ${name} with Tesseract's layout (${gray.w} × ${gray.h})…`);
      const found = await findLines(gray, order);
      last = {
        name, gray, order, clean, dict, layer, mode, note,
        boxes: found.boxes, src: found.boxes.map(() => "ocr"), fixed: found.boxes.map(() => null),
        findMs: found.ms, layerMs: layer ? layer.ms + spotMs : 0, spotMs,
      };
    }
    const fromLayer = last.src.filter((s) => s === "layer").length;
    relay.log("page", { name, w: gray.w, h: gray.h, lines: last.boxes.length, from_layer: fromLayer, mode, order, clean, note, ms: Math.round(last.findMs) });
    showPage();
    await recogniseLast();
  } catch (e) {
    relay.error(e);
    status(`Reading failed: ${e.message}`, "error");
  } finally {
    busy = false;
  }
}

// Tesseract reads every line of the page that needs OCR in one recognize() call; lines from a PDF's text layer keep
// their text. The text lands in reading order.
async function recogniseLast() {
  if (!last || !current) return;
  const { gray, boxes } = last;
  const tc = performance.now();
  const idx = [], crops = [];
  boxes.forEach((b, i) => {
    if (last.src[i] !== "ocr") return;
    const c = cropOf(gray, b);
    if (c) { idx.push(i); crops.push(last.clean ? cleanLine(c) : c); }
  });
  const cropMs = performance.now() - tc;
  const t1 = performance.now();
  const out = crops.length ? await recognize(gpu, current.model, crops, { dict: last.dict }) : [];
  const recMs = performance.now() - t1;
  const texts = boxes.map((_, i) => (last.fixed[i] === null ? "" : last.fixed[i]));
  idx.forEach((i, k) => { texts[i] = tidy(((out[k] && out[k].text) || "").trim()); });
  last.texts = texts;
  last.timings = {
    find: last.findMs,
    crop: cropMs,
    gpu: out.ms ? out.ms.prep + out.ms.gpu + out.ms.readback : recMs,
    decode: out.ms ? out.ms.decode : 0,
    layer: last.layerMs,
  };
  last.counts = { lines: boxes.length, read: idx.length, layer: boxes.length - last.src.filter((s) => s === "ocr").length, text: texts.filter(Boolean).length };
  relay.log("recognize", {
    code: current.code, variant: current.variant, lines: idx.length, with_text: last.counts.text,
    crop_ms: Math.round(cropMs), rec_ms: Math.round(recMs), tess: out.ms,
  });
  runs += 1;
  $("result").dataset.run = String(runs);
  showText();
  const how = last.counts.layer
    ? (idx.length ? `${last.counts.layer} from the PDF's text layer, ${idx.length} read with ${current.name} (${current.variant})` : "all from the PDF's text layer, no OCR")
    : `read with ${current.name} (${current.variant})`;
  status(`${boxes.length} lines in ${last.name}; ${how}.${last.note ? " " + last.note : ""}`, "ready");
}

// ------------------------------------------------------------------ output

function showPage() {
  $("result").hidden = false;
  const { gray, boxes } = last;
  const scale = Math.min(1, 1400 / Math.max(gray.w, gray.h));
  const cv = $("canvas");
  cv.width = Math.round(gray.w * scale);
  cv.height = Math.round(gray.h * scale);
  const ctx = cv.getContext("2d");
  const src = new OffscreenCanvas(gray.w, gray.h);
  const id = new ImageData(gray.w, gray.h);
  for (let i = 0; i < gray.data.length; i++) { const v = gray.data[i]; id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
  src.getContext("2d").putImageData(id, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(src, 0, 0, cv.width, cv.height);
  const fs = Math.max(11, Math.round(13 * scale));
  ctx.font = `600 ${fs}px system-ui, -apple-system, sans-serif`;
  ctx.textBaseline = "top";
  ctx.lineWidth = Math.max(1, 2 * scale);
  boxes.forEach((b, i) => {
    const [x0, y0, x1, y1] = b.map((v) => v * scale);
    const color = last.src[i] === "layer" ? LAYER_COLOR : BOX_COLOR;
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = color;
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const label = String(i + 1);
    const tw = ctx.measureText(label).width + 6, th = fs + 4;
    ctx.globalAlpha = 0.95;
    ctx.fillRect(x0, y0, tw, th);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x0 + 3, y0 + 2);
  });
  ctx.globalAlpha = 1;
  const nLayer = last.src.filter((s) => s === "layer").length;
  $("legend").innerHTML = nLayer
    ? `<span><i class="layer"></i>line from the PDF's text layer</span>${nLayer < boxes.length ? `<span><i class="box"></i>line OCR'd (not in the text layer)</span>` : ""}<span>numbered in ${last.order === "columns" ? "column bands" : "the PDF's order"}</span>`
    : `<span><i class="box"></i>line box, numbered in reading order</span><span>found by Tesseract's layout, ${ORDER_LABEL[last.order]}</span>`;
  const info = gpu.info;
  const adapter = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ");
  $("adapter").textContent = `WebGPU adapter: ${adapter || "not named by the browser"}`;
}

function showText() {
  const { texts, timings, counts } = last;
  const ol = $("lines");
  ol.innerHTML = "";
  texts.forEach((t, i) => {
    const li = document.createElement("li");
    li.dataset.src = last.src[i];
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = `${i + 1}`;
    const tx = document.createElement("span");
    tx.className = t ? "t" : "t empty";
    tx.textContent = t || "no text";
    li.append(n, tx);
    ol.appendChild(li);
  });
  $("count").textContent = `${counts.lines} lines, ${counts.text} with text${counts.layer ? `, ${counts.layer} from the text layer` : ""}`;
  const rows = [];
  if (last.layer) rows.push([last.spotMs ? "PDF text layer (read, grouped into lines, a sample OCR'd to check it)" : "PDF text layer (read, grouped into lines)", timings.layer]);
  if (counts.read || !counts.layer) {
    rows.push(
      [counts.layer ? "Line finding: Tesseract layout, lines the text layer misses" : `Line finding: Tesseract layout, ${ORDER_LABEL[last.order]}`, timings.find],
      [last.clean ? "Crop and clean-up (flat-field, tight crop)" : "Crop (page to line images, 3 px padding)", timings.crop],
      ["Recognition on GPU (prep, kernels, readback)", timings.gpu],
      [last.dict && current.model.dict ? "Decode (beam search with Tesseract's word lists)" : "Decode (beam search)", timings.decode],
    );
  }
  $("timings").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${ms(v)}</td></tr>`).join("");
  $("copy").disabled = !plainText();
  $("export").disabled = !counts.lines;
  $("stats").textContent = `Page ${last.gray.w} × ${last.gray.h} px. Recognition model: ${current.name}, ${current.variant}.${last.note ? " " + last.note : ""}`;
}

function plainText() {
  if (!last || !last.texts) return "";
  return last.texts.filter(Boolean).join("\n");
}

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Absolutely positioned text layer, as anything-to-text's asHtml writes it: one div per line, placed by percentage of
// the page so the markup carries the geometry; font size follows the box height through container query units.
function asHtml() {
  const { gray, boxes, texts, name } = last;
  const W = gray.w, H = gray.h;
  const medH = medianOf(boxes.map((b) => b[3] - b[1])) || 20;
  const segs = boxes.map((b, i) => {
    const [x0, y0, x1, y1] = b;
    const left = (x0 / W * 100).toFixed(3), top = (y0 / H * 100).toFixed(3);
    const w = ((x1 - x0) / W * 100).toFixed(3), h = ((y1 - y0) / H * 100).toFixed(3);
    // Font size: fit the box's height and its width (about 0.55 em a character), never above 2x the median line.
    const hFit = (y1 - y0) * 0.78, wFit = (x1 - x0) / (0.55 * Math.max(1, texts[i].length)), cap = 2 * medH * 0.78;
    const fs = (Math.min(hFit, wFit, cap) / H * 100).toFixed(3);
    return `<div class="seg" title="line ${i + 1}" style="left:${left}%;top:${top}%;width:${w}%;height:${h}%;font-size:${fs}cqh;">${escapeHtml(texts[i])}</div>`;
  }).join("\n");
  const nLayer = last.src.filter((s) => s === "layer").length;
  const ocrd = `read by Tesseract ${current.name} (${current.variant})`;
  const from = !nLayer ? ocrd : nLayer === boxes.length ? "the PDF's text layer" : `the PDF's text layer; other lines ${ocrd}`;
  return '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n' +
    `<title>${escapeHtml(name)}: positioned text</title>\n<style>\n` +
    'body{margin:0;background:#525659;font-family:Arial,Helvetica,sans-serif;}\n' +
    'h2{color:#e8e8e8;font:13px Arial;font-weight:normal;margin:16px auto 6px;max-width:900px;}\n' +
    '.page{position:relative;width:100%;max-width:900px;margin:0 auto 24px;background:#fff;\n' +
    '  border:1px solid #222;box-shadow:0 1px 6px rgba(0,0,0,0.4);container-type:size;}\n' +
    '.seg{position:absolute;white-space:nowrap;overflow:visible;line-height:1;color:#111;}\n' +
    '</style>\n</head>\n<body>\n' +
    `<h2>${escapeHtml(name)} (${escapeHtml(from)})</h2>\n` +
    `<div class="page" style="aspect-ratio:${W} / ${H};">\n${segs}\n</div>\n` +
    '</body>\n</html>\n';
}

function download(name, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(plainText());
    $("copy").textContent = "Copied";
    setTimeout(() => { $("copy").textContent = "Copy text"; }, 1500);
  } catch (e) { relay.warn("copy failed", { e: String(e) }); }
});
$("export").addEventListener("click", () => {
  if (!last || !last.texts) return;
  const base = last.name.replace(/\.[^.]+$/, "") || "page";
  download(`${base}-positioned.html`, asHtml(), "text/html;charset=utf-8");
  relay.log("export", { lines: last.boxes.length, with_text: last.counts.text });
});

// Changing the order or the clean-up rereads the page already open; Tesseract's layout is reused.
function rereadLast() {
  if (busy) { status("Still reading; choose again in a moment.", ""); return; }
  if (last) runPage(last.gray, last.name, last.layer);
}
$("clean").addEventListener("change", rereadLast);
$("dict").addEventListener("change", rereadLast);
$("order").addEventListener("change", rereadLast);
$("pdfmode").addEventListener("change", rereadLast);
$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) handle(f); });
$("pdfpage").addEventListener("change", (e) => readPdfPage(Number(e.target.value)).catch((err) => { relay.error(err); status(`Could not read that page: ${err.message}`, "error"); }));
$("pdfprev").addEventListener("click", () => { if (pdf && pdf.page > 1) readPdfPage(pdf.page - 1); });
$("pdfnext").addEventListener("click", () => { if (pdf && pdf.page < pdf.n) readPdfPage(pdf.page + 1); });
$("lang").addEventListener("change", switchModel);
document.querySelectorAll('input[name="variant"]').forEach((r) => r.addEventListener("change", switchModel));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) handle(f); });
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) handle(item.getAsFile());
});

// ------------------------------------------------------------------ start

async function boot() {
  if (!navigator.gpu) {
    relay.warn("no WebGPU", { ua: navigator.userAgent });
    status("This page needs WebGPU, which this browser does not offer. Use a recent Chrome, Edge or Safari with hardware acceleration on.", "error");
    $("file").disabled = true;
    return;
  }
  try {
    status("Loading the Tesseract engine…");
    const [index, engineGpu] = await Promise.all([
      fetch("langs.json", { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`langs.json: HTTP ${r.status}`); return r.json(); }),
      Gpu.create(),
    ]);
    langs = index;
    byCode = new Map(langs.entries.map((e) => [e.code, e]));
    gpu = engineGpu;
    populatePicker();
    await switchModel();
    $("file").disabled = false;
    // Warm Tesseract's layout engine (about 7 MB with its model) while the reader picks a page.
    tessLayout(new Uint8Array(32 * 32).fill(255), 32, 32).catch((e) => relay.warn("layout warm-up failed", { e: String(e) }));
    relay.log("ready", { build: BUILD, lang: langs.default, adapter: gpu.info });
  } catch (e) {
    relay.error(e);
    status(`Could not start: ${e.message}`, "error");
  }
}
boot();
