// The page: an image in, its text lines found on the GPU (PP-OCRv6's detector with layout.js, or Tesseract's own
// layout from tesslayout.js), every line read by Tesseract's LSTM recognizer on WebGPU in one batch, and the text in
// reading order. Every component is Apache-2.0 or MIT. Logs go through relay.js (?log=1 on Oskar's devices).
import { Gpu, recognize } from "./engine.js";
import { loadTraineddata } from "./traineddata.js";
import { DetNet } from "./det/engine.js";
import { tessLayout } from "./tesslayout.js";

// layout.js is a classic (UMD) script, loaded before this module; it sets window.TessLayout.
const detLayout = (...args) => window.TessLayout.layout(...args);

const BUILD = "2026-11-20-tess4"; // keep equal to build.json
const DET_BASE = "det/models/tiny/";
const DET_LONG_SIDE = 1280;
const TESS_CDN = (variant) => `https://cdn.jsdelivr.net/gh/tesseract-ocr/tessdata_${variant}@main/`;
const PAD = 3; // px of page around each line's box, clamped to the page
const FINDER_LABEL = { detector: "Detector (PP-OCRv6)", tess: "Tesseract layout" };
const ORDER_LABEL = { bands: "columns", xycut: "XY-cut" };
const BOX_COLOR = "#2f5fd0";

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
let detNet = null;       // PP-OCRv6 detector (det/engine.js)
let langs = null;        // langs.json
let byCode = new Map();
const models = new Map(); // "variant/path" -> Promise<{model, ms, bytes, variant, code, name}>
let current = null;      // the recogniser model in use
let last = null;         // {name, gray, boxes (reading order, page px), finder, order, findMs, texts, timings, counts}
let detCache = null;     // {gray, prob, H, W, scale}: the detector's map for one page, reused when only the order changes
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
  const key = `${v}/${info.path}`;
  if (!models.has(key)) {
    const p = (async () => {
      const t0 = performance.now();
      const model = await loadTraineddata(gpu, TESS_CDN(v) + info.path);
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
  const variant = selectedVariant();
  const v = resolveVariant(entry, variant);
  const key = `${v}/${entry[v].path}`;
  const fresh = !models.has(key);
  busy = true;
  try {
    status(`Loading ${entry.name} (${v})…`);
    const m = await getModel(entry, variant);
    if (!fresh) m.cached = true;
    current = m;
    $("model-info").textContent = describeModel(m);
    if (last) await recogniseLast();
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

async function handle(file) {
  if (!detNet || busy) return;
  if (!/^image\//.test(file.type)) {
    status(`${file.name} is not an image. Save the page as PNG or JPEG first (PDF pages are not read by this page yet).`, "error");
    return;
  }
  try {
    const gray = await imageToGray(file);
    relay.log("image", { w: gray.w, h: gray.h, bytes: file.size, type: file.type });
    await runPage(gray, file.name);
  } catch (e) {
    relay.error(e);
    status(`Could not read ${file.name}: ${e.message}`, "error");
  }
}

// ------------------------------------------------------------------ reading

// Line boxes in reading order, as [x0, y0, x1, y1] in page pixels.
async function findLines(gray, finder, order) {
  if (finder === "tess") {
    const t0 = performance.now();
    const r = await tessLayout(gray.data, gray.w, gray.h);
    const took = performance.now() - t0;
    relay.log("tess layout", { lines: r.lines.length, ms: Math.round(took), core_ms: Math.round(r.ms) });
    return { boxes: r.lines.map((ln) => ln.box), ms: took };
  }
  // The detector's time is kept with its map, so a reread that only changes the order still reports what finding the lines cost.
  if (!detCache || detCache.gray !== gray) {
    const t0 = performance.now();
    const out = await detNet.run(gray.data, gray.w, gray.h, DET_LONG_SIDE);
    const detMs = performance.now() - t0;
    detCache = { gray, prob: out.prob, H: out.H, W: out.W, scale: out.scale, ms: detMs };
    relay.log("det", { w: gray.w, h: gray.h, H: out.H, W: out.W, ms: Math.round(detMs) });
  }
  const t1 = performance.now();
  const { prob, H, W, scale } = detCache;
  // scale is map pixels per page pixel (det/README.md), so the layout takes page pixels per map pixel
  const res = detLayout(prob, H, W, 1 / scale, 1 / scale, gray.w, gray.h, { order });
  const seq = res.order.slice();
  const seen = new Set(seq);
  for (let i = 0; i < res.lines.length; i++) if (!seen.has(i)) seq.push(i); // a line the order missed goes last
  const boxes = seq.map((i) => res.lines[i].box);
  const layoutMs = performance.now() - t1;
  relay.log("layout", { order, lines: boxes.length, blocks: res.blocks.length, skew: res.skew, ms: Math.round(layoutMs) });
  return { boxes, ms: detCache.ms + layoutMs };
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

async function runPage(gray, name) {
  busy = true;
  try {
    const finder = $("finder").value;
    const order = $("order").value;
    status(`Finding the lines of ${name} with ${FINDER_LABEL[finder]} (${gray.w} × ${gray.h})…`);
    const found = await findLines(gray, finder, order);
    last = { name, gray, boxes: found.boxes, finder, order, findMs: found.ms };
    relay.log("page", { name, w: gray.w, h: gray.h, lines: found.boxes.length, finder, order, ms: Math.round(found.ms) });
    showPage();
    await recogniseLast();
  } catch (e) {
    relay.error(e);
    status(`Reading failed: ${e.message}`, "error");
  } finally {
    busy = false;
  }
}

// Tesseract reads every line of the page in one recognize() call; the text lands in reading order.
async function recogniseLast() {
  if (!last || !current) return;
  const { gray, boxes } = last;
  const tc = performance.now();
  const idx = [], crops = [];
  boxes.forEach((b, i) => {
    const c = cropOf(gray, b);
    if (c) { idx.push(i); crops.push(c); }
  });
  const cropMs = performance.now() - tc;
  const t1 = performance.now();
  const out = crops.length ? await recognize(gpu, current.model, crops) : [];
  const recMs = performance.now() - t1;
  const texts = boxes.map(() => "");
  idx.forEach((i, k) => { texts[i] = ((out[k] && out[k].text) || "").trim(); });
  last.texts = texts;
  last.timings = {
    find: last.findMs,
    crop: cropMs,
    gpu: out.ms ? out.ms.prep + out.ms.gpu + out.ms.readback : recMs,
    decode: out.ms ? out.ms.decode : 0,
  };
  last.counts = { lines: boxes.length, read: idx.length, text: texts.filter(Boolean).length };
  relay.log("recognize", {
    code: current.code, variant: current.variant, lines: idx.length, with_text: last.counts.text,
    crop_ms: Math.round(cropMs), rec_ms: Math.round(recMs), tess: out.ms,
  });
  runs += 1;
  $("result").dataset.run = String(runs);
  showText();
  status(`${boxes.length} lines in ${last.name}, found by ${FINDER_LABEL[last.finder]}; read with ${current.name} (${current.variant}).`, "ready");
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
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = BOX_COLOR;
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.globalAlpha = 0.9;
    ctx.strokeStyle = BOX_COLOR;
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
  $("legend").innerHTML = `<span><i class="box"></i>line box, numbered in reading order</span><span>found by ${FINDER_LABEL[last.finder]}${last.finder === "detector" ? `, ${ORDER_LABEL[last.order]}` : ""}</span>`;
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
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = `${i + 1}`;
    const tx = document.createElement("span");
    tx.className = t ? "t" : "t empty";
    tx.textContent = t || "no text";
    li.append(n, tx);
    ol.appendChild(li);
  });
  $("count").textContent = `${counts.lines} lines, ${counts.text} with text`;
  const finding = last.finder === "detector" ? `Line finding: detector, ${ORDER_LABEL[last.order]}` : "Line finding: Tesseract layout";
  const rows = [
    [finding, timings.find],
    ["Crop (page to line images, 3 px padding)", timings.crop],
    ["Recognition on GPU (prep, kernels, readback)", timings.gpu],
    ["Decode (beam search)", timings.decode],
  ];
  $("timings").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${ms(v)}</td></tr>`).join("");
  $("copy").disabled = !plainText();
  $("export").disabled = !counts.lines;
  $("stats").textContent = `Page ${last.gray.w} × ${last.gray.h} px. Recognition model: ${current.name}, ${current.variant}.`;
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
  const segs = boxes.map((b, i) => {
    const [x0, y0, x1, y1] = b;
    const left = (x0 / W * 100).toFixed(3), top = (y0 / H * 100).toFixed(3);
    const w = ((x1 - x0) / W * 100).toFixed(3), h = ((y1 - y0) / H * 100).toFixed(3);
    const fs = ((y1 - y0) / H * 100 * 0.78).toFixed(3);
    return `<div class="seg" title="line ${i + 1}" style="left:${left}%;top:${top}%;width:${w}%;height:${h}%;font-size:${fs}cqh;">${escapeHtml(texts[i])}</div>`;
  }).join("\n");
  const from = `lines by ${FINDER_LABEL[last.finder]}, read by Tesseract ${current.name} (${current.variant})`;
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

function syncOrder() {
  $("order").disabled = $("finder").value !== "detector";
}
// Changing how lines are found or ordered rereads the page already open; the detector's map is reused.
function rereadLast() {
  if (busy) { status("Still reading; choose again in a moment.", ""); return; }
  if (last) runPage(last.gray, last.name);
}
$("finder").addEventListener("change", () => { syncOrder(); rereadLast(); });
$("order").addEventListener("change", rereadLast);
$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) handle(f); });
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
    status("Loading the line detector and the Tesseract engine…");
    const [index, engineGpu] = await Promise.all([
      fetch("langs.json", { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`langs.json: HTTP ${r.status}`); return r.json(); }),
      Gpu.create(),
    ]);
    langs = index;
    byCode = new Map(langs.entries.map((e) => [e.code, e]));
    gpu = engineGpu;
    const t0 = performance.now();
    detNet = await DetNet.load(gpu, DET_BASE);
    relay.log("detector ready", { ms: Math.round(performance.now() - t0), base: DET_BASE, gpu: gpu.info });
    populatePicker();
    syncOrder();
    await switchModel();
    $("file").disabled = false;
    relay.log("ready", { build: BUILD, lang: langs.default, adapter: gpu.info });
  } catch (e) {
    relay.error(e);
    status(`Could not start: ${e.message}`, "error");
  }
}
boot();
