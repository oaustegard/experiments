// The page: an image in, Indra OCR (its layout, reading order and math) on the GPU, every non-math line
// read by Tesseract's LSTM recognizer on WebGPU in one batch, and the text in reading order.
// Logs go through relay.js (?log=1 on Oskar's devices).
import { Indra } from "https://austegard.com/experiments/indra-browser/web/indra.js";
import { rot90 } from "https://austegard.com/experiments/indra-browser/web/imgproc.js";
import { Gpu, recognize } from "./engine.js";
import { loadTraineddata } from "./traineddata.js";

const BUILD = "2026-11-19-tess2"; // keep equal to build.json
const INDRA_MODELS = "https://austegard.com/experiments/indra-browser/web/models/f16/";
const TESS_CDN = (variant) => `https://cdn.jsdelivr.net/gh/tesseract-ocr/tessdata_${variant}@main/`;
const PAD = 3; // px of page around each line's box, clamped to the page
const MATH_CONF = 1; // Indra's math lines carry confidence exactly 1 and LaTeX between $ signs

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

const COLORS = { line: "#2f5fd0", math: "#7a3fb5" };
const ms = (v) => `${Math.round(v)} ms`;

function status(text, kind = "", frac = null) {
  $("status").className = `status ${kind}`;
  $("status-text").textContent = text;
  const bar = $("status-bar");
  bar.hidden = frac === null;
  if (frac !== null) bar.value = frac;
}

// ------------------------------------------------------------------ state

let gpu = null;          // Tesseract's engine device (engine.js)
let ocr = null;          // Indra (indra.js)
let langs = null;        // langs.json
let byCode = new Map();
const models = new Map(); // "variant/path" -> Promise<{model, ms, bytes, variant, code, name}>
let current = null;      // the model in use
let last = null;         // {name, gray, frame, page, indra: {...}, texts, timings}
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
  // cv2's RGB -> GRAY weights, 14-bit fixed point, as indra-browser does
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
  if (!ocr || busy) return;
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

// Indra's boxes are in the frame the page is read in: the input turned by its rotation (indra.js turnPage),
// and a vertical page keeps its own frame. Crops and the drawing both use this frame.
function frameOf(gray, page) {
  if (page.vertical || !page.rotation) return gray;
  return rot90(gray, -(page.rotation / 90));
}

const isMath = (ln) => ln.text.includes("$") && ln.conf === MATH_CONF;

// one crop per line that is not math: axis-aligned, padded, clamped to the page
function cropLines(frame, lines) {
  const idx = [], crops = [];
  lines.forEach((ln, i) => {
    if (isMath(ln)) return;
    const [x0, y0, x1, y1] = ln.box;
    const sx = Math.max(0, Math.floor(x0) - PAD), sy = Math.max(0, Math.floor(y0) - PAD);
    const ex = Math.min(frame.w, Math.ceil(x1) + PAD), ey = Math.min(frame.h, Math.ceil(y1) + PAD);
    const w = ex - sx, h = ey - sy;
    if (w < 2 || h < 2) return;
    const data = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const o = (sy + y) * frame.w + sx;
      data.set(frame.data.subarray(o, o + w), y * w);
    }
    idx.push(i);
    crops.push({ data, width: w, height: h });
  });
  return { idx, crops };
}

async function runPage(gray, name) {
  busy = true;
  try {
    status(`Finding and ordering the lines of ${name} (${gray.w} × ${gray.h})…`);
    const t0 = performance.now();
    const page = await ocr.read(gray);
    const indraMs = performance.now() - t0;
    const frame = frameOf(gray, page);
    last = { name, gray, frame, page, indraMs };
    relay.log("page", { name, w: gray.w, h: gray.h, lines: page.lines.length, ms: Math.round(indraMs), rotation: page.rotation, skew: page.skew, indra: page.timings });
    showPage();
    await recogniseLast();
  } catch (e) {
    relay.error(e);
    status(`Reading failed: ${e.message}`, "error");
  } finally {
    busy = false;
  }
}

// Tesseract reads every non-math line of the page in one recognize() call; the text lands in reading order.
async function recogniseLast() {
  const { page, frame } = last;
  if (!current) return;
  const tc = performance.now();
  const { idx, crops } = cropLines(frame, page.lines);
  const cropMs = performance.now() - tc;
  const t1 = performance.now();
  const out = crops.length ? await recognize(gpu, current.model, crops) : [];
  const recMs = performance.now() - t1;
  const texts = page.lines.map((ln) => (isMath(ln) ? ln.text : ""));
  idx.forEach((i, k) => { texts[i] = (out[k] && out[k].text ? out[k].text : "").trim(); });
  last.texts = texts;
  last.timings = {
    layout: page.timings.orient_ms + page.timings.det_ms + page.timings.order_ms,
    indraRead: page.timings.rec_ms,
    crop: cropMs,
    tess: out.ms ? out.ms.prep + out.ms.gpu + out.ms.readback : recMs,
    decode: out.ms ? out.ms.decode : 0,
  };
  last.counts = { lines: page.lines.length, tess: idx.length, math: page.lines.length - idx.length };
  relay.log("recognize", { code: current.code, variant: current.variant, lines: idx.length, math: last.counts.math, ms: Math.round(recMs), crop_ms: Math.round(cropMs), tess: out.ms, empty: texts.filter((t, i) => !isMath(page.lines[i]) && !t).length });
  showText();
  status(`${last.counts.lines} lines in ${last.name}; ${idx.length} read by Tesseract (${current.name}, ${current.variant}).`, "ready");
}

// ------------------------------------------------------------------ output

function showPage() {
  $("result").hidden = false;
  const frame = last.frame;
  const scale = Math.min(1, 1400 / Math.max(frame.w, frame.h));
  const cv = $("canvas");
  cv.width = Math.round(frame.w * scale);
  cv.height = Math.round(frame.h * scale);
  const ctx = cv.getContext("2d");
  const src = new OffscreenCanvas(frame.w, frame.h);
  const id = new ImageData(frame.w, frame.h);
  for (let i = 0; i < frame.data.length; i++) { const v = frame.data[i]; id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
  src.getContext("2d").putImageData(id, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(src, 0, 0, cv.width, cv.height);
  ctx.lineWidth = Math.max(1, 2 * scale);
  for (const ln of last.page.lines) {
    const [x0, y0, x1, y1] = ln.box.map((v) => v * scale);
    ctx.fillStyle = isMath(ln) ? COLORS.math : COLORS.line;
    ctx.globalAlpha = 0.22;
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.globalAlpha = 0.85;
    ctx.strokeStyle = ctx.fillStyle;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
  ctx.globalAlpha = 1;
  $("legend").innerHTML = `<span><i class="tess"></i>line read by Tesseract</span><span><i class="mathl"></i>math line (Indra's LaTeX)</span>`;
  const info = gpu.info;
  const adapter = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ");
  $("adapter").textContent = `WebGPU adapter: ${adapter || "not named by the browser"}`;
}

function showText() {
  const { page, texts, timings, counts } = last;
  const ol = $("lines");
  ol.innerHTML = "";
  const compare = $("compare").checked;
  page.lines.forEach((ln, i) => {
    const li = document.createElement("li");
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = `${i + 1}`;
    const t = document.createElement("span");
    const math = isMath(ln);
    t.className = `t${math ? " math" : ""}`;
    t.textContent = texts[i] || (math ? ln.text : "");
    li.append(n, t);
    if (compare && !math) {
      const ind = document.createElement("div");
      ind.className = "ind";
      const b = document.createElement("b");
      b.textContent = "Indra";
      ind.append(b, ln.text);
      li.appendChild(ind);
    }
    ol.appendChild(li);
  });
  $("count").textContent = `${counts.lines} lines, ${counts.tess} by Tesseract${counts.math ? `, ${counts.math} math` : ""}`;
  const rows = [
    ["Indra layout (orientation, detection, reading order)", timings.layout],
    ["Indra's own line reader, for comparison", timings.indraRead],
    ["Crop (page to line images)", timings.crop],
    ["Tesseract GPU (prep, kernels, readback)", timings.tess],
    ["Decode (beam search)", timings.decode],
  ];
  $("timings").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${ms(v)}</td></tr>`).join("");
  const text = plainText();
  $("copy").disabled = !text;
  $("stats").textContent = `Page read in ${ms(last.indraMs)} with Indra. Model: ${current.name}, ${current.variant}.`;
}

function plainText() {
  if (!last || !last.texts) return "";
  return last.texts.filter(Boolean).join("\n");
}

$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(plainText());
    $("copy").textContent = "Copied";
    setTimeout(() => { $("copy").textContent = "Copy text"; }, 1500);
  } catch (e) { relay.warn("copy failed", { e: String(e) }); }
});
$("compare").addEventListener("change", () => { if (last && last.texts) showText(); });
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
    status("Loading the layout models and the Tesseract engine…");
    const [index, engineGpu] = await Promise.all([
      fetch("langs.json", { cache: "no-store" }).then((r) => { if (!r.ok) throw new Error(`langs.json: HTTP ${r.status}`); return r.json(); }),
      Gpu.create(),
    ]);
    langs = index;
    byCode = new Map(langs.entries.map((e) => [e.code, e]));
    gpu = engineGpu;
    const t0 = performance.now();
    ocr = await Indra.load(INDRA_MODELS, { log: (m) => relay.log(m), math: "auto" });
    relay.log("layout ready", { ms: Math.round(performance.now() - t0), gpu: gpu.info });
    populatePicker();
    await switchModel();
    $("file").disabled = false;
    relay.log("ready", { build: BUILD, lang: langs.default, adapter: gpu.info });
  } catch (e) {
    relay.error(e);
    status(`Could not start: ${e.message}`, "error");
  }
}
boot();
