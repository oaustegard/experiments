// The page: a file or PDF in, Indra OCR (indra.js) on the GPU, the page with its lines boxed and
// the markdown out. Logs go through relay.js (?log=1 on Oskar's devices).
import { Indra } from "./indra.js";

const BUILD = "2026-11-18-b"; // keep equal to build.json
const PACK = "models/f16/";
const PDFJS = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";
const PDFJS_WORKER = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";
const DPI = 200, MAX_PAGE_PIXELS = 100e6;

const $ = (id) => document.getElementById(id);
const relay = (window.Relay || { start() { return this; }, log() {}, warn() {}, error() {} }).start({ page: "indra-browser", build: BUILD });
$("build").textContent = BUILD;
fetch("build.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((b) => {
  if (!b || !b.build || b.build === BUILD) return;
  relay.warn("stale page", { running: BUILD, current: b.build });
  let tried = null;
  try { tried = sessionStorage.getItem("indra:reloaded-for"); sessionStorage.setItem("indra:reloaded-for", b.build); } catch (e) {}
  if (tried === b.build) return;
  const url = new URL(location.href);
  url.searchParams.set("b", b.build);
  location.replace(url.href);
}).catch(() => {});

const COLORS = {
  Text: "#3b6fd9", "Section-Header": "#c4610f", "Page-Header": "#8f8f8a", "Page-Footer": "#8f8f8a", "List-Group": "#2b8a3e",
  Table: "#a8328f", Caption: "#13808c", Footnote: "#8f7400", "Table-Of-Contents": "#6b46c1", Form: "#b4372b",
};

function status(text, kind = "", frac = null) {
  const s = $("status");
  s.className = `status ${kind}`;
  $("status-text").textContent = text;
  const bar = $("status-bar");
  bar.hidden = frac === null;
  if (frac !== null) bar.value = frac;
}

let ocr = null, fetched = 0;
async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const b = await r.arrayBuffer();
  fetched += b.byteLength;
  return b;
}

async function boot() {
  try {
    status("Loading the models…", "", null);
    const t0 = performance.now();
    ocr = await Indra.load(PACK, { fetchBytes, log: (m) => relay.log(m) });
    const ms = performance.now() - t0;
    relay.log("ready", { ms: Math.round(ms), gpu: ocr.gpu.label, bytes: fetched });
    status(`Ready: ${(fetched / 1e6).toFixed(2)} MB of models loaded in ${(ms / 1000).toFixed(1)} s (${ocr.gpu.label || "WebGPU"}). Choose a page.`, "ready");
  } catch (e) {
    relay.error(e);
    status(`Could not start: ${e.message}. This page needs WebGPU (recent Chrome, Edge or Safari).`, "error");
  }
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
  return toGray(ctx.getImageData(0, 0, bm.width, bm.height).data, bm.width, bm.height);
}

let pdf = null, pdfPage = 1;
async function pdfToGray(n) {
  const page = await pdf.getPage(n);
  const vp1 = page.getViewport({ scale: 1 });
  const k = Math.min(DPI / 72, Math.sqrt(MAX_PAGE_PIXELS / Math.max(vp1.width * vp1.height, 1)));
  const vp = page.getViewport({ scale: k });
  const w = Math.round(vp.width), h = Math.round(vp.height);
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  await page.render({ canvasContext: ctx, viewport: vp, background: "white" }).promise;
  return toGray(ctx.getImageData(0, 0, w, h).data, w, h);
}

let busy = false, last = null;
async function handle(file) {
  if (!ocr || busy) return;
  pdf = null;
  $("pdf-nav").hidden = true;
  try {
    if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
      status("Opening the PDF…");
      const lib = await import(PDFJS);
      lib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
      pdf = await lib.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise;
      pdfPage = 1;
      relay.log("pdf", { pages: pdf.numPages, bytes: file.size });
      await readPdfPage();
    } else {
      const gray = await imageToGray(file);
      relay.log("image", { w: gray.w, h: gray.h, bytes: file.size, type: file.type });
      await run(gray, file.name);
    }
  } catch (e) {
    relay.error(e);
    status(`Could not read ${file.name}: ${e.message}`, "error");
  }
}

async function readPdfPage() {
  $("pdf-nav").hidden = pdf.numPages < 2;
  $("page-of").textContent = `page ${pdfPage} of ${pdf.numPages}`;
  $("prev").disabled = pdfPage <= 1;
  $("next").disabled = pdfPage >= pdf.numPages;
  status(`Rendering page ${pdfPage}…`);
  await run(await pdfToGray(pdfPage), `page ${pdfPage}`);
}

async function run(gray, name) {
  busy = true;
  try {
    ocr.furniture = $("furniture").checked;
    ocr.noLm = !$("lm").checked;
    status(`Reading ${name} (${gray.w} × ${gray.h})…`, "", null);
    const before = fetched;
    const t0 = performance.now();
    const page = await ocr.read(gray);
    const ms = performance.now() - t0;
    last = { page, gray, name };
    show(page, gray);
    const t = page.timings;
    const extra = fetched - before;
    const msg = `${page.lines.length} lines in ${(ms / 1000).toFixed(2)} s` +
      ` (orientation ${t.orient_ms.toFixed(0)} ms, detection ${t.det_ms.toFixed(0)} ms, reading ${t.rec_ms.toFixed(0)} ms, order ${t.order_ms.toFixed(0)} ms)` +
      (page.rotation ? `; the page was turned ${page.rotation}°` : "") + (page.skew ? `; straightened by ${page.skew}°` : "") +
      (extra ? `; fetched ${(extra / 1e6).toFixed(2)} MB more models for this page` : "");
    relay.log("page", { name, w: gray.w, h: gray.h, lines: page.lines.length, ms: Math.round(ms), timings: t, rotation: page.rotation, skew: page.skew, fetched: extra });
    status(msg, "ready");
    $("stats").textContent = msg;
  } catch (e) {
    relay.error(e);
    status(`Reading failed: ${e.message}`, "error");
  } finally {
    busy = false;
  }
}

// ------------------------------------------------------------------ output

function show(page, gray) {
  $("result").hidden = false;
  $("md").textContent = page.markdown ?? page.text;
  // the page as the boxes see it: turned upright (a vertical page keeps its own frame)
  let w = gray.w, h = gray.h;
  const k = page.vertical ? 0 : ((page.rotation / 90) % 4 + 4) % 4;
  if (k % 2) [w, h] = [h, w];
  const scale = Math.min(1, 1400 / Math.max(w, h));
  const cv = $("canvas");
  cv.width = Math.round(w * scale);
  cv.height = Math.round(h * scale);
  const ctx = cv.getContext("2d");
  const src = new OffscreenCanvas(gray.w, gray.h);
  const id = new ImageData(gray.w, gray.h);
  for (let i = 0; i < gray.data.length; i++) { const v = gray.data[i]; id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = v; id.data[4 * i + 3] = 255; }
  src.getContext("2d").putImageData(id, 0, 0);
  ctx.save();
  ctx.scale(scale, scale);
  // rotation = degrees counter-clockwise the input was turned to make it upright
  if (k === 1) { ctx.translate(0, h); ctx.rotate(-Math.PI / 2); }
  else if (k === 2) { ctx.translate(w, h); ctx.rotate(Math.PI); }
  else if (k === 3) { ctx.translate(w, 0); ctx.rotate(Math.PI / 2); }
  ctx.drawImage(src, 0, 0);
  ctx.restore();
  ctx.lineWidth = Math.max(1, 2 * scale);
  const labelOf = new Map();
  for (const b of page.blocks) {
    ctx.strokeStyle = COLORS[b.label] ?? "#3b6fd9";
    ctx.globalAlpha = 0.9;
    const [x0, y0, x1, y1] = b.bbox.map((v) => v * scale);
    ctx.strokeRect(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4);
    labelOf.set(b.label, true);
  }
  ctx.globalAlpha = 0.25;
  ctx.fillStyle = "#2f5fd0";
  for (const ln of page.lines) {
    const [x0, y0, x1, y1] = ln.box.map((v) => v * scale);
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  }
  ctx.globalAlpha = 1;
  $("legend").innerHTML = [...labelOf.keys()].map((l) => `<span><i style="background:${COLORS[l]}"></i>${l.replace(/-/g, " ")}</span>`).join("");
}

$("copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("md").textContent); $("copy").textContent = "Copied"; setTimeout(() => { $("copy").textContent = "Copy"; }, 1500); } catch (e) { relay.warn("copy failed", { e: String(e) }); }
});
$("save").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([$("md").textContent], { type: "text/markdown" }));
  a.download = `${(last?.name ?? "page").replace(/[^\w.-]+/g, "_")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) handle(f); });
$("prev").addEventListener("click", () => { if (pdf && pdfPage > 1 && !busy) { pdfPage--; readPdfPage(); } });
$("next").addEventListener("click", () => { if (pdf && pdfPage < pdf.numPages && !busy) { pdfPage++; readPdfPage(); } });
for (const id of ["furniture", "lm"]) $(id).addEventListener("change", () => { if (last && !busy) run(last.gray, last.name); });
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) handle(f); });
window.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files ?? [])][0]; if (f) handle(f); });

boot();
