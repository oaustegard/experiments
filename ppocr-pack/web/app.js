// Demo page: PP-OCRv6 tiny or small, packed weights, WGSL on WebGPU. Logs go through relay.js (?log=1 on Oskar's devices).
import { Gpu } from "./det/gpu.js";
import { DetNet } from "./det/engine.js";
import { RecNet } from "./rec/engine.js";
import { TextPipeline } from "./pipeline.js";

const BUILD = "2026-11-24-a"; // keep equal to build.json
const $ = (id) => document.getElementById(id);
const relay = (window.Relay || { start() { return this; }, log() {}, warn() {}, error() {} }).start({ page: "ppocr-pack", build: BUILD });
$("build").textContent = BUILD;
fetch("build.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((b) => {
  if (!b || !b.build || b.build === BUILD) return;
  relay.warn("stale page", { running: BUILD, current: b.build });
  let tried = null;
  try { tried = sessionStorage.getItem("ppocr:reloaded-for"); sessionStorage.setItem("ppocr:reloaded-for", b.build); } catch (e) {}
  if (tried === b.build) return;
  const url = new URL(location.href);
  url.searchParams.set("b", b.build);
  location.replace(url.toString());
}).catch(() => {});

const status = (msg, err) => { const s = $("status"); s.textContent = msg; s.className = err ? "err" : "muted"; };
const fmtB = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + " MB" : (n / 1e3).toFixed(0) + " kB");
const ms = (v) => (v == null ? "" : v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");

let gpu = null;
const nets = {};      // size -> { pipe, loadMs, bytes }
let image = null;     // { canvas, name }
let busy = false;

function fail(where, e) {
  relay.error(e);
  status(`${where}: ${(e && e.message) || e}`, true);
}

async function ensureGpu() {
  if (gpu) return gpu;
  gpu = await Gpu.create();
  relay.log("gpu", { info: gpu.info || null });
  return gpu;
}

function modelBytes(since) {
  let wire = 0, decoded = 0;
  for (const e of performance.getEntriesByType("resource").slice(since)) {
    if (!/\/models\//.test(e.name)) continue;
    wire += e.transferSize || e.encodedBodySize || 0;
    decoded += e.decodedBodySize || 0;
  }
  return { wire, decoded };
}

async function load(size, longSide) {
  if (nets[size]) { nets[size].pipe.longSide = longSide; return nets[size]; }
  const g = await ensureGpu();
  const mark = performance.getEntriesByType("resource").length;
  const t0 = performance.now();
  const det = await DetNet.load(g, `det/models/${size}/`);
  const tDet = performance.now() - t0;
  const rec = await RecNet.load(g, `rec/models/${size}/`);
  const loadMs = performance.now() - t0;
  await new Promise((r) => setTimeout(r, 50)); // let resource timing entries land
  const bytes = modelBytes(mark);
  nets[size] = { pipe: new TextPipeline({ det, rec, longSide }), loadMs, bytes };
  relay.log("models ready", { size, loadMs: Math.round(loadMs), detLoadMs: Math.round(tDet), bytes });
  return nets[size];
}

function draw(lines, hl) {
  const cv = $("cv"), ctx = cv.getContext("2d");
  cv.width = image.canvas.width; cv.height = image.canvas.height;
  ctx.drawImage(image.canvas, 0, 0);
  const style = getComputedStyle(document.documentElement).getPropertyValue("--box").trim() || "#e0301e";
  ctx.lineWidth = Math.max(2, cv.width / 400);
  lines.forEach((l, i) => {
    ctx.strokeStyle = i === hl ? "#2f5fd0" : style;
    ctx.beginPath();
    l.box.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = style;
    ctx.font = `${Math.max(11, cv.width / 70)}px system-ui, sans-serif`;
    ctx.fillText(String(i + 1), l.box[0][0] + 2, Math.max(10, l.box[0][1] - 3));
  });
}

function show(lines, st, size) {
  const pre = $("text");
  pre.className = "";
  pre.textContent = "";
  lines.forEach((l, i) => {
    const sp = document.createElement("span");
    sp.textContent = l.text + "\n";
    sp.onmouseenter = () => { sp.className = "hl"; draw(lines, i); };
    sp.onmouseleave = () => { sp.className = ""; draw(lines, -1); };
    pre.appendChild(sp);
  });
  if (!lines.length) pre.textContent = "(no text found)";
  const t = lines.timing || {};
  const rows = [["model", size], ["lines", lines.length], ["load (fetch, unpack, upload)", st.fresh ? ms(st.loadMs) : "cached"],
    ["detect (GPU)", ms(t.det_ms)], ["postprocess", ms(t.post_ms)], ["crop", ms(t.crop_ms)], ["recognize (GPU)", ms(t.rec_ms)],
    ["total after load", ms((t.det_ms || 0) + (t.post_ms || 0) + (t.crop_ms || 0) + (t.rec_ms || 0))],
    ["model download", st.bytes.wire ? fmtB(st.bytes.wire) + (st.bytes.decoded && st.bytes.decoded !== st.bytes.wire ? ` (${fmtB(st.bytes.decoded)} decoded)` : "") : "cached by the browser"]];
  const body = $("stats").querySelector("tbody");
  body.textContent = "";
  for (const [k, v] of rows) { const tr = body.insertRow(); tr.insertCell().textContent = k; tr.insertCell().textContent = String(v); }
}

async function run() {
  if (busy || !image) return;
  busy = true; $("run").disabled = true;
  const size = $("size").value;
  const longSide = +(new URLSearchParams(location.search).get("long") || $("long").value);
  try {
    status(nets[size] ? "Reading." : `Loading the ${size} model.`);
    const fresh = !nets[size];
    const st = await load(size, longSide);
    status("Reading.");
    const t0 = performance.now();
    const lines = await st.pipe.run(image.canvas);
    const total = performance.now() - t0;
    show(lines, { ...st, fresh }, size);
    draw(lines, -1);
    status(`${lines.length} lines in ${ms(total)}.`);
    relay.log("result", { size, image: image.name, w: image.canvas.width, h: image.canvas.height, longSide, lines: lines.length,
      chars: lines.reduce((a, l) => a + l.text.length, 0), timing: lines.timing, loadMs: fresh ? Math.round(st.loadMs) : null,
      bytes: st.bytes, meanScore: lines.length ? +(lines.reduce((a, l) => a + l.score, 0) / lines.length).toFixed(3) : null,
      text: lines.map((l) => l.text).join(" | ").slice(0, 600) });
  } catch (e) {
    fail("Run failed", e);
  } finally {
    busy = false; $("run").disabled = false;
  }
}

async function setImage(src, name) {
  const bmp = await createImageBitmap(src);
  const cv = document.createElement("canvas");
  cv.width = bmp.width; cv.height = bmp.height;
  cv.getContext("2d").drawImage(bmp, 0, 0);
  image = { canvas: cv, name };
  draw([], -1);
  $("text").className = "muted"; $("text").textContent = "Run to read the image.";
  relay.log("image", { name, w: cv.width, h: cv.height });
}

async function loadSample() {
  const url = $("sample").value;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  await setImage(await r.blob(), url);
}

async function main() {
  relay.log("start", { build: BUILD, webgpu: !!navigator.gpu });
  if (!navigator.gpu) {
    $("nogpu").hidden = false; $("ui").hidden = true;
    relay.warn("no WebGPU");
    return;
  }
  $("sample").onchange = () => { $("file").value = ""; loadSample().catch((e) => fail("Sample", e)); };
  $("file").onchange = () => { const f = $("file").files[0]; if (f) setImage(f, f.name).catch((e) => fail("Image", e)); };
  $("run").onclick = run;
  await loadSample();
  $("run").disabled = false;
  status("Ready. Press Run.");
  if (new URLSearchParams(location.search).get("auto") === "1") run();
}
main().catch((e) => fail("Start", e));
window.addEventListener("error", (e) => relay.error(e.error || e.message));
