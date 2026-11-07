import { TextureDiffusion } from "../../wgsl-diffusion/web/diffusion.js";

const MODEL = "../../wgsl-diffusion/web/model";
const $ = (id) => document.getElementById(id);
const canvas = $("c"), ctx2d = canvas.getContext("2d"), statusEl = $("status"), bar = $("bar").firstElementChild;
const segVal = (id) => $(id).querySelector("[aria-pressed=true]").dataset.v;

// ------------------------------------------------------------ worker RPC
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let rpcId = 0;
const pending = new Map();
worker.onmessage = (e) => { const p = pending.get(e.data.id); pending.delete(e.data.id); e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data); };
const call = (type, msg = {}, transfer = []) => new Promise((resolve, reject) => {
  const id = ++rpcId; pending.set(id, { resolve, reject }); worker.postMessage({ type, id, ...msg }, transfer);
});

// ------------------------------------------------------------ state
let table, names, bank = [];             // bank[c]: { rgb, S } stored tile per class (seed 0 of the bridge bank)
const fresh = new Map();                 // `${cls}:${seed}:${S}` -> rgb of a generated tile
let img = null;                          // { W, H, rgba } at working size
let seg = null;                          // { K, labels, stats, solid }
let regions = [];                        // { cls, scale, seed, S, manual }
let collage = null;                      // ImageData
let model = null, device = null, gpuError = null, generating = false, cancel = false, freshMode = false;
const timing = {};
window.app = { timing, renders: 0, ready: false };

const status = (s) => { statusEl.textContent = s; };
const fmt = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`);

// ------------------------------------------------------------ loading
async function loadBank() {
  table = await (await fetch("classes.json")).json();
  names = table.names;
  const bmp = await createImageBitmap(await (await fetch("bank.png")).blob());
  const S = table.S, cols = 8, oc = new OffscreenCanvas(bmp.width, bmp.height), c = oc.getContext("2d");
  c.drawImage(bmp, 0, 0);
  const px = c.getImageData(0, 0, bmp.width, bmp.height).data;
  bank = names.map((_, k) => {
    const rgb = new Uint8Array(3 * S * S), x0 = (k % cols) * S, y0 = Math.floor(k / cols) * S;
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const i = 4 * ((y0 + y) * bmp.width + x0 + x), o = 3 * (y * S + x);
      rgb[o] = px[i]; rgb[o + 1] = px[i + 1]; rgb[o + 2] = px[i + 2];
    }
    return { rgb, S };
  });
  await call("table", { table });
}

async function loadImage(src) {
  const blob = src instanceof Blob ? src : await (await fetch(src)).blob();
  const bmp = await createImageBitmap(blob);
  const D = +segVal("detail"), s = Math.min(1, D / Math.max(bmp.width, bmp.height));
  const W = Math.max(16, Math.round(bmp.width * s)), H = Math.max(16, Math.round(bmp.height * s));
  const oc = new OffscreenCanvas(W, H), c = oc.getContext("2d");
  c.imageSmoothingQuality = "high";
  c.drawImage(bmp, 0, 0, W, H);
  epoch++;
  if (generating) cancel = true;
  seg = null;                        // no render may reach the worker between its new image and new regions
  img = { W, H, rgba: c.getImageData(0, 0, W, H).data, bmp };
  canvas.width = W; canvas.height = H;
  await call("image", { W, H, rgba: img.rgba.slice().buffer });
  await resegment();
}

// ------------------------------------------------------------ pipeline
// Bumped whenever the regions are recut: generation and renders started for the old regions stop.
let epoch = 0;

async function resegment() {
  if (!img) return;
  epoch++;
  if (generating) cancel = true;
  status("Finding regions…");
  const r = await call("segment", { K: +$("K").value });
  seg = { K: r.K, labels: r.labels, stats: r.stats, solid: new ImageData(new Uint8ClampedArray(r.solid), img.W, img.H) };
  timing.segment = r.ms.segment; timing.describe = r.ms.describe;
  regions = Array.from({ length: seg.K }, () => ({ cls: 0, scale: 1, seed: 0, S: 128, manual: false }));
  await choose();
}

async function choose() {
  if (!seg) return;
  const sc = segVal("scale");
  const { pick } = await call("choose", { opts: { mode: segVal("mode"), distinct: $("distinct").checked,
    scales: sc === "auto" ? [1, 2] : [+sc], seed: (Math.random() * 1e9) | 0,
    // DTD photographed most freckled textures on faces, so those tiles grow eyes: menu only
    exclude: ["freckled"] } });
  pick.forEach((p, k) => { const r = regions[k]; if (!r.manual) { r.cls = p.cls; r.scale = p.scale; r.seed = 0; } });
  buildList();
  await render();
  if (freshMode && model) generate({ onlyMissing: true });
}

function tileFor(r) {
  const key = `${r.cls}:${r.seed}:${r.S}`;
  if (r.seed && fresh.has(key)) return { key, rgb: fresh.get(key), S: r.S };
  return { key: `bank:${r.cls}`, rgb: bank[r.cls].rgb, S: bank[r.cls].S };
}

let rendering = null, again = false;
async function render() {
  if (!seg) return;
  if (rendering) { again = true; return rendering; }
  rendering = (async () => {
    try {
      do {
        again = false;
        if (!seg) break;
        const e0 = epoch, W = img.W, H = img.H;
        const tiles = regions.map((r) => { const t = tileFor(r); return { key: t.key, rgb: t.rgb.slice().buffer, S: t.S, scale: r.scale }; });
        const opts = { tint: segVal("tint"), feather: +$("feather").value, strength: +$("strength").value, align: $("align").checked };
        let r;
        try { r = await call("render", { tiles, opts }, tiles.map((t) => t.rgb)); }
        catch (e) { if (e0 === epoch) throw e; continue; }       // the regions changed under it: drop it
        if (e0 !== epoch) continue;
        collage = new ImageData(new Uint8ClampedArray(r.rgba), W, H);
        timing.render = r.ms;
        window.app.renders++;
        draw();
      } while (again);
    } catch (e) {
      status(`Error: ${e.message}`); console.error(e);
    } finally {
      rendering = null;
    }
  })();
  await rendering;
  if (!generating && seg) status(summary());
}

function summary() {
  const t = `${seg.K} regions · regions ${fmt(timing.segment + timing.describe)} · collage ${fmt(timing.render)}`;
  const g = timing.texture ? ` · ${fmt(timing.texture)} per ${timing.textureS}px texture` : "";
  const gpu = model ? "" : gpuError ? ` · ${gpuError}` : " · loading the model…";
  return t + g + gpu;
}

function draw() {
  if (!img) return;
  const v = segVal("view");
  if (v === "original") { ctx2d.drawImage(img.bmp, 0, 0, img.W, img.H); return; }
  if (v === "regions") { ctx2d.putImageData(seg.solid, 0, 0); return; }
  if (collage) ctx2d.putImageData(collage, 0, 0);
}

// ------------------------------------------------------------ region list
function buildList() {
  const box = $("regions");
  box.innerHTML = "";
  const order = regions.map((_, k) => k).sort((a, b) => seg.stats[b].frac - seg.stats[a].frac);
  const sorted = names.map((n, i) => [n, i]).sort();
  for (const k of order) {
    const r = regions[k], m = seg.stats[k].mean;
    const row = document.createElement("div");
    row.className = "reg"; row.id = `reg${k}`;
    const sw = document.createElement("div");
    sw.className = "sw"; sw.style.background = `lab(${m[0]}% ${m[1]} ${m[2]})`;
    const sel = document.createElement("select");
    sel.setAttribute("aria-label", `Texture for region ${k + 1}`);
    for (const [n, i] of sorted) sel.add(new Option(n, i));
    sel.value = r.cls;
    sel.addEventListener("change", async () => {
      r.cls = +sel.value; r.manual = true; r.seed = 0;
      await render();
      if (freshMode && model) generate({ only: [k] });
    });
    const pct = document.createElement("span");
    pct.className = "pct"; pct.textContent = `${Math.round(100 * seg.stats[k].frac)}%`;
    const re = document.createElement("button");
    re.className = "btn"; re.textContent = "↻"; re.title = "Dream a new texture of this class";
    re.disabled = !model;
    re.addEventListener("click", () => generate({ only: [k] }));
    row.append(sw, sel, pct, re);
    box.append(row);
  }
}

canvas.addEventListener("click", (e) => {
  if (!seg) return;
  const b = canvas.getBoundingClientRect();
  const x = Math.floor(((e.clientX - b.left) / b.width) * img.W), y = Math.floor(((e.clientY - b.top) / b.height) * img.H);
  const k = seg.labels[y * img.W + x];
  document.querySelectorAll(".reg.hot").forEach((r) => r.classList.remove("hot"));
  const row = $(`reg${k}`);
  if (row) { row.classList.add("hot"); row.scrollIntoView({ block: "nearest", behavior: "smooth" }); row.querySelector("select").focus(); }
});

// ------------------------------------------------------------ generation (WebGPU)
async function initGPU() {
  try {
    if (!navigator.gpu) throw new Error("no WebGPU here, so stored textures only");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter, so stored textures only");
    device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    device.lost.then((i) => { model = null; gpuError = `GPU device lost: ${i.message}`; status(gpuError); });
    const t0 = performance.now();
    model = await TextureDiffusion.create(device, MODEL, { weights: "q8" });
    timing.modelLoad = performance.now() - t0;
    for (const b of $("size").children) if (192 * (+b.dataset.v) ** 2 * 4 > device.limits.maxStorageBufferBindingSize) b.disabled = true;
    $("go").disabled = false;
    document.querySelectorAll(".reg .btn").forEach((b) => { b.disabled = false; });
  } catch (e) {
    gpuError = e.message;
  }
  if (seg && !generating) status(summary());
}

function toRGB(d, S) {
  const u = new Uint8Array(3 * S * S), n = S * S;
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) u[3 * i + k] = Math.round(Math.min(1, Math.max(0, (d[k * n + i] + 1) / 2)) * 255);
  return u;
}

let queue = Promise.resolve(), stops = 0;
function generate({ only = null, onlyMissing = false } = {}) {
  if (!model) return Promise.resolve();
  // Stop: cancel the running batch and every batch queued behind it
  if (generating && !only && !onlyMissing) { cancel = true; stops++; return queue; }
  const s0 = stops;
  queue = queue.then(() => (s0 === stops ? runGenerate(only, onlyMissing) : null)).catch((e) => { console.error(e); });
  return queue;
}

async function runGenerate(only, onlyMissing) {
  if (!seg) return;
  generating = true; cancel = false; freshMode = true;
  const e0 = epoch;
  $("go").textContent = "Stop";
  const S = +segVal("size"), steps = +$("steps").value;
  let ks = only || regions.map((_, k) => k).sort((a, b) => seg.stats[b].frac - seg.stats[a].frac);
  if (onlyMissing) ks = ks.filter((k) => !regions[k].seed);
  const t0 = performance.now();
  let done = 0;
  try {
    for (const k of ks) {
      if (cancel || e0 !== epoch) break;
      const r = regions[k], seed = 1 + ((Math.random() * 1e9) | 0), t1 = performance.now();
      status(`Dreaming ${names[r.cls]} for region ${done + 1} of ${ks.length}…`);
      const ok = await model.sample({ cls: r.cls, H: S, W: S, steps, cfg: 3, seed, circ: true,
        onStep: (i) => { bar.style.width = `${(100 * (done + (i + 1) / steps)) / ks.length}%`; },
        shouldStop: () => cancel || e0 !== epoch });
      if (!ok || e0 !== epoch) break;
      const d = await model.read(model.dBuf, 3 * S * S);
      timing.texture = performance.now() - t1; timing.textureS = S;
      fresh.set(`${r.cls}:${seed}:${S}`, toRGB(d, S));
      r.seed = seed; r.S = S;
      done++;
      render();
    }
  } catch (e) {
    status(`Error: ${e.message}`); console.error(e);
  } finally {
    generating = false;
    $("go").textContent = "Generate fresh textures";
    bar.style.width = "0";
  }
  if (e0 !== epoch) return;            // recut meanwhile: choose() queues the new regions' textures
  await render();
  timing.generateAll = performance.now() - t0;
  status(`${done} texture${done === 1 ? "" : "s"} in ${fmt(timing.generateAll)} · ` + summary());
}

// ------------------------------------------------------------ controls
function segControl(id, fn) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || b.disabled) return;
    for (const x of $(id).children) x.setAttribute("aria-pressed", x === b);
    fn(b.dataset.v);
  });
}

async function main() {
  status("Loading stored textures…");
  await loadBank();
  for (const id of ["K", "strength", "feather", "steps"]) $(id).addEventListener("input", () => { $(`${id}V`).textContent = $(id).value; });
  $("K").addEventListener("change", resegment);
  $("strength").addEventListener("change", render);
  $("feather").addEventListener("change", render);
  $("align").addEventListener("change", render);
  $("distinct").addEventListener("change", () => { regions.forEach((r) => { r.manual = false; }); choose(); });
  segControl("detail", () => source && loadImage(source));
  segControl("mode", () => { regions.forEach((r) => { r.manual = false; }); choose(); });
  segControl("scale", () => { regions.forEach((r) => { r.manual = false; }); choose(); });
  segControl("tint", render);
  segControl("view", draw);
  segControl("size", () => {});
  $("go").addEventListener("click", () => generate());
  $("open").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", () => { const f = $("file").files[0]; if (f) openBlob(f); });
  $("sample").addEventListener("click", () => openURL("sample.jpg"));
  $("save").addEventListener("click", save);
  const stage = $("stage");
  stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("drag"); });
  stage.addEventListener("dragleave", () => stage.classList.remove("drag"));
  stage.addEventListener("drop", (e) => {
    e.preventDefault(); stage.classList.remove("drag");
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/")); if (f) openBlob(f);
  });
  addEventListener("paste", (e) => {
    const it = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith("image/"));
    if (it) openBlob(it.getAsFile());
  });
  const q = new URLSearchParams(location.search);
  if (q.has("K")) { $("K").value = q.get("K"); $("KV").textContent = q.get("K"); }
  await openURL(q.get("img") || "sample.jpg");
  window.app.ready = true;
  initGPU().then(() => { window.app.gpu = !!model; });
}

let source = null;     // what the current picture came from, to reload at another detail level
async function openBlob(b) { source = b; await loadImage(b); }
async function openURL(u) { source = u; await loadImage(u); }

async function save() {
  if (!collage) return;
  const oc = new OffscreenCanvas(img.W, img.H);
  oc.getContext("2d").putImageData(collage, 0, 0);
  const blob = await oc.convertToBlob({ type: "image/png" });
  const a = document.createElement("a");
  a.download = "texture-collage.png";
  a.href = URL.createObjectURL(blob);
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

window.app.generate = (o) => { if (o?.size) for (const b of $("size").children) b.setAttribute("aria-pressed", b.dataset.v === String(o.size));
  if (o?.steps) $("steps").value = o.steps; return generate(o?.only ? { only: o.only } : {}); };
window.app.regions = () => regions.map((r, k) => ({ ...r, name: names[r.cls], frac: seg.stats[k].frac }));

main().catch((e) => { status(`Error: ${e.message}`); console.error(e); });
