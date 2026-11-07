import { SplatDiffusion, paintToBytes } from "../../splat-diffusion/web/splatdiff.js";
import { recolourSplats, labelMap, makeScene, storageShader, sceneBuffers, sceneUniforms, bakeCollage, chooseLabels, viewerHTML } from "./splatcollage.js";

const MODEL = "../../splat-diffusion/web/model512";
const NK = 512;
const $ = (id) => document.getElementById(id);
const gl = $("gl"), flat = $("flat"), statusEl = $("status"), bar = $("bar").firstElementChild;
const segVal = (id) => $(id).querySelector("[aria-pressed=true]").dataset.v;
const status = (s) => { statusEl.textContent = s; };
const fmt = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`);
const kb = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)} MB` : `${Math.round(n / 1e3)} kB`);

// ------------------------------------------------------------ worker RPC
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let rpcId = 0;
const pending = new Map();
worker.onmessage = (e) => { const p = pending.get(e.data.id); pending.delete(e.data.id); e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data); };
const call = (type, msg = {}, transfer = []) => new Promise((resolve, reject) => {
  const id = ++rpcId; pending.set(id, { resolve, reject }); worker.postMessage({ type, id, ...msg }, transfer);
});

// ------------------------------------------------------------ state
let table, names, bank, bankMeta;
let device, ctx, format, pipe, bufs = null, ubuf, bindGroup = null;
let img = null, seg = null, regions = [], scene = null, model = null, modelLoading = null;
const fresh = new Map();          // `${cls}:${seed}` -> { splats, stats, theta, gcoh }
const view = { x0: 0, y0: 0, scale: 1 };     // output pixel -> reference px
const timing = {};
let epoch = 0, generating = false, cancel = false, freshMode = false;
window.app = { timing, renders: 0, ready: false };

// ------------------------------------------------------------ GPU
async function initGPU() {
  if (!navigator.gpu) throw new Error("This page needs WebGPU: Chrome or Edge 113+, Safari 26+, or Firefox 141+.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter.");
  device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  device.lost.then((i) => status(`GPU device lost: ${i.message}`));
  ctx = gl.getContext("webgpu");
  format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });
  const mod = device.createShaderModule({ code: storageShader() });
  pipe = device.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format }] } });
  ubuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}

function upload() {
  const b = sceneBuffers(scene);
  const mk = (arr) => {
    const buf = device.createBuffer({ size: Math.max(16, arr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, arr);
    return buf;
  };
  if (bufs) for (const x of Object.values(bufs)) x.destroy();
  bufs = { sp: mk(b.sp), labels: mk(b.labels), xf: mk(b.xf) };
  bindGroup = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: ubuf } }, { binding: 1, resource: { buffer: bufs.sp } },
    { binding: 2, resource: { buffer: bufs.labels } }, { binding: 3, resource: { buffer: bufs.xf } }] });
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), r = $("stage").getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (gl.width !== w || gl.height !== h) { gl.width = w; gl.height = h; flat.width = w; flat.height = h; }
}

function fit() {
  if (!img) return;
  resize();
  view.scale = Math.max(img.W / gl.width, img.H / gl.height) * 1.04;
  view.x0 = img.W / 2 - (gl.width / 2) * view.scale;
  view.y0 = img.H / 2 - (gl.height / 2) * view.scale;
  draw();
}

let drawQueued = false;
function draw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(async () => {
    drawQueued = false;
    if (!scene || !bindGroup) return;
    resize();
    const v = segVal("view");
    flat.style.display = v === "collage" ? "none" : "block";
    if (v !== "collage") drawFlat(v);
    device.queue.writeBuffer(ubuf, 0, sceneUniforms(scene, view));
    const t0 = performance.now();
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(),
      loadOp: "clear", storeOp: "store", clearValue: [0.04, 0.04, 0.05, 1] }] });
    pass.setPipeline(pipe); pass.setBindGroup(0, bindGroup); pass.draw(3); pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    timing.frame = performance.now() - t0;
    window.app.renders++;
  });
}

// the original picture or its flat region colours, under the same view
function drawFlat(v) {
  const c = flat.getContext("2d");
  c.fillStyle = "#0a0a0d"; c.fillRect(0, 0, flat.width, flat.height);
  c.imageSmoothingEnabled = v === "original";
  const src = v === "original" ? img.bmp : seg.solidCanvas;
  c.drawImage(src, -view.x0 / view.scale, -view.y0 / view.scale, img.W / view.scale, img.H / view.scale);
}

// ------------------------------------------------------------ loading
async function loadBank() {
  [table, bankMeta] = await Promise.all([(await fetch("classes.json")).json(), (await fetch("bank_meta.json")).json()]);
  names = table.names;
  bank = new Float32Array(await (await fetch("bank.bin")).arrayBuffer());
  await call("table", { table });
}

let source = null;
async function loadImage(src) {
  source = src;
  const blob = src instanceof Blob ? src : await (await fetch(src)).blob();
  const bmp = await createImageBitmap(blob);
  // regions are cut at 768 px on the long side, as in the pixel collage
  const s = Math.min(1, 768 / Math.max(bmp.width, bmp.height));
  const W = Math.max(16, Math.round(bmp.width * s)), H = Math.max(16, Math.round(bmp.height * s));
  const oc = new OffscreenCanvas(W, H), c = oc.getContext("2d");
  c.imageSmoothingQuality = "high";
  c.drawImage(bmp, 0, 0, W, H);
  epoch++; if (generating) cancel = true;
  seg = null; scene = null;
  img = { W, H, bmp };
  await call("image", { W, H, rgba: c.getImageData(0, 0, W, H).data.buffer });
  await resegment();
  fit();
}

async function resegment() {
  if (!img) return;
  epoch++; if (generating) cancel = true;
  status("Finding regions…");
  const r = await call("segment", { K: +$("K").value });
  const solidCanvas = new OffscreenCanvas(img.W, img.H);
  solidCanvas.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(r.solid), img.W, img.H), 0, 0);
  seg = { K: r.K, W: img.W, H: img.H, labels: r.labels, stats: r.stats, rdesc: r.rdesc, solidCanvas };
  timing.segment = r.ms.segment; timing.describe = r.ms.describe;
  regions = Array.from({ length: seg.K }, (_, k) => ({ cls: 0, scale: 1, seed: 0, manual: false, off: [hash(k, 1), hash(k, 2)] }));
  await choose();
}

const hash = (a, b) => { const x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return x - Math.floor(x); };

async function choose() {
  if (!seg) return;
  const sc = segVal("scale");
  const { pick } = await call("choose", { opts: { mode: segVal("pick"), distinct: $("distinct").checked,
    scales: sc === "auto" ? [1, 2] : [+sc], seed: (Math.random() * 1e9) | 0, exclude: ["freckled"] } });
  pick.forEach((p, k) => { const r = regions[k]; if (!r.manual) { r.cls = p.cls; r.scale = p.scale; r.seed = 0; } });
  buildList();
  rebuild();
  if (freshMode && model) generate({ onlyMissing: true });
}

function texFor(r) {
  if (r.seed && fresh.has(`${r.cls}:${r.seed}`)) return fresh.get(`${r.cls}:${r.seed}`);
  const m = bankMeta.classes[r.cls];
  return { splats: bank.subarray(r.cls * NK * 12, (r.cls + 1) * NK * 12), stats: m.stats, theta: m.theta, gcoh: m.gcoh };
}

// Scene from the current regions and controls: recolour each region's splats, place them, upload.
function rebuild() {
  if (!seg) return;
  const t0 = performance.now();
  const tint = segVal("tint"), strength = +$("strength").value, align = $("align").checked;
  const regs = regions.map((r, k) => {
    const t = texFor(r), rd = seg.rdesc[k];
    let phi = 0;
    if (align && t.gcoh > 0.15 && rd.f[3] > 0.15) phi = rd.theta - t.theta;
    return { splats: recolourSplats(t.splats, NK, t.stats, seg.stats[k], { tint, strength }), phi, scale: r.scale, off: r.off };
  });
  scene = makeScene({ W0: img.W, H0: img.H, lm: labelMap(seg, +segVal("lw")), regions: regs, K: NK, mode: segVal("mode"), blend: +$("blend").value });
  upload();
  timing.build = performance.now() - t0;
  window.app.sceneBytes = null;
  draw();
  if (!generating) status(summary());
}

function summary() {
  const t = `${seg.K} regions · cut ${fmt(timing.segment + timing.describe)} · shader data ${fmt(timing.build)}`;
  const f = timing.frame ? ` · frame ${fmt(timing.frame)}` : "";
  const g = timing.texture ? ` · ${fmt(timing.texture)} per texture` : "";
  return t + f + g + (model ? "" : modelLoading ? " · loading the model…" : "");
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
    sel.addEventListener("change", () => {
      r.cls = +sel.value; r.manual = true; r.seed = 0;
      rebuild();
      if (freshMode && model) generate({ only: [k] });
    });
    const pct = document.createElement("span");
    pct.className = "pct"; pct.textContent = `${Math.round(100 * seg.stats[k].frac)}%`;
    const re = document.createElement("button");
    re.className = "btn"; re.textContent = "↻"; re.title = "Dream a new texture of this kind";
    re.addEventListener("click", () => generate({ only: [k] }));
    row.append(sw, sel, pct, re);
    box.append(row);
  }
}

// ------------------------------------------------------------ fresh textures
async function ensureModel() {
  if (model) return model;
  if (!modelLoading) modelLoading = SplatDiffusion.create(device, MODEL).then((m) => { model = m; $("go").disabled = false; return m; });
  return modelLoading;
}

let queue = Promise.resolve(), stops = 0;
function generate({ only = null, onlyMissing = false } = {}) {
  if (generating && !only && !onlyMissing) { cancel = true; stops++; return queue; }
  const s0 = stops;
  queue = queue.then(() => (s0 === stops ? runGenerate(only, onlyMissing) : null)).catch((e) => { console.error(e); status(`Error: ${e.message}`); });
  return queue;
}

async function runGenerate(only, onlyMissing) {
  if (!seg) return;
  generating = true; cancel = false; freshMode = true;
  const e0 = epoch, steps = +$("steps").value;
  $("go").textContent = "Stop";
  let done = 0;
  const t0 = performance.now();
  try {
    status("Loading the model (19 MB)…");
    const m = await ensureModel();
    let ks = only || regions.map((_, k) => k).sort((a, b) => seg.stats[b].frac - seg.stats[a].frac);
    if (onlyMissing) ks = ks.filter((k) => !regions[k].seed);
    for (const k of ks) {
      if (cancel || e0 !== epoch) break;
      const r = regions[k], seed = 1 + ((Math.random() * 1e9) | 0), t1 = performance.now();
      status(`Dreaming ${names[r.cls]} for region ${done + 1} of ${ks.length}…`);
      const ok = await m.sample({ cls: r.cls, steps, cfg: 3, seed,
        onStep: (i) => { bar.style.width = `${(100 * (done + (i + 1) / steps)) / ks.length}%`; },
        shouldStop: () => cancel || e0 !== epoch });
      if (!ok || e0 !== epoch) break;
      const splats = await m.splats();
      const rgba = await paintToBytes(device, m.b.splats, NK, 128);
      const d = await call("describeTile", { S: 128, rgba: rgba.buffer }, [rgba.buffer]);
      fresh.set(`${r.cls}:${seed}`, { splats, stats: d.stats, theta: d.theta, gcoh: d.gcoh });
      timing.texture = performance.now() - t1;
      if (e0 !== epoch) break;
      r.seed = seed; done++;
      rebuild();
    }
  } finally {
    generating = false;
    $("go").textContent = "Generate fresh textures";
    bar.style.width = "0";
  }
  if (e0 === epoch && seg) status(`${done} texture${done === 1 ? "" : "s"} in ${fmt(performance.now() - t0)} · ` + summary());
}

// ------------------------------------------------------------ export
function download(blob, name) {
  const a = document.createElement("a");
  a.download = name; a.href = URL.createObjectURL(blob); a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function exportWGSL() {
  const src = bakeCollage(scene);
  download(new Blob([src], { type: "text/plain" }), "collage.wgsl");
  status(`collage.wgsl: ${kb(src.length)} · ` + summary());
}

function exportHTML() {
  const src = bakeCollage(scene);
  download(new Blob([viewerHTML(src, scene.W0, scene.H0)], { type: "text/html" }), "collage.html");
}

// Render the collage at f output pixels per reference pixel into a texture and save it as PNG.
async function exportPNG() {
  const f = +segVal("pngw"), W = Math.round(scene.W0 * f), H = Math.round(scene.H0 * f);
  if (W > device.limits.maxTextureDimension2D || H > device.limits.maxTextureDimension2D) { status("Too large for this GPU"); return; }
  status(`Rendering ${W}×${H}…`);
  const mod = device.createShaderModule({ code: storageShader() });
  const p2 = device.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] } });
  const u = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(u, 0, sceneUniforms(scene, { x0: 0, y0: 0, scale: 1 / f }));
  const bg = device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: u } }, { binding: 1, resource: { buffer: bufs.sp } },
    { binding: 2, resource: { buffer: bufs.labels } }, { binding: 3, resource: { buffer: bufs.xf } }] });
  const tex = device.createTexture({ size: [W, H], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const bpr = Math.ceil((W * 4) / 256) * 256;
  const rb = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
  pass.setPipeline(p2); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [W, H]);
  const t0 = performance.now();
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  timing.png = performance.now() - t0;
  const src = new Uint8Array(rb.getMappedRange()), out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) out.set(src.subarray(y * bpr, y * bpr + W * 4), y * W * 4);
  rb.unmap(); rb.destroy(); tex.destroy(); u.destroy();
  const oc = new OffscreenCanvas(W, H);
  oc.getContext("2d").putImageData(new ImageData(out, W, H), 0, 0);
  download(await oc.convertToBlob({ type: "image/png" }), `collage-${W}.png`);
  status(`${W}×${H} rendered in ${fmt(timing.png)} · ` + summary());
}

// ------------------------------------------------------------ interaction
function segControl(id, fn) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || b.disabled) return;
    for (const x of $(id).children) x.setAttribute("aria-pressed", x === b);
    fn(b.dataset.v);
  });
}

function zoomAt(cx, cy, f) {
  const r = gl.getBoundingClientRect(), d = gl.width / r.width, px = (cx - r.left) * d, py = (cy - r.top) * d;
  view.x0 += px * view.scale * (1 - f); view.y0 += py * view.scale * (1 - f); view.scale *= f;
  draw();
}

function pointerControls() {
  gl.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY * 0.0015)); }, { passive: false });
  const pts = new Map();
  let pinch = 0, moved = 0;
  gl.addEventListener("pointerdown", (e) => { gl.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); moved = 0; gl.classList.add("drag"); });
  const up = (e) => {
    const o = pts.get(e.pointerId); pts.delete(e.pointerId); pinch = 0; gl.classList.remove("drag");
    if (o && moved < 4 && pts.size === 0) pick(e.clientX, e.clientY);
  };
  gl.addEventListener("pointerup", up); gl.addEventListener("pointercancel", up);
  gl.addEventListener("pointermove", (e) => {
    const o = pts.get(e.pointerId); if (!o) return;
    const r = gl.getBoundingClientRect(), d = gl.width / r.width;
    if (pts.size === 2) {
      pts.set(e.pointerId, [e.clientX, e.clientY]);
      const [a, b] = [...pts.values()], dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, pinch / dist);
      pinch = dist; moved += 10; return;
    }
    moved += Math.abs(e.clientX - o[0]) + Math.abs(e.clientY - o[1]);
    view.x0 -= (e.clientX - o[0]) * d * view.scale; view.y0 -= (e.clientY - o[1]) * d * view.scale;
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    draw();
  });
}

// a click (not a drag) highlights the region under it
function pick(cx, cy) {
  if (!scene) return;
  const r = gl.getBoundingClientRect(), d = gl.width / r.width;
  const px = view.x0 + (cx - r.left) * d * view.scale, py = view.y0 + (cy - r.top) * d * view.scale;
  if (px < 0 || py < 0 || px >= scene.W0 || py >= scene.H0) return;
  const k = chooseLabels(scene, px, py, 0).l1;
  document.querySelectorAll(".reg.hot").forEach((x) => x.classList.remove("hot"));
  const row = $(`reg${k}`);
  if (row) { row.classList.add("hot"); row.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
}

async function main() {
  try { await initGPU(); } catch (e) { status(e.message); return; }
  status("Loading stored textures…");
  await loadBank();
  for (const id of ["K", "strength", "steps", "blend"]) $(id).addEventListener("input", () => { $(`${id}V`).textContent = $(id).value; });
  $("K").addEventListener("change", resegment);
  $("strength").addEventListener("change", rebuild);
  $("blend").addEventListener("change", rebuild);
  $("align").addEventListener("change", rebuild);
  $("distinct").addEventListener("change", () => { regions.forEach((r) => { r.manual = false; }); choose(); });
  segControl("pick", () => { regions.forEach((r) => { r.manual = false; }); choose(); });
  segControl("scale", () => { regions.forEach((r) => { r.manual = false; }); choose(); });
  segControl("tint", rebuild);
  segControl("lw", rebuild);
  segControl("mode", rebuild);
  segControl("view", draw);
  segControl("pngw", () => {});
  $("go").addEventListener("click", () => generate());
  $("fit").addEventListener("click", fit);
  $("wgsl").addEventListener("click", exportWGSL);
  $("html").addEventListener("click", exportHTML);
  $("png").addEventListener("click", () => exportPNG().catch((e) => status(`Error: ${e.message}`)));
  $("open").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", () => { const f = $("file").files[0]; if (f) loadImage(f); });
  $("sample").addEventListener("click", () => loadImage("../web/sample.jpg"));
  const stage = $("stage");
  stage.addEventListener("dragover", (e) => { e.preventDefault(); });
  stage.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/")); if (f) loadImage(f);
  });
  addEventListener("paste", (e) => {
    const it = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith("image/"));
    if (it) loadImage(it.getAsFile());
  });
  addEventListener("resize", () => { const s = view.scale; resize(); view.scale = s; draw(); });
  pointerControls();
  const q = new URLSearchParams(location.search);
  if (q.has("K")) { $("K").value = q.get("K"); $("KV").textContent = q.get("K"); }
  await loadImage(q.get("img") || "../web/sample.jpg");
  window.app.ready = true;
  ensureModel().then(() => { window.app.gpu = true; if (seg && !generating) status(summary()); })
    .catch((e) => { window.app.gpu = false; status(`Model: ${e.message}`); });
}

window.app.generate = (o = {}) => { if (o.steps) $("steps").value = o.steps; return generate(o.only ? { only: o.only } : {}); };
window.app.regions = () => regions.map((r, k) => ({ ...r, name: names[r.cls], frac: seg.stats[k].frac }));
window.app.baked = () => bakeCollage(scene);
window.app.scene = () => scene;
// what the Stop button does, callable whatever the button reads: cancel the running batch and drop queued ones
window.app.stop = () => { cancel = true; stops++; };
window.app.exportPNG = exportPNG;

main().catch((e) => { status(`Error: ${e.message}`); console.error(e); });
