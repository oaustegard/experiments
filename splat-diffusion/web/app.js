import { Painter, SplatDiffusion, bakeShader, tileMean } from "./splatdiff.js";

const $ = (id) => document.getElementById(id);
const canvas = $("c"), status = $("status"), bar = $("bar").firstElementChild;
const PRETTY = ["marbled", "veined", "crystalline", "swirly", "paisley", "honeycombed", "scaly", "cracked",
  "lacelike", "bubbly", "woven", "knitted", "braided", "frilly", "spiralled", "stratified", "zigzagged", "cobwebbed"];

let model, painter, device, names, running = false, cancel = false, ready = false;
// 1,024 splats by default; 512 on phones, where the model and the paint cost half as much
const MODELS = { 1024: "./model", 512: "./model512" };
const PHONE = /iPhone|iPad|Android/i.test(navigator.userAgent) || (navigator.maxTouchPoints > 1 && Math.min(screen.width, screen.height) < 820);
const cache = {};
let K = PHONE ? 512 : 1024;
let seed = (Math.random() * 1e9) | 0;
const view = { tile: 384, ox: 0, oy: 0 };   // tile size in CSS px, pan offset in tile units

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  draw();
}

// Drawing only ever happens inside requestAnimationFrame, at most once per frame. Safari presents a
// WebGPU canvas on its animation-frame tick; frames submitted from inside the sampling loop could be
// dropped, which showed as the texture vanishing part-way through generation on an iPhone.
let drawQueued = false;
function draw() {
  if (!ready || drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => {
    drawQueued = false;
    const dpr = canvas.width / innerWidth;
    painter.draw(view.tile * dpr, view.ox, view.oy);
  });
}

async function generate() {
  if (running) { cancel = true; return; }
  running = true; cancel = false;
  $("go").textContent = "Stop";
  const cls = +$("cls").value, steps = +$("steps").value, cfg = +$("cfg").value;
  const t0 = performance.now();
  try {
    const done = await model.sample({ cls, steps, cfg, seed,
      onStep: async (i) => {
        ready = true;
        painter.setSplats(model.b.splats, model.K);   // marks the cached tile stale; drawn on the next frame
        draw();
        bar.style.width = `${(100 * (i + 1)) / steps}%`;
        status.textContent = `${names[cls]} · step ${i + 1}/${steps} · ${((performance.now() - t0) / (i + 1)).toFixed(0)} ms/step`;
      },
      shouldStop: () => cancel });
    await device.queue.onSubmittedWorkDone();
    painter.setSplats(model.b.splats, model.K);
    painter.mean = await tileMean(device, model.b.splats, model.K);   // centre of the stochastic-tiling blend
    draw();
    status.textContent = done ? `${names[cls]} · ${model.K} splats · ${steps} steps in ${((performance.now() - t0) / 1000).toFixed(2)} s · seed ${seed}` : "Stopped";
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    console.error(e);
  }
  running = false;
  $("go").textContent = "Generate";
  seed = (Math.random() * 1e9) | 0;
}

async function exportWGSL() {
  if (!ready) return;
  const src = bakeShader(await model.splats(), model.K);
  const a = document.createElement("a");
  a.download = `${names[+$("cls").value]}-splats.wgsl`;
  a.href = URL.createObjectURL(new Blob([src], { type: "text/plain" }));
  a.click();
}

function pickRandom() { $("cls").value = names.indexOf(PRETTY[(Math.random() * PRETTY.length) | 0]); }

function pointerControls() {
  const pts = new Map();
  let pinch = 0;
  canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); });
  const up = (e) => { pts.delete(e.pointerId); pinch = 0; };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId)) return;
    const [px, py] = pts.get(e.pointerId);
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      const dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, dist / pinch);
      pinch = dist;
    } else {
      view.ox -= (e.clientX - px) / view.tile;
      view.oy -= (e.clientY - py) / view.tile;
      draw();
    }
  });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
}

// keep the texture point under (x, y) fixed while the tile size changes
function zoomAt(x, y, f) {
  const t1 = Math.min(6000, Math.max(24, view.tile * f));
  view.ox += x / view.tile - x / t1;
  view.oy += y / view.tile - y / t1;
  view.tile = t1;
  draw();
}

async function loadModel(k) {
  if (!cache[k]) {
    try {
      cache[k] = await SplatDiffusion.create(device, MODELS[k]);
    } catch (e) {
      const other = k === 1024 ? 512 : 1024;
      const o = [...$("splats").options].find((x) => x.value === String(k));
      if (o) o.remove();
      if (!$("splats").options.length) throw e;
      console.warn(`no ${k}-splat model (${e.message}); using ${other}`);
      K = other;
      $("splats").value = String(other);
      return loadModel(other);
    }
  }
  return cache[k];
}

async function main() {
  const adapter = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { $("nogpu").style.display = "grid"; $("panel").style.display = "none"; return; }
  device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  device.lost.then((i) => { status.textContent = `GPU device lost: ${i.message}`; });
  painter = new Painter(device, canvas, navigator.gpu.getPreferredCanvasFormat());
  const q0 = new URLSearchParams(location.search);
  if (q0.has("k") && MODELS[q0.get("k")]) K = +q0.get("k");
  model = await loadModel(K);
  names = model.man.names;
  const sel = $("cls");
  names.map((n, i) => [n, i]).sort().forEach(([n, i]) => sel.add(new Option(n, i)));
  sel.value = names.indexOf("marbled");
  const q = new URLSearchParams(location.search);
  if (q.has("t") && names.includes(q.get("t"))) sel.value = names.indexOf(q.get("t"));
  for (const k of ["steps", "cfg"]) if (q.has(k)) { $(k).value = q.get(k); }
  if (q.has("seed")) seed = +q.get("seed");
  if (q.has("tile")) view.tile = +q.get("tile");
  for (const id of ["steps", "cfg"]) {
    $(`${id}V`).textContent = $(id).value;
    $(id).addEventListener("input", () => { $(`${id}V`).textContent = $(id).value; });
  }
  addEventListener("resize", resize);
  resize();
  pointerControls();
  $("go").addEventListener("click", generate);
  $("rnd").addEventListener("click", () => { pickRandom(); if (!running) generate(); });
  $("wgsl").addEventListener("click", exportWGSL);
  $("stoch").addEventListener("change", () => { painter.stochastic = $("stoch").checked; draw(); });
  if (q0.has("grid")) { $("stoch").checked = false; painter.stochastic = false; }
  $("splats").value = String(K);
  $("splats").addEventListener("change", async () => {
    if (running) { cancel = true; while (running) await new Promise((r) => setTimeout(r, 50)); }
    K = +$("splats").value;
    status.textContent = `Loading the ${K}-splat model…`;
    model = await loadModel(K);
    generate();
  });
  $("min").addEventListener("click", () => {
    const p = $("panel"); p.classList.toggle("min"); $("min").textContent = p.classList.contains("min") ? "+" : "–";
  });
  status.textContent = "Ready";
  generate();
}

main().catch((e) => { status.textContent = `Error: ${e.message}`; console.error(e); });
