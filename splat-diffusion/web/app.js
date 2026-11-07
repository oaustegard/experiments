import { Painter, SplatDiffusion, bakeShader } from "./splatdiff.js";

const $ = (id) => document.getElementById(id);
const canvas = $("c"), status = $("status"), bar = $("bar").firstElementChild;
const PRETTY = ["marbled", "veined", "crystalline", "swirly", "paisley", "honeycombed", "scaly", "cracked",
  "lacelike", "bubbly", "woven", "knitted", "braided", "frilly", "spiralled", "stratified", "zigzagged", "cobwebbed"];

let model, painter, device, names, running = false, cancel = false, ready = false;
let seed = (Math.random() * 1e9) | 0;
const view = { tile: 384, ox: 0, oy: 0 };   // tile size in CSS px, pan offset in tile units

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  draw();
}

function draw() {
  if (!ready) return;
  const dpr = canvas.width / innerWidth;
  painter.draw(model.b.splats, model.K, view.tile * dpr, view.ox, view.oy);
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
        draw();
        bar.style.width = `${(100 * (i + 1)) / steps}%`;
        status.textContent = `${names[cls]} · step ${i + 1}/${steps} · ${((performance.now() - t0) / (i + 1)).toFixed(0)} ms/step`;
      },
      shouldStop: () => cancel });
    await device.queue.onSubmittedWorkDone();
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

async function main() {
  const adapter = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { $("nogpu").style.display = "grid"; $("panel").style.display = "none"; return; }
  device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  device.lost.then((i) => { status.textContent = `GPU device lost: ${i.message}`; });
  painter = new Painter(device, canvas, navigator.gpu.getPreferredCanvasFormat());
  model = await SplatDiffusion.create(device, "./model");
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
  $("min").addEventListener("click", () => {
    const p = $("panel"); p.classList.toggle("min"); $("min").textContent = p.classList.contains("min") ? "+" : "–";
  });
  status.textContent = "Ready";
  generate();
}

main().catch((e) => { status.textContent = `Error: ${e.message}`; console.error(e); });
