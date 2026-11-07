import { TextureDiffusion, Viewer } from "./diffusion.js";

const $ = (id) => document.getElementById(id);
const canvas = $("c"), status = $("status"), bar = $("bar").firstElementChild;
const PRETTY = ["marbled", "veined", "crystalline", "swirly", "paisley", "honeycombed", "scaly", "cracked",
  "lacelike", "bubbly", "woven", "knitted", "braided", "frilly", "spiralled", "stratified", "zigzagged", "cobwebbed"];

let model, viewer, device, names, running = false, cancel = false, dreamTimer = null;
let size = 128, seed = (Math.random() * 1e9) | 0, last = null;

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  draw();
}

function draw() {
  if (!viewer || !last) return;
  const dpr = canvas.width / innerWidth;
  viewer.draw(model.dBuf, last.H, last.W, +$("zoom").value * dpr, $("smooth").checked);
}

async function generate() {
  if (running) { cancel = true; return; }
  running = true; cancel = false;
  $("go").textContent = "Stop";
  const cls = +$("cls").value, steps = +$("steps").value, cfg = +$("cfg").value;
  const H = size, W = size, circ = $("seam").checked;
  const t0 = performance.now();
  try {
    const done = await model.sample({ cls, H, W, steps, cfg, seed, circ,
      onStep: async (i) => {
        last = { H, W };          // set here: the display buffer for this size exists once sampling starts
        draw();                   // queued behind step i, so it shows that step's x0 estimate
        bar.style.width = `${(100 * (i + 1)) / steps}%`;
        const per = (performance.now() - t0) / (i + 1);
        status.textContent = `${names[cls]} · step ${i + 1}/${steps} · ${per.toFixed(0)} ms/step`;
      },
      shouldStop: () => cancel });
    await device.queue.onSubmittedWorkDone();
    draw();
    const ms = performance.now() - t0;
    status.textContent = done
      ? `${names[cls]} · ${W}×${H} · ${steps} steps in ${(ms / 1000).toFixed(2)} s · seed ${seed}`
      : "Stopped";
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    console.error(e);
  }
  running = false;
  $("go").textContent = "Generate";
  seed = (Math.random() * 1e9) | 0;
  if ($("dream").checked && !cancel) dreamTimer = setTimeout(dreamNext, 2500);
}

function pickRandom() {
  const n = PRETTY[(Math.random() * PRETTY.length) | 0];
  $("cls").value = names.indexOf(n);
}

function dreamNext() {
  dreamTimer = null;
  if (!$("dream").checked) return;
  pickRandom();
  generate();
}

async function save() {
  if (!last) return;
  const { H, W } = last;
  const d = await model.read(model.dBuf, 3 * H * W);
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d"), img = ctx.createImageData(W, H);
  for (let i = 0; i < H * W; i++) {
    for (let k = 0; k < 3; k++) img.data[4 * i + k] = Math.round(Math.min(1, Math.max(0, (d[k * H * W + i] + 1) / 2)) * 255);
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const a = document.createElement("a");
  a.download = `${names[+$("cls").value]}-${W}.png`;
  a.href = c.toDataURL("image/png");
  a.click();
}

async function main() {
  if (!navigator.gpu) { $("nogpu").style.display = "grid"; $("panel").style.display = "none"; return; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { $("nogpu").style.display = "grid"; $("panel").style.display = "none"; return; }
  device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize } });
  device.lost.then((i) => { status.textContent = `GPU device lost: ${i.message}`; });
  viewer = new Viewer(device, canvas, navigator.gpu.getPreferredCanvasFormat());
  model = await TextureDiffusion.create(device, "./model");
  names = model.man.names;
  const sel = $("cls");
  names.map((n, i) => [n, i]).sort().forEach(([n, i]) => sel.add(new Option(n, i)));
  sel.value = names.indexOf("marbled");
  // shareable settings: ?t=veined&size=256&steps=24&cfg=3&seed=123&zoom=2
  const q = new URLSearchParams(location.search);
  if (q.has("t") && names.includes(q.get("t"))) sel.value = names.indexOf(q.get("t"));
  if (q.has("size")) {
    size = +q.get("size");
    for (const x of $("size").children) x.setAttribute("aria-pressed", +x.dataset.v === size);
  }
  for (const k of ["steps", "cfg", "zoom"]) if (q.has(k)) { $(k).value = q.get(k); $(`${k}V`).textContent = $(k).value; }
  if (q.has("seed")) seed = +q.get("seed");
  addEventListener("resize", resize);
  resize();
  for (const id of ["steps", "cfg", "zoom"]) {
    $(id).addEventListener("input", () => { $(`${id}V`).textContent = $(id).value; if (id === "zoom") draw(); });
  }
  $("smooth").addEventListener("change", draw);
  $("size").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $("size").children) x.setAttribute("aria-pressed", x === b);
    size = +b.dataset.v;
  });
  $("go").addEventListener("click", generate);
  $("rnd").addEventListener("click", () => { pickRandom(); if (!running) generate(); });
  $("save").addEventListener("click", save);
  $("dream").addEventListener("change", () => { if ($("dream").checked && !running) dreamNext(); });
  $("min").addEventListener("click", () => {
    const p = $("panel"); p.classList.toggle("min");
    $("min").textContent = p.classList.contains("min") ? "+" : "–";
  });
  status.textContent = "Ready";
  generate();
}

main().catch((e) => { status.textContent = `Error: ${e.message}`; console.error(e); });
