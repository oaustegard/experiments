import { ImageDiffusion, Viewer } from "./diffusion.js";

const $ = (id) => document.getElementById(id);
const canvas = $("c"), status = $("status"), bar = $("bar").firstElementChild;
const S = 64;
const SRC = '<a href="https://github.com/oaustegard/experiments/tree/main/wgsl-image-diffusion">Source</a>';
const TAIL = "It makes 64×64 images; the large square shows one scaled up. Earlier results collect below it: " +
  "click one to draw it again from the same random seed. ";
const MODELS = {
  animals: { dir: "./model", title: "Animal Diffusion", label: "Animal", what: "animal",
    sub: "A 23M-parameter diffusion model with self-attention, in hand-written WGSL, running on your GPU. " +
      'It was trained on the 15,803 close-up animal-face photos of <a href="https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq">AFHQv2</a> ' +
      "(cats, dogs, and wild animals such as lions, tigers, foxes and wolves). " },
  faces: { dir: "./model-faces", title: "Face Diffusion", label: "Face", what: "kind of face",
    sub: "The same 23M-parameter diffusion model with self-attention, in hand-written WGSL, running on your GPU, " +
      'trained on the 162,770 aligned celebrity photos of <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">CelebA</a>. ' +
      "Its four classes come from two of the dataset's annotated labels, male or female and smiling or not. " },
};
const MK = MODELS[new URLSearchParams(location.search).get("m")] ? new URLSearchParams(location.search).get("m") : "animals";
const M = MODELS[MK];

let model, viewer, device, names, running = false, cancel = false, autoTimer = null;
let cls = 0, seed = (Math.random() * 1e9) | 0, shown = false;

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  draw();
}

// Draw only inside requestAnimationFrame (Safari on iOS presents unreliably from other callbacks);
// draw() just asks for the next frame.
let frameQueued = false;
function draw() {
  if (!viewer || !shown || frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    viewer.draw(model.dBuf, S, S, canvas.width / S, $("smooth").checked);
  });
}

function setCls(v) {
  cls = v;
  for (const x of $("cls").children) x.setAttribute("aria-pressed", +x.dataset.v === v);
}

async function pixels() {
  const d = await model.read(model.dBuf, 3 * S * S);
  const img = new ImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    for (let k = 0; k < 3; k++) img.data[4 * i + k] = Math.round(Math.min(1, Math.max(0, (d[k * S * S + i] + 1) / 2)) * 255);
    img.data[4 * i + 3] = 255;
  }
  return img;
}

async function addToHistory(rec) {
  const c = document.createElement("canvas");
  c.width = S; c.height = S;
  c.getContext("2d").putImageData(await pixels(), 0, 0);
  c.title = `${names[rec.cls]} · seed ${rec.seed} · ${rec.steps} steps · guidance ${rec.cfg}`;
  c.addEventListener("click", () => {
    if (running) return;
    setCls(rec.cls);
    $("steps").value = rec.steps; $("stepsV").textContent = rec.steps;
    $("cfg").value = rec.cfg; $("cfgV").textContent = rec.cfg;
    generate(rec.seed);
  });
  const h = $("hist");
  h.prepend(c);
  while (h.children.length > 48) h.lastChild.remove();
}

async function generate(fixedSeed) {
  if (running) { cancel = true; return; }
  clearTimeout(autoTimer);
  running = true; cancel = false;
  $("go").textContent = "Stop";
  if (fixedSeed !== undefined) seed = fixedSeed;
  const steps = +$("steps").value, cfg = +$("cfg").value, rec = { cls, seed, steps, cfg };
  const t0 = performance.now();
  try {
    const done = await model.sample({ cls, H: S, W: S, steps, cfg, seed,
      onStep: async (i) => {
        shown = true;
        draw();                   // queued behind step i, so it shows that step's estimate of the clean image
        bar.style.width = `${(100 * (i + 1)) / steps}%`;
        const per = (performance.now() - t0) / (i + 1);
        status.textContent = `${names[cls]} · step ${i + 1}/${steps} · ${per.toFixed(0)} ms/step`;
      },
      shouldStop: () => cancel });
    await device.queue.onSubmittedWorkDone();
    draw();
    const ms = performance.now() - t0;
    if (done) {
      status.textContent = `${names[cls]} · ${steps} steps in ${(ms / 1000).toFixed(2)} s · seed ${seed}`;
      await addToHistory(rec);
    } else status.textContent = "Stopped";
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    console.error(e);
  }
  running = false;
  $("go").textContent = "Generate";
  seed = (Math.random() * 1e9) | 0;
  if ($("auto").checked && !cancel) autoTimer = setTimeout(() => {
    if (!$("auto").checked) return;
    setCls((Math.random() * 3) | 0);
    generate();
  }, 1200);
}

async function save() {
  if (!shown) return;
  const c = document.createElement("canvas");
  c.width = S; c.height = S;
  c.getContext("2d").putImageData(await pixels(), 0, 0);
  const a = document.createElement("a");
  a.download = `${names[cls]}-${seed}.png`;
  a.href = c.toDataURL("image/png");
  a.click();
}

async function main() {
  $("title").textContent = M.title;
  $("sub").innerHTML = M.sub + TAIL + SRC;
  $("clsLabel").textContent = M.label;
  $("what").textContent = M.what;
  for (const b of $("mdl").children) {
    b.setAttribute("aria-pressed", b.dataset.v === MK);
    b.addEventListener("click", () => { if (b.dataset.v !== MK) location.search = `?m=${b.dataset.v}`; });
  }
  const fail = () => { $("nogpu").style.display = "block"; $("app").style.display = "none"; };
  if (!navigator.gpu) return fail();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return fail();
  device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize } });
  device.lost.then((i) => { status.textContent = `GPU device lost: ${i.message}`; });
  viewer = new Viewer(device, canvas, navigator.gpu.getPreferredCanvasFormat());
  model = await ImageDiffusion.create(device, M.dir);
  names = model.man.names;
  for (const [i, n] of names.entries()) {
    const b = document.createElement("button");
    b.dataset.v = i; b.textContent = n[0].toUpperCase() + n.slice(1);
    $("cls").append(b);
  }
  setCls(cls);
  // shareable settings: ?m=faces&a=man,%20smiling&steps=24&cfg=2&seed=123 (m: animals or faces; a: a class name)
  const q = new URLSearchParams(location.search);
  if (q.has("a") && names.includes(q.get("a"))) setCls(names.indexOf(q.get("a")));
  for (const k of ["steps", "cfg"]) if (q.has(k)) { $(k).value = q.get(k); $(`${k}V`).textContent = $(k).value; }
  if (q.has("seed")) seed = +q.get("seed");
  addEventListener("resize", resize);
  resize();
  for (const id of ["steps", "cfg"]) $(id).addEventListener("input", () => { $(`${id}V`).textContent = $(id).value; });
  $("smooth").addEventListener("change", draw);
  $("cls").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    setCls(+b.dataset.v);
    if (!running) generate();
  });
  $("go").addEventListener("click", () => generate());
  $("save").addEventListener("click", save);
  $("auto").addEventListener("change", () => { if ($("auto").checked && !running) generate(); });
  status.textContent = "Ready";
  generate();
}

main().catch((e) => { status.textContent = `Error: ${e.message}`; console.error(e); });
