import { ImageDiffusion, Viewer } from "./diffusion.js";

const $ = (id) => document.getElementById(id);
const canvas = $("c"), ov = $("ov"), status = $("status"), bar = $("bar").firstElementChild;
const S = 64;
const SRC = '<a href="https://github.com/oaustegard/experiments/tree/main/wgsl-image-diffusion">Source</a>';
const TAIL = "It makes 64×64 images; the large square shows one scaled up. ";
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
const q0 = new URLSearchParams(location.search);
const MK = MODELS[q0.get("m")] ? q0.get("m") : "animals";
const M = MODELS[MK];

let model, viewer, device, names, running = false, cancel = false, autoTimer = null;
let cls = 0, seed = (Math.random() * 1e9) | 0, shown = false;
// Picture mode: start from a picture (src) instead of pure noise. mask marks the pixels to regenerate.
let mode = "noise", tool = "paint", hasPicture = false, editing = false;
const src = document.createElement("canvas"), mask = document.createElement("canvas");
src.width = src.height = mask.width = mask.height = S;
const sctx = src.getContext("2d", { willReadFrequently: true }), mctx = mask.getContext("2d", { willReadFrequently: true });
const octx = ov.getContext("2d");
ov.width = ov.height = S;

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

// The picture being edited, with the mask tinted over it, drawn on a 2D canvas over the WebGPU one.
function showPicture(on) {
  editing = on && mode === "picture" && hasPicture;
  ov.style.display = editing ? "block" : "none";
  if (!editing) return;
  octx.clearRect(0, 0, S, S);
  octx.drawImage(src, 0, 0);
  const tint = document.createElement("canvas");
  tint.width = tint.height = S;
  const t = tint.getContext("2d");
  t.fillStyle = "#ff2fd0"; t.fillRect(0, 0, S, S);
  t.globalCompositeOperation = "destination-in"; t.drawImage(mask, 0, 0);
  octx.globalAlpha = 0.5; octx.drawImage(tint, 0, 0); octx.globalAlpha = 1;
}

function setMode(m) {
  mode = m;
  for (const b of $("mode").children) b.setAttribute("aria-pressed", b.dataset.v === m);
  $("picRows").hidden = m !== "picture";
  $("go").textContent = m === "picture" ? "Transform" : "Generate";
  showPicture(true);
}

function setTool(t) {
  tool = t;
  for (const b of $("tool").children) b.setAttribute("aria-pressed", b.dataset.v === t);
}

function setCls(v) {
  cls = v;
  for (const x of $("cls").children) x.setAttribute("aria-pressed", +x.dataset.v === v);
}

// cls as the sampler takes it: an index, or weights over the classes when blending
function classArg() {
  const b = +$("blend").value, a = +$("blendAmt").value;
  if (b < 0 || b === cls || a === 0) return cls;
  const w = new Array(names.length + 1).fill(0);
  w[cls] = 1 - a; w[b] = a;
  return w;
}

function label() {
  const b = +$("blend").value, a = +$("blendAmt").value;
  return b < 0 || b === cls || a === 0 ? names[cls] : `${names[cls]} + ${Math.round(a * 100)}% ${names[b]}`;
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

function loadPicture(drawFn) {
  sctx.clearRect(0, 0, S, S);
  drawFn(sctx);
  mctx.clearRect(0, 0, S, S);
  hasPicture = true;
  setMode("picture");
}

async function useResult() {
  if (!shown) return;
  const img = await pixels();
  loadPicture((c) => c.putImageData(img, 0, 0));
}

function upload(file) {
  const im = new Image();
  im.onload = () => {
    const s = Math.min(im.width, im.height);        // centre square crop
    loadPicture((c) => { c.imageSmoothingQuality = "high"; c.drawImage(im, (im.width - s) / 2, (im.height - s) / 2, s, s, 0, 0, S, S); });
    URL.revokeObjectURL(im.src);
    status.textContent = "Picture loaded: pick a strength and press Transform";
  };
  im.src = URL.createObjectURL(file);
}

function blankPicture() {
  loadPicture((c) => { c.fillStyle = "#7f7f7f"; c.fillRect(0, 0, S, S); });
  setTool("paint");
  status.textContent = "Sketch shapes and colours, then press Transform";
}

// image and mask as the sampler wants them: 3 x S x S in [-1, 1], and S x S with 1 = regenerate
function pictureArrays() {
  const p = sctx.getImageData(0, 0, S, S).data, m = mctx.getImageData(0, 0, S, S).data;
  const init = new Float32Array(3 * S * S), mk = new Float32Array(S * S);
  let any = false;
  for (let i = 0; i < S * S; i++) {
    for (let k = 0; k < 3; k++) init[k * S * S + i] = p[4 * i + k] / 127.5 - 1;
    mk[i] = m[4 * i + 3] > 127 ? 1 : 0;
    any ||= mk[i] > 0;
  }
  return { init, mask: any ? mk : null };
}

// painting on the stage: colour onto the picture, or the mask
let drawing = false, last = null;
function stroke(e) {
  const r = ov.getBoundingClientRect();
  const x = ((e.clientX - r.left) / r.width) * S, y = ((e.clientY - r.top) / r.height) * S;
  const ctx = tool === "mask" ? mctx : sctx;
  const rad = +$("brush").value / 2;
  ctx.fillStyle = tool === "mask" ? "#fff" : $("color").value;
  const pts = last ? Math.max(1, Math.ceil(Math.hypot(x - last[0], y - last[1]) / Math.max(0.5, rad / 2))) : 1;
  for (let k = 1; k <= pts; k++) {
    const px = last ? last[0] + ((x - last[0]) * k) / pts : x, py = last ? last[1] + ((y - last[1]) * k) / pts : y;
    ctx.beginPath(); ctx.arc(px, py, rad, 0, 2 * Math.PI); ctx.fill();
  }
  last = [x, y];
  showPicture(true);
}

async function addToHistory(rec) {
  const c = document.createElement("canvas");
  c.width = S; c.height = S;
  c.getContext("2d").putImageData(await pixels(), 0, 0);
  c.title = `${rec.label} · seed ${rec.seed} · ${rec.steps} steps · guidance ${rec.cfg} — click to edit this one`;
  c.addEventListener("click", () => { if (!running) loadPicture((x) => x.drawImage(c, 0, 0)); });
  const h = $("hist");
  h.prepend(c);
  while (h.children.length > 48) h.lastChild.remove();
}

async function generate() {
  if (running) { cancel = true; return; }
  clearTimeout(autoTimer);
  const picture = mode === "picture";
  if (picture && !hasPicture) { status.textContent = "Upload a photo, start a sketch, or use the current result first"; return; }
  running = true; cancel = false;
  $("go").textContent = "Stop";
  const steps = +$("steps").value, cfg = +$("cfg").value, strength = +$("strength").value;
  const rec = { label: label(), seed, steps, cfg };
  const opts = { cls: classArg(), H: S, W: S, steps, cfg, seed };
  if (picture) Object.assign(opts, pictureArrays(), { strength });
  showPicture(false);
  const t0 = performance.now();
  try {
    const done = await model.sample({ ...opts,
      onStep: async (i, t, n) => {
        shown = true;
        draw();                   // shows that step's estimate of the clean image
        bar.style.width = `${(100 * (i + 1)) / n}%`;
        const per = (performance.now() - t0) / (i + 1);
        status.textContent = `${rec.label} · step ${i + 1}/${n} · ${per.toFixed(0)} ms/step`;
      },
      shouldStop: () => cancel });
    await device.queue.onSubmittedWorkDone();
    draw();
    const ms = performance.now() - t0;
    if (done) {
      status.textContent = `${rec.label} · ${(ms / 1000).toFixed(2)} s · seed ${seed}` + (picture ? ` · strength ${strength}` : "");
      await addToHistory(rec);
    } else status.textContent = "Stopped";
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
    console.error(e);
  }
  running = false;
  $("go").textContent = mode === "picture" ? "Transform" : "Generate";
  seed = (Math.random() * 1e9) | 0;
  if ($("auto").checked && !cancel) autoTimer = setTimeout(() => {
    if (!$("auto").checked) return;
    if (mode === "noise") setCls((Math.random() * names.length) | 0);
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
    // a model whose weights are not published yet stays disabled
    if (b.dataset.v !== MK) fetch(`${MODELS[b.dataset.v].dir}/manifest.json`, { method: "HEAD" }).then((r) => {
      if (!r.ok) { b.disabled = true; b.title = "Still training"; }
    }).catch(() => { b.disabled = true; });
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
  const cap = (n) => n[0].toUpperCase() + n.slice(1);
  for (const [i, n] of names.entries()) {
    const b = document.createElement("button");
    b.dataset.v = i; b.textContent = cap(n);
    $("cls").append(b);
    $("blend").add(new Option(cap(n), i));
  }
  setCls(cls);
  // shareable settings: ?m=faces&a=man,%20smiling&steps=24&cfg=2&seed=123 (m: animals or faces; a: a class name)
  const q = q0;
  if (q.has("a") && names.includes(q.get("a"))) setCls(names.indexOf(q.get("a")));
  for (const k of ["steps", "cfg"]) if (q.has(k)) { $(k).value = q.get(k); $(`${k}V`).textContent = $(k).value; }
  if (q.has("seed")) seed = +q.get("seed");
  addEventListener("resize", resize);
  resize();
  for (const id of ["steps", "cfg", "strength", "brush", "blendAmt"]) {
    $(id).addEventListener("input", () => { $(`${id}V`).textContent = $(id).value; });
  }
  $("smooth").addEventListener("change", () => { draw(); ov.style.imageRendering = $("smooth").checked ? "auto" : "pixelated"; });
  $("cls").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    setCls(+b.dataset.v);
    if (!running && (mode === "noise" || hasPicture)) generate();
  });
  $("mode").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setMode(b.dataset.v); });
  $("tool").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { setTool(b.dataset.v); showPicture(true); } });
  $("upBtn").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", () => { if ($("file").files[0]) upload($("file").files[0]); $("file").value = ""; });
  $("useRes").addEventListener("click", useResult);
  $("sketch").addEventListener("click", blankPicture);
  $("clearMask").addEventListener("click", () => { mctx.clearRect(0, 0, S, S); showPicture(true); });
  $("showSrc").addEventListener("click", () => showPicture(!editing));
  ov.addEventListener("pointerdown", (e) => { if (running) return; drawing = true; last = null; ov.setPointerCapture(e.pointerId); stroke(e); });
  ov.addEventListener("pointermove", (e) => { if (drawing) stroke(e); });
  for (const ev of ["pointerup", "pointercancel"]) ov.addEventListener(ev, () => { drawing = false; last = null; });
  $("go").addEventListener("click", () => generate());
  $("save").addEventListener("click", save);
  $("auto").addEventListener("change", () => { if ($("auto").checked && !running) generate(); });
  setMode("noise");
  status.textContent = "Ready";
  generate();
}

main().catch((e) => { status.textContent = `Error: ${e.message}`; console.error(e); });
