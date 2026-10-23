// Live camera tagger: MobileCLIP-S0's vision tower on WebGPU (transformers.js), scored against
// 4,585 RAM++ tags embedded offline with the same model's text tower (run/embed_tags.py).
// The phone never runs a text model; a frame costs one vision pass plus a 4,585 x 512 dot product.
import { CLIPVisionModelWithProjection, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0";

const MODEL_ID = "Xenova/mobileclip_s0";
const SIZE = 256;          // preprocessor_config.json: shortest edge 256, center crop 256, /255, no mean/std
const TOP_K = 5;
const EMA = 0.5;           // weight of the newest frame in the smoothed embedding (camera mode only)

const $ = (id) => document.getElementById(id);
const video = $("video"), still = $("still"), statusEl = $("status"), tagsEl = $("tags"), statsEl = $("stats");
// Canvas drawImage downscaling does not low-pass filter: on COCO it lands at cos 0.975-0.996 to
// PIL's resize, exactly like naive bilinear. So the crop is drawn 1:1 (halved first while it is
// over 4x the target) and resampled in JS with PIL's separable triangle filter.
const src2d = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
const halves = [];
const filters = new Map();

let model, device, bank, scales = null, tags, nGeneral, dim, smooth = null, paused = false, facing = "environment", stream = null;
const timing = { prep: 0, model: 0, score: 0, interval: 0, last: 0 };

function f16ToF32(u16) {
  if (typeof Float16Array !== "undefined") return Float32Array.from(new Float16Array(u16.buffer));
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

// Loading screen: one row per stage, byte progress where there are bytes, a note when nothing moves.
const MB = (b) => (b / 1e6).toFixed(1);
let lastProgress = performance.now();
function stage(id, state, frac, label) {
  const el = $("st-" + id), bar = el.querySelector(".bar");
  el.className = "stage " + state;
  bar.classList.toggle("indet", state === "busy" && frac == null);
  if (frac != null) bar.style.width = `${(100 * frac).toFixed(1)}%`;
  if (label != null) el.querySelector(".head span:last-child").textContent = label;
  lastProgress = performance.now();
}
const stallTimer = setInterval(() => {
  const idle = (performance.now() - lastProgress) / 1000;
  $("note").textContent = idle > 15 ? `No progress for ${idle.toFixed(0)} s. Still trying; a slow network shows here.` : "";
}, 1000);

async function fetchWithProgress(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get("content-length") || 0;
  if (!r.body) return r.arrayBuffer();
  const reader = r.body.getReader(), parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onBytes(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}

// Two banks: "big" (RAM++ + Open Images V7, int8 rows + float32 row scales) and "small"
// (RAM++ only, float16). ?vocab=small picks the small one.
const VOCAB = new URLSearchParams(location.search).get("vocab") === "small" ? "small" : "big";
const BANKS = {
  small: { meta: "tags.json", bin: "tags.f16.bin" },
  big: { meta: "tags-big.json", bin: "tags-big.i8.bin" },
};

async function loadBank() {
  stage("tags", "busy", 0);
  const [meta, bin] = await Promise.all([
    fetch(BANKS[VOCAB].meta).then((r) => r.json()),
    fetchWithProgress(BANKS[VOCAB].bin, (g, t) => stage("tags", "busy", t ? g / t : null, t ? `${MB(g)} / ${MB(t)} MB` : `${MB(g)} MB`)),
  ]);
  tags = meta.tags; dim = meta.dim; nGeneral = meta.n_general ?? tags.length;
  if (meta.dtype === "float16") bank = f16ToF32(new Uint16Array(bin));
  else {
    bank = new Int8Array(bin, 0, tags.length * dim);
    scales = new Float32Array(bin, tags.length * dim, tags.length);
  }
  stage("tags", "done", 1, `${tags.length.toLocaleString()} tags · ${MB(bin.byteLength)} MB`);
}

async function loadModel() {
  device = "gpu" in navigator && (await navigator.gpu.requestAdapter().catch(() => null)) ? "webgpu" : "wasm";
  // fp32 on WebGPU: METHODS.md records transformers.js int8 on webgpu collapsing embeddings.
  stage("model", "busy", 0);
  const progress_callback = (e) => {
    if (e.status === "progress" && /\.onnx/.test(e.file)) {
      const frac = e.total ? e.loaded / e.total : null;  // no content-length: show bytes on an indeterminate bar
      stage("model", "busy", frac, e.total ? `${MB(e.loaded)} / ${MB(e.total)} MB` : `${MB(e.loaded)} MB`);
    } else if (e.status === "done" && /\.onnx/.test(e.file)) {
      stage("model", "busy", 1, "preparing GPU session");
    } else if (e.status === "initiate" || e.status === "download") lastProgress = performance.now();
  };
  model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { device, dtype: device === "webgpu" ? "fp32" : "q8", progress_callback });
  stage("model", "done", 1, device === "webgpu" ? "46 MB · WebGPU" : "12 MB · CPU (no WebGPU)");
}

// PIL's BILINEAR resample as a 1-D filter: for each output index, the first input index and weights.
function filter(inSize) {
  if (filters.has(inSize)) return filters.get(inSize);
  const scale = inSize / SIZE, support = Math.max(scale, 1), f = [];
  for (let o = 0; o < SIZE; o++) {
    const c = (o + 0.5) * scale, lo = Math.max(0, Math.floor(c - support + 0.5)), hi = Math.min(inSize, Math.floor(c + support + 0.5));
    const w = []; let sum = 0;
    for (let i = lo; i < hi; i++) { const x = Math.max(0, 1 - Math.abs((i - c + 0.5) / support)); w.push(x); sum += x; }
    f.push([lo, Float32Array.from(w, (x) => x / sum)]);
  }
  filters.set(inSize, f);
  return f;
}

// Center square of any drawable -> 1x3x256x256 float32 in [0, 1].
function pixels(src, w, h) {
  let s = Math.min(w, h), from = src, sx = Math.floor((w - s) / 2), sy = Math.floor((h - s) / 2);  // integer offsets, as PIL crops
  for (let i = 0; s > 4 * SIZE; i++) {          // camera frames over 1024 px: cheap halving first
    const half = Math.floor(s / 2);
    if (!halves[i]) halves[i] = document.createElement("canvas").getContext("2d");
    const c = halves[i];
    if (c.canvas.width !== half) c.canvas.width = c.canvas.height = half;
    c.drawImage(from, sx, sy, s, s, 0, 0, half, half);
    from = c.canvas; s = half; sx = sy = 0;
  }
  if (src2d.canvas.width !== s) src2d.canvas.width = src2d.canvas.height = s;
  src2d.drawImage(from, sx, sy, s, s, 0, 0, s, s);
  const rgba = src2d.getImageData(0, 0, s, s).data, f = filter(s);
  const tmp = new Float32Array(3 * s * SIZE);  // horizontal pass: s rows x SIZE cols x 3
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < SIZE; x++) {
      const [lo, wt] = f[x]; let r = 0, g = 0, b = 0;
      for (let k = 0, p = 4 * (y * s + lo); k < wt.length; k++, p += 4) { r += wt[k] * rgba[p]; g += wt[k] * rgba[p + 1]; b += wt[k] * rgba[p + 2]; }
      const q = 3 * (y * SIZE + x); tmp[q] = r; tmp[q + 1] = g; tmp[q + 2] = b;
    }
  }
  const n = SIZE * SIZE, out = new Float32Array(3 * n);
  for (let y = 0; y < SIZE; y++) {
    const [lo, wt] = f[y];
    for (let x = 0; x < SIZE; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0, q = 3 * (lo * SIZE + x); k < wt.length; k++, q += 3 * SIZE) { r += wt[k] * tmp[q]; g += wt[k] * tmp[q + 1]; b += wt[k] * tmp[q + 2]; }
      // PIL rounds to uint8 between passes and at the end; only the final rounding is kept here.
      const i = y * SIZE + x;
      out[i] = Math.min(255, Math.max(0, Math.round(r))) / 255;
      out[n + i] = Math.min(255, Math.max(0, Math.round(g))) / 255;
      out[2 * n + i] = Math.min(255, Math.max(0, Math.round(b))) / 255;
    }
  }
  return new Tensor("float32", out, [1, 3, SIZE, SIZE]);
}

async function embed(src, w, h) {
  const t0 = performance.now(), pixel_values = pixels(src, w, h), t1 = performance.now();
  const { image_embeds } = await model({ pixel_values });
  timing.prep = t1 - t0; timing.model = performance.now() - t1;
  const v = Float32Array.from(image_embeds.data);
  let norm = 0; for (const x of v) norm += x * x; norm = Math.sqrt(norm);
  for (let i = 0; i < v.length; i++) v[i] /= norm;
  return v;
}

// Top 5 among the general RAM++ rows and top 3 among the Open Images rows after them. Ranking
// all rows together let rare specific labels push out the general ones (results/eval_vocab.json).
const TOP_SPECIFIC = 3;
function topk(s, from, to, k) {
  const top = [];
  for (let r = from; r < to; r++) {
    if (top.length < k || s[r] > top[top.length - 1][1]) {
      top.push([r, s[r]]); top.sort((a, b) => b[1] - a[1]); if (top.length > k) top.pop();
    }
  }
  return top.map(([r, x]) => ({ tag: tags[r], score: x }));
}

function score(v) {
  const t0 = performance.now(), n = tags.length, s = new Float32Array(n);
  for (let r = 0, o = 0; r < n; r++, o += dim) {
    let acc = 0;
    for (let c = 0; c < dim; c++) acc += bank[o + c] * v[c];
    s[r] = scales ? acc * scales[r] : acc;
  }
  const out = topk(s, 0, nGeneral, TOP_K);
  out.specific = topk(s, nGeneral, n, TOP_SPECIFIC);
  timing.score = performance.now() - t0;
  return out;
}

function render(top) {
  const hi = top[0].score, lo = Math.min(...top.map((t) => t.score)) - 0.02;
  tagsEl.innerHTML = top.map((t) => `<li><span class="name"><span class="fill" style="width:${
    (100 * (t.score - lo) / (hi - lo)).toFixed(0)}%"></span>${t.tag}</span><span class="score">${t.score.toFixed(3)}</span></li>`).join("");
  $("specific").innerHTML = top.specific.length
    ? top.specific.map((t) => `<span class="chip">${t.tag} <em>${t.score.toFixed(2)}</em></span>`).join("") : "";
  const fps = timing.interval ? (1000 / timing.interval).toFixed(1) : "–";
  statsEl.innerHTML = `${fps} fps<br>resize ${timing.prep.toFixed(1)} · model ${timing.model.toFixed(1)} · tags ${timing.score.toFixed(1)} ms · ${device}`;
}

async function tick() {
  if (!paused && stream && video.readyState >= 2) {
    const v = await embed(video, video.videoWidth, video.videoHeight);
    if (smooth) {
      let norm = 0;
      for (let i = 0; i < v.length; i++) { smooth[i] = EMA * v[i] + (1 - EMA) * smooth[i]; norm += smooth[i] ** 2; }
      norm = Math.sqrt(norm); for (let i = 0; i < v.length; i++) smooth[i] /= norm;
    } else smooth = v;
    render(score(smooth));
    const now = performance.now();
    if (timing.last) timing.interval = timing.interval ? 0.8 * timing.interval + 0.2 * (now - timing.last) : now - timing.last;
    timing.last = now;
  }
  requestAnimationFrame(tick);
}

async function startCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  video.srcObject = stream; await video.play();
  video.style.display = ""; still.style.display = "none";
  video.style.transform = facing === "user" ? "scaleX(-1)" : "";
  smooth = null; timing.last = 0; timing.interval = 0;
}

async function tagImage(url) {
  const img = new Image(); img.crossOrigin = "anonymous"; img.src = url; await img.decode();
  const v = await embed(img, img.naturalWidth, img.naturalHeight);
  const top = score(v);
  return { emb: Array.from(v), top, specific: top.specific, img };
}

async function showStill(url) {
  paused = true; $("pause").textContent = "Camera";
  const { top } = await tagImage(url);
  still.src = url; still.style.display = "block"; video.style.display = "none";
  render(top);
}

$("pause").onclick = async () => {
  if (still.style.display === "block") { paused = false; $("pause").textContent = "Pause"; return startCamera().catch(camFail); }
  paused = !paused; $("pause").textContent = paused ? "Resume" : "Pause";
};
$("flip").onclick = () => { facing = facing === "user" ? "environment" : "user"; startCamera().catch(camFail); };
$("file").onchange = (e) => { const f = e.target.files[0]; if (f) showStill(URL.createObjectURL(f)); };
function camFail(err) { statusEl.textContent = `No camera (${err.name || err.message}). Pick a photo instead.`; }

window.tagger = { tagImage, ready: null };
stage("runtime", "done", 1, "loaded");
window.tagger.ready = (async () => {
  try {
    const t0 = performance.now();
    await Promise.all([loadBank(), loadModel()]);
    // Warm-up pass: the first WebGPU run compiles shaders.
    stage("warm", "busy", null, "first run");
    await embed(src2d.canvas, SIZE, SIZE);
    stage("warm", "done", 1, `${(timing.model / 1000).toFixed(1)} s`);
    clearInterval(stallTimer); $("note").textContent = "";
    statusEl.textContent = `Ready in ${((performance.now() - t0) / 1000).toFixed(1)} s · MobileCLIP-S0 on ${device} · ${tags.length.toLocaleString()} tags`;
    $("pause").disabled = $("flip").disabled = false;
    requestAnimationFrame(tick);
    const q = new URLSearchParams(location.search).get("img");
    if (q) { $("boot").hidden = true; await showStill(q); return; }
    // Camera starts on a tap: iOS wants a gesture for the permission prompt, and the user sees it coming.
    $("start").hidden = false;
    $("start").onclick = async () => {
      $("start").disabled = true; $("start").textContent = "Starting camera…";
      await startCamera().catch(camFail);
      $("boot").hidden = true;
    };
  } catch (err) {
    console.error(err); clearInterval(stallTimer);
    $("err").textContent = `Failed: ${err.message}`; statusEl.textContent = `Failed: ${err.message}`;
    throw err;
  }
})();
