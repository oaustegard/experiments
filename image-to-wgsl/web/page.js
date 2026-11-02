// Gallery viewer plus the in-browser encoder ("Your image").
import { IMAGES } from "./shaders.js";

const $ = (id) => document.getElementById(id);
const T = 6;                 // the shader paints N * (t / T)^2 splats at time t > 0
// Phones and tablets: 1,024 splats crashed an iPhone tab (memory), so they stop at 256 and train on the
// smaller pixel budget. Other devices: up to 2,048 splats, twice the pixels per step and twice the steps,
// which on an M5 Max took 1,024 splats from 25.5 to 26.3 dB (GPU bridge, 2026-11-02). Which counts are
// offered also depends on a timed probe of this device's training step.
const MOBILE = /iPhone|iPad|Android/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
const PROFILE = MOBILE ? { maxN: 256, pixBudget: 2 ** 21, stepsScale: 1, maxDim: 512 }
                       : { maxN: 2048, pixBudget: 2 ** 22, stepsScale: 2, maxDim: 768 };
const MAXDIM = PROFILE.maxDim;   // uploaded images are fitted at most this many pixels on the longer side
const TRAIN = { init: "structure", importance: 0.5, pixBudget: PROFILE.pixBudget, stepsScale: PROFILE.stepsScale };
// TF.js trains on its WebGL backend. Its WebGPU backend computed wrong gradients past 32 splats per
// chunk and crashed the iPhone tab when limited to 32 (GPU bridge, 2026-11-02).
const TF_URLS = ["https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"];
const LABEL = { orig: "Original", avif: "AVIF, same bytes", webp: "WebP, same bytes", jpeg: "JPEG, same bytes" };
let dev, ctx, fmt, ub, cur, pipe, bg, anim = 0, view = "shader", ks = [16, 32, 64, 128, 256, 512, 1024];

const ksFor = (n) => { const a = []; for (let k = 16; k < n; k *= 2) a.push(k); a.push(n); return a; };

async function init() {
  const ad = await navigator.gpu?.requestAdapter();
  if (!ad) { $("stage").innerHTML = '<p class="err" style="padding:16px">WebGPU is not available in this browser, so the shaders cannot run. A recent Chrome, Edge or Safari has it.</p>'; return false; }
  dev = await ad.requestDevice();
  ctx = $("cv").getContext("webgpu");
  fmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device: dev, format: fmt, alphaMode: "premultiplied" });
  ub = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.lost.then((i) => console.log("device lost: " + i.message));
  window.gpuDone = () => dev.queue.onSubmittedWorkDone();   // for headless tests
  return true;
}

function load(entry) {
  cur = entry;
  const mod = dev.createShaderModule({ code: entry.wgsl });
  pipe = dev.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" }, fragment: { module: mod, entryPoint: "fs", targets: [{ format: fmt }] } });
  bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }] });
  $("src").textContent = entry.wgsl;
  for (const b of $("picker").children) b.setAttribute("aria-pressed", String(b.dataset.id === entry.id));
  ks = ksFor(entry.n);
  $("k").max = ks.length - 1;
  $("k").value = ks.length - 1;
  // comparison buttons: whatever this entry has
  const views = ["shader", "orig", ...Object.keys(entry.cmp)];
  $("views").replaceChildren(...views.map((v) => {
    const b = document.createElement("button");
    b.dataset.view = v; b.textContent = v === "shader" ? "Shader" : LABEL[v] || v;
    b.onclick = () => setView(v);
    return b;
  }));
  const full = entry.points[entry.points.length - 1];
  $("cmpnote").textContent = `Full shader: ${full.bytes.toLocaleString()} bytes, ${full.psnr} dB.` + Object.entries(entry.cmp).map(([k, c]) =>
    ` ${k.toUpperCase()} at or under that size: ${c.bytes.toLocaleString()} bytes, ${c.psnr} dB${c.scale > 1 ? ` (encoded at 1/${c.scale} size)` : ""}.`).join("");
  $("dl").hidden = !entry.own;
  if (entry.own) {
    $("dlw").href = URL.createObjectURL(new Blob([entry.wgsl], { type: "text/plain" }));
    $("dlh").href = URL.createObjectURL(new Blob([standalone(entry.wgsl)], { type: "text/html" }));
  }
  setView(views.includes(view) ? view : "shader");
  draw(timeFor(ks[$("k").value]));
}

function timeFor(k) { return k >= cur.n ? 0 : T * Math.sqrt((k + 0.5) / cur.n); }

function draw(t) {
  const cv = $("cv");
  const w = Math.max(1, Math.round(cv.clientWidth * devicePixelRatio)), h = Math.max(1, Math.round(cv.clientHeight * devicePixelRatio));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  dev.queue.writeBuffer(ub, 0, new Float32Array([w, h, t, 0]));
  const enc = dev.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  dev.queue.submit([enc.finish()]);
  readout(t);
}

function readout(t) {
  const n = t <= 0 ? cur.n : Math.min(cur.n, Math.floor(cur.n * Math.min(1, t * t / (T * T))));
  const p = cur.points.filter((q) => q.n <= Math.max(n, 16)).pop();
  $("readout").innerHTML = `<b>${n}</b> splats · <b>${p.bytes.toLocaleString()}</b> bytes · <b>${p.psnr}</b> dB` + (n === p.n ? "" : ` <span class="note">(nearest measured: ${p.n})</span>`);
}

function setView(v) {
  view = v;
  for (const b of document.querySelectorAll("[data-view]")) b.setAttribute("aria-pressed", String(b.dataset.view === v));
  const img = $("cmpimg");
  if (v === "shader") { img.hidden = true; $("tag").textContent = "shader"; return; }
  if (v !== "orig" && !cur.cmp[v]) { img.hidden = true; $("tag").textContent = `no ${v} fits`; return; }
  img.src = cur.src?.[v] ?? (v === "orig" ? `img/${cur.id}_orig.jpg` : `img/${cur.id}_${v}.${v}`);
  img.hidden = false;
  $("tag").textContent = v === "orig" ? "original" : `${v.toUpperCase()} ${cur.cmp[v].bytes.toLocaleString()} B`;
}

function paint() {
  cancelAnimationFrame(anim);
  setView("shader");
  const t0 = performance.now();
  const step = () => {
    const t = (performance.now() - t0) / 1000;
    if (t >= T) { $("k").value = ks.length - 1; draw(0); return; }
    draw(Math.max(t, 1e-3));
    anim = requestAnimationFrame(step);
  };
  step();
}

function addButton(e, own = false) {
  let b = [...$("picker").children].find((x) => x.dataset.id === e.id);
  if (!b) {
    b = document.createElement("button");
    b.textContent = e.id; b.dataset.id = e.id;
    $("picker").insertBefore(b, $("picker").querySelector(".own"));
  }
  b.onclick = () => { cancelAnimationFrame(anim); $("enc").hidden = !own; load(e); };
}

// ------------------------------------------------------------------ standalone page for downloads

function standalone(wgsl) {
  return `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Image as a shader</title>
<style>html,body{margin:0;height:100%;background:#fff}canvas{width:100%;height:100%;display:block}</style><canvas></canvas><script type="module">
const code=${JSON.stringify(wgsl)};const cv=document.querySelector("canvas");const ad=await navigator.gpu?.requestAdapter();
if(!ad){document.body.textContent="WebGPU is not available in this browser.";throw 0;}
const dev=await ad.requestDevice(),ctx=cv.getContext("webgpu"),fmt=navigator.gpu.getPreferredCanvasFormat();ctx.configure({device:dev,format:fmt,alphaMode:"premultiplied"});
const mod=dev.createShaderModule({code});const pipe=dev.createRenderPipeline({layout:"auto",vertex:{module:mod,entryPoint:"vs"},fragment:{module:mod,entryPoint:"fs",targets:[{format:fmt}]}});
const ub=dev.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const bg=dev.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ub}}]});
let t0=performance.now();const draw=(t)=>{const w=cv.width=cv.clientWidth*devicePixelRatio,h=cv.height=cv.clientHeight*devicePixelRatio;dev.queue.writeBuffer(ub,0,new Float32Array([w,h,t,0]));
const e=dev.createCommandEncoder(),p=e.beginRenderPass({colorAttachments:[{view:ctx.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:[1,1,1,1]}]});
p.setPipeline(pipe);p.setBindGroup(0,bg);p.draw(3);p.end();dev.queue.submit([e.finish()]);};
const step=()=>{const t=(performance.now()-t0)/1000;if(t>=6){draw(0);return;}draw(Math.max(t,1e-3));requestAnimationFrame(step);};step();
addEventListener("resize",()=>draw(0));cv.onclick=()=>{t0=performance.now();step();};
</script>`;
}

// ------------------------------------------------------------------ the encoder

let tfReady = null, trainer = null, upload = null, probed = null;

// Fill the splat menu with the counts this device can train: up to PROFILE.maxN, each labelled with an
// estimate from a timed probe of one training step; the default is the largest that fits in about a minute.
async function sizeMenu() {
  if (!probed) {
    $("encstat").textContent = "Timing this GPU…";
    const tf = await getTF();
    trainer ??= await import("./trainer.js");
    // a step costs a fixed overhead plus a part proportional to splats x pixels: two probes fit both
    const small = await trainer.probe(tf, { n: 64, nPix: 1024 }), big = await trainer.probe(tf, { n: 256, nPix: 8192 });
    const b = Math.max(0, (big - small) / (256 * 8192 - 64 * 1024)), a0 = Math.max(0, small - b * 64 * 1024);
    probed = { a: a0, b, small, big, backend: tf.getBackend() };
  }
  const opts = [];
  for (let n = 64; n <= PROFILE.maxN; n *= 2) {
      const work = (k) => k * Math.min(8192, Math.max(1024, Math.floor(PROFILE.pixBudget / k)));
    const sizes = trainer.checkpoints(n), sc = PROFILE.stepsScale, ms = (k) => probed.a + probed.b * work(k);
    const sec = (sizes.slice(0, -1).reduce((acc, k) => acc + 100 * sc * ms(k), 0) + 450 * sc * ms(n)) / 1000 + 2;
    if (sec <= 300 || n <= 256) opts.push([n, sec]);
  }
  const pick = opts.filter(([, s]) => s <= 75).pop() || opts[0];
  const fmt = (s) => s < 90 ? `about ${Math.max(5, Math.round(s / 5) * 5)} s` : `about ${Math.round(s / 60)} min`;
  $("nsel").replaceChildren(...opts.map(([n, s]) => { const o = document.createElement("option"); o.value = n; o.textContent = `${n} · ${fmt(s)}`; o.selected = n === pick[0]; return o; }));
}

function loadScript(src) {
  return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res;
    s.onerror = () => rej(new Error("could not load " + src)); document.head.append(s); });
}

async function getTF() {
  tfReady ??= (async () => {
    for (const u of TF_URLS) await loadScript(u);
    // shaders that take tensor shapes as uniforms: no recompiles as the splat list grows (2x faster on an M5 Max)
    tf.env().set("WEBGL_USE_SHAPES_UNIFORMS", true);
    if (!(await tf.setBackend("webgl"))) throw new Error("TensorFlow.js could not start its WebGL backend");
    await tf.ready();
    return window.tf;
  })();
  return tfReady;
}

function psnrU8(a, b) {      // RGBA byte arrays, alpha ignored
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i += 4) for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; s += d * d; n++; }
  return 10 * Math.log10(255 * 255 / Math.max(s / n, 1e-12));
}

function canvas2d(w, h) { const c = document.createElement("canvas"); c.width = w; c.height = h; return [c, c.getContext("2d")]; }
const toBlob = (c, type, q) => new Promise((r) => c.toBlob(r, type, q));

async function readUpload(file) {
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, MAXDIM / Math.max(bmp.width, bmp.height));
  const W = Math.max(16, Math.round(bmp.width * s)), H = Math.max(16, Math.round(bmp.height * s));
  const [c, x] = canvas2d(W, H);
  x.imageSmoothingQuality = "high";
  x.drawImage(bmp, 0, 0, W, H);
  const rgba = x.getImageData(0, 0, W, H).data;
  const png = URL.createObjectURL(await toBlob(c, "image/png"));
  return { W, H, rgba, canvas: c, png, name: file.name };
}

// Best encode of `type` at or under `budget` bytes, at full, half or quarter size, decoded back to W x H.
async function bestUnder(up, type, budget) {
  let best = null;
  for (const sc of [1, 2, 4]) {
    const w = Math.max(8, Math.round(up.W / sc)), h = Math.max(8, Math.round(up.H / sc));
    const [c, x] = canvas2d(w, h);
    x.imageSmoothingQuality = "high"; x.drawImage(up.canvas, 0, 0, w, h);
    for (let q = 0.02; q <= 0.96; q += 0.04) {
      const blob = await toBlob(c, type, q);
      if (!blob || blob.type !== type) return null;           // this browser cannot encode the type
      if (blob.size > budget) break;
      const bmp = await createImageBitmap(blob);
      const [d, y] = canvas2d(up.W, up.H);
      y.imageSmoothingQuality = "high"; y.drawImage(bmp, 0, 0, up.W, up.H);
      const p = psnrU8(y.getImageData(0, 0, up.W, up.H).data, up.rgba);
      if (!best || p > best.psnr) best = { psnr: +p.toFixed(2), bytes: blob.size, scale: sc, q: +q.toFixed(2), url: URL.createObjectURL(blob) };
    }
  }
  return best;
}

function showPreview({ rgb, width, height }) {
  const c = $("prev");
  if (c.width !== width || c.height !== height) { c.width = width; c.height = height; }
  const x = c.getContext("2d"), im = x.createImageData(width, height);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    im.data[j] = rgb[i] * 255; im.data[j + 1] = rgb[i + 1] * 255; im.data[j + 2] = rgb[i + 2] * 255; im.data[j + 3] = 255;
  }
  x.putImageData(im, 0, 0);
  c.hidden = false;
}

async function runEncode() {
  const n = +$("nsel").value;
  $("go").disabled = $("file").disabled = $("nsel").disabled = true;
  $("prog").hidden = false; $("prog").value = 0;
  try {
    $("encstat").textContent = "Loading TensorFlow.js…";
    const tf = await getTF();
    trainer ??= await import("./trainer.js");
    $("encstat").textContent = `Training on ${tf.getBackend()}…`;
    cancelAnimationFrame(anim); setView("shader");
    const r = await trainer.encode(tf, { rgba: upload.rgba, width: upload.W, height: upload.H }, {
      n, ...TRAIN, ...(window.__encodeOpts || {}),         // tests shorten training through this
      onProgress: (p) => {
        $("prog").value = p.step / p.totalSteps;
        $("encstat").textContent = `${p.phase} · step ${p.step} of ${p.totalSteps} · about ${p.psnrSample.toFixed(1)} dB · ${Math.round(p.seconds)} s`;
      },
      onPreview: showPreview,
    });
    $("encstat").textContent = "Measuring the shader and the JPEG/WebP files of the same size…";
    const wgsl = trainer.emitWGSL(r);
    const points = [];
    for (const p of r.points) points.push({ n: p.n, bytes: await trainer.gzipSize(trainer.emitWGSL(r, p.n)), psnr: +p.psnr.toFixed(2) });
    const budget = points[points.length - 1].bytes, cmp = {}, src = { orig: upload.png };
    for (const [k, type] of [["jpeg", "image/jpeg"], ["webp", "image/webp"]]) {
      const b = await bestUnder(upload, type, budget);
      if (b) { src[k] = b.url; cmp[k] = { psnr: b.psnr, bytes: b.bytes, scale: b.scale }; }
    }
    const entry = { id: "your image", own: true, wgsl, n, W: r.W, H: r.H, points, cmp, src };
    window.__encodeResult = { ...entry, seconds: r.seconds, backend: r.backend, P: Array.from(r.P), bg: Array.from(r.bg), A: r.A };
    $("prev").hidden = true;
    $("encstat").textContent = `Done in ${Math.round(r.seconds)} s on ${r.backend}: ${points[points.length - 1].bytes.toLocaleString()} bytes, ${points[points.length - 1].psnr} dB at ${upload.W}×${upload.H}.`;
    addButton(entry, true);
    load(entry);
    paint();
  } catch (e) {
    $("encstat").textContent = "Encoding failed: " + (e?.message || e);
    console.log("encode error", e?.stack || e);
    $("prev").hidden = true;
  } finally {
    $("prog").hidden = true;
    $("go").disabled = !upload; $("file").disabled = $("nsel").disabled = false;
  }
}

if (await init()) {
  const own = document.createElement("button");
  own.className = "own"; own.textContent = "Your image…"; own.dataset.id = "__own";
  own.onclick = () => { $("enc").hidden = false; for (const b of $("picker").children) b.setAttribute("aria-pressed", String(b === own)); $("file").focus(); };
  $("picker").append(own);
  for (const e of IMAGES) addButton(e);
  $("file").onchange = async () => {
    const f = $("file").files[0];
    if (!f) return;
    try {
      upload = await readUpload(f);
      await sizeMenu();
      $("go").disabled = false;
      $("encstat").textContent = `${f.name}: ${upload.W}×${upload.H} after scaling. Press Encode.`;
    } catch (e) { $("encstat").textContent = "Could not read that image: " + (e?.message || e); }
  };
  $("go").onclick = runEncode;
  $("k").oninput = () => { cancelAnimationFrame(anim); setView("shader"); draw(timeFor(ks[$("k").value])); };
  $("paint").onclick = paint;
  addEventListener("resize", () => { if (view === "shader") draw(timeFor(ks[$("k").value])); });
  load(IMAGES[0]);
  paint();
}
