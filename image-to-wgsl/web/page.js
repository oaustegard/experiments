// Gallery viewer plus the in-browser encoder ("Your image").
import { IMAGES } from "./shaders.js";
import { emitInstanced, createRenderer } from "./instanced.js";
import { gzipSize, emitWGSL, checkpoints } from "./trainer.js";
import { encodeGPU, probeGPU } from "./gputrain.js";

const $ = (id) => document.getElementById(id);
const T = 6;                 // the shader paints N * (t / T)^2 splats at time t > 0
// Training runs in WebGPU compute shaders (gputrain.js): every pixel every step, each pixel meeting only
// the splats that reach its 16x16 tile. On an M5 Max that fitted 2,048 splats of a 484x512 image to 28.1 dB
// in 2.3 s, where TF.js took 63 s for 26.8 dB. 8,192 splats reached 30.1 dB in 4.1 s there and 30.2 dB in
// 61 s on an iPhone (GPU bridge, 2026-11-02). TF.js on WebGL is the fallback: add ?tf to the URL, or it
// takes over if the WebGPU trainer fails. Under TF.js, phones stop at 256 splats (1,024 crashed an iPhone
// tab) and other devices at 2,048. Which counts are offered also depends on timed steps on this device.
const MOBILE = /iPhone|iPad|Android/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
const FORCE_TF = new URLSearchParams(location.search).has("tf");
const GPU_MAXN = 8192;
const TF_PROFILE = MOBILE ? { maxN: 256, pixBudget: 2 ** 21, stepsScale: 1 } : { maxN: 2048, pixBudget: 2 ** 22, stepsScale: 2 };
const MAXDIM = MOBILE ? 512 : 768;   // uploaded images are fitted at most this many pixels on the longer side
const TRAIN_TF = { init: "structure", importance: 0.5, pixBudget: TF_PROFILE.pixBudget, stepsScale: TF_PROFILE.stepsScale };
const GPU_STEPS = { stepsStage: 150, stepsFinal: 500, stepsQat: 200 };   // 3x or 8x as many changed 2,048 splats by under 0.15 dB
// TF.js trains on its WebGL backend. Its WebGPU backend computed wrong gradients past 32 splats per
// chunk and crashed the iPhone tab when limited to 32 (GPU bridge, 2026-11-02).
const TF_URLS = ["https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"];
const LABEL = { orig: "Original", avif: "AVIF, same bytes", webp: "WebP, same bytes", jpeg: "JPEG, same bytes" };
let dev, ctx, fmt, ub, cur, pipe, bg, rend = null, anim = 0, view = "shader", ks = [16, 32, 64, 128, 256, 512, 1024];

const ksFor = (n) => { const a = []; for (let k = 16; k < n; k *= 2) a.push(k); a.push(n); return a; };

async function init() {
  const ad = await navigator.gpu?.requestAdapter();
  if (!ad) { $("stage").innerHTML = '<p class="err" style="padding:16px">WebGPU is not available in this browser, so the shaders cannot run. A recent Chrome, Edge or Safari has it.</p>'; return false; }
  // float32 blending makes the instanced renderer's painting exact; without it, it blends in float16
  // the trainer keeps a snapshot of the image per prefix: allow storage buffers as large as the adapter does
  dev = await ad.requestDevice({ requiredFeatures: ad.features.has("float32-blendable") ? ["float32-blendable"] : [],
    requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
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
  // Draw with the instanced renderer (one quad per splat, instanced.js) when the entry carries its splats;
  // the per-pixel shader is the file whose size is measured, and the fallback.
  rend?.destroy(); rend = null;
  if (entry.P) {
    entry.inst ??= emitInstanced({ P: Int32Array.from(entry.P), bg: Int32Array.from(entry.bg), A: entry.A, n: entry.n });
    rend = createRenderer(dev, fmt, entry.inst);
    if (entry.instGz == null) gzipSize(entry.inst).then((g) => { entry.instGz = g; if (cur === entry) note(entry); });
  } else {
    const mod = dev.createShaderModule({ code: entry.wgsl });
    pipe = dev.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" }, fragment: { module: mod, entryPoint: "fs", targets: [{ format: fmt }] } });
    bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }] });
  }
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
  note(entry);
  $("dl").hidden = !entry.own;
  if (entry.own) {
    $("dlw").href = URL.createObjectURL(new Blob([entry.wgsl], { type: "text/plain" }));
    $("dlh").href = URL.createObjectURL(new Blob([standalone(entry)], { type: "text/html" }));
  }
  cam.z = 1; cam.x = cam.y = 0.5; applyCam();
  setView(views.includes(view) ? view : "shader");
  draw(timeFor(ks[$("k").value]));
}

// ------------------------------------------------------------------ zoom and pan
// The camera zooms by z about image point (x, y), given as fractions of the image's width and height. The
// instanced renderer redraws the splats at that scale; comparison images get the same view as a CSS
// transform, with pixelated scaling so their pixels show.
const cam = { z: 1, x: 0.5, y: 0.5 }, ZMAX = 64;
let painting = false, redrawPending = false;
function camGeom() { const st = $("stage"), A = cur.A ?? cur.W / cur.H; return { A, Sw: st.clientWidth, Sh: st.clientHeight, k: Math.min(st.clientWidth / A, st.clientHeight) * cam.z }; }
function clampCam() {
  cam.z = Math.min(ZMAX, Math.max(1, cam.z));
  const m = 0.5 / cam.z;   // keep the view over the image
  cam.x = Math.min(1 - m, Math.max(m, cam.x)); cam.y = Math.min(1 - m, Math.max(m, cam.y));
}
function applyCam() {
  const img = $("cmpimg"), { A, k } = camGeom(), k1 = k / cam.z;
  img.style.transform = cam.z === 1 ? "" : `scale(${cam.z}) translate(${(0.5 - cam.x) * A * k1}px, ${(0.5 - cam.y) * k1}px)`;
  img.style.imageRendering = cam.z > 1 ? "pixelated" : "";
  $("zlvl").textContent = cam.z === 1 ? "" : `${cam.z < 9.95 ? cam.z.toFixed(1) : Math.round(cam.z)}×`;
  $("stage").classList.toggle("zoomed", cam.z > 1);
}
function redraw() {
  applyCam();
  if (painting || redrawPending) return;   // a running paint-in draws every frame anyway
  redrawPending = true;
  requestAnimationFrame(() => { redrawPending = false; if (pv) drawPreview(); else draw(timeFor(ks[$("k").value])); });
}
function zoomAt(mx, my, f) {           // keep the image point under (mx, my) in place
  let g = camGeom();
  const qx = cam.x * g.A + (mx - g.Sw / 2) / g.k, qy = cam.y + (my - g.Sh / 2) / g.k;
  cam.z = Math.min(ZMAX, Math.max(1, cam.z * f)); g = camGeom();
  cam.x = (qx - (mx - g.Sw / 2) / g.k) / g.A; cam.y = qy - (my - g.Sh / 2) / g.k;
  clampCam(); redraw();
}
function panBy(dx, dy) { const g = camGeom(); cam.x -= dx / g.k / g.A; cam.y -= dy / g.k; clampCam(); redraw(); }
function resetCam() { cam.z = 1; cam.x = cam.y = 0.5; redraw(); }
function initZoom() {
  const st = $("stage"), pts = new Map();
  const at = (e) => { const r = st.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  st.addEventListener("wheel", (e) => { e.preventDefault(); const [x, y] = at(e); zoomAt(x, y, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.002))); }, { passive: false });
  st.addEventListener("pointerdown", (e) => { if (e.target.closest("button")) return; st.setPointerCapture(e.pointerId); pts.set(e.pointerId, at(e)); });
  st.addEventListener("pointermove", (e) => {
    const prev = pts.get(e.pointerId); if (!prev) return;
    const p = at(e);
    if (pts.size === 1) panBy(p[0] - prev[0], p[1] - prev[1]);
    else {                             // two fingers: pan with the midpoint, zoom with the spread
      const o = [...pts].find(([id]) => id !== e.pointerId)[1];
      const d0 = Math.hypot(prev[0] - o[0], prev[1] - o[1]), d1 = Math.hypot(p[0] - o[0], p[1] - o[1]);
      panBy((p[0] - prev[0]) / 2, (p[1] - prev[1]) / 2);
      if (d0 > 0) zoomAt((p[0] + o[0]) / 2, (p[1] + o[1]) / 2, d1 / d0);
    }
    pts.set(e.pointerId, p);
  });
  for (const ev of ["pointerup", "pointercancel"]) st.addEventListener(ev, (e) => pts.delete(e.pointerId));
  st.addEventListener("dblclick", (e) => { if (!e.target.closest("button")) resetCam(); });
  $("zin").onclick = () => zoomAt(st.clientWidth / 2, st.clientHeight / 2, 2);
  $("zout").onclick = () => zoomAt(st.clientWidth / 2, st.clientHeight / 2, 0.5);
  $("zfit").onclick = resetCam;
}

function note(entry) {
  const full = entry.points[entry.points.length - 1];
  $("cmpnote").textContent = `Full shader: ${full.bytes.toLocaleString()} bytes, ${full.psnr} dB.` + Object.entries(entry.cmp).map(([k, c]) =>
    ` ${k.toUpperCase()} at or under that size: ${c.bytes.toLocaleString()} bytes, ${c.psnr} dB${c.scale > 1 ? ` (encoded at 1/${c.scale} size)` : ""}.`).join("") +
    (entry.instGz ? ` Drawn by an instanced version of the same data (${entry.instGz.toLocaleString()} bytes gzipped), one quad per splat.` : "");
}

const nAt = (t) => t <= 0 ? cur.n : Math.min(cur.n, Math.floor(cur.n * Math.min(1, t * t / (T * T))));

function timeFor(k) { return k >= cur.n ? 0 : T * Math.sqrt((k + 0.5) / cur.n); }

function draw(t) {
  const cv = $("cv");
  const w = Math.max(1, Math.round(cv.clientWidth * devicePixelRatio)), h = Math.max(1, Math.round(cv.clientHeight * devicePixelRatio));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  if (rend) { rend.draw(ctx.getCurrentTexture().createView(), w, h, nAt(t), cam); readout(t); return; }
  dev.queue.writeBuffer(ub, 0, new Float32Array([w, h, t, 0]));
  const enc = dev.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  dev.queue.submit([enc.finish()]);
  readout(t);
}

function readout(t) {
  const n = nAt(t);
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

function stopPaint() { cancelAnimationFrame(anim); painting = false; }

function paint() {
  stopPaint();
  setView("shader");
  const t0 = performance.now();
  painting = true;
  const step = () => {
    const t = (performance.now() - t0) / 1000;
    if (t >= T) { painting = false; $("k").value = ks.length - 1; draw(0); return; }
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
  b.onclick = () => { stopPaint(); $("enc").hidden = !own; load(e); };
}

// ------------------------------------------------------------------ standalone page for downloads

// Opens on the finished image. "Paint it in" replays the painting; the slider steps through the measured
// prefixes (16, 32, ... splats) with their bytes and PSNR from the encoder. Scroll or pinch zooms and drag
// pans, redrawing the splats at each scale. Shader only: no source image or comparisons.
function standalone(entry) {
  const pts = entry.points.map((p) => [p.n, p.bytes, p.psnr]);
  return `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Image as a shader</title>
<style>html,body{margin:0;height:100%;background:#fff;color:#222;font:14px/1.4 system-ui,sans-serif}body{display:flex;flex-direction:column}
canvas{flex:1;min-height:0;width:100%;display:block;touch-action:none;cursor:zoom-in}.bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;padding:10px 16px;border-top:1px solid #ddd}
.bar input{flex:1;min-width:140px}.bar button{font:inherit;padding:6px 12px}#ro,#zl{font-variant-numeric:tabular-nums}.near,.hint{color:#777}</style>
<canvas title="Scroll or pinch to zoom, drag to pan, double-click to fit"></canvas><div class="bar"><button id="go">Paint it in</button><input id="k" type="range" min="0" step="1" aria-label="Splats drawn"><span id="ro"></span>
<span class="hint">Scroll or pinch to zoom; drag to pan.</span><span id="zl"></span><button id="fit">Fit</button></div>
<script type="module">
const code=${JSON.stringify(entry.inst)},N=${entry.n},A=${entry.A},PTS=${JSON.stringify(pts)},T=6;
const $=(id)=>document.getElementById(id),cv=document.querySelector("canvas"),k=$("k"),ro=$("ro"),zl=$("zl"),ad=await navigator.gpu?.requestAdapter();
if(!ad){document.body.textContent="WebGPU is not available in this browser.";throw 0;}
const dev=await ad.requestDevice({requiredFeatures:ad.features.has("float32-blendable")?["float32-blendable"]:[]}),ctx=cv.getContext("webgpu"),fmt=navigator.gpu.getPreferredCanvasFormat();ctx.configure({device:dev,format:fmt,alphaMode:"premultiplied"});
${createRenderer.toString()}
const rd=createRenderer(dev,fmt,code),cam={z:1,x:.5,y:.5};let cur=N,anim=0,painting=false,pend=false;k.max=PTS.length-1;k.value=PTS.length-1;
const show=(n)=>{cur=n;const w=Math.max(1,Math.round(cv.clientWidth*devicePixelRatio)),h=Math.max(1,Math.round(cv.clientHeight*devicePixelRatio));
if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}rd.draw(ctx.getCurrentTexture().createView(),w,h,n,cam);
const i=Math.max(0,PTS.findLastIndex((q)=>q[0]<=n)),p=PTS[i];k.value=i;
ro.innerHTML="<b>"+n.toLocaleString()+"</b> splats · <b>"+p[1].toLocaleString()+"</b> bytes · <b>"+p[2]+"</b> dB"+(n===p[0]?"":' <span class="near">(nearest measured: '+p[0]+")</span>");};
const stop=()=>{cancelAnimationFrame(anim);painting=false;};
k.oninput=()=>{stop();show(PTS[k.value][0]);};
$("go").onclick=()=>{stop();painting=true;const t0=performance.now();
const step=()=>{const t=(performance.now()-t0)/1000;if(t>=T){painting=false;show(N);return;}show(Math.min(N,Math.floor(N*t*t/(T*T))));anim=requestAnimationFrame(step);};step();};
const geo=()=>({Sw:cv.clientWidth,Sh:cv.clientHeight,k:Math.min(cv.clientWidth/A,cv.clientHeight)*cam.z});
const clamp=()=>{cam.z=Math.min(64,Math.max(1,cam.z));const m=.5/cam.z;cam.x=Math.min(1-m,Math.max(m,cam.x));cam.y=Math.min(1-m,Math.max(m,cam.y));
zl.textContent=cam.z===1?"":(cam.z<9.95?cam.z.toFixed(1):Math.round(cam.z))+"×";cv.style.cursor=cam.z>1?"grab":"zoom-in";};
const redraw=()=>{if(painting||pend)return;pend=true;requestAnimationFrame(()=>{pend=false;show(cur);});};
const zoomAt=(mx,my,f)=>{let g=geo();const qx=cam.x*A+(mx-g.Sw/2)/g.k,qy=cam.y+(my-g.Sh/2)/g.k;cam.z=Math.min(64,Math.max(1,cam.z*f));g=geo();
cam.x=(qx-(mx-g.Sw/2)/g.k)/A;cam.y=qy-(my-g.Sh/2)/g.k;clamp();redraw();};
const panBy=(dx,dy)=>{const g=geo();cam.x-=dx/g.k/A;cam.y-=dy/g.k;clamp();redraw();};
const at=(e)=>{const r=cv.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top];},pts=new Map();
cv.addEventListener("wheel",(e)=>{e.preventDefault();const[x,y]=at(e);zoomAt(x,y,Math.exp(-e.deltaY*(e.deltaMode===1?.05:.002)));},{passive:false});
cv.onpointerdown=(e)=>{cv.setPointerCapture(e.pointerId);pts.set(e.pointerId,at(e));};
cv.onpointermove=(e)=>{const pr=pts.get(e.pointerId);if(!pr)return;const p=at(e);if(pts.size===1)panBy(p[0]-pr[0],p[1]-pr[1]);
else{const o=[...pts].find(([i])=>i!==e.pointerId)[1],d0=Math.hypot(pr[0]-o[0],pr[1]-o[1]),d1=Math.hypot(p[0]-o[0],p[1]-o[1]);panBy((p[0]-pr[0])/2,(p[1]-pr[1])/2);if(d0>0)zoomAt((p[0]+o[0])/2,(p[1]+o[1])/2,d1/d0);}pts.set(e.pointerId,p);};
cv.onpointerup=cv.onpointercancel=(e)=>pts.delete(e.pointerId);
const fit=()=>{cam.z=1;cam.x=cam.y=.5;clamp();redraw();};cv.ondblclick=fit;$("fit").onclick=fit;
addEventListener("resize",redraw);show(N);
</script>`;
}

// ------------------------------------------------------------------ the encoder

let tfReady = null, trainer = null, upload = null, probed = null, tfProbe = null;

// Fill the splat menu with the counts this device can train, each labelled with an estimate from timed
// training steps; the default is the largest that fits in about 30 s (WebGPU) or a minute (TF.js).
// The WebGPU probe runs on the uploaded image, since a step's cost follows its pixel count.
async function sizeMenu() {
  probed = null;
  const opts = [];
  if (!FORCE_TF) {
    $("encstat").textContent = "Timing this GPU…";
    try {
      const img = { rgba: upload.rgba, width: upload.W, height: upload.H };
      const lo = await probeGPU(dev, img, { n: 256 }), hi = await probeGPU(dev, img, { n: 2048 });
      // a step costs a fixed part plus a part proportional to splats
      const b = Math.max(0, (hi - lo) / (2048 - 256)), a = Math.max(0.05, lo - 256 * b), ms = (k) => a + b * k;
      probed = { gpu: true, a, b };
      for (let n = 64; n <= GPU_MAXN; n *= 2) {
        const sizes = checkpoints(n);
        const sec = (sizes.slice(0, -1).reduce((acc, k) => acc + GPU_STEPS.stepsStage * ms(k), 0) + (GPU_STEPS.stepsFinal + GPU_STEPS.stepsQat) * ms(n)) / 1000 + 1 + 0.1 * sizes.length;
        if (sec <= 300 || n <= 256) opts.push([n, sec]);
      }
    } catch (e) { console.log("WebGPU trainer probe failed; using TF.js", e?.stack || e); probed = null; opts.length = 0; }
  }
  if (!probed) {
    $("encstat").textContent = "Loading TensorFlow.js and timing this GPU…";
    const tf = await getTF();
    trainer ??= await import("./trainer.js");
    if (!tfProbe) {
      // a step costs a fixed overhead plus a part proportional to splats x pixels: two probes fit both
      const small = await trainer.probe(tf, { n: 64, nPix: 1024 }), big = await trainer.probe(tf, { n: 256, nPix: 8192 });
      const b = Math.max(0, (big - small) / (256 * 8192 - 64 * 1024)), a0 = Math.max(0, small - b * 64 * 1024);
      tfProbe = { gpu: false, a: a0, b, small, big, backend: tf.getBackend() };
    }
    probed = tfProbe;
    for (let n = 64; n <= TF_PROFILE.maxN; n *= 2) {
      const work = (k) => k * Math.min(8192, Math.max(1024, Math.floor(TF_PROFILE.pixBudget / k)));
      const sizes = checkpoints(n), sc = TF_PROFILE.stepsScale, ms = (k) => probed.a + probed.b * work(k);
      const sec = (sizes.slice(0, -1).reduce((acc, k) => acc + 100 * sc * ms(k), 0) + 450 * sc * ms(n)) / 1000 + 2;
      if (sec <= 300 || n <= 256) opts.push([n, sec]);
    }
  }
  const pick = opts.filter(([, s]) => s <= (probed.gpu ? 30 : 75)).pop() || opts[0];
  const fmt = (s) => s < 90 ? `about ${Math.max(5, Math.round(s / 5) * 5)} s` : `about ${Math.round(s / 60)} min`;
  $("nsel").replaceChildren(...opts.map(([n, s]) => { const o = document.createElement("option"); o.value = n; o.textContent = `${n.toLocaleString()} · ${fmt(s)}`; o.selected = n === pick[0]; return o; }));
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

// Live preview for the WebGPU trainer: its splats so far, drawn on the main canvas by the instanced
// renderer. A preview compiles a shader, so they come at most every 0.7 s.
let lastPreview = 0, pv = null;
function previewSplats(p) {
  if (performance.now() - lastPreview < 700) return;
  pv?.r.destroy();
  pv = { r: createRenderer(dev, fmt, emitInstanced(p)), n: p.n };
  drawPreview();
  lastPreview = performance.now();
}
function drawPreview() {
  const cv = $("cv"), w = Math.max(1, Math.round(cv.clientWidth * devicePixelRatio)), h = Math.max(1, Math.round(cv.clientHeight * devicePixelRatio));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  pv.r.draw(ctx.getCurrentTexture().createView(), w, h, pv.n, cam);
}
function endPreview() { pv?.r.destroy(); pv = null; }

async function runEncode() {
  let n = +$("nsel").value;
  $("go").disabled = $("file").disabled = $("nsel").disabled = true;
  $("prog").hidden = false; $("prog").value = 0;
  const img = { rgba: upload.rgba, width: upload.W, height: upload.H }, extra = window.__encodeOpts || {};   // tests shorten training through __encodeOpts
  const onProgress = (p) => {
    $("prog").value = p.step / p.totalSteps;
    $("encstat").textContent = `${p.phase} · step ${p.step} of ${p.totalSteps} · about ${p.psnrSample.toFixed(1)} dB · ${Math.round(p.seconds)} s`;
  };
  try {
    stopPaint(); setView("shader");
    let r = null, fellBack = "";
    if (probed?.gpu) {
      $("encstat").textContent = "Training on WebGPU…";
      lastPreview = 0;
      try { r = await encodeGPU(dev, img, { n, ...GPU_STEPS, ...extra, onProgress, onPreview: previewSplats }); }
      catch (e) { console.log("WebGPU trainer failed; falling back to TF.js", e?.stack || e); fellBack = ` (the WebGPU trainer failed: ${e?.message || e})`; }
    }
    if (!r) {
      $("encstat").textContent = "Loading TensorFlow.js…" + fellBack;
      const tf = await getTF();
      trainer ??= await import("./trainer.js");
      n = Math.min(n, TF_PROFILE.maxN);
      $("encstat").textContent = `Training on ${tf.getBackend()}…`;
      r = await trainer.encode(tf, img, { n, ...TRAIN_TF, ...extra, onProgress, onPreview: showPreview });
    }
    endPreview();
    $("encstat").textContent = "Measuring the shader and the JPEG/WebP files of the same size…";
    const wgsl = emitWGSL(r);
    const points = [];
    for (const p of r.points) points.push({ n: p.n, bytes: await gzipSize(emitWGSL(r, p.n)), psnr: +p.psnr.toFixed(2) });
    const budget = points[points.length - 1].bytes, cmp = {}, src = { orig: upload.png };
    for (const [k, type] of [["jpeg", "image/jpeg"], ["webp", "image/webp"]]) {
      const b = await bestUnder(upload, type, budget);
      if (b) { src[k] = b.url; cmp[k] = { psnr: b.psnr, bytes: b.bytes, scale: b.scale }; }
    }
    const entry = { id: "your image", own: true, wgsl, n, W: r.W, H: r.H, points, cmp, src, P: Array.from(r.P), bg: Array.from(r.bg), A: r.A };
    window.__encodeResult = { ...entry, seconds: r.seconds, backend: r.backend };
    $("prev").hidden = true;
    const on = r.backend === "webgpu-tiles" ? "WebGPU" : `TensorFlow.js (${r.backend})`;
    $("encstat").textContent = `Done in ${r.seconds < 10 ? r.seconds.toFixed(1) : Math.round(r.seconds)} s on ${on}${fellBack}: ${n.toLocaleString()} splats, ${points[points.length - 1].bytes.toLocaleString()} bytes, ${points[points.length - 1].psnr} dB at ${upload.W}×${upload.H}.`;
    addButton(entry, true);
    load(entry);
    paint();
  } catch (e) {
    $("encstat").textContent = "Encoding failed: " + (e?.message || e);
    console.log("encode error", e?.stack || e);
    $("prev").hidden = true;
    endPreview();
    if (cur) draw(timeFor(ks[$("k").value]));   // repaint what the canvas showed before the previews
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
  $("k").oninput = () => { stopPaint(); setView("shader"); draw(timeFor(ks[$("k").value])); };
  $("paint").onclick = paint;
  addEventListener("resize", redraw);
  initZoom();
  load(IMAGES[0]);
  paint();
}
