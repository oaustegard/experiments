// Gallery viewer plus the in-browser encoder ("Your image").
import { IMAGES } from "./shaders.js";
import { emitInstanced, createRenderer } from "./instanced.js";
import { gzipSize, emitWGSL, emitBin, checkpoints, normaliseFocus } from "./trainer.js";
import { encodeGPU, probeGPU } from "./gputrain.js";
import { DeepZoom } from "./deepzoom.js";

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
const MAXSRC = 4096;                 // and kept at most this size, for the detail tiles a zoomed view trains
// Splats per 256-pixel detail tile. On six 2040-px DIV2K photos (M5 Max, GPU bridge, 2026-11-03) 256 splats a tile added
// a median 5.7 dB at full size over the zoomed base list; 1,024 a tile added 4.3 dB more on the two photos tried and
// beat bicubic upscaling of the fitted raster, at 0.5-0.65 s a tile there. Phones get 512: not timed on a phone.
const DZ_TILE = MOBILE ? 512 : 1024;
const TRAIN_TF = { init: "structure", importance: 0.5, pixBudget: TF_PROFILE.pixBudget, stepsScale: TF_PROFILE.stepsScale };
const GPU_STEPS = { stepsStage: 150, stepsFinal: 500, stepsQat: 200 };   // 3x or 8x as many changed 2,048 splats by under 0.15 dB
// TF.js trains on its WebGL backend. Its WebGPU backend computed wrong gradients past 32 splats per
// chunk and crashed the iPhone tab when limited to 32 (GPU bridge, 2026-11-02).
const TF_URLS = ["https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"];
const LABEL = { orig: "Original", avif: "AVIF, same bytes", webp: "WebP, same bytes", jpeg: "JPEG, same bytes" };
let dev, ctx, fmt, ub, cur, pipe, bg, rend = null, anim = 0, view = "shader", ks = [16, 32, 64, 128, 256, 512, 1024], dz = null;

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
    if (dz && dz !== entry.dz) dz.paused = true;
    dz = entry.dz ?? null;
    if (dz) { dz.attach(rend); dz.paused = false; }
    if (entry.instGz == null) gzipSize(entry.inst).then((g) => { entry.instGz = g; if (cur === entry) note(entry); });
  } else {
    if (dz) dz.paused = true;
    dz = null;
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
  // every picture exports: the gallery's lists and the visitor's own image alike (built once per entry)
  if (!entry.dl) {
    const bin = emitBin(entry), url = (part, type) => URL.createObjectURL(new Blob([part], { type }));
    entry.dl = { wgsl: url(entry.wgsl, "text/plain"), html: url(standalone(entry), "text/html"), bin: url(bin, "application/octet-stream"), kb: (bin.byteLength / 1000).toFixed(1) };
  }
  const name = entry.own ? "image" : entry.id;
  $("dlw").href = entry.dl.wgsl; $("dlw").download = `${name}.wgsl`;
  $("dlh").href = entry.dl.html; $("dlh").download = `${name}-shader.html`;
  $("dlb").href = entry.dl.bin; $("dlb").download = `${name}.bin`;
  $("dlb").textContent = `Export the splat list (.bin, ${entry.dl.kb} KB)`;
  cam.z = 1; cam.x = cam.y = 0.5; applyCam();
  $("dzrow").hidden = !dz; dzStatus();
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
  if (focusShown) return;                // the focus map is a flat preview; zoom the shader once it exists
  let g = camGeom();
  const qx = cam.x * g.A + (mx - g.Sw / 2) / g.k, qy = cam.y + (my - g.Sh / 2) / g.k;
  cam.z = Math.min(ZMAX, Math.max(1, cam.z * f)); g = camGeom();
  cam.x = (qx - (mx - g.Sw / 2) / g.k) / g.A; cam.y = qy - (my - g.Sh / 2) / g.k;
  clampCam(); redraw();
}
function panBy(dx, dy) { if (focusShown) return; const g = camGeom(); cam.x -= dx / g.k / g.A; cam.y -= dy / g.k; clampCam(); redraw(); }
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
  const inF = (q) => q.pf != null ? ` (${q.pf} dB in focus)` : "";
  $("cmpnote").textContent = (entry.focus ? `Trained with focus on ${entry.focus}. ` : "") +
    `Full shader: ${full.bytes.toLocaleString()} bytes, ${full.psnr} dB${inF(full)}.` + Object.entries(entry.cmp).map(([k, c]) =>
    ` ${k.toUpperCase()} at or under that size: ${c.bytes.toLocaleString()} bytes, ${c.psnr} dB${inF(c)}${c.scale > 1 ? ` (encoded at 1/${c.scale} size)` : ""}.`).join("") +
    (entry.instGz ? ` Drawn by an instanced version of the same data (${entry.instGz.toLocaleString()} bytes gzipped), one quad per splat.` : "");
}

const nAt = (t) => t <= 0 ? cur.n : Math.min(cur.n, Math.floor(cur.n * Math.min(1, t * t / (T * T))));

function timeFor(k) { return k >= cur.n ? 0 : T * Math.sqrt((k + 0.5) / cur.n); }

function draw(t) {
  const cv = $("cv");
  const w = Math.max(1, Math.round(cv.clientWidth * devicePixelRatio)), h = Math.max(1, Math.round(cv.clientHeight * devicePixelRatio));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  if (rend) {
    const n = nAt(t), nd = dz && $("dzon").checked && n === cur.n ? dz.countFor(viewLevel()) : 0;
    rend.draw(ctx.getCurrentTexture().createView(), w, h, n, cam, nd);
    readout(t, nd);
    if (n === cur.n) wantDetail();
    return;
  }
  dev.queue.writeBuffer(ub, 0, new Float32Array([w, h, t, 0]));
  const enc = dev.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [1, 1, 1, 1] }] });
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  dev.queue.submit([enc.finish()]);
  readout(t);
}

function readout(t, nd = 0) {
  const n = nAt(t);
  const p = cur.points.filter((q) => q.n <= Math.max(n, 16)).pop();
  $("tag").textContent = view === "shader" && !focusShown ? (nd ? "shader + detail" : "shader") : $("tag").textContent;
  $("readout").innerHTML = `<b>${n}</b> splats · <b>${p.bytes.toLocaleString()}</b> bytes · <b>${p.psnr}</b> dB` +
    (p.pf != null ? ` · <b>${p.pf}</b> dB in focus` : "") + (n === p.n ? "" : ` <span class="note">(nearest measured: ${p.n})</span>`) +
    (nd ? ` · <b>+${nd.toLocaleString()}</b> detail splats` : "");
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
  b.onclick = () => { stopPaint(); hideFocus(); $("enc").hidden = !own; load(e); };
}

// ------------------------------------------------------------------ deep zoom
// For your own image the page keeps the source (up to MAXSRC on its longer side). When the screen shows more
// pixels than the list was fitted at, detail tiles for the visible part train in the background (deepzoom.js) and
// draw over the list, so zooming in adds detail from the source rather than only smoother edges.
function viewLevel() {          // device pixels per pixel of the fitted image, as a pyramid level
  const { A, Sw, Sh } = camGeom();
  return dz.levelFor(Math.min(Sw / A, Sh) * cam.z * devicePixelRatio / cur.H);
}
let wantTimer = 0;
function wantDetail() {         // the tiles this view needs, once the view has settled for a moment
  if (!dz || !$("dzon").checked) return;
  clearTimeout(wantTimer);
  wantTimer = setTimeout(() => {
    const L = viewLevel();
    if (L < 1) { dz.want([]); dzStatus(); return; }
    const { A, k } = camGeom(), hw = camGeom().Sw / 2 / k / A, hh = camGeom().Sh / 2 / k;
    dz.want(dz.visible(L, Math.max(0, cam.x - hw), Math.max(0, cam.y - hh), Math.min(1, cam.x + hw), Math.min(1, cam.y + hh)));
    dzStatus();
  }, 150);
}
function dzStatus() {
  if (!dz) return;
  const st = dz.stats(), L = viewLevel(), top = dz.levels[dz.levels.length - 1];
  const trained = st.reduce((a, l) => a + l.trained, 0), splats = st.reduce((a, l) => a + l.splats, 0);
  let msg = `Source kept at ${top.W.toLocaleString()}×${top.H.toLocaleString()}: ${dz.levels.length} detail level${dz.levels.length > 1 ? "s" : ""} above the fitted ${cur.W}×${cur.H}.`;
  if (dz.error) msg += ` Detail training failed: ${dz.error.message || dz.error}.`;
  else if (dz.pending()) msg += ` Training detail for this view: ${dz.pending()} tile${dz.pending() > 1 ? "s" : ""} to go.`;
  else if (L > 0) msg += ` This view shows level ${L} (${dz.levels[L - 1].W.toLocaleString()}×${dz.levels[L - 1].H.toLocaleString()}).`;
  else msg += " Zoom in to add detail.";
  if (trained) msg += ` ${trained} tile${trained > 1 ? "s" : ""}, ${splats.toLocaleString()} detail splats so far.`;
  $("dzstat").textContent = msg;
}
function makeDeepZoom(entry, up) {
  const src = up.src, Ws = src.width, Hs = src.height;
  const crop = async (lv, x0, y0, x1, y1) => {       // a rectangle of a level, resampled from the kept source
    const w = x1 - x0, h = y1 - y0, [c, x] = canvas2d(w, h);
    x.imageSmoothingQuality = "high";
    x.drawImage(src, x0 * Ws / lv.W, y0 * Hs / lv.H, w * Ws / lv.W, h * Hs / lv.H, 0, 0, w, h);
    return x.getImageData(0, 0, w, h).data;
  };
  const d = new DeepZoom(dev, { base: { P: Int32Array.from(entry.P), bg: entry.bg, A: entry.A, n: entry.n }, W0: up.W, H0: up.H, Ws, Hs, crop, nTile: DZ_TILE,
    ...(window.__dzOpts || {}) });   // tests shorten tile training through __dzOpts
  if (!d.levels.length) return null;
  d.onTile = (r) => { if (r?.error) console.log("deep zoom error", r.error?.stack || r.error); if (cur?.dz === d) { dzStatus(); redraw(); } };
  return d;
}

// ------------------------------------------------------------------ standalone page for downloads

// Opens on the finished image. "Paint it in" replays the painting; the slider steps through the measured
// prefixes (16, 32, ... splats) with their bytes and PSNR from the encoder. Scroll or pinch zooms and drag
// pans, redrawing the splats at each scale. Shader only: no source image or comparisons.
function standalone(entry) {
  const pts = entry.points.map((p) => [p.n, p.bytes, p.psnr, p.pf ?? null]);
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
ro.innerHTML="<b>"+n.toLocaleString()+"</b> splats · <b>"+p[1].toLocaleString()+"</b> bytes · <b>"+p[2]+"</b> dB"+(p[3]!=null?" · <b>"+p[3]+"</b> dB in focus":"")+(n===p[0]?"":' <span class="near">(nearest measured: '+p[0]+")</span>");};
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

function psnrU8w(a, b, w) {  // the same, each pixel weighted by w (mean 1), as trainer.js psnr8w
  let s = 0, ws = 0;
  for (let i = 0, j = 0; i < a.length; i += 4, j++) { let d2 = 0; for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; d2 += d * d; } s += w[j] * d2; ws += w[j]; }
  return 10 * Math.log10(255 * 255 / Math.max(s / (3 * ws), 1e-12));
}

function psnrU8(a, b) {      // RGBA byte arrays, alpha ignored
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i += 4) for (let c = 0; c < 3; c++) { const d = a[i + c] - b[i + c]; s += d * d; n++; }
  return 10 * Math.log10(255 * 255 / Math.max(s / n, 1e-12));
}

function canvas2d(w, h) { const c = document.createElement("canvas"); c.width = w; c.height = h; return [c, c.getContext("2d")]; }
const toBlob = (c, type, q) => new Promise((r) => c.toBlob(r, type, q));

async function readUpload(file) {
  let bmp = await createImageBitmap(file);
  const s = Math.min(1, MAXDIM / Math.max(bmp.width, bmp.height));
  const W = Math.max(16, Math.round(bmp.width * s)), H = Math.max(16, Math.round(bmp.height * s));
  const [c, x] = canvas2d(W, H);
  x.imageSmoothingQuality = "high";
  x.drawImage(bmp, 0, 0, W, H);
  const rgba = x.getImageData(0, 0, W, H).data;
  const png = URL.createObjectURL(await toBlob(c, "image/png"));
  // the source, for deep zoom: at most MAXSRC on its longer side
  const t = Math.min(1, MAXSRC / Math.max(bmp.width, bmp.height));
  if (t < 1) { const big = bmp; bmp = await createImageBitmap(big, { resizeWidth: Math.round(big.width * t), resizeHeight: Math.round(big.height * t), resizeQuality: "high" }); big.close(); }
  return { W, H, rgba, canvas: c, png, name: file.name, img: { rgba, width: W, height: H }, src: bmp };
}

// Best encode of `type` at or under `budget` bytes, at full, half or quarter size, decoded back to W x H.
async function bestUnder(up, type, budget, focus = null) {
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
      const px = y.getImageData(0, 0, up.W, up.H).data, p = psnrU8(px, up.rgba);
      if (!best || p > best.psnr) best = { psnr: +p.toFixed(2), bytes: blob.size, scale: sc, q: +q.toFixed(2), url: URL.createObjectURL(blob),
        pf: focus ? +psnrU8w(px, up.rgba, focus).toFixed(2) : null };
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

// ------------------------------------------------------------------ focus (foveation)
// A focus map is a weight per pixel (focus.js): a spot, faces, or a band of depth around a tapped spot. Until
// the next encode the stage shows it over the image, bright where error counts most; tapping the stage moves
// the spot or picks the depth. The trainers take it floored at 1/20 and normalised (trainer.js normaliseFocus).
const FOCUS_LABEL = { point: "a spot", face: "faces", depth: "a distance" };
const focusState = { mode: "", point: { x: 0.5, y: 0.4 }, depthAt: null, map: null };
let focusMod = null, focusToken = 0, focusShown = false;
async function updateFocus() {
  const mode = $("fsel").value, size = +$("fsize").value, token = ++focusToken;
  focusState.mode = mode;
  $("fsize").hidden = $("fsizel").hidden = !(mode === "point" || mode === "depth");
  if (!mode) { focusState.map = null; $("fstat").textContent = ""; hideFocus(); return; }
  if (!upload) { $("fstat").textContent = "Choose an image first."; return; }
  const img = upload.img;
  try {
    focusMod ??= await import("./focus.js");
    let m = null, msg = "";
    if (mode === "point") { m = await focusMod.focusMap("point", img, { point: focusState.point, radius: size }); msg = "Tap the picture to move the spot."; }
    else if (mode === "face") {
      $("fstat").textContent = "Looking for faces…";
      const faces = await focusMod.detectFaces(img);
      m = faces.length ? await focusMod.focusMap("face", img, { faces }) : null;
      msg = faces.length ? `Found ${faces.length} face${faces.length > 1 ? "s" : ""}.` : "No face found: the whole image will be trained evenly.";
    } else {
      $("fstat").textContent = "Estimating depth…";
      const d = await focusMod.depthMap(img);
      if (focusState.depthAt == null) focusState.depthAt = d[Math.floor(img.height * 0.45) * img.width + Math.floor(img.width / 2)];
      m = await focusMod.focusMap("depth", img, { depth: focusState.depthAt, width: size * 0.75 });
      msg = "Tap the picture to pick the distance in focus.";
    }
    if (token !== focusToken) return;     // a newer request superseded this one
    focusState.map = m; $("fstat").textContent = msg;
    if (m) showFocus(); else hideFocus();
  } catch (e) {
    if (token !== focusToken) return;
    focusState.map = null; hideFocus();
    $("fstat").textContent = "Focus failed: " + (e?.message || e) + ". The whole image will be trained evenly.";
    console.log("focus error", e?.stack || e);
  }
}
function showFocus() {           // the image, darkened where the focus weight is low
  const m = focusState.map, { W, H, rgba } = upload, c = $("prev");
  if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
  let mx = 0; for (let i = 0; i < m.length; i++) if (m[i] > mx) mx = m[i];
  const x = c.getContext("2d"), im = x.createImageData(W, H);
  for (let i = 0; i < W * H; i++) { const k = 0.2 + 0.8 * Math.max(m[i] / mx, 0.05); for (let ch = 0; ch < 3; ch++) im.data[i * 4 + ch] = rgba[i * 4 + ch] * k; im.data[i * 4 + 3] = 255; }
  x.putImageData(im, 0, 0);
  resetCam(); setView("shader"); c.hidden = false; $("tag").textContent = "focus map"; focusShown = true;
}
function hideFocus() { if (focusShown) { focusShown = false; $("prev").hidden = true; setView(view); } }
async function tapFocus(e) {     // stage click while a tappable focus map is shown
  if (!upload || !focusShown || !(focusState.mode === "point" || focusState.mode === "depth")) return;
  const r = $("stage").getBoundingClientRect(), k = Math.min(r.width / upload.W, r.height / upload.H);
  const x = (e.clientX - r.left - (r.width - upload.W * k) / 2) / (upload.W * k), y = (e.clientY - r.top - (r.height - upload.H * k) / 2) / (upload.H * k);
  if (x < 0 || y < 0 || x > 1 || y > 1) return;
  if (focusState.mode === "point") focusState.point = { x, y };
  else { const d = await focusMod.depthMap(upload.img); focusState.depthAt = d[Math.min(upload.H - 1, Math.floor(y * upload.H)) * upload.W + Math.min(upload.W - 1, Math.floor(x * upload.W))]; }
  updateFocus();
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
  if (dz) dz.paused = true;                 // the GPU is the encoder's until it finishes
  try {
    stopPaint(); setView("shader"); hideFocus();
    const focus = focusState.map ? normaliseFocus(focusState.map, upload.W * upload.H) : null, focusLabel = focus ? FOCUS_LABEL[focusState.mode] : null;
    let r = null, fellBack = "";
    if (probed?.gpu) {
      $("encstat").textContent = "Training on WebGPU…";
      lastPreview = 0;
      try { r = await encodeGPU(dev, img, { n, ...GPU_STEPS, ...extra, focus, onProgress, onPreview: previewSplats }); }
      catch (e) { console.log("WebGPU trainer failed; falling back to TF.js", e?.stack || e); fellBack = ` (the WebGPU trainer failed: ${e?.message || e})`; }
    }
    if (!r) {
      $("encstat").textContent = "Loading TensorFlow.js…" + fellBack;
      const tf = await getTF();
      trainer ??= await import("./trainer.js");
      n = Math.min(n, TF_PROFILE.maxN);
      $("encstat").textContent = `Training on ${tf.getBackend()}…`;
      r = await trainer.encode(tf, img, { n, ...TRAIN_TF, ...extra, focus, onProgress, onPreview: showPreview });
    }
    endPreview();
    $("encstat").textContent = "Measuring the shader and the JPEG/WebP files of the same size…";
    const wgsl = emitWGSL(r);
    const points = [];
    for (const p of r.points) points.push({ n: p.n, bytes: await gzipSize(emitWGSL(r, p.n)), psnr: +p.psnr.toFixed(2), pf: p.psnrFocus != null ? +p.psnrFocus.toFixed(2) : null });
    const budget = points[points.length - 1].bytes, cmp = {}, src = { orig: upload.png };
    for (const [k, type] of [["jpeg", "image/jpeg"], ["webp", "image/webp"]]) {
      const b = await bestUnder(upload, type, budget, focus);
      if (b) { src[k] = b.url; cmp[k] = { psnr: b.psnr, bytes: b.bytes, scale: b.scale, pf: b.pf }; }
    }
    const entry = { id: "your image", own: true, wgsl, n, W: r.W, H: r.H, points, cmp, src, P: Array.from(r.P), bg: Array.from(r.bg), A: r.A, focus: focusLabel };
    const old = [...$("picker").children].find((x) => x.dataset.id === entry.id);
    if (dz && old) dz.paused = true;                  // the previous upload's detail stops training
    entry.dz = makeDeepZoom(entry, upload);
    window.__encodeResult = { ...entry, seconds: r.seconds, backend: r.backend };
    $("prev").hidden = true;
    const on = r.backend === "webgpu-tiles" ? "WebGPU" : `TensorFlow.js (${r.backend})`;
    $("encstat").textContent = `Done in ${r.seconds < 10 ? r.seconds.toFixed(1) : Math.round(r.seconds)} s on ${on}${fellBack}: ${n.toLocaleString()} splats, ${points[points.length - 1].bytes.toLocaleString()} bytes, ${points[points.length - 1].psnr} dB${points[points.length - 1].pf != null ? ` (${points[points.length - 1].pf} dB in focus)` : ""} at ${upload.W}×${upload.H}.`;
    addButton(entry, true);
    load(entry);
    paint();
  } catch (e) {
    $("encstat").textContent = "Encoding failed: " + (e?.message || e);
    console.log("encode error", e?.stack || e);
    if (dz) { dz.paused = false; dz.pump(); }
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
  own.onclick = () => { $("enc").hidden = false; for (const b of $("picker").children) b.setAttribute("aria-pressed", String(b === own)); if (focusState.map) showFocus(); $("file").focus(); };
  $("picker").append(own);
  for (const e of IMAGES) addButton(e);
  $("file").onchange = async () => {
    const f = $("file").files[0];
    if (!f) return;
    try {
      upload = await readUpload(f);
      focusState.depthAt = null; focusState.point = { x: 0.5, y: 0.4 };
      updateFocus();                       // runs alongside the timing below
      await sizeMenu();
      $("go").disabled = false;
      $("encstat").textContent = `${f.name}: ${upload.W}×${upload.H} after scaling. Press Encode.`;
    } catch (e) { $("encstat").textContent = "Could not read that image: " + (e?.message || e); }
  };
  $("go").onclick = runEncode;
  $("fsel").onchange = updateFocus;
  let fsizeTimer = 0;
  $("fsize").oninput = () => { clearTimeout(fsizeTimer); fsizeTimer = setTimeout(updateFocus, 120); };
  $("stage").addEventListener("click", tapFocus);
  $("k").oninput = () => { stopPaint(); setView("shader"); draw(timeFor(ks[$("k").value])); };
  $("paint").onclick = paint;
  $("dzon").onchange = () => { if (dz) { dz.paused = !$("dzon").checked; if (!dz.paused) dz.pump(); } redraw(); };
  addEventListener("resize", redraw);
  initZoom();
  load(IMAGES[0]);
  paint();
}
