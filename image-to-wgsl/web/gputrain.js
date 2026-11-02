// Footprint-limited splat trainer in WebGPU compute shaders.
//
// TF.js trains densely: every sampled pixel evaluates every splat, so a step costs pixels x splats and
// only a few thousand pixels can be sampled. Here a step uses every pixel of the image, and each pixel
// only meets the splats that reach its 16x16 tile, as in the tile rasterisers of 3D Gaussian splatting.
// Painting order is the list order, so per-tile lists need no sorting.
//
// One step, all on the GPU (a fixed sequence of compute dispatches):
//   prep      decode each splat (quantized during the rounding stage), find its tile rectangle (weight >= 1/cut)
//   scanArea  prefix sum of rectangle areas -> each splat's slots in the slot map
//   count     per tile, how many splats reach it
//   scanTiles prefix sum -> each tile's list offset
//   fill      per tile, the ordered list of splats; slotMap[splat slot] = list entry
//   forward   per pixel, paint the tile's list; keep colour C_k and log-transmittance L_k at each prefix k
//   loss      per-prefix squared error -> mse_k, the gradient scale of the mean-log-MSE prefix loss
//   backward  per pixel, walk the list again: dC_k/dw_i = T_ik (col_i - B_i), T_ik = exp(L_k - L_<=i), no
//             division; chain rule to the 10 fields; a workgroup sum per (tile, splat) -> partials
//   gather    per splat, sum its partials through the slot map (no float atomics needed)
//   adam      per field, Adam in the same step units as trainer.js (STEP_FRAC)
// The model matches trainer.js and codec.py: same fields, decode and prefix loss. Differences: every
// pixel every step, weight clamped at 0.9999, tails below 1/cut ignored, background fixed at the
// quadrant means.
import { LAYOUT, checkpoints } from "./trainer.js";

const B = LAYOUT.bits, MAX = B.map((b) => 2 ** b - 1), MBG = 2 ** LAYOUT.bbg - 1;
const SPAN = (LAYOUT.sHi - LAYOUT.sLo) / MAX[2];
const STEP_FRAC = [0.004, 0.004, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03];
const TILE = 16, KMAX = 12;

function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const f = (v) => { let s = String(Math.fround(v)); if (!/[.e]/.test(s)) s += "."; return s; };
const arr = (xs) => `array<f32,${xs.length}>(${xs.map(f).join(",")})`;

function wgsl() {
  // a prefix sum of src[0..n) into dst[0..n], with dst[n] = total, in one workgroup
  const scan = (name, src, dst, nExpr, after = "") => `
@compute @workgroup_size(256) fn ${name}(@builtin(local_invocation_index) li: u32) {
  let n = ${nExpr}; let ch = (n + 255u) / 256u; let lo = min(li * ch, n); let hi = min(lo + ch, n);
  var s = 0u; for (var i = lo; i < hi; i++) { s += ${src}[i]; }
  scanSh[li] = s; workgroupBarrier();
  for (var o = 1u; o < 256u; o *= 2u) { var v = 0u; if (li >= o) { v = scanSh[li - o]; } workgroupBarrier(); scanSh[li] += v; workgroupBarrier(); }
  var run = scanSh[li] - s;
  for (var i = lo; i < hi; i++) { let a = ${src}[i]; ${dst}[i] = run; run += a; }
  if (li == 255u) { ${dst}[n] = scanSh[255]; ${after} }
}`;
  return `
struct Cfg { W: u32, H: u32, TX: u32, TY: u32, N: u32, K: u32, quant: u32, evalMode: u32,
  A: f32, cut: f32, lr: f32, t: f32, cap: u32, wsum: f32, b1: f32, b2: f32,
  ck: array<vec4u, 3>, bg: array<vec4f, 4> }
@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var<storage, read_write> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> spl: array<vec4f>;      // 3 per splat
@group(0) @binding(3) var<storage, read_write> rects: array<vec4i>;
@group(0) @binding(4) var<storage, read_write> area: array<u32>;
@group(0) @binding(5) var<storage, read_write> slotOff: array<u32>;
@group(0) @binding(6) var<storage, read_write> tileCount: array<u32>;
@group(0) @binding(7) var<storage, read_write> tileOff: array<u32>;
@group(0) @binding(8) var<storage, read_write> tileList: array<u32>;
@group(0) @binding(9) var<storage, read_write> slotMap: array<u32>;
@group(0) @binding(10) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> tgt: array<vec4f>;      // rgb, weight
@group(0) @binding(12) var<storage, read_write> pix: array<vec4f>;      // [K][P]: C rgb, L
@group(0) @binding(13) var<storage, read_write> tileLoss: array<f32>;   // [T][K]
@group(0) @binding(14) var<storage, read_write> stats: array<f32>;      // mse[K], scale[K], loss
@group(0) @binding(15) var<storage, read_write> partials: array<f32>;   // [cap][10]
@group(0) @binding(16) var<storage, read_write> grad: array<f32>;
@group(0) @binding(17) var<storage, read_write> adamM: array<f32>;
@group(0) @binding(18) var<storage, read_write> adamV: array<f32>;
const MAXS = ${arr(MAX)};
const SCALE = ${arr(MAX.map((m, i) => m * STEP_FRAC[i]))};
const LN2 = 0.6931471805599453;
fn ck(i: u32) -> u32 { return cfg.ck[i / 4u][i % 4u]; }
fn bgAt(q: vec2f) -> vec3f { let h = q.x / cfg.A; return mix(mix(cfg.bg[0].rgb, cfg.bg[1].rgb, h), mix(cfg.bg[2].rgb, cfg.bg[3].rgb, h), q.y); }

@compute @workgroup_size(64) fn prep(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x; if (j >= cfg.N) { return; }
  var q: array<f32, 10>;
  for (var i = 0u; i < 10u; i++) { var v = clamp(params[j * 10u + i], 0., MAXS[i]); if (cfg.quant == 1u) { v = round(v); } q[i] = v; }
  let cx = q[0] * (cfg.A / MAXS[0]); let cy = q[1] / MAXS[1];
  let sx = exp2(${f(LAYOUT.sLo)} + q[2] * ${f(SPAN)}); let sy = exp2(${f(LAYOUT.sLo)} + q[3] * ${f(SPAN)});
  let th = q[4] * ${f(Math.PI / 2 ** B[4])}; let M = exp2(q[9]); let al = (q[8] + 1.) * ${f(1 / 2 ** B[8])};
  spl[j * 3u] = vec4f(cx, cy, 1. / sx, 1. / sy);
  spl[j * 3u + 1u] = vec4f(cos(th), sin(th), M, al);
  spl[j * 3u + 2u] = vec4f(q[5] / MAXS[5], q[6] / MAXS[6], q[7] / MAXS[7], f32(j));
  // tile rectangle: weight al * exp(-.5 * r^(2M)) >= 1/cut inside radius r (sigma units)
  let ec = 2. * log(max(al * cfg.cut, 1.0001));
  let R = exp(log(ec) / (2. * M)) * max(sx, sy) * f32(cfg.H);
  let cp = vec2f(cx, cy) * f32(cfg.H);
  let lo = vec2i(floor((cp - R) / ${TILE}.)); let hi = vec2i(floor((cp + R) / ${TILE}.));
  let t0 = max(lo, vec2i(0)); let t1 = min(hi, vec2i(i32(cfg.TX) - 1, i32(cfg.TY) - 1));
  if (t1.x < t0.x || t1.y < t0.y) { rects[j] = vec4i(1, 1, 0, 0); area[j] = 0u; return; }
  rects[j] = vec4i(t0, t1); area[j] = u32((t1.x - t0.x + 1) * (t1.y - t0.y + 1));
}
var<workgroup> scanSh: array<u32, 256>;
${scan("scanArea", "area", "slotOff", "cfg.N")}
${scan("scanTiles", "tileCount", "tileOff", "cfg.TX * cfg.TY", "if (scanSh[255] > cfg.cap) { atomicStore(&counters[0], 1u); }")}
fn inRect(r: vec4i, t: vec2i) -> bool { return t.x >= r.x && t.x <= r.z && t.y >= r.y && t.y <= r.w; }
var<workgroup> cnt: atomic<u32>;
@compute @workgroup_size(256) fn count(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&cnt, 0u); } workgroupBarrier();
  let t = vec2i(wg.xy);
  for (var j = li; j < cfg.N; j += 256u) { if (inRect(rects[j], t)) { atomicAdd(&cnt, 1u); } }
  workgroupBarrier();
  if (li == 0u) { tileCount[wg.y * cfg.TX + wg.x] = atomicLoad(&cnt); }
}
var<workgroup> fsh: array<u32, 256>;
var<workgroup> frun: u32;
@compute @workgroup_size(256) fn fill(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let t = vec2i(wg.xy); let ti = wg.y * cfg.TX + wg.x; let base0 = tileOff[ti];
  if (li == 0u) { frun = 0u; } workgroupBarrier();
  for (var base = 0u; base < cfg.N; base += 256u) {
    let j = base + li; var flag = 0u; var r = vec4i(0);
    if (j < cfg.N) { r = rects[j]; if (inRect(r, t)) { flag = 1u; } }
    fsh[li] = flag; workgroupBarrier();
    for (var o = 1u; o < 256u; o *= 2u) { var v = 0u; if (li >= o) { v = fsh[li - o]; } workgroupBarrier(); fsh[li] += v; workgroupBarrier(); }
    if (flag == 1u) {
      let p = base0 + frun + fsh[li] - 1u;
      let s = slotOff[j] + u32((t.y - r.y) * (r.z - r.x + 1) + (t.x - r.x));
      if (p < cfg.cap) { tileList[p] = j; }
      if (s < cfg.cap) { slotMap[s] = p; }
    }
    workgroupBarrier();
    if (li == 255u) { frun += fsh[255]; }
    workgroupBarrier();
  }
}
// splat weight at q: (w, a, b, d2), a and b in sigma units
fn wgt(q: vec2f, v0: vec4f, v1: vec4f) -> vec4f {
  let d = q - v0.xy; let a = (d.x * v1.x + d.y * v1.y) * v0.z; let b = (d.y * v1.x - d.x * v1.y) * v0.w;
  let d2 = a * a + b * b; let e = pow(min(d2, 40.) + 1e-12, v1.z);
  return vec4f(v1.w * exp(-.5 * e), a, b, d2);
}
// workgroup memory per kernel stays under WebGPU's guaranteed 16 KB:
// forward 12 KB (sb) + 1 KB (redf), backward 3 KB (sbb) + 10 KB (red)
var<workgroup> sb: array<vec4f, 768>;      // forward: 256 splats x 3
var<workgroup> redf: array<f32, 256>;      // forward, loss
var<workgroup> sbb: array<vec4f, 192>;     // backward: 64 splats x 3
var<workgroup> red: array<f32, 2560>;      // backward: 256 threads x 10
var<workgroup> anyNZ: u32;
var<workgroup> lrw: vec2u;
fn listRange(ti: u32) -> vec2u { let s = min(tileOff[ti], cfg.cap); let e = min(tileOff[ti + 1u], cfg.cap); return vec2u(s, e - s); }
@compute @workgroup_size(16, 16) fn forward(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let px = wg.x * ${TILE}u + li % ${TILE}u; let py = wg.y * ${TILE}u + li / ${TILE}u;
  let valid = px < cfg.W && py < cfg.H; let P = cfg.W * cfg.H; let pi = py * cfg.W + px;
  let q = vec2f((f32(px) + .5) / f32(cfg.H), (f32(py) + .5) / f32(cfg.H));
  let ti = wg.y * cfg.TX + wg.x;
  if (li == 0u) { lrw = listRange(ti); }
  let lr = workgroupUniformLoad(&lrw);   // one value per tile, uniform for the barriers below
  var c = bgAt(q); var L = 0.; var kp = 0u;
  var snaps: array<vec4f, ${KMAX}>;
  for (var base = 0u; base < lr.y; base += 256u) {
    if (base + li < lr.y) { let j = tileList[lr.x + base + li]; sb[li * 3u] = spl[j * 3u]; sb[li * 3u + 1u] = spl[j * 3u + 1u]; sb[li * 3u + 2u] = spl[j * 3u + 2u]; }
    workgroupBarrier();
    let m = min(256u, lr.y - base);
    for (var i = 0u; i < m; i++) {
      let v2 = sb[i * 3u + 2u]; let j = u32(v2.w);
      while (kp < cfg.K && j >= ck(kp)) { snaps[kp] = vec4f(c, L); kp++; }
      let w = min(wgt(q, sb[i * 3u], sb[i * 3u + 1u]).x, .9999);
      c = mix(c, v2.rgb, w); L += log(1. - w);
    }
    workgroupBarrier();
  }
  while (kp < cfg.K) { snaps[kp] = vec4f(c, L); kp++; }
  let T = tgt[min(pi, P - 1u)];
  for (var k = 0u; k < cfg.K; k++) {
    var e = 0.;
    if (valid) {
      pix[k * P + pi] = snaps[k];
      // evalMode 1: 8-bit squared error (as psnr8); 2: the same, times the focus weight (as psnr8w)
      if (cfg.evalMode >= 1u) { let d = round(clamp(snaps[k].rgb, vec3f(0.), vec3f(1.)) * 255.) - round(T.rgb * 255.); e = dot(d, d) * select(1., T.w, cfg.evalMode == 2u); }
      else { let d = snaps[k].rgb - T.rgb; e = dot(d, d) * T.w; }
    }
    redf[li] = e; workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { redf[li] += redf[li + s]; } workgroupBarrier(); }
    if (li == 0u) { tileLoss[ti * cfg.K + k] = redf[0]; }
    workgroupBarrier();
  }
}
@compute @workgroup_size(256) fn loss(@builtin(local_invocation_index) li: u32) {
  let T = cfg.TX * cfg.TY; var tot = 0.;
  for (var k = 0u; k < cfg.K; k++) {
    var s = 0.; for (var t = li; t < T; t += 256u) { s += tileLoss[t * cfg.K + k]; }
    redf[li] = s; workgroupBarrier();
    for (var o = 128u; o > 0u; o >>= 1u) { if (li < o) { redf[li] += redf[li + o]; } workgroupBarrier(); }
    if (li == 0u) {
      let mse = max(redf[0] / (3. * cfg.wsum), 1e-12);
      stats[k] = mse; stats[cfg.K + k] = (2. / f32(cfg.K)) / (mse * 3. * cfg.wsum); tot += log(mse);
    }
    workgroupBarrier();
  }
  if (li == 0u) { stats[2u * cfg.K] = tot / f32(cfg.K); }
}
@compute @workgroup_size(16, 16) fn backward(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let px = wg.x * ${TILE}u + li % ${TILE}u; let py = wg.y * ${TILE}u + li / ${TILE}u;
  let valid = px < cfg.W && py < cfg.H; let P = cfg.W * cfg.H; let pi = min(py * cfg.W + px, P - 1u);
  let q = vec2f((f32(px) + .5) / f32(cfg.H), (f32(py) + .5) / f32(cfg.H));
  let ti = wg.y * cfg.TX + wg.x;
  if (li == 0u) { lrw = listRange(ti); }
  let lr = workgroupUniformLoad(&lrw);   // one value per tile, uniform for the barriers below
  let T = tgt[pi];
  var gk: array<vec3f, ${KMAX}>; var Lk: array<f32, ${KMAX}>;
  for (var k = 0u; k < cfg.K; k++) {
    let s = pix[k * P + pi]; Lk[k] = s.w;
    gk[k] = select(vec3f(0.), stats[cfg.K + k] * T.w * (s.rgb - T.rgb), valid);
  }
  var Bc = bgAt(q); var Lle = 0.;
  for (var base = 0u; base < lr.y; base += 64u) {
    if (li < 64u && base + li < lr.y) { let j = tileList[lr.x + base + li]; sbb[li * 3u] = spl[j * 3u]; sbb[li * 3u + 1u] = spl[j * 3u + 1u]; sbb[li * 3u + 2u] = spl[j * 3u + 2u]; }
    workgroupBarrier();
    let m = min(64u, lr.y - base);
    for (var i = 0u; i < m; i++) {
      let v0 = sbb[i * 3u]; let v1 = sbb[i * 3u + 1u]; let v2 = sbb[i * 3u + 2u]; let j = u32(v2.w);
      let r = wgt(q, v0, v1); let w = r.x; let wc = min(w, .9999); let Ln = Lle + log(1. - wc);
      var dW = 0.; var dC = vec3f(0.);
      for (var k = 0u; k < cfg.K; k++) { if (ck(k) > j) { let Tk = exp(Lk[k] - Ln); dW += Tk * dot(gk[k], v2.rgb - Bc); dC += Tk * wc * gk[k]; } }
      // explicit zeros: an uninitialised var in this loop kept the previous splat's values here (SwiftShader)
      var out = array<f32, 10>(0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
      if (valid && w > 1e-12) {
        let gw = select(0., dW, w < .9999);
        let a = r.y; let b = r.z; let d2 = r.w; let e0 = min(d2, 40.) + 1e-12; let M = v1.z; let e = pow(e0, M);
        let G = gw * (-.5 * w) * M * e / e0 * select(0., 1., d2 < 40.);
        let ga = G * 2. * a; let gb = G * 2. * b;
        let cs = v1.x; let sn = v1.y; let isx = v0.z; let isy = v0.w;
        out[0] = (ga * (-cs * isx) + gb * (sn * isy)) * (cfg.A / MAXS[0]);
        out[1] = (ga * (-sn * isx) + gb * (-cs * isy)) / MAXS[1];
        out[2] = -ga * a * LN2 * ${f(SPAN)};
        out[3] = -gb * b * LN2 * ${f(SPAN)};
        out[4] = (ga * b * isx / isy - gb * a * isy / isx) * ${f(Math.PI / 2 ** B[4])};
        out[5] = dC.r / MAXS[5]; out[6] = dC.g / MAXS[6]; out[7] = dC.b / MAXS[7];
        out[8] = gw * w / v1.w * ${f(1 / 2 ** B[8])};
        out[9] = gw * (-.5 * w) * e * log(e0) * M * LN2;
        anyNZ = 1u;
      }
      for (var f = 0u; f < 10u; f++) { red[li * 10u + f] = out[f]; }
      if (workgroupUniformLoad(&anyNZ) == 1u) {
        for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { for (var f = 0u; f < 10u; f++) { red[li * 10u + f] += red[(li + s) * 10u + f]; } } workgroupBarrier(); }
      }
      if (li == 0u) { let p = lr.x + base + i; for (var f = 0u; f < 10u; f++) { partials[p * 10u + f] = red[f]; } anyNZ = 0u; }
      workgroupBarrier();
      Bc = mix(Bc, v2.rgb, wc); Lle = Ln;
    }
    workgroupBarrier();
  }
}
@compute @workgroup_size(64) fn gather(@builtin(global_invocation_id) g: vec3u) {
  let j = g.x; if (j >= cfg.N) { return; }
  let s0 = slotOff[j]; let s1 = min(slotOff[j + 1u], cfg.cap);
  var acc = array<f32, 10>(0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
  for (var s = s0; s < s1; s++) { let p = slotMap[s]; if (p < cfg.cap) { for (var f = 0u; f < 10u; f++) { acc[f] += partials[p * 10u + f]; } } }
  for (var f = 0u; f < 10u; f++) { grad[j * 10u + f] = acc[f]; }
}
@compute @workgroup_size(64) fn adam(@builtin(global_invocation_id) g: vec3u) {
  let id = g.x; if (id >= cfg.N * 10u) { return; }
  let fi = id % 10u; let gb = grad[id] * SCALE[fi];
  let m = cfg.b1 * adamM[id] + (1. - cfg.b1) * gb; let v = cfg.b2 * adamV[id] + (1. - cfg.b2) * gb * gb;
  adamM[id] = m; adamV[id] = v;
  let mh = m / (1. - pow(cfg.b1, cfg.t)); let vh = v / (1. - pow(cfg.b2, cfg.t));
  params[id] = clamp(params[id] - cfg.lr * mh / (sqrt(vh) + 1e-8) * SCALE[fi], 0., MAXS[fi]);
}
`;
}

// Bindings each entry point uses (layout "auto" makes one bind group layout per pipeline from these).
const USES = {
  prep: [0, 1, 2, 3, 4], scanArea: [0, 4, 5], count: [0, 3, 6], scanTiles: [0, 6, 7, 10], fill: [0, 3, 5, 7, 8, 9],
  forward: [0, 2, 7, 8, 11, 12, 13], loss: [0, 13, 14], backward: [0, 2, 7, 8, 11, 12, 14, 15], gather: [0, 5, 9, 15, 16],
  adam: [0, 1, 16, 17, 18],
};

export class GpuTrainer {
  constructor(dev, img, { nMax, cut = 1024, focus = null } = {}) {
    this.dev = dev; this.W = img.width; this.H = img.height; this.A = this.W / this.H; this.P = this.W * this.H;
    this.TX = Math.ceil(this.W / TILE); this.TY = Math.ceil(this.H / TILE); this.T = this.TX * this.TY;
    this.nMax = nMax; this.cut = cut; this.K = checkpoints(nMax).length;
    const U = GPUBufferUsage, S = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const buf = (bytes) => dev.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: S });
    this.cfgBuf = dev.createBuffer({ size: 176, usage: U.UNIFORM | U.COPY_DST });
    const N = nMax;
    this.b = { params: buf(N * 40), spl: buf(N * 48), rects: buf(N * 16), area: buf(N * 4), slotOff: buf((N + 1) * 4),
      tileCount: buf(this.T * 4), tileOff: buf((this.T + 1) * 4), counters: buf(16), tgt: buf(this.P * 16),
      pix: buf(this.K * this.P * 16), tileLoss: buf(this.T * this.K * 4), stats: buf((2 * this.K + 1) * 4),
      grad: buf(N * 40), adamM: buf(N * 40), adamV: buf(N * 40) };
    // target and per-pixel weights (focus); weights normalised to mean 1
    const t = new Float32Array(this.P * 4);
    let ws = 0;
    for (let i = 0; i < this.P; i++) { const w = focus ? focus[i] : 1; ws += w; }
    const wn = this.P / ws;
    for (let i = 0; i < this.P; i++) {
      for (let c = 0; c < 3; c++) t[i * 4 + c] = img.rgba[i * 4 + c] / 255;
      t[i * 4 + 3] = (focus ? focus[i] : 1) * wn;
    }
    dev.queue.writeBuffer(this.b.tgt, 0, t);
    this.wsum = this.P;
    this.tgtArr = t;
    this.mod = dev.createShaderModule({ code: wgsl() });
    this.pipes = {};
    for (const name of Object.keys(USES)) this.pipes[name] = dev.createComputePipeline({ layout: "auto", compute: { module: this.mod, entryPoint: name } });
    this.setCap(Math.max(this.T * 48, nMax * 16));
  }
  setCap(cap) {
    const U = GPUBufferUsage, S = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    for (const k of ["tileList", "slotMap", "partials"]) this.b[k]?.destroy();
    this.cap = cap;
    this.b.tileList = this.dev.createBuffer({ size: cap * 4, usage: S });
    this.b.slotMap = this.dev.createBuffer({ size: cap * 4, usage: S });
    this.b.partials = this.dev.createBuffer({ size: cap * 40, usage: S });
    const byIdx = ["cfg", "params", "spl", "rects", "area", "slotOff", "tileCount", "tileOff", "tileList", "slotMap", "counters",
                   "tgt", "pix", "tileLoss", "stats", "partials", "grad", "adamM", "adamV"];
    this.groups = {};
    for (const [name, uses] of Object.entries(USES)) {
      this.groups[name] = this.dev.createBindGroup({ layout: this.pipes[name].getBindGroupLayout(0),
        entries: uses.map((i) => ({ binding: i, resource: { buffer: i === 0 ? this.cfgBuf : this.b[byIdx[i]] } })) });
    }
  }
  writeCfg({ N, quant = 0, evalMode = 0, lr = 0, t = 1, b1 = 0.9, b2 = 0.99, bg }) {
    const ks = checkpoints(N); this.Kcur = ks.length;
    const ab = new ArrayBuffer(176), u = new Uint32Array(ab), fl = new Float32Array(ab);
    u[0] = this.W; u[1] = this.H; u[2] = this.TX; u[3] = this.TY; u[4] = N; u[5] = ks.length; u[6] = quant; u[7] = evalMode;
    fl[8] = this.A; fl[9] = this.cut; fl[10] = lr; fl[11] = t; u[12] = this.cap; fl[13] = this.wsum; fl[14] = b1; fl[15] = b2;
    for (let i = 0; i < KMAX; i++) u[16 + i] = i < ks.length ? ks[i] : 0xffffffff;
    for (let i = 0; i < 4; i++) for (let c = 0; c < 3; c++) fl[28 + i * 4 + c] = bg[i * 3 + c] / MBG;
    this.dev.queue.writeBuffer(this.cfgBuf, 0, ab);
  }
  // one pass of the pipeline; `train` adds backward, gather and adam
  encode(N, train) {
    const enc = this.dev.createCommandEncoder(), p = enc.beginComputePass();
    const run = (name, x, y = 1) => { p.setPipeline(this.pipes[name]); p.setBindGroup(0, this.groups[name]); p.dispatchWorkgroups(x, y); };
    run("prep", Math.ceil(N / 64)); run("scanArea", 1); run("count", this.TX, this.TY); run("scanTiles", 1);
    run("fill", this.TX, this.TY); run("forward", this.TX, this.TY); run("loss", 1);
    if (train) { run("backward", this.TX, this.TY); run("gather", Math.ceil(N / 64)); run("adam", Math.ceil(N * 10 / 64)); }
    p.end();
    return enc;
  }
  async read(name, bytes, offset = 0) {
    const rb = this.dev.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(this.b[name], offset, rb, 0, Math.ceil(bytes / 4) * 4);
    this.dev.queue.submit([enc.finish()]); await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy(); return out;
  }
  destroy() { for (const b of Object.values(this.b)) b.destroy(); this.cfgBuf.destroy(); }
  async overflow() {
    const rb = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(this.b.counters, 0, rb, 0, 16); enc.clearBuffer(this.b.counters, 0, 16);
    this.dev.queue.submit([enc.finish()]); await rb.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap(); rb.destroy();
    if (v[0]) { const tot = new Uint32Array((await this.read("tileOff", (this.T + 1) * 4)).buffer)[this.T]; return tot; }
    return 0;
  }
}

// ---------------------------------------------------------------- the training schedule (as trainer.js encode)

export async function encodeGPU(dev, img, opts = {}) {
  // focus: per-pixel weights (W*H, any scale) trained on and reported as psnrFocus; focusEval: weights only
  // reported, so an unfocused run can be scored on the same map
  const { n = 1024, seed = 0, onProgress = null, onPreview = null, init = "structure", focus = null, focusEval = null,
          stepsStage = 150, stepsFinal = 500, stepsQat = 200, cut = 1024 } = opts;
  const W = img.width, H = img.height, A = W / H, HW = W * H, rand = rng(seed);
  const tr = new GpuTrainer(dev, img, { nMax: n, cut, focus });
  const t0 = performance.now();
  // small target, structure tensor and splat init (same rules as trainer.js)
  const sw = Math.max(16, Math.min(W, 128)), sh = Math.max(8, Math.round(sw / A)), C = sw * sh;
  const small = new Float32Array(C * 3), cnt = new Float32Array(C);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const ci = Math.min(sh - 1, Math.floor(y * sh / H)) * sw + Math.min(sw - 1, Math.floor(x * sw / W)), i = y * W + x;
    for (let c = 0; c < 3; c++) small[ci * 3 + c] += img.rgba[i * 4 + c] / 255; cnt[ci]++;
  }
  for (let i = 0; i < C; i++) for (let c = 0; c < 3; c++) small[i * 3 + c] /= Math.max(1, cnt[i]);
  const fsmall = new Float32Array(C).fill(1);
  if (focus) { fsmall.fill(0); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const ci = Math.min(sh - 1, Math.floor(y * sh / H)) * sw + Math.min(sw - 1, Math.floor(x * sw / W)); fsmall[ci] += focus[y * W + x] / Math.max(1, cnt[ci]); } }
  const J = [new Float32Array(C), new Float32Array(C), new Float32Array(C)], Lum = new Float32Array(C);
  for (let i = 0; i < C; i++) Lum[i] = 0.299 * small[i * 3] + 0.587 * small[i * 3 + 1] + 0.114 * small[i * 3 + 2];
  const at = (x, y) => Lum[Math.min(sh - 1, Math.max(0, y)) * sw + Math.min(sw - 1, Math.max(0, x))];
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) { const gx = (at(x + 1, y) - at(x - 1, y)) / 2, gy = (at(x, y + 1) - at(x, y - 1)) / 2, i = y * sw + x; J[0][i] = gx * gx; J[1][i] = gx * gy; J[2][i] = gy * gy; }
  for (const a of J) for (let pass = 0; pass < 2; pass++) { const b = Float32Array.from(a); for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) { let s = 0, k = 0; for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= sw || yy >= sh) continue; s += b[yy * sw + xx]; k++; } a[y * sw + x] = s / k; } }
  function newSplat(i, size) {
    const ix = i % sw, iy = Math.floor(i / sw), sigma = 0.6 * Math.sqrt(A / size);
    let ls = Math.log2(sigma) + (rand() * 0.6 - 0.3), ratio = 1, th = rand() * Math.PI;
    if (init === "structure") {
      const a = J[0][i], b = J[1][i], c = J[2][i], tr2 = a + c, d = Math.sqrt(Math.max(0, (a - c) ** 2 / 4 + b * b));
      ratio = Math.min(4, Math.max(1, Math.sqrt((tr2 / 2 + d + 1e-6) / (tr2 / 2 - d + 1e-6))));
      th = (((0.5 * Math.atan2(2 * b, a - c) + Math.PI / 2) % Math.PI) + Math.PI) % Math.PI;
    }
    const q = (l) => (l - LAYOUT.sLo) / (LAYOUT.sHi - LAYOUT.sLo) * MAX[2], r2 = Math.log2(ratio) / 2;
    return [(ix + rand()) / sw * MAX[0], (iy + rand()) / sh * MAX[1], q(ls + r2), q(ls - r2) + (init === "structure" ? 0 : rand() * 4 - 2),
      th / (Math.PI / 2 ** B[4]), small[i * 3] * MAX[5], small[i * 3 + 1] * MAX[5], small[i * 3 + 2] * MAX[5], 0.75 * 2 ** B[8] - 1, 0]
      .map((v, f2) => Math.min(Math.max(v, 0), MAX[f2]));
  }
  // background: quadrant means, quantized to its 6 bits (fixed during training)
  const bg = new Int32Array(12);
  for (let qd = 0; qd < 4; qd++) {
    const x0 = qd % 2 ? W >> 1 : 0, x1 = qd % 2 ? W : W >> 1, y0 = qd > 1 ? H >> 1 : 0, y1 = qd > 1 ? H : H >> 1;
    const s = [0, 0, 0]; let k = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { for (let c = 0; c < 3; c++) s[c] += img.rgba[(y * W + x) * 4 + c]; k++; }
    for (let c = 0; c < 3; c++) bg[qd * 3 + c] = Math.round(s[c] / k / 255 * MBG);
  }
  const sizes = checkpoints(n), sched = sizes.map((s, i) => (i === sizes.length - 1 ? stepsFinal : stepsStage));
  const totalSteps = sched.reduce((a, b) => a + b, 0) + stepsQat;
  let P = new Float32Array(0), done = 0, overflows = 0;
  const zero = (N) => { tr.dev.queue.writeBuffer(tr.b.adamM, 0, new Float32Array(N * 10)); tr.dev.queue.writeBuffer(tr.b.adamV, 0, new Float32Array(N * 10)); };
  async function errorCells(N) {           // per-cell squared error of the current render (last prefix)
    if (N === 0) { const e = new Float32Array(C); const bgc = (h, v, c) => ((bg[c] * (1 - h) + bg[3 + c] * h) * (1 - v) + (bg[6 + c] * (1 - h) + bg[9 + c] * h) * v) / MBG;
      for (let i = 0; i < C; i++) { const h = (i % sw + 0.5) / sw, v = (Math.floor(i / sw) + 0.5) / sh; let s = 1e-8; for (let c = 0; c < 3; c++) s += (bgc(h, v, c) - small[i * 3 + c]) ** 2; e[i] = s * fsmall[i]; } return e; }
    tr.writeCfg({ N, bg }); tr.dev.queue.submit([tr.encode(N, false).finish()]);
    const K = checkpoints(N).length, cur = await tr.read("pix", HW * 16, (K - 1) * HW * 16);
    const e = new Float32Array(C), k = new Float32Array(C);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const ci = Math.min(sh - 1, Math.floor(y * sh / H)) * sw + Math.min(sw - 1, Math.floor(x * sw / W)), i = y * W + x;
      let s = 0; for (let c = 0; c < 3; c++) s += (cur[i * 4 + c] - tr.tgtArr[i * 4 + c]) ** 2; e[ci] += s * tr.tgtArr[i * 4 + 3]; k[ci]++;
    }
    for (let i = 0; i < C; i++) e[i] = e[i] / Math.max(1, k[i]) + 1e-8;
    return e;
  }
  async function optimise(N, steps, lr0, lr1, quant, phase) {
    zero(N);
    for (let it = 0; it < steps; it++) {
      const fr = it / Math.max(1, steps - 1), lr = lr1 + 0.5 * (lr0 - lr1) * (1 + Math.cos(Math.PI * fr));
      tr.writeCfg({ N, quant, lr, t: it + 1, bg });
      tr.dev.queue.submit([tr.encode(N, true).finish()]);
      done++;
      if (it % 10 === 9 || it === steps - 1) {
        const ov = await tr.overflow();
        if (ov) { overflows++; tr.setCap(Math.ceil(ov * 1.5)); }
        const st = await tr.read("stats", (2 * tr.Kcur + 1) * 4);
        onProgress?.({ phase, n: N, step: done, totalSteps, psnrSample: -10 * Math.log10(st[tr.Kcur - 1]), seconds: (performance.now() - t0) / 1000 });
      }
    }
    P = (await tr.read("params", N * 40)).slice(0, N * 10);
  }
  for (let si = 0; si < sizes.length; si++) {
    const size = sizes[si], nOld = P.length / 10;
    const e = await errorCells(nOld);
    const keys = []; for (let i = 0; i < C; i++) keys.push([Math.log(rand() + 1e-300) / e[i], i]);
    keys.sort((p, q) => q[0] - p[0]);
    const P2 = new Float32Array(size * 10); P2.set(P);
    for (let k = 0; k < size - nOld; k++) P2.set(newSplat(keys[k][1], size), (nOld + k) * 10);
    P = P2; dev.queue.writeBuffer(tr.b.params, 0, P);
    const final = si === sizes.length - 1;
    await optimise(size, sched[si], 1.0, final ? 0.03 : 0.1, 0, `${size} splats`);
    if (onPreview) onPreview({ P: Int32Array.from(P, Math.round), bg, A, n: size });
  }
  await optimise(n, stepsQat, 0.3, 0.02, 1, "rounding to bits");
  // final: rounded parameters, measured at every prefix with 8-bit rounding (as psnr8)
  const Pq = new Int32Array(n * 10); for (let i = 0; i < n * 10; i++) Pq[i] = Math.min(MAX[i % 10], Math.max(0, Math.round(P[i])));
  dev.queue.writeBuffer(tr.b.params, 0, new Float32Array(Pq));
  tr.wsum = HW; tr.writeCfg({ N: n, quant: 1, evalMode: 1, bg });
  dev.queue.submit([tr.encode(n, false).finish()]);
  const st = await tr.read("stats", (2 * tr.Kcur + 1) * 4);
  // with evalMode on, st[i] is the mean squared difference of 8-bit-rounded colours (as psnr8)
  const db = (m) => 10 * Math.log10(255 * 255 / Math.max(m, 1e-12));
  const points = checkpoints(n).map((k, i) => ({ n: k, psnr: db(st[i]) }));
  const evalW = focusEval ?? focus;
  if (evalW) {   // and weighted by the focus map (mean 1), as trainer.js psnrFocus
    if (evalW !== focus) {
      let ws = 0; for (let i = 0; i < HW; i++) ws += evalW[i];
      const t = Float32Array.from(tr.tgtArr); for (let i = 0; i < HW; i++) t[i * 4 + 3] = evalW[i] * HW / ws;
      dev.queue.writeBuffer(tr.b.tgt, 0, t);
    }
    tr.writeCfg({ N: n, quant: 1, evalMode: 2, bg });
    dev.queue.submit([tr.encode(n, false).finish()]);
    const sf = await tr.read("stats", (2 * tr.Kcur + 1) * 4);
    points.forEach((p, i) => { p.psnrFocus = db(sf[i]); });
  }
  tr.destroy();
  return { P: Pq, bg, A, W, H, n, points, seconds: (performance.now() - t0) / 1000, backend: "webgpu-tiles", steps: totalSteps, overflows };
}

// Milliseconds per training step with n splats on this image, from `steps` timed steps after a warm-up
// (the page's time estimates). Splats are spread at random at the size the init gives n of them.
export async function probeGPU(dev, img, { n = 256, steps = 20 } = {}) {
  const tr = new GpuTrainer(dev, img, { nMax: n }), rand = rng(1), A = img.width / img.height;
  const qs = (Math.log2(0.6 * Math.sqrt(A / n)) - LAYOUT.sLo) / (LAYOUT.sHi - LAYOUT.sLo) * MAX[2], P = new Float32Array(n * 10);
  for (let i = 0; i < n; i++) P.set([rand() * MAX[0], rand() * MAX[1], qs, qs, rand() * MAX[4], rand() * MAX[5], rand() * MAX[6], rand() * MAX[7], 0.75 * 2 ** B[8] - 1, 0], i * 10);
  dev.queue.writeBuffer(tr.b.params, 0, P);
  const bg = new Int32Array(12).fill(MBG >> 1);
  const step = (t) => { tr.writeCfg({ N: n, lr: 0.05, t, bg }); dev.queue.submit([tr.encode(n, true).finish()]); };
  step(1);
  const ov = await tr.overflow(); if (ov) tr.setCap(Math.ceil(ov * 1.5));
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) { step(i + 2); if (i % 10 === 9) await tr.overflow(); }   // the encoder reads back every 10 steps too
  await dev.queue.onSubmittedWorkDone();
  const ms = (performance.now() - t0) / steps;
  tr.destroy();
  return ms;
}

// Gradient check helper: one forward + backward on given params; returns {loss, grad, mse}.
export async function lossAndGrad(dev, img, P, bg, { cut = 1e9, quant = 0, focus = null } = {}) {
  const N = P.length / 10, tr = new GpuTrainer(dev, img, { nMax: N, cut, focus });
  dev.queue.writeBuffer(tr.b.params, 0, Float32Array.from(P));
  tr.writeCfg({ N, quant, lr: 0, t: 1, bg });
  dev.queue.submit([tr.encode(N, true).finish()]);
  const st = await tr.read("stats", (2 * tr.Kcur + 1) * 4), grad = await tr.read("grad", N * 40);
  return { loss: st[2 * tr.Kcur], mse: Array.from(st.slice(0, tr.Kcur)), grad: Array.from(grad.slice(0, N * 10)), overflow: await tr.overflow() };
}
