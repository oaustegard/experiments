// Line crops on the GPU, written straight into a recognizer batch [B, H, Wm] as ink (0 = paper).
// Each line follows indra.cropLine: the box padded and cut from the page (paper outside it),
// then resized as imgproc.resize does it on the CPU — INTER_AREA decimation in float32 sums,
// exact-integer area blocks, or OpenCV's 11-bit fixed-point linear interpolation — so the
// batch holds the same values the CPU path builds.

import { areaTab, coefs } from "./imgproc.js";
import { registerKernel } from "./gpu.js";

const PAD_X = 0.3, PAD_Y = 0.2, WIDTH_MULTIPLE = 8;

const pyRound = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };

// the crop rectangle and output width of rec_preproc.crop_line
export function cropPlan(box, height, window) {
  const padY = (window * (1 + 2 * PAD_Y) - 1) / 2;
  const H = height * window;
  const [x0, y0, x1, y1] = box;
  const h = Math.max(1.0, y1 - y0);
  const ix0 = Math.floor(x0 - PAD_X * h), iy0 = Math.floor(y0 - padY * h);
  const cw = Math.max(1, Math.ceil(x1 + PAD_X * h) - ix0), ch = Math.max(1, Math.ceil(y1 + padY * h) - iy0);
  const outW = Math.max(WIDTH_MULTIPLE, Math.min(4096, pyRound((cw * H) / ch)));
  return { ix0, iy0, cw, ch, outW, H };
}

const CROP = /* wgsl */ `
struct P { B: u32, H: u32, Wm: u32, W: u32, PH: u32, nx: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> page: array<u32>;
@group(0) @binding(2) var<storage, read> lines: array<u32>;   // 12 words per line
@group(0) @binding(3) var<storage, read> ti: array<u32>;      // integer tables
@group(0) @binding(4) var<storage, read> tf: array<f32>;      // float tables
@group(0) @binding(5) var<storage, read> lut: array<f32>;     // ink of a grey level: [paper-white, dark]
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
fn px(L: u32, cx: i32, cy: i32) -> u32 {
  let gx = bitcast<i32>(lines[L + 1u]) + cx; let gy = bitcast<i32>(lines[L + 2u]) + cy;
  if (gx < 0 || gy < 0 || gx >= i32(p.W) || gy >= i32(p.PH)) { return select(255u, 0u, lines[L + 6u] == 1u); }
  let i = u32(gy) * p.W + u32(gx);
  return (page[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.B * p.H * p.Wm) { return; }
  let x = idx % p.Wm; let yy = (idx / p.Wm) % p.H; let b = idx / (p.Wm * p.H);
  let L = b * 12u;
  let outW = lines[L + 5u];
  if (x >= outW) { y[idx] = 0.0; return; }
  let mode = lines[L];
  let cw = lines[L + 3u]; let ch = lines[L + 4u];
  var v = 0u;
  if (mode == 0u) {
    v = px(L, i32(x), i32(yy));
  } else if (mode == 1u) {
    // area decimation: per row j (weight beta), the row's sum over its columns (weights alpha)
    let xo = lines[L + 7u] + 2u * x; let yo = lines[L + 8u] + 2u * yy;
    let x0 = ti[xo]; let xn = ti[xo + 1u]; let y0 = ti[yo]; let yn = ti[yo + 1u];
    var sum = 0.0;
    for (var j = 0u; j < yn; j++) {
      let sy = i32(ti[y0 + j]);
      var buf = 0.0;
      for (var i = 0u; i < xn; i++) { buf = buf + f32(px(L, i32(ti[x0 + i]), sy)) * tf[x0 + i]; }
      if (j == 0u) { sum = tf[y0] * buf; } else { sum = sum + tf[y0 + j] * buf; }
    }
    v = u32(clamp(round(sum), 0.0, 255.0));
  } else if (mode == 2u) {
    // exact k x k blocks
    let kx = lines[L + 9u]; let ky = lines[L + 10u];
    var s = 0u;
    for (var j = 0u; j < ky; j++) { for (var i = 0u; i < kx; i++) { s += px(L, i32(x * kx + i), i32(yy * ky + j)); } }
    if (kx == 2u && ky == 2u) { v = (s + 2u) >> 2u; }
    else { v = u32(clamp(round(f32(s) * bitcast<f32>(lines[L + 11u])), 0.0, 255.0)); }
  } else {
    // OpenCV fixed-point bilinear: tables (index, a0, a1) per column and row
    let xo = lines[L + 7u] + 3u * x; let yo = lines[L + 8u] + 3u * yy;
    let xi = ti[xo]; let a0 = i32(ti[xo + 1u]); let a1 = i32(ti[xo + 2u]);
    let sy = ti[yo]; let b0 = i32(ti[yo + 1u]); let b1 = i32(ti[yo + 2u]);
    let xj = min(xi + 1u, cw - 1u); let sy1 = min(sy + 1u, ch - 1u);
    let r0 = i32(px(L, i32(xi), i32(sy))) * a0 + i32(px(L, i32(xj), i32(sy))) * a1;
    let r1 = i32(px(L, i32(xi), i32(sy1))) * a0 + i32(px(L, i32(xj), i32(sy1))) * a1;
    let t = (((r0 >> 4u) * b0) >> 16u) + (((r1 >> 4u) * b1) >> 16u);
    v = u32(clamp((t + 2) >> 2u, 0, 255));
  }
  y[idx] = lut[v + select(0u, 256u, lines[L + 6u] == 1u)];
}`;
registerKernel(CROP, "crop");

// rows [r0, r0 + h) of x [B, Hx, W] -> [B, h, W]
const ROWS = /* wgsl */ `
struct P { B: u32, Hx: u32, h: u32, W: u32, r0: u32, nx: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.B * p.h * p.W) { return; }
  let xx = idx % p.W; let r = (idx / p.W) % p.h; let b = idx / (p.W * p.h);
  y[idx] = x[(b * p.Hx + r + p.r0) * p.W + xx];
}`;
registerKernel(ROWS, "rows");

const LUT = (() => {
  const t = new Float32Array(512);
  for (let v = 0; v < 256; v++) { const f = Math.fround(v / 255); t[v] = Math.fround(1 - f); t[256 + v] = f; }
  return t;
})();

export class GpuCrops {
  constructor(gpu) { this.gpu = gpu; this.lut = gpu.upload(LUT); }

  setPage(gray) {
    this.page?.destroy();
    const n = Math.ceil(gray.data.length / 4) * 4;
    const u8 = new Uint8Array(n);
    u8.set(gray.data);
    this.page = this.gpu.upload(new Uint32Array(u8.buffer));
    this.W = gray.w;
    this.PH = gray.h;
  }

  // Record into the open pass: the batch [B, H, Wm] of `items` [{plan, dark}]; returns the buffer.
  record(items, H, Wm) {
    const gpu = this.gpu;
    const lines = new Uint32Array(items.length * 12);
    const ti = [], tf = [];
    const f32bits = new Float32Array(1), u32v = new Uint32Array(f32bits.buffer);
    items.forEach(({ plan, dark }, b) => {
      const { ix0, iy0, cw, ch, outW } = plan;
      const L = b * 12;
      lines[L + 1] = ix0 >>> 0; lines[L + 2] = iy0 >>> 0; lines[L + 3] = cw; lines[L + 4] = ch; lines[L + 5] = outW; lines[L + 6] = dark ? 1 : 0;
      const sx = cw / outW, sy = ch / H;
      if (outW === cw && H === ch) { lines[L] = 0; return; }
      if (ch > H && sx >= 1 && sy >= 1) {
        const ix = Math.round(sx), iy = Math.round(sy);
        if (Math.abs(sx - ix) < 1e-12 && Math.abs(sy - iy) < 1e-12) {
          lines[L] = 2; lines[L + 9] = ix; lines[L + 10] = iy;
          f32bits[0] = Math.fround(1 / (ix * iy)); lines[L + 11] = u32v[0];
          return;
        }
        lines[L] = 1;
        for (const [axis, ss, ds, sc] of [[7, cw, outW, sx], [8, ch, H, sy]]) {
          const tab = areaTab(ss, ds, sc);
          // per destination index: [start, count] at the head, then (source, weight) entries
          const head = ti.length;
          lines[L + axis] = head;
          for (let d = 0; d < ds; d++) { ti.push(0, 0); tf.push(0, 0); }
          for (const [d, s, a] of tab) {
            if (ti[head + 2 * d + 1] === 0) ti[head + 2 * d] = ti.length;
            ti[head + 2 * d + 1]++;
            ti.push(s);
            tf.push(a);
          }
        }
        return;
      }
      // linear (INTER_LINEAR, or INTER_AREA when it upscales: OpenCV's area coefficients)
      lines[L] = 3;
      const area = ch > H;
      for (const [axis, ss, ds] of [[7, cw, outW], [8, ch, H]]) {
        const c = coefs(ss, ds, area);
        lines[L + axis] = ti.length;
        for (let d = 0; d < ds; d++) { ti.push(c.idx[d], c.a0[d], c.a1[d]); tf.push(0, 0, 0); }
      }
    });
    const bl = gpu.upload(lines), bti = gpu.upload(Uint32Array.from(ti.length ? ti : [0])), btf = gpu.upload(Float32Array.from(tf.length ? tf : [0]));
    const out = gpu.buffer(4 * items.length * H * Wm);
    const total = items.length * H * Wm, grid = gpu.grid(total);
    gpu.dispatch(CROP, [this.page, bl, bti, btf, this.lut, out], [items.length, H, Wm, this.W, this.PH, grid[0], 0, 0], grid);
    this.tmp = (this.tmp ?? []).concat([bl, bti, btf]);
    return out;
  }

  rows(x, B, Hx, h, W, r0) {
    const gpu = this.gpu;
    const out = gpu.buffer(4 * B * h * W);
    const total = B * h * W, grid = gpu.grid(total);
    gpu.dispatch(ROWS, [x, out], [B, Hx, h, W, r0, grid[0], 0, 0], grid);
    return out;
  }

  // after the pass that used them has been submitted
  release() { for (const b of this.tmp ?? []) b.destroy(); this.tmp = []; }
}
