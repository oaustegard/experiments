// The deskew on the GPU: the skew search's rotated row sums for many angles in one dispatch, and
// the full-page rotation. Both sample as imgproc.warpAffine does (OpenCV 5's float32 bilinear
// warp through the inverse matrix, round half to even), on 8-bit images packed 4 pixels per u32.

import { areaTab, img, rotationMatrix } from "./imgproc.js";
import { registerKernel } from "./gpu.js";

const SAMPLE = /* wgsl */ `
fn px(x: i32, y: i32) -> f32 {
  var xx = x; var yy = y;
  if (x < 0 || x >= i32(p.W) || y < 0 || y >= i32(p.H)) {
    if (p.rep == 0u) { return f32(p.border); }
    xx = clamp(x, 0, i32(p.W) - 1); yy = clamp(y, 0, i32(p.H) - 1);
  }
  let i = u32(yy) * p.W + u32(xx);
  return f32((src[i >> 2u] >> ((i & 3u) * 8u)) & 255u);
}
fn sample(m: array<f32, 6>, x: u32, y: u32) -> u32 {
  let by0 = m[1] * f32(y); let by1 = m[4] * f32(y);
  let sx = (m[0] * f32(x) + by0) + m[2];
  let sy = (m[3] * f32(x) + by1) + m[5];
  let fx0 = floor(sx); let fy0 = floor(sy);
  let x0 = i32(fx0); let y0 = i32(fy0);
  let fx = sx - fx0; let fy = sy - fy0;
  let p00 = px(x0, y0); let p01 = px(x0 + 1, y0); let p10 = px(x0, y0 + 1); let p11 = px(x0 + 1, y0 + 1);
  let top = p00 + fx * (p01 - p00);
  let bot = p10 + fx * (p11 - p10);
  return u32(clamp(round(top + fy * (bot - top)), 0.0, 255.0));
}`;

// Row sums of the image rotated by each angle: one workgroup per (row, angle).
const ROWSUM = /* wgsl */ `
struct P { W: u32, H: u32, A: u32, rep: u32, border: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> mats: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
var<workgroup> red: array<u32, 64>;
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let y = wg.x; let a = wg.y;
  var m: array<f32, 6>;
  for (var k = 0u; k < 6u; k++) { m[k] = mats[a * 6u + k]; }
  var s = 0u;
  for (var x = l; x < p.W; x += 64u) { s += sample(m, x, y); }
  red[l] = s;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (l < k) { red[l] += red[l + k]; }
    workgroupBarrier();
  }
  if (l == 0u) { out[a * p.H + y] = red[0]; }
}`;

function inverse(Mf) {
  const D = Mf[0] * Mf[4] - Mf[1] * Mf[3];
  const id = D !== 0 ? 1 / D : 0;
  const a11 = Mf[4] * id, a22 = Mf[0] * id, a12 = -Mf[1] * id, a21 = -Mf[3] * id;
  return [a11, a12, -a11 * Mf[2] - a12 * Mf[5], a21, a22, -a21 * Mf[2] - a22 * Mf[5]];
}

function packed(gpu, im) {
  const n = Math.ceil(im.data.length / 4) * 4;
  const u8 = new Uint8Array(n);
  u8.set(im.data);
  return gpu.upload(new Uint32Array(u8.buffer));
}

export class Deskew {
  constructor(gpu) { this.gpu = gpu; }

  // sum over rows of (row sum difference)^2 of `ink` turned by each angle (border 0)
  async sharpness(ink, angles) {
    const gpu = this.gpu;
    gpu.scope = "pre";
    const src = packed(gpu, ink);
    const mats = new Float32Array(angles.length * 6);
    angles.forEach((a, i) => mats.set(inverse(rotationMatrix(ink.w / 2, ink.h / 2, a, 1)), i * 6));
    const mb = gpu.upload(mats);
    const out = gpu.buffer(4 * angles.length * ink.h);
    gpu.begin();
    gpu.dispatch(ROWSUM, [src, mb, out], [ink.w, ink.h, angles.length, 0, 0, 0, 0, 0], [ink.h, angles.length]);
    const [res] = await gpu.end([[out, 4 * angles.length * ink.h]]);
    src.destroy(); mb.destroy(); out.destroy();
    const rows = new Uint32Array(res);
    return angles.map((_, a) => {
      let s = 0;
      for (let y = 1; y < ink.h; y++) { const d = rows[a * ink.h + y] - rows[a * ink.h + y - 1]; s += d * d; }
      return s;
    });
  }

  // the page turned by `deg` about its centre, borders replicated (pipeline.rotate_page)
  async rotate(gray, deg) {
    const gpu = this.gpu;
    gpu.scope = "pre";
    const W = gray.w, H = gray.h;
    // each thread writes 4 whole bytes of the packed output, so it handles 4 consecutive pixel indices
    const n4 = Math.ceil((W * H) / 4);
    const src = packed(gpu, gray);
    const mb = gpu.upload(new Float32Array(inverse(rotationMatrix(W / 2, H / 2, deg, 1))));
    const out = gpu.buffer(4 * n4);
    const grid = gpu.grid(n4);
    gpu.begin();
    gpu.dispatch(WARP2, [src, mb, out], [W, H, n4, 1, 0, grid[0], 0, 0], grid);
    const [res] = await gpu.end([[out, 4 * n4]]);
    src.destroy(); mb.destroy(); out.destroy();
    return img(W, H, new Uint8Array(res, 0, W * H).slice());
  }
}

// One thread per packed output word: pixel indices 4t .. 4t+3 (row-major across row ends).
const WARP2 = /* wgsl */ `
struct P { W: u32, H: u32, N4: u32, rep: u32, border: u32, nx: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> mats: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
${SAMPLE}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x + g.y * p.nx * 64u;
  if (t >= p.N4) { return; }
  var m: array<f32, 6>;
  for (var k = 0u; k < 6u; k++) { m[k] = mats[k]; }
  var word = 0u;
  for (var j = 0u; j < 4u; j++) {
    let i = t * 4u + j;
    if (i < p.W * p.H) { word |= sample(m, i % p.W, i / p.W) << (8u * j); }
  }
  out[t] = word;
}`;

// INTER_AREA decimation (imgproc.areaDecimate) on the GPU, in the same order of float32 sums:
// for each destination pixel, over its source rows (weight beta), the row's weighted sum over its
// source columns (weights alpha). Tables: per destination column / row, [start, count] into
// (source index, weight) lists. One thread per 4 packed output pixels.
const AREA = /* wgsl */ `
struct P { W: u32, H: u32, dw: u32, dh: u32, N4: u32, nx: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> xr: array<u32>;
@group(0) @binding(3) var<storage, read> xs: array<u32>;
@group(0) @binding(4) var<storage, read> xa: array<f32>;
@group(0) @binding(5) var<storage, read> yr: array<u32>;
@group(0) @binding(6) var<storage, read> ys: array<u32>;
@group(0) @binding(7) var<storage, read> ya: array<f32>;
@group(0) @binding(8) var<storage, read_write> out: array<u32>;
fn at(i: u32) -> f32 { return f32((src[i >> 2u] >> ((i & 3u) * 8u)) & 255u); }
fn one(x: u32, y: u32) -> u32 {
  var sum = 0.0;
  let y0 = yr[2u * y]; let yn = yr[2u * y + 1u];
  let x0 = xr[2u * x]; let xn = xr[2u * x + 1u];
  for (var j = 0u; j < yn; j++) {
    let row = ys[y0 + j] * p.W;
    var buf = 0.0;
    for (var i = 0u; i < xn; i++) { buf = buf + at(row + xs[x0 + i]) * xa[x0 + i]; }
    if (j == 0u) { sum = ya[y0] * buf; } else { sum = sum + ya[y0 + j] * buf; }
  }
  return u32(clamp(round(sum), 0.0, 255.0));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x + g.y * p.nx * 64u;
  if (t >= p.N4) { return; }
  var word = 0u;
  for (var j = 0u; j < 4u; j++) {
    let i = t * 4u + j;
    if (i < p.dw * p.dh) { word |= one(i % p.dw, i / p.dw) << (8u * j); }
  }
  out[t] = word;
}`;

function tables(ssize, dsize, scale) {
  const tab = areaTab(ssize, dsize, scale); // [d, s, w], grouped by d in order
  const r = new Uint32Array(2 * dsize), si = new Uint32Array(tab.length), w = new Float32Array(tab.length);
  tab.forEach(([d, s, a], k) => { if (r[2 * d + 1] === 0) r[2 * d] = k; r[2 * d + 1]++; si[k] = s; w[k] = a; });
  return [r, si, w];
}

Deskew.prototype.area = async function (src, dw, dh) {
  const gpu = this.gpu;
  gpu.scope = "pre";
  const [xr, xs, xa] = tables(src.w, dw, src.w / dw), [yr, ys, ya] = tables(src.h, dh, src.h / dh);
  const bufs = [packed(gpu, src), ...[xr, xs, xa, yr, ys, ya].map((a) => gpu.upload(a))];
  const n4 = Math.ceil((dw * dh) / 4);
  const out = gpu.buffer(4 * n4);
  const grid = gpu.grid(n4);
  gpu.begin();
  gpu.dispatch(AREA, [...bufs, out], [src.w, src.h, dw, dh, n4, grid[0], 0, 0], grid);
  const [res] = await gpu.end([[out, 4 * n4]]);
  for (const b of [...bufs, out]) b.destroy();
  return img(dw, dh, new Uint8Array(res, 0, dw * dh).slice());
};

registerKernel(ROWSUM, "deskew.rowsum");
registerKernel(WARP2, "deskew.rotate");
registerKernel(AREA, "area");
