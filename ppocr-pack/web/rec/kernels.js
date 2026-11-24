// WGSL for the recognizer op program (run/export_rec.py writes it, run/ref_rec.py is the spec).
//
// Every kernel takes a uniform struct P at binding 0, then its storage buffers in order. Activations are
// f32: a "chw" buffer is [B][C][H][W], a "tc" buffer [B][T][C]. Weights, biases and norm parameters are
// one f32 array (engine.js dequantises the packed file once at load). Uniform structs are padded to a
// multiple of 16 bytes. 2-D workgroups are 8x8, 1-D ones 64 (256 for row reductions and the matmul).
//
// The conv, pool, squeeze-excite and add kernels come from tesseract-gpu/web/det/kernels.js (the PP-OCRv6
// detector port), with a batch index folded into the z dimension and two more activations.

const COMMON = /* wgsl */ `
fn erf_approx(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-a * a);
  return select(-y, y, x >= 0.0);
}
// act: 0 none, 1 relu, 2 hard sigmoid, 3 exact GELU, 4 sigmoid, 5 hard swish, 6 swish
fn activate(v: f32, act: u32, ha: f32, hb: f32) -> f32 {
  var r = v;
  if (act == 1u) { r = max(v, 0.0); }
  if (act == 2u) { r = clamp(ha * v + hb, 0.0, 1.0); }
  if (act == 3u) { r = 0.5 * v * (1.0 + erf_approx(v * 0.70710678)); }
  if (act == 4u) { r = 1.0 / (1.0 + exp(-v)); }
  if (act == 5u) { r = v * clamp(ha * v + hb, 0.0, 1.0); }
  if (act == 6u) { r = v * (1.0 / (1.0 + exp(-v))); }
  return r;
}
`;

const CONV_P = /* wgsl */ `
struct P {
  ic: u32, ih: u32, iw: u32, oc: u32, oh: u32, ow: u32, icg: u32, ocg: u32,
  kh: u32, kw: u32, sh: u32, sw: u32, pt: u32, pl: u32, dh: u32, dw: u32,
  act: u32, wo: u32, bo: u32, nb: u32, ha: f32, hb: f32, z0: u32, z1: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
`;

// Conv with explicit pads, dilation, groups and a fused activation. Weights [O, Icg, kh, kw] at p.wo,
// bias at p.bo. One output channel per invocation; dispatch z = B * O. Out-of-image taps are read at a
// clamped index and multiplied by 0 (no branches in the inner loop, which SwiftShader runs faster).
export const CONV = COMMON + CONV_P + /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y;
  let b = g.z / p.oc; let o = g.z - b * p.oc;
  if (ox >= p.ow || oy >= p.oh || b >= p.nb) { return; }
  var acc = w[p.bo + o];
  let grp = o / p.ocg;
  let y0 = i32(oy * p.sh) - i32(p.pt);
  let x0 = i32(ox * p.sw) - i32(p.pl);
  for (var ic: u32 = 0u; ic < p.icg; ic = ic + 1u) {
    let cb = (b * p.ic + grp * p.icg + ic) * p.ih * p.iw;
    let wb = p.wo + (o * p.icg + ic) * p.kh * p.kw;
    for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
      let iy = y0 + i32(ky * p.dh);
      let cy = u32(clamp(iy, 0, i32(p.ih) - 1));
      let rowok = select(0.0, 1.0, iy >= 0 && iy < i32(p.ih));
      for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
        let ix = x0 + i32(kx * p.dw);
        let cx = u32(clamp(ix, 0, i32(p.iw) - 1));
        let m = rowok * select(0.0, 1.0, ix >= 0 && ix < i32(p.iw));
        acc = acc + m * x[cb + cy * p.iw + cx] * w[wb + ky * p.kw + kx];
      }
    }
  }
  y[((b * p.oc + o) * p.oh + oy) * p.ow + ox] = activate(acc, p.act, p.ha, p.hb);
}
`;

// Same conv, four output channels per invocation (needs Ocg % 4 == 0 so the four share a group).
// Each input tap is loaded once and feeds four multiply-adds. Dispatch z = B * O / 4.
export const CONV4 = COMMON + CONV_P + /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y;
  let q = p.oc / 4u;
  let b = g.z / q; let ob = (g.z - b * q) * 4u;
  if (ox >= p.ow || oy >= p.oh || b >= p.nb) { return; }
  let grp = ob / p.ocg;
  var a0 = w[p.bo + ob];
  var a1 = w[p.bo + ob + 1u];
  var a2 = w[p.bo + ob + 2u];
  var a3 = w[p.bo + ob + 3u];
  let y0 = i32(oy * p.sh) - i32(p.pt);
  let x0 = i32(ox * p.sw) - i32(p.pl);
  let kk = p.kh * p.kw;
  let stride = p.icg * kk;
  for (var ic: u32 = 0u; ic < p.icg; ic = ic + 1u) {
    let cb = (b * p.ic + grp * p.icg + ic) * p.ih * p.iw;
    let wb0 = p.wo + (ob * p.icg + ic) * kk;
    for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
      let iy = y0 + i32(ky * p.dh);
      let cy = u32(clamp(iy, 0, i32(p.ih) - 1));
      let rowok = select(0.0, 1.0, iy >= 0 && iy < i32(p.ih));
      for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
        let ix = x0 + i32(kx * p.dw);
        let cx = u32(clamp(ix, 0, i32(p.iw) - 1));
        let v = rowok * select(0.0, 1.0, ix >= 0 && ix < i32(p.iw)) * x[cb + cy * p.iw + cx];
        let k = ky * p.kw + kx;
        a0 = a0 + v * w[wb0 + k];
        a1 = a1 + v * w[wb0 + stride + k];
        a2 = a2 + v * w[wb0 + 2u * stride + k];
        a3 = a3 + v * w[wb0 + 3u * stride + k];
      }
    }
  }
  let plane = p.oh * p.ow;
  let base = ((b * p.oc + ob) * p.oh + oy) * p.ow + ox;
  y[base] = activate(a0, p.act, p.ha, p.hb);
  y[base + plane] = activate(a1, p.act, p.ha, p.hb);
  y[base + 2u * plane] = activate(a2, p.act, p.ha, p.hb);
  y[base + 3u * plane] = activate(a3, p.act, p.ha, p.hb);
}
`;

// Max pool (mode 0; pads excluded from the max, ONNX semantics) or average pool (mode 1; divisor
// kh * kw when cip, else the count of in-image taps). Dispatch z = B * C.
export const POOL = /* wgsl */ `
struct P {
  c: u32, ih: u32, iw: u32, oh: u32, ow: u32, kh: u32, kw: u32, sh: u32,
  sw: u32, pt: u32, pl: u32, mode: u32, cip: u32, nb: u32, z0: u32, z1: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y; let bc = g.z;
  if (ox >= p.ow || oy >= p.oh || bc >= p.c * p.nb) { return; }
  var m = -3.4e38;
  var s = 0.0;
  var n = 0.0;
  for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
    let iy = i32(oy * p.sh + ky) - i32(p.pt);
    if (iy < 0 || iy >= i32(p.ih)) { continue; }
    for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
      let ix = i32(ox * p.sw + kx) - i32(p.pl);
      if (ix < 0 || ix >= i32(p.iw)) { continue; }
      let v = x[(bc * p.ih + u32(iy)) * p.iw + u32(ix)];
      m = max(m, v);
      s = s + v;
      n = n + 1.0;
    }
  }
  var r = m;
  if (p.mode == 1u) { r = s / select(n, f32(p.kh * p.kw), p.cip == 1u); }
  y[(bc * p.oh + oy) * p.ow + ox] = r;
}
`;

// Copies a [B][cs][h][w] input into channels coff.. of a [B][cout][h][w] output (Concat on channels).
export const COPY = /* wgsl */ `
struct P { cs: u32, h: u32, wd: u32, coff: u32, cout: u32, nb: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let xx = g.x; let yy = g.y;
  let b = g.z / p.cs; let c = g.z - b * p.cs;
  if (xx >= p.wd || yy >= p.h || b >= p.nb) { return; }
  y[((b * p.cout + c + p.coff) * p.h + yy) * p.wd + xx] = x[((b * p.cs + c) * p.h + yy) * p.wd + xx];
}
`;

// Squeeze-excite pool: the mean of each of the n = B * C planes of hw values (one thread per plane).
export const GAP = /* wgsl */ `
struct P { n: u32, hw: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  if (c >= p.n) { return; }
  // Kahan-compensated sum: a plain f32 running sum over many pixels drifts by ~1e-4 relative, which
  // the gate then carries through the network (tesseract-gpu/web/det/README.md).
  var s = 0.0;
  var comp = 0.0;
  for (var i: u32 = 0u; i < p.hw; i = i + 1u) {
    let yv = x[c * p.hw + i] - comp;
    let t = s + yv;
    comp = (t - s) - yv;
    s = t;
  }
  m[c] = s / f32(p.hw);
}
`;

// Squeeze-excite apply: y = x * gate[plane] (+ x when res). Indexed as i = gid.x + gid.y * nx.
export const SCALE = /* wgsl */ `
struct P { n: u32, nx: u32, hw: u32, res: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * p.nx;
  if (i >= p.n) { return; }
  let v = x[i];
  var r = v * gate[i / p.hw];
  if (p.res == 1u) { r = r + v; }
  y[i] = r;
}
`;

// Elementwise a + b.
export const ADD = /* wgsl */ `
struct P { n: u32, nx: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * p.nx;
  if (i >= p.n) { return; }
  y[i] = a[i] + b[i];
}
`;

// Per batch item, [r][s] -> [s][r]: img2seq is r = C, s = H * W; seq2img is r = T, s = C.
// Dispatch (ceil(s / 8), ceil(r / 8), B).
export const TRANSPOSE = /* wgsl */ `
struct P { r: u32, s: u32, nb: u32, z0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let si = g.x; let ri = g.y; let b = g.z;
  if (si >= p.s || ri >= p.r || b >= p.nb) { return; }
  y[(b * p.s + si) * p.r + ri] = x[(b * p.r + ri) * p.s + si];
}
`;

// Linear layer Y[m, n] = act(sum_k X[m, k] W[k, n] + bias[n]) over m < M = B * T rows, with W [K, N]
// row-major at p.wo and the bias at p.bo. A workgroup of 256 computes a 64 x 64 output tile, 4 x 4 per
// thread, stepping K by 16 through workgroup memory (the tiled pattern of _lib/wgsl/kernels.js
// matmul_tiled, with bounds for any M, N, K). Dispatch (ceil(N / 64), ceil(M / 64)).
export const LINEAR = COMMON + /* wgsl */ `
struct P { m: u32, k: u32, n: u32, wo: u32, bo: u32, act: u32, ha: f32, hb: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
var<workgroup> xs: array<f32, 1024>;
var<workgroup> ws: array<f32, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r0 = wid.y * 64u; let n0 = wid.x * 64u;
  let tx = l % 16u; let ty = l / 16u;
  var acc = array<vec4<f32>, 4>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
  for (var k0: u32 = 0u; k0 < p.k; k0 = k0 + 16u) {
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let e = l * 4u + i;
      let xr = r0 + e / 16u; let xk = k0 + e % 16u;
      let okx = xr < p.m && xk < p.k;
      xs[e] = select(0.0, x[select(0u, xr * p.k + xk, okx)], okx);
      let wk = k0 + e / 64u; let wn = n0 + e % 64u;
      let okw = wk < p.k && wn < p.n;
      ws[e] = select(0.0, w[select(p.wo, p.wo + wk * p.n + wn, okw)], okw);
    }
    workgroupBarrier();
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      let wv = vec4<f32>(ws[k * 64u + tx * 4u], ws[k * 64u + tx * 4u + 1u], ws[k * 64u + tx * 4u + 2u], ws[k * 64u + tx * 4u + 3u]);
      for (var i: u32 = 0u; i < 4u; i = i + 1u) { acc[i] = acc[i] + xs[(ty * 4u + i) * 16u + k] * wv; }
    }
    workgroupBarrier();
  }
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let r = r0 + ty * 4u + i;
    if (r >= p.m) { break; }
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
      let n = n0 + tx * 4u + c;
      if (n < p.n) { y[r * p.n + n] = activate(acc[i][c] + w[p.bo + n], p.act, p.ha, p.hb); }
    }
  }
}
`;

// LayerNorm over the last axis, one thread per row: y = (x - mean) / sqrt(var + eps) * g + b, the
// variance taken around the mean (two passes), as the ONNX ReduceMean/Sub/Pow/Sqrt chain computes it.
export const LN = /* wgsl */ `
struct P { m: u32, d: u32, go: u32, bo: u32, eps: f32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x;
  if (r >= p.m) { return; }
  let base = r * p.d;
  var s = 0.0;
  for (var j: u32 = 0u; j < p.d; j = j + 1u) { s = s + x[base + j]; }
  let mean = s / f32(p.d);
  var v = 0.0;
  for (var j: u32 = 0u; j < p.d; j = j + 1u) { let d = x[base + j] - mean; v = v + d * d; }
  let sd = sqrt(v / f32(p.d) + p.eps);
  for (var j: u32 = 0u; j < p.d; j = j + 1u) {
    y[base + j] = (x[base + j] - mean) / sd * w[p.go + j] + w[p.bo + j];
  }
}
`;

// Multi-head self-attention over a whole line, no mask. qkv [B][T][3 * D] laid out as [3][nh][hd] per
// row (D = nh * hd); out [B][T][D] with head h at columns h * hd. One thread per (query, head): pass 1
// finds the max score, pass 2 sums exp and the weighted values. hd <= 32. Dispatch (ceil(T / 64), nh, B).
export const ATTN = /* wgsl */ `
struct P { t: u32, nh: u32, hd: u32, nb: u32, scale: f32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x; let h = g.y; let b = g.z;
  if (t >= p.t || b >= p.nb) { return; }
  let D = p.nh * p.hd;
  let row = 3u * D;
  let base = b * p.t * row;
  var q = array<f32, 32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  var acc = array<f32, 32>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var d: u32 = 0u; d < p.hd; d = d + 1u) { q[d] = qkv[base + t * row + h * p.hd + d] * p.scale; }
  var mx = -3.4e38;
  for (var s: u32 = 0u; s < p.t; s = s + 1u) {
    let kb = base + s * row + D + h * p.hd;
    var sc = 0.0;
    for (var d: u32 = 0u; d < p.hd; d = d + 1u) { sc = sc + q[d] * qkv[kb + d]; }
    mx = max(mx, sc);
  }
  var sum = 0.0;
  for (var s: u32 = 0u; s < p.t; s = s + 1u) {
    let kb = base + s * row + D + h * p.hd;
    var sc = 0.0;
    for (var d: u32 = 0u; d < p.hd; d = d + 1u) { sc = sc + q[d] * qkv[kb + d]; }
    let e = exp(sc - mx);
    sum = sum + e;
    let vb = kb + D;
    for (var d: u32 = 0u; d < p.hd; d = d + 1u) { acc[d] = acc[d] + e * qkv[vb + d]; }
  }
  let ob = (b * p.t + t) * D + h * p.hd;
  for (var d: u32 = 0u; d < p.hd; d = d + 1u) { o[ob + d] = acc[d] / sum; }
}
`;

// Softmax over each row of n values, one workgroup of 256 per row (row = wid.x + wid.y * 65535).
export const SOFTMAX = /* wgsl */ `
struct P { m: u32, n: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let row = wid.x + wid.y * 65535u;
  if (row >= p.m) { return; }
  let base = row * p.n;
  var mx = -3.4e38;
  for (var j = l; j < p.n; j = j + 256u) { mx = max(mx, x[base + j]); }
  red[l] = mx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (l < s) { red[l] = max(red[l], red[l + s]); }
    workgroupBarrier();
  }
  mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = l; j < p.n; j = j + 256u) { sum = sum + exp(x[base + j] - mx); }
  red[l] = sum;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (l < s) { red[l] = red[l] + red[l + s]; }
    workgroupBarrier();
  }
  let tot = red[0];
  for (var j = l; j < p.n; j = j + 256u) { y[base + j] = exp(x[base + j] - mx) / tot; }
}
`;
