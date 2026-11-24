// WGSL for the detector op program (run/export_det.py writes it, run/ref_det.py is the spec).
//
// Every kernel takes a uniform struct P at binding 0, then its storage buffers in order.
// Activations are f32 CHW. Weights and biases are one f32 array (see WEIGHTS). Uniform structs are
// padded to a multiple of 16 bytes.
// Dispatch sizes are chosen in engine.js; 2-D workgroups are 8x8, 1-D are 64.

const COMMON = /* wgsl */ `
fn erf_approx(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * exp(-a * a);
  return select(-y, y, x >= 0.0);
}
fn activate(v: f32, act: u32, ha: f32, hb: f32) -> f32 {
  var r = v;
  if (act == 1u) { r = max(v, 0.0); }
  if (act == 2u) { r = clamp(ha * v + hb, 0.0, 1.0); }
  if (act == 3u) { r = 0.5 * v * (1.0 + erf_approx(v * 0.70710678)); }
  if (act == 4u) { r = 1.0 / (1.0 + exp(-v)); }
  return r;
}
`;

// Parameters: weights and biases as f32 (engine.js dequantises the packed f16 hi/lo planes once
// at load: hi + lo, rounded to f32, the same value run/ref_det.py uses).
const WEIGHTS = /* wgsl */ `
@group(0) @binding(3) var<storage, read> w: array<f32>;
fn wf(i: u32) -> f32 { return w[i]; }
`;

// Conv with explicit pads, dilation, groups, and a fused activation. Weights [O, Icg, kh, kw].
// One output channel per invocation; out-of-image taps are read at a clamped index and multiplied
// by 0 (no branches in the inner loop, which SwiftShader runs much faster).
export const CONV = COMMON + /* wgsl */ `
struct P {
  ic: u32, ih: u32, iw: u32, oc: u32, oh: u32, ow: u32, icg: u32, ocg: u32,
  kh: u32, kw: u32, sh: u32, sw: u32, pt: u32, pl: u32, dh: u32, dw: u32,
  act: u32, wo: u32, bo: u32, z2: u32, ha: f32, hb: f32, z0: u32, z1: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
` + WEIGHTS + /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y; let o = g.z;
  if (ox >= p.ow || oy >= p.oh || o >= p.oc) { return; }
  var acc = w[p.bo + o];
  let grp = o / p.ocg;
  let y0 = i32(oy * p.sh) - i32(p.pt);
  let x0 = i32(ox * p.sw) - i32(p.pl);
  for (var ic: u32 = 0u; ic < p.icg; ic = ic + 1u) {
    let cb = (grp * p.icg + ic) * p.ih * p.iw;
    let wb = p.wo + (o * p.icg + ic) * p.kh * p.kw;
    for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
      let iy = y0 + i32(ky * p.dh);
      let cy = u32(clamp(iy, 0, i32(p.ih) - 1));
      let rowok = select(0.0, 1.0, iy >= 0 && iy < i32(p.ih));
      for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
        let ix = x0 + i32(kx * p.dw);
        let cx = u32(clamp(ix, 0, i32(p.iw) - 1));
        let m = rowok * select(0.0, 1.0, ix >= 0 && ix < i32(p.iw));
        acc = acc + m * x[cb + cy * p.iw + cx] * wf(wb + ky * p.kw + kx);
      }
    }
  }
  y[(o * p.oh + oy) * p.ow + ox] = activate(acc, p.act, p.ha, p.hb);
}
`;

// Same conv, four output channels per invocation (needs Ocg % 4 == 0 so the four share a group).
// Each input tap is loaded once and feeds four multiply-adds. Dispatch z = O / 4.
export const CONV4 = COMMON + /* wgsl */ `
struct P {
  ic: u32, ih: u32, iw: u32, oc: u32, oh: u32, ow: u32, icg: u32, ocg: u32,
  kh: u32, kw: u32, sh: u32, sw: u32, pt: u32, pl: u32, dh: u32, dw: u32,
  act: u32, wo: u32, bo: u32, z2: u32, ha: f32, hb: f32, z0: u32, z1: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
` + WEIGHTS + /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y; let ob = g.z * 4u;
  if (ox >= p.ow || oy >= p.oh || ob >= p.oc) { return; }
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
    let cb = (grp * p.icg + ic) * p.ih * p.iw;
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
        a0 = a0 + v * wf(wb0 + k);
        a1 = a1 + v * wf(wb0 + stride + k);
        a2 = a2 + v * wf(wb0 + 2u * stride + k);
        a3 = a3 + v * wf(wb0 + 3u * stride + k);
      }
    }
  }
  let base = (ob * p.oh + oy) * p.ow + ox;
  y[base] = activate(a0, p.act, p.ha, p.hb);
  y[base + p.oh * p.ow] = activate(a1, p.act, p.ha, p.hb);
  y[base + 2u * p.oh * p.ow] = activate(a2, p.act, p.ha, p.hb);
  y[base + 3u * p.oh * p.ow] = activate(a3, p.act, p.ha, p.hb);
}
`;

// ConvTranspose, kernel 2 stride 2 no pads (each output pixel has one tap). Weights [Cin, Cout, 2, 2].
export const CONVT = COMMON + /* wgsl */ `
struct P {
  ic: u32, ih: u32, iw: u32, oc: u32, oh: u32, ow: u32, act: u32, wo: u32,
  bo: u32, z2: u32, ha: f32, hb: f32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
` + WEIGHTS + /* wgsl */ `
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y; let o = g.z;
  if (ox >= p.ow || oy >= p.oh || o >= p.oc) { return; }
  let iy = oy / 2u; let ky = oy - iy * 2u;
  let ix = ox / 2u; let kx = ox - ix * 2u;
  var acc = w[p.bo + o];
  for (var ic: u32 = 0u; ic < p.ic; ic = ic + 1u) {
    acc = acc + x[(ic * p.ih + iy) * p.iw + ix] * wf(p.wo + ((ic * p.oc + o) * 2u + ky) * 2u + kx);
  }
  y[(o * p.oh + oy) * p.ow + ox] = activate(acc, p.act, p.ha, p.hb);
}
`;

// Max pool, pads excluded from the max (ONNX semantics).
export const MAXPOOL = /* wgsl */ `
struct P { c: u32, ih: u32, iw: u32, oh: u32, ow: u32, kh: u32, kw: u32, sh: u32, sw: u32, pt: u32, pl: u32, z0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y; let c = g.z;
  if (ox >= p.ow || oy >= p.oh || c >= p.c) { return; }
  var m = -3.4e38;
  for (var ky: u32 = 0u; ky < p.kh; ky = ky + 1u) {
    let iy = i32(oy * p.sh + ky) - i32(p.pt);
    if (iy < 0 || iy >= i32(p.ih)) { continue; }
    for (var kx: u32 = 0u; kx < p.kw; kx = kx + 1u) {
      let ix = i32(ox * p.sw + kx) - i32(p.pl);
      if (ix < 0 || ix >= i32(p.iw)) { continue; }
      m = max(m, x[(c * p.ih + u32(iy)) * p.iw + u32(ix)]);
    }
  }
  y[(c * p.oh + oy) * p.ow + ox] = m;
}
`;

// Copies a channel slice of x into y at channel offset coff (Concat along channels).
export const COPY = /* wgsl */ `
struct P { cs: u32, h: u32, wd: u32, coff: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let xx = g.x; let yy = g.y; let c = g.z;
  if (xx >= p.wd || yy >= p.h || c >= p.cs) { return; }
  y[((c + p.coff) * p.h + yy) * p.wd + xx] = x[(c * p.h + yy) * p.wd + xx];
}
`;

// Nearest-neighbour upsample by an integer factor (asymmetric, floor).
export const RESIZE = /* wgsl */ `
struct P { c: u32, ih: u32, iw: u32, oh: u32, ow: u32, s: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y; let c = g.z;
  if (ox >= p.ow || oy >= p.oh || c >= p.c) { return; }
  y[(c * p.oh + oy) * p.ow + ox] = x[(c * p.ih + oy / p.s) * p.iw + ox / p.s];
}
`;

// Squeeze-excite pool: channel means over h*w (one thread per channel).
export const GAP = /* wgsl */ `
struct P { c: u32, hw: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  if (c >= p.c) { return; }
  // Kahan-compensated sum: a plain f32 running sum over tens of thousands of pixels drifts by
  // ~1e-4 relative, which the gate then carries through the network.
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

// Squeeze-excite apply: y = x * gate[channel] (+ x when res). Indexed as i = gid.x + gid.y * nx.
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
