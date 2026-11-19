// Convolve 3x3 (1 -> 9 features) + FullyConnected (9+1 -> 16) + tanh, and Maxpool 3x3, over a
// packed batch of lines. Layouts are those documented in run/make_fixtures.py.
// Both functions only record dispatches; the caller begins and submits.
//
//   recordConv(gpu, { x, pads, w, s, tanhT, out, lines, H, intMode })
//   recordPool(gpu, { conv, out, lines, H })
//
// Line records: { W, in_off, pad_off, pool_off } (offsets in elements of the packed axis).
// Border index for pad lookup (ref.border_positions, raster order), with nI = min(2, W):
//   base(y) = (y > 0 ? W : 0) + max(0, y-1) * nI    rows above y, all of them border or interior
//   border row (y == 0 or y == H-1):  index = base(y) + x
//   interior row, border column:      index = base(y) + (x > 0 ? 1 : 0)

import { Gpu } from "../gpu.js";

const WG = 64;

// Tesseract's TanhTable with linear interpolation (lstm/functions.h Tanh, ref._lut).
const TANH_FN = `
fn tanh_lut(x: f32) -> f32 {
  let a = abs(x);
  var v = 1.0;
  if (a < 4095.0 / 256.0) {
    let xs = a * 256.0;
    let i = u32(floor(xs));
    let t0 = tanhT[i];
    let t1 = tanhT[i + 1u];
    v = t0 + (t1 - t0) * (xs - f32(i));
  }
  return select(v, -v, x < 0.0);
}`;

const CONV_WGSL = (intMode) => `
@group(0) @binding(0) var<uniform> P: vec4<u32>;            // H, nLines, total, -
@group(0) @binding(1) var<storage, read> L: array<u32>;     // per line: W, in_off, pad_off, pool_off
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> PADS: array<f32>;  // [sum nb, 9]
@group(0) @binding(4) var<storage, read> WT: array<f32>;    // [16, 10]
@group(0) @binding(5) var<storage, read> S: array<f32>;     // [16] int row scales
@group(0) @binding(6) var<storage, read> tanhT: array<f32>; // [4096]
@group(0) @binding(7) var<storage, read_write> OUT: array<f32>; // [sum H*W, 16]

const INT_MODE: bool = ${intMode ? "true" : "false"};

${TANH_FN}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let g = gid.x + gid.y * nwg.x * ${WG}u;
  if (g >= P.z) { return; }
  let H = P.x;

  var li = 0u;
  for (var i = 0u; i < P.y; i++) {
    let off = L[4u * i + 1u];
    let size = H * L[4u * i];
    if (g >= off && g < off + size) { li = i; }
  }
  let W = L[4u * li];
  let in_off = L[4u * li + 1u];
  let pad_off = L[4u * li + 2u];
  let p = g - in_off;
  let y = p / W;
  let x = p - y * W;

  let border_row = (y == 0u) || (y == H - 1u);
  let border_pix = border_row || x == 0u || x == W - 1u;
  var bi = 0u;
  if (border_pix) {
    var base = 0u;
    if (y > 0u) { base = W + (y - 1u) * min(2u, W); }
    if (border_row) {
      bi = base + x;
    } else {
      bi = base;
      if (x > 0u) { bi = base + 1u; }
    }
  }

  // slot k = (dx+1)*3 + (dy+1): dx moves along x, dy along y.
  var f: array<f32, 9>;
  for (var k = 0u; k < 9u; k++) {
    let yy = i32(y) + i32(k % 3u) - 1;
    let xx = i32(x) + i32(k / 3u) - 1;
    if (yy >= 0 && yy < i32(H) && xx >= 0 && xx < i32(W)) {
      f[k] = X[in_off + u32(yy) * W + u32(xx)];
    } else {
      f[k] = PADS[(pad_off + bi) * 9u + k];
    }
  }

  for (var o = 0u; o < 16u; o++) {
    let wb = o * 10u;
    var acc = 0.0;
    for (var k = 0u; k < 9u; k++) { acc += WT[wb + k] * f[k]; }
    var v: f32;
    if (INT_MODE) {
      v = (acc + WT[wb + 9u] * 127.0) * S[o];
    } else {
      v = acc + WT[wb + 9u];
    }
    let t = tanh_lut(v);
    var q = t;
    if (INT_MODE) {
      let r = 127.0 * clamp(t, -1.0, 1.0);
      q = clamp(sign(r) * floor(abs(r) + 0.5), -127.0, 127.0);
    }
    OUT[g * 16u + o] = q;
  }
}`;

const POOL_WGSL = `
@group(0) @binding(0) var<uniform> P: vec4<u32>;            // H, nLines, total pool rows, -
@group(0) @binding(1) var<storage, read> L: array<u32>;     // per line: W, in_off, pad_off, pool_off
@group(0) @binding(2) var<storage, read> CONV: array<f32>;  // [sum H*W, 16]
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>; // [sum Hp*T, 16]

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let g = gid.x + gid.y * nwg.x * ${WG}u;
  if (g >= P.z) { return; }
  let Hp = P.x / 3u;

  var li = 0u;
  for (var i = 0u; i < P.y; i++) {
    let T = L[4u * i] / 3u;
    let off = L[4u * i + 3u];
    if (g >= off && g < off + Hp * T) { li = i; }
  }
  let W = L[4u * li];
  let in_off = L[4u * li + 1u];
  let T = W / 3u;
  let p = g - L[4u * li + 3u];
  let py = p / T;
  let px = p - py * T;

  for (var o = 0u; o < 16u; o++) {
    var m = -3.0e38;
    for (var dy = 0u; dy < 3u; dy++) {
      for (var dx = 0u; dx < 3u; dx++) {
        let y = py * 3u + dy;
        let x = px * 3u + dx;
        m = max(m, CONV[(in_off + y * W + x) * 16u + o]);
      }
    }
    OUT[g * 16u + o] = m;
  }
}`;

// [W, in_off, pad_off, pool_off] per line, as u32.
function lineTable(lines) {
  const t = new Uint32Array(4 * lines.length);
  lines.forEach((l, i) => {
    t[4 * i] = l.W;
    t[4 * i + 1] = l.in_off;
    t[4 * i + 2] = l.pad_off | 0;
    t[4 * i + 3] = l.pool_off | 0;
  });
  return t;
}

export function recordConv(gpu, { x, pads, w, s, tanhT, out, lines, H, intMode }) {
  if (intMode && !s) throw new Error("recordConv: int mode needs the per-row scales s");
  const total = lines.reduce((n, l) => n + H * l.W, 0);
  const sBuf = intMode ? s : gpu.buffer(64); // float mode never reads s; binding must exist
  const u = gpu.uniform(new Uint32Array([H, lines.length, total, 0]));
  const L = gpu.upload(lineTable(lines));
  gpu.dispatch(CONV_WGSL(!!intMode), [u, L, x, pads, w, sBuf, tanhT, out],
    Gpu.grid(Math.ceil(total / WG)));
}

export function recordPool(gpu, { conv, out, lines, H }) {
  if (H % 3) throw new Error("recordPool: H must be a multiple of 3");
  const total = lines.reduce((n, l) => n + (H / 3) * Math.floor(l.W / 3), 0);
  const u = gpu.uniform(new Uint32Array([H, lines.length, total, 0]));
  const L = gpu.upload(lineTable(lines));
  gpu.dispatch(POOL_WGSL, [u, L, conv, out], Gpu.grid(Math.ceil(total / WG)));
}
