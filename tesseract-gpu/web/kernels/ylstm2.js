// v2 (tesseract-gpu round 2): gate-interleaved [K][ns] vec4 weights (interleaveGates in
// xlstm2.js), 4 columns per workgroup. Original header follows.
// y-summarising LSTM (Tesseract "Lfys" SummLSTM after XYTranspose) as one WebGPU dispatch.
//
//   recordYLstm(gpu, { pool, w, s, tanhT, logT, out, lines, H, ns, intMode })
//
// pool  f32 [sum Hp*T_i, 16]  packed Maxpool output; line i, row y, column x is row
//                              (pool_off_i + y*T_i + x)
// w     f32 [4*ns, 16+ns+1]   gate blocks CI, GI, GF1, GO; columns: x (16), prev out (ns), bias
//                              (int mode: integer codes)
// s     f32 [4*ns]            per-row scales (int mode); ignored in float mode (may be null)
// tanhT, logT f32 [4096]      Tesseract lookup tables, x = i/256, linear interpolation
// out   f32 [sum T_i, ns]     ylstm: last step output of every column, at row t_off_i + x
// lines                       manifest lines: {T, pool_off, t_off}
//
// One workgroup per column (x of a line), 64 threads; thread j < ns owns hidden unit j and
// its four gate rows. The previous output lives in workgroup memory. Columns are independent,
// so the whole layer is a single dispatch. Only gpu.dispatch is called; the caller submits.
import { Gpu } from "../gpu.js";

const COLS = 4; // columns per workgroup: each weight vec4 read serves 4 columns

const WGSL = /* wgsl */ `
struct Params { hp: u32, ns: u32, intMode: u32, ncols: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pool: array<f32>;
@group(0) @binding(2) var<storage, read> w4: array<vec4<f32>>;   // [K][ns] of (CI, GI, GF1, GO)
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read> tanhT: array<f32>;
@group(0) @binding(5) var<storage, read> logT: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read> colMeta: array<vec4<u32>>;

const C: u32 = ${COLS}u;
var<workgroup> prev: array<f32, ${64 * COLS}>;
var<workgroup> xin: array<f32, ${16 * COLS}>;

fn tanhF(x: f32) -> f32 {
  let a = abs(x);
  var v: f32 = 1.0;
  if (a < 4095.0 / 256.0) {
    let xs = a * 256.0;
    let i = u32(xs);
    let f = xs - f32(i);
    let t0 = tanhT[i];
    v = t0 + (tanhT[i + 1u] - t0) * f;
  }
  return select(v, -v, x < 0.0);
}

fn logisticF(x: f32) -> f32 {
  let a = abs(x);
  var v: f32 = 1.0;
  if (a < 4095.0 / 256.0) {
    let xs = a * 256.0;
    let i = u32(xs);
    let f = xs - f32(i);
    let l0 = logT[i];
    v = l0 + (logT[i + 1u] - l0) * f;
  }
  return select(v, 1.0 - v, x < 0.0);
}

// Tesseract's quant in int mode: round half away from zero of 127 * clamp(v, -1, 1).
fn quant(v: f32) -> f32 {
  let c = clamp(v, -1.0, 1.0) * 127.0;
  return clamp(sign(c) * floor(abs(c) + 0.5), -127.0, 127.0);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid3: vec3<u32>) {
  let col0 = (wid.y * nwg.x + wid.x) * C;
  if (col0 >= P.ncols) { return; }   // uniform per workgroup
  let lid = lid3.x;
  let ns = P.ns;
  let K = 16u + ns + 1u;
  let isInt = P.intMode == 1u;
  let j = lid;
  let owns = j < ns;
  let bias = select(1.0, 127.0, isInt);
  var cm: array<vec4<u32>, ${COLS}>;
  for (var c = 0u; c < C; c++) {
    cm[c] = colMeta[min(col0 + c, P.ncols - 1u)];
    prev[c * 64u + lid] = 0.0;
  }
  workgroupBarrier();

  var state: array<f32, ${COLS}>;
  for (var c = 0u; c < C; c++) { state[c] = 0.0; }
  for (var y = 0u; y < P.hp; y = y + 1u) {
    if (lid < 16u) {
      for (var c = 0u; c < C; c++) {
        let m = cm[c];
        xin[c * 16u + lid] = pool[(m.x + y * m.y) * 16u + lid];
      }
    }
    workgroupBarrier();
    var newOut: array<f32, ${COLS}>;
    if (owns) {
      var acc: array<vec4<f32>, ${COLS}>;
      for (var c = 0u; c < C; c++) { acc[c] = vec4<f32>(0.0); }
      for (var k = 0u; k < 16u; k = k + 1u) {
        let wv = w4[k * ns + j];
        for (var c = 0u; c < C; c++) { acc[c] += wv * xin[c * 16u + k]; }
      }
      for (var k = 0u; k < ns; k = k + 1u) {
        let wv = w4[(16u + k) * ns + j];
        for (var c = 0u; c < C; c++) { acc[c] += wv * prev[c * 64u + k]; }
      }
      let wb = w4[(K - 1u) * ns + j];
      for (var c = 0u; c < C; c++) {
        var d = acc[c] + wb * bias;
        if (isInt) { d = d * vec4<f32>(s[j], s[ns + j], s[2u * ns + j], s[3u * ns + j]); }
        let ci = tanhF(d.x);
        let gi = logisticF(d.y);
        let gf = logisticF(d.z);
        let go = logisticF(d.w);
        state[c] = clamp(state[c] * gf + ci * gi, -100.0, 100.0);
        let o = tanhF(state[c]) * go;
        newOut[c] = select(o, quant(o), isInt);
      }
    }
    workgroupBarrier();
    if (owns) {
      for (var c = 0u; c < C; c++) { prev[c * 64u + j] = newOut[c]; }
    }
    workgroupBarrier();
  }
  if (owns) {
    for (var c = 0u; c < C; c++) {
      if (col0 + c < P.ncols) { out[cm[c].z * ns + j] = prev[c * 64u + j]; }
    }
  }
}
`;

export function recordYLstm(gpu, { pool, w, s, tanhT, logT, out, lines, H, ns, intMode }) {
  const cols = [];
  for (const ln of lines) {
    for (let x = 0; x < ln.T; x++) cols.push(ln.pool_off + x, ln.T, ln.t_off + x, 0);
  }
  const n = cols.length / 4;
  const colMeta = gpu.upload(new Uint32Array(cols));
  const params = gpu.uniform(new Uint32Array([Math.floor(H / 3), ns, intMode ? 1 : 0, n]));
  const scale = s ?? gpu.upload(new Float32Array(4 * ns).fill(1));
  gpu.dispatch(WGSL, [params, pool, w, scale, tanhT, logT, out, colMeta], Gpu.grid(Math.ceil(n / COLS)));
}
