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

const WGSL = /* wgsl */ `
struct Params { hp: u32, ns: u32, intMode: u32, ncols: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pool: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read> tanhT: array<f32>;
@group(0) @binding(5) var<storage, read> logT: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read> colMeta: array<vec4<u32>>;

var<workgroup> prev: array<f32, 64>;
var<workgroup> xin: array<f32, 16>;

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
  let col = wid.y * nwg.x + wid.x;
  if (col >= P.ncols) { return; }   // whole workgroup returns together: barriers stay uniform
  let m = colMeta[col];             // x: first pool row (pool_off + x), y: T, z: output row
  let lid = lid3.x;
  let ns = P.ns;
  let L = 16u + ns + 1u;
  let isInt = P.intMode == 1u;
  let j = lid;
  let owns = j < ns;

  prev[lid] = 0.0;
  workgroupBarrier();

  var state: f32 = 0.0;
  for (var y = 0u; y < P.hp; y = y + 1u) {
    if (lid < 16u) {
      xin[lid] = pool[(m.x + y * m.y) * 16u + lid];
    }
    workgroupBarrier();

    var newOut: f32 = 0.0;
    if (owns) {
      var dots: array<f32, 4>;
      for (var g = 0u; g < 4u; g = g + 1u) {
        let r = g * ns + j;
        let base = r * L;
        var d: f32 = 0.0;
        for (var k = 0u; k < 16u; k = k + 1u) {
          d = d + w[base + k] * xin[k];
        }
        for (var k = 0u; k < ns; k = k + 1u) {
          d = d + w[base + 16u + k] * prev[k];
        }
        let b = w[base + L - 1u];
        if (isInt) {
          dots[g] = (d + b * 127.0) * s[r];
        } else {
          dots[g] = d + b;
        }
      }
      let ci = tanhF(dots[0]);
      let gi = logisticF(dots[1]);
      let gf = logisticF(dots[2]);
      let go = logisticF(dots[3]);
      state = clamp(state * gf + ci * gi, -100.0, 100.0);
      let o = tanhF(state) * go;
      newOut = select(o, quant(o), isInt);
    }
    workgroupBarrier();
    if (owns) {
      prev[j] = newOut;
    }
    workgroupBarrier();
  }
  if (owns) {
    out[m.z * ns + j] = prev[j];
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
  gpu.dispatch(WGSL, [params, pool, w, scale, tanhT, logT, out, colMeta], Gpu.grid(n));
}
