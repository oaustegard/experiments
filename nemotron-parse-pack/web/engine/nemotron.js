// NVIDIA Nemotron Parse 2.0 on hand-written WebGPU kernels, reading a nemotron-pack/1 pack (run/export.py).
//
//   const m = await NemotronParse.load({ base: "models/default/", onProgress });
//   const r = await m.parse(imageBitmapOrCanvasOrImageData, { onToken, signal });
//   // r = { raw, tokens, ms: { encode, decode }, blocks: [{ cls, bbox: [x0, y0, x1, y1], text }] }
//
// Built on the whisper-pack engine (whisper-pack/web/whisper.js): the same uniform header, recorded
// steps (pipeline, bind group, dispatch size), packed weights dequantized in the kernel, one decoder
// token per submit with cached self- and cross-attention keys and values. What is new here:
//   - weights in three formats per tensor (int8 per row, 4-bit with a scale per group, f16 read as
//     f32), chosen by the pack's scheme, streamed shard by shard straight into GPU buffers
//   - a ViT-H encoder over up to 13,320 tokens: attention keeps a running softmax per (query, head)
//     so no score matrix exists, and every big op is cut into bounded dispatches
//   - greedy pick with the repetition penalty on the GPU, so a token costs a 1 KB readback
// Activations are f32 throughout.

import { Tokenizer, toBlocks, resizedSize } from "./postprocess.js";
export * from "./postprocess.js";

const WG = 64;
const QB = 4;   // queries per attention thread

const HEAD = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, f: vec4<f32> }
struct D { n: vec4<u32> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> d: D;
fn erf(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return sign(x) * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf(x * 0.7071067811865476)); }
fn i8x4(w: u32) -> vec4<f32> {
  let i = bitcast<i32>(w);
  return vec4<f32>(vec4<i32>((i << 24u) >> 24u, (i << 16u) >> 24u, (i << 8u) >> 24u, i >> 24u));
}
fn nib4(w: u32, sh: vec4<u32>) -> vec4<f32> {
  return vec4<f32>((vec4<u32>(w) >> sh) & vec4<u32>(15u)) - vec4<f32>(8.0);
}
`;

// Y[row, col0 + n] = act(sum_k W[w0 + n, k] X[x0 + r, k] + B[s0 + n]), one thread per (n, block of R rows):
// each weight word is decoded once and applied to R rows. K, the format and R are constants.
// a = (T, x row0, y row0, flags)  b = (w row0 in this buffer, scale/bias row0, y col0, y row width)  c.z = N
// flags: 1 gelu, 2 add into Y, 4 y row += d.n.x
const linearWGSL = (fmt, K, G, R) => {
  const sb = fmt === "f32" ? 3 : 4;
  const body = fmt === "int8" ? /* wgsl */ `
  let wo = wr * ${K / 4}u;
  for (var b = 0u; b < ${K / 4}u; b++) {
    let wv = i8x4(W[wo + b]);
    for (var t = 0u; t < ${R}u; t++) { acc[t] += dot(wv, X[rr[t] + b]); }
  }
  let sc = S[sr];` : fmt === "int4g" ? /* wgsl */ `
  let wo = wr * ${K / 8}u;
  let so = sr * ${K / G}u;
  for (var gi = 0u; gi < ${K / G}u; gi++) {
    var part: array<f32, ${R}>;
    for (var t = 0u; t < ${R}u; t++) { part[t] = 0.0; }
    for (var j = 0u; j < ${G / 8}u; j++) {
      let w = W[wo + gi * ${G / 8}u + j];
      let lo = nib4(w, vec4<u32>(0u, 4u, 8u, 12u));
      let hi = nib4(w, vec4<u32>(16u, 20u, 24u, 28u));
      let xb = gi * ${G / 4}u + 2u * j;
      for (var t = 0u; t < ${R}u; t++) { part[t] += dot(lo, X[rr[t] + xb]) + dot(hi, X[rr[t] + xb + 1u]); }
    }
    let s = S[so + gi];
    for (var t = 0u; t < ${R}u; t++) { acc[t] += part[t] * s; }
  }
  let sc = 1.0;` : /* wgsl */ `
  let wo = wr * ${K / 4}u;
  for (var b = 0u; b < ${K / 4}u; b++) {
    let wv = W[wo + b];
    for (var t = 0u; t < ${R}u; t++) { acc[t] += dot(wv, X[rr[t] + b]); }
  }
  let sc = 1.0;`;
  return HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<${fmt === "f32" ? "vec4<f32>" : "u32"}>;
${fmt === "f32" ? "" : "@group(0) @binding(3) var<storage, read> S: array<f32>;"}
@group(0) @binding(${sb}) var<storage, read> B: array<f32>;
@group(0) @binding(${sb + 1}) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(${sb + 2}) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let N = p.c.z; let T = p.a.x; let fl = p.a.w;
  let n = i % N;
  let r0 = (i / N) * ${R}u;
  var rr: array<u32, ${R}>;
  for (var t = 0u; t < ${R}u; t++) { rr[t] = (p.a.y + min(r0 + t, T - 1u)) * ${K / 4}u; }
  var acc: array<f32, ${R}>;
  let wr = p.b.x + n; let sr = p.b.y + n;
  ${body}
  let bias = B[sr];
  for (var t = 0u; t < ${R}u; t++) {
    let r = r0 + t;
    if (r < T) {
      var y = acc[t] * sc + bias;
      if ((fl & 1u) != 0u) { y = gelu(y); }
      var row = p.a.z + r;
      if ((fl & 4u) != 0u) { row += d.n.x; }
      let yi = row * p.b.w + p.b.z + n;
      if ((fl & 2u) != 0u) { Y[yi] = Y[yi] + y; } else { Y[yi] = y; }
    }
  }
}`;
};

// LayerNorm over rows. a = (T, D, x row0, y row0)  f.x = eps. `inplace` normalizes Y's own rows.
const normWGSL = (inplace) => HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
${inplace ? "" : "@group(0) @binding(4) var<storage, read> X: array<f32>;"}
@group(0) @binding(${inplace ? 4 : 5}) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x + g.y * p.c.x;
  if (t >= p.c.y) { return; }
  let Dm = p.a.y;
  let yo = (p.a.w + t) * Dm;
  let xo = ${inplace ? "yo" : "(p.a.z + t) * Dm"};
  var mu = 0.0;
  for (var k = 0u; k < Dm; k++) { mu += ${inplace ? "Y" : "X"}[xo + k]; }
  mu = mu / f32(Dm);
  var v = 0.0;
  for (var k = 0u; k < Dm; k++) { let e = ${inplace ? "Y" : "X"}[xo + k] - mu; v += e * e; }
  let r = inverseSqrt(v / f32(Dm) + p.f.x);
  for (var k = 0u; k < Dm; k++) { Y[yo + k] = (${inplace ? "Y" : "X"}[xo + k] - mu) * r * G[k] + B[k]; }
  _ = d.n.x;
}`;

// Encoder self-attention without a score matrix. One thread per (block of ${QB} queries, head) walks
// every key once, keeping a running max, normalizer and weighted value sum per query (the softmax is
// rescaled whenever a larger score arrives). H, head dim and width are constants.
// a = (queries in this dispatch, first query row, Tk, 0)
const attnWGSL = (H, hd, Dm) => {
  const HV = hd / 4;
  return HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> KK: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> O: array<vec4<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let h = i % ${H}u;
  let b0 = (i / ${H}u) * ${QB}u;
  let Tq = p.a.x; let Tk = p.a.z;
  var q: array<vec4<f32>, ${QB * HV}>;
  var acc: array<vec4<f32>, ${QB * HV}>;
  var m: array<f32, ${QB}>;
  var z: array<f32, ${QB}>;
  for (var b = 0u; b < ${QB}u; b++) {
    let qo = (p.a.y + min(b0 + b, Tq - 1u)) * ${Dm / 4}u + h * ${HV}u;
    for (var e = 0u; e < ${HV}u; e++) { q[b * ${HV}u + e] = Q[qo + e] * ${1 / Math.sqrt(hd)}; }
    m[b] = -1e30;
  }
  var kv: array<vec4<f32>, ${HV}>;
  var vv: array<vec4<f32>, ${HV}>;
  for (var j = 0u; j < Tk; j++) {
    let ko = j * ${Dm / 4}u + h * ${HV}u;
    for (var e = 0u; e < ${HV}u; e++) { kv[e] = KK[ko + e]; vv[e] = V[ko + e]; }
    for (var b = 0u; b < ${QB}u; b++) {
      let bo = b * ${HV}u;
      var s = 0.0;
      for (var e = 0u; e < ${HV}u; e++) { s += dot(q[bo + e], kv[e]); }
      if (s > m[b]) {
        let c = exp(m[b] - s);
        z[b] *= c;
        for (var e = 0u; e < ${HV}u; e++) { acc[bo + e] *= c; }
        m[b] = s;
      }
      let w = exp(s - m[b]);
      z[b] += w;
      for (var e = 0u; e < ${HV}u; e++) { acc[bo + e] += w * vv[e]; }
    }
  }
  for (var b = 0u; b < ${QB}u; b++) {
    if (b0 + b < Tq) {
      let oo = (p.a.y + b0 + b) * ${Dm / 4}u + h * ${HV}u;
      let inv = 1.0 / z[b];
      for (var e = 0u; e < ${HV}u; e++) { O[oo + e] = acc[b * ${HV}u + e] * inv; }
    }
  }
  _ = d.n.x;
}`;
};

// Tiled matmul (draft by Gemini, checked against the row kernel above): a workgroup of 8 x 8 threads
// computes an (8 RB) x (8 CB) block of Y. Per tile of KT inputs it loads the X rows and the dequantized
// weight columns into workgroup memory once, and every thread accumulates its RB x CB block from there,
// so global memory is read once per block instead of once per output.
// Same uniforms as linearWGSL; dispatch (ceil(N / (8 CB)), ceil(T / (8 RB))).
const linearTiledWGSL = (fmt, K, G, KT, RB, CB) => HEAD + `
${fmt === 'f32' ? `
@group(0) @binding(2) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
` : `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
`}

var<workgroup> As: array<vec4<f32>, ${8 * RB * (KT / 4)}>;
var<workgroup> Bs: array<vec4<f32>, ${8 * CB * (KT / 4)}>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  _ = d.n.x;
  let T = p.a.x;
  let xRow0 = p.a.y;
  let yRow0 = p.a.z;
  let flags = p.a.w;
  let wRow0 = p.b.x;
  let sRow0 = p.b.y;
  let yCol0 = p.b.z;
  let Ny = p.b.w;
  let N = p.c.z;

  let lx = lid.x;
  let ly = lid.y;
  let li = ly * 8u + lx;
  let wx = wid.x;
  let wy = wid.y;

  let rBase = wy * ${8 * RB}u;
  let cBase = wx * ${8 * CB}u;
  let r0 = rBase + ly * ${RB}u;
  let c0 = cBase + lx * ${CB}u;

  var acc: array<f32, ${RB * CB}>;
  for (var idx = 0u; idx < ${RB * CB}u; idx = idx + 1u) {
    acc[idx] = 0.0;
  }

  for (var k0 = 0u; k0 < ${K}u; k0 = k0 + ${KT}u) {
    for (var idx = li; idx < ${8 * RB * (KT / 4)}u; idx = idx + 64u) {
      let tr = idx / ${KT / 4}u;
      let q = idx % ${KT / 4}u;
      let r = min(rBase + tr, T - 1u);
      As[idx] = X[(xRow0 + r) * ${K / 4}u + (k0 / 4u) + q];
    }

${fmt === 'int4g' ? `
    for (var idx = li; idx < ${8 * CB * (KT / 8)}u; idx = idx + 64u) {
      let tc = idx / ${KT / 8}u;
      let q2 = idx % ${KT / 8}u;
      let n = min(cBase + tc, N - 1u);
      let wr = wRow0 + n;
      let sr = sRow0 + n;
      let k = k0 + q2 * 8u;
      let w = W[wr * ${K / 8}u + (k / 8u)];
      let sc = S[sr * ${K / G}u + (k / ${G}u)];
      let bIdx = tc * ${KT / 4}u + q2 * 2u;
      Bs[bIdx] = nib4(w, vec4<u32>(0u, 4u, 8u, 12u)) * sc;
      Bs[bIdx + 1u] = nib4(w, vec4<u32>(16u, 20u, 24u, 28u)) * sc;
    }
` : fmt === 'int8' ? `
    for (var idx = li; idx < ${8 * CB * (KT / 4)}u; idx = idx + 64u) {
      let tc = idx / ${KT / 4}u;
      let q = idx % ${KT / 4}u;
      let n = min(cBase + tc, N - 1u);
      let wr = wRow0 + n;
      let w = W[wr * ${K / 4}u + (k0 / 4u) + q];
      Bs[idx] = i8x4(w);
    }
` : `
    for (var idx = li; idx < ${8 * CB * (KT / 4)}u; idx = idx + 64u) {
      let tc = idx / ${KT / 4}u;
      let q = idx % ${KT / 4}u;
      let n = min(cBase + tc, N - 1u);
      let wr = wRow0 + n;
      Bs[idx] = W[wr * ${K / 4}u + (k0 / 4u) + q];
    }
`}

    workgroupBarrier();

    var a: array<vec4<f32>, ${RB}>;
    var b: array<vec4<f32>, ${CB}>;
    for (var q = 0u; q < ${KT / 4}u; q = q + 1u) {
      for (var i = 0u; i < ${RB}u; i = i + 1u) {
        a[i] = As[(ly * ${RB}u + i) * ${KT / 4}u + q];
      }
      for (var j = 0u; j < ${CB}u; j = j + 1u) {
        b[j] = Bs[(lx * ${CB}u + j) * ${KT / 4}u + q];
      }
      for (var i = 0u; i < ${RB}u; i = i + 1u) {
        for (var j = 0u; j < ${CB}u; j = j + 1u) {
          acc[i * ${CB}u + j] = acc[i * ${CB}u + j] + dot(a[i], b[j]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var i = 0u; i < ${RB}u; i = i + 1u) {
    let r = r0 + i;
    if (r < T) {
      for (var j = 0u; j < ${CB}u; j = j + 1u) {
        let n = c0 + j;
        if (n < N) {
          let sr = sRow0 + n;
          let sc = ${fmt === 'int8' ? 'S[sr]' : '1.0'};
          var y = acc[i * ${CB}u + j] * sc + B[sr];
          if ((flags & 1u) != 0u) {
            y = gelu(y);
          }
          let yIdx = (yRow0 + r) * Ny + yCol0 + n;
          if ((flags & 2u) != 0u) {
            Y[yIdx] = Y[yIdx] + y;
          } else {
            Y[yIdx] = y;
          }
        }
      }
    }
  }
}
`;

// Tiled attention (draft by Gemini, checked against attnWGSL): 64 threads of a workgroup share one head
// and stream tiles of KT keys (K and V) through workgroup memory; each thread owns QB queries and keeps
// the same running softmax as attnWGSL. a = (Tq, first query row, Tk, 0); dispatch (H, ceil(Tq / (64 QB))).
const attnTiledWGSL = (H, hd, Dm, QB, KT) => HEAD + `
const HV: u32 = ${hd >>> 2}u;
const DM4: u32 = ${Dm >>> 2}u;
const QB_C: u32 = ${QB}u;
const KT_C: u32 = ${KT}u;
const TILE_SZ: u32 = ${KT * (hd >>> 2)}u;
const SCALE: f32 = ${(1.0 / Math.sqrt(hd)).toExponential(9)};

@group(0) @binding(2) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> KK: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> O: array<vec4<f32>>;

var<workgroup> Ks: array<vec4<f32>, ${KT * (hd >>> 2)}>;
var<workgroup> Vs: array<vec4<f32>, ${KT * (hd >>> 2)}>;

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  _ = d.n.x;
  let li = lid.x;
  let h = wid.x;
  let qTile = wid.y;

  let Tq = p.a.x;
  let qRow0 = p.a.y;
  let Tk = p.a.z;
  let hBase = h * HV;

  var qLoc: array<vec4<f32>, ${QB * (hd >>> 2)}>;
  var acc: array<vec4<f32>, ${QB * (hd >>> 2)}>;
  var m: array<f32, ${QB}>;
  var z: array<f32, ${QB}>;
  var kLoc: array<vec4<f32>, ${hd >>> 2}>;
  var vLoc: array<vec4<f32>, ${hd >>> 2}>;

  for (var b: u32 = 0u; b < QB_C; b = b + 1u) {
    let qi = qTile * 64u * QB_C + li * QB_C + b;
    let qClamped = min(qi, select(0u, Tq - 1u, Tq > 0u));
    let qBase = (qRow0 + qClamped) * DM4 + hBase;
    m[b] = -1e30;
    z[b] = 0.0;
    for (var e: u32 = 0u; e < HV; e = e + 1u) {
      qLoc[b * HV + e] = Q[qBase + e] * SCALE;
      acc[b * HV + e] = vec4<f32>(0.0);
    }
  }

  for (var k0: u32 = 0u; k0 < Tk; k0 = k0 + KT_C) {
    for (var idx: u32 = li; idx < TILE_SZ; idx = idx + 64u) {
      let j = idx / HV;
      let e = idx % HV;
      let kRow = k0 + j;
      if (kRow < Tk) {
        let kvBase = kRow * DM4 + hBase + e;
        Ks[idx] = KK[kvBase];
        Vs[idx] = V[kvBase];
      } else {
        Ks[idx] = vec4<f32>(0.0);
        Vs[idx] = vec4<f32>(0.0);
      }
    }

    workgroupBarrier();

    let jLim = min(KT_C, Tk - k0);
    for (var j: u32 = 0u; j < jLim; j = j + 1u) {
      let sBase = j * HV;
      for (var e: u32 = 0u; e < HV; e = e + 1u) {
        kLoc[e] = Ks[sBase + e];
        vLoc[e] = Vs[sBase + e];
      }

      for (var b: u32 = 0u; b < QB_C; b = b + 1u) {
        var s: f32 = 0.0;
        let qbBase = b * HV;
        for (var e: u32 = 0u; e < HV; e = e + 1u) {
          s = s + dot(qLoc[qbBase + e], kLoc[e]);
        }

        let mOld = m[b];
        if (s > mOld) {
          let c = exp(mOld - s);
          z[b] = z[b] * c;
          for (var e: u32 = 0u; e < HV; e = e + 1u) {
            acc[qbBase + e] = acc[qbBase + e] * c;
          }
          m[b] = s;
        }
        let w = exp(s - m[b]);
        z[b] = z[b] + w;
        for (var e: u32 = 0u; e < HV; e = e + 1u) {
          acc[qbBase + e] = acc[qbBase + e] + w * vLoc[e];
        }
      }
    }

    workgroupBarrier();
  }

  for (var b: u32 = 0u; b < QB_C; b = b + 1u) {
    let qi = qTile * 64u * QB_C + li * QB_C + b;
    if (qi < Tq) {
      let invZ = 1.0 / z[b];
      let oBase = (qRow0 + qi) * DM4 + hBase;
      let qbBase = b * HV;
      for (var e: u32 = 0u; e < HV; e = e + 1u) {
        O[oBase + e] = acc[qbBase + e] * invZ;
      }
    }
  }
}
`;

// The two tiled kernels with their inner loops written out (named scalars instead of indexed local
// arrays), so the accumulators can live in registers whatever the backend's loop unroller does.
const seq = (n, f, sep = "\n") => Array.from({ length: n }, (_, i) => f(i)).join(sep);
const linearTiledUnrolledWGSL = (fmt, K, G, KT, RB, CB) => {
  const KQ = KT / 4;
  const src = linearTiledWGSL(fmt, K, G, KT, RB, CB).slice(HEAD.length);
  const head = src.slice(0, src.indexOf("  var acc: array<f32"));
  const load = src.slice(src.indexOf("  for (var k0 = 0u;"), src.indexOf("    var a: array<vec4<f32>"));
  return HEAD + head + seq(RB * CB, (i) => `  var s${i} = 0.0;`) + "\n" + load + `
    for (var q = 0u; q < ${KQ}u; q = q + 1u) {
${seq(RB, (i) => `      let a${i} = As[(ly * ${RB}u + ${i}u) * ${KQ}u + q];`)}
${seq(CB, (j) => `      let b${j} = Bs[(lx * ${CB}u + ${j}u) * ${KQ}u + q];`)}
${seq(RB * CB, (i) => `      s${i} += dot(a${Math.floor(i / CB)}, b${i % CB});`)}
    }
    workgroupBarrier();
  }
  var acc: array<f32, ${RB * CB}>;
${seq(RB * CB, (i) => `  acc[${i}] = s${i};`)}
` + src.slice(src.indexOf("  for (var i = 0u; i < " + RB + "u; i = i + 1u) {\n    let r = r0 + i;"));
};

const attnTiledUnrolledWGSL = (H, hd, Dm, QB, KT, direct) => {
  const HV = hd / 4;
  const kk = (e) => (direct ? `Ks[sb + ${e}u]` : `k${e}`), vv = (e) => (direct ? `Vs[sb + ${e}u]` : `v${e}`);   // direct: no local copy of the key
  return HEAD + `
@group(0) @binding(2) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> KK: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> O: array<vec4<f32>>;
var<workgroup> Ks: array<vec4<f32>, ${KT * HV}>;
var<workgroup> Vs: array<vec4<f32>, ${KT * HV}>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  _ = d.n.x;
  let li = lid.x;
  let Tq = p.a.x; let qRow0 = p.a.y; let Tk = p.a.z;
  let hBase = wid.x * ${HV}u;
  let qi0 = wid.y * ${64 * QB}u + li * ${QB}u;
${seq(QB, (b) => `  let qb${b} = (qRow0 + min(qi0 + ${b}u, Tq - 1u)) * ${Dm / 4}u + hBase;
  var m${b} = -1e30; var z${b} = 0.0;
${seq(HV, (e) => `  let q${b}_${e} = Q[qb${b} + ${e}u] * ${1 / Math.sqrt(hd)}; var a${b}_${e} = vec4<f32>(0.0);`)}`)}
  for (var k0 = 0u; k0 < Tk; k0 = k0 + ${KT}u) {
    for (var idx = li; idx < ${KT * HV}u; idx = idx + 64u) {
      let kRow = k0 + idx / ${HV}u;
      if (kRow < Tk) {
        let kvBase = kRow * ${Dm / 4}u + hBase + idx % ${HV}u;
        Ks[idx] = KK[kvBase];
        Vs[idx] = V[kvBase];
      } else {
        Ks[idx] = vec4<f32>(0.0);
        Vs[idx] = vec4<f32>(0.0);
      }
    }
    workgroupBarrier();
    let jLim = min(${KT}u, Tk - k0);
    for (var j = 0u; j < jLim; j = j + 1u) {
      let sb = j * ${HV}u;
${direct ? "" : seq(HV, (e) => `      let k${e} = Ks[sb + ${e}u]; let v${e} = Vs[sb + ${e}u];`)}
${seq(QB, (b) => `      {
        let s = ${seq(HV, (e) => `dot(q${b}_${e}, ${kk(e)})`, " + ")};
        if (s > m${b}) {
          let c = exp(m${b} - s);
          z${b} *= c;
          ${seq(HV, (e) => `a${b}_${e} *= c;`, " ")}
          m${b} = s;
        }
        let w = exp(s - m${b});
        z${b} += w;
        ${seq(HV, (e) => `a${b}_${e} += w * ${vv(e)};`, " ")}
      }`)}
    }
    workgroupBarrier();
  }
${seq(QB, (b) => `  if (qi0 + ${b}u < Tq) {
    let inv = 1.0 / z${b};
    let ob = (qRow0 + qi0 + ${b}u) * ${Dm / 4}u + hBase;
    ${seq(HV, (e) => `O[ob + ${e}u] = a${b}_${e} * inv;`, " ")}
  }`)}
}`;
};

// Attention as three matmul-shaped passes over a chunk of queries (tune.attn = "mat"): scores into a
// buffer [query, head, key], softmax in place, then the weighted sum of values. Each thread keeps only
// a few accumulators, like the row linear kernel. Tk4 = Tk rounded up to 4 is the score row stride.
// a = (queries in this chunk, first query row, Tk, Tk4)
// Scores: one thread per (query, head, 8 keys).
const attnScoreWGSL = (H, hd, Dm) => {
  const HV = hd / 4;
  return HEAD + `
@group(0) @binding(2) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> KK: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> SC: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let Tk = p.a.z;
  let JB = (Tk + 7u) / 8u;
  let jb = i % JB; let rest = i / JB;
  let h = rest % ${H}u; let q = rest / ${H}u;
  let qo = (p.a.y + q) * ${Dm / 4}u + h * ${HV}u;
${seq(HV, (e) => `  let q${e} = Q[qo + ${e}u] * ${1 / Math.sqrt(hd)};`)}
  let so = (q * ${H}u + h) * p.a.w;
  for (var t = 0u; t < 8u; t++) {
    let j = jb * 8u + t;
    if (j < Tk) {
      let ko = j * ${Dm / 4}u + h * ${HV}u;
      SC[so + j] = ${seq(HV, (e) => `dot(q${e}, KK[ko + ${e}u])`, " + ")};
    }
  }
  _ = d.n.x;
}`;
};
// Softmax of each (query, head) row in place, left unnormalized; 1 / sum goes to ZI. Padding keys get 0.
const attnSoftmaxWGSL = (H) => HEAD + `
@group(0) @binding(2) var<storage, read_write> SC: array<f32>;
@group(0) @binding(3) var<storage, read_write> ZI: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let Tk = p.a.z;
  let so = i * p.a.w;
  var m = -1e30;
  for (var j = 0u; j < Tk; j++) { m = max(m, SC[so + j]); }
  var z = 0.0;
  for (var j = 0u; j < Tk; j++) { let w = exp(SC[so + j] - m); SC[so + j] = w; z += w; }
  for (var j = Tk; j < p.a.w; j++) { SC[so + j] = 0.0; }
  ZI[i] = 1.0 / z;
  _ = d.n.x;
}`;
// Mix: one thread per (block of QM queries, head, vec4 of the head's values) walks the keys 4 at a time.
const attnMixWGSL = (H, hd, Dm, QM) => {
  const HV = hd / 4;
  return HEAD + `
@group(0) @binding(2) var<storage, read> SC: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ZI: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> O: array<vec4<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let Tq = p.a.x; let Tk = p.a.z; let T4 = p.a.w / 4u;
  let e = i % ${HV}u; let h = (i / ${HV}u) % ${H}u; let q0 = (i / ${HV * H}u) * ${QM}u;
${seq(QM, (t) => `  let r${t} = (min(q0 + ${t}u, Tq - 1u) * ${H}u + h) * T4; var a${t} = vec4<f32>(0.0);`)}
  let vo = h * ${HV}u + e;
  for (var j4 = 0u; j4 < T4; j4++) {
    let j = j4 * 4u;
    let v0 = V[min(j, Tk - 1u) * ${Dm / 4}u + vo];
    let v1 = V[min(j + 1u, Tk - 1u) * ${Dm / 4}u + vo];
    let v2 = V[min(j + 2u, Tk - 1u) * ${Dm / 4}u + vo];
    let v3 = V[min(j + 3u, Tk - 1u) * ${Dm / 4}u + vo];
${seq(QM, (t) => `    { let w = SC[r${t} + j4]; a${t} += w.x * v0 + w.y * v1 + w.z * v2 + w.w * v3; }`)}
  }
${seq(QM, (t) => `  if (q0 + ${t}u < Tq) { O[(p.a.y + q0 + ${t}u) * ${Dm / 4}u + vo] = a${t} * ZI[(q0 + ${t}u) * ${H}u + h]; }`)}
  _ = d.n.x;
}`;
};

// Neck: rows of 4 neighbouring tokens become one conv2 input row in the weight's (channel, kw) order.
// Y[h * Wn + w, c * 4 + kw] = X[h * Wp + 4 w + kw, c].  a = (C, Wp, Wn, 0); columns past 4 Wn are dropped, as the conv does.
const REGROUP = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let C = p.a.x;
  let t = i / (4u * C); let rem = i % (4u * C);
  let src = (t / p.a.z) * p.a.y + 4u * (t % p.a.z) + (rem & 3u);
  Y[i] = X[src * C + (rem >> 2u)];
  _ = d.n.x;
}`;

// Token embedding: Y[n] = table[id, n] * f.x, id = d.n.y, table part starting at row d.n.z.
const embedWGSL = (fmt, K, G) => HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> TB: array<${fmt === "f32" ? "f32" : "u32"}>;
${fmt === "f32" ? "" : "@group(0) @binding(3) var<storage, read> TS: array<f32>;"}
@group(0) @binding(${fmt === "f32" ? 3 : 4}) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  if (n >= ${K}u) { return; }
  let id = d.n.y; let row = id - d.n.z;
  ${fmt === "int8" ? `let bi = row * ${K}u + n;
  let w = bitcast<i32>(TB[bi / 4u]);
  let v = f32((w << (24u - 8u * (bi % 4u))) >> 24u) * TS[id];`
    : fmt === "int4g" ? `let bi = row * ${K}u + n;
  let v = (f32((TB[bi / 8u] >> (4u * (bi % 8u))) & 15u) - 8.0) * TS[id * ${K / G}u + n / ${G}u];`
      : `let v = TB[row * ${K}u + n];`}
  Y[n] = v * p.f.x;
}`;

// Attention for one query row (the decoder), split so the keys run in parallel (from whisper.js).
// ASCORE: SC[h * Tk + j] = q_h . k_{j,h} / 8, one thread per (key, head).
// AMIX:   O[h * 64 + e] = sum_j softmax(SC[h])_j V[j, h, e], one thread per output value.
// a = (Tk, H, D, flags)  flags: 1 Tk = d.n.x + 1
const ASCORE = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> KK: array<f32>;
@group(0) @binding(4) var<storage, read_write> SC: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  let H = p.a.y; let Dm = p.a.z;
  var Tk = p.a.x;
  if ((p.a.w & 1u) != 0u) { Tk = d.n.x + 1u; }
  if (i >= Tk * H) { return; }
  let j = i / H; let h = i % H;
  let qo = h * 64u; let ko = j * Dm + h * 64u;
  var s = vec4<f32>(0.0);
  for (var e = 0u; e < 64u; e += 4u) {
    s += vec4<f32>(Q[qo + e], Q[qo + e + 1u], Q[qo + e + 2u], Q[qo + e + 3u])
       * vec4<f32>(KK[ko + e], KK[ko + e + 1u], KK[ko + e + 2u], KK[ko + e + 3u]);
  }
  SC[h * Tk + j] = (s.x + s.y + s.z + s.w) * 0.125;
}`;
const AMIX = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> SC: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let H = p.a.y; let Dm = p.a.z;
  if (i >= Dm) { return; }
  var Tk = p.a.x;
  if ((p.a.w & 1u) != 0u) { Tk = d.n.x + 1u; }
  let h = i / 64u;
  let so = h * Tk;
  var mx = -1e30;
  for (var j = 0u; j < Tk; j++) { mx = max(mx, SC[so + j]); }
  var z = 0.0;
  var acc = 0.0;
  for (var j = 0u; j < Tk; j++) {
    let w = exp(SC[so + j] - mx);
    z += w;
    acc += w * V[j * Dm + i];
  }
  O[i] = acc / z + f32(H) * 0.0;
}`;

// Greedy pick with the repetition penalty: the best (value, index) of each chunk of the logits.
// a = (V, chunk size, 0, 0)  f.x = penalty.  SEEN[v] != 0 marks tokens already in the sequence.
const PICK = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> L: array<f32>;
@group(0) @binding(3) var<storage, read> SEEN: array<u32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  if (c >= p.c.y) { return; }
  let v0 = c * p.a.y; let v1 = min(v0 + p.a.y, p.a.x);
  var best = -3.0e38; var bi = v0;
  for (var v = v0; v < v1; v++) {
    var x = L[v];
    if (SEEN[v] != 0u) { if (x < 0.0) { x = x * p.f.x; } else { x = x / p.f.x; } }
    if (x > best) { best = x; bi = v; }
  }
  O[2u * c] = best; O[2u * c + 1u] = f32(bi);
  _ = d.n.x;
}`;

const SHADERS = { regroup: REGROUP, ascore: ASCORE, amix: AMIX, pick: PICK, norm: normWGSL(false), norm_in: normWGSL(true) };

function halfToFloat(u16) {
  if (typeof Float16Array !== "undefined") return new Float32Array(new Float16Array(u16.buffer, u16.byteOffset, u16.length));
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], e = (h >> 10) & 31, f = h & 1023;
    const v = e === 0 ? f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * 2 ** (e - 15);
    out[i] = h & 0x8000 ? -v : v;
  }
  return out;
}

async function gunzip(bytes) {
  return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
}

const aborted = (signal) => { if (signal?.aborted) throw new DOMException("aborted", "AbortError"); };

// PIL's Image.resize(BILINEAR) for a shrink: a triangle filter widened by the scale, rows then
// columns through an 8-bit intermediate. rgba: Uint8ClampedArray [h, w, 4]. Returns RGB Uint8Array.
export function resizeBilinear(rgba, w, h, nw, nh) {
  const coeffs = (inSize, outSize) => {
    const scale = inSize / outSize, fs = Math.max(scale, 1), support = fs;
    const out = [];
    for (let x = 0; x < outSize; x++) {
      const center = (x + 0.5) * scale;
      const lo = Math.max(0, Math.trunc(center - support + 0.5)), hi = Math.min(inSize, Math.trunc(center + support + 0.5));
      const k = new Float64Array(hi - lo);
      let sum = 0;
      for (let i = 0; i < k.length; i++) { const v = Math.max(0, 1 - Math.abs((i + lo - center + 0.5) / fs)); k[i] = v; sum += v; }
      for (let i = 0; i < k.length; i++) k[i] /= sum;
      out.push({ lo, k });
    }
    return out;
  };
  const cx = coeffs(w, nw), cy = coeffs(h, nh);
  const tmp = new Uint8Array(h * nw * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < nw; x++) {
    const { lo, k } = cx[x];
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let i = 0; i < k.length; i++) s += k[i] * rgba[(y * w + lo + i) * 4 + c];
      tmp[(y * nw + x) * 3 + c] = Math.max(0, Math.min(255, Math.round(s)));
    }
  }
  const out = new Uint8Array(nh * nw * 3);
  for (let y = 0; y < nh; y++) {
    const { lo, k } = cy[y];
    for (let x = 0; x < nw; x++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let i = 0; i < k.length; i++) s += k[i] * tmp[((lo + i) * nw + x) * 3 + c];
      out[(y * nw + x) * 3 + c] = Math.max(0, Math.min(255, Math.round(s)));
    }
  }
  return out;
}

export class NemotronParse {
  // opts: base (pack directory URL), device? (a GPUDevice; otherwise one is requested with the adapter's
  // buffer limits), onProgress({phase: "download" | "upload" | "ready", loaded, total}),
  // size? ({height, width} canvas, multiples of 16; default the pack's 2048 x 1664),
  // cache? (false skips the Cache API), budget? (multiply-adds per dispatch, default 4e9).
  static async load(opts = {}) {
    const base = opts.base.endsWith("/") ? opts.base : opts.base + "/";
    let device = opts.device;
    if (!device) {
      if (!navigator.gpu) throw new Error("WebGPU is not available");
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("no WebGPU adapter");
      device = await adapter.requestDevice({ requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
    }
    const manifest = await (await fetch(base + "manifest.json", { cache: "no-cache" })).json();
    if (manifest.format !== "nemotron-pack/1") throw new Error("not a nemotron-pack/1 manifest");
    const m = new NemotronParse(device, manifest, opts);
    m.tok = new Tokenizer(await (await fetch(base + "tokenizer.json")).json());
    await m._loadWeights(base);
    const size = opts.size || manifest.image;
    m.size = { height: size.height, width: size.width };
    opts.onProgress?.({ phase: "ready", loaded: manifest.bytes, total: manifest.bytes });
    return m;
  }

  constructor(device, manifest, opts = {}) {
    this.dev = device;
    this.opts = opts;
    this.m = manifest;
    this.dims = manifest.dims;
    this.gen = manifest.generation;
    this.budget = opts.budget || 4e9;
    // Encoder kernel choice and tile shapes; setTune() changes them (benchmarks, A/B checks).
    this.tune = { lin: "tiled", attn: "tiled", KT: 16, RB: 8, CB: 8, QB: 4, AKT: 16, unroll: false, direct: false, tiledBudget: 3e10, QM: 8, matBytes: 1 << 27, ...(opts.tune || {}) };
    this._tag = "";
    this.maxBind = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    this.pipes = {};
    this.zero = {};
    this.w = {};
    this.cpu = {};
    this.plan = null;
    this.dyn = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  // ---------------------------------------------------------------- weights

  // Fetch the shards (4 at a time, through the Cache API) and write every record into its GPU buffer
  // as the shard holding it arrives; nothing but the shard in hand is kept in JS memory.
  async _loadWeights(base) {
    const dev = this.dev, man = this.m, sinks = [];
    const gpuParts = (rows, rowBytes, label) => {   // one buffer, or several when rows exceed the binding limit
      const per = Math.max(1, Math.min(rows, Math.floor(this.maxBind / rowBytes)));
      const parts = [];
      for (let n0 = 0; n0 < rows; n0 += per) {
        const n = Math.min(per, rows - n0);
        parts.push({ n0, n, buf: dev.createBuffer({ label, size: Math.max(16, Math.ceil(n * rowBytes / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }) });
      }
      return parts;
    };
    const write = (parts, rowBytes, rel, u8) => {   // u8 lands at byte `rel` of the parts' concatenation
      for (const part of parts) {
        const p0 = part.n0 * rowBytes, p1 = p0 + part.n * rowBytes;
        const s0 = Math.max(rel, p0), s1 = Math.min(rel + u8.length, p1);
        if (s1 <= s0) continue;
        let src = u8.subarray(s0 - rel, s1 - rel);
        if (src.length % 4) { const pad = new Uint8Array(Math.ceil(src.length / 4) * 4); pad.set(src); src = pad; }
        dev.queue.writeBuffer(part.buf, s0 - p0, src.buffer, src.byteOffset, src.length);
      }
    };
    const rawSink = (rec, parts, rowBytes) => sinks.push({ off: rec[0], len: rec[1], put: (rel, u8) => write(parts, rowBytes, rel, u8) });
    const halfSink = (rec, parts, rowBytesF32) => sinks.push({ off: rec[0], len: rec[1], put: (rel, u8) => {
      const f = halfToFloat(new Uint16Array(u8.slice().buffer));
      write(parts, rowBytesF32, rel * 2, new Uint8Array(f.buffer));
    } });
    const cpuSink = (rec) => {
      const dst = new Uint8Array(rec[1]);
      sinks.push({ off: rec[0], len: rec[1], put: (rel, u8) => dst.set(u8, rel) });
      return dst;
    };
    for (const [name, t] of Object.entries(man.tensors)) {
      if (/pos_embed|cls_token/.test(name)) {   // used on the CPU (position grid interpolation, prefix rows)
        this.cpu[name] = { t, q: cpuSink(t.q || t.w), s: t.s ? cpuSink(t.s) : null };
      } else if (t.shape.length === 1) {
        const parts = gpuParts(1, t.shape[0] * 4, name);
        halfSink(t.w, parts, t.shape[0] * 4);
        this.w[name] = { B: parts[0].buf, N: t.shape[0] };
      } else {
        const [N, K] = t.shape;
        const w = { fmt: t.kind === "f16" ? "f32" : t.kind, N, K, G: t.group || 0 };
        if (K % 8) throw new Error(name + ": K must be a multiple of 8");
        if (t.kind === "f16") {
          w.parts = gpuParts(N, K * 4, name);
          halfSink(t.w, w.parts, K * 4);
        } else {
          const rowBytes = t.kind === "int8" ? K : K / 2;
          w.parts = gpuParts(N, rowBytes, name);
          rawSink(t.q, w.parts, rowBytes);
          const sn = t.kind === "int8" ? N : N * K / t.group;
          const sp = gpuParts(1, sn * 4, name + ".s");
          halfSink(t.s, sp, sn * 4);
          w.S = sp[0].buf;
        }
        this.w[name] = w;
      }
    }
    sinks.sort((a, b) => a.off - b.off);

    let cache = null;
    if (this.opts.cache !== false) { try { cache = await caches.open("nemotron-pack-v1"); } catch (e) { cache = null; } }
    const starts = [];
    let at = 0;
    for (const s of man.shards) { starts.push(at); at += s.bytes; }
    const total = man.shards.reduce((n, s) => n + (s.gzip || s.bytes), 0);
    const got = new Array(man.shards.length).fill(0);
    const progress = (phase) => this.opts.onProgress?.({ phase, loaded: got.reduce((a, b) => a + b, 0), total });
    const fetchShard = async (i) => {
      const s = man.shards[i], url = base + s.file, key = url + "?v=" + man.id;
      let res = cache ? await cache.match(key).catch(() => null) : null;
      if (!res) {
        res = await fetch(url);
        if (!res.ok) throw new Error(s.file + ": HTTP " + res.status);
        if (cache) cache.put(key, res.clone()).catch(() => {});   // over quota: carry on uncached
      }
      const chunks = [];
      let n = 0;
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        n += value.length;
        got[i] = Math.min(n, (s.gzip || s.bytes) - 1);
        progress("download");
      }
      let bytes = new Uint8Array(n);
      n = 0;
      for (const c of chunks) { bytes.set(c, n); n += c.length; }
      if (bytes.length !== s.bytes && bytes[0] === 0x1f && bytes[1] === 0x8b) bytes = await gunzip(bytes);   // host sent no Content-Encoding
      if (bytes.length !== s.bytes) throw new Error(`${s.file}: ${bytes.length} bytes, expected ${s.bytes}`);
      const a0 = starts[i], a1 = a0 + s.bytes;
      for (const k of sinks) {
        if (k.off >= a1) break;
        const s0 = Math.max(a0, k.off), s1 = Math.min(a1, k.off + k.len);
        if (s1 > s0) k.put(s0 - k.off, bytes.subarray(s0 - a0, s1 - a0));
      }
      got[i] = s.gzip || s.bytes;
      progress("upload");
    };
    let next = 0;
    const worker = async () => { while (next < man.shards.length) await fetchShard(next++); };
    await Promise.all(Array.from({ length: Math.min(this.opts.parallel || 4, man.shards.length) }, worker));
    await dev.queue.onSubmittedWorkDone();

    for (const [name, c] of Object.entries(this.cpu)) {   // dequantize the CPU-side tensors
      const t = c.t, n = t.shape[0] * t.shape[1];
      let f;
      if (t.kind === "f16") f = halfToFloat(new Uint16Array(c.q.buffer, 0, n));
      else {
        const s = halfToFloat(new Uint16Array(c.s.buffer, 0, c.s.length / 2));
        f = new Float32Array(n);
        const [N, K] = t.shape;
        if (t.kind === "int8") {
          const q = new Int8Array(c.q.buffer);
          for (let r = 0; r < N; r++) for (let k = 0; k < K; k++) f[r * K + k] = q[r * K + k] * s[r];
        } else {
          const G = t.group, gs = K / G;
          for (let r = 0; r < N; r++) for (let k = 0; k < K; k++) {
            const i = r * K + k;
            f[i] = (((c.q[i >> 1] >> (4 * (i & 1))) & 15) - 8) * s[r * gs + Math.floor(k / G)];
          }
        }
      }
      this.cpu[name] = f;
    }
  }

  // ---------------------------------------------------------------- recording

  _pipe(kind, code) {
    if (!this.pipes[kind]) {
      const module = this.dev.createShaderModule({ code: code || SHADERS[kind] });
      this.pipes[kind] = this.dev.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    }
    return this.pipes[kind];
  }

  _buf(floats, label, own) {
    const size = Math.max(16, floats * 4);
    if (size > this.maxBind) throw new Error(`${label}: ${size} bytes exceeds this device's storage binding limit (${this.maxBind})`);
    const b = this.dev.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (own) own.push(b);
    return b;
  }

  _zeros(n) {
    if (!this.zero[n]) this.zero[n] = this._buf(n, "zero" + n);
    return this.zero[n];
  }

  // One recorded dispatch. threads = number of threads wanted; params = { a, b, cz, f }.
  _step(list, kind, params, bufs, threads, code, own) {
    const pipe = this._pipe(kind, code);
    const groups = Math.ceil(threads / WG);
    const gx = Math.min(groups, 16384), gy = Math.ceil(groups / gx);
    const u = new ArrayBuffer(64), u32 = new Uint32Array(u), f32 = new Float32Array(u);
    u32.set(params.a || [], 0);
    u32.set(params.b || [], 4);
    u32.set([gx * WG, threads, params.cz || 0, 0], 8);
    const [dx, dy] = params.dispatch || [gx, gy];
    f32.set(params.f || [], 12);
    const ub = this.dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint8Array(ub.getMappedRange()).set(new Uint8Array(u));
    ub.unmap();
    if (own) own.push(ub);
    const entries = [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: this.dyn } },
      ...bufs.map((buffer, i) => ({ binding: i + 2, resource: { buffer } }))];
    list.push({ pipe, group: this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries }), gx: dx, gy: dy, tag: this._tag });
  }

  // Change the encoder kernels ({lin, attn: "tiled" | "row", KT, RB, CB, QB, AKT, tiledBudget}); the next encode re-plans.
  setTune(t) {
    this.tune = { ...this.tune, ...t };
    if (this.plan) this.plan.stale = true;
  }

  // Y[yRow0 + r, yCol0 + n] = W[n0 + n] . X[xRow0 + r] + bias, r < T, n < N; cut into dispatches of
  // at most `budget` multiply-adds and one per weight buffer part.
  _linear(list, w, X, Y, T, o = {}, own) {
    const { flags = 0, xRow0 = 0, yRow0 = 0, n0 = 0, yCol0 = 0 } = o;
    const N = o.N ?? w.N, K = w.K, Ny = o.Ny ?? N;
    const R = T >= 8 ? 8 : 1;
    const B = o.bias || this._zeros(w.N);
    const tn = this.tune;
    if (tn.lin === "tiled" && T >= 8 && !(flags & 4) && K % tn.KT === 0 && tn.KT % 8 === 0 && (w.fmt !== "int4g" || w.G % 8 === 0)
        && (8 * tn.RB + 8 * tn.CB) * tn.KT * 4 <= 16384) {
      const { KT, RB, CB } = tn, tr = 8 * RB, tc = 8 * CB;
      const kind = `lint_${w.fmt}_${K}_${w.G}_${KT}_${RB}_${CB}_${tn.unroll ? 1 : 0}`;
      const code = this.pipes[kind] ? null : (tn.unroll ? linearTiledUnrolledWGSL : linearTiledWGSL)(w.fmt, K, w.G, KT, RB, CB);
      for (const part of w.parts) {
        const a0 = Math.max(n0, part.n0), a1 = Math.min(n0 + N, part.n0 + part.n);
        if (a1 <= a0) continue;
        const rows = Math.max(tr, Math.floor(tn.tiledBudget / ((a1 - a0) * K) / tr) * tr);
        for (let r = 0; r < T; r += rows) {
          const t = Math.min(rows, T - r);
          this._step(list, kind, { a: [t, xRow0 + r, yRow0 + r, flags], b: [a0 - part.n0, a0, yCol0 + a0 - n0, Ny], cz: a1 - a0,
            dispatch: [Math.ceil((a1 - a0) / tc), Math.ceil(t / tr)] },
            w.fmt === "f32" ? [part.buf, B, X, Y] : [part.buf, w.S, B, X, Y], 1, code, own);
        }
      }
      return;
    }
    const kind = `lin_${w.fmt}_${K}_${w.G}_${R}`, code = this.pipes[kind] ? null : linearWGSL(w.fmt, K, w.G, R);
    for (const part of w.parts) {
      const a0 = Math.max(n0, part.n0), a1 = Math.min(n0 + N, part.n0 + part.n);
      if (a1 <= a0) continue;
      const rows = Math.max(R, Math.floor(this.budget / ((a1 - a0) * K) / R) * R);
      for (let r = 0; r < T; r += rows) {
        const t = Math.min(rows, T - r);
        this._step(list, kind, { a: [t, xRow0 + r, yRow0 + r, flags], b: [a0 - part.n0, a0, yCol0 + a0 - n0, Ny], cz: a1 - a0 },
          w.fmt === "f32" ? [part.buf, B, X, Y] : [part.buf, w.S, B, X, Y], Math.ceil(t / R) * (a1 - a0), code, own);
      }
    }
  }

  _norm(list, n, X, Y, T, Dm, eps, xRow0 = 0, yRow0 = 0, own) {
    if (X === Y) this._step(list, "norm_in", { a: [T, Dm, yRow0, yRow0], f: [eps] }, [n.G, n.B, Y], T, null, own);
    else this._step(list, "norm", { a: [T, Dm, xRow0, yRow0], f: [eps] }, [n.G, n.B, X, Y], T, null, own);
  }

  // ---------------------------------------------------------------- encoder plan

  // Buffers and steps for one canvas size. Called by parse(); a different size replaces the plan.
  _plan(height, width) {
    if (this.plan && !this.plan.stale && this.plan.height === height && this.plan.width === width) return this.plan;
    const dm = this.dims, P = dm.patch;
    if (height % P || width % P || width < 4 * P || height > dm.pos_grid * P || width > dm.pos_grid * P) {
      throw new Error(`canvas ${height}x${width}: both sides must be multiples of ${P}, at most ${dm.pos_grid * P}, width at least ${4 * P}`);
    }
    if (this.plan) for (const b of this.plan.own) b.destroy();
    const own = [];
    const Hp = height / P, Wp = width / P, Tp = Hp * Wp, PRE = dm.enc_prefix, T = Tp + PRE, Wn = Math.floor(Wp / 4), Tn = Hp * Wn, Te = Tn + 1;
    const E = dm.enc_width, EH = dm.enc_heads, F = dm.enc_mlp, D = dm.d_model, PK = 3 * P * P;
    const W = (n) => { const w = this.w[n]; if (!w) throw new Error("pack has no tensor " + n); return w; };
    const R = "encoder.model_encoder.radio_model.model.";
    const b = {};
    b.patches = this._buf(Tp * PK, "patches", own);
    b.x0 = this._buf(T * E, "x0", own);
    for (const k of ["x", "xn", "q", "k", "v"]) b[k] = this._buf(T * E, k, own);
    const mlpRows = Math.max(8, Math.min(Math.ceil(T / 8) * 8, Math.floor(Math.min(this.maxBind, 1 << 27) / (F * 4) / 8) * 8));
    b.h = this._buf(mlpRows * F, "h", own);
    b.c1 = this._buf(Tp * D, "c1", own);
    b.c2 = this._buf(Tn * 4 * D, "c2", own);
    b.enc = this._buf(Te * D, "enc", own);

    // prefix tokens and the position grid: bilinear (align_corners) to max(Hp, Wp) squared, then the top-left window
    const x0 = new Float32Array(T * E);
    x0.set(this.cpu[R + "patch_generator.cls_token.token"].subarray(0, PRE * E));
    const pos = this.cpu[R + "patch_generator.pos_embed"], Gd = dm.pos_grid, M = Math.max(Hp, Wp);
    const sc = M > 1 ? (Gd - 1) / (M - 1) : 0;
    for (let i = 0; i < Hp; i++) {
      const sy = i * sc, y0 = Math.min(Math.floor(sy), Gd - 1), y1 = Math.min(y0 + 1, Gd - 1), fy = sy - y0;
      for (let j = 0; j < Wp; j++) {
        const sx = j * sc, xa = Math.min(Math.floor(sx), Gd - 1), xb = Math.min(xa + 1, Gd - 1), fx = sx - xa;
        const o = (PRE + i * Wp + j) * E;
        const p00 = (y0 * Gd + xa) * E, p01 = (y0 * Gd + xb) * E, p10 = (y1 * Gd + xa) * E, p11 = (y1 * Gd + xb) * E;
        if (fy === 0 && fx === 0) { x0.set(pos.subarray(p00, p00 + E), o); continue; }
        const w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx, w10 = fy * (1 - fx), w11 = fy * fx;
        for (let c = 0; c < E; c++) x0[o + c] = w00 * pos[p00 + c] + w01 * pos[p01 + c] + w10 * pos[p10 + c] + w11 * pos[p11 + c];
      }
    }
    this.dev.queue.writeBuffer(b.x0, 0, x0);

    const stages = [];   // each a list of steps, awaited as a unit
    let s = [{ copy: [b.x0, b.x, T * E * 4], tag: "embed" }];
    this._tag = "embed";
    this._linear(s, W(R + "patch_generator.embedder.weight"), b.patches, b.x, Tp, { flags: 2, yRow0: PRE }, own);
    stages.push(s);
    const attnKind = `attn_${EH}_${E / EH}_${E}`, tn = this.tune, hd = E / EH;
    const T4 = Math.ceil(T / 4) * 4;
    const matQ = Math.max(1, Math.min(T, Math.floor(Math.min(this.maxBind, tn.matBytes) / (EH * T4 * 4))));
    if (tn.attn === "mat") { b.sc = this._buf(matQ * EH * T4, "sc", own); b.zi = this._buf(matQ * EH, "zi", own); }
    const qPer = Math.max(QB, Math.floor(this.budget / (T * EH * (E / EH) * 2) / QB) * QB);
    for (let l = 0; l < dm.enc_layers; l++) {
      const L = `${R}blocks.${l}.`;
      const n1 = { G: W(L + "norm1.weight").B, B: W(L + "norm1.bias").B }, n2 = { G: W(L + "norm2.weight").B, B: W(L + "norm2.bias").B };
      const qkv = W(L + "attn.qkv.weight"), qb = W(L + "attn.qkv.bias").B;
      s = [];
      this._tag = "norm";
      this._norm(s, n1, b.x, b.xn, T, E, dm.enc_eps, 0, 0, own);
      this._tag = "qkv";
      [b.q, b.k, b.v].forEach((Y, j) => this._linear(s, qkv, b.xn, Y, T, { n0: j * E, N: E, bias: qb }, own));
      this._tag = "attn";
      if (tn.attn === "mat") {
        for (let r = 0; r < T; r += matQ) {
          const t = Math.min(matQ, T - r), a = [t, r, T, T4];
          this._step(s, `asc_${EH}_${hd}_${E}`, { a }, [b.q, b.k, b.sc], t * EH * Math.ceil(T / 8), attnScoreWGSL(EH, hd, E), own);
          this._step(s, `asm_${EH}`, { a }, [b.sc, b.zi], t * EH, attnSoftmaxWGSL(EH), own);
          this._step(s, `amx_${EH}_${hd}_${E}_${tn.QM}`, { a }, [b.sc, b.zi, b.v, b.xn], Math.ceil(t / tn.QM) * EH * (hd / 4), attnMixWGSL(EH, hd, E, tn.QM), own);
        }
      } else if (tn.attn === "tiled" && 2 * tn.AKT * hd * 4 <= 16384) {
        const per = 64 * tn.QB, kind = `attnt_${EH}_${hd}_${E}_${tn.QB}_${tn.AKT}_${tn.unroll ? 1 : 0}_${tn.direct ? 1 : 0}`;
        const qt = Math.max(per, Math.floor(tn.tiledBudget / (T * EH * hd * 2) / per) * per);
        for (let r = 0; r < T; r += qt) {
          const t = Math.min(qt, T - r);
          this._step(s, kind, { a: [t, r, T, 0], dispatch: [EH, Math.ceil(t / per)] }, [b.q, b.k, b.v, b.xn], 1,
            this.pipes[kind] ? null : (tn.unroll ? attnTiledUnrolledWGSL : attnTiledWGSL)(EH, hd, E, tn.QB, tn.AKT, tn.direct), own);
        }
      } else {
        for (let r = 0; r < T; r += qPer) {
          const t = Math.min(qPer, T - r);
          this._step(s, attnKind, { a: [t, r, T, 0] }, [b.q, b.k, b.v, b.xn], Math.ceil(t / QB) * EH, attnWGSL(EH, E / EH, E), own);
        }
      }
      this._tag = "proj";
      this._linear(s, W(L + "attn.proj.weight"), b.xn, b.x, T, { flags: 2, bias: W(L + "attn.proj.bias").B }, own);
      this._tag = "norm";
      this._norm(s, n2, b.x, b.xn, T, E, dm.enc_eps, 0, 0, own);
      for (let r = 0; r < T; r += mlpRows) {
        const t = Math.min(mlpRows, T - r);
        this._tag = "fc1";
        this._linear(s, W(L + "mlp.fc1.weight"), b.xn, b.h, t, { flags: 1, xRow0: r, bias: W(L + "mlp.fc1.bias").B }, own);
        this._tag = "fc2";
        this._linear(s, W(L + "mlp.fc2.weight"), b.h, b.x, t, { flags: 2, yRow0: r, bias: W(L + "mlp.fc2.bias").B }, own);
      }
      stages.push(s);
    }
    this._tag = "neck";
    // neck: conv1 (1x1) -> LN -> conv2 (1x4 stride 4) -> LN; the three summary tokens -> sum_proj -> LN, appended
    s = [];
    const ln = (k) => ({ G: W(`encoder.${k}.weight`).B, B: W(`encoder.${k}.bias`).B });
    this._linear(s, W("encoder.conv1.weight"), b.x, b.c1, Tp, { xRow0: PRE, bias: W("encoder.conv1.bias").B }, own);
    this._norm(s, ln("layer_norm1"), b.c1, b.c1, Tp, D, dm.neck_eps, 0, 0, own);
    this._step(s, "regroup", { a: [D, Wp, Wn, 0] }, [b.c1, b.c2], Tn * 4 * D, null, own);
    this._linear(s, W("encoder.conv2.weight"), b.c2, b.enc, Tn, {}, own);
    this._norm(s, ln("layer_norm2"), b.enc, b.enc, Tn, D, dm.neck_eps, 0, 0, own);
    this._linear(s, W("encoder.sum_proj.weight"), b.x, b.enc, 1, { yRow0: Tn, bias: W("encoder.sum_proj.bias").B }, own);
    this._norm(s, ln("layer_norm3"), b.enc, b.enc, 1, D, dm.neck_eps, 0, Tn, own);
    // cross-attention keys and values for every decoder layer, once per page
    const cross = [];
    for (let l = 0; l < dm.dec_layers; l++) {
      const L = `decoder.layers.${l}.encoder_attn.`;
      const xk = this._buf(Te * D, `d${l}.xk`, own), xv = this._buf(Te * D, `d${l}.xv`, own);
      this._linear(s, W(L + "k_proj.weight"), b.enc, xk, Te, { bias: W(L + "k_proj.bias").B }, own);
      this._linear(s, W(L + "v_proj.weight"), b.enc, xv, Te, { bias: W(L + "v_proj.bias").B }, own);
      cross.push({ xk, xv });
    }
    stages.push(s);
    this._tag = "";
    this.plan = { height, width, Hp, Wp, Tp, T, Te, b, stages, cross, own, PK };
    this.b = b;
    if (this.dec) {   // decoder steps bind the cross buffers: rebuild
      for (const x of [...this.dec.own, ...this.dec.read, this.dec.b.logits, this.dec.b.seen, this.dec.b.pick]) x.destroy();
      this.dec = null;
    }
    return this.plan;
  }

  // ---------------------------------------------------------------- decoder steps

  // Record the per-token steps with a self-attention cache of `cap` rows (grown by doubling).
  _buildDec(cap) {
    const dm = this.dims, D = dm.d_model, H = dm.dec_heads, F = dm.dec_ffn, Te = this.plan.Te;
    const V = this.w["decoder.embed_tokens.weight"].N;   // table rows (a few more than the tokenizer has tokens)
    if (D / H !== 64) throw new Error("decoder head dim must be 64");
    const W = (n) => this.w[n];
    const old = this.dec, own = [];
    const b = {};
    for (const k of ["de", "dx", "dn", "dq", "da"]) b[k] = this._buf(D, k, own);
    b.dh = this._buf(F, "dh", own);
    b.sc = this._buf(H * Math.max(cap, Te), "sc", own);
    b.logits = old ? old.b.logits : this._buf(V, "logits");
    b.seen = old ? old.b.seen : this._buf(V, "seen");
    const CH = 512, chunks = Math.ceil(V / CH);
    b.pick = old ? old.b.pick : this._buf(chunks * 2, "pick");
    const table = W("decoder.embed_tokens.weight");
    this.vocab = V;
    const embed = table.parts.map((part) => {
      const st = [];
      this._step(st, `embed_${table.fmt}_${D}_${table.G}`, { f: [dm.embed_scale] }, table.fmt === "f32" ? [part.buf, b.de] : [part.buf, table.S, b.de], D,
        embedWGSL(table.fmt, D, table.G), own);
      return { n0: part.n0, n: part.n, step: st[0] };
    });
    const ln = (k) => ({ G: W(k + ".weight").B, B: W(k + ".bias").B });
    const copies = [];
    const s = [];
    this._norm(s, ln("decoder.layernorm_embedding"), b.de, b.dx, 1, D, dm.dec_eps, 0, 0, own);
    const attn1 = (K, Vb, Tk, flags) => {
      this._step(s, "ascore", { a: [Tk, H, D, flags] }, [b.dq, K, b.sc], Math.max(Tk, cap) * H, null, own);
      this._step(s, "amix", { a: [Tk, H, D, flags] }, [b.sc, Vb, b.da], D, null, own);
    };
    const kv = [];
    for (let l = 0; l < dm.dec_layers; l++) {
      const L = `decoder.layers.${l}.`;
      const lin = (k, X, Y, o = {}) => this._linear(s, W(L + k + ".weight"), X, Y, 1, { ...o, bias: W(L + k + ".bias").B }, own);
      const kc = this._buf(cap * D, `d${l}.kc`, own), vc = this._buf(cap * D, `d${l}.vc`, own);
      if (old) copies.push([old.kv[l].kc, kc, old.cap * D * 4], [old.kv[l].vc, vc, old.cap * D * 4]);
      kv.push({ kc, vc });
      this._norm(s, ln(L + "self_attn_layer_norm"), b.dx, b.dn, 1, D, dm.dec_eps, 0, 0, own);
      lin("self_attn.q_proj", b.dn, b.dq);
      lin("self_attn.k_proj", b.dn, kc, { flags: 4, Ny: D });
      lin("self_attn.v_proj", b.dn, vc, { flags: 4, Ny: D });
      attn1(kc, vc, 0, 1);
      lin("self_attn.out_proj", b.da, b.dx, { flags: 2 });
      this._norm(s, ln(L + "encoder_attn_layer_norm"), b.dx, b.dn, 1, D, dm.dec_eps, 0, 0, own);
      lin("encoder_attn.q_proj", b.dn, b.dq);
      attn1(this.plan.cross[l].xk, this.plan.cross[l].xv, Te, 0);
      lin("encoder_attn.out_proj", b.da, b.dx, { flags: 2 });
      this._norm(s, ln(L + "final_layer_norm"), b.dx, b.dn, 1, D, dm.dec_eps, 0, 0, own);
      lin("fc1", b.dn, b.dh, { flags: 1 });
      lin("fc2", b.dh, b.dx, { flags: 2 });
    }
    this._norm(s, ln("decoder.layer_norm"), b.dx, b.dn, 1, D, dm.dec_eps, 0, 0, own);
    const head = [];
    this._linear(head, table, b.dn, b.logits, 1, {}, own);
    this._step(head, "pick", { a: [V, CH, 0, 0], f: [this.gen.repetition_penalty] }, [b.logits, b.seen, b.pick], chunks, null, own);
    if (copies.length) {
      const enc = this.dev.createCommandEncoder();
      for (const [src, dst, n] of copies) enc.copyBufferToBuffer(src, 0, dst, 0, n);
      this.dev.queue.submit([enc.finish()]);
    }
    if (old) for (const x of old.own) x.destroy();
    const read = old ? old.read : [0, 1].map(() => this.dev.createBuffer({ size: chunks * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    this.dec = { cap, b, steps: s, head, embed, kv, own, read, chunks };
    Object.assign(this.b, b);
  }

  // ---------------------------------------------------------------- running

  _run(steps) {
    const enc = this.dev.createCommandEncoder();
    let pass = null;
    for (const st of steps) {
      if (st.copy) {
        if (pass) { pass.end(); pass = null; }
        enc.copyBufferToBuffer(st.copy[0], 0, st.copy[1], 0, st.copy[2]);
        continue;
      }
      if (!pass) pass = enc.beginComputePass();
      pass.setPipeline(st.pipe);
      pass.setBindGroup(0, st.group);
      pass.dispatchWorkgroups(st.gx, st.gy);
    }
    if (pass) pass.end();
    return enc;
  }

  // Canvas-sized RGB (white-padded, centred) as normalized patch rows [Tp, 3 * 16 * 16] in (c, y, x) order.
  // image: ImageBitmap, canvas, HTMLImageElement, or ImageData-like {data (RGBA), width, height}.
  preprocess(image, size = this.size) {
    const TH = size.height, TW = size.width, P = this.dims.patch;
    let w = image.width, h = image.height, rgba = image.data;
    if (image.naturalWidth) { w = image.naturalWidth; h = image.naturalHeight; }
    if (!rgba) {
      const cv = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : Object.assign(document.createElement("canvas"), { width: w, height: h });
      const ctx = cv.getContext("2d", { willReadFrequently: true });
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(image, 0, 0);
      rgba = ctx.getImageData(0, 0, w, h).data;
    }
    const [nw, nh] = resizedSize(w, h, TW, TH);
    let rgb, step = 3;
    if (nw !== w || nh !== h) rgb = resizeBilinear(rgba, w, h, nw, nh);
    else { rgb = rgba; step = 4; }
    const padL = Math.floor((TW - nw) / 2), padT = Math.floor((TH - nh) / 2);
    const { mean, std } = this.m.image;
    const Wp = TW / P, PK = 3 * P * P;
    const out = new Float32Array((TH / P) * Wp * PK);
    const lut = [0, 1, 2].map((c) => Float32Array.from({ length: 256 }, (_, v) => (Math.fround(v / 255) - mean[c]) / std[c]));
    for (let y = 0; y < TH; y++) {
      const sy = y - padT, py = Math.floor(y / P), yy = y % P;
      for (let x = 0; x < TW; x++) {
        const sx = x - padL, inside = sy >= 0 && sy < nh && sx >= 0 && sx < nw;
        const o = (py * Wp + Math.floor(x / P)) * PK + yy * P + (x % P), si = (sy * nw + sx) * step;
        for (let c = 0; c < 3; c++) out[o + c * P * P] = lut[c][inside ? rgb[si + c] : 255];
      }
    }
    return { patches: out, width: w, height: h };
  }

  // Run the encoder and neck on patch rows from preprocess(); fills the cross-attention caches.
  async encode(patches, opts = {}) {
    const plan = this._plan(this.size.height, this.size.width);
    if (patches.length !== plan.Tp * plan.PK) throw new Error("patches do not match the canvas size");
    this.dev.queue.writeBuffer(plan.b.patches, 0, patches);
    this.dev.queue.writeBuffer(this.dyn, 0, new Uint32Array([0, 0, 0, 0]));
    const prof = opts.profile ? {} : null, last = plan.stages.length - 1;
    for (let i = 0; i < plan.stages.length; i++) {
      aborted(opts.signal);
      const steps = plan.stages[i];
      if (prof) {   // wall time per kind of step, each kind waited for; opts.layers runs only that many blocks
        if (opts.layers && i > opts.layers && i < last) continue;
        for (let j = 0; j < steps.length;) {
          let e = j + 1;
          while (e < steps.length && steps[e].tag === steps[j].tag) e++;
          const t0 = performance.now();
          for (let k = j; k < e; k += 16) this.dev.queue.submit([this._run(steps.slice(k, Math.min(e, k + 16))).finish()]);
          await this.dev.queue.onSubmittedWorkDone();
          prof[steps[j].tag] = (prof[steps[j].tag] || 0) + performance.now() - t0;
          j = e;
        }
      } else {
        for (let j = 0; j < steps.length; j += 16) this.dev.queue.submit([this._run(steps.slice(j, j + 16)).finish()]);
        await this.dev.queue.onSubmittedWorkDone();
      }
      await opts.onStage?.(i + 1, plan.stages.length);
    }
    return prof;
  }

  // Feed one token at position `pos`; returns the next token (greedy, repetition penalty applied) when `want`.
  async _token(id, pos, want) {
    if (!this.dec) this._buildDec(this.opts.cacheRows || 1024);
    if (pos >= this.dec.cap) this._buildDec(this.dec.cap * 2);
    const dec = this.dec, part = dec.embed.find((e) => id >= e.n0 && id < e.n0 + e.n);
    this.dev.queue.writeBuffer(this.dyn, 0, new Uint32Array([pos, id, part.n0, 0]));
    this.dev.queue.writeBuffer(dec.b.seen, id * 4, new Uint32Array([1]));
    const enc = this._run(want ? [part.step, ...dec.steps, ...dec.head] : [part.step, ...dec.steps]);
    if (!want) { this.dev.queue.submit([enc.finish()]); return -1; }
    const rb = dec.read[pos & 1];
    enc.copyBufferToBuffer(dec.b.pick, 0, rb, 0, dec.chunks * 8);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const o = new Float32Array(rb.getMappedRange());
    let best = -Infinity, bi = 0;
    for (let c = 0; c < dec.chunks; c++) if (o[2 * c] > best) { best = o[2 * c]; bi = o[2 * c + 1]; }
    rb.unmap();
    return bi;
  }

  // Greedy decode against the page last given to encode(). Returns generated ids, end-of-sequence excluded.
  // opts: maxTokens, onToken(id, text), onFirst() (called when the first step's logits are in b.logits), signal.
  async decode(promptIds, opts = {}) {
    const maxTokens = opts.maxTokens ?? this.gen.max_new_tokens;
    if (!this.dec) this._buildDec(this.opts.cacheRows || 1024);
    const clr = this.dev.createCommandEncoder();
    clr.clearBuffer(this.dec.b.seen);
    this.dev.queue.submit([clr.finish()]);
    let pos = 0, next = -1;
    for (let i = 0; i < promptIds.length; i++) next = await this._token(promptIds[i], pos++, i === promptIds.length - 1);
    if (opts.onFirst) await opts.onFirst();
    const out = [], stream = this.tok.stream();
    while (out.length < maxTokens && next !== this.gen.eos) {
      aborted(opts.signal);
      out.push(next);
      opts.onToken?.(next, stream.push(next));
      next = await this._token(next, pos++, true);
    }
    return out;
  }

  // image -> { raw, tokens, ms: { encode, decode }, blocks: [{ cls, bbox (original image pixels), text }] }
  // opts: prompt, maxTokens, onToken(id, text), signal, size ({height, width}), tableFormat ("markdown" |
  // "html" | "latex"), textFormat ("markdown" | "plain"), onStage(done, total) during encoding.
  async parse(image, opts = {}) {
    if (opts.size) this.size = { height: opts.size.height, width: opts.size.width };
    const size = this.size;
    this._plan(size.height, size.width);
    let t = performance.now();
    const pre = this.preprocess(image, size);
    await this.encode(pre.patches, opts);
    const encode = performance.now() - t;
    t = performance.now();
    const tokens = await this.decode(this.tok.encodePrompt(opts.prompt || this.gen.prompt), opts);
    const decode = performance.now() - t;
    const raw = this.tok.decode(tokens);
    return { raw, tokens, ms: { encode, decode }, blocks: toBlocks(raw, pre.width, pre.height, size.width, size.height, opts) };
  }

  // Copy an activation buffer back, for parity checks ("enc", "logits", "x", ...).
  async probe(name, floats) {
    const src = this.b[name];
    const rb = this.dev.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rb, 0, floats * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return out;
  }

  destroy() {
    if (this.plan) for (const b of this.plan.own) b.destroy();
    if (this.dec) for (const b of this.dec.own) b.destroy();
  }
}
