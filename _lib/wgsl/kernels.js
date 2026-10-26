// Kernel library for the WGSL encoders: one template per block, shared by every engine. An engine
// records a block with the matching helper (matmul, norm, attention, pool, l2), which picks the
// kernel variant and lays out its uniform; SeqEngine compiles only the variants a model records.
//
// Weight formats plug into the matmul as a WGSL snippet for one k-block. The loop around it, the
// k-split reduction and the epilogue are common:
//   q4   onnxruntime MatMulNBits, [N, K] 4-bit, block 32, low nibble first, f32 scales, packed 4-bit
//        zero points; the block scale is applied once per block to the partial sums.
//   i8   int8 [N, K] (4 per u32) in a separate buffer Wt, one f32 scale per output column.
//   f32  f32 [N, K] in the weight buffer.
// Uniform: a = (format offsets x3 in words, N), b = (K, rows selector, bias word offset or NONE,
// mode: bit 0 GELU (erf), bit 1 accumulate into Y), f.x = alpha.

import { BYTE, HEADER, NONE } from "./runtime.js";

// Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7): the exact-erf GELU.
export const GELU = /* wgsl */ `
fn gelu(x: f32) -> f32 {
  let a = abs(x) * 0.7071067811865476;
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return 0.5 * x * (1.0 + sign(x) * y);
}
`;

// Each format: extra bindings after W, k values per block (BK), the K and N multiples it needs,
// per-thread setup, the block body (adds block b's contribution to acc[t] for the 8 rows), and the
// per-column factor applied in the epilogue.
export const FORMATS = {
  q4: {
    bind: [], BK: 32, kMul: 256,
    setup: `let zrow = (nb + 1u) / 2u;`,
    body: `
    let sc = bitcast<f32>(W[p.a.y + n * nb + b]);
    let z = f32((wbyte(p.a.z * 4u + n * zrow + b / 2u) >> (4u * (b % 2u))) & 15u);
    var part = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var w = 0u; w < 4u; w++) {
      let word = W[p.a.x + n * (K / 8u) + b * 4u + w];
      let lo = vec4<f32>(f32(word & 15u), f32((word >> 4u) & 15u), f32((word >> 8u) & 15u), f32((word >> 12u) & 15u)) - z;
      let hi = vec4<f32>(f32((word >> 16u) & 15u), f32((word >> 20u) & 15u), f32((word >> 24u) & 15u), f32(word >> 28u)) - z;
      let k4 = b * 8u + w * 2u;
      for (var t = 0u; t < 8u; t++) {
        part[t] += dot(lo, X[rr[t] + k4]) + dot(hi, X[rr[t] + k4 + 1u]);
      }
    }
    for (var t = 0u; t < 8u; t++) { acc[t] += sc * part[t]; }`,
    col: `1.0`,
  },
  i8: {
    bind: ["Wt"], BK: 4, kMul: 32,
    setup: ``,
    body: `
    let wv = i8x4(Wt[p.a.x + n * K4 + b]);
    for (var t = 0u; t < 8u; t++) { acc[t] += dot(wv, X[rr[t] + b]); }`,
    col: `bitcast<f32>(W[p.a.y + n])`,
  },
  f32: {
    bind: [], BK: 4, kMul: 32,
    setup: ``,
    body: `
    let o = p.a.x + n * K + b * 4u;
    let wv = vec4<f32>(bitcast<f32>(W[o]), bitcast<f32>(W[o + 1u]), bitcast<f32>(W[o + 2u]), bitcast<f32>(W[o + 3u]));
    for (var t = 0u; t < 8u; t++) { acc[t] += dot(wv, X[rr[t] + b]); }`,
    col: `1.0`,
  },
};

// Y[r, n] = epilogue(alpha * col(n) * sum_k X[r, k] W[n, k]). Workgroup: 32 columns x 8 k-splits,
// 8 rows; each k-split walks K / 8 contiguous values in blocks of BK, so each thread decodes its
// weights once and applies them to all 8 rows.
function matmulWGSL(fmt) {
  const F = FORMATS[fmt];
  const names = ["W", ...F.bind];
  const binds = names.map((nm, i) => `@group(0) @binding(${2 + i}) var<storage, read> ${nm}: array<u32>;`).join("\n");
  const x = 2 + names.length;
  return HEADER + /* wgsl */ `
${binds}
@group(0) @binding(${x}) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(${x + 1}) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 2048>;
${BYTE}
${GELU}
fn i8x4(w: u32) -> vec4<f32> {
  return vec4<f32>(f32(bitcast<i32>(w << 24u) >> 24u), f32(bitcast<i32>(w << 16u) >> 24u),
                   f32(bitcast<i32>(w << 8u) >> 24u), f32(bitcast<i32>(w) >> 24u));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let N = p.a.w; let K = p.b.x; let K4 = K / 4u;
  let rows = dn.n[p.b.y];
  let nl = l % 32u; let ks = l / 32u;
  let n = wid.x * 32u + nl;
  let r0 = wid.y * 8u;
  let nb = K / ${F.BK}u; let bps = nb / 8u;
  ${F.setup}
  var rr: array<u32, 8>;
  for (var t = 0u; t < 8u; t++) { rr[t] = min(r0 + t, rows - 1u) * K4; }
  var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var b = ks * bps; b < (ks + 1u) * bps; b++) {${F.body}
  }
  for (var t = 0u; t < 8u; t++) { red[(ks * 8u + t) * 32u + nl] = acc[t]; }
  workgroupBarrier();
  let t = l / 32u;
  let r = r0 + t;
  if (r >= rows) { return; }
  var s = 0.0;
  for (var j = 0u; j < 8u; j++) { s += red[(j * 8u + t) * 32u + nl]; }
  s = s * ${F.col} * p.f.x;
  if (p.b.z != ${NONE}u) { s += bitcast<f32>(W[p.b.z + n]); }
  if ((p.b.w & 1u) != 0u) { s = gelu(s); }
  if ((p.b.w & 2u) != 0u) { Y[r * N + n] += s; } else { Y[r * N + n] = s; }
}`;
}

// Row normalization over D <= 512 values, one workgroup of 64 per row, thread l holding elements
// l + 64 i. kind rms: x / rms(x) * w; kind ln: (x - mean) / std * w (+ bias).
// mode 0: y = norm(x).  mode 1: y = norm(y + x) (residual first, BERT post-LN).
// mode 2: y = (y + norm(x)) * scalar (residual after, Gemma sandwich norm with layer scalar).
// Uniform: a = (D, weight word offset, bias word offset or NONE, mode), b = (rows selector, rows per
// selected unit), f = (eps, scalar).
function normWGSL(kind) {
  return HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wid.x + wid.y * 65535u;
  if (r >= dn.n[p.b.x] * p.b.y) { return; }
  let D = p.a.x; let base = r * D; let mode = p.a.w;
  var v = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  var c = 0u;
  var s = 0.0;
  for (var j = l; j < D; j += 64u) {
    var a = x[base + j];
    if (mode == 1u) { a += y[base + j]; }
    v[c] = a; c++;
    ${kind === "rms" ? "s += a * a;" : "s += a;"}
  }
  red[l] = s;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  ${kind === "rms" ? `
  let mu = 0.0;
  let inv = 1.0 / sqrt(red[0] / f32(D) + p.f.x);` : `
  let mu = red[0] / f32(D);
  workgroupBarrier();
  var q = 0.0;
  for (var i = 0u; i < c; i++) { let d = v[i] - mu; q += d * d; }
  red[l] = q;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(D) + p.f.x);`}
  c = 0u;
  for (var j = l; j < D; j += 64u) {
    var o = (v[c] - mu) * inv * bitcast<f32>(W[p.a.y + j]);
    if (p.a.z != ${NONE}u) { o += bitcast<f32>(W[p.a.z + j]); }
    if (mode == 2u) { y[base + j] = (y[base + j] + o) * p.f.y; } else { y[base + j] = o; }
    c++;
  }
}`;
}

// Bidirectional attention inside each packed sequence, grouped-query (nkv divides nh), optionally
// limited to |i - j| <= window / 2. One workgroup of 128 per (token, query head): thread j scores
// key j (sequences are at most 128 tokens), then thread i sums values i + 128 e. Head dim hd is a
// multiple of 4 up to 512. Layout: separate q [T, nh, hd], k and v [T, nkv, hd] buffers, or (fused)
// one [T, (nh + 2 nkv) hd] buffer bound three times. Output o [T, nh, hd].
// Uniform: a = (nh, nkv, hd, fused), b.x = window or 0, f.x = score scale.
const ATTN = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> seqs: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> Q: array<f32>;
@group(0) @binding(5) var<storage, read> Kv: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> V: array<f32>;
@group(0) @binding(7) var<storage, read_write> O: array<f32>;
var<workgroup> qs: array<vec4<f32>, 128>;
var<workgroup> pr: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let t = wid.x; let h = wid.y;
  if (t >= dn.n.x) { return; }
  let nh = p.a.x; let nkv = p.a.y; let hd = p.a.z; let hd4 = hd / 4u;
  var qst = nh * hd; var kst = nkv * hd; var ko = 0u; var vo = 0u;
  if (p.a.w == 1u) { qst = (nh + 2u * nkv) * hd; kst = qst; ko = nh * hd; vo = (nh + nkv) * hd; }
  let kvh = h / (nh / nkv);
  let tk = tok[t];
  let sq = seqs[tk.z];
  let start = sq.x; let len = sq.y;
  let qb = t * qst + h * hd;
  for (var j = i; j < hd4; j += 128u) { qs[j] = vec4<f32>(Q[qb + 4u * j], Q[qb + 4u * j + 1u], Q[qb + 4u * j + 2u], Q[qb + 4u * j + 3u]); }
  workgroupBarrier();
  var sc = -3.0e38;
  if (i < len) {
    let pi = i32(tk.y); let pj = i32(i);
    if (p.b.x == 0u || u32(abs(pi - pj)) <= p.b.x / 2u) {
      let kb = ((start + i) * kst + ko + kvh * hd) / 4u;
      var d = 0.0;
      for (var j = 0u; j < hd4; j++) { d += dot(qs[j], Kv[kb + j]); }
      sc = d * p.f.x;
    }
  }
  red[i] = sc;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (i < s) { red[i] = max(red[i], red[i + s]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  var e = 0.0;
  if (sc > -1.0e38) { e = exp(sc - mx); }
  pr[i] = e; red[i] = e;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (i < s) { red[i] += red[i + s]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var j = 0u; j < len; j++) {
    let vb = (start + j) * kst + vo + kvh * hd;
    var c = 0u;
    for (var d = i; d < hd; d += 128u) { acc[c] += pr[j] * V[vb + d]; c++; }
  }
  var c = 0u;
  for (var d = i; d < hd; d += 128u) { O[(t * nh + h) * hd + d] = acc[c] * inv; c++; }
}`;

// Mean over each sequence's tokens: [tokens, H] -> [sequences, H].  a.x = H.
const POOL = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> seqs: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let b = wid.x;
  if (b >= dn.n.y) { return; }
  let H = p.a.x; let sq = seqs[b];
  for (var j = l; j < H; j += 256u) {
    var s = 0.0;
    for (var t = 0u; t < sq.y; t++) { s += x[(sq.x + t) * H + j]; }
    y[b * H + j] = s / f32(sq.y);
  }
}`;

// L2-normalize rows of D values.  a.x = D.
const L2 = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let b = wid.x;
  if (b >= dn.n.y) { return; }
  let D = p.a.x;
  var s = 0.0;
  for (var j = i; j < D; j += 256u) { let a = x[b * D + j]; s += a * a; }
  red[i] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (i < k) { red[i] += red[i + k]; } workgroupBarrier(); }
  let inv = 1.0 / max(sqrt(red[0]), 1e-12);
  for (var j = i; j < D; j += 256u) { y[b * D + j] = x[b * D + j] * inv; }
}`;

export const KERNELS = {
  matmul_q4: matmulWGSL("q4"), matmul_i8: matmulWGSL("i8"), matmul_f32: matmulWGSL("f32"),
  norm_rms: normWGSL("rms"), norm_ln: normWGSL("ln"), attn: ATTN, pool: POOL, l2: L2,
};

function split(n) { return n <= 65535 ? [n] : [65535, Math.ceil(n / 65535)]; }

// The helpers below record one block on engine `e` (a SeqEngine). Weight offsets are in words,
// relative to the binding passed as W (a window of the weight buffer, or the whole of it).

// w = { fmt, offs: [o0, o1, o2], N, K }, W: weight binding, Wt: the int8 buffer for fmt i8.
// opts: rowsSel (0 tokens, 1 sequences), alpha, bias (word offset), gelu, accumulate.
export function matmul(e, w, W, X, Y, { rowsSel = 0, alpha = 1, bias = NONE, gelu = false, accumulate = false, Wt } = {}) {
  const F = FORMATS[w.fmt];
  if (!F) throw new Error(`unknown weight format ${w.fmt}`);
  if (w.N % 32 || w.K % F.kMul) throw new Error(`matmul ${w.fmt}: N ${w.N} must be a multiple of 32 and K ${w.K} of ${F.kMul}`);
  const offs = w.offs.concat([0, 0, 0]).slice(0, 3);
  const bufs = w.fmt === "i8" ? [W, Wt, X, Y] : [W, X, Y];
  e.step(`matmul_${w.fmt}`, { a: [...offs, w.N], b: [w.K, rowsSel, bias, (gelu ? 1 : 0) | (accumulate ? 2 : 0)], f: [alpha] }, bufs,
    (n) => [w.N / 32, Math.ceil(n[rowsSel] / 8)]);
}

// kind "rms" | "ln"; wOff / bOff word offsets in W (bOff NONE for none); mode 0 | 1 | 2 as above.
export function norm(e, kind, W, wOff, X, Y, { D, eps, bOff = NONE, mode = 0, scalar = 1, rowsSel = 0, mult = 1 }) {
  if (D > 512) throw new Error(`norm: D ${D} over 512`);
  e.step(`norm_${kind}`, { a: [D, wOff, bOff, mode], b: [rowsSel, mult], f: [eps, scalar] }, [W, X, Y], (n) => split(n[rowsSel] * mult));
}

// Q, K, V: buffers (the same one three times when fused); O: [T, nh, hd].
export function attention(e, Q, K, V, O, { nh, nkv = nh, hd, fused = false, window = 0, scale }) {
  if (hd % 4 || hd > 512 || nh % nkv) throw new Error(`attention: head dim ${hd} (multiple of 4, <= 512), heads ${nh}/${nkv}`);
  e.step("attn", { a: [nh, nkv, hd, fused ? 1 : 0], b: [window], f: [scale ?? 1 / Math.sqrt(hd)] },
    [e.b.tok, e.b.seqs, Q, K, V, O], (n) => [n[0], nh]);
}

export function pool(e, X, Y, H) { e.step("pool", { a: [H] }, [e.b.seqs, X, Y], (n) => [n[1]]); }

export function l2(e, X, Y, D) { e.step("l2", { a: [D] }, [X, Y], (n) => [n[1]]); }
