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

// Row normalization over D values (a multiple of 64, at most 512), one workgroup of 64 per row,
// thread l holding elements l + 64 i. kind rms: x / rms(x) * w; kind ln: (x - mean) / std * w (+ bias).
// mode 0: y = norm(x).  mode 1: y = norm(y + x) (residual first, BERT post-LN).
// mode 2: y = (y + norm(x)) * scalar (residual after, Gemma sandwich norm with layer scalar).
// D, mode and the bias are WGSL constants, one pipeline per combination: with them as uniforms the
// kernel ran 22-32% slower on Apple Metal (gpu-bridge job wgsl-kern-1).
// Uniform: a = (weight word offset, bias word offset), b = (rows selector, rows per selected unit),
// f = (eps, scalar).
function normWGSL(kind, D, mode, bias) {
  const per = D / 64;
  return HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 64>;
const D = ${D}u;
const PER = ${per}u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wid.x + wid.y * 65535u;
  if (r >= dn.n[p.b.x] * p.b.y) { return; }
  let base = r * D;
  var v: array<f32, ${per}>;
  var s = 0.0;
  for (var i = 0u; i < PER; i++) {
    var a = x[base + l + 64u * i];
    ${mode === 1 ? "a += y[base + l + 64u * i];" : ""}
    v[i] = a;
    ${kind === "rms" ? "s += a * a;" : "s += a;"}
  }
  red[l] = s;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  ${kind === "rms" ? `
  let inv = 1.0 / sqrt(red[0] / f32(D) + p.f.x);` : `
  let mu = red[0] / f32(D);
  workgroupBarrier();
  var q = 0.0;
  for (var i = 0u; i < PER; i++) { v[i] -= mu; q += v[i] * v[i]; }
  red[l] = q;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(D) + p.f.x);`}
  for (var i = 0u; i < PER; i++) {
    let j = l + 64u * i;
    var o = v[i] * inv * bitcast<f32>(W[p.a.x + j]);
    ${bias ? "o += bitcast<f32>(W[p.a.y + j]);" : ""}
    ${mode === 2 ? "y[base + j] = (y[base + j] + o) * p.f.y;" : "y[base + j] = o;"}
  }
}`;
}

// Bidirectional attention inside each packed sequence, grouped-query (nkv divides nh), optionally
// limited to |i - j| <= window / 2. One workgroup of 128 per (token, query head): thread j scores
// key j (sequences are at most 128 tokens), then thread i sums values i + 128 e. Head dim hd is a
// multiple of 4 up to 512. Layout: separate q [T, nh, hd], k and v [T, nkv, hd] buffers, or (fused)
// one [T, (nh + 2 nkv) hd] buffer bound three times. Output o [T, nh, hd].
// The head dim and the layout are WGSL constants, one pipeline per combination.
// Uniform: a = (nh, nkv), b.x = window or 0, f.x = score scale.
const attnWGSL = (hd, fused) => HEADER + /* wgsl */ `
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
  let nh = p.a.x; let nkv = p.a.y; let hd = ${hd}u; let hd4 = ${hd / 4}u;
  ${fused ? "let qst = (nh + 2u * nkv) * hd; let kst = qst; let ko = nh * hd; let vo = (nh + nkv) * hd;"
          : "let qst = nh * hd; let kst = nkv * hd; let ko = 0u; let vo = 0u;"}
  let kvh = h / (nh / nkv);
  let tk = tok[t];
  let sq = seqs[tk.z];
  let start = sq.x; let len = sq.y;
  let qb = t * qst + h * hd;
  if (i < hd4) { qs[i] = vec4<f32>(Q[qb + 4u * i], Q[qb + 4u * i + 1u], Q[qb + 4u * i + 2u], Q[qb + 4u * i + 3u]); }
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
  ${hd >= 128 ? `
  var acc = array<f32, ${hd / 128}>();
  for (var j = 0u; j < len; j++) {
    let vb = (start + j) * kst + vo + kvh * hd + i;
    for (var e = 0u; e < ${hd / 128}u; e++) { acc[e] += pr[j] * V[vb + 128u * e]; }
  }
  for (var e = 0u; e < ${hd / 128}u; e++) { O[(t * nh + h) * hd + i + 128u * e] = acc[e] * inv; }` : `
  if (i < hd) {
    var a = 0.0;
    for (var j = 0u; j < len; j++) { a += pr[j] * V[(start + j) * kst + vo + kvh * hd + i]; }
    O[(t * nh + h) * hd + i] = a * inv;
  }`}
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

// Fixed kernels; norm and attention variants are generated per shape by their helpers.
export const KERNELS = {
  matmul_q4: matmulWGSL("q4"), matmul_i8: matmulWGSL("i8"), matmul_f32: matmulWGSL("f32"), pool: POOL, l2: L2,
};

// Name a shape-specialized variant and add its source to e.shaders on first use.
function variant(e, name, make) {
  e.shaders[name] ??= make();
  return name;
}

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
  if (D > 512 || D % 64) throw new Error(`norm: D ${D} must be a multiple of 64 up to 512`);
  const bias = bOff !== NONE;
  const k = variant(e, `norm_${kind}_${D}_${mode}${bias ? "_b" : ""}`, () => normWGSL(kind, D, mode, bias));
  e.step(k, { a: [wOff, bias ? bOff : 0], b: [rowsSel, mult], f: [eps, scalar] }, [W, X, Y], (n) => split(n[rowsSel] * mult));
}

// Q, K, V: buffers (the same one three times when fused); O: [T, nh, hd].
export function attention(e, Q, K, V, O, { nh, nkv = nh, hd, fused = false, window = 0, scale }) {
  if (hd % 4 || hd > 512 || (hd > 128 && hd % 128) || nh % nkv) throw new Error(`attention: head dim ${hd} (multiple of 4 up to 128, or of 128 up to 512), heads ${nh}/${nkv}`);
  const k = variant(e, `attn_${hd}${fused ? "_fused" : ""}`, () => attnWGSL(hd, fused));
  e.step(k, { a: [nh, nkv], b: [window], f: [scale ?? 1 / Math.sqrt(hd)] },
    [e.b.tok, e.b.seqs, Q, K, V, O], (n) => [n[0], nh]);
}

export function pool(e, X, Y, H) { e.step("pool", { a: [H] }, [e.b.seqs, X, Y], (n) => [n[1]]); }

export function l2(e, X, Y, D) { e.step("l2", { a: [D] }, [X, Y], (n) => [n[1]]); }

// Attention over sequences longer than one workgroup (up to 1024 keys), from browser-decider; an engine
// that uses it passes maxSeq up to 1024 to SeqEngine.
// One workgroup per (token, head); each of the 128 threads scores keys i, i+128, ... of the token's
// window (the whole sequence for global layers, |pi - pj| <= window/2 for local ones), then the
// first hd threads accumulate V. Fused [T, 3H] qkv layout, as in attnWGSL(hd, true).
const attnLongWGSL = (hd) => HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> seqs: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> Q: array<f32>;
@group(0) @binding(5) var<storage, read> Kv: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> V: array<f32>;
@group(0) @binding(7) var<storage, read_write> O: array<f32>;
var<workgroup> qs: array<vec4<f32>, ${hd / 4}>;
var<workgroup> pr: array<f32, 1024>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let t = wid.x; let h = wid.y;
  if (t >= dn.n.x) { return; }
  let nh = p.a.x; let hd = ${hd}u; let hd4 = ${hd / 4}u;
  let st = 3u * nh * hd; let ko = nh * hd; let vo = 2u * nh * hd;
  let tk = tok[t];
  let sq = seqs[tk.z];
  let start = sq.x; let len = sq.y; let pi = tk.y;
  var lo = 0u; var hi = len;
  if (p.b.x != 0u) {
    let w2 = p.b.x / 2u;
    lo = select(0u, pi - w2, pi > w2);
    hi = min(len, pi + w2 + 1u);
  }
  let n = hi - lo;
  let qb = t * st + h * hd;
  if (i < hd4) { qs[i] = vec4<f32>(Q[qb + 4u * i], Q[qb + 4u * i + 1u], Q[qb + 4u * i + 2u], Q[qb + 4u * i + 3u]); }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var j = i; j < n; j += 128u) {
    let kb = ((start + lo + j) * st + ko + h * hd) / 4u;
    var d = 0.0;
    for (var c = 0u; c < hd4; c++) { d += dot(qs[c], Kv[kb + c]); }
    let s = d * p.f.x;
    pr[j] = s;
    mx = max(mx, s);
  }
  red[i] = mx;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (i < s) { red[i] = max(red[i], red[i + s]); } workgroupBarrier(); }
  let M = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = i; j < n; j += 128u) { let e = exp(pr[j] - M); pr[j] = e; sum += e; }
  red[i] = sum;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (i < s) { red[i] += red[i + s]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  if (i < hd) {
    var a = 0.0;
    for (var j = 0u; j < n; j++) { a += pr[j] * V[(start + lo + j) * st + vo + h * hd + i]; }
    O[(t * nh + h) * hd + i] = a * inv;
  }
}`;

export function attentionLong(e, QKV, O, { nh, hd, window = 0 }) {
  if (hd % 4 || hd > 128) throw new Error(`attentionLong: head dim ${hd}`);
  const k = variant(e, `attnlong_${hd}`, () => attnLongWGSL(hd));
  e.step(k, { a: [nh], b: [window], f: [1 / Math.sqrt(hd)] }, [e.b.tok, e.b.seqs, QKV, QKV, QKV, O], (n) => [n[0], nh]);
}

// Copy rows `idx` (token indexes into the packed batch) of X [T, H] into Y [P, H].  a.x = H, a.y = P.
const GATHER = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wid.x; let H = p.a.x;
  if (r >= p.a.y || dn.n.x == 0u) { return; }  // dn read so the auto layout keeps binding 1
  let src = idx[r];
  for (var j = l; j < H; j += 64u) { Y[r * H + j] = X[src * H + j]; }
}`;
KERNELS.gather = GATHER;

// ---- Token-table embedding --------------------------------------------------------------------------
// y[t, j] = alpha * table(id)[j] (+ pos[position, j] + type0[j], BERT). One workgroup of 64 per token,
// thread l holding columns l + 64 i. Row formats, each with one fp16 scale per 32 values:
//   t4  symmetric 4-bit, two per byte, low nibble = even column, value (code - 8) * scale
//   t3  3-bit, eight per 3 bytes (little-endian bitstream), a zero-point byte per 32, (code - zp) * scale
// H, the format and the options are WGSL constants. Uniform: a = (codes, scales, zero points) byte
// offsets and the position-table word offset; b.x = type0 word offset; f.x = alpha.
const EMBED_FMT = {
  t4: `let b = wbyte(p.a.x + id * (H / 2u) + j / 2u);
    var v = (f32((b >> ((j & 1u) * 4u)) & 15u) - 8.0) * s;`,
  t3: `let cb = p.a.x + id * (H / 8u) * 3u + (j / 8u) * 3u;
    let word = wbyte(cb) | (wbyte(cb + 1u) << 8u) | (wbyte(cb + 2u) << 16u);
    let z = f32(wbyte(p.a.z + id * G + j / 32u));
    var v = (f32((word >> (3u * (j & 7u))) & 7u) - z) * s;`,
};
const embedWGSL = (fmt, H, alpha, pos) => HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${BYTE}
const H = ${H}u;
const G = ${H / 32}u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x + wid.y * 65535u;
  if (t >= dn.n.x) { return; }
  let id = tok[t].x;
  for (var j = l; j < H; j += 64u) {
    let s = wf16(p.a.y + (id * G + j / 32u) * 2u);
    ${EMBED_FMT[fmt]}
    ${alpha ? "v = v * p.f.x;" : ""}
    ${pos ? "v += wf32(p.a.w + tok[t].y * H + j) + wf32(p.b.x + j);" : ""}
    y[t * H + j] = v;
  }
}`;

// t = { fmt: "t4" | "t3", codes, scales, zp } byte offsets in W; opts.alpha, opts.pos / opts.type0
// (word offsets of BERT's position and token-type rows).
export function embed(e, W, Y, t, { H, alpha = 1, pos = NONE, type0 = 0 }) {
  if (!EMBED_FMT[t.fmt] || H % 64) throw new Error(`embed: format ${t.fmt}, H ${H} (a multiple of 64)`);
  const k = variant(e, `embed_${t.fmt}_${H}${alpha !== 1 ? "_a" : ""}${pos !== NONE ? "_pos" : ""}`, () => embedWGSL(t.fmt, H, alpha !== 1, pos !== NONE));
  e.step(k, { a: [t.codes, t.scales, t.zp ?? 0, pos === NONE ? 0 : pos], b: [type0], f: [alpha] }, [W, e.b.tok, Y], (n) => split(n[0]));
}

// ---- Rotary position embedding --------------------------------------------------------------------
// ropeTable(thetas, hd, maxSeq): [cos | sin] blocks, each [table][pos][hd / 2], angles formed in float32
// as the exported graphs do. rope() rotates in place (rotate_half convention) the first `parts` blocks
// of nh heads in each row of X: parts 2 for the q and k of a fused [T, (q k v)] row, 1 for a plain
// [T, nh * hd] buffer. hd, maxSeq, the table count and parts are WGSL constants; uniform a = (nh, row
// stride, table index).
export function ropeTable(thetas, hd, maxSeq) {
  const half = hd / 2, nt = thetas.length, out = new Float32Array(2 * nt * maxSeq * half);
  thetas.forEach((theta, sel) => {
    for (let i = 0; i < half; i++) {
      const inv = Math.fround(1 / Math.fround(Math.pow(theta, Math.fround((2 * i) / hd))));
      for (let pos = 0; pos < maxSeq; pos++) {
        const ang = Math.fround(pos * inv);
        out[(sel * maxSeq + pos) * half + i] = Math.cos(ang);
        out[nt * maxSeq * half + (sel * maxSeq + pos) * half + i] = Math.sin(ang);
      }
    }
  });
  return out;
}

const ropeWGSL = (hd, maxSeq, nt, parts) => HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> rope: array<f32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
const D = ${hd}u;
const HALF = ${hd / 2}u;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x + wid.y * 65535u;
  if (t >= dn.n.x) { return; }
  let nh = p.a.x;
  let tb = (p.a.z * ${maxSeq}u + tok[t].y) * HALF;
  let sinOff = ${nt * maxSeq}u * HALF;
  for (var i = l; i < ${parts}u * nh * HALF; i += 128u) {
    let part = i / (nh * HALF);
    let h = (i / HALF) % nh; let d = i % HALF;
    let base = t * p.a.y + part * nh * D + h * D;
    let c = rope[tb + d]; let s = rope[sinOff + tb + d];
    let x1 = x[base + d]; let x2 = x[base + d + HALF];
    x[base + d] = x1 * c - x2 * s;
    x[base + d + HALF] = x2 * c + x1 * s;
  }
}`;

// table: the ropeTable buffer for `tables` thetas; sel picks one; stride defaults to parts * nh * hd.
export function rope(e, table, X, { nh, hd, maxSeq, tables = 1, sel = 0, parts = 1, stride }) {
  if (hd % 2 || sel >= tables) throw new Error(`rope: head dim ${hd}, table ${sel} of ${tables}`);
  const k = variant(e, `rope_${hd}_${maxSeq}_${tables}_${parts}`, () => ropeWGSL(hd, maxSeq, tables, parts));
  e.step(k, { a: [nh, stride ?? parts * nh * hd, sel] }, [table, e.b.tok, X], (n) => split(n[0]));
}

// ---- Gated activation -----------------------------------------------------------------------------
// Y[r, j] = act(A[r * sa + oa + j]) * B[r * sb + ob + j] for j < I, over rows * mult rows. A and B may
// be the same buffer (ModernBERT's [gate | up] row: sa = sb = 2I, ob = I); with inPlace, Y is A
// (EmbeddingGemma's gelumul, LFM's SwiGLU). act: gelu (erf), gelu_tanh (clamped as in EG2), silu.
// I and act are WGSL constants; uniform a = (sa, oa, sb, ob), b = (rows selector, mult).
const ACT = {
  gelu: "gelu(x)",
  gelu_tanh: "0.5 * x * (1.0 + tanh(clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0)))",
  silu: "x / (1.0 + exp(-x))",
};
const gatedWGSL = (act, I, inPlace) => HEADER + /* wgsl */ `
${inPlace ? `@group(0) @binding(2) var<storage, read_write> A: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;`
          : `@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;`}
${act === "gelu" ? GELU : ""}
const I = ${I}u;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let idx = (wid.x + wid.y * 65535u) * 256u + l;
  if (idx >= dn.n[p.b.x] * p.b.y * I) { return; }
  let r = idx / I; let j = idx % I;
  let x = A[r * p.a.x + p.a.y + j];
  ${inPlace ? "A" : "Y"}[idx] = ${ACT[act]} * B[r * p.a.z + p.a.w + j];
}`;

export function gated(e, A, B, Y, { I, act = "gelu", sa, oa = 0, sb, ob = 0, rowsSel = 0, mult = 1 }) {
  if (!ACT[act]) throw new Error(`gated: activation ${act}`);
  const inPlace = Y === A;
  if (inPlace && ((sa ?? I) !== I || oa)) throw new Error("gated in place: A's rows must be [rows, I] from offset 0");
  const k = variant(e, `gated_${act}_${I}${inPlace ? "_ip" : ""}`, () => gatedWGSL(act, I, inPlace));
  e.step(k, { a: [sa ?? I, oa, sb ?? I, ob], b: [rowsSel, mult] }, inPlace ? [A, B] : [A, B, Y],
    (n) => split(Math.ceil(n[rowsSel] * mult * I / 256)));
}
