// A WebGPU engine, in hand-written WGSL, for SauerkrautLM-MoEdernColBERT 1.5M and 27M
// (VAGOsolutions' MoEdernBERT: hash embedding, attention + mixture-of-experts layers, ColBERT head).
// It reads the checkpoint's own model.safetensors and 1_Dense/model.safetensors, so there is no
// export step; run/mcb.py is the NumPy/PyTorch spec it follows (itself at cosine 1.0 to the
// official sentence-transformers pipeline).
//
//   const eng = await MoEdernEngine.create(device, { config, dense }, model, { maxTokens: 4096, maxSeqs: 64 });
//   const toks = await eng.embed([ids0, ids1, ...]);   // Float32Array[T * 128], one unit row per token
//
// Every token gets a vector (ColBERT). Callers drop document skiplist tokens themselves; queries are
// [CLS] [Q] ... [SEP] padded with [MASK] to 32 and keep all 32. A sequence is at most 512 tokens.
// Matmuls run on a 64 x 64 tiled f32 kernel (opts.kernel "template" switches to the shared 8-row
// kernels.js matmul). The MoE runs sparsely: a router kernel appends each (token, slot) to its
// expert's list, and grouped matmuls walk only those rows.
// The 115M size's gated-conv layers are not implemented.

import { BYTE, HEADER, SeqEngine } from "../../_lib/wgsl/runtime.js";
import { GELU, KERNELS, matmul } from "../../_lib/wgsl/kernels.js";

export const MAX_LEN = 512;

// { name: { dtype, shape, data: Float32Array } } from a safetensors file (F32 only).
export function readSafetensors(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = Number(dv.getBigUint64(0, true));
  const head = JSON.parse(new TextDecoder().decode(u8.subarray(8, 8 + n)));
  const out = {};
  for (const [k, v] of Object.entries(head)) {
    if (k === "__metadata__") continue;
    if (v.dtype !== "F32") throw new Error(`${k}: dtype ${v.dtype}`);
    const [a, b] = v.data_offsets;
    const bytes = u8.slice(8 + n + a, 8 + n + b);   // copy: aligns to 4
    out[k] = { shape: v.shape, data: new Float32Array(bytes.buffer) };
  }
  return out;
}

const SHADERS = {
  // Hash embedding: e = LayerNorm(P @ table[id]), table row of p.a.y values, projection [H, P].
  // One workgroup of 128 per token. p.a = (H, P, table word offset, projection word offset),
  // p.b = (LN weight, LN bias), p.f.x = eps.
  embed: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${BYTE}
var<workgroup> row: array<f32, 256>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x + wid.y * 65535u;
  if (t >= dn.n.x) { return; }
  let H = p.a.x; let P = p.a.y; let id = tok[t].x;
  for (var k = l; k < P; k += 128u) { row[k] = wf32(p.a.z + id * P + k); }
  workgroupBarrier();
  var v = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  var c = 0u; var s = 0.0;
  for (var j = l; j < H; j += 128u) {
    var a = 0.0;
    for (var k = 0u; k < P; k++) { a += wf32(p.a.w + j * P + k) * row[k]; }
    v[c] = a; c++; s += a;
  }
  red[l] = s;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let mu = red[0] / f32(H);
  workgroupBarrier();
  var q = 0.0;
  for (var i = 0u; i < c; i++) { let d = v[i] - mu; q += d * d; }
  red[l] = q;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(H) + p.f.x);
  c = 0u;
  for (var j = l; j < H; j += 128u) {
    y[t * H + j] = (v[c] - mu) * inv * wf32(p.b.x + j) + wf32(p.b.y + j);
    c++;
  }
}`,

  // RMSNorm rows of D <= 1024: y = x / rms(x) * w.  p.a = (D, weight word offset), p.f.x = eps.
  rms: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${BYTE}
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wid.x + wid.y * 65535u;
  if (r >= dn.n.x) { return; }
  let D = p.a.x; let base = r * D;
  var s = 0.0;
  for (var j = l; j < D; j += 128u) { let a = x[base + j]; s += a * a; }
  red[l] = s;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / f32(D) + p.f.x);
  for (var j = l; j < D; j += 128u) { y[base + j] = x[base + j] * inv * wf32(p.a.y + j); }
}`,

  // RoPE in place on a [T, heads * hd] buffer (rotate_half convention). p.a = (heads, hd).
  rope: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> rope: array<f32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x + wid.y * 65535u;
  if (t >= dn.n.x) { return; }
  let nh = p.a.x; let D = p.a.y; let half = D / 2u;
  let pos = tok[t].y;
  let sinOff = ${MAX_LEN}u * half;
  for (var i = l; i < nh * half; i += 64u) {
    let h = i / half; let d = i % half;
    let base = (t * nh + h) * D;
    let c = rope[pos * half + d]; let s = rope[sinOff + pos * half + d];
    let x1 = x[base + d]; let x2 = x[base + d + half];
    x[base + d] = x1 * c - x2 * s;
    x[base + d + half] = x2 * c + x1 * s;
  }
}`,

  // Bidirectional grouped-query attention inside each packed sequence of up to 512 tokens, head dim 64.
  // One workgroup of 64 per (token, query head): scores for all keys into workgroup memory, softmax,
  // then thread d sums dimension d over the values. p.a = (nh, nkv), p.f.x = scale.
  attn: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> seqs: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> V: array<f32>;
@group(0) @binding(7) var<storage, read_write> O: array<f32>;
var<workgroup> qs: array<vec4<f32>, 16>;
var<workgroup> sc: array<f32, ${MAX_LEN}>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let t = wid.x; let h = wid.y;
  if (t >= dn.n.x) { return; }
  let nh = p.a.x; let nkv = p.a.y;
  let kvh = h / (nh / nkv);
  let sq = seqs[tok[t].z];
  let start = sq.x; let len = sq.y;
  if (i < 16u) { qs[i] = Q[(t * nh + h) * 16u + i]; }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var j = i; j < len; j += 64u) {
    let kb = ((start + j) * nkv + kvh) * 16u;
    var d = 0.0;
    for (var c = 0u; c < 16u; c++) { d += dot(qs[c], K[kb + c]); }
    let s = d * p.f.x;
    sc[j] = s; mx = max(mx, s);
  }
  red[i] = mx;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (i < s) { red[i] = max(red[i], red[i + s]); } workgroupBarrier(); }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = i; j < len; j += 64u) { let e = exp(sc[j] - m); sc[j] = e; sum += e; }
  red[i] = sum;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (i < s) { red[i] += red[i + s]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  var acc = 0.0;
  for (var j = 0u; j < len; j++) { acc += sc[j] * V[((start + j) * nkv + kvh) * 64u + i]; }
  O[(t * nh + h) * 64u + i] = acc * inv;
}`,

  // Router: softmax over E <= 8 experts, top-2, renormalized; appends slot 2t+k to expert e's list
  // and stores its gate. One workgroup of 64 per token. p.a = (H, router word offset, E, list cap).
  route: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> list: array<u32>;
@group(0) @binding(6) var<storage, read_write> gw: array<f32>;
${BYTE}
var<workgroup> red: array<f32, 512>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x + wid.y * 65535u;
  if (t >= dn.n.x) { return; }
  let H = p.a.x; let E = p.a.z;
  var part = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var j = l; j < H; j += 64u) {
    let a = x[t * H + j];
    for (var e = 0u; e < E; e++) { part[e] += a * wf32(p.a.y + e * H + j); }
  }
  for (var e = 0u; e < 8u; e++) { red[e * 64u + l] = part[e]; }
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) {
    if (l < k) { for (var e = 0u; e < 8u; e++) { red[e * 64u + l] += red[e * 64u + l + k]; } }
    workgroupBarrier();
  }
  if (l != 0u) { return; }
  var mx = -3.0e38;
  for (var e = 0u; e < E; e++) { mx = max(mx, red[e * 64u]); }
  var pr = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  var z = 0.0;
  for (var e = 0u; e < E; e++) { pr[e] = exp(red[e * 64u] - mx); z += pr[e]; }
  var i0 = 0u;
  for (var e = 1u; e < E; e++) { if (pr[e] > pr[i0]) { i0 = e; } }
  var i1 = select(0u, 1u, i0 == 0u);
  for (var e = 0u; e < E; e++) { if (e != i0 && pr[e] > pr[i1]) { i1 = e; } }
  let w0 = pr[i0] / z; let w1 = pr[i1] / z;
  let n = w0 + w1 + 1e-9;
  gw[2u * t] = w0 / n; gw[2u * t + 1u] = w1 / n;
  list[i0 * p.a.w + atomicAdd(&cnt[i0], 1u)] = 2u * t;
  list[i1 * p.a.w + atomicAdd(&cnt[i1], 1u)] = 2u * t + 1u;
}`,

  zero: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read_write> cnt: array<u32>;
@compute @workgroup_size(8)
fn main(@builtin(local_invocation_index) l: u32) { cnt[l] = p.a.x * dn.n.w; }`,

  // Grouped matmul over the routed rows of expert e = wid.z (at most one slot per token, so the
  // grid covers T rows per expert and idle workgroups exit at once): Y[slot, n] = sum_k X[row(slot), k] W_e[n, k],
  // W_e f32 [N, K] at word p.a.x + e * N * K. row(slot) = slot / 2 (p.b.x = 1, reading token rows)
  // or slot (p.b.x = 0). Same 32 columns x 8 k-splits x 8 rows layout as the shared template.
  // p.a = (weight word offset, N, K, list cap).
  gmm: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> cnt: array<u32>;
@group(0) @binding(4) var<storage, read> list: array<u32>;
@group(0) @binding(5) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 2048>;
var<workgroup> wrows: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let e = wid.z;
  if (dn.n.x == 0u) { return; }
  if (l == 0u) { wrows = cnt[e]; }
  let rows = workgroupUniformLoad(&wrows);
  let r0 = wid.y * 8u;
  if (r0 >= rows) { return; }
  let N = p.a.y; let K = p.a.z; let K4 = K / 4u;
  let nl = l % 32u; let ks = l / 32u;
  let n = wid.x * 32u + nl;
  let wb = p.a.x + e * N * K + n * K;
  let bps = K4 / 8u;
  var rr: array<u32, 8>;
  var slot: array<u32, 8>;
  for (var t = 0u; t < 8u; t++) {
    let s = list[e * p.a.w + min(r0 + t, rows - 1u)];
    slot[t] = s;
    rr[t] = select(s, s / 2u, p.b.x == 1u) * K4;
  }
  var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var b = ks * bps; b < (ks + 1u) * bps; b++) {
    let o = wb + b * 4u;
    let wv = vec4<f32>(bitcast<f32>(W[o]), bitcast<f32>(W[o + 1u]), bitcast<f32>(W[o + 2u]), bitcast<f32>(W[o + 3u]));
    for (var t = 0u; t < 8u; t++) { acc[t] += dot(wv, X[rr[t] + b]); }
  }
  for (var t = 0u; t < 8u; t++) { red[(ks * 8u + t) * 32u + nl] = acc[t]; }
  workgroupBarrier();
  let t = l / 32u;
  if (r0 + t >= rows) { return; }
  var s = 0.0;
  for (var j = 0u; j < 8u; j++) { s += red[(j * 8u + t) * 32u + nl]; }
  Y[slot[t] * N + n] = s;
}`,

  // Tiled f32 matmul for batches: Y[out(r), n] (+)= sum_k X[in(r), k] W_e[n, k], W_e [N, K] at word
  // p.a.x + e * N * K, e = wid.z. 64 x 64 output tile per workgroup of 256, 4 x 4 per thread, K in
  // steps of 16 through workgroup memory. p.b.x row mode: 0 dense (r < tokens, in = out = r);
  // 1 expert rows, in = slot / 2 (a token), out = slot; 2 expert rows, in = out = slot.
  // p.b.y = 1 accumulates into Y. p.a = (weight word offset, N, K, list cap). N % 64 = 0, K % 16 = 0.
  mmt: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> cnt: array<u32>;
@group(0) @binding(4) var<storage, read> list: array<u32>;
@group(0) @binding(5) var<storage, read> X: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
var<workgroup> xs: array<f32, 1024>;
var<workgroup> ws: array<f32, 1024>;
var<workgroup> wrows: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let e = wid.z; let mode = p.b.x;
  if (l == 0u) { wrows = select(cnt[e], dn.n.x, mode == 0u); }
  let rows = workgroupUniformLoad(&wrows);
  let r0 = wid.y * 64u;
  if (r0 >= rows) { return; }
  let N = p.a.y; let K = p.a.z; let n0 = wid.x * 64u;
  let wb = p.a.x + e * N * K;
  let tx = l % 16u; let ty = l / 16u;
  // the 4 tile rows this thread loads (r = l / 4 for loading; 4 k values each)
  let lr = l / 4u; let lk = (l % 4u) * 4u;
  var inRow = 0u; var valid = r0 + lr < rows;
  if (valid) {
    if (mode == 0u) { inRow = r0 + lr; }
    else { let s = list[e * p.a.w + r0 + lr]; inRow = select(s, s / 2u, mode == 1u); }
  }
  var acc: array<vec4<f32>, 4>;
  for (var i = 0u; i < 4u; i++) { acc[i] = vec4<f32>(0.0); }
  for (var k0 = 0u; k0 < K; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      xs[lr * 16u + lk + i] = select(0.0, X[inRow * K + k0 + lk + i], valid);
      ws[lr * 16u + lk + i] = W[wb + (n0 + lr) * K + k0 + lk + i];
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      let w = vec4<f32>(ws[(tx * 4u) * 16u + k], ws[(tx * 4u + 1u) * 16u + k], ws[(tx * 4u + 2u) * 16u + k], ws[(tx * 4u + 3u) * 16u + k]);
      for (var i = 0u; i < 4u; i++) { acc[i] += xs[(ty * 4u + i) * 16u + k] * w; }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let r = r0 + ty * 4u + i;
    if (r >= rows) { break; }
    var o = r;
    if (mode != 0u) { o = list[e * p.a.w + r]; }
    for (var c = 0u; c < 4u; c++) {
      let idx = o * N + n0 + tx * 4u + c;
      if (p.b.y == 1u) { Y[idx] += acc[i][c]; } else { Y[idx] = acc[i][c]; }
    }
  }
}`,

  // GeGLU: g[r, j] = gelu(u[r, j]) * u[r, I + j] over p.b.x rows-per-token x tokens.  p.a.x = I.
  glu: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> g: array<f32>;
${GELU}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let I = p.a.x;
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= dn.n.x * p.b.x * I) { return; }
  let r = idx / I; let j = idx % I;
  g[idx] = gelu(u[r * 2u * I + j]) * u[r * 2u * I + I + j];
}`,

  // x[t] += gw[2t] od[2t] + gw[2t+1] od[2t+1].  p.a.x = H.
  combine: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> od: array<f32>;
@group(0) @binding(3) var<storage, read> gw: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let H = p.a.x;
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= dn.n.x * H) { return; }
  let t = idx / H; let j = idx % H;
  x[idx] += gw[2u * t] * od[(2u * t) * H + j] + gw[2u * t + 1u] * od[(2u * t + 1u) * H + j];
}`,

  // L2-normalize each token row of 128.
  l2tok: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let r = wid.x + wid.y * 65535u;
  if (r >= dn.n.x) { return; }
  let a = x[r * 128u + i + p.a.x];
  red[i] = a * a;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (i < k) { red[i] += red[i + k]; } workgroupBarrier(); }
  y[r * 128u + i] = a / max(sqrt(red[0]), 1e-12);
}`,
};

const rows2d = (n) => (n <= 65535 ? [n] : [65535, Math.ceil(n / 65535)]);
const flat = (n) => { const g = Math.ceil(n / 256); return g <= 65535 ? [g] : [65535, Math.ceil(g / 65535)]; };

export class MoEdernEngine extends SeqEngine {
  // manifest: { config (config.json), dense: 1_Dense safetensors bytes }; data: model.safetensors bytes.
  constructor(device, manifest, opts) {
    super(device, manifest, opts);
    this.dim = 128;
    this.kernel = opts.kernel ?? "tiled";
  }

  async _build(bytes) {
    const c = this.m.config, T = this.maxTokens, B = this.maxSeqs;
    if (c.layer_types.some((x) => x !== "attn_moe")) throw new Error("gated-conv layers (115M) are not implemented");
    const H = c.hidden_size, nh = c.num_attention_heads, nkv = c.num_key_value_heads, hd = H / nh;
    const E = c.num_routed_experts, He = c.expert_dim, P = c.hash_projections;
    if (hd !== 64 || H > 1024 || P > 256 || E > 8 || c.top_k !== 2) throw new Error("unexpected config");
    const st = readSafetensors(bytes), dn = readSafetensors(this.m.dense);
    const g = (k) => { const t = st[k]; if (!t) throw new Error(`missing ${k}`); return t.data; };

    // One packed f32 weight buffer; experts transposed host-side from [E, K, N] to [E, N, K], gate and
    // up stacked into one [E, 2He, H] block (and the shared expert's into [2He, H]).
    const parts = []; let words = 0;
    const add = (arr) => { const o = words; parts.push(arr); words += arr.length; return o; };
    const tr = (src, Kd, Nd, dst, off) => { for (let k = 0; k < Kd; k++) for (let n = 0; n < Nd; n++) dst[off + n * Kd + k] = src[k * Nd + n]; };
    const off = {
      table: add(g("embeddings.hash_embeddings.weight")), proj: add(g("embeddings.projection.weight")),
      lnw: add(g("embeddings.norm.weight")), lnb: add(g("embeddings.norm.bias")),
      final: add(g("norm.weight")), dense: add(dn["linear.weight"].data), layers: [],
    };
    for (let i = 0; i < c.layer_types.length; i++) {
      const p = `layers.${i}.`;
      const gu = new Float32Array(E * 2 * He * H), down = new Float32Array(E * H * He);
      const Wg = g(p + "ffn.W_gate"), Wu = g(p + "ffn.W_up"), Wd = g(p + "ffn.W_down");
      for (let e = 0; e < E; e++) {
        tr(Wg.subarray(e * H * He, (e + 1) * H * He), H, He, gu, e * 2 * He * H);
        tr(Wu.subarray(e * H * He, (e + 1) * H * He), H, He, gu, e * 2 * He * H + He * H);
        tr(Wd.subarray(e * He * H, (e + 1) * He * H), He, H, down, e * H * He);
      }
      const sgu = new Float32Array(2 * He * H);
      sgu.set(g(p + "ffn.shared_gate.weight")); sgu.set(g(p + "ffn.shared_up.weight"), He * H);
      off.layers.push({
        anorm: add(g(p + "attention.norm.weight")), q: add(g(p + "attention.Q_proj.weight")),
        k: add(g(p + "attention.K_proj.weight")), v: add(g(p + "attention.V_proj.weight")),
        o: add(g(p + "attention.O_proj.weight")), fnorm: add(g(p + "ffn_norm.weight")),
        router: add(g(p + "ffn.router.weight")), gu: add(gu), down: add(down),
        sgu: add(sgu), sdown: add(g(p + "ffn.shared_down.weight")),
      });
    }
    const all = new Float32Array(words);
    let o = 0; for (const a of parts) { all.set(a, o); o += a.length; }
    this.W = this.upload(all, "weights");
    this.weightBytes = all.byteLength;

    // RoPE table [pos][hd/2]: cos block then sin block, angles in float32 like torch.
    const half = hd / 2, rope = new Float32Array(2 * MAX_LEN * half);
    for (let i = 0; i < half; i++) {
      const inv = Math.fround(1 / Math.fround(Math.pow(c.rope_theta, Math.fround((2 * i) / hd))));
      for (let pos = 0; pos < MAX_LEN; pos++) {
        const ang = Math.fround(pos * inv);
        rope[pos * half + i] = Math.cos(ang); rope[MAX_LEN * half + pos * half + i] = Math.sin(ang);
      }
    }
    this.rope = this.upload(rope, "rope");

    const U = GPUBufferUsage, buf = (n, l, x) => this.buf(n, l, x);
    const b = this.b = {
      tok: buf(T * 4, "tok", U.COPY_DST), seqs: buf(B * 4, "seqs", U.COPY_DST),
      x: buf(T * H, "x"), a: buf(T * H, "a"), q: buf(T * H, "q"), k: buf(T * nkv * hd, "k"),
      v: buf(T * nkv * hd, "v"), o: buf(T * H, "o"), gw: buf(2 * T, "gw"), cnt: buf(8, "cnt"),
      list: buf(E * 2 * T, "list"), u: buf(2 * T * 2 * He, "u"), g: buf(2 * T * He, "g"),
      od: buf(2 * T * H, "od"), d: buf(T * 128, "d"), out: buf(T * 128, "out"),
    };
    this.readTok = this.device.createBuffer({ size: T * 128 * 4, usage: U.MAP_READ | U.COPY_DST, label: "readback tokens" });
    this._owned.push(this.readTok);
    this.shaders = { ...KERNELS, ...SHADERS };
    const step = (...a) => this.step(...a), W = this.W, eps = c.norm_eps, cap = 2 * T;
    // tiled: the 64 x 64 batch kernel (mmt) everywhere; template: the shared 8-row kernels.js
    // matmul and the 8-row grouped gmm (the shape the query engines were tuned on).
    const tiled = this.kernel === "tiled";
    const mm = (wOff, N, K, X, Y, opts = {}) => tiled
      ? step("mmt", { a: [wOff, N, K, cap], b: [0, opts.accumulate ? 1 : 0] }, [W, b.cnt, b.list, X, Y], (n) => [N / 64, Math.ceil(n[0] / 64), 1])
      : matmul(this, { fmt: "f32", offs: [wOff], N, K }, W, X, Y, opts);
    const gmm = (wOff, N, K, X, Y, mode) => tiled
      ? step("mmt", { a: [wOff, N, K, cap], b: [mode, 0] }, [W, b.cnt, b.list, X, Y], (n) => [N / 64, Math.ceil(n[0] / 64), E])
      : step("gmm", { a: [wOff, N, K, cap], b: [mode === 1 ? 1 : 0] }, [W, b.cnt, b.list, X, Y], (n) => [N / 32, Math.ceil(n[0] / 8), E]);
    const rms = (wOff, X, Y) => step("rms", { a: [H, wOff], f: [eps] }, [W, X, Y], (n) => rows2d(n[0]));

    step("embed", { a: [H, P, off.table, off.proj], b: [off.lnw, off.lnb], f: [1e-5] }, [W, b.tok, b.x], (n) => rows2d(n[0]));
    for (const L of off.layers) {
      rms(L.anorm, b.x, b.a);
      mm(L.q, H, H, b.a, b.q); mm(L.k, nkv * hd, H, b.a, b.k); mm(L.v, nkv * hd, H, b.a, b.v);
      step("rope", { a: [nh, hd] }, [this.rope, b.tok, b.q], (n) => rows2d(n[0]));
      step("rope", { a: [nkv, hd] }, [this.rope, b.tok, b.k], (n) => rows2d(n[0]));
      step("attn", { a: [nh, nkv], f: [1 / Math.sqrt(hd)] }, [b.tok, b.seqs, b.q, b.k, b.v, b.o], (n) => [n[0], nh]);
      mm(L.o, H, H, b.o, b.x, { accumulate: true });
      rms(L.fnorm, b.x, b.a);
      step("zero", {}, [b.cnt], () => [1]);
      step("route", { a: [H, L.router, E, cap] }, [W, b.a, b.cnt, b.list, b.gw], (n) => rows2d(n[0]));
      gmm(L.gu, 2 * He, H, b.a, b.u, 1);
      step("glu", { a: [He], b: [2] }, [b.u, b.g], (n) => flat(2 * n[0] * He));
      gmm(L.down, H, He, b.g, b.od, 2);
      step("combine", { a: [H] }, [b.od, b.gw, b.x], (n) => flat(n[0] * H));
      // shared expert, accumulated straight into the residual
      mm(L.sgu, 2 * He, H, b.a, b.u);
      step("glu", { a: [He], b: [1] }, [b.u, b.g], (n) => flat(n[0] * He));
      mm(L.sdown, H, He, b.g, b.x, { accumulate: true });
    }
    rms(off.final, b.x, b.a);
    mm(off.dense, 128, H, b.a, b.d);
    step("l2tok", {}, [b.d, b.out], (n) => rows2d(n[0]));
  }

  // SeqEngine._record with sequences up to MAX_LEN tokens.
  _record(enc, batch, nSteps = Infinity) {
    const T = batch.reduce((s, x) => s + x.length, 0), B = batch.length;
    if (!B || T > this.maxTokens || B > this.maxSeqs) throw new Error(`batch of ${B} sequences / ${T} tokens exceeds the engine's limits`);
    const tok = new Uint32Array(T * 4), seqs = new Uint32Array(B * 4);
    let o = 0;
    batch.forEach((ids, b) => {
      if (!ids.length || ids.length > MAX_LEN) throw new Error(`sequence length ${ids.length} not in 1..${MAX_LEN}`);
      seqs.set([o, ids.length], b * 4);
      ids.forEach((id, j) => tok.set([id, j, b], (o + j) * 4));
      o += ids.length;
    });
    const q = this.device.queue;
    q.writeBuffer(this.b.tok, 0, tok);
    q.writeBuffer(this.b.seqs, 0, seqs);
    q.writeBuffer(this.dyn, 0, new Uint32Array([T, B, 0, 0]));
    const pass = enc.beginComputePass();
    const n = [T, B];
    for (const s of this.steps.slice(0, nSteps)) {
      pass.setPipeline(this.pipes[s.kind]);
      pass.setBindGroup(0, s.bg);
      pass.dispatchWorkgroups(...s.grid(n));
    }
    pass.end();
    return T;
  }

  // Token vectors for the whole batch: Float32Array of T * 128 (sequences back to back).
  async embed(batch) {
    const enc = this.device.createCommandEncoder();
    const T = this._record(enc, batch), n = T * 128 * 4;
    enc.copyBufferToBuffer(this.b.out, 0, this.readTok, 0, n);
    this.device.queue.submit([enc.finish()]);
    await this.readTok.mapAsync(GPUMapMode.READ, 0, n);
    const out = new Float32Array(this.readTok.getMappedRange(0, n).slice(0));
    this.readTok.unmap();
    return out;
  }
}
