// LFM3-1.7B on hand-written WGSL, weights in the rotated-GPTQ 4-bit format of run/export3.py
// (web/manifest3.json). Same streaming loader and surface as lfm.js:
//
//   const eng = await Lfm3Engine.create(device, manifest, url, { onProgress });
//   eng.reset();
//   let logits = await eng.forward(promptIds);   // prefill; Float32Array(vocab) for the last token
//   logits = await eng.forward([next]);          // one decode step
//
// Every matrix except the token table is stored in a randomized-Hadamard basis: before each matmul
// the activations are rotated in 256-wide blocks (x' = H diag(signs) x / 16, kernel ROT), then the
// matmul decodes 4-bit codes to Lloyd-Max levels times an fp16 scale per 32 weights.
//
// Per layer (sandwich norms): h = norm(x); x += post_norm(op(h)); x += post_ffn_norm(mlp(ffn_norm(x))).
//   conv: in_proj -> [B | C | x]; y = C * causal depthwise conv7(B * x) with a 6-step state; out_proj
//   attention: fused q | k | v | gate projection; RMS norm per 128-wide head on q and k; RoPE on
//   pairs (i, i + 64), the pairs past a full-attention layer's 32 frequencies left unrotated;
//   16 query heads over 4 KV heads, sliding layers seeing the last 256 positions; each head's
//   output times sigmoid(gate); o_proj
//   mlp: fused w1 | w3, silu(w1 x) * w3 x, w2

import { HEADER, BYTE, NONE, compile, split } from "../../_lib/wgsl/runtime.js";
import { EMBED, GEMV, LAST } from "./lfm.js";

const U = GPUBufferUsage;
export const MAX_T = 128;
export const MAX_CTX = 2048;

// Rotate rows of X [rows, K] in 256-wide blocks into Y: signs, fast Walsh-Hadamard transform, / 16.
// Grid (K / 256, rows). p.a.x = K, p.b.x = rows selector (0: tokens, 1: one row).
const ROT = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> v: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let K = p.a.x; let i = wid.y * K + wid.x * 256u + l;
  v[l] = X[i] * S[l];
  for (var h = 1u; h < 256u; h <<= 1u) {
    workgroupBarrier();
    let a = v[l]; let b = v[l ^ h];
    workgroupBarrier();
    v[l] = select(b - a, a + b, (l & h) == 0u);
  }
  Y[i] = v[l] * 0.0625;
}`;

// 4-bit Lloyd-Max matmul, Y[r, n] = scale * sum_k LV[code] X'[r, k]. Layout per matrix: codes
// [N, K/8] words (code k at bits 4 (k % 8)), then fp16 scales [N, K/32]. Workgroup: 32 columns x 8
// k-splits, TILE rows (1 for decode). p.a = (codes word offset, scales word offset, 0, N), p.b.x = K.
function gqWGSL(levels, TILE) {
  const lv = levels.map((x) => x.toFixed(9)).join(", ");
  return HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
const LV = array<f32, 16>(${lv});
var<workgroup> red: array<f32, ${256 * TILE}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let N = p.a.w; let K = p.b.x; let K4 = K / 4u;
  let rows = dn.n.x;
  let nl = l % 32u; let ks = l / 32u;
  let n = wid.x * 32u + nl;
  let r0 = wid.y * ${TILE}u;
  let nb = K / 32u; let bps = nb / 8u;
  var rr: array<u32, ${TILE}>;
  for (var t = 0u; t < ${TILE}u; t++) { rr[t] = min(r0 + t, rows - 1u) * K4; }
  var acc: array<f32, ${TILE}>;
  for (var t = 0u; t < ${TILE}u; t++) { acc[t] = 0.0; }
  for (var b = ks * bps; b < (ks + 1u) * bps; b++) {
    let si = n * nb + b;
    let sc = unpack2x16float(W[p.a.y + si / 2u])[si % 2u];
    var part: array<f32, ${TILE}>;
    for (var t = 0u; t < ${TILE}u; t++) { part[t] = 0.0; }
    for (var w = 0u; w < 4u; w++) {
      let word = W[p.a.x + n * (K / 8u) + b * 4u + w];
      let lo = vec4<f32>(LV[word & 15u], LV[(word >> 4u) & 15u], LV[(word >> 8u) & 15u], LV[(word >> 12u) & 15u]);
      let hi = vec4<f32>(LV[(word >> 16u) & 15u], LV[(word >> 20u) & 15u], LV[(word >> 24u) & 15u], LV[word >> 28u]);
      let k4 = b * 8u + w * 2u;
      for (var t = 0u; t < ${TILE}u; t++) { part[t] += dot(lo, X[rr[t] + k4]) + dot(hi, X[rr[t] + k4 + 1u]); }
    }
    for (var t = 0u; t < ${TILE}u; t++) { acc[t] += sc * part[t]; }
  }
  for (var t = 0u; t < ${TILE}u; t++) { red[(ks * ${TILE}u + t) * 32u + nl] = acc[t]; }
  workgroupBarrier();
  if (l >= 32u * ${TILE}u) { return; }
  let t = l / 32u; let r = r0 + t;
  if (r >= rows) { return; }
  var s = 0.0;
  for (var j = 0u; j < 8u; j++) { s += red[(j * ${TILE}u + t) * 32u + nl]; }
  Y[r * N + n] = s;
}`;
}

// RMS norm of rows of D: mode 0 Y = norm(X) * w; mode 1 Y += norm(X) * w (the sandwich norms' residual add).
// Rows: dn.n[p.b.x] * p.b.y.
function rmsWGSL(D, mode) {
  const WG = 256, per = D / WG;
  return HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, ${WG}>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wid.x + wid.y * 65535u;
  if (r >= dn.n[p.b.x] * p.b.y) { return; }
  let base = r * ${D}u;
  var v: array<f32, ${per}>;
  var s = 0.0;
  for (var i = 0u; i < ${per}u; i++) { let a = x[base + l + ${WG}u * i]; v[i] = a; s += a * a; }
  red[l] = s;
  workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / ${D}.0 + p.f.x);
  for (var i = 0u; i < ${per}u; i++) {
    let j = l + ${WG}u * i;
    let o = v[i] * inv * bitcast<f32>(W[p.a.x + j]);
    ${mode ? "y[base + j] += o;" : "y[base + j] = o;"}
  }
}`;
}

// Short convolution: BCX rows [B | C | x] (3D wide); y[t, c] = C[t, c] * sum_j w[c, j] ext[t + j],
// ext = the 6 cached values of B * x, then this pass's. Grid (D / 64, T).
const CONV = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read> BCX: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
const D = 2048u; const L = 7u;
fn ext(i: u32, c: u32) -> f32 {
  if (i < L - 1u) { return S[i * D + c]; }
  let r = (i - (L - 1u)) * 3u * D;
  return BCX[r + c] * BCX[r + 2u * D + c];
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let c = wid.x * 64u + l; let t = wid.y;
  var o = 0.0;
  for (var j = 0u; j < L; j++) { o += bitcast<f32>(W[p.a.x + c * L + j]) * ext(t + j, c); }
  Y[t * D + c] = BCX[t * 3u * D + D + c] * o;
}`;

// Conv state after the pass: the last 6 values of ext, per channel.
const CONV_STATE = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
@group(0) @binding(3) var<storage, read> BCX: array<f32>;
const D = 2048u; const M = 6u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let c = wid.x * 64u + l; let T = dn.n.x;
  var old: array<f32, 6>;
  for (var i = 0u; i < M; i++) { old[i] = S[i * D + c]; }
  for (var i = 0u; i < M; i++) {
    let e = T + i;
    var v = 0.0;
    if (e < M) { v = old[e]; } else { let r = (e - M) * 3u * D; v = BCX[r + c] * BCX[r + 2u * D + c]; }
    S[i * D + c] = v;
  }
}`;

// q and k: RMS norm per 128-wide head, RoPE on pairs (l, l + 64) from the layer kind's cos/sin
// table [ctx, 64], k into the cache at its position; v copied into the cache. Grid (T, 16 + 4),
// one workgroup of 128 per head. QKVG rows: q (2048) | k (512) | v (512) | gate (16) | padding.
const QK = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read> QKVG: array<f32>;
@group(0) @binding(5) var<storage, read_write> Q: array<f32>;
@group(0) @binding(6) var<storage, read_write> Kc: array<f32>;
@group(0) @binding(7) var<storage, read_write> Vc: array<f32>;
var<workgroup> v: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x; let h = wid.y; let NH = 16u; let NKV = 4u; let RS = p.a.w;
  let pos = dn.n.z + t;
  let isq = h < NH;
  let src = t * RS + select(NH * 128u + (h - NH) * 128u, h * 128u, isq);
  let x = QKVG[src + l];
  red[l] = x * x;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let nw = bitcast<f32>(W[select(p.a.y, p.a.x, isq) + l]);
  v[l] = x / sqrt(red[0] / 128.0 + p.f.x) * nw;
  workgroupBarrier();
  let i = l % 64u;
  let c = R[pos * 64u + i]; let s = R[p.b.x + pos * 64u + i];
  let a = v[i]; let b = v[i + 64u];
  let o = select(b * c + a * s, a * c - b * s, l < 64u);
  if (isq) {
    Q[(t * NH + h) * 128u + l] = o;
  } else {
    let kh = h - NH; let co = (pos * NKV + kh) * 128u;
    Kc[co + l] = o;
    Vc[co + l] = QKVG[t * RS + NH * 128u + NKV * 128u + kh * 128u + l];
  }
}`;

// Causal grouped-query attention over the cache, keys from max(0, pos - window + 1) (window 0: all),
// then times sigmoid(gate). Grid (T, 16), one workgroup of 128 (thread d sums column d of V).
const ATTN = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> Kc: array<f32>;
@group(0) @binding(4) var<storage, read> Vc: array<f32>;
@group(0) @binding(5) var<storage, read> QKVG: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
var<workgroup> sc: array<f32, ${MAX_CTX}>;
var<workgroup> q: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x; let h = wid.y; let NH = 16u; let NKV = 4u;
  let kvh = h / (NH / NKV);
  let pos = dn.n.z + t;
  let win = p.a.x;
  var j0 = 0u;
  if (win > 0u && pos + 1u > win) { j0 = pos + 1u - win; }
  q[l] = Q[(t * NH + h) * 128u + l];
  workgroupBarrier();
  var m = -3.0e38;
  for (var j = j0 + l; j <= pos; j += 128u) {
    let ko = (j * NKV + kvh) * 128u;
    var d = 0.0;
    for (var e = 0u; e < 128u; e++) { d += q[e] * Kc[ko + e]; }
    d *= p.f.x;
    sc[j - j0] = d; m = max(m, d);
  }
  red[l] = m;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (l < k) { red[l] = max(red[l], red[l + k]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var j = j0 + l; j <= pos; j += 128u) { let e = exp(sc[j - j0] - mx); sc[j - j0] = e; s += e; }
  red[l] = s;
  workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let tot = red[0];
  var acc = 0.0;
  for (var j = j0; j <= pos; j++) { acc += sc[j - j0] * Vc[(j * NKV + kvh) * 128u + l]; }
  let g = QKVG[t * p.a.w + 3072u + h];
  O[(t * NH + h) * 128u + l] = acc / tot / (1.0 + exp(-g));
}`;

// M[t, i] = silu(U[t, i]) * U[t, F + i] over T * F values (U rows are w1 | w3 outputs).
const SWIGLU = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> Uu: array<f32>;
@group(0) @binding(3) var<storage, read_write> M: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let F = p.a.x;
  let i = (wid.x + wid.y * 65535u) * 256u + l;
  if (i >= dn.n.x * F) { return; }
  let t = i / F; let c = i % F;
  let g = Uu[t * 2u * F + c];
  M[i] = g / (1.0 + exp(-g)) * Uu[t * 2u * F + F + c];
}`;

// cos, sin tables [MAX_CTX, 64] for a layer kind: frequency i < angles follows the full head
// dimension's progression; the rest are zero (cos 1, sin 0), which leaves those pairs unrotated.
function ropeTable(theta, angles, hd = 128) {
  const cos = new Float32Array(MAX_CTX * 64), sin = new Float32Array(MAX_CTX * 64);
  for (let pos = 0; pos < MAX_CTX; pos++) for (let i = 0; i < 64; i++) {
    const f = i < angles ? pos / Math.pow(theta, (2 * i) / hd) : 0;
    cos[pos * 64 + i] = Math.cos(f); sin[pos * 64 + i] = Math.sin(f);
  }
  const out = new Float32Array(MAX_CTX * 128);
  out.set(cos); out.set(sin, MAX_CTX * 64);
  return out;
}

// The weight file is hosted as parts (manifest `parts`: url, size); this joins their bodies into one
// stream for _load. `fetchPart(url)` may return a cached Response.
export function partsSource(parts, fetchPart = (u) => fetch(u)) {
  let k = 0, reader = null;
  return { body: new ReadableStream({
    async pull(ctrl) {
      for (;;) {
        if (!reader) {
          if (k >= parts.length) { ctrl.close(); return; }
          const res = await fetchPart(parts[k].url);
          if (!res.ok) throw new Error(`weights part ${k}: HTTP ${res.status}`);
          reader = res.body.getReader();
        }
        const { done, value } = await reader.read();
        if (done) { reader = null; k++; continue; }
        ctrl.enqueue(value); return;
      }
    },
  }) };
}

export class Lfm3Engine {
  static async create(device, manifest, url, opts = {}) {
    const e = new Lfm3Engine(device, manifest);
    await e._load(url, opts);
    e._build();
    e.pipes = await compile(device, e.shaders);
    for (const list of Object.values(e.lists)) for (const s of list) {
      s.bg = device.createBindGroup({ layout: e.pipes[s.kind].getBindGroupLayout(0), entries: s.entries, label: s.kind });
      delete s.entries;
    }
    await device.queue.onSubmittedWorkDone();
    return e;
  }

  constructor(device, m) {
    this.device = device; this.m = m; this.owned = []; this.past = 0;
    const own = { embed: EMBED, lmhead: GEMV, last: LAST, rot: ROT, gq1: gqWGSL(m.levels, 1), gq8: gqWGSL(m.levels, 8),
      rms: rmsWGSL(m.hidden, 0), rmsadd: rmsWGSL(m.hidden, 1), conv: CONV, conv_state: CONV_STATE, qk: QK, attn: ATTN, swiglu: SWIGLU };
    // layout "auto" drops bindings a shader never reads, and every step binds both uniforms
    for (const k in own) own[k] = own[k].replace(/(fn main\([^)]*\)[^{]*\{)/, "$1\n  _ = p.a.x; _ = dn.n.x;");
    this.shaders = own;
  }

  buf(bytes, label) {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage: U.STORAGE | U.COPY_DST | U.COPY_SRC, label });
    this.owned.push(b);
    return b;
  }

  upload(arr, label) { const b = this.buf(arr.byteLength, label); this.device.queue.writeBuffer(b, 0, arr); return b; }

  // Stream the weight file; every chunk goes straight to the buffers whose byte ranges it covers.
  async _load(url, { onProgress, source } = {}) {
    const m = this.m, dev = this.device;
    this.W = {};
    const B = m.buffers.slice().sort((a, b) => a.offset - b.offset);
    for (const b of B) this.W[b.name] = this.buf(b.size, b.name);
    const res = source ?? await fetch(url);
    if (res.ok === false) throw new Error(`weights: HTTP ${res.status}`);
    const reader = (res.body ?? res).getReader();
    let pos = 0, bi = 0, carry = new Uint8Array(0), sinceDrain = 0;
    for (;;) {
      const { done, value } = await reader.read();
      let chunk = value ?? new Uint8Array(0);
      if (carry.length) { const c = new Uint8Array(carry.length + chunk.length); c.set(carry); c.set(chunk, carry.length); chunk = c; }
      const use = done ? chunk.length : chunk.length - (chunk.length % 4);
      const end = pos + use;
      while (bi < B.length && B[bi].offset + B[bi].size <= pos) bi++;
      for (let k = bi; k < B.length && B[k].offset < end; k++) {
        const a = Math.max(pos, B[k].offset), b = Math.min(end, B[k].offset + B[k].size);
        if (b > a) dev.queue.writeBuffer(this.W[B[k].name], a - B[k].offset, chunk, a - pos, b - a);
      }
      pos = end; sinceDrain += use;
      carry = chunk.slice(use);
      onProgress?.(pos, m.file_bytes);
      if (sinceDrain > 32e6) { await dev.queue.onSubmittedWorkDone(); sinceDrain = 0; }
      if (done || pos >= m.file_bytes) { if (!done) reader.cancel().catch(() => {}); break; }
    }
    if (pos < m.file_bytes) throw new Error(`weights: stream ended at ${pos} of ${m.file_bytes} bytes`);
    await dev.queue.onSubmittedWorkDone();
  }

  _build() {
    const m = this.m, D = m.hidden, F = m.ffn, f = 4;
    this.dyn = this.device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST, label: "dyn" });
    this.owned.push(this.dyn);
    const maxN = Math.max(...m.layers.map((L) => (L.qkvg ?? L.in).N));
    this.b = {
      tok: this.buf(MAX_T * 4, "tok"), x: this.buf(MAX_T * D * f, "x"), h: this.buf(MAX_T * D * f, "h"), hr: this.buf(MAX_T * 7680 * f, "hr"),
      big: this.buf(MAX_T * maxN * f, "big"), y: this.buf(MAX_T * D * f, "y"), o: this.buf(MAX_T * D * f, "o"),
      q: this.buf(MAX_T * D * f, "q"), u: this.buf(MAX_T * 2 * F * f, "u"), mm: this.buf(MAX_T * F * f, "mm"),
      last: this.buf(D * f, "last"), lastn: this.buf(D * f, "lastn"), logits: this.buf(m.vocab * f, "logits"),
      signs: this.upload(new Float32Array(m.signs), "signs"),
      rope_full: this.upload(ropeTable(m.rope.full_attention.theta, m.rope.full_attention.angles), "rope_full"),
      rope_sliding: this.upload(ropeTable(m.rope.sliding_attention.theta, m.rope.sliding_attention.angles), "rope_sliding"),
    };
    this.read = this.device.createBuffer({ size: m.vocab * f, usage: U.MAP_READ | U.COPY_DST, label: "readback" });
    this.owned.push(this.read);
    const kv = MAX_CTX * m.kv_heads * m.head_dim * f;
    this.state = m.layers.map((L, i) => L.kind === "conv" ? { conv: this.buf(6 * D * f, `conv${i}`) }
      : { kc: this.buf(kv, `kc${i}`), vc: this.buf(kv, `vc${i}`) });
    this.lists = { prefill: this._record(false), decode: this._record(true) };
  }

  step(kind, params, bufs, grid) {
    const pad = (x) => (x ?? []).concat(Array(4 - (x ?? []).length).fill(0));
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 8).set(pad(params.a).concat(pad(params.b)));
    new Float32Array(u, 32, 4).set(pad(params.f));
    const ub = this.device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST, label: kind });
    this.device.queue.writeBuffer(ub, 0, u);
    this.owned.push(ub);
    const entries = [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: this.dyn } }]
      .concat(bufs.map((x, i) => ({ binding: i + 2, resource: { buffer: x } })));
    this.steps.push({ kind, entries, grid });
  }

  _record(single) {
    const m = this.m, b = this.b, W = this.W, D = m.hidden, F = m.ffn, NH = m.heads, NKV = m.kv_heads;
    this.steps = [];
    const T = (n) => n[0];
    const rot = (X, Y, K) => this.step("rot", { a: [K] }, [b.signs, X, Y], (n) => [K / 256, T(n)]);
    const gq = (spec, X, Y) => this.step(single ? "gq1" : "gq8", { a: [0, spec.N * spec.K / 8, 0, spec.N], b: [spec.K] },
      [W[spec.buf], X, Y], (n) => [spec.N / 32, single ? 1 : Math.ceil(T(n) / 8)]);
    const rms = (kind, wbuf, off, X, Y, sel = 0) => this.step(kind, { a: [off], b: [sel, 1], f: [m.eps] }, [wbuf, X, Y], (n) => split(n[sel]));

    this.step("embed", { a: [m.embed.q, m.embed.s, m.embed.z, D] }, [W.embed, b.tok, b.x], (n) => [T(n)]);
    m.layers.forEach((L, i) => {
      const nw = W[L.norms], st = this.state[i];
      rms("rms", nw, 0, b.x, b.h);
      rot(b.h, b.hr, D);
      if (L.kind === "conv") {
        gq(L.in, b.hr, b.big);
        this.step("conv", { a: [4 * D] }, [nw, st.conv, b.big, b.y], (n) => [D / 64, T(n)]);
        this.step("conv_state", {}, [st.conv, b.big], () => [D / 64]);
      } else {
        gq(L.qkvg, b.hr, b.big);
        const rope = L.kind === "sliding_attention" ? b.rope_sliding : b.rope_full;
        this.step("qk", { a: [4 * D, 4 * D + m.head_dim, 0, L.qkvg.N], b: [MAX_CTX * 64], f: [m.eps] },
          [nw, rope, b.big, b.q, st.kc, st.vc], (n) => [T(n), NH + NKV]);
        this.step("attn", { a: [L.kind === "sliding_attention" ? m.window : 0, 0, 0, L.qkvg.N], f: [m.head_dim ** -0.5] },
          [b.q, st.kc, st.vc, b.big, b.y], (n) => [T(n), NH]);
      }
      rot(b.y, b.hr, D);
      gq(L.out, b.hr, b.o);
      rms("rmsadd", nw, D, b.o, b.x);
      rms("rms", nw, 2 * D, b.x, b.h);
      rot(b.h, b.hr, D);
      gq(L.w13, b.hr, b.u);
      this.step("swiglu", { a: [F] }, [b.u, b.mm], (n) => split(Math.ceil(T(n) * F / 256)));
      rot(b.mm, b.hr, F);
      gq(L.w2, b.hr, b.o);
      rms("rmsadd", nw, 3 * D, b.o, b.x);
    });
    this.step("last", { a: [D] }, [b.x, b.last], () => [D / 256]);
    rms("rms", W.final, 0, b.last, b.lastn, 1);
    this.step("lmhead", { a: [m.embed.q, m.embed.s, m.embed.z, m.vocab], b: [D, 0, NONE, 0] }, [W.embed, b.lastn, b.logits], () => split(m.vocab / 32));
    const s = this.steps; this.steps = null;
    return s;
  }

  reset() {
    this.past = 0;
    const enc = this.device.createCommandEncoder();
    for (const st of this.state) if (st.conv) enc.clearBuffer(st.conv);
    this.device.queue.submit([enc.finish()]);
  }

  async forward(ids, { read = true } = {}) {
    if (!ids.length) throw new Error("forward: no token ids");
    if (this.past + ids.length > MAX_CTX) throw new Error(`context full (${MAX_CTX} tokens)`);
    for (let o = 0; o < ids.length; o += MAX_T) {
      const chunk = ids.slice(o, o + MAX_T), n = chunk.length, final = o + MAX_T >= ids.length;
      const q = this.device.queue;
      q.writeBuffer(this.b.tok, 0, new Uint32Array(chunk));
      q.writeBuffer(this.dyn, 0, new Uint32Array([n, 1, this.past, 0]));
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const s of (n === 1 ? this.lists.decode : this.lists.prefill)) {
        pass.setPipeline(this.pipes[s.kind]); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.grid([n, 1]));
      }
      pass.end();
      if (final && read) enc.copyBufferToBuffer(this.b.logits, 0, this.read, 0, this.m.vocab * 4);
      q.submit([enc.finish()]);
      this.past += n;
    }
    if (!read) return null;
    await this.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.read.getMappedRange().slice(0));
    this.read.unmap();
    return out;
  }

  destroy() { for (const x of this.owned) x.destroy(); this.owned = []; }
}
