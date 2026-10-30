// LFM2 decoder (LFM2.5-350M) on hand-written WGSL. Reads onnx-community's model_q4.onnx_data as
// a stream and writes each chunk straight into per-operator GPU buffers (web/manifest.json maps
// file ranges to buffers), so the page never holds the weights in JS memory: the peak is the GPU
// copy plus one network chunk. The q4 matmul is the shared MatMulNBits kernel in _lib/wgsl.
//
//   const eng = await LfmEngine.create(device, manifest, url, { onProgress });
//   eng.reset();
//   let logits = await eng.forward(promptIds);   // prefill; Float32Array(vocab) for the last token
//   logits = await eng.forward([next]);          // one decode step
//
// Layer kinds: "conv" (in_proj -> B, C, x; y = C * causal depthwise conv3(B * x) -> out_proj) and
// "full_attention" (q/k/v, per-head RMS norm on q and k, RoPE, grouped-query attention over a KV
// cache, o_proj). Every layer ends in a SwiGLU feed-forward block. Activations are f32.

import { HEADER, BYTE, NONE, compile, split } from "../../_lib/wgsl/runtime.js";
import { KERNELS, matmul } from "../../_lib/wgsl/kernels.js";

const U = GPUBufferUsage;
export const MAX_T = 128;     // tokens per forward pass; longer prompts go in chunks
export const MAX_CTX = 2048;  // KV cache length (the attention kernel keeps one score per key in workgroup memory)

// Token embedding from the 4-bit table: (q - zero point) * block scale. Grid: one workgroup per token.
const EMBED = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> tok: array<u32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${BYTE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let D = p.a.w; let r = wid.x; let id = tok[r];
  let nb = D / 32u; let zrow = (nb + 1u) / 2u;
  for (var k = l; k < D; k += 256u) {
    let q = f32((wbyte(p.a.x * 4u + id * (D / 2u) + k / 2u) >> (4u * (k % 2u))) & 15u);
    let b = k / 32u;
    let z = f32((wbyte(p.a.z * 4u + id * zrow + b / 2u) >> (4u * (b % 2u))) & 15u);
    Y[r * D + k] = (q - z) * bitcast<f32>(W[p.a.y + id * nb + b]);
  }
}`;

// RMS norm of rows of D values: y = x / sqrt(mean(x^2) + eps) * w. Rows: dn.n[p.b.x] * p.b.y.
function rmsWGSL(D) {
  const WG = Math.min(D, 256), per = D / WG;
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
  for (var i = 0u; i < ${per}u; i++) { let j = l + ${WG}u * i; y[base + j] = v[i] * inv * bitcast<f32>(W[p.a.x + j]); }
}`;
}

// Single-row q4 matmul for decode steps: the shared kernel's layout (32 columns x 8 k-splits per
// workgroup) without its 8-row tile, which at one row would repeat every dot product 8 times.
const GEMV = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
${BYTE}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let N = p.a.w; let K = p.b.x;
  let nl = l % 32u; let ks = l / 32u;
  let n = (wid.x + wid.y * 65535u) * 32u + nl;
  let nb = K / 32u; let bps = nb / 8u; let zrow = (nb + 1u) / 2u;
  var acc = 0.0;
  if (n < N) {
    for (var b = ks * bps; b < (ks + 1u) * bps; b++) {
      let sc = bitcast<f32>(W[p.a.y + n * nb + b]);
      let z = f32((wbyte(p.a.z * 4u + n * zrow + b / 2u) >> (4u * (b % 2u))) & 15u);
      var part = 0.0;
      for (var w = 0u; w < 4u; w++) {
        let word = W[p.a.x + n * (K / 8u) + b * 4u + w];
        let lo = vec4<f32>(f32(word & 15u), f32((word >> 4u) & 15u), f32((word >> 8u) & 15u), f32((word >> 12u) & 15u)) - z;
        let hi = vec4<f32>(f32((word >> 16u) & 15u), f32((word >> 20u) & 15u), f32((word >> 24u) & 15u), f32(word >> 28u)) - z;
        let k4 = b * 8u + w * 2u;
        part += dot(lo, X[k4]) + dot(hi, X[k4 + 1u]);
      }
      acc += sc * part;
    }
  }
  red[l] = acc;
  workgroupBarrier();
  if (ks != 0u || n >= N) { return; }
  var s = 0.0;
  for (var j = 0u; j < 8u; j++) { s += red[j * 32u + nl]; }
  if ((p.b.w & 2u) != 0u) { Y[n] += s; } else { Y[n] = s; }
}`;

// Short convolution, output: y[t, c] = C[t, c] * sum_j w[c, j] * ext[t + 1 + j], where ext is the
// 3 cached values of B * x followed by this pass's B * x (the exported graph keeps the last T of
// the T + 1 conv outputs). BCX rows are [B | C | x], 3 * D wide.
const CONV = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read> BCX: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
const D = 1024u;
fn ext(i: u32, c: u32) -> f32 {
  if (i < 3u) { return S[i * D + c]; }
  let r = (i - 3u) * 3u * D;
  return BCX[r + c] * BCX[r + 2u * D + c];
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let c = wid.x * 64u + l; let t = wid.y;
  var o = 0.0;
  for (var j = 0u; j < 3u; j++) { o += bitcast<f32>(W[p.a.x + c * 3u + j]) * ext(t + 1u + j, c); }
  Y[t * D + c] = BCX[t * 3u * D + D + c] * o;
}`;

// Conv state after the pass: the last 3 values of ext. Each thread owns one channel.
const CONV_STATE = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
@group(0) @binding(3) var<storage, read> BCX: array<f32>;
const D = 1024u;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let c = wid.x * 64u + l; let T = dn.n.x;
  let old = array<f32, 3>(S[c], S[D + c], S[2u * D + c]);
  var nw: array<f32, 3>;
  for (var i = 0u; i < 3u; i++) {
    let e = T + i;
    if (e < 3u) { nw[i] = old[e]; } else { let r = (e - 3u) * 3u * D; nw[i] = BCX[r + c] * BCX[r + 2u * D + c]; }
  }
  for (var i = 0u; i < 3u; i++) { S[i * D + c] = nw[i]; }
}`;

// RoPE (non-interleaved halves, cos/sin rows from the exported tables) on q in place and on k,
// then k and v into the cache at positions past .. past + T - 1. Cache rows: [kv head][64].
const ROPE_KV = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> R: array<f32>;
@group(0) @binding(3) var<storage, read_write> Q: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read> V: array<f32>;
@group(0) @binding(6) var<storage, read_write> Kc: array<f32>;
@group(0) @binding(7) var<storage, read_write> Vc: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let t = wid.x; let h = wid.y; let NH = p.a.z; let NKV = p.a.w;
  let pos = dn.n.z + t;
  let c = R[p.a.x + pos * 32u + i]; let s = R[p.a.y + pos * 32u + i];
  let qo = (t * NH + h) * 64u;
  let q1 = Q[qo + i]; let q2 = Q[qo + i + 32u];
  Q[qo + i] = q1 * c - q2 * s; Q[qo + i + 32u] = q2 * c + q1 * s;
  if (h < NKV) {
    let ko = (t * NKV + h) * 64u; let co = (pos * NKV + h) * 64u;
    let k1 = K[ko + i]; let k2 = K[ko + i + 32u];
    Kc[co + i] = k1 * c - k2 * s; Kc[co + i + 32u] = k2 * c + k1 * s;
    Vc[co + i] = V[ko + i]; Vc[co + i + 32u] = V[ko + i + 32u];
  }
}`;

// Causal grouped-query attention over the cache. Grid (T, heads), one workgroup of 64 per query
// head: scores for every visible key in workgroup memory, softmax, then thread d sums column d of V.
const ATTN = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> Kc: array<f32>;
@group(0) @binding(4) var<storage, read> Vc: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
var<workgroup> sc: array<f32, ${MAX_CTX}>;
var<workgroup> q: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x; let h = wid.y; let NH = p.a.z; let NKV = p.a.w;
  let kvh = h / (NH / NKV);
  let n = dn.n.z + t + 1u;
  q[l] = Q[(t * NH + h) * 64u + l];
  workgroupBarrier();
  var m = -3.0e38;
  for (var j = l; j < n; j += 64u) {
    let ko = (j * NKV + kvh) * 64u;
    var d = 0.0;
    for (var e = 0u; e < 64u; e++) { d += q[e] * Kc[ko + e]; }
    d *= p.f.x;
    sc[j] = d; m = max(m, d);
  }
  red[l] = m;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (l < k) { red[l] = max(red[l], red[l + k]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var j = l; j < n; j += 64u) { let e = exp(sc[j] - mx); sc[j] = e; s += e; }
  red[l] = s;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (l < k) { red[l] += red[l + k]; } workgroupBarrier(); }
  let tot = red[0];
  var acc = 0.0;
  for (var j = 0u; j < n; j++) { acc += sc[j] * Vc[(j * NKV + kvh) * 64u + l]; }
  O[(t * NH + h) * 64u + l] = acc / tot;
}`;

// g = silu(g) * u over T * F values.
const SWIGLU = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
@group(0) @binding(3) var<storage, read> Up: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = (wid.x + wid.y * 65535u) * 256u + l;
  if (i >= dn.n.x * p.a.x) { return; }
  let g = G[i];
  G[i] = g / (1.0 + exp(-g)) * Up[i];
}`;

// Copy the last token's row of X into L.
const LAST = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> L: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = wid.x * 256u + l;
  L[i] = X[(dn.n.x - 1u) * p.a.x + i];
}`;

export class LfmEngine {
  // url: the model_q4.onnx_data to stream (or a Response / ReadableStream source via opts.source).
  static async create(device, manifest, url, opts = {}) {
    const e = new LfmEngine(device, manifest);
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
    this.device = device; this.m = m; this.owned = [];
    // layout "auto" drops bindings a shader never reads, and every step binds both uniforms, so
    // each of this file's kernels touches p and dn once.
    const own = { embed: EMBED, rms1024: rmsWGSL(m.hidden), rms64: rmsWGSL(m.head_dim),
      gemv: GEMV, conv: CONV, conv_state: CONV_STATE, rope_kv: ROPE_KV, attn: ATTN, swiglu: SWIGLU, last: LAST };
    for (const k in own) own[k] = own[k].replace(/(fn main\([^)]*\)[^{]*\{)/, "$1\n  _ = p.a.x; _ = dn.n.x;");
    this.shaders = { matmul_q4: KERNELS.matmul_q4, ...own };
    this.past = 0;
  }

  buf(bytes, label, usage = 0) {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage: U.STORAGE | U.COPY_DST | U.COPY_SRC | usage, label });
    this.owned.push(b);
    return b;
  }

  // Stream the weight file into the per-operator buffers. Chunks are cut at 4-byte file positions
  // (every piece starts and ends on one), and every 32 MB the queue is drained so writeBuffer's
  // staging copies are released before more arrive.
  async _load(url, { onProgress, source } = {}) {
    const m = this.m, dev = this.device;
    this.W = {};
    for (const [name, size] of Object.entries(m.buffers)) this.W[name] = this.buf(size, name);
    for (const t of m.inline) dev.queue.writeBuffer(this.W[t.buf], t.dst, new Float32Array(t.f32));
    const res = source ?? await fetch(url);
    if (res.ok === false) throw new Error(`weights: HTTP ${res.status}`);
    const total = Number(res.headers?.get?.("content-length")) || m.file_bytes;
    const reader = (res.body ?? res).getReader();
    const P = m.pieces;
    let pos = 0, pi = 0, carry = new Uint8Array(0), sinceDrain = 0;
    const consume = (bytes, start) => {        // bytes covers file [start, start + bytes.length), aligned
      const end = start + bytes.length;
      while (pi < P.length && P[pi].offset + P[pi].length <= start) pi++;
      for (let k = pi; k < P.length && P[k].offset < end; k++) {
        const a = Math.max(start, P[k].offset), b = Math.min(end, P[k].offset + P[k].length);
        if (b > a) dev.queue.writeBuffer(this.W[P[k].buf], P[k].dst + (a - P[k].offset), bytes, a - start, b - a);
      }
    };
    for (;;) {
      const { done, value } = await reader.read();
      let chunk = value ?? new Uint8Array(0);
      if (carry.length) { const c = new Uint8Array(carry.length + chunk.length); c.set(carry); c.set(chunk, carry.length); chunk = c; }
      const use = done ? chunk.length : chunk.length - (chunk.length % 4);
      if (use) consume(chunk.subarray(0, use), pos);
      pos += use; sinceDrain += use;
      carry = chunk.slice(use);
      onProgress?.(pos, total);
      if (sinceDrain > 32e6) { await dev.queue.onSubmittedWorkDone(); sinceDrain = 0; }
      if (done || pos >= m.file_bytes) { if (!done) reader.cancel().catch(() => {}); break; }
    }
    if (pos < m.file_bytes) throw new Error(`weights: stream ended at ${pos} of ${m.file_bytes} bytes`);
    await dev.queue.onSubmittedWorkDone();
  }

  _build() {
    const m = this.m, D = m.hidden, F = m.ffn, NH = m.heads, NKV = m.kv_heads, HD = m.head_dim;
    const f = 4;
    this.dyn = this.device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST, label: "dyn" });
    this.owned.push(this.dyn);
    const b = this.b = {
      tok: this.buf(MAX_T * 4, "tok"), x: this.buf(MAX_T * D * f, "x"), h: this.buf(MAX_T * D * f, "h"),
      bcx: this.buf(MAX_T * 3 * D * f, "bcx"), y: this.buf(MAX_T * D * f, "y"),
      q: this.buf(MAX_T * D * f, "q"), k: this.buf(MAX_T * NKV * HD * f, "k"), v: this.buf(MAX_T * NKV * HD * f, "v"),
      qn: this.buf(MAX_T * D * f, "qn"), kn: this.buf(MAX_T * NKV * HD * f, "kn"),
      g: this.buf(MAX_T * F * f, "g"), u: this.buf(MAX_T * F * f, "u"),
      last: this.buf(D * f, "last"), lastn: this.buf(D * f, "lastn"),
      logits: this.buf(m.vocab * f, "logits"),
    };
    this.read = this.device.createBuffer({ size: m.vocab * f, usage: U.MAP_READ | U.COPY_DST, label: "readback" });
    this.owned.push(this.read);
    this.state = m.layers.map((L, i) => L.kind === "conv"
      ? { conv: this.buf(3 * D * f, `conv${i}`) }
      : { kc: this.buf(MAX_CTX * NKV * HD * f, `kc${i}`), vc: this.buf(MAX_CTX * NKV * HD * f, `vc${i}`) });
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

  // One list of dispatches for a pass of T tokens; `single` records the decode variant (T = 1).
  _record(single) {
    const m = this.m, b = this.b, W = this.W, D = m.hidden, NH = m.heads, NKV = m.kv_heads;
    this.steps = [];
    const T = (n) => n[0];
    const mm = (w, X, Y, accumulate = false) => {
      if (single) this.step("gemv", { a: [w.q, w.s, w.z, w.N], b: [w.K, 0, NONE, accumulate ? 2 : 0] }, [W[w.buf], X, Y], () => split(w.N / 32));
      else matmul(this, { fmt: "q4", offs: [w.q, w.s, w.z], N: w.N, K: w.K }, W[w.buf], X, Y, { accumulate });
    };
    const rms = (wbuf, off, X, Y, rows, Dn, eps, sel = 0) =>
      this.step(Dn === D ? "rms1024" : "rms64", { a: [off], b: [sel, rows], f: [eps] }, [wbuf, X, Y], (n) => split(n[sel] * rows));

    this.step("embed", { a: [m.embed.q, m.embed.s, m.embed.z, D] }, [W.embed, b.tok, b.x], (n) => [T(n)]);
    m.layers.forEach((L, i) => {
      const nw = W[L.norms], st = this.state[i];
      rms(nw, L.op_norm, b.x, b.h, 1, D, m.eps);
      if (L.kind === "conv") {
        mm(L.in_proj, b.h, b.bcx);
        this.step("conv", { a: [L.conv_w] }, [nw, st.conv, b.bcx, b.y], (n) => [D / 64, T(n)]);
        this.step("conv_state", {}, [st.conv, b.bcx], () => [D / 64]);
        mm(L.out_proj, b.y, b.x, true);
      } else {
        mm(L.q, b.h, b.q); mm(L.k, b.h, b.k); mm(L.v, b.h, b.v);
        rms(nw, L.q_norm, b.q, b.qn, NH, m.head_dim, m.eps);
        rms(nw, L.k_norm, b.k, b.kn, NKV, m.head_dim, m.eps);
        this.step("rope_kv", { a: [m.rope.cos, m.rope.sin, NH, NKV] }, [W.rope, b.qn, b.kn, b.v, st.kc, st.vc], (n) => [T(n), NH]);
        this.step("attn", { a: [0, 0, NH, NKV], f: [m.scale] }, [b.qn, st.kc, st.vc, b.y], (n) => [T(n), NH]);
        mm(L.o, b.y, b.x, true);
      }
      rms(nw, L.ffn_norm, b.x, b.h, 1, D, m.eps);
      mm(L.gate, b.h, b.g); mm(L.up, b.h, b.u);
      this.step("swiglu", { a: [L.gate.N] }, [b.g, b.u], (n) => split(Math.ceil(T(n) * L.gate.N / 256)));
      mm(L.down, b.g, b.x, true);
    });
    this.step("last", { a: [D] }, [b.x, b.last], () => [D / 256]);
    rms(W.final, m.final_norm, b.last, b.lastn, 1, D, m.final_eps, 1);
    this.step("gemv", { a: [m.embed.q, m.embed.s, m.embed.z, m.vocab], b: [D, 0, NONE, 0] }, [W.embed, b.lastn, b.logits], () => split(m.vocab / 32));
    const s = this.steps; this.steps = null;
    return s;
  }

  reset() {
    this.past = 0;
    const enc = this.device.createCommandEncoder();
    for (const st of this.state) if (st.conv) enc.clearBuffer(st.conv);
    this.device.queue.submit([enc.finish()]);
  }

  // Run ids (appended after the cached context) and return the last position's logits.
  async forward(ids, { read = true, nSteps = Infinity } = {}) {
    if (!ids.length) throw new Error("forward: no token ids");
    if (this.past + ids.length > MAX_CTX) throw new Error(`context full (${MAX_CTX} tokens)`);
    for (let o = 0; o < ids.length; o += MAX_T) {
      const chunk = ids.slice(o, o + MAX_T), n = chunk.length, final = o + MAX_T >= ids.length;
      const q = this.device.queue;
      q.writeBuffer(this.b.tok, 0, new Uint32Array(chunk));
      q.writeBuffer(this.dyn, 0, new Uint32Array([n, 1, this.past, 0]));
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const s of (n === 1 ? this.lists.decode : this.lists.prefill).slice(0, nSteps)) {
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

  // Debugging: copy `count` floats of activation buffer b[name] after the last forward.
  async probe(name, count) {
    const rb = this.device.createBuffer({ size: count * 4, usage: U.MAP_READ | U.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.b[name], 0, rb, 0, count * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.destroy();
    return out;
  }

  destroy() { for (const x of this.owned) x.destroy(); this.owned = []; }
}
