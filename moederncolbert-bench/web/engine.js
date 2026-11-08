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
// expert's list, and grouped matmuls walk only those rows. Those MoE ops live in _lib/wgsl/kernels.js
// (kernel library step 4); the hash embedding, RMSNorm and 512-token attention are still this file's.
// The 115M size's gated-conv layers are not implemented.

import { BYTE, HEADER, SeqEngine } from "../../_lib/wgsl/runtime.js";
import { KERNELS, combine, expertMatmul, gated, l2tok, matmul, matmulTiled, rope, ropeTable, route } from "../../_lib/wgsl/kernels.js";

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

};

const rows2d = (n) => (n <= 65535 ? [n] : [65535, Math.ceil(n / 65535)]);

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
    this.rope = this.upload(ropeTable([c.rope_theta], hd, MAX_LEN), "rope");

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
      ? matmulTiled(this, W, X, Y, { wOff, N, K, cnt: b.cnt, list: b.list, accumulate: opts.accumulate })
      : matmul(this, { fmt: "f32", offs: [wOff], N, K }, W, X, Y, opts);
    const gmm = (wOff, N, K, X, Y, mode) => expertMatmul(this, W, b.cnt, b.list, X, Y, { wOff, N, K, E, cap, mode, kernel: tiled ? "tiled" : "rows" });
    const rms = (wOff, X, Y) => step("rms", { a: [H, wOff], f: [eps] }, [W, X, Y], (n) => rows2d(n[0]));

    step("embed", { a: [H, P, off.table, off.proj], b: [off.lnw, off.lnb], f: [1e-5] }, [W, b.tok, b.x], (n) => rows2d(n[0]));
    for (const L of off.layers) {
      rms(L.anorm, b.x, b.a);
      mm(L.q, H, H, b.a, b.q); mm(L.k, nkv * hd, H, b.a, b.k); mm(L.v, nkv * hd, H, b.a, b.v);
      rope(this, this.rope, b.q, { nh, hd, maxSeq: MAX_LEN });
      rope(this, this.rope, b.k, { nh: nkv, hd, maxSeq: MAX_LEN });
      step("attn", { a: [nh, nkv], f: [1 / Math.sqrt(hd)] }, [b.tok, b.seqs, b.q, b.k, b.v, b.o], (n) => [n[0], nh]);
      mm(L.o, H, H, b.o, b.x, { accumulate: true });
      rms(L.fnorm, b.x, b.a);
      route(this, W, b.a, b.cnt, b.list, b.gw, { H, wOff: L.router, E, cap });
      gmm(L.gu, 2 * He, H, b.a, b.u, 1);
      gated(this, b.u, b.u, b.g, { I: He, sa: 2 * He, sb: 2 * He, ob: He, mult: 2 });
      gmm(L.down, H, He, b.g, b.od, 2);
      combine(this, b.od, b.gw, b.x, { H });
      // shared expert, accumulated straight into the residual
      mm(L.sgu, 2 * He, H, b.a, b.u);
      gated(this, b.u, b.u, b.g, { I: He, sa: 2 * He, sb: 2 * He, ob: He });
      mm(L.sdown, H, He, b.g, b.x, { accumulate: true });
    }
    rms(off.final, b.x, b.a);
    mm(off.dense, 128, H, b.a, b.d);
    l2tok(this, b.d, b.out, { D: 128 });
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
