// WebGPU engine, in hand-written WGSL, for linkup-sparseup-embed-v2: a SPLADE head on ModernBERT-base
// (22 layers, hidden 768, 12 heads, GLU 1152, global attention every third layer, sliding window 128
// elsewhere). It reads a pack written by pack.py (manifest.json + weights.bin) and returns, per text,
// the folded sparse vector the model card's encode() returns: [vpad] f32, zero outside ~34k active dims.
//
//   const eng = await SparseEngine.create(device, manifest, weightsBytes, { maxTokens: 512, maxSeq: 512 });
//   const vecs = await eng.embed([{ ids, pool }, ...]);   // Float32Array[B * vpad]
//
// pool[j] = 0 for the prefix token ("[Q]" / "[D]"), which is attended but not pooled; CLS/SEP are pooled.
//
// The encoder is the ettin-native recipe on the shared kernels (_lib/wgsl/kernels.js): t8 token-table
// gather, LayerNorm, int8 or q4 matmuls, RoPE with two thetas, attentionLong (sequences to 512 here),
// the GELU GLU. New here:
//   - the MLM head (dense + GELU fused into the matmul epilogue, then LayerNorm) and the tied decoder,
//     an int8 matmul over the same token-table bytes the gather reads (50,400 padded rows), with its bias;
//   - `splade`: one workgroup per token computes w = log1p(relu(logit - shift) / sigma) on the fly,
//     keeps each thread's top-12 in registers, finds the row's exact 12th-largest w by a radix select
//     over the candidates' float bits, then scatters every w >= cutoff into the output with atomicMax
//     at fold[v]. Max-pool over positions and the vocab fold are both maxima, so one atomicMax on the
//     folded slot does both. w >= 0, so its f32 bits order like u32.
// The weights live in two GPU buffers (pack regions "main" and "layers") so each binding stays under
// the default 128 MiB maxStorageBufferBindingSize. Logits take maxTokens * 50,400 * 4 bytes
// (103 MB at 512 tokens).

import { HEADER, NONE, SeqEngine, split } from "../../../_lib/wgsl/runtime.js";
import { KERNELS, attentionLong, embed, gated, matmul, norm, rope, ropeTable } from "../../../_lib/wgsl/kernels.js";

// Zero the output slots of the batch's sequences: n[1] * V u32.
const CLEAR = (V) => HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  _ = p.a.x;   // auto layout drops unread bindings
  let i = (wid.x + wid.y * 65535u) * 256u + l;
  if (i < dn.n.y * ${V}u) { out[i] = 0u; }
}`;

// p.a.y = fold word offset in W, p.f = (shift, sigma).
const SPLADE = (V, K) => HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> logit: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<atomic<u32>>;
var<workgroup> cnt: atomic<u32>;
const V = ${V}u;
const K = ${K}u;
// log1p(relu(x - shift) / sigma), with log1p(r) = log(u) * r / (u - 1), u = 1 + r (Goldberg).
fn act(x: f32) -> f32 {
  let r = max(x - p.f.x, 0.0) / p.f.y;
  let u = 1.0 + r;
  if (u == 1.0) { return r; }
  return log(u) * (r / (u - 1.0));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x + wid.y * 65535u;
  if (t >= dn.n.x) { return; }
  let row = t * V;
  // This thread's K largest w over columns l, l + 256, ... (descending; zeros fill, w >= 0).
  var top: array<f32, K>;
  for (var i = 0u; i < K; i++) { top[i] = 0.0; }
  for (var v = l; v < V; v += 256u) {
    let w = act(logit[row + v]);
    if (w > top[K - 1u]) {
      var j = K - 1u;
      loop {
        if (j == 0u || top[j - 1u] >= w) { break; }
        top[j] = top[j - 1u];
        j--;
      }
      top[j] = w;
    }
  }
  // Radix select: the largest bit pattern c with #{candidates >= c} >= K is the row's K-th largest w
  // (with multiplicity). The union of per-thread top-K lists gives the same answer as the full row:
  // a thread with K values >= c contributes K either way, any other contributes all of its values >= c.
  var pre = 0u;
  for (var b = 30i; b >= 0i; b--) {
    let tr = pre | (1u << u32(b));
    if (l == 0u) { atomicStore(&cnt, 0u); }
    workgroupBarrier();
    var c = 0u;
    for (var i = 0u; i < K; i++) { c += select(0u, 1u, bitcast<u32>(top[i]) >= tr); }
    if (c > 0u) { atomicAdd(&cnt, c); }
    workgroupBarrier();
    if (atomicLoad(&cnt) >= K) { pre = tr; }
    workgroupBarrier();
  }
  let cut = bitcast<f32>(pre);
  let tk = tok[t];
  if (tk.w == 0u) { return; }   // prefix token: attended, not pooled
  let ob = tk.z * V;
  for (var v = l; v < V; v += 256u) {
    let w = act(logit[row + v]);
    if (w >= cut && w > 0.0) { atomicMax(&out[ob + W[p.a.y + v]], bitcast<u32>(w)); }
  }
}`;

export class SparseEngine extends SeqEngine {
  // manifest: pack manifest.json; data: weights.bin bytes. opts.maxTokens bounds one batch (all
  // sequences together, default 512), opts.maxSeq one sequence (default 512, at most 1024), opts.maxSeqs
  // the sequence count (default 16).
  static create(device, manifest, data, opts = {}) {
    return super.create(device, structuredClone(manifest), data, { maxSeq: 512, maxSeqs: 16, ...opts });
  }

  constructor(device, manifest, opts) {
    super(device, manifest, opts);
    this.dim = manifest.vpad;
    if (this.maxSeq > 1024) throw new Error("maxSeq above 1024 (attentionLong's limit)");
  }

  async _build(bytes) {
    const m = this.m, T = this.maxTokens, B = this.maxSeqs, S = this.maxSeq;
    const H = m.hidden, nh = m.heads, D = H / nh, I = m.inter, V = m.vpad;
    if (H % 64 || D !== 64 || V % 32) throw new Error("unexpected manifest shape");
    const region = (name) => {
      const r = m.regions[name];
      return this.upload(bytes.subarray(r.offset, r.offset + r.length), `weights ${name}`);
    };
    this.Wm = region("main");
    this.Wl = region("layers");
    const buf = (n, label, extra) => this.buf(n, label, extra), U = GPUBufferUsage;
    const B_ = this.b = {
      tok: buf(T * 4, "tok", U.COPY_DST), seqs: buf(B * 4, "seqs", U.COPY_DST),
      e: buf(T * H, "e"), h: buf(T * H, "h"), a: buf(T * H, "a"), o: buf(T * H, "o"),
      qkv: buf(T * 3 * H, "qkv"), u: buf(T * 2 * I, "u"), g: buf(T * I, "g"),
      logits: buf(T * V, "logits"), out: buf(B * V, "out"),
    };
    this.shaders = { ...KERNELS, clear: CLEAR(V), splade: SPLADE(V, m.top_k) };
    this.ropeBuf = this.upload(ropeTable([m.theta_global, m.theta_local], D, S), "rope");

    const word = (t) => { if (t.offset % 4) throw new Error("unaligned tensor"); return t.offset / 4; };
    const mm = (w, X, Y, { gelu = false, accumulate = false, bias = NONE } = {}) => {
      const Wb = w.region === "main" ? this.Wm : this.Wl;
      const offs = w.fmt === "i8" ? [word(w.q), word(w.s)] : [word(w.q), word(w.s), word(w.zp)];
      matmul(this, { fmt: w.fmt, offs, N: w.N, K: w.K }, Wb, X, Y, { bias, gelu, accumulate, Wt: Wb });
    };
    const ln = (t, X, Y) => norm(this, "ln", this.Wm, word(t), X, Y, { D: H, eps: m.eps });

    embed(this, this.Wm, B_.e, { fmt: "t8", codes: m.table.codes.offset, scales: m.table.scales.offset }, { H });
    ln(m.emb_norm, B_.e, B_.h);
    for (const L of m.layers) {
      let X = B_.h;
      if (L.attn_norm) { ln(L.attn_norm, B_.h, B_.a); X = B_.a; }
      mm(L.Wqkv, X, B_.qkv);
      rope(this, this.ropeBuf, B_.qkv, { nh, hd: D, maxSeq: S, tables: 2, sel: L.global ? 0 : 1, parts: 2, stride: 3 * H });
      attentionLong(this, B_.qkv, B_.o, { nh, hd: D, window: L.global ? 0 : m.window });
      mm(L.Wo, B_.o, B_.h, { accumulate: true });
      ln(L.mlp_norm, B_.h, B_.a);
      mm(L.Wi, B_.a, B_.u);
      gated(this, B_.u, B_.u, B_.g, { I, sa: 2 * I, sb: 2 * I, ob: I });
      mm(L.Wo_mlp, B_.g, B_.h, { accumulate: true });
    }
    ln(m.final_norm, B_.h, B_.a);
    mm(m.head.dense, B_.a, B_.e, { gelu: true });
    ln(m.head.norm, B_.e, B_.a);
    this.step("clear", {}, [B_.out], (n) => split(Math.ceil(n[1] * V / 256)));
    mm({ fmt: "i8", region: "main", q: m.table.codes, s: m.table.scales, N: V, K: H }, B_.a, B_.logits, { bias: word(m.decoder_bias) });
    this.step("splade", { a: [0, word(m.fold)], f: [m.shift, m.sigma] }, [this.Wm, B_.tok, B_.logits, B_.out], (n) => split(n[0]));
  }

  // As SeqEngine._record, with each batch entry { ids, pool } (or a bare id list: all pooled) and the
  // pool flag in the token's fourth word.
  _record(enc, batch, nSteps = Infinity) {
    batch = batch.map((x) => (Array.isArray(x) || ArrayBuffer.isView(x) ? { ids: x, pool: null } : x));
    const T = batch.reduce((s, x) => s + x.ids.length, 0), B = batch.length;
    if (!B || T > this.maxTokens || B > this.maxSeqs) throw new Error(`batch of ${B} sequences / ${T} tokens exceeds the engine's limits`);
    const tok = new Uint32Array(T * 4), seqs = new Uint32Array(B * 4);
    let o = 0;
    batch.forEach(({ ids, pool }, b) => {
      if (!ids.length || ids.length > this.maxSeq) throw new Error(`sequence length ${ids.length} not in 1..${this.maxSeq}`);
      seqs.set([o, ids.length], b * 4);
      for (let j = 0; j < ids.length; j++) tok.set([ids[j], j, b, pool ? pool[j] : 1], (o + j) * 4);
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
    return B;
  }
}

// Folded dense rows -> [{ id: weight }] sparse maps, optionally keeping each row's topK largest.
export function toSparse(vecs, V, B, topK = 0) {
  const out = [];
  for (let b = 0; b < B; b++) {
    let ent = [];
    for (let v = 0; v < V; v++) { const w = vecs[b * V + v]; if (w > 0) ent.push([v, w]); }
    if (topK && ent.length > topK) ent = ent.sort((x, y) => y[1] - x[1]).slice(0, topK);
    out.push(Object.fromEntries(ent));
  }
  return out;
}
