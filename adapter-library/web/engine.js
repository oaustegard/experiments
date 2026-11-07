// SpikeEngine: one frozen Ettin-17m base with swappable LoRA spikes, in hand-written WGSL.
//
// browser-decider's DeciderEngine (copied below, its comments kept) plus three things for the
// adapter-library system: (1) every Wqkv / Wo / Wi / Wo_mlp matmul gets a LoRA side path,
// y += B (A x), as two f32 matmuls reading a separate buffer, with the rank padded to 32 for the
// kernel's tile; setSpike() rewrites that buffer and the JS decision head, so switching spikes costs one
// buffer write. (2) score() can also return a pooled, L2-normalised key: the mean of the final hidden
// states over a token range (the state), the kNN memory's key. (3) maxPos defaults to 2048 so the
// pooled rows fit the gather. run/pack_system.py's LoraRef is the NumPy spec.
//
// Original header:
// DeciderEngine: Ettin-17m (ModernBERT) as a typed-decision cross-encoder, in hand-written WGSL.
//
//   const eng = await DeciderEngine.create(device, manifest, weightsBin, { maxTokens: 4096, maxSeqs: 16 });
//   const logits = await eng.score([{ ids, pos }, ...]);   // Float32Array per sequence, one logit per [MASK]
//
// This is ettin-native's SmallEngine (web/engine.js there) with three changes: sequences up to 1024
// tokens (SeqEngine's maxSeq option), attention that loops over keys beyond one workgroup
// (attentionLong in _lib/wgsl/kernels.js), and, in place of mean pooling + dense + l2, a gather of the
// final hidden state at each option's [MASK] position. The decision head (256 -> 256 GELU, LayerNorm,
// -> 1) runs in JS on those few rows. run/pack.py writes the weights and its `Ref` class is the
// NumPy spec this follows.

import { SeqEngine } from "../../_lib/wgsl/runtime.js";
import { KERNELS, attentionLong, embed, gated, matmul, norm, rope, ropeTable } from "../../_lib/wgsl/kernels.js";

const MAX_SEQ = 1024;   // decisions run to ~700 tokens; attentionLong holds 1024 keys

const NONE = 0xffffffff;
const HEADER = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, f: vec4<f32> };
struct Dyn { n: vec4<u32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> dn: Dyn;
`;
const BYTE = /* wgsl */ `
fn wbyte(i: u32) -> u32 { return (W[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
fn wf16(byteOff: u32) -> f32 { return unpack2x16float(W[byteOff >> 2u])[(byteOff >> 1u) & 1u]; }
`;

// Same erf approximation as the WGSL GELU (Abramowitz-Stegun 7.1.26).
function gelu(x) {
  const a = Math.abs(x) * 0.7071067811865476, t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return 0.5 * x * (1 + Math.sign(x) * y);
}

const LR = 32;   // LoRA rank as stored on the GPU (the spikes' rank 8, zero-padded to the matmul tile)

function f16(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

export class SpikeEngine extends SeqEngine {
  static create(device, manifest, data, opts = {}) {
    return super.create(device, structuredClone(manifest), data, opts);
  }

  constructor(device, manifest, opts) {
    super(device, manifest, { ...opts, maxSeq: MAX_SEQ });
    this.maxPos = opts.maxPos ?? 2048;  // option positions plus pooled state rows per batch
    this.dim = manifest.hidden;
  }

  async _build(bytes) {
    const m = this.m, T = this.maxTokens;
    const H = m.hidden, nh = m.heads, D = H / nh, I = m.inter;
    if (m.arch !== "modernbert" || m.table.bits !== 4 || H % 32) throw new Error("unexpected manifest");
    bytes = this.uploadPacked(bytes);
    const f32 = (t) => new Float32Array(bytes.buffer, bytes.byteOffset + t.offset, t.length / 4).slice();
    this.head = null;   // set by setSpike()

    const U = GPUBufferUsage, buf = (n, label, extra) => this.buf(n, label, extra);
    const half = D / 2;
    this.rope = this.upload(ropeTable([m.theta_global, m.theta_local], D, MAX_SEQ), "rope");
    const B_ = {
      tok: buf(T * 4, "tok", U.COPY_DST), seqs: buf(this.maxSeqs * 4, "seqs", U.COPY_DST),
      e: buf(T * H, "e"), h: buf(T * H, "h"), a: buf(T * H, "a"), o: buf(T * H, "o"),
      qkv: buf(T * 3 * H, "qkv"), u: buf(T * 2 * I, "u"), g: buf(T * I, "g"),
      idx: buf(this.maxPos, "idx", U.COPY_DST), out: buf(this.maxPos * H, "out"),
      lt: buf(T * LR, "lora tmp"),
    };
    // LoRA buffer layout: per layer and module, A [LR, K] then B [N, LR], f32 words
    this.loraLayout = [];
    let lw = 0;
    for (const L of m.layers) {
      const o = {};
      for (const k of ["Wqkv", "Wo", "Wi", "Wo_mlp"]) {
        const K = L[k].K, N = L[k].N;
        o[k] = { K, N, offA: lw, offB: lw + LR * K };
        lw += LR * K + N * LR;
      }
      this.loraLayout.push(o);
    }
    this.loraWords = lw;
    this.loraW = this.upload(new Float32Array(lw), "lora A/B (zero until setSpike)");
    this.b = B_;
    this.shaders = { ...KERNELS };
    const word = (t) => { if (t.offset % 4) throw new Error("unaligned f32 tensor"); return t.offset / 4; };

    const lins = [];
    const walk = (o) => { if (o && typeof o === "object") { if (o.q && o.s) lins.push(o); else Object.values(o).forEach(walk); } };
    walk(m.layers);
    let words = 0;
    for (const w of lins) { if (w.K % 32 || w.N % 32) throw new Error("matmul shape"); w._wt = words; words += w.N * w.K / 4; }
    const wt = new Int8Array(words * 4);
    for (const w of lins) {
      const src = new Int8Array(bytes.buffer, bytes.byteOffset + w.q.offset, w.K * w.N), dst = w._wt * 4;
      for (let k = 0; k < w.K; k++) for (let n = 0; n < w.N; n++) wt[dst + n * w.K + k] = src[k * w.N + n];
    }
    this.Wt = this.upload(wt, "int8 weights [N, K]");
    const mm = (w, X, Y, mode = 0) => matmul(this, { fmt: "i8", offs: [w._wt, word(w.s)], N: w.N, K: w.K }, this.W, X, Y,
      { rowsSel: 0, bias: NONE, gelu: !!(mode & 1), accumulate: !!(mode & 2), Wt: this.Wt });
    const lora = (i, k, X, Y) => {
      const s = this.loraLayout[i][k];
      matmul(this, { fmt: "f32", offs: [s.offA], N: LR, K: s.K }, this.loraW, X, B_.lt, { rowsSel: 0, bias: NONE });
      matmul(this, { fmt: "f32", offs: [s.offB], N: s.N, K: LR }, this.loraW, B_.lt, Y, { rowsSel: 0, bias: NONE, accumulate: true });
    };
    const ln = (w, X, Y) => norm(this, "ln", this.W, word(w), X, Y, { D: H, eps: m.eps, bOff: NONE, mode: 0, rowsSel: 0 });

    const tb = m.table;
    embed(this, this.W, B_.e, { fmt: "t4", codes: tb.codes.offset, scales: tb.scales.offset }, { H });
    ln(m.emb_norm, B_.e, B_.h);
    m.layers.forEach((L, li) => {
      let X = B_.h;
      if (L.attn_norm) { ln(L.attn_norm, B_.h, B_.a); X = B_.a; }
      mm(L.Wqkv, X, B_.qkv);
      lora(li, "Wqkv", X, B_.qkv);
      rope(this, this.rope, B_.qkv, { nh, hd: D, maxSeq: MAX_SEQ, tables: 2, sel: L.global ? 0 : 1, parts: 2, stride: 3 * H });
      attentionLong(this, B_.qkv, B_.o, { nh, hd: D, window: L.global ? 0 : m.window });
      mm(L.Wo, B_.o, B_.h, 2);
      lora(li, "Wo", B_.o, B_.h);
      ln(L.mlp_norm, B_.h, B_.a);
      mm(L.Wi, B_.a, B_.u);
      lora(li, "Wi", B_.a, B_.u);
      gated(this, B_.u, B_.u, B_.g, { I, sa: 2 * I, sb: 2 * I, ob: I });
      mm(L.Wo_mlp, B_.g, B_.h, 2);
      lora(li, "Wo_mlp", B_.g, B_.h);
    });
    ln(m.final_norm, B_.h, B_.a);
    // gather's row count is per batch, so it is recorded with a uniform the score() call rewrites
    this.gatherU = this.device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST, label: "gather" });
    this._owned.push(this.gatherU);
    this.step("gather", { a: [H, 0] }, [B_.idx, B_.a, B_.out], () => [this.maxPos]);
    this._gatherStep = this.steps[this.steps.length - 1];
    this._gatherStep.entries[0] = { binding: 0, resource: { buffer: this.gatherU } };
  }

  // Swap in a spike: sm is its manifest part (pack_system.py pack_spike), bytes its .bin.
  setSpike(sm, bytes) {
    const u16 = (t) => new Uint16Array(bytes.buffer, bytes.byteOffset + t.offset, t.length / 2);
    const w = new Float32Array(this.loraWords);
    sm.layers.forEach((L, i) => {
      for (const k of ["Wqkv", "Wo", "Wi", "Wo_mlp"]) {
        const s = this.loraLayout[i][k], r = L[k].r, A = f16(u16(L[k].A)), B = f16(u16(L[k].B));
        if (L[k].K !== s.K || L[k].N !== s.N || r > LR) throw new Error(`spike shape mismatch at layer ${i} ${k}`);
        w.set(A, s.offA);                                           // rows r..LR-1 stay zero
        for (let n = 0; n < s.N; n++) for (let j = 0; j < r; j++) w[s.offB + n * LR + j] = B[n * r + j];
      }
    });
    this.device.queue.writeBuffer(this.loraW, 0, w);
    const h = sm.head;
    this.head = { w0: f16(u16(h.w0)), b0: f16(u16(h.b0)), lnw: f16(u16(h.ln_w)), lnb: f16(u16(h.ln_b)), w1: f16(u16(h.w1)), b1: h.b1 };
    this.spike = sm.id ?? "router";
  }

  // seqs: [{ ids, pos, pool? }] with pos the [MASK] indexes inside each sequence and pool an optional
  // [start, end) token range to mean-pool into a key. Returns one Float32Array of raw logits per
  // sequence (temperature not applied); the array's .keys holds each sequence's key or null.
  async score(seqs) {
    const H = this.m.hidden;
    const idx = [], poolAt = [];
    let off = 0;
    for (const s of seqs) { for (const p of s.pos) idx.push(off + p); off += s.ids.length; }
    off = 0;
    for (const s of seqs) {
      if (s.pool) { poolAt.push([idx.length, s.pool[1] - s.pool[0]]); for (let t = s.pool[0]; t < s.pool[1]; t++) idx.push(off + t); }
      else poolAt.push(null);
      off += s.ids.length;
    }
    const nOpt = seqs.reduce((a, s) => a + s.pos.length, 0);
    if (idx.length > this.maxPos) throw new Error(`${idx.length} options exceed maxPos ${this.maxPos}`);
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 2).set([H, idx.length]);
    this.device.queue.writeBuffer(this.gatherU, 0, u);
    this.device.queue.writeBuffer(this.b.idx, 0, new Uint32Array(idx));
    const enc = this.device.createCommandEncoder();
    this._record(enc, seqs.map((s) => s.ids));
    const n = idx.length * H * 4;
    if (!this.readPos) {
      this.readPos = this.device.createBuffer({ size: this.maxPos * H * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "readback rows" });
      this._owned.push(this.readPos);
    }
    enc.copyBufferToBuffer(this.b.out, 0, this.readPos, 0, n);
    this.device.queue.submit([enc.finish()]);
    await this.readPos.mapAsync(GPUMapMode.READ, 0, n);
    const rows = new Float32Array(this.readPos.getMappedRange(0, n).slice(0));
    this.readPos.unmap();
    const z = this.headLogits(rows, nOpt);
    const out = [];
    let j = 0;
    for (const s of seqs) { out.push(z.subarray(j, j + s.pos.length)); j += s.pos.length; }
    out.keys = poolAt.map((pa) => {
      if (!pa) return null;
      const k = new Float32Array(H);
      for (let t = 0; t < pa[1]; t++) { const r = rows.subarray((pa[0] + t) * H, (pa[0] + t + 1) * H); for (let d = 0; d < H; d++) k[d] += r[d]; }
      let nrm = 0;
      for (let d = 0; d < H; d++) { k[d] /= pa[1]; nrm += k[d] * k[d]; }
      nrm = Math.sqrt(nrm) + 1e-12;
      for (let d = 0; d < H; d++) k[d] /= nrm;
      return k;
    });
    return out;
  }

  headLogits(rows, P) {
    const H = this.m.hidden, { w0, b0, lnw, lnb, w1, b1 } = this.head, eps = this.m.eps;
    const z = new Float32Array(P), h = new Float32Array(H);
    for (let r = 0; r < P; r++) {
      const x = rows.subarray(r * H, r * H + H);
      let mu = 0;
      for (let o = 0; o < H; o++) {
        let s = b0[o];
        const w = w0.subarray(o * H, o * H + H);
        for (let k = 0; k < H; k++) s += w[k] * x[k];
        h[o] = gelu(s); mu += h[o];
      }
      mu /= H;
      let v = 0;
      for (let o = 0; o < H; o++) v += (h[o] - mu) ** 2;
      const inv = 1 / Math.sqrt(v / H + eps);
      let s = b1;
      for (let o = 0; o < H; o++) s += ((h[o] - mu) * inv * lnw[o] + lnb[o]) * w1[o];
      z[r] = s;
    }
    return z;
  }
}
