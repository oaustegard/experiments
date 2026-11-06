// Training backends for the creator. Every backend has the same interface, which is also the interface
// proposed to the WGSL backward-pass experiment (bsky-account-decider session, 2026-11-05):
//
//   const b = await Backend.create(device, manifest, weightsBin, { init: { sm, bytes }, ... });
//   await b.step(seqs, targets, { lr, weightDecay }) -> { loss }   // seqs [{ ids, pos, key? }], targets Float32Array per seq
//   await b.score(seqs)          -> [Float32Array raw logits per seq]
//   await b.exportSpike()        -> { sm, bytes } in adapter-library pack_spike format
//   await b.loadSpike(sm, bytes)
//
// HeadBackend trains only the decision head, on a frozen encoder (the base plus the start spike's LoRA)
// run by adapter-library's SpikeEngine. The encoder never changes, so each sequence's [MASK] rows are
// computed once on the GPU and cached; an epoch is then plain JS over the cached rows.
// LoraBackend wraps an external trainer module (the WGSL backward pass) that implements the interface.

import { SpikeEngine } from "../../adapter-library/web/engine.js";
import { MODS, packSpike, softmax, unpackSpike } from "./creator.js";

// GELU as the engine computes it (erf by Abramowitz-Stegun 7.1.26), and its derivative x -> Phi(x) + x phi(x)
// with the same Phi, so the head here reproduces the engine's head to float rounding.
function erfAS(x) {
  const a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return Math.sign(x) * y;
}
// d/da of the approximation itself (not of the exact erf), so the gradient is exact for the function computed
function derfAS(x) {
  const a = Math.abs(x), t = 1 / (1 + 0.3275911 * a), e = Math.exp(-a * a);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const dpoly = (((5 * 1.061405429 * t - 4 * 1.453152027) * t + 3 * 1.421413741) * t - 2 * 0.284496736) * t + 0.254829592;
  return dpoly * 0.3275911 * t * t * e + poly * 2 * a * e;
}
export const gelu = (x) => 0.5 * x * (1 + erfAS(x * Math.SQRT1_2));
export const dgelu = (x) => 0.5 * (1 + erfAS(x * Math.SQRT1_2)) + 0.5 * x * derfAS(x * Math.SQRT1_2) * Math.SQRT1_2;

// SpikeEngine with the gathered [MASK] rows returned instead of head logits.
export class RowEngine extends SpikeEngine {
  async rows(seqs) {
    const H = this.m.hidden, idx = [];
    let off = 0;
    for (const s of seqs) { for (const p of s.pos) idx.push(off + p); off += s.ids.length; }
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
    const out = [];
    let j = 0;
    for (const s of seqs) { out.push(rows.subarray(j * H, (j + s.pos.length) * H)); j += s.pos.length; }
    return out;
  }
}

// Greedy batches under the engine's limits (tokens, sequences, gathered rows).
export function batchesOf(seqs, { maxTokens, maxSeqs, maxPos }) {
  const out = [];
  let cur = [], tok = 0, pos = 0;
  for (const s of seqs) {
    if (s.ids.length > maxTokens) throw new Error(`a sequence of ${s.ids.length} tokens exceeds maxTokens ${maxTokens}`);
    if (cur.length && (tok + s.ids.length > maxTokens || cur.length >= maxSeqs || pos + s.pos.length > maxPos)) { out.push(cur); cur = []; tok = 0; pos = 0; }
    cur.push(s); tok += s.ids.length; pos += s.pos.length;
  }
  if (cur.length) out.push(cur);
  return out;
}

// ---- the decision head: Linear(H, H) -> GELU -> LayerNorm -> Linear(H, 1), w0 stored [out, in] ----

export function headForward(p, x, H, eps, keep = null) {
  const s = new Float64Array(H), h = new Float64Array(H);
  let mu = 0;
  for (let o = 0; o < H; o++) {
    let a = p.b0[o];
    const w = o * H;
    for (let k = 0; k < H; k++) a += p.w0[w + k] * x[k];
    s[o] = a; h[o] = gelu(a); mu += h[o];
  }
  mu /= H;
  let v = 0;
  for (let o = 0; o < H; o++) v += (h[o] - mu) ** 2;
  const inv = 1 / Math.sqrt(v / H + eps), nrm = new Float64Array(H);
  let z = p.b1[0];
  for (let o = 0; o < H; o++) { nrm[o] = (h[o] - mu) * inv; z += (nrm[o] * p.lnw[o] + p.lnb[o]) * p.w1[o]; }
  if (keep) { keep.s = s; keep.nrm = nrm; keep.inv = inv; }
  return z;
}

// Soft cross-entropy of softmax(head logits) against targets, summed over seqs and divided by their
// number; gradients are accumulated into g (same keys as p). rows[i] is seq i's [P_i * H] block.
export function headLossGrad(p, rows, targets, H, eps, g = null) {
  let loss = 0;
  const B = rows.length;
  for (let i = 0; i < B; i++) {
    const R = rows[i], t = targets[i], P = t.length, keeps = [], z = new Float64Array(P);
    for (let j = 0; j < P; j++) { keeps.push({}); z[j] = headForward(p, R.subarray(j * H, (j + 1) * H), H, eps, keeps[j]); }
    const q = softmax(z);
    for (let j = 0; j < P; j++) if (t[j] > 0) loss -= t[j] * Math.log(Math.max(q[j], 1e-12));
    if (!g) continue;
    for (let j = 0; j < P; j++) {
      const dz = (q[j] - t[j]) / B;
      if (dz === 0) continue;
      const { s, nrm, inv } = keeps[j], x = R.subarray(j * H, (j + 1) * H);
      g.b1[0] += dz;
      const dn = new Float64Array(H);
      let mdn = 0, mdnn = 0;
      for (let o = 0; o < H; o++) {
        const y = nrm[o] * p.lnw[o] + p.lnb[o];
        g.w1[o] += dz * y;
        const dy = dz * p.w1[o];
        g.lnw[o] += dy * nrm[o]; g.lnb[o] += dy;
        dn[o] = dy * p.lnw[o]; mdn += dn[o]; mdnn += dn[o] * nrm[o];
      }
      mdn /= H; mdnn /= H;
      for (let o = 0; o < H; o++) {
        const ds = inv * (dn[o] - mdn - nrm[o] * mdnn) * dgelu(s[o]);
        if (ds === 0) continue;
        g.b0[o] += ds;
        const w = o * H;
        for (let k = 0; k < H; k++) g.w0[w + k] += ds * x[k];
      }
    }
  }
  return loss / B;
}

const HEAD_KEYS = ["w0", "b0", "lnw", "lnb", "w1", "b1"];
const DECAY = new Set(["w0", "w1"]);

function fnv(bytes) { let h = 0x811c9dc5; for (let i = 0; i < bytes.length; i += 7) { h ^= bytes[i]; h = Math.imul(h, 0x01000193); } return (h >>> 0) ^ bytes.length; }
// A cheap fingerprint of a spike's LoRA weights: the row cache stays valid while it is unchanged.
function loraSig(sm, bytes) {
  let lo = Infinity, hi = 0;
  for (const L of sm.layers) for (const k of MODS) for (const t of [L[k].A, L[k].B]) { lo = Math.min(lo, t.offset); hi = Math.max(hi, t.offset + t.length); }
  return fnv(bytes.subarray(lo, hi));
}

export class HeadBackend {
  static async create(device, manifest, weights, opts = {}) {
    const { maxTokens = 4096, maxSeqs = 16, maxPos = 2048 } = opts;
    const engine = await RowEngine.create(device, manifest, weights, { maxTokens, maxSeqs, maxPos });
    const b = new HeadBackend(engine, manifest, { maxTokens, maxSeqs, maxPos });
    if (!opts.init) throw new Error("HeadBackend needs a start spike (init: { sm, bytes })");
    await b.loadSpike(opts.init.sm, opts.init.bytes);
    return b;
  }

  constructor(engine, manifest, limits) {
    this.engine = engine; this.m = manifest; this.H = manifest.hidden; this.eps = manifest.eps; this.limits = limits;
    this.cache = new Map(); this.sig = null; this.encodeMs = 0; this.encoded = 0;
    this.name = "head only (frozen encoder)";
  }

  async loadSpike(sm, bytes) {
    bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const sig = loraSig(sm, bytes);
    if (sig !== this.sig) { this.engine.setSpike(sm, bytes); this.cache.clear(); this.sig = sig; }
    const u = unpackSpike(sm, bytes);
    this.layers = u.layers;
    this.p = { ...u.head, b1: Float32Array.of(u.head.b1) };
    this.adam = { t: 0, m: Object.fromEntries(HEAD_KEYS.map((k) => [k, new Float32Array(this.p[k].length)])),
                  v: Object.fromEntries(HEAD_KEYS.map((k) => [k, new Float32Array(this.p[k].length)])) };
  }

  // The frozen rows of each seq, from the cache or the GPU; seqs without a key are not cached.
  async rows(seqs) {
    const miss = seqs.filter((s) => !(s.key && this.cache.has(s.key)));
    const fresh = new Map();
    if (miss.length) {
      const t = performance.now();
      for (const batch of batchesOf(miss, this.limits)) {
        const r = await this.engine.rows(batch);
        batch.forEach((s, i) => { const v = r[i].slice(); if (s.key) this.cache.set(s.key, v); else fresh.set(s, v); });
      }
      this.encodeMs += performance.now() - t; this.encoded += miss.length;
    }
    return seqs.map((s) => (s.key ? this.cache.get(s.key) : fresh.get(s)));
  }

  // Encode ahead of training (progress callback per batch), so the first epoch is not slow.
  async warm(seqs, onProgress = () => {}) {
    const todo = seqs.filter((s) => !this.cache.has(s.key));
    const bs = batchesOf(todo, this.limits);
    let done = 0;
    for (const b of bs) { await this.rows(b); done += b.length; onProgress(done, todo.length); }
  }

  async step(seqs, targets, { lr = 1e-3, weightDecay = 0.01 } = {}) {
    const rows = await this.rows(seqs);
    const g = Object.fromEntries(HEAD_KEYS.map((k) => [k, new Float32Array(this.p[k].length)]));
    const loss = headLossGrad(this.p, rows, targets, this.H, this.eps, g);
    // clip the global gradient norm at 1.0, as run/lora.py does
    let gn = 0;
    for (const k of HEAD_KEYS) for (const x of g[k]) gn += x * x;
    gn = Math.sqrt(gn);
    if (gn > 1) for (const k of HEAD_KEYS) { const gk = g[k]; for (let i = 0; i < gk.length; i++) gk[i] /= gn; }
    const A = this.adam, b1 = 0.9, b2 = 0.999, e = 1e-8;
    A.t++;
    const c1 = 1 - b1 ** A.t, c2 = 1 - b2 ** A.t;
    for (const k of HEAD_KEYS) {
      const p = this.p[k], m = A.m[k], v = A.v[k], gk = g[k], wd = DECAY.has(k) ? weightDecay : 0;
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * gk[i];
        v[i] = b2 * v[i] + (1 - b2) * gk[i] * gk[i];
        p[i] -= lr * ((m[i] / c1) / (Math.sqrt(v[i] / c2) + e) + wd * p[i]);
      }
    }
    return { loss };
  }

  async score(seqs) {
    const rows = await this.rows(seqs), H = this.H;
    return rows.map((R, i) => {
      const P = seqs[i].pos.length, z = new Float32Array(P);
      for (let j = 0; j < P; j++) z[j] = headForward(this.p, R.subarray(j * H, (j + 1) * H), H, this.eps);
      return z;
    });
  }

  // weights and AdamW moments, for pausing and resuming (creator.js train); the frozen encoder is the start spike's
  saveState() {
    const c = (o) => Object.fromEntries(HEAD_KEYS.map((k) => [k, o[k].slice()]));
    return { kind: "head", sig: this.sig, p: c(this.p), adam: { t: this.adam.t, m: c(this.adam.m), v: c(this.adam.v) } };
  }

  loadState(st) {
    if (st.kind !== "head") throw new Error(`a ${st.kind} checkpoint cannot resume on the head-only backend`);
    if (st.sig !== this.sig) throw new Error("this checkpoint was made on a different start spike");
    const c = (o) => Object.fromEntries(HEAD_KEYS.map((k) => [k, Float32Array.from(o[k])]));
    this.p = c(st.p);
    this.adam = { t: st.adam.t, m: c(st.adam.m), v: c(st.adam.v) };
  }

  exportSpike() {
    const h = this.p;
    return packSpike({ layers: this.layers, head: { w0: h.w0, b0: h.b0, lnw: h.lnw, lnb: h.lnb, w1: h.w1, b1: h.b1[0] } });
  }

  destroy() { this.engine.destroy?.(); }
}

// ettin-wgsl-train's Trainer (web/train.js there): forward, backward and AdamW for rank-r LoRA on Wqkv,
// Wo and Wi plus the head, in WGSL, checked against PyTorch to 3.9e-6 per gradient tensor. Its API is
// padded batches and one flat parameter buffer; this adapter maps the creator's interface onto it.
// Differences that matter: its B is stored without the alpha/r scale (packed spikes carry it), and it has
// no LoRA on Wo_mlp, so a start spike's Wo_mlp delta is dropped (this.dropped reports its size) and the
// exported spike's Wo_mlp is zero. Body lr = lr, head lr = headMult * lr (run/lora.py: 5e-4 and 8e-4).
export class WgslTrainerBackend {
  static async create(device, manifest, weights, opts = {}) {
    const { Trainer, pad, rank = 8, alpha = 16, maxSeqs = 16, maxRows = 16 * 1024, headMult = 1.6 } = opts;
    const t = await Trainer.create(device, { manifest, weights, maxRows, maxSeqs, maxPool: maxSeqs * 32, maxSeg: maxSeqs, rank, alpha });
    const b = new WgslTrainerBackend(t, { pad, rank, alpha, maxSeqs, maxRows, headMult, L: manifest.layers.length });
    if (opts.init) await b.loadSpike(opts.init.sm, opts.init.bytes);
    return b;
  }

  constructor(t, o) { this.t = t; Object.assign(this, o); this.s = o.alpha / o.rank; this.step_ = 0; this.dropped = 0; this.name = "LoRA + head (WGSL backward pass, ettin-wgsl-train)"; }

  // the LoRA targets the trainer has, read from its layout (Wo_mlp only if it was added)
  mods() { return MODS.filter((k) => `L0.${k}.A` in this.t.poff); }

  // [{ ids, pos }] -> the trainer's padded batch; targets optional
  collate(seqs, targets = null) {
    const B = seqs.length, S = Math.max(...seqs.map((x) => x.ids.length));
    const ids = new Int32Array(B * S).fill(this.pad), lens = new Int32Array(B), pos = [], seg = [], gold = [];
    seqs.forEach((x, b) => {
      ids.set(x.ids, b * S); lens[b] = x.ids.length; seg.push(x.pos.length);
      x.pos.forEach((p, j) => { pos.push(b * S + p); gold.push(targets ? targets[b][j] : 0); });
    });
    return { ids, B, S, lens, pos: Int32Array.from(pos), seg: Int32Array.from(seg), gold: Float32Array.from(gold) };
  }

  batches(seqs) {
    const out = [];
    let cur = [], S = 0;
    for (const x of seqs) {
      const S2 = Math.max(S, x.ids.length);
      if (cur.length && (cur.length >= this.maxSeqs || (cur.length + 1) * S2 > this.maxRows)) { out.push(cur); cur = []; S = 0; }
      cur.push(x); S = Math.max(S, x.ids.length);
    }
    if (cur.length) out.push(cur);
    return out;
  }

  async step(seqs, targets, { lr = 5e-4, weightDecay = 0.01 } = {}) {
    const t = this.t;
    t.loadBatch(this.collate(seqs, targets));
    t.forwardBackward({ T: 1, slot: 0 });
    t.adamStep({ lr, lrHead: lr * this.headMult, wd: weightDecay, step: ++this.step_, clip: 1.0 });
    const loss = (await t.read(t.LOSS, 1))[0];
    return { loss };
  }

  async score(seqs) {
    const t = this.t, out = [];
    for (const b of this.batches(seqs)) {
      t.loadBatch(this.collate(b));
      t.forward();
      const z = await t.read(t.LOGIT, b.reduce((a, x) => a + x.pos.length, 0));
      let j = 0;
      for (const x of b) { out.push(z.slice(j, j + x.pos.length)); j += x.pos.length; }
    }
    return out;
  }

  async loadSpike(sm, bytes) {
    let u = unpackSpike(sm, bytes);
    if (u.layers.some((L) => MODS.some((k) => L[k].r > this.rank))) u = truncateRank(u, this.rank);
    const t = this.t, flat = new Float32Array(t.nParams), off = t.poff, r = this.rank;
    let dropped = 0;
    const mods = this.mods();
    u.layers.forEach((L, i) => {
      for (const k of mods) {
        const m = L[k], A = new Float32Array(r * m.K), B = new Float32Array(m.N * r);
        A.set(m.A.subarray(0, m.r * m.K));                                      // rows m.r..r-1 stay zero
        for (let n = 0; n < m.N; n++) for (let j = 0; j < m.r; j++) B[n * r + j] = m.B[n * m.r + j] / this.s;
        flat.set(A, off[`L${i}.${k}.A`]); flat.set(B, off[`L${i}.${k}.B`]);
      }
      if (!mods.includes("Wo_mlp")) for (const x of L.Wo_mlp.B) dropped = Math.max(dropped, Math.abs(x));
    });
    const h = u.head;
    flat.set(h.w0, off["head.w0"]); flat.set(h.b0, off["head.b0"]); flat.set(h.lnw, off["head.ln_w"]); flat.set(h.lnb, off["head.ln_b"]);
    flat.set(h.w1, off["head.w1"]); flat[off["head.b1"]] = h.b1;
    t.setParams(flat);
    await t.flush();
    this.step_ = 0; this.dropped = dropped;
  }

  // the whole flat parameter vector, both AdamW moments and the step count, for pausing and resuming
  async saveState() {
    const t = this.t;
    return { kind: "wgsl", nParams: t.nParams, rank: this.rank, mods: this.mods(), prm: await t.read(t.PRM), am: await t.read(t.AM), av: await t.read(t.AV), step: this.step_ };
  }

  async loadState(st) {
    const t = this.t;
    if (st.kind !== "wgsl" || st.nParams !== t.nParams || st.rank !== this.rank || st.mods.join() !== this.mods().join())
      throw new Error("this checkpoint was made with a different trainer layout");
    t.setParams(Float32Array.from(st.prm));   // also queues clears of the moments, so flush before writing them
    await t.flush();
    t.device.queue.writeBuffer(t.AM, 0, Float32Array.from(st.am));
    t.device.queue.writeBuffer(t.AV, 0, Float32Array.from(st.av));
    await t.flush();
    this.step_ = st.step;
  }

  async exportSpike() {
    const t = this.t, p = await t.read(t.PRM), off = t.poff, r = this.rank, H = 256;
    const layers = Array.from({ length: this.L }, (_, i) => {
      const o = {}, mods = this.mods();
      for (const k of mods) {
        const [, K] = t.layout.find((x) => x.name === `L${i}.${k}.A`).shape, [N] = t.layout.find((x) => x.name === `L${i}.${k}.B`).shape;
        const A = p.slice(off[`L${i}.${k}.A`], off[`L${i}.${k}.A`] + r * K), B = p.slice(off[`L${i}.${k}.B`], off[`L${i}.${k}.B`] + N * r);
        for (let j = 0; j < B.length; j++) B[j] *= this.s;
        o[k] = { A, B, r, K, N };
      }
      if (!mods.includes("Wo_mlp")) o.Wo_mlp = { A: new Float32Array(384), B: new Float32Array(H), r: 1, K: 384, N: H };
      return o;
    });
    const g = (n, len) => p.slice(off[n], off[n] + len);
    return packSpike({ layers, head: { w0: g("head.w0", H * H), b0: g("head.b0", H), lnw: g("head.ln_w", H), lnb: g("head.ln_b", H), w1: g("head.w1", H), b1: p[off["head.b1"]] } });
  }
}

// A learned library merge (after P²-MOPD, arXiv 2611.03953) on ettin-wgsl-train's Trainer, unchanged.
// The new spike's delta per module m is  sum_k alpha[k][m] * B_k A_k  +  B_res A_res : K frozen library
// spikes plus a trainable rank-8 residual. The trainer runs at rank 8 + 8K with the residual in slots 0..7
// and spike k in slots 8 + 8k .. 15 + 8k (A_slot = A_k, B_slot = alpha * B_k / s). After each backward pass
// the adapter reads the gradients, takes dL/dalpha = sum(G_B[slot k] * B_k) / s (exact, since B_slot is
// linear in alpha), zeroes the library slots' gradients so AdamW and its clipping see only the residual and
// the head, steps, then writes the library slots back from the new alpha. alpha has its own Adam (no decay).
// mode: "alpha+res" (the method), "uniform+res" (alpha frozen at 1/K), "alpha" (no residual).
export class LearnedMergeBackend {
  static async create(device, manifest, weights, opts = {}) {
    const { Trainer, pad, lib, mode = "alpha+res", head, maxSeqs = 16, maxRows = 16 * 1024, headMult = 1.6, alphaMult = 20, seed = 0 } = opts;
    const K = lib.length, rank = 8 + 8 * K;
    const t = await Trainer.create(device, { manifest, weights, maxRows, maxSeqs, maxPool: maxSeqs * 32, maxSeg: maxSeqs, rank, alpha: 2 * rank });
    const b = new LearnedMergeBackend(t, { pad, lib, mode, headInit: head, K, rank, maxSeqs, maxRows, headMult, alphaMult, L: manifest.layers.length, seed });
    b.reset();
    await t.flush();
    return b;
  }

  constructor(t, o) {
    Object.assign(this, o); this.t = t; this.s = 2;   // trainer scale alpha/rank = 2, as rank-8 alpha-16 spikes
    this.inner = new WgslTrainerBackend(t, { pad: o.pad, rank: o.rank, alpha: 2 * o.rank, maxSeqs: o.maxSeqs, maxRows: o.maxRows, headMult: o.headMult, L: o.L });
    this.mods = this.inner.mods();
    this.name = `learned library merge (${o.mode}, ${o.K} spikes + rank-8 residual)`;
  }

  // alpha to 1/K, residual to a fresh LoRA (A kaiming-uniform as nn.Linear, B zero), head from headInit
  reset() {
    const t = this.t, off = t.poff, r = this.rank, s = this.s, rand = rngU(this.seed + 11);
    this.alpha = this.lib.map(() => this.layersOf(() => 1 / this.K));
    this.am = this.lib.map(() => this.layersOf(() => 0)); this.av = this.lib.map(() => this.layersOf(() => 0)); this.at = 0;
    const flat = new Float32Array(t.nParams);
    for (let i = 0; i < this.L; i++) for (const m of this.mods) {
      const K_ = this.lib[0].layers[i][m].K, bound = 1 / Math.sqrt(K_);
      for (let j = 0; j < 8 * K_; j++) flat[off[`L${i}.${m}.A`] + j] = this.mode === "alpha" ? 0 : (rand() * 2 - 1) * bound;
    }
    const h = this.headInit;
    flat.set(h.w0, off["head.w0"]); flat.set(h.b0, off["head.b0"]); flat.set(h.lnw, off["head.ln_w"]); flat.set(h.lnb, off["head.ln_b"]);
    flat.set(h.w1, off["head.w1"]); flat[off["head.b1"]] = h.b1;
    this.writeLib(flat);
    t.setParams(flat);
    this.inner.step_ = 0;
  }
  layersOf(f) { return Array.from({ length: this.L }, () => Object.fromEntries(this.mods.map((m) => [m, f()]))); }

  // the library slots of flat params (A rows 8+8k.., B columns 8+8k..) from the current alpha
  writeLib(flat) {
    const off = this.t.poff, r = this.rank, s = this.s;
    this.lib.forEach((sp, k) => {
      for (let i = 0; i < this.L; i++) for (const m of this.mods) {
        const { A, B, r: rk, K, N } = sp.layers[i][m], a = this.alpha[k][i][m], j0 = 8 + 8 * k;
        flat.set(A.subarray(0, rk * K), off[`L${i}.${m}.A`] + j0 * K);
        const bo = off[`L${i}.${m}.B`];
        for (let n = 0; n < N; n++) for (let j = 0; j < rk; j++) flat[bo + n * r + j0 + j] = (a * B[n * rk + j]) / s;
      }
    });
  }

  // dL/dalpha from the gradient buffer; zero the slots AdamW must not move
  alphaGrads(grd) {
    const off = this.t.poff, r = this.rank, s = this.s;
    const g = this.lib.map(() => this.layersOf(() => 0));
    for (let i = 0; i < this.L; i++) for (const m of this.mods) {
      const bo = off[`L${i}.${m}.B`], ao = off[`L${i}.${m}.A`];
      this.lib.forEach((sp, k) => {
        const { B, r: rk, K, N } = sp.layers[i][m], j0 = 8 + 8 * k;
        let d = 0;
        for (let n = 0; n < N; n++) for (let j = 0; j < rk; j++) { d += grd[bo + n * r + j0 + j] * B[n * rk + j] / s; grd[bo + n * r + j0 + j] = 0; }
        g[k][i][m] = d;
        grd.fill(0, ao + j0 * K, ao + (j0 + 8) * K);
      });
      if (this.mode === "alpha") {
        const K = this.lib[0].layers[i][m].K, N = this.lib[0].layers[i][m].N;
        grd.fill(0, ao, ao + 8 * K);
        for (let n = 0; n < N; n++) for (let j = 0; j < 8; j++) grd[bo + n * r + j] = 0;
      }
    }
    return g;
  }

  async lossAt(seqs, targets) {   // forward + loss only (for the gradient check)
    const t = this.t;
    t.loadBatch(this.inner.collate(seqs, targets)); t.forwardBackward({ T: 1, slot: 0 });
    return (await t.read(t.LOSS, 1))[0];
  }

  async step(seqs, targets, { lr = 5e-4, weightDecay = 0.01 } = {}) {
    const t = this.t;
    t.loadBatch(this.inner.collate(seqs, targets));
    t.forwardBackward({ T: 1, slot: 0 });
    const loss = (await t.read(t.LOSS, 1))[0], grd = await t.read(t.GRD);
    const ga = this.alphaGrads(grd);
    this.device().queue.writeBuffer(t.GRD, 0, grd);
    t.adamStep({ lr, lrHead: lr * this.headMult, wd: weightDecay, step: ++this.inner.step_, clip: 1.0 });
    if (this.mode !== "uniform+res") {
      const la = lr * this.alphaMult, b1 = 0.9, b2 = 0.999; this.at++;
      const c1 = 1 - b1 ** this.at, c2 = 1 - b2 ** this.at;
      this.lib.forEach((_, k) => { for (let i = 0; i < this.L; i++) for (const m of this.mods) {
        const gk = ga[k][i][m], M = this.am[k][i], V = this.av[k][i];
        M[m] = b1 * M[m] + (1 - b1) * gk; V[m] = b2 * V[m] + (1 - b2) * gk * gk;
        this.alpha[k][i][m] -= la * (M[m] / c1) / (Math.sqrt(V[m] / c2) + 1e-8);
      } });
    }
    const prm = await t.read(t.PRM);
    this.writeLib(prm);
    this.device().queue.writeBuffer(t.PRM, 0, prm);
    return { loss };
  }
  device() { return this.t.device; }

  score(seqs) { return this.inner.score(seqs); }

  // the trained spike at rank 8 + 8K (library slots carry alpha): packed B carries s
  async exportSpike() {
    const t = this.t, p = await t.read(t.PRM), off = t.poff, r = this.rank, H = 256;
    const layers = Array.from({ length: this.L }, (_, i) => Object.fromEntries(this.mods.map((m) => {
      const [, K] = t.layout.find((x) => x.name === `L${i}.${m}.A`).shape, [N] = t.layout.find((x) => x.name === `L${i}.${m}.B`).shape;
      const A = p.slice(off[`L${i}.${m}.A`], off[`L${i}.${m}.A`] + r * K), B = p.slice(off[`L${i}.${m}.B`], off[`L${i}.${m}.B`] + N * r);
      for (let j = 0; j < B.length; j++) B[j] *= this.s;
      return [m, { A, B, r, K, N }];
    })));
    const g = (n, len) => p.slice(off[n], off[n] + len);
    const out = packSpike({ layers, head: { w0: g("head.w0", H * H), b0: g("head.b0", H), lnw: g("head.ln_w", H), lnb: g("head.ln_b", H), w1: g("head.w1", H), b1: p[off["head.b1"]] } });
    out.sm.alpha = this.alpha;
    return out;
  }

  // a snapshot this backend made: params as they were, alpha restored
  async loadSpike(sm, bytes) {
    if (!sm.alpha) throw new Error("LearnedMergeBackend loads only its own snapshots");
    const u = unpackSpike(sm, bytes), t = this.t, off = t.poff, r = this.rank, flat = new Float32Array(t.nParams);
    u.layers.forEach((L, i) => { for (const m of this.mods) { flat.set(L[m].A, off[`L${i}.${m}.A`]); L[m].B.forEach((x, j) => { flat[off[`L${i}.${m}.B`] + j] = x / this.s; }); } });
    const h = u.head;
    flat.set(h.w0, off["head.w0"]); flat.set(h.b0, off["head.b0"]); flat.set(h.lnw, off["head.ln_w"]); flat.set(h.lnb, off["head.ln_b"]);
    flat.set(h.w1, off["head.w1"]); flat[off["head.b1"]] = h.b1;
    this.alpha = structuredClone(sm.alpha);
    t.setParams(flat); await t.flush();
  }
}

function rngU(seed) { let a = (seed >>> 0) || 1; return () => { a = (a + 0x6d2b79f5) | 0; let x = Math.imul(a ^ (a >>> 15), 1 | a); x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x; return ((x ^ (x >>> 14)) >>> 0) / 4294967296; }; }

// An external trainer (the WGSL backward pass) behind the same interface. The module must export
// LoraTrainer (or a default export) with create(device, manifest, weights, opts) and the methods above.
export class LoraBackend {
  static async load(url) {
    const mod = await import(url);
    const T = mod.LoraTrainer ?? mod.Trainer ?? mod.default;
    if (!T?.create) throw new Error(`${url} exports neither LoraTrainer nor Trainer with create()`);
    return T;
  }

  static async create(device, manifest, weights, opts = {}) {
    const mod = await import(opts.module);
    if (mod.Trainer?.create && mod.Trainer.prototype?.forwardBackward)   // ettin-wgsl-train's API
      return WgslTrainerBackend.create(device, manifest, weights, { ...opts, Trainer: mod.Trainer });
    const T = await LoraBackend.load(opts.module);
    const t = await T.create(device, manifest, weights, { rank: 8, alpha: 16, maxTokens: 4096, maxSeqs: 16, maxPos: 2048, seed: 0, ...opts.trainer });
    const b = new LoraBackend(t);
    if (opts.init) await b.loadSpike(opts.init.sm, opts.init.bytes);
    return b;
  }

  constructor(t) { this.t = t; this.name = "LoRA (WGSL backward pass)"; }
  step(seqs, targets, o) { return this.t.step(seqs, targets, o); }
  score(seqs) { return this.t.score(seqs); }
  exportSpike() { return this.t.exportSpike(); }
  loadSpike(sm, bytes) { return this.t.loadSpike(sm, bytes); }
  get saveState() { return this.t.saveState ? () => this.t.saveState() : undefined; }
  loadState(st) { return this.t.loadState(st); }
  destroy() { this.t.destroy?.(); }
}

// Truncate an unpacked spike's LoRA to rank r per module by SVD of its delta B A, split as
// B' = U sqrt(S), A' = sqrt(S) V^T (adapter-library run/lora.py merged_adapter, without its alpha/rank
// rescale, since packed B already carries the scale). For a rank-8 trainer started from a merged library.
// Subspace iteration on (B A)^T (B A) with a Jacobi eigensolve of the r x r projection.
export function truncateRank(u, r = 8, iters = 30) {
  const layers = u.layers.map((L) => Object.fromEntries(MODS.map((k) => {
    const { A, B, r: R, K, N } = L[k];
    if (R <= r) return [k, L[k]];
    // M = B A as N x K, applied implicitly: Mv = B (A v), M^T w = A^T (B^T w)
    const Av = (v) => { const o = new Float64Array(R); for (let j = 0; j < R; j++) { let s = 0; for (let c = 0; c < K; c++) s += A[j * K + c] * v[c]; o[j] = s; } return o; };
    const Bv = (v) => { const o = new Float64Array(N); for (let n = 0; n < N; n++) { let s = 0; for (let j = 0; j < R; j++) s += B[n * R + j] * v[j]; o[n] = s; } return o; };
    const Btw = (w) => { const o = new Float64Array(R); for (let n = 0; n < N; n++) for (let j = 0; j < R; j++) o[j] += B[n * R + j] * w[n]; return o; };
    const Atv = (v) => { const o = new Float64Array(K); for (let j = 0; j < R; j++) for (let c = 0; c < K; c++) o[c] += A[j * K + c] * v[j]; return o; };
    const M = (v) => Bv(Av(v)), Mt = (w) => Atv(Btw(w));
    let Q = Array.from({ length: r }, (_, j) => { const v = new Float64Array(K); for (let c = 0; c < K; c++) v[c] = Math.sin(1 + j * 7.13 + c * 0.917); return v; });
    const orth = (Vs) => { for (let j = 0; j < Vs.length; j++) { for (let i = 0; i < j; i++) { let d = 0; for (let c = 0; c < Vs[j].length; c++) d += Vs[j][c] * Vs[i][c]; for (let c = 0; c < Vs[j].length; c++) Vs[j][c] -= d * Vs[i][c]; } let nn = 0; for (const x of Vs[j]) nn += x * x; nn = Math.sqrt(nn) || 1; for (let c = 0; c < Vs[j].length; c++) Vs[j][c] /= nn; } return Vs; };
    Q = orth(Q);
    for (let it = 0; it < iters; it++) Q = orth(Q.map((q) => Mt(M(q))));
    // Rayleigh-Ritz: eigenvectors of G = (MQ)^T (MQ) rotate Q onto right singular vectors
    const MQ = Q.map(M);
    const G = Array.from({ length: r }, (_, i) => Array.from({ length: r }, (_, j) => { let s = 0; for (let n = 0; n < N; n++) s += MQ[i][n] * MQ[j][n]; return s; }));
    const V = Array.from({ length: r }, (_, i) => Array.from({ length: r }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 50; sweep++) {
      let off = 0;
      for (let p = 0; p < r; p++) for (let q = p + 1; q < r; q++) off += G[p][q] ** 2;
      if (off < 1e-20) break;
      for (let p = 0; p < r; p++) for (let q = p + 1; q < r; q++) {
        if (Math.abs(G[p][q]) < 1e-30) continue;
        const th = 0.5 * Math.atan2(2 * G[p][q], G[q][q] - G[p][p]), c = Math.cos(th), s = Math.sin(th);
        for (let i = 0; i < r; i++) { const gp = G[i][p], gq = G[i][q]; G[i][p] = c * gp - s * gq; G[i][q] = s * gp + c * gq; }
        for (let i = 0; i < r; i++) { const gp = G[p][i], gq = G[q][i]; G[p][i] = c * gp - s * gq; G[q][i] = s * gp + c * gq; }
        for (let i = 0; i < r; i++) { const vp = V[i][p], vq = V[i][q]; V[i][p] = c * vp - s * vq; V[i][q] = s * vp + c * vq; }
      }
    }
    const order = Array.from({ length: r }, (_, j) => j).sort((a, b) => G[b][b] - G[a][a]);
    const A2 = new Float32Array(r * K), B2 = new Float32Array(N * r);
    order.forEach((e, jj) => {
      const v = new Float64Array(K);
      for (let i = 0; i < r; i++) for (let c = 0; c < K; c++) v[c] += V[i][e] * Q[i][c];
      const mv = M(v);
      let sig = 0; for (const x of mv) sig += x * x; sig = Math.sqrt(sig);
      const rs = Math.sqrt(sig) || 0;
      for (let c = 0; c < K; c++) A2[jj * K + c] = rs * v[c];
      for (let n = 0; n < N; n++) B2[n * r + jj] = sig > 0 ? (mv[n] / sig) * rs : 0;
    });
    return [k, { A: A2, B: B2, r, K, N }];
  })));
  return { layers, head: u.head };
}
