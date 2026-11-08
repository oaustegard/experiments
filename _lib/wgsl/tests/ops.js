// Per-op parity and timing for the shared kernels in ../kernels.js.
//
// Each case records one op through the real SeqEngine path (step, _finish, _record), on random inputs
// and a packed batch of mixed-length sequences, and compares its output with a float64 reference in
// this file. A case passes when max |gpu - ref| <= tol * max |ref|. With opts.reps > 0 it also times
// the op: reps dispatches in one pass, submit to done, median of opts.rounds.
//
//   const { cases, runOps } = await import(".../_lib/wgsl/tests/ops.js");
//   const res = await runOps(device, { only: "norm", reps: 0 });   // [{ name, pass, err, ... }]
//
// Run headless: node _lib/wgsl/tests/check_ops.mjs. On a real GPU: gpu-bridge/jobs/wgsl-ops.

import { SeqEngine, NONE } from "../runtime.js";
import { KERNELS, matmul, norm, attention, attentionLong, pool, l2, embed, rope, ropeTable, gated, route, expertMatmul, matmulTiled, combine, l2tok } from "../kernels.js";

const U = GPUBufferUsage;

// ---- deterministic inputs ----------------------------------------------------------------------
export function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export const randn = (r, n, sc = 1) => Float32Array.from({ length: n }, () => sc * Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()));

// GELU with the same erf approximation as the kernel (A&S 7.1.26, |error| < 1.5e-7).
export function gelu(x) {
  const a = Math.abs(x) * Math.SQRT1_2, t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return 0.5 * x * (1 + Math.sign(x) * y);
}

// A packed batch: sequence lengths -> { batch (token id lists), T, B, starts, lens }.
function packed(lens) {
  const starts = []; let o = 0;
  for (const l of lens) { starts.push(o); o += l; }
  return { batch: lens.map((l) => Array.from({ length: l }, (_, j) => j)), T: o, B: lens.length, starts, lens };
}
const LENS = [7, 33, 1, 128, 20];   // 189 tokens: a single token, a full 128, and odd lengths
const LONG = [300, 1, 129, 700];     // attentionLong: past one workgroup, under its 1024

// float32 -> IEEE half bits (round to nearest even), for fp16 scales.
function toHalf(x) {
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0];
  const sgn = (u >>> 16) & 0x8000, e = ((u >>> 23) & 255) - 127 + 15, m = u & 0x7fffff;
  if (e <= 0) return sgn;
  let h = sgn | (e << 10) | (m >> 13);
  const rest = m & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (h & 1))) h++;
  return h;
}
function fromHalf(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15);
}

// ---- weight-buffer builder: word-addressed u32 store ----------------------------------------------
class Words {
  constructor() { this.w = []; }
  f32(a) { const o = this.w.length; const u = new Uint32Array(Float32Array.from(a).buffer); for (const x of u) this.w.push(x); return o; }
  u32(a) { const o = this.w.length; for (const x of a) this.w.push(x >>> 0); return o; }
  get array() { return Uint32Array.from(this.w.length ? this.w : [0]); }
}

// byte-addressed builder for tables whose offsets are in bytes
class Bytes {
  constructor() { this.b = []; }
  put(u8) { while (this.b.length % 4) this.b.push(0); const o = this.b.length; for (const x of u8) this.b.push(x); return o; }
  get array() { const a = new Uint8Array(Math.ceil(this.b.length / 4) * 4 || 4); a.set(this.b); return new Uint32Array(a.buffer); }
}

// ---- harness: one engine per case, recorded through SeqEngine's own step / _finish / _record -------
async function harness(device, bt, make) {
  const e = new SeqEngine(device, {}, { maxTokens: bt.T, maxSeqs: bt.B, maxSeq: Math.max(...bt.lens) });
  e._owned = [];
  e.dyn = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST, label: "dyn" });
  e._owned.push(e.dyn);
  e.steps = [];
  e.shaders = { ...KERNELS };
  e.b = { tok: e.buf(bt.T * 4, "tok", U.COPY_DST), seqs: e.buf(bt.B * 4, "seqs", U.COPY_DST) };
  const out = make(e);           // records the op, returns { buf, n } to read
  await e._finish();
  return { e, out };
}

async function readBuf(device, buf, n) {
  const rb = device.createBuffer({ size: n * 4, usage: U.MAP_READ | U.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const x = new Float32Array(rb.getMappedRange().slice(0));
  rb.destroy();
  return x;
}

async function timeIt(device, e, bt, reps, rounds) {
  const n = [bt.T, bt.B], ms = [];
  for (let r = 0; r < rounds; r++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = 0; i < reps; i++) for (const s of e.steps) { pass.setPipeline(e.pipes[s.kind]); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.grid(n)); }
    pass.end();
    const t = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    ms.push((performance.now() - t) / reps);
  }
  ms.sort((a, b) => a - b);
  return ms[ms.length >> 1];
}

// ---- cases ---------------------------------------------------------------------------------------
// Each: { name, op, run(device, r) -> { e, bt, out: {buf, n}, ref: Float64Array|Float32Array, init?: [[buf, data]] } }

function matmulCase(fmt, { N, K, rowsSel = 0, bias = false, geluOn = false, accumulate = false, alpha = 1 }) {
  const tag = [fmt, `${N}x${K}`, rowsSel ? "seqrows" : "", bias ? "bias" : "", geluOn ? "gelu" : "", accumulate ? "acc" : "", alpha !== 1 ? "alpha" : ""].filter(Boolean).join("_");
  return {
    name: `matmul_${tag}`, op: "matmul",
    async run(device, r) {
      const bt = packed(LENS), rows = rowsSel ? bt.B : bt.T;
      const X = randn(r, rows * K), Wd = new Float64Array(N * K);   // dequantized weights [N, K]
      const words = new Words();
      const offs = [0, 0, 0];
      let colScale = new Float64Array(N).fill(1), WtArr = null;
      if (fmt === "f32") {
        const w = randn(r, N * K, 1 / Math.sqrt(K));
        offs[0] = words.f32(w); w.forEach((v, i) => { Wd[i] = v; });
      } else if (fmt === "i8") {
        const q = Int8Array.from({ length: N * K }, () => Math.floor(r() * 255) - 127);
        const sc = Float32Array.from({ length: N }, () => (0.5 + r()) / (64 * Math.sqrt(K)));
        WtArr = new Uint32Array(q.buffer.slice(0));
        offs[0] = 0; offs[1] = words.f32(sc);
        q.forEach((v, i) => { Wd[i] = v; });
        colScale = Float64Array.from(sc);
      } else if (fmt === "q4") {
        // onnxruntime MatMulNBits: [N, K/32 blocks, 16 bytes], low nibble first; f32 scale per block;
        // packed 4-bit zero points, ceil(nb / 2) bytes per row.
        const nb = K / 32, zrow = (nb + 1) >> 1;
        const nib = Uint8Array.from({ length: N * K }, () => Math.floor(r() * 16));
        const zp = Uint8Array.from({ length: N * nb }, () => Math.floor(r() * 16));
        const sc = Float32Array.from({ length: N * nb }, () => (0.5 + r()) / (4 * Math.sqrt(K)));
        const qw = new Uint32Array(N * K / 8);
        for (let i = 0; i < N * K; i++) qw[i >> 3] |= nib[i] << (4 * (i & 7));
        const zb = new Uint8Array(Math.ceil(N * zrow / 4) * 4);
        for (let n = 0; n < N; n++) for (let b = 0; b < nb; b++) zb[n * zrow + (b >> 1)] |= zp[n * nb + b] << (4 * (b & 1));
        offs[0] = words.u32(qw); offs[1] = words.f32(sc); offs[2] = words.u32(new Uint32Array(zb.buffer));
        for (let n = 0; n < N; n++) for (let k = 0; k < K; k++) Wd[n * K + k] = sc[n * nb + (k >> 5)] * (nib[n * K + k] - zp[n * nb + (k >> 5)]);
      }
      const bv = bias ? randn(r, N) : null;
      const bOff = bias ? words.f32(bv) : NONE;
      const Y0 = accumulate ? randn(r, rows * N) : null;
      const ref = new Float64Array(rows * N);
      for (let i = 0; i < rows; i++) for (let n = 0; n < N; n++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += X[i * K + k] * Wd[n * K + k];
        s *= alpha * colScale[n];
        if (bias) s += bv[n];
        if (geluOn) s = gelu(s);
        ref[i * N + n] = accumulate ? Y0[i * N + n] + s : s;
      }
      const { e, out } = await harness(device, bt, (e) => {
        const W = e.upload(words.array, "W"), Wt = WtArr ? e.upload(WtArr, "Wt") : undefined;
        const Xb = e.upload(X, "X"), Yb = e.buf(rows * N, "Y", U.COPY_DST);
        if (Y0) device.queue.writeBuffer(Yb, 0, Y0);
        matmul(e, { fmt, offs, N, K }, W, Xb, Yb, { rowsSel, alpha, bias: bOff, gelu: geluOn, accumulate, Wt });
        return { buf: Yb, n: rows * N, reset: Y0 ? () => device.queue.writeBuffer(Yb, 0, Y0) : null };
      });
      return { e, bt, out, ref };
    },
  };
}

function normCase(kind, D, mode, bias) {
  return {
    name: `norm_${kind}_${D}_m${mode}${bias ? "_b" : ""}`, op: "norm",
    async run(device, r) {
      const bt = packed(LENS), R = bt.T, eps = 1e-6, scalar = 0.75;
      const x = randn(r, R * D), y0 = randn(r, R * D), w = randn(r, D), bv = randn(r, D);
      const words = new Words(); words.f32([0, 0, 0]);   // nonzero offsets exercise p.a
      const wOff = words.f32(w), bOff = bias ? words.f32(bv) : NONE;
      const ref = new Float64Array(R * D);
      for (let i = 0; i < R; i++) {
        const a = Array.from({ length: D }, (_, j) => x[i * D + j] + (mode === 1 ? y0[i * D + j] : 0));
        let inv, mu = 0;
        if (kind === "rms") inv = 1 / Math.sqrt(a.reduce((s, v) => s + v * v, 0) / D + eps);
        else { mu = a.reduce((s, v) => s + v, 0) / D; inv = 1 / Math.sqrt(a.reduce((s, v) => s + (v - mu) ** 2, 0) / D + eps); }
        for (let j = 0; j < D; j++) {
          let o = (a[j] - mu) * inv * w[j] + (bias ? bv[j] : 0);
          ref[i * D + j] = mode === 2 ? (y0[i * D + j] + o) * scalar : o;
        }
      }
      const { e, out } = await harness(device, bt, (e) => {
        const W = e.upload(words.array, "W"), X = e.upload(x, "x"), Y = e.buf(R * D, "y", U.COPY_DST);
        device.queue.writeBuffer(Y, 0, y0);
        norm(e, kind, W, wOff, X, Y, { D, eps, bOff, mode, scalar });
        return { buf: Y, n: R * D, reset: mode ? () => device.queue.writeBuffer(Y, 0, y0) : null };
      });
      return { e, bt, out, ref };
    },
  };
}

function attnCase({ nh, nkv, hd, fused, window = 0, long = false }) {
  return {
    name: `attn${long ? "long" : ""}_h${nh}kv${nkv}_d${hd}${fused ? "_fused" : ""}${window ? `_w${window}` : ""}`, op: long ? "attentionLong" : "attention",
    async run(device, r) {
      const bt = packed(long ? LONG : LENS), T = bt.T, scale = 1 / Math.sqrt(hd);
      const qst = fused ? (nh + 2 * nkv) * hd : nh * hd, kst = fused ? qst : nkv * hd;
      const ko = fused ? nh * hd : 0, vo = fused ? (nh + nkv) * hd : 0;
      const Q = randn(r, T * qst), Kd = fused ? Q : randn(r, T * kst), Vd = fused ? Q : randn(r, T * kst);
      const ref = new Float64Array(T * nh * hd), g = nh / nkv;
      for (let s = 0; s < bt.B; s++) for (let pi = 0; pi < bt.lens[s]; pi++) {
        const t = bt.starts[s] + pi;
        for (let h = 0; h < nh; h++) {
          const kv = Math.floor(h / g), sc = [];
          for (let pj = 0; pj < bt.lens[s]; pj++) {
            if (window && Math.abs(pi - pj) > Math.floor(window / 2)) { sc.push(-Infinity); continue; }
            let d = 0; const j = bt.starts[s] + pj;
            for (let c = 0; c < hd; c++) d += Q[t * qst + h * hd + c] * Kd[j * kst + ko + kv * hd + c];
            sc.push(d * scale);
          }
          const mx = Math.max(...sc), ex = sc.map((v) => Math.exp(v - mx)), z = ex.reduce((a, b) => a + b, 0);
          for (let c = 0; c < hd; c++) {
            let a = 0;
            for (let pj = 0; pj < bt.lens[s]; pj++) a += ex[pj] * Vd[(bt.starts[s] + pj) * kst + vo + kv * hd + c];
            ref[(t * nh + h) * hd + c] = a / z;
          }
        }
      }
      const { e, out } = await harness(device, bt, (e) => {
        const Qb = e.upload(Q, "Q"), Kb = fused ? Qb : e.upload(Kd, "K"), Vb = fused ? Qb : e.upload(Vd, "V");
        const O = e.buf(T * nh * hd, "O");
        if (long) attentionLong(e, Qb, O, { nh, hd, window });
        else attention(e, Qb, Kb, Vb, O, { nh, nkv, hd, fused, window, scale });
        return { buf: O, n: T * nh * hd };
      });
      return { e, bt, out, ref };
    },
  };
}

const poolCase = (H) => ({
  name: `pool_${H}`, op: "pool",
  async run(device, r) {
    const bt = packed(LENS), x = randn(r, bt.T * H), ref = new Float64Array(bt.B * H);
    for (let s = 0; s < bt.B; s++) for (let j = 0; j < H; j++) {
      let a = 0; for (let t = 0; t < bt.lens[s]; t++) a += x[(bt.starts[s] + t) * H + j];
      ref[s * H + j] = a / bt.lens[s];
    }
    const { e, out } = await harness(device, bt, (e) => {
      const X = e.upload(x, "x"), Y = e.buf(bt.B * H, "y");
      pool(e, X, Y, H);
      return { buf: Y, n: bt.B * H };
    });
    return { e, bt, out, ref };
  },
});

// Token-table embedding: V tokens, fp16 scale per 32; t3 with zero points; alpha; BERT position rows.
function embedCase(fmt, H, { alpha = 1, pos = false } = {}) {
  return {
    name: `embed_${fmt}_${H}${alpha !== 1 ? "_alpha" : ""}${pos ? "_pos" : ""}`, op: "embed",
    async run(device, r) {
      const bt = packed(LENS), V = 97, G = H / 32;
      const codes = Uint8Array.from({ length: V * H }, () => Math.floor(r() * (fmt === "t4" ? 16 : 8)));
      const zp = Uint8Array.from({ length: V * G }, () => Math.floor(r() * 8));
      const sc = Uint16Array.from({ length: V * G }, () => toHalf((0.2 + r()) * 0.05));
      const by = new Bytes();
      let packedCodes;
      if (fmt === "t4") { packedCodes = new Uint8Array(V * H / 2); for (let i = 0; i < V * H; i++) packedCodes[i >> 1] |= codes[i] << (4 * (i & 1)); }
      else { packedCodes = new Uint8Array(V * H * 3 / 8); for (let i = 0; i < V * H; i += 8) { let w = 0; for (let k = 0; k < 8; k++) w |= codes[i + k] << (3 * k); packedCodes.set([w & 255, (w >> 8) & 255, (w >> 16) & 255], i * 3 / 8); } }
      const t = { fmt, codes: by.put(packedCodes), scales: by.put(new Uint8Array(sc.buffer)), zp: by.put(zp) };
      const words = by.array, maxPos = 128;
      const pw = pos ? randn(r, maxPos * H) : null, t0 = pos ? randn(r, H) : null;
      const wl = new Words(); wl.w = Array.from(words);
      const posOff = pos ? wl.f32(pw) : NONE, typeOff = pos ? wl.f32(t0) : 0;
      const ids = bt.batch.map((s) => s.map(() => Math.floor(r() * V)));
      const ref = new Float64Array(bt.T * H);
      let o = 0;
      ids.forEach((s) => s.forEach((id, p) => {
        for (let j = 0; j < H; j++) {
          const sv = fromHalf(sc[id * G + (j >> 5)]), c = codes[id * H + j];
          let v = (c - (fmt === "t4" ? 8 : zp[id * G + (j >> 5)])) * sv * alpha;
          if (pos) v += pw[p * H + j] + t0[j];
          ref[o * H + j] = v;
        }
        o++;
      }));
      bt.batch = ids;
      const { e, out } = await harness(device, bt, (e) => {
        const W = e.upload(wl.array, "W"), Y = e.buf(bt.T * H, "y");
        embed(e, W, Y, t, { H, alpha, pos: posOff, type0: typeOff });
        return { buf: Y, n: bt.T * H };
      });
      return { e, bt, out, ref };
    },
  };
}

// RoPE in place: fused [T, (q k v)] rows with two tables, or a plain [T, nh * hd] buffer.
function ropeCase({ nh, hd, parts, tables, sel }) {
  return {
    name: `rope_h${nh}_d${hd}_p${parts}_t${tables}s${sel}`, op: "rope",
    async run(device, r) {
      const bt = packed(LENS), maxSeq = 128, stride = parts === 2 ? 3 * nh * hd : nh * hd, half = hd / 2;
      const tab = ropeTable([160000, 10000].slice(0, tables), hd, maxSeq);
      const x = randn(r, bt.T * stride), ref = Float64Array.from(x);
      bt.batch.forEach((s, b) => s.forEach((_, p) => {
        const t = bt.starts[b] + p;
        for (let part = 0; part < parts; part++) for (let h = 0; h < nh; h++) for (let d = 0; d < half; d++) {
          const base = t * stride + part * nh * hd + h * hd;
          const c = tab[(sel * maxSeq + p) * half + d], sn = tab[tables * maxSeq * half + (sel * maxSeq + p) * half + d];
          const x1 = x[base + d], x2 = x[base + d + half];
          ref[base + d] = x1 * c - x2 * sn; ref[base + d + half] = x2 * c + x1 * sn;
        }
      }));
      const { e, out } = await harness(device, bt, (e) => {
        const T = e.upload(tab, "rope"), X = e.buf(bt.T * stride, "x", U.COPY_DST);
        device.queue.writeBuffer(X, 0, x);
        rope(e, T, X, { nh, hd, maxSeq, tables, sel, parts, stride });
        return { buf: X, n: bt.T * stride, reset: () => device.queue.writeBuffer(X, 0, x) };
      });
      return { e, bt, out, ref };
    },
  };
}

// Gated activation: ModernBERT [gate | up] rows (gelu), EG2's in-place gelu_tanh against a strided B, LFM's silu.
function gatedCase(act, I, layout, mult = 1) {
  return {
    name: `gated_${act}_${I}_${layout}${mult > 1 ? `_x${mult}` : ""}`, op: "gated",
    async run(device, r) {
      const bt = packed(LENS), R = bt.T * mult;
      const act_ = { gelu: (x) => x * 0.5 * (1 + erfRef(x / Math.SQRT2)), gelu_tanh: (x) => 0.5 * x * (1 + Math.tanh(Math.max(-15, Math.min(15, 0.7978845608028654 * (x + 0.044715 * x ** 3))))), silu: (x) => x / (1 + Math.exp(-x)) }[act];
      let A, B, sa, oa = 0, sb, ob = 0, inPlace = false;
      if (layout === "cat") { A = B = randn(r, R * 2 * I, 2); sa = sb = 2 * I; ob = I; }
      else { inPlace = true; A = randn(r, R * I, 2); sa = I; sb = 2 * I + 32; ob = 16; B = randn(r, R * sb); }
      const ref = new Float64Array(R * I);
      for (let i = 0; i < R; i++) for (let j = 0; j < I; j++) ref[i * I + j] = act_(A[i * sa + oa + j]) * B[i * sb + ob + j];
      const { e, out } = await harness(device, bt, (e) => {
        const Ab = e.buf(A.length, "A", U.COPY_DST); device.queue.writeBuffer(Ab, 0, A);
        const Bb = layout === "cat" ? Ab : e.upload(B, "B"), Y = inPlace ? Ab : e.buf(R * I, "Y");
        gated(e, Ab, Bb, Y, { I, act, sa, oa, sb, ob, mult });
        return { buf: Y, n: R * I, reset: inPlace ? () => device.queue.writeBuffer(Ab, 0, A) : null };
      });
      return { e, bt, out, ref };
    },
  };
}
// erf to double precision (W. J. Cody's rational approximations are overkill here: Abramowitz & Stegun
// 7.1.26 is the kernel's own, so the reference uses a series / continued fraction instead).
function erfRef(x) {
  const ax = Math.abs(x);
  if (ax < 2.5) { let s = 0, t = ax, n = 0; while (Math.abs(t) > 1e-17 * Math.abs(s) || n < 3) { s += t / (2 * n + 1); n++; t *= -ax * ax / n; if (n > 200) break; } return Math.sign(x) * 2 / Math.sqrt(Math.PI) * s; }
  let f = 0; for (let n = 60; n >= 1; n--) f = n / 2 / (ax + f);
  return Math.sign(x) * (1 - Math.exp(-ax * ax) / Math.sqrt(Math.PI) / (ax + f));
}

// Copy rows idx of X [T, H] into Y [P, H] (the decider's [MASK] positions).
const gatherCase = (H, P) => ({
  name: `gather_${H}_${P}`, op: "gather",
  async run(device, r) {
    const bt = packed(LENS), x = randn(r, bt.T * H);
    const idx = Uint32Array.from({ length: P }, () => Math.floor(r() * bt.T)), ref = new Float64Array(P * H);
    idx.forEach((t, i) => { for (let j = 0; j < H; j++) ref[i * H + j] = x[t * H + j]; });
    const { e, out } = await harness(device, bt, (e) => {
      const I = e.upload(idx, "idx"), X = e.upload(x, "x"), Y = e.buf(P * H, "y");
      e.step("gather", { a: [H, P] }, [I, X, Y], () => [P]);
      return { buf: Y, n: P * H };
    });
    return { e, bt, out, ref };
  },
});

const l2Case = (D) => ({
  name: `l2_${D}`, op: "l2",
  async run(device, r) {
    const bt = packed(LENS), x = randn(r, bt.B * D), ref = new Float64Array(bt.B * D);
    for (let s = 0; s < bt.B; s++) {
      let q = 0; for (let j = 0; j < D; j++) q += x[s * D + j] ** 2;
      const inv = 1 / Math.max(Math.sqrt(q), 1e-12);
      for (let j = 0; j < D; j++) ref[s * D + j] = x[s * D + j] * inv;
    }
    const { e, out } = await harness(device, bt, (e) => {
      const X = e.upload(x, "x"), Y = e.buf(bt.B * D, "y");
      l2(e, X, Y, D);
      return { buf: Y, n: bt.B * D };
    });
    return { e, bt, out, ref };
  },
});

// The shapes the shipped engines use (EG2: H 768, hd 256, nh 3 / nkv 1, window 512, q4; Ettin-17m: H 256,
// hd 64, nh 4, fused qkv, int8 or f32, window 128, and in the decider up to 1024 tokens with
// attentionLong; MoEdernColBERT: rms, f32), at sizes SwiftShader runs in seconds.
// ---- MoE ops (step 4) ------------------------------------------------------------------------------
// A random top-2 assignment: two distinct experts per token, each expert's slot list in shuffled order.
function assign(r, T, E) {
  const ex = new Int32Array(2 * T), lists = Array.from({ length: E }, () => []);
  for (let t = 0; t < T; t++) {
    const a = Math.floor(r() * E); let b = Math.floor(r() * (E - 1)); if (b >= a) b++;
    ex[2 * t] = a; ex[2 * t + 1] = b; lists[a].push(2 * t); lists[b].push(2 * t + 1);
  }
  for (const l of lists) for (let i = l.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [l[i], l[j]] = [l[j], l[i]]; }
  const cap = 2 * T, cnt = new Uint32Array(8), list = new Uint32Array(E * cap);
  lists.forEach((l, e) => { cnt[e] = l.length; list.set(l, e * cap); });
  return { ex, cnt, list, cap };
}

// Router: compared as the gates followed by each slot's expert (rebuilt from cnt and list, so a lost or
// duplicated slot shows up as a wrong expert).
function routeCase(E, H) {
  return {
    name: `moe_route_e${E}_h${H}`, op: "route",
    async run(device, r) {
      const bt = packed(LENS), T = bt.T, cap = 2 * T;
      const X = randn(r, T * H), Wr = randn(r, E * H, 1 / Math.sqrt(H));
      const ref = new Float64Array(4 * T);
      for (let t = 0; t < T; t++) {
        const lg = Array.from({ length: E }, (_, e) => { let a = 0; for (let j = 0; j < H; j++) a += X[t * H + j] * Wr[e * H + j]; return a; });
        const mx = Math.max(...lg), pr = lg.map((v) => Math.exp(v - mx)), z = pr.reduce((a, b) => a + b, 0);
        const i0 = pr.indexOf(Math.max(...pr)); let i1 = i0 === 0 ? 1 : 0;
        for (let e = 0; e < E; e++) if (e !== i0 && pr[e] > pr[i1]) i1 = e;
        const w0 = pr[i0] / z, w1 = pr[i1] / z, n = w0 + w1 + 1e-9;
        ref[2 * t] = w0 / n; ref[2 * t + 1] = w1 / n; ref[2 * T + 2 * t] = i0; ref[2 * T + 2 * t + 1] = i1;
      }
      let cntB, listB;
      const { e, out } = await harness(device, bt, (e) => {
        const words = new Words(), wOff = words.f32(Wr), W = e.upload(words.array, "W"), Xb = e.upload(X, "x");
        cntB = e.buf(8, "cnt"); listB = e.buf(E * cap, "list");
        const gw = e.buf(2 * T, "gw");
        route(e, W, Xb, cntB, listB, gw, { H, wOff, E, cap });
        return { buf: gw, n: 2 * T, post: async (g) => {
          const cnt = new Uint32Array((await readBuf(device, cntB, 8)).buffer), list = new Uint32Array((await readBuf(device, listB, E * cap)).buffer);
          const ex = new Float32Array(2 * T).fill(-1);
          for (let k = 0; k < E; k++) for (let i = 0; i < cnt[k]; i++) { const s = list[k * cap + i]; ex[s] = ex[s] === -1 ? k : 99; }
          const o = new Float32Array(4 * T); o.set(g); o.set(ex, 2 * T); return o;
        } };
      });
      return { e, bt, out, ref };
    },
  };
}

// Grouped matmul over a host-made assignment. mode 1: token rows in, slot rows out; mode 2: slot rows both.
function expertCase(kernel, mode, { E, N, K }) {
  return {
    name: `moe_${kernel === "tiled" ? "matmul_tiled" : "gmm"}_mode${mode}_e${E}_${N}x${K}`, op: "expertMatmul",
    async run(device, r) {
      const bt = packed(LENS), T = bt.T, { ex, cnt, list, cap } = assign(r, T, E);
      const rows = mode === 1 ? T : 2 * T, X = randn(r, rows * K), Wd = randn(r, E * N * K, 1 / Math.sqrt(K));
      const ref = new Float64Array(2 * T * N);
      for (let s = 0; s < 2 * T; s++) {
        const xr = mode === 1 ? s >> 1 : s, w = ex[s] * N * K;
        for (let n = 0; n < N; n++) { let a = 0; for (let k = 0; k < K; k++) a += X[xr * K + k] * Wd[w + n * K + k]; ref[s * N + n] = a; }
      }
      const { e, out } = await harness(device, bt, (e) => {
        const words = new Words(); words.f32([1, 2, 3]); const wOff = words.f32(Wd);
        const W = e.upload(words.array, "W"), Xb = e.upload(X, "x"), Y = e.buf(2 * T * N, "y");
        expertMatmul(e, W, e.upload(cnt, "cnt"), e.upload(list, "list"), Xb, Y, { wOff, N, K, E, cap, mode, kernel });
        return { buf: Y, n: 2 * T * N };
      });
      return { e, bt, out, ref };
    },
  };
}

function tiledCase({ N, K, accumulate }) {
  return {
    name: `matmul_tiled_${N}x${K}${accumulate ? "_acc" : ""}`, op: "matmulTiled",
    async run(device, r) {
      const bt = packed(LENS), T = bt.T, X = randn(r, T * K), Wd = randn(r, N * K, 1 / Math.sqrt(K)), base = randn(r, T * N);
      const ref = new Float64Array(T * N);
      for (let t = 0; t < T; t++) for (let n = 0; n < N; n++) { let a = accumulate ? base[t * N + n] : 0; for (let k = 0; k < K; k++) a += X[t * K + k] * Wd[n * K + k]; ref[t * N + n] = a; }
      const { e, out } = await harness(device, bt, (e) => {
        const words = new Words(); words.f32([7]); const wOff = words.f32(Wd);
        const W = e.upload(words.array, "W"), Y = e.buf(T * N, "y", U.COPY_DST), dummy = e.buf(4, "unused");
        device.queue.writeBuffer(Y, 0, base);
        matmulTiled(e, W, e.upload(X, "x"), Y, { wOff, N, K, cnt: dummy, list: dummy, accumulate });
        return { buf: Y, n: T * N, reset: () => device.queue.writeBuffer(Y, 0, base) };
      });
      return { e, bt, out, ref };
    },
  };
}

function combineCase(H) {
  return {
    name: `moe_combine_${H}`, op: "combine",
    async run(device, r) {
      const bt = packed(LENS), T = bt.T, od = randn(r, 2 * T * H), gw = Float32Array.from({ length: 2 * T }, () => r()), x = randn(r, T * H);
      const ref = Float64Array.from({ length: T * H }, (_, i) => { const t = Math.floor(i / H), j = i % H; return x[i] + gw[2 * t] * od[2 * t * H + j] + gw[2 * t + 1] * od[(2 * t + 1) * H + j]; });
      const { e, out } = await harness(device, bt, (e) => {
        const X = e.buf(T * H, "x", U.COPY_DST);
        device.queue.writeBuffer(X, 0, x);
        combine(e, e.upload(od, "od"), e.upload(gw, "gw"), X, { H });
        return { buf: X, n: T * H, reset: () => device.queue.writeBuffer(X, 0, x) };
      });
      return { e, bt, out, ref };
    },
  };
}

function l2tokCase(D) {
  return {
    name: `l2tok_${D}`, op: "l2tok",
    async run(device, r) {
      const bt = packed(LENS), T = bt.T, x = randn(r, T * D), ref = new Float64Array(T * D);
      for (let t = 0; t < T; t++) { let s = 0; for (let j = 0; j < D; j++) s += x[t * D + j] ** 2; s = Math.sqrt(s); for (let j = 0; j < D; j++) ref[t * D + j] = x[t * D + j] / s; }
      const { e, out } = await harness(device, bt, (e) => {
        const Y = e.buf(T * D, "y");
        l2tok(e, e.upload(x, "x"), Y, { D });
        return { buf: Y, n: T * D };
      });
      return { e, bt, out, ref };
    },
  };
}

export const cases = [
  matmulCase("f32", { N: 64, K: 64 }),
  matmulCase("f32", { N: 96, K: 256, bias: true, geluOn: true }),
  matmulCase("f32", { N: 32, K: 128, rowsSel: 1, alpha: 0.5 }),
  matmulCase("i8", { N: 64, K: 256 }),
  matmulCase("i8", { N: 32, K: 96, accumulate: true, bias: true }),
  matmulCase("q4", { N: 64, K: 256 }),
  matmulCase("q4", { N: 32, K: 768, geluOn: true, accumulate: true }),
  normCase("rms", 256, 0, false),
  normCase("rms", 768 / 2, 2, false),
  normCase("ln", 256, 0, false),
  normCase("ln", 256, 1, true),
  normCase("ln", 512, 0, true),
  normCase("rms", 64, 1, false),
  attnCase({ nh: 4, nkv: 4, hd: 64, fused: true }),
  attnCase({ nh: 4, nkv: 4, hd: 64, fused: true, window: 16 }),
  attnCase({ nh: 4, nkv: 2, hd: 32, fused: false }),
  attnCase({ nh: 3, nkv: 1, hd: 256, fused: false, window: 64 }),
  attnCase({ nh: 4, nkv: 4, hd: 64, fused: true, long: true }),
  attnCase({ nh: 4, nkv: 4, hd: 64, fused: true, long: true, window: 128 }),
  gatherCase(256, 37),
  embedCase("t4", 256),
  embedCase("t3", 512, { alpha: 22.627 }),
  embedCase("t3", 256, { pos: true }),
  ropeCase({ nh: 4, hd: 64, parts: 2, tables: 2, sel: 1 }),
  ropeCase({ nh: 3, hd: 32, parts: 1, tables: 1, sel: 0 }),
  gatedCase("gelu", 384, "cat"),
  gatedCase("gelu", 96, "cat", 2),
  gatedCase("gelu_tanh", 256, "inplace"),
  gatedCase("silu", 160, "inplace"),
  poolCase(256),
  l2Case(768),
  routeCase(8, 96),
  routeCase(5, 256),
  expertCase("tiled", 1, { E: 4, N: 128, K: 96 }),
  expertCase("tiled", 2, { E: 8, N: 64, K: 48 }),
  expertCase("rows", 1, { E: 4, N: 96, K: 128 }),
  expertCase("rows", 2, { E: 8, N: 64, K: 64 }),
  tiledCase({ N: 128, K: 80, accumulate: false }),
  tiledCase({ N: 64, K: 256, accumulate: true }),
  combineCase(96),
  l2tokCase(128),
  l2tokCase(64),
];

// opts: only (substring of case name or op), seed, tol (default 1e-4), reps (0: no timing), rounds.
export async function runOps(device, { only = "", seed = 1, tol = 1e-4, reps = 0, rounds = 5, log = () => {} } = {}) {
  const res = [];
  for (const c of cases) {
    if (only && !c.name.includes(only) && c.op !== only) continue;
    const r = rng(seed + res.length * 7919);
    let row = { name: c.name, op: c.op };
    try {
      const { e, bt, out, ref } = await c.run(device, r);
      const enc = device.createCommandEncoder();
      e._record(enc, bt.batch);
      device.queue.submit([enc.finish()]);
      let got = await readBuf(device, out.buf, out.n);
      if (out.post) got = await out.post(got);   // a case that compares a derived view (the router)
      let maxErr = 0, maxRef = 0, dot = 0, ga = 0, rb = 0;
      for (let i = 0; i < got.length; i++) {
        maxErr = Math.max(maxErr, Math.abs(got[i] - ref[i])); maxRef = Math.max(maxRef, Math.abs(ref[i]));
        dot += got[i] * ref[i]; ga += got[i] * got[i]; rb += ref[i] * ref[i];
      }
      const rel = maxErr / Math.max(maxRef, 1e-30);
      row = { ...row, pass: Number.isFinite(rel) && rel <= tol, rel, maxErr, cos: dot / Math.sqrt(ga * rb), n: out.n, dispatches: e.steps.length };
      if (reps) {
        if (out.reset) out.reset();   // accumulate / residual cases: timing only, values no longer checked
        row.ms = await timeIt(device, e, bt, reps, rounds);
      }
      e.destroy();
    } catch (err) {
      row = { ...row, pass: false, error: String(err.message || err) };
    }
    log(`${row.pass ? "ok  " : "FAIL"} ${row.name} rel ${row.rel?.toExponential(2)}${row.ms != null ? ` ${row.ms.toFixed(3)} ms` : ""}${row.error ? " " + row.error : ""}`);
    res.push(row);
  }
  return res;
}
