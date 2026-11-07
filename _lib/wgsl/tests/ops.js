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
import { KERNELS, matmul, norm, attention, attentionLong, pool, l2 } from "../kernels.js";

const U = GPUBufferUsage;

// ---- deterministic inputs ----------------------------------------------------------------------
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const randn = (r, n, sc = 1) => Float32Array.from({ length: n }, () => sc * Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()));

// GELU with the same erf approximation as the kernel (A&S 7.1.26, |error| < 1.5e-7).
function gelu(x) {
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

// ---- weight-buffer builder: word-addressed u32 store ----------------------------------------------
class Words {
  constructor() { this.w = []; }
  f32(a) { const o = this.w.length; const u = new Uint32Array(Float32Array.from(a).buffer); for (const x of u) this.w.push(x); return o; }
  u32(a) { const o = this.w.length; for (const x of a) this.w.push(x >>> 0); return o; }
  get array() { return Uint32Array.from(this.w.length ? this.w : [0]); }
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
  poolCase(256),
  l2Case(768),
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
      const got = await readBuf(device, out.buf, out.n);
      let maxErr = 0, maxRef = 0, dot = 0, ga = 0, rb = 0;
      for (let i = 0; i < out.n; i++) {
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
