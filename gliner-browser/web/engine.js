// TextEngine: GLiNER's Ettin text encoder (ModernBERT) in hand-written WGSL, returning the final
// hidden states at chosen token positions (each word's first subtoken).
//
//   const eng = await TextEngine.create(device, manifest, weightsBin, { maxTokens: 8192, maxSeqs: 512, maxPos: 8192 });
//   const job = eng.submit([ids0, ids1, ...], idx);   // idx: Uint32Array of token indexes into the packed batch
//   const { rows, gpuMs, readMs } = await job.read(); // rows: Float32Array(idx.length * hidden)
//
// Copied from browser-decider/web/engine.js (DeciderEngine), which runs Ettin-17m the same way. What
// changed for gliner-bi-edge-v2.0's Ettin-32m and its packs:
//   - nothing in the forward pass was hard-wired to 17m: hidden, heads, inter, layer count, which
//     layers are global, the window and the RoPE thetas all come from the manifest (32m: 384 / 6 / 576,
//     10 layers, global every 3rd, window 128). The shared kernels take hidden 384 (norm and embed need
//     a multiple of 64, matmul K and N multiples of 32) and head dim 64.
//   - fp32 packs: an f32 token table (own `embed_f32` shader; the shared embed kernel has t4/t3/t8 only)
//     and f32 matmuls (the shared `matmul_f32` template), alongside the t4 table and int8 matmuls.
//   - only the encoder's tensors are uploaded (the GLiNER manifest also lists the label encoder and the
//     span head, which run in JavaScript); the fp32 encoder is 127.5 MB in one binding.
//   - no decision head: the gathered rows are read back for the JavaScript span head; the gather is
//     dispatched over exactly the batch's row count, in 2-D past 65,535 rows.
//   - submit() and read() are separate so a caller can run the CPU head on one batch while the GPU
//     encodes the next.
//   - optional span head (opts.ghead, from the GLiNER manifest): the per-word start / end MLPs and the
//     two halves of out_project.0 run on the gathered rows in the same submit, then the span x label
//     scores (relu(a_s + b_{s+k}) . q_c + r_c, GRef's factorised form), so a batch reads back
//     [rows, K, C] scores instead of [rows, hidden] word vectors. Head weights int8 or fp32, as packed.
// run/gref.py (GRefBi.reps -> browser-decider's Ref.hidden, GRef.span_logits) is the NumPy spec.

import { SeqEngine, HEADER, BYTE, split } from "../../_lib/wgsl/runtime.js";
import { KERNELS, attentionLong, embed, gated, matmul, norm, rope, ropeTable } from "../../_lib/wgsl/kernels.js";
import { Pack } from "./jshead.js";

export const MAX_SEQ = 1024;   // attentionLong holds 1,024 keys; the RoPE table is built to this length
const NONE = 0xffffffff;

// y[t, :] = f32 table row tok[t].x; p.a.x = table word offset. Reads every binding (METHODS: an unread
// binding invalidates the bind group).
const embedF32 = (H) => HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${BYTE}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x + wid.y * 65535u;
  if (t >= dn.n.x) { return; }
  let base = p.a.x + tok[t].x * ${H}u;
  for (var j = l; j < ${H}u; j += 64u) { y[t * ${H}u + j] = wf32(base + j); }
}`;

// Y[r, :] = X[idx[r], :] for r < p.a.y rows of p.a.x floats, grid split over x and y.
const GATHER2 = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r = wid.x + wid.y * 65535u; let H = p.a.x;
  if (r >= p.a.y || dn.n.x == 0u) { return; }
  let src = idx[r];
  if (src == 0xffffffffu) { for (var j = l; j < H; j += 64u) { Y[r * H + j] = 0.0; } return; }   // a word with no subtoken
  for (var j = l; j < H; j += 64u) { Y[r * H + j] = X[src * H + j]; }
}`;

// Y[i] = max(Y[i], 0) over the first dn.n[p.a.y] rows of p.a.x floats.
const RELU = HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let i = (wid.x + wid.y * 65535u) * 64u + l;
  if (i < dn.n[p.a.y] * p.a.x) { Y[i] = max(Y[i], 0.0); }
}`;

// Span x label scores for the gathered word rows: Z[(r * K + k) * C + c] = R[c] + sum_j relu(A[r, j] +
// B[r + k, j]) Q[c, j] for k < lim[r] (the words left in r's text, at most K); other entries are not
// written (the caller fills them with -Infinity). One workgroup of 64 per (r, k), thread l holding
// j = l + 64 i; labels four at a time with a tree reduction. Rows dn.n.z, labels dn.n.w. No early
// return: every thread reaches every barrier, and lim is only read to gate the writes.
const scoresWGSL = (D, K) => HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read> Q: array<f32>;
@group(0) @binding(5) var<storage, read> R: array<f32>;
@group(0) @binding(6) var<storage, read> lim: array<u32>;
@group(0) @binding(7) var<storage, read_write> Z: array<f32>;
const D = ${D}u;
const K = ${K}u;
const PER = ${D / 64}u;
var<workgroup> red: array<vec4<f32>, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let g = wid.x + wid.y * 65535u;
  let rows = dn.n.z; let C = dn.n.w + 0u * p.a.x;   // p read so binding 0 stays in the layout
  let live = g < rows * K;
  let r = min(g / K, max(rows, 1u) - 1u); let k = g % K;
  let valid = live && k < lim[r];
  var v: array<f32, ${D / 64}>;
  for (var i = 0u; i < PER; i++) {
    let j = l + 64u * i;
    var x = 0.0;
    if (valid) { x = max(A[r * D + j] + B[(r + k) * D + j], 0.0); }
    v[i] = x;
  }
  for (var c0 = 0u; c0 < C; c0 += 4u) {
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0u; i < PER; i++) {
      let j = l + 64u * i;
      acc += v[i] * vec4<f32>(Q[c0 * D + j], Q[(c0 + 1u) * D + j], Q[(c0 + 2u) * D + j], Q[(c0 + 3u) * D + j]);
    }
    red[l] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
      if (l < s) { red[l] += red[l + s]; }
      workgroupBarrier();
    }
    if (l == 0u && valid) {
      let o = g * C;
      for (var t = 0u; t < 4u; t++) { if (c0 + t < C) { Z[o + c0 + t] = red[0][t] + R[c0 + t]; } }
    }
    workgroupBarrier();
  }
}`;

// The text-encoder part of a GLiNER packed manifest (browser-decider's schema at the top level).
export function encoderManifest(m) {
  const keys = ["arch", "hidden", "heads", "inter", "eps", "window", "global_every", "theta_global", "theta_local",
    "table", "emb_norm", "layers", "final_norm", "variant"];
  return structuredClone(Object.fromEntries(keys.map((k) => [k, m[k]])));
}

export class TextEngine extends SeqEngine {
  static create(device, manifest, data, opts = {}) {
    return super.create(device, encoderManifest(manifest), data, opts);
  }

  constructor(device, manifest, opts) {
    super(device, manifest, { ...opts, maxSeq: MAX_SEQ });
    this.maxPos = opts.maxPos ?? 8192;
    this.dim = manifest.hidden;
    this._nPos = 1;
    this.ghead = opts.ghead ? structuredClone(opts.ghead) : null;   // span head records (offsets into the weight file)
    this.K = opts.maxWidth ?? 12;
    this.maxLabels = opts.maxLabels ?? 128;
    this.scoreCap = opts.scoreCap ?? this.maxPos * this.K * 64;      // floats in the scores buffer: rows * K * C must fit
  }

  // Most word rows one batch may hold with C labels in the GPU head (scores buffer), and whether it runs at all.
  headRows(C) { return this.ghead && C <= this.maxLabels ? Math.min(this.maxPos, Math.floor(this.scoreCap / (this.K * Math.max(C, 1)))) : 0; }

  async _build(bytes) {
    const m = this.m, T = this.maxTokens;
    const H = m.hidden, nh = m.heads, D = H / nh, I = m.inter;
    if (m.arch !== "modernbert") throw new Error(`text encoder arch ${m.arch}`);
    if (H % 64 || D % 4 || D > 128) throw new Error(`hidden ${H} / head dim ${D} not supported by the kernels`);
    if (![4, 32].includes(m.table.bits)) throw new Error(`token table bits ${m.table.bits} (engine runs 4 and 32)`);
    const head = this.ghead ? this._headWeights(bytes) : null;   // from the original file offsets, before packing
    bytes = this.uploadPacked(bytes);   // rewrites this.m's offsets to the packed buffer
    const U = GPUBufferUsage, buf = (n, label, extra) => this.buf(n, label, extra);
    this.rope = this.upload(ropeTable([m.theta_global, m.theta_local], D, MAX_SEQ), "rope");
    const B_ = {
      tok: buf(T * 4, "tok", U.COPY_DST), seqs: buf(this.maxSeqs * 4, "seqs", U.COPY_DST),
      e: buf(T * H, "e"), h: buf(T * H, "h"), a: buf(T * H, "a"), o: buf(T * H, "o"),
      qkv: buf(T * 3 * H, "qkv"), u: buf(T * 2 * I, "u"), g: buf(T * I, "g"),
      idx: buf(this.maxPos, "idx", U.COPY_DST), out: buf(this.maxPos * H, "out"),
    };
    this.b = B_;
    this.shaders = { ...KERNELS, embed_f32: embedF32(H), gather2: GATHER2 };
    const word = (t) => { if (t.offset % 4) throw new Error("unaligned f32 tensor"); return t.offset / 4; };

    // int8 matmuls: transpose the [K, N] codes once into the [N, K] layout the i8 kernel reads (as DeciderEngine)
    const lins = [];
    const walk = (o) => { if (o && typeof o === "object") { if ("K" in o && "N" in o) lins.push(o); else Object.values(o).forEach(walk); } };
    walk(m.layers);
    let words = 0;
    for (const w of lins) {
      if (w.K % 32 || w.N % 32) throw new Error(`matmul shape ${w.N} x ${w.K}`);
      if (w.q) { w._wt = words; words += w.N * w.K / 4; }
    }
    if (words) {
      const wt = new Int8Array(words * 4);
      for (const w of lins) {
        if (!w.q) continue;
        const src = new Int8Array(bytes.buffer, bytes.byteOffset + w.q.offset, w.K * w.N), dst = w._wt * 4;
        for (let k = 0; k < w.K; k++) for (let n = 0; n < w.N; n++) wt[dst + n * w.K + k] = src[k * w.N + n];
      }
      this.Wt = this.upload(wt, "int8 weights [N, K]");
    }
    const mm = (w, X, Y, mode = 0) => {
      const fmt = w.q ? { fmt: "i8", offs: [w._wt, word(w.s)], N: w.N, K: w.K } : { fmt: "f32", offs: [word(w.f32_t)], N: w.N, K: w.K };
      matmul(this, fmt, this.W, X, Y, { rowsSel: 0, bias: NONE, gelu: !!(mode & 1), accumulate: !!(mode & 2), Wt: this.Wt });
    };
    const ln = (w, X, Y) => norm(this, "ln", this.W, word(w), X, Y, { D: H, eps: m.eps, bOff: NONE, mode: 0, rowsSel: 0 });

    const tb = m.table;
    if (tb.bits === 4) embed(this, this.W, B_.e, { fmt: "t4", codes: tb.codes.offset, scales: tb.scales.offset }, { H });
    else this.step("embed_f32", { a: [word(tb.f32)] }, [this.W, B_.tok, B_.e], (n) => split(n[0]));
    ln(m.emb_norm, B_.e, B_.h);
    for (const L of m.layers) {
      let X = B_.h;
      if (L.attn_norm) { ln(L.attn_norm, B_.h, B_.a); X = B_.a; }
      mm(L.Wqkv, X, B_.qkv);
      rope(this, this.rope, B_.qkv, { nh, hd: D, maxSeq: MAX_SEQ, tables: 2, sel: L.global ? 0 : 1, parts: 2, stride: 3 * H });
      attentionLong(this, B_.qkv, B_.o, { nh, hd: D, window: L.global ? 0 : m.window });
      mm(L.Wo, B_.o, B_.h, 2);
      ln(L.mlp_norm, B_.h, B_.a);
      mm(L.Wi, B_.a, B_.u);
      gated(this, B_.u, B_.u, B_.g, { I, sa: 2 * I, sb: 2 * I, ob: I });
      mm(L.Wo_mlp, B_.g, B_.h, 2);
    }
    ln(m.final_norm, B_.h, B_.a);
    // the gather's row count changes per batch: its own uniform, rewritten by submit()
    this.gatherU = this.device.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST, label: "gather" });
    this._owned.push(this.gatherU);
    this.step("gather2", { a: [H, 0] }, [B_.idx, B_.a, B_.out], () => split(this._nPos));
    this.steps[this.steps.length - 1].entries[0] = { binding: 0, resource: { buffer: this.gatherU } };
    this.readPos = this.device.createBuffer({ size: this.maxPos * H * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "readback rows" });
    this._owned.push(this.readPos);
    this.nEnc = this.steps.length;   // steps past this are the span head, recorded only when a batch asks for it
    if (head) this._buildHead(head);
  }

  // Head weights as two GPU buffers: HW (f32 words: scales, biases, fp32 matrices) and HWt (int8 codes,
  // [N, K] rows, the layout matmul_i8 reads). out_project.0 is split at H into the start and end halves.
  _headWeights(bytes) {
    const pack = new Pack(bytes), gh = this.ghead;
    const f32 = [], i8 = [];
    let nf = 0, ni = 0;
    const addF = (a) => { f32.push(a); const o = nf; nf += a.length; return o; };
    const addI = (a) => { i8.push(a); const o = ni; ni += a.length; return o / 4; };
    const lin = (L) => L.s ? { fmt: "i8", offs: [addI(L.w), addF(L.s)], N: L.N, K: L.K } : { fmt: "f32", offs: [addF(L.w)], N: L.N, K: L.K };
    const mlp = (r) => ({ l0: lin(pack.lin(r.l0)), b0: addF(pack.f32(r.b0)), l1: lin(pack.lin(r.l1)), b1: addF(pack.f32(r.b1)) });
    const start = mlp(gh.start), end = mlp(gh.end);
    const o0 = pack.lin(gh.out.l0), H = start.l1.N;
    if (o0.K !== 2 * H) throw new Error("out_project.0 must take concat(start, end)");
    const W1a = lin(Pack.kslice(o0, 0, H)), W1b = lin(Pack.kslice(o0, H, H)), b1 = addF(pack.f32(gh.out.b0));
    for (const w of [start.l0, start.l1, end.l0, end.l1, W1a, W1b]) if (w.K % 32 || w.N % 32) throw new Error(`head matmul shape ${w.N} x ${w.K}`);
    const F = new Float32Array(nf), I = new Int8Array(Math.max(ni, 4));
    let o = 0; for (const a of f32) { F.set(a, o); o += a.length; }
    o = 0; for (const a of i8) { I.set(a, o); o += a.length; }
    return { start, end, W1a, W1b, b1, H, D: W1a.N, F, I, hasI8: ni > 0 };
  }

  // Record the head after the gather: start / end MLPs (ReLU after each layer: GRef applies one to the
  // MLP output too), the out_project.0 halves, then the scores. Rows are dn.n.z (the gathered words).
  _buildHead(h) {
    const U = GPUBufferUsage, P = this.maxPos, H = h.H, D = h.D, I0 = h.start.l0.N;
    if (D % 64 || D > 4096) throw new Error(`head width ${D}`);
    this.headD = D;
    const HW = this.upload(h.F, "head f32"), HWt = h.hasI8 ? this.upload(h.I, "head int8 [N, K]") : null;
    const Lp = Math.ceil(this.maxLabels / 4) * 4 + 4;
    const B_ = this.hb = {
      hid: this.buf(P * I0, "head mlp hidden"), S: this.buf(P * H, "start"), E: this.buf(P * H, "end"),
      A: this.buf(P * D, "a"), B: this.buf(P * D, "b"),
      Q: this.buf(Lp * D, "label q", U.COPY_DST), R: this.buf(Lp, "label r", U.COPY_DST),
      lim: this.buf(P, "lim", U.COPY_DST), Z: this.buf(this.scoreCap, "scores"),
    };
    this.shaders.relu = RELU;
    this.shaders.span_scores = scoresWGSL(D, this.K);
    const mm = (w, X, Y, bias) => matmul(this, w, HW, X, Y, { rowsSel: 2, bias, Wt: HWt ?? undefined });
    const relu = (Y, N) => this.step("relu", { a: [N, 2] }, [Y], (n) => split(Math.ceil(n[2] * N / 64)));
    for (const [m, Y] of [[h.start, B_.S], [h.end, B_.E]]) {
      mm(m.l0, this.b.out, B_.hid, m.b0); relu(B_.hid, I0);
      mm(m.l1, B_.hid, Y, m.b1); relu(Y, H);
    }
    mm(h.W1a, B_.S, B_.A, h.b1);
    mm(h.W1b, B_.E, B_.B);
    this.nWords = this.steps.length;   // steps up to here give A and B
    this.step("span_scores", {}, [B_.A, B_.B, B_.Q, B_.R, B_.lim, B_.Z], (n) => split(n[2] * this.K));
    this.readZ = this.device.createBuffer({ size: this.scoreCap * 4, usage: U.MAP_READ | U.COPY_DST, label: "readback scores" });
    this._owned.push(this.readZ);
  }

  // The current label set for the GPU head: q [C, D] and r [C] (SpanHead.labels, in the label order).
  setLabels(q, r, C) {
    if (!this.hb) throw new Error("engine built without a span head");
    if (C > this.maxLabels) throw new Error(`${C} labels exceed the GPU head's ${this.maxLabels}`);
    if (C) {
      this.device.queue.writeBuffer(this.hb.Q, 0, q.subarray(0, C * this.headD));
      this.device.queue.writeBuffer(this.hb.R, 0, Float32Array.from(r.subarray(0, C)));
    }
    this.C = C;
  }

  // Encode a batch (arrays of token ids, each at most MAX_SEQ) and gather the final hidden rows at
  // idx (indexes into the packed token axis: sequence offset + position; NONE gives a zero row). Returns
  // { read() } whose promise gives { rows, gpuMs, readMs }. One readback buffer per kind: read() a job
  // before submitting the next.
  // head (the engine's span head, after setLabels): { lim: Uint32Array per row, words: false } adds the
  // head and reads back scores [rows, K, C] (rows unset, entries with k >= lim[r] unwritten); words: true
  // stops after out_project.0 and reads back A and B [rows, D] each (to time JavaScript scoring against it).
  submit(batch, idx, head = null) {
    const H = this.dim, P = idx.length;
    if (P > this.maxPos) throw new Error(`${P} rows exceed maxPos ${this.maxPos}`);
    if (this._pending) throw new Error("read() the previous batch before submitting another");
    if (head && P > this.headRows(this.C)) throw new Error(`${P} rows exceed the GPU head's ${this.headRows(this.C)} at ${this.C} labels`);
    this._nPos = Math.max(P, 1);
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 2).set([H, P]);
    const q = this.device.queue;
    q.writeBuffer(this.gatherU, 0, u);
    if (P) q.writeBuffer(this.b.idx, 0, idx);
    const enc = this.device.createCommandEncoder();
    this._record(enc, batch, this.nEnc);
    let rd, n, kind;
    if (!head) { kind = "rows"; rd = this.readPos; n = P * H * 4; if (n) enc.copyBufferToBuffer(this.b.out, 0, rd, 0, n); }
    else {
      const T = batch.reduce((a, x) => a + x.length, 0), nn = [T, batch.length, P, this.C];
      q.writeBuffer(this.dyn, 0, new Uint32Array(nn));   // after _record's [T, B, 0, 0]: same submit, last write wins
      q.writeBuffer(this.hb.lim, 0, head.lim);
      const pass = enc.beginComputePass();
      for (const s of this.steps.slice(this.nEnc, head.words ? this.nWords : this.steps.length)) {
        pass.setPipeline(this.pipes[s.kind]); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(...s.grid(nn));
      }
      pass.end();
      if (head.words) {
        kind = "ab"; n = P * this.headD * 4;
        this.readAB ??= (() => { const b = this.device.createBuffer({ size: 2 * this.maxPos * this.headD * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "readback a, b" }); this._owned.push(b); return b; })();
        rd = this.readAB;
        if (n) { enc.copyBufferToBuffer(this.hb.A, 0, rd, 0, n); enc.copyBufferToBuffer(this.hb.B, 0, rd, n, n); }
        n *= 2;
      } else { kind = "scores"; rd = this.readZ; n = P * this.K * this.C * 4; if (n) enc.copyBufferToBuffer(this.hb.Z, 0, rd, 0, n); }
    }
    q.submit([enc.finish()]);
    const t0 = performance.now();
    this._pending = true;
    return {
      read: async () => {
        await this.device.queue.onSubmittedWorkDone();
        const t1 = performance.now();
        let out = new Float32Array(0);
        if (n) {
          await rd.mapAsync(GPUMapMode.READ, 0, n);
          out = new Float32Array(rd.getMappedRange(0, n).slice(0));
          rd.unmap();
        }
        this._pending = false;
        const r = { gpuMs: t1 - t0, readMs: performance.now() - t1 };
        if (kind === "ab") { const h = out.length / 2; r.A = out.subarray(0, h); r.B = out.subarray(h); } else r[kind] = out;
        return r;
      },
    };
  }
}
