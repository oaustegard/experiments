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
// run/gref.py (GRefBi.reps -> browser-decider's Ref.hidden) is the NumPy spec.

import { SeqEngine, HEADER, BYTE, split } from "../../_lib/wgsl/runtime.js";
import { KERNELS, attentionLong, embed, gated, matmul, norm, rope, ropeTable } from "../../_lib/wgsl/kernels.js";

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
  for (var j = l; j < H; j += 64u) { Y[r * H + j] = X[src * H + j]; }
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
  }

  async _build(bytes) {
    const m = this.m, T = this.maxTokens;
    const H = m.hidden, nh = m.heads, D = H / nh, I = m.inter;
    if (m.arch !== "modernbert") throw new Error(`text encoder arch ${m.arch}`);
    if (H % 64 || D % 4 || D > 128) throw new Error(`hidden ${H} / head dim ${D} not supported by the kernels`);
    if (![4, 32].includes(m.table.bits)) throw new Error(`token table bits ${m.table.bits} (engine runs 4 and 32)`);
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
  }

  // Encode a batch (arrays of token ids, each at most MAX_SEQ) and gather the final hidden rows at
  // idx (indexes into the packed token axis: sequence offset + position). Returns { read() } whose
  // promise gives { rows, gpuMs, readMs }. One readback buffer: read() a job before submitting the next.
  submit(batch, idx) {
    const H = this.dim, P = idx.length;
    if (P > this.maxPos) throw new Error(`${P} rows exceed maxPos ${this.maxPos}`);
    if (this._pending) throw new Error("read() the previous batch before submitting another");
    this._nPos = Math.max(P, 1);
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 2).set([H, P]);
    this.device.queue.writeBuffer(this.gatherU, 0, u);
    if (P) this.device.queue.writeBuffer(this.b.idx, 0, idx);
    const enc = this.device.createCommandEncoder();
    this._record(enc, batch);
    const n = P * H * 4;
    if (n) enc.copyBufferToBuffer(this.b.out, 0, this.readPos, 0, n);
    this.device.queue.submit([enc.finish()]);
    const t0 = performance.now();
    this._pending = true;
    return {
      read: async () => {
        await this.device.queue.onSubmittedWorkDone();
        const t1 = performance.now();
        let rows = new Float32Array(0);
        if (n) {
          await this.readPos.mapAsync(GPUMapMode.READ, 0, n);
          rows = new Float32Array(this.readPos.getMappedRange(0, n).slice(0));
          this.readPos.unmap();
        }
        this._pending = false;
        return { rows, gpuMs: t1 - t0, readMs: performance.now() - t1 };
      },
    };
  }
}
