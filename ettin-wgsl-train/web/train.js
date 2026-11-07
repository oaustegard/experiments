// LoRA training of a packed Ettin-17m (ModernBERT) decision model in WebGPU compute shaders.
//
// The frozen base is the browser-decider page format (int8 matmuls with a scale per output column, a
// 4-bit token table), dequantized to f32 once on upload. Trainable: a rank-r LoRA on every Wqkv, Wo and
// Wi and Wo_mlp (the MLP output; `targets` picks a subset), and the decision head (Linear, GELU,
// LayerNorm, Linear). One flat parameter vector holds them, in run/reference.py's layout: per layer A then B
// for each target in TARGETS order; then head w0, b0, ln_w, ln_b, w1, b1. `trainer.layout` lists it.
//
// One step: forward through the 7 layers keeping only each layer's input (checkpointing), the head and
// the segment-softmax cross-entropy loss, then for each layer from the top: recompute its forward from
// the saved input, and differentiate it. Attention backward runs in two passes without atomics: per
// query row (dQ and D = dO.O) and per key row (dK, dV), both recomputing the probabilities from the
// saved log-sum-exp. Layer 0's input needs no gradient, so its dX is skipped. Weight gradients that sum
// over every token (LoRA A and B) use a split-K matmul and a reduction. AdamW (PyTorch's update order),
// gradient clipping by global norm, all on the GPU; nothing is read back unless asked.
//
// Buffers are separate GPUBuffers per tensor, so no dispatch binds one buffer as both read and write.
// Uniforms live in one buffer with a 256-byte slot per dispatch (dynamic offsets); bind groups are cached.

import { GELU, REDUCE, SMAX, wgsl, grid } from "../../_lib/wgsl/train/spec.js";
import { kernels } from "../../_lib/wgsl/train/kernels.js";

// The kernels and the spec format live in _lib/wgsl/train/ (kernel library step 3); re-exported for
// pages and bridge jobs that import them from here.
export { GELU, REDUCE, kernels, wgsl, grid };

const SLOT = 256;           // uniform slot bytes (minUniformBufferOffsetAlignment)
const MAX_SLOTS = 2048;
const KCHUNK = 256;         // rows per split-K slice
export const TARGETS = ["Wqkv", "Wo", "Wi", "Wo_mlp"];   // PyTorch's recipes match child name "Wo", so attn.Wo and mlp.Wo

const f16 = (h) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
};

export class Trainer {
  // opts: { manifest, weights (ArrayBuffer | Uint8Array of weights.bin), maxRows, maxSeqs, maxPool, maxSeg, rank, alpha }
  static async create(device, opts) {
    const t = new Trainer(device, opts);
    await t.#init();
    return t;
  }

  constructor(device, { manifest, weights, maxRows, maxSeqs = 64, maxPool = 256, maxSeg = 128, rank = 8, alpha = 16, targets = TARGETS }) {
    this.device = device;
    this.man = manifest;
    this.wbytes = weights instanceof Uint8Array ? weights : new Uint8Array(weights);
    const H = manifest.hidden, NH = manifest.heads;
    this.C = { H, NH, D: H / NH, I: manifest.inter, L: manifest.layers.length, r: rank, s: alpha / rank };
    if (targets.some((t) => !TARGETS.includes(t))) throw new Error(`targets must be among ${TARGETS}`);
    this.targets = TARGETS.filter((t) => targets.includes(t));
    if (H !== 256 || this.C.D !== 64) throw new Error(`kernels assume hidden 256, head dim 64 (got ${H}, ${this.C.D})`);
    this.max = { R: maxRows, B: maxSeqs, P: maxPool, G: maxSeg };
    this.ops = [];
    this.bgCache = new Map();
    this.nid = 0;
  }

  async #init() {
    const dev = this.device, { H, I, L, r, NH, D } = this.C, m = this.man, R = this.max.R;
    const K = kernels(this.C);
    this.K = {};
    await Promise.all(Object.entries(K).map(async ([name, k]) => {
      const code = wgsl(name, k);
      const mod = dev.createShaderModule({ code, label: name });
      const entries = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
        ...k.b.map(([, mode], i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: mode === "rw" ? "storage" : "read-only-storage" } }))];
      const bgl = dev.createBindGroupLayout({ entries, label: name });
      let pipe;
      try {
        pipe = await dev.createComputePipelineAsync({ layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: mod, entryPoint: "main" }, label: name });
      } catch (e) {
        const info = await mod.getCompilationInfo();
        throw new Error(`${name}: ${e.message}\n${info.messages.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("\n")}`);
      }
      this.K[name] = { pipe, bgl, u: k.u.map((s) => s.split(":")), nb: k.b.length };
    }));
    this.ubo = dev.createBuffer({ size: SLOT * MAX_SLOTS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "uniforms" });
    this.udata = new ArrayBuffer(SLOT * MAX_SLOTS);
    this.uview = new DataView(this.udata);

    // frozen weights, dequantized: per layer attn_norm, Wqkv [3H,H], Wo [H,H], mlp_norm, Wi [2I,H], Wo_mlp [H,I] (PyTorch [out, in])
    const tf = (rec) => new Float32Array(this.wbytes.buffer.slice(this.wbytes.byteOffset + rec.offset, this.wbytes.byteOffset + rec.offset + rec.length));
    const tu8 = (rec) => this.wbytes.slice(rec.offset, rec.offset + rec.length);
    const lin = (w) => {   // file: q [K, N] int8, s [N] -> f32 [N, K]
      const q = new Int8Array(tu8(w.q).buffer), s = tf(w.s), out = new Float32Array(w.N * w.K);
      for (let k = 0; k < w.K; k++) for (let n = 0; n < w.N; n++) out[n * w.K + k] = Math.fround(q[k * w.N + n] * s[n]);
      return out;
    };
    const parts = [], off = {};
    let n = 0;
    const put = (name, arr) => { off[name] = n; parts.push(arr); n += arr.length; };
    put("emb_norm", tf(m.emb_norm));
    put("final_norm", tf(m.final_norm));
    m.layers.forEach((Ly, i) => {
      if (Ly.attn_norm) put(`L${i}.attn_norm`, tf(Ly.attn_norm));
      put(`L${i}.Wqkv`, lin(Ly.Wqkv));
      put(`L${i}.Wo`, lin(Ly.Wo));
      put(`L${i}.mlp_norm`, tf(Ly.mlp_norm));
      put(`L${i}.Wi`, lin(Ly.Wi));
      put(`L${i}.Wo_mlp`, lin(Ly.Wo_mlp));
    });
    const wb = new Float32Array(n);
    let o = 0;
    for (const a of parts) { wb.set(a, o); o += a.length; }
    this.woff = off;
    this.WB = this.#buf("WB", wb.length, wb);
    const T = m.table;
    const codes = tu8(T.codes);
    this.CODES = this.#buf("CODES", codes.length / 4, new Uint32Array(codes.buffer, codes.byteOffset, codes.length / 4));
    const sh = new Uint16Array(tu8(T.scales).buffer), sc = new Float32Array(sh.length);
    for (let i = 0; i < sh.length; i++) sc[i] = f16(sh[i]);
    this.SCALES = this.#buf("SCALES", sc.length, sc);
    // RoPE cos / sin [SMAX, D/2], computed the way PyTorch does it in float32
    const half = D / 2, cs = new Float32Array(2 * SMAX * half), fr = Math.fround;
    const rope = (theta) => {
      for (let j = 0; j < half; j++) {
        const inv = fr(1 / fr(theta ** fr((2 * j) / D)));
        for (let p = 0; p < SMAX; p++) { const a = fr(p * inv); cs[p * half + j] = Math.cos(a); cs[SMAX * half + p * half + j] = Math.sin(a); }
      }
    };
    if (m.theta_global !== m.theta_local) throw new Error("separate local / global RoPE tables not implemented");
    rope(m.theta_global);
    this.CS = this.#buf("CS", cs.length, cs);

    // trainable parameters: per layer [Wqkv A, B, Wo A, B, Wi A, B], then head
    const tg = [["Wqkv", 3 * H, H], ["Wo", H, H], ["Wi", 2 * I, H], ["Wo_mlp", H, I]].filter(([t]) => this.targets.includes(t));
    const lay = [];
    let pn = 0;
    const add = (name, size, shape) => { lay.push({ name, offset: pn, shape }); pn += size; };
    for (let i = 0; i < L; i++) for (const [t, N, Kin] of tg) { add(`L${i}.${t}.A`, r * Kin, [r, Kin]); add(`L${i}.${t}.B`, N * r, [N, r]); }
    add("head.w0", H * H, [H, H]); add("head.b0", H, [H]); add("head.ln_w", H, [H]); add("head.ln_b", H, [H]); add("head.w1", H, [1, H]); add("head.b1", 1, [1]);
    this.layout = lay;
    this.poff = Object.fromEntries(lay.map((x) => [x.name, x.offset]));
    this.nParams = pn;
    this.headOff = this.poff["head.w0"];
    for (const nm of ["PRM", "GRD", "AM", "AV"]) this[nm] = this.#buf(nm, pn);
    this.NPART = 64;
    this.PARTN = this.#buf("PARTN", this.NPART);
    this.NRM = this.#buf("NRM", 4);

    // activations (sized for maxRows tokens) and gradients
    const P = this.max.P;
    this.XS = Array.from({ length: L + 1 }, (_, i) => this.#buf(`x${i}`, R * H));
    const A = (name, size) => { this[name] = this.#buf(name, size); };
    A("Hn", R * H); A("ST1", 2 * R); A("QKV", R * 3 * H); A("QR", R * 2 * H); A("O", R * H); A("LSE", R * NH);
    A("XM", R * H); A("H2", R * H); A("ST2", 2 * R); A("U", R * 2 * I); A("G", R * I);
    A("T1", R * r); A("T2", R * r); A("T3", R * r); A("T4", R * r);
    A("DXA", R * H); A("DXB", R * H); A("DG", R * I); A("DU", R * 2 * I); A("DT", R * r); A("DH2", R * H); A("DO", R * H);
    A("DH", R * H); A("DXM", R * H); A("DQR", R * 2 * H); A("DQKV", R * 3 * H); A("DD", R * NH);
    this.ZMAX = Math.ceil(R / KCHUNK);
    A("PART", this.ZMAX * 3 * H * r);
    A("HP", P * H); A("STF", 2 * P); A("HF", P * H); A("Z1", P * H); A("A1", P * H); A("STH", 2 * P); A("N1", P * H);
    A("LOGIT", P); A("DL", P); A("DN1", P * H); A("DA1", P * H); A("DZ1", P * H); A("DHF", P * H); A("DHP", P * H);
    this.IDS = this.#buf("IDS", R, null, "u32");
    this.LENS = this.#buf("LENS", this.max.B, null, "u32");
    this.POS = this.#buf("POS", P, null, "u32");
    this.SEG = this.#buf("SEG", this.max.G, null, "u32");
    this.GOLD = this.#buf("GOLD", P);
    this.LOSS = this.#buf("LOSS", 4096);
  }

  #buf(label, n, init = null) {
    const size = Math.max(16, Math.ceil((n * 4) / 16) * 16);
    const b = this.device.createBuffer({ size, label, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: !!init });
    if (init) {
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(init.buffer, init.byteOffset, init.byteLength));
      b.unmap();
    }
    b._id = this.nid++;
    b._n = n;
    return b;
  }

  // record one dispatch
  op(name, bufs, vals, groups) {
    const k = this.K[name];
    if (!k) throw new Error(`no kernel ${name}`);
    if (bufs.length !== k.nb || bufs.some((b) => !b)) throw new Error(`${name}: expected ${k.nb} buffers`);
    this.ops.push({ k, name, bufs, vals, groups: Array.isArray(groups) ? groups : grid(groups) });
  }

  clear(buf) { this.ops.push({ clear: buf }); }

  // submit the recorded dispatches; wait = await the GPU (queue order already keeps later uploads behind it)
  async flush(wait = true) {
    if (!this.ops.length) { if (wait) await this.device.queue.onSubmittedWorkDone(); return; }
    const dev = this.device;
    let slots = 0;
    for (const o of this.ops) {
      if (o.clear) continue;
      if (slots >= MAX_SLOTS) throw new Error("too many dispatches in one flush");
      o.slot = slots++;
      o.k.u.forEach(([n, t], i) => {
        const v = o.vals[n] ?? 0;
        if (t === "f") this.uview.setFloat32(o.slot * SLOT + 4 * i, v, true);
        else this.uview.setUint32(o.slot * SLOT + 4 * i, v, true);
      });
    }
    dev.queue.writeBuffer(this.ubo, 0, this.udata, 0, slots * SLOT);
    const enc = dev.createCommandEncoder();
    let pass = null;
    for (const o of this.ops) {
      if (o.clear) {
        if (pass) { pass.end(); pass = null; }
        enc.clearBuffer(o.clear);
        continue;
      }
      if (!pass) pass = enc.beginComputePass();
      const key = o.name + "|" + o.bufs.map((b) => b._id).join(",");
      let bg = this.bgCache.get(key);
      if (!bg) {
        bg = dev.createBindGroup({ layout: o.k.bgl, entries: [{ binding: 0, resource: { buffer: this.ubo, offset: 0, size: SLOT } },
          ...o.bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))] });
        this.bgCache.set(key, bg);
      }
      pass.setPipeline(o.k.pipe);
      pass.setBindGroup(0, bg, [o.slot * SLOT]);
      pass.dispatchWorkgroups(...o.groups);
    }
    if (pass) pass.end();
    dev.queue.submit([enc.finish()]);
    this.ops = [];
    if (wait) await dev.queue.onSubmittedWorkDone();
  }

  async read(buf, n = buf._n, type = Float32Array) {
    await this.flush();
    const bytes = Math.ceil((n * 4) / 4) * 4;
    const st = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, st, 0, Math.ceil(bytes / 4) * 4);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new type(st.getMappedRange(0, bytes).slice(0));
    st.unmap();
    st.destroy();
    return out;
  }

  setParams(flat) {
    if (flat.length !== this.nParams) throw new Error(`params: ${flat.length} != ${this.nParams}`);
    this.device.queue.writeBuffer(this.PRM, 0, flat);
    this.clear(this.AM);
    this.clear(this.AV);
  }

  // batch: { ids Int32Array [B*S], B, S, lens [B], pos [nP] (flat row b*S+t), seg [nSeg], gold [nP] }
  loadBatch(bt) {
    const q = this.device.queue;
    if (bt.B * bt.S > this.max.R || bt.B > this.max.B || bt.pos.length > this.max.P || bt.seg.length > this.max.G || bt.S > SMAX)
      throw new Error(`batch ${bt.B}x${bt.S} (${bt.pos.length} options) exceeds the trainer's sizes`);
    q.writeBuffer(this.IDS, 0, Uint32Array.from(bt.ids));
    q.writeBuffer(this.LENS, 0, Uint32Array.from(bt.lens));
    q.writeBuffer(this.POS, 0, Uint32Array.from(bt.pos));
    q.writeBuffer(this.SEG, 0, Uint32Array.from(bt.seg));
    if (bt.gold) q.writeBuffer(this.GOLD, 0, Float32Array.from(bt.gold));
    this.b = { R: bt.B * bt.S, B: bt.B, S: bt.S, nP: bt.pos.length, nSeg: bt.seg.length };
  }

  // C (=|+=) alpha * X W^T for a frozen W [N, K] at WB offset (the forward linear)
  #lin(X, wname, N, Kin, Y, R, acc = 0) {
    this.op("mm", [X, this.WB, Y], { M: R, N, K: Kin, kChunk: Kin, aOff: 0, aRS: Kin, aCS: 1, bOff: this.woff[wname], bRS: 1, bCS: Kin,
      cOff: 0, cRS: N, cCS: 1, acc, alpha: 1 }, [Math.ceil(N / 64), Math.ceil(R / 64), 1]);
  }

  // dX (=|+=) dY W for a frozen W [N, K]
  #linT(DY, wname, N, Kin, DX, R, acc = 0) {
    this.op("mm", [DY, this.WB, DX], { M: R, N: Kin, K: N, kChunk: N, aOff: 0, aRS: N, aCS: 1, bOff: this.woff[wname], bRS: Kin, bCS: 1,
      cOff: 0, cRS: Kin, cCS: 1, acc, alpha: 1 }, [Math.ceil(Kin / 64), Math.ceil(R / 64), 1]);
  }

  // LoRA forward: T = X A^T, Y += s T B^T
  #has(pfx) { return this.targets.includes(pfx.split(".")[1]); }

  #loraF(X, pfx, N, Kin, T, Y, R, tOnly = false) {
    if (!this.#has(pfx)) return;
    const { r, s } = this.C;
    this.op("mm", [X, this.PRM, T], { M: R, N: r, K: Kin, kChunk: Kin, aRS: Kin, aCS: 1, bOff: this.poff[pfx + ".A"], bRS: 1, bCS: Kin, cRS: r, cCS: 1, alpha: 1 },
      [1, Math.ceil(R / 64), 1]);
    if (tOnly) return;
    this.op("mm", [T, this.PRM, Y], { M: R, N, K: r, kChunk: r, aRS: r, aCS: 1, bOff: this.poff[pfx + ".B"], bRS: 1, bCS: r, cRS: N, cCS: 1, acc: 1, alpha: s },
      [Math.ceil(N / 64), Math.ceil(R / 64), 1]);
  }

  // split-K product into GRD: G[name] (layout [M, N], row-major) = alpha * op(A) op(B), K = rows R
  #gradK(A, B, name, M, N, R, a, b, alpha) {
    const Z = Math.ceil(R / KCHUNK);
    this.op("mm", [A, B, this.PART], { M, N, K: R, kChunk: KCHUNK, ...a, ...b, cOff: 0, cRS: N, cCS: 1, cZS: M * N, alpha: 1 },
      [Math.ceil(N / 64), Math.ceil(M / 64), Z]);
    this.op("sumz", [this.PART, this.GRD], { M, N, Z, cOff: this.poff[name], cRS: N, cCS: 1, alpha }, Math.ceil((M * N) / 256));
  }

  // LoRA backward. DY [R, N] is the gradient of the LoRA'd output, X [R, K] its input, T = X A^T.
  // Writes dA, dB into GRD; DT = s DY B; DX += DT A (when DX given).
  #loraB(DY, X, T, pfx, N, Kin, R, DX) {
    if (!this.#has(pfx)) return;
    const { r, s } = this.C;
    this.op("mm", [DY, this.PRM, this.DT], { M: R, N: r, K: N, kChunk: N, aRS: N, aCS: 1, bOff: this.poff[pfx + ".B"], bRS: r, bCS: 1, cRS: r, cCS: 1, alpha: s },
      [1, Math.ceil(R / 64), 1]);
    // dB [N, r] = s DY^T T
    this.#gradK(DY, T, pfx + ".B", N, r, R, { aOff: 0, aRS: 1, aCS: N }, { bOff: 0, bRS: r, bCS: 1 }, s);
    // dA [r, K] = DT^T X
    this.#gradK(this.DT, X, pfx + ".A", r, Kin, R, { aOff: 0, aRS: 1, aCS: r }, { bOff: 0, bRS: Kin, bCS: 1 }, 1);
    if (DX) this.op("mm", [this.DT, this.PRM, DX], { M: R, N: Kin, K: r, kChunk: r, aRS: r, aCS: 1, bOff: this.poff[pfx + ".A"], bRS: Kin, bCS: 1, cRS: Kin, cCS: 1, acc: 1, alpha: 1 },
      [Math.ceil(Kin / 64), Math.ceil(R / 64), 1]);
  }

  // forward of layer i from XS[i]; writes XS[i+1] unless recompute
  #layer(i, recompute = false) {
    const { H, I } = this.C, { R, S } = this.b, Ly = this.man.layers[i], X = this.XS[i], eps = this.man.eps;
    let h = X;
    if (Ly.attn_norm) { this.op("ln", [X, this.WB, this.Hn, this.ST1], { R, eps, wOff: this.woff[`L${i}.attn_norm`] }, R); h = this.Hn; }
    this.#lin(h, `L${i}.Wqkv`, 3 * H, H, this.QKV, R);
    this.#loraF(h, `L${i}.Wqkv`, 3 * H, H, this.T1, this.QKV, R);
    this.op("rope", [this.QKV, this.CS, this.QR], { R, S, sinOff: SMAX * (this.C.D / 2) }, R);
    const att = { R, S, isLocal: Ly.global ? 0 : 1, win: this.man.window >> 1 };
    this.op("attn", [this.QR, this.QKV, this.LENS, this.O, this.LSE], att, R * this.C.NH);
    this.op("copy", [X, this.XM], { n: R * H }, Math.ceil((R * H) / 256));
    this.#lin(this.O, `L${i}.Wo`, H, H, this.XM, R, 1);
    this.#loraF(this.O, `L${i}.Wo`, H, H, this.T2, this.XM, R);
    this.op("ln", [this.XM, this.WB, this.H2, this.ST2], { R, eps, wOff: this.woff[`L${i}.mlp_norm`] }, R);
    this.#lin(this.H2, `L${i}.Wi`, 2 * I, H, this.U, R);
    this.#loraF(this.H2, `L${i}.Wi`, 2 * I, H, this.T3, this.U, R);
    this.op("geglu", [this.U, this.G], { n: R * I }, Math.ceil((R * I) / 256));
    if (recompute) { this.#loraF(this.G, `L${i}.Wo_mlp`, H, I, this.T4, null, R, true); return { h, att }; }
    this.op("copy", [this.XM, this.XS[i + 1]], { n: R * H }, Math.ceil((R * H) / 256));
    this.#lin(this.G, `L${i}.Wo_mlp`, H, I, this.XS[i + 1], R, 1);
    this.#loraF(this.G, `L${i}.Wo_mlp`, H, I, this.T4, this.XS[i + 1], R);
    return { h, att };
  }

  #headForward() {
    const { H } = this.C, { nP } = this.b, eps = this.man.eps, L = this.C.L;
    this.op("gather", [this.POS, this.XS[L], this.HP], { R: nP }, nP);
    this.op("ln", [this.HP, this.WB, this.HF, this.STF], { R: nP, eps, wOff: this.woff.final_norm }, nP);
    this.op("mm", [this.HF, this.PRM, this.Z1], { M: nP, N: H, K: H, kChunk: H, aRS: H, aCS: 1, bOff: this.poff["head.w0"], bRS: 1, bCS: H, cRS: H, cCS: 1, alpha: 1 },
      [Math.ceil(H / 64), Math.ceil(nP / 64), 1]);
    this.op("bias_gelu", [this.PRM, this.Z1, this.A1], { R: nP, bOff: this.poff["head.b0"] }, nP);
    this.op("ln", [this.A1, this.PRM, this.N1, this.STH], { R: nP, eps: 1e-5, wOff: this.poff["head.ln_w"], bOff: this.poff["head.ln_b"], hasB: 1 }, nP);
    this.op("head_out", [this.N1, this.PRM, this.LOGIT], { R: nP, wOff: this.poff["head.w1"], bOff: this.poff["head.b1"] }, nP);
  }

  forward() {
    const { R } = this.b;
    this.op("embed", [this.IDS, this.CODES, this.SCALES, this.WB, this.XS[0]], { R, eps: this.man.eps, wOff: this.woff.emb_norm }, R);
    for (let i = 0; i < this.C.L; i++) this.#layer(i);
    this.#headForward();
  }

  // forward, loss (written to LOSS[slot]) and every gradient into GRD
  forwardBackward({ T, slot = 0 }) {
    const { H, I, L, NH } = this.C, { R, S, nP, nSeg } = this.b;
    this.forward();
    this.op("loss", [this.LOGIT, this.SEG, this.GOLD, this.DL, this.LOSS], { nSeg, T, slot }, 1);
    // head backward
    this.op("head_out_bwd", [this.DL, this.N1, this.PRM, this.DN1, this.GRD], { R: nP, wOff: this.poff["head.w1"], gwOff: this.poff["head.w1"], gbOff: this.poff["head.b1"] }, 1);
    this.op("ln_bwd", [this.A1, this.STH, this.PRM, this.DN1, this.DA1], { R: nP, wOff: this.poff["head.ln_w"] }, nP);
    this.op("ln_pgrad", [this.A1, this.STH, this.DN1, this.GRD], { R: nP, gwOff: this.poff["head.ln_w"], gbOff: this.poff["head.ln_b"] }, 1);
    this.op("gelu_bwd", [this.Z1, this.DA1, this.DZ1], { n: nP * H }, Math.ceil((nP * H) / 256));
    this.op("mm", [this.DZ1, this.HF, this.GRD], { M: H, N: H, K: nP, kChunk: nP, aRS: 1, aCS: H, bRS: H, bCS: 1, cOff: this.poff["head.w0"], cRS: H, cCS: 1, alpha: 1 },
      [Math.ceil(H / 64), Math.ceil(H / 64), 1]);
    this.op("colsum", [this.DZ1, this.GRD], { R: nP, gOff: this.poff["head.b0"] }, 1);
    this.op("mm", [this.DZ1, this.PRM, this.DHF], { M: nP, N: H, K: H, kChunk: H, aRS: H, aCS: 1, bOff: this.poff["head.w0"], bRS: H, bCS: 1, cRS: H, cCS: 1, alpha: 1 },
      [Math.ceil(H / 64), Math.ceil(nP / 64), 1]);
    this.op("ln_bwd", [this.HP, this.STF, this.WB, this.DHF, this.DHP], { R: nP, wOff: this.woff.final_norm }, nP);
    let dxo = this.DXA, dxi = this.DXB;
    this.clear(dxo);
    this.op("scatter", [this.POS, this.DHP, dxo], { R: nP }, nP);
    // layers, top down
    for (let i = L - 1; i >= 0; i--) {
      const Ly = this.man.layers[i], X = this.XS[i], pf = `L${i}.`;
      const { h, att } = this.#layer(i, true);
      // MLP
      this.#linT(dxo, pf + "Wo_mlp", H, I, this.DG, R);
      this.#loraB(dxo, this.G, this.T4, pf + "Wo_mlp", H, I, R, this.DG);
      this.op("geglu_bwd", [this.U, this.DG, this.DU], { n: R * I }, Math.ceil((R * I) / 256));
      this.#linT(this.DU, pf + "Wi", 2 * I, H, this.DH2, R);
      this.#loraB(this.DU, this.H2, this.T3, pf + "Wi", 2 * I, H, R, this.DH2);
      this.op("copy", [dxo, this.DXM], { n: R * H }, Math.ceil((R * H) / 256));
      this.op("ln_bwd", [this.XM, this.ST2, this.WB, this.DH2, this.DXM], { R, wOff: this.woff[pf + "mlp_norm"], acc: 1 }, R);
      // attention output projection
      this.#linT(this.DXM, pf + "Wo", H, H, this.DO, R);
      this.#loraB(this.DXM, this.O, this.T2, pf + "Wo", H, H, R, this.DO);
      this.op("attn_bq", [this.QR, this.QKV, this.LENS, this.LSE, this.DO, this.O, this.DQR, this.DD], att, R * NH);
      this.op("attn_bkv", [this.QR, this.QKV, this.LENS, this.LSE, this.DO, this.DD, this.DQR, this.DQKV], att, R * NH);
      this.op("rope_bwd", [this.DQR, this.CS, this.DQKV], { R, S, sinOff: SMAX * (this.C.D / 2) }, R);
      const needDX = i > 0;
      if (needDX) this.#linT(this.DQKV, pf + "Wqkv", 3 * H, H, this.DH, R);
      this.#loraB(this.DQKV, h, this.T1, pf + "Wqkv", 3 * H, H, R, needDX ? this.DH : null);
      if (needDX) {
        this.op("copy", [this.DXM, dxi], { n: R * H }, Math.ceil((R * H) / 256));
        if (Ly.attn_norm) this.op("ln_bwd", [X, this.ST1, this.WB, this.DH, dxi], { R, wOff: this.woff[pf + "attn_norm"], acc: 1 }, R);
        else this.op("copy", [this.DH, dxi], { n: R * H, acc: 1 }, Math.ceil((R * H) / 256));
        [dxo, dxi] = [dxi, dxo];
      }
    }
  }

  // AdamW with global-norm clipping, PyTorch's update order. step counts from 1.
  adamStep({ lr, lrHead = lr, wd = 0.01, step, clip = 1.0, b1 = 0.9, b2 = 0.999, eps = 1e-8 }) {
    const n = this.nParams;
    this.op("sqsum", [this.GRD, this.PARTN], { n }, this.NPART);
    this.op("norm", [this.PARTN, this.NRM], { n: this.NPART }, 1);
    this.op("adamw", [this.PRM, this.GRD, this.AM, this.AV, this.NRM], { n, headOff: this.headOff, lr, lrHead, wd, b1, b2, eps,
      bc1: 1 - b1 ** step, bc2s: Math.sqrt(1 - b2 ** step), clip }, Math.ceil(n / 256));
  }
}
