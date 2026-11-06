// Full fine-tune of an Ettin-17m embedder in WebGPU compute shaders, against a teacher's vectors.
//
// Every weight trains, the token table included: table [V, H] -> LayerNorm (no bias) -> 7 ModernBERT
// layers -> final LayerNorm -> mean pool over real tokens -> Dense 256 -> 768 (no bias). Loss: mean over
// the batch of 1 - cos(student, teacher), as ettin-native's distill.py. The flat f32 parameter vector is
// run/model.py's layout(): table, emb_norm, per layer [attn_norm (not in layer 0), Wqkv, Wo, mlp_norm,
// Wi, Wo_mlp], final_norm, dense.
//
// Layer forward and backward kernels (attention in two passes without atomics, GeGLU, RoPE, the tiled
// matmul, AdamW) are ettin-wgsl-train's. New here: weight gradients for every linear (split-K matmul over
// the batch's rows), LayerNorm weight gradients (per-chunk partial sums, then a reduction), the token
// table gradient (the batch's unique tokens and the rows holding each, built on the CPU, one workgroup per
// unique token: no atomics), mean pooling and the cosine loss.

import { kernels as baseKernels, wgsl, grid, REDUCE } from "../../ettin-wgsl-train/web/train.js";

const SMAX = 1024, SLOT = 256, MAX_SLOTS = 4096, KCHUNK = 512, LNCH = 256;

function kernels(C) {
  const { H, OUT } = C;
  const K = baseKernels(C);
  K.embed_f = { u: ["R", "eps:f", "tOff", "wOff"], b: [["IDS", "r", "u32"], ["PRM", "r", "f32"], ["XE", "rw", "f32"], ["X", "rw", "f32"], ["ST", "rw", "f32"]],
    wg: H, pre: REDUCE(H), body: `
  if (gi >= p.R) { return; }
  let j = lid.x;
  let v = PRM[p.tOff + IDS[gi] * ${H}u + j];
  XE[gi * ${H}u + j] = v;
  let mu = wsum(v, j) / ${H}.0;
  let d = v - mu;
  let rs = inverseSqrt(wsum(d * d, j) / ${H}.0 + p.eps);
  X[gi * ${H}u + j] = d * rs * PRM[p.wOff + j];
  if (j == 0u) { ST[2u * gi] = mu; ST[2u * gi + 1u] = rs; }` };
  // LayerNorm weight gradient, partial sums over chunks of rows: PART[z, c] = sum_r DY[r, c] * xhat[r, c]
  K.ln_wpart = { u: ["R", "chunk"], b: [["X", "r", "f32"], ["ST", "r", "f32"], ["DY", "r", "f32"], ["PART", "rw", "f32"]], wg: H, body: `
  let c = lid.x;
  let r0 = gi * p.chunk;
  let r1 = min(p.R, r0 + p.chunk);
  var s = 0.0;
  for (var r = r0; r < r1; r = r + 1u) { s = s + DY[r * ${H}u + c] * (X[r * ${H}u + c] - ST[2u * r]) * ST[2u * r + 1u]; }
  PART[gi * ${H}u + c] = s;` };
  K.pool = { u: ["B", "S"], b: [["X", "r", "f32"], ["LENS", "r", "u32"], ["PL", "rw", "f32"]], wg: H, body: `
  if (gi >= p.B) { return; }
  let n = LENS[gi];
  var s = 0.0;
  for (var t = 0u; t < n; t = t + 1u) { s = s + X[(gi * p.S + t) * ${H}u + lid.x]; }
  PL[gi * ${H}u + lid.x] = s / f32(n);` };
  K.unpool = { u: ["R", "S"], b: [["DPL", "r", "f32"], ["LENS", "r", "u32"], ["DX", "rw", "f32"]], wg: H, body: `
  if (gi >= p.R) { return; }
  let b = gi / p.S;
  let t = gi % p.S;
  let n = LENS[b];
  var v = 0.0;
  if (t < n) { v = DPL[b * ${H}u + lid.x] / f32(n); }
  DX[gi * ${H}u + lid.x] = v;` };
  // loss = mean_b (1 - cos(y_b, t_b)); one workgroup walks the batch
  K.cosloss = { u: ["B", "slot"], b: [["Y", "r", "f32"], ["TGT", "r", "f32"], ["DY", "rw", "f32"], ["LOSS", "rw", "f32"]], wg: 256, pre: REDUCE(256), body: `
  let t = lid.x;
  var tot = 0.0;
  for (var b = 0u; b < p.B; b = b + 1u) {
    var yt = 0.0;
    var yy = 0.0;
    var tt = 0.0;
    for (var j = t; j < ${OUT}u; j = j + 256u) { let y = Y[b * ${OUT}u + j]; let g = TGT[b * ${OUT}u + j]; yt = yt + y * g; yy = yy + y * y; tt = tt + g * g; }
    yt = wsum(yt, t);
    yy = wsum(yy, t);
    tt = wsum(tt, t);
    let ny = sqrt(yy);
    let nt = sqrt(tt);
    let cs = yt / (ny * nt);
    tot = tot + 1.0 - cs;
    let k = -1.0 / f32(p.B);
    for (var j = t; j < ${OUT}u; j = j + 256u) {
      let y = Y[b * ${OUT}u + j];
      DY[b * ${OUT}u + j] = k * (TGT[b * ${OUT}u + j] / (ny * nt) - cs * y / yy);
    }
  }
  if (t == 0u) { LOSS[p.slot] = tot / f32(p.B); }` };
  // token-table gradient: one workgroup per unique token u, summing the rows that hold it
  K.tgrad = { u: ["nU", "tOff"], b: [["UTOK", "r", "u32"], ["USTART", "r", "u32"], ["UROWS", "r", "u32"], ["DXE", "r", "f32"], ["G", "rw", "f32"]], wg: H, body: `
  if (gi >= p.nU) { return; }
  var s = 0.0;
  for (var k = USTART[gi]; k < USTART[gi + 1u]; k = k + 1u) { s = s + DXE[UROWS[k] * ${H}u + lid.x]; }
  G[p.tOff + UTOK[gi] * ${H}u + lid.x] = s;` };
  return K;
}

export function layoutFor(V, { H = 256, I = 384, L = 7, OUT = 768 } = {}) {
  const lay = [];
  let n = 0;
  const add = (name, shape) => { lay.push({ name, shape, offset: n }); n += shape.reduce((a, b) => a * b, 1); };
  add("table", [V, H]); add("emb_norm", [H]);
  for (let i = 0; i < L; i++) {
    if (i > 0) add(`L${i}.attn_norm`, [H]);
    add(`L${i}.Wqkv`, [3 * H, H]); add(`L${i}.Wo`, [H, H]); add(`L${i}.mlp_norm`, [H]); add(`L${i}.Wi`, [2 * I, H]); add(`L${i}.Wo_mlp`, [H, I]);
  }
  add("final_norm", [H]); add("dense", [OUT, H]);
  return { layout: lay, n };
}

export class DistillTrainer {
  // opts: { V, maxRows, maxSeqs, window (local attention, 128), theta, eps }
  static async create(device, opts) {
    const t = new DistillTrainer(device, opts);
    await t.#init();
    return t;
  }

  constructor(device, { V, maxRows, maxSeqs = 128, window = 128, theta = 160000, eps = 1e-5, H = 256, NH = 4, I = 384, L = 7, OUT = 768 }) {
    this.device = device;
    this.C = { H, NH, D: H / NH, I, L, OUT, V, r: 8 };
    this.cfg = { window, theta, eps, global: Array.from({ length: L }, (_, i) => i % 3 === 0) };
    this.max = { R: maxRows, B: maxSeqs };
    this.ops = [];
    this.bgCache = new Map();
    this.nid = 0;
    const { layout, n } = layoutFor(V, { H, I, L, OUT });
    this.layout = layout;
    this.poff = Object.fromEntries(layout.map((x) => [x.name, x.offset]));
    this.nParams = n;
  }

  async #init() {
    const dev = this.device, { H, I, L, NH, D, OUT } = this.C, R = this.max.R, B = this.max.B;
    const K = kernels(this.C);
    this.K = {};
    await Promise.all(Object.entries(K).map(async ([name, k]) => {
      const mod = dev.createShaderModule({ code: wgsl(name, k), label: name });
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
    const half = D / 2, cs = new Float32Array(2 * SMAX * half), fr = Math.fround;
    for (let j = 0; j < half; j++) {
      const inv = fr(1 / fr(this.cfg.theta ** fr((2 * j) / D)));
      for (let p = 0; p < SMAX; p++) { const a = fr(p * inv); cs[p * half + j] = Math.cos(a); cs[SMAX * half + p * half + j] = Math.sin(a); }
    }
    this.CS = this.#buf("CS", cs.length, cs);
    for (const nm of ["PRM", "GRD", "AM", "AV"]) this[nm] = this.#buf(nm, this.nParams);
    this.NPART = 256;
    this.PARTN = this.#buf("PARTN", this.NPART);
    this.NRM = this.#buf("NRM", 4);
    this.XS = Array.from({ length: L + 1 }, (_, i) => this.#buf(`x${i}`, R * H));
    const A = (name, size) => { this[name] = this.#buf(name, size); };
    A("XE", R * H); A("ST0", 2 * R); A("DXE", R * H);
    A("Hn", R * H); A("ST1", 2 * R); A("QKV", R * 3 * H); A("QR", R * 2 * H); A("O", R * H); A("LSE", R * NH);
    A("XM", R * H); A("H2", R * H); A("ST2", 2 * R); A("U", R * 2 * I); A("G", R * I);
    A("DXA", R * H); A("DXB", R * H); A("DG", R * I); A("DU", R * 2 * I); A("DH2", R * H); A("DO", R * H);
    A("DH", R * H); A("DXM", R * H); A("DQR", R * 2 * H); A("DQKV", R * 3 * H); A("DD", R * NH);
    A("HF", R * H); A("STF", 2 * R); A("DHF", R * H);
    A("PL", B * H); A("Y", B * OUT); A("TGT", B * OUT); A("DY", B * OUT); A("DPL", B * H);
    this.ZMAX = Math.ceil(R / KCHUNK);
    A("PART", Math.max(this.ZMAX * 3 * H * H, Math.ceil(R / LNCH) * H));
    this.IDS = this.#buf("IDS", R);
    this.LENS = this.#buf("LENS", B);
    this.UTOK = this.#buf("UTOK", R);
    this.USTART = this.#buf("USTART", R + 1);
    this.UROWS = this.#buf("UROWS", R);
    this.LOSS = this.#buf("LOSS", 4096);
  }

  #buf(label, n, init = null) {
    const size = Math.max(16, Math.ceil((n * 4) / 16) * 16);
    const b = this.device.createBuffer({ size, label, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: !!init });
    if (init) { new Uint8Array(b.getMappedRange()).set(new Uint8Array(init.buffer, init.byteOffset, init.byteLength)); b.unmap(); }
    b._id = this.nid++;
    b._n = n;
    return b;
  }

  op(name, bufs, vals, groups) {
    const k = this.K[name];
    if (!k) throw new Error(`no kernel ${name}`);
    if (bufs.length !== k.nb || bufs.some((b) => !b)) throw new Error(`${name}: expected ${k.nb} buffers`);
    this.ops.push({ k, name, bufs, vals, groups: Array.isArray(groups) ? groups : grid(groups) });
  }

  clear(buf) { this.ops.push({ clear: buf }); }

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
      if (o.clear) { if (pass) { pass.end(); pass = null; } enc.clearBuffer(o.clear); continue; }
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
    const bytes = n * 4;
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

  // batch: { ids [B*S] (padded), B, S, lens [B], tgt Float32Array [B*768] (optional) }
  loadBatch(bt) {
    const q = this.device.queue, R = bt.B * bt.S;
    if (R > this.max.R || bt.B > this.max.B || bt.S > SMAX) throw new Error(`batch ${bt.B}x${bt.S} exceeds the trainer's sizes`);
    const ids = Uint32Array.from(bt.ids);
    q.writeBuffer(this.IDS, 0, ids);
    q.writeBuffer(this.LENS, 0, Uint32Array.from(bt.lens));
    if (bt.tgt) q.writeBuffer(this.TGT, 0, bt.tgt);
    // unique real tokens and their rows (padding rows carry no gradient)
    const rowsOf = new Map();
    for (let b = 0; b < bt.B; b++) for (let t = 0; t < bt.lens[b]; t++) {
      const r = b * bt.S + t, tok = ids[r];
      let l = rowsOf.get(tok);
      if (!l) rowsOf.set(tok, (l = []));
      l.push(r);
    }
    const nU = rowsOf.size, utok = new Uint32Array(nU), ust = new Uint32Array(nU + 1), urows = new Uint32Array(R);
    let u = 0, k = 0;
    for (const [tok, l] of rowsOf) { utok[u] = tok; ust[u] = k; for (const r of l) urows[k++] = r; u++; }
    ust[nU] = k;
    q.writeBuffer(this.UTOK, 0, utok);
    q.writeBuffer(this.USTART, 0, ust);
    q.writeBuffer(this.UROWS, 0, urows.subarray(0, Math.max(1, k)));
    this.b = { R, B: bt.B, S: bt.S, nU };
  }

  #lin(X, w, N, Kin, Y, R, acc = 0) {
    this.op("mm", [X, this.PRM, Y], { M: R, N, K: Kin, kChunk: Kin, aOff: 0, aRS: Kin, aCS: 1, bOff: this.poff[w], bRS: 1, bCS: Kin,
      cOff: 0, cRS: N, cCS: 1, acc, alpha: 1 }, [Math.ceil(N / 64), Math.ceil(R / 64), 1]);
  }

  #linT(DY, w, N, Kin, DX, R, acc = 0) {
    this.op("mm", [DY, this.PRM, DX], { M: R, N: Kin, K: N, kChunk: N, aOff: 0, aRS: N, aCS: 1, bOff: this.poff[w], bRS: Kin, bCS: 1,
      cOff: 0, cRS: Kin, cCS: 1, acc, alpha: 1 }, [Math.ceil(Kin / 64), Math.ceil(R / 64), 1]);
  }

  // GRD[w] (=) DY^T X: dW [N, Kin] summed over R rows, split-K then reduced
  #wgrad(DY, X, w, N, Kin, R) {
    const Z = Math.ceil(R / KCHUNK);
    this.op("mm", [DY, X, this.PART], { M: N, N: Kin, K: R, kChunk: KCHUNK, aOff: 0, aRS: 1, aCS: N, bOff: 0, bRS: Kin, bCS: 1, cOff: 0, cRS: Kin, cCS: 1, cZS: N * Kin, alpha: 1 },
      [Math.ceil(Kin / 64), Math.ceil(N / 64), Z]);
    this.op("sumz", [this.PART, this.GRD], { M: N, N: Kin, Z, cOff: this.poff[w], cRS: Kin, cCS: 1, alpha: 1 }, Math.ceil((N * Kin) / 256));
  }

  #lngrad(X, ST, DY, w, R) {
    const Z = Math.ceil(R / LNCH), { H } = this.C;
    this.op("ln_wpart", [X, ST, DY, this.PART], { R, chunk: LNCH }, Z);
    this.op("sumz", [this.PART, this.GRD], { M: 1, N: H, Z, cOff: this.poff[w], cRS: H, cCS: 1, alpha: 1 }, 1);
  }

  #layer(i, recompute = false) {
    const { H, I } = this.C, { R, S } = this.b, X = this.XS[i], eps = this.cfg.eps;
    let h = X;
    if (i > 0) { this.op("ln", [X, this.PRM, this.Hn, this.ST1], { R, eps, wOff: this.poff[`L${i}.attn_norm`] }, R); h = this.Hn; }
    this.#lin(h, `L${i}.Wqkv`, 3 * H, H, this.QKV, R);
    this.op("rope", [this.QKV, this.CS, this.QR], { R, S, sinOff: SMAX * (this.C.D / 2) }, R);
    const att = { R, S, isLocal: this.cfg.global[i] ? 0 : 1, win: this.cfg.window >> 1 };
    this.op("attn", [this.QR, this.QKV, this.LENS, this.O, this.LSE], att, R * this.C.NH);
    this.op("copy", [X, this.XM], { n: R * H }, Math.ceil((R * H) / 256));
    this.#lin(this.O, `L${i}.Wo`, H, H, this.XM, R, 1);
    this.op("ln", [this.XM, this.PRM, this.H2, this.ST2], { R, eps, wOff: this.poff[`L${i}.mlp_norm`] }, R);
    this.#lin(this.H2, `L${i}.Wi`, 2 * I, H, this.U, R);
    this.op("geglu", [this.U, this.G], { n: R * I }, Math.ceil((R * I) / 256));
    if (recompute) return { h, att };
    this.op("copy", [this.XM, this.XS[i + 1]], { n: R * H }, Math.ceil((R * H) / 256));
    this.#lin(this.G, `L${i}.Wo_mlp`, H, I, this.XS[i + 1], R, 1);
    return { h, att };
  }

  // forward to the 768-d outputs in Y
  forward() {
    const { R, B, S } = this.b, { H, OUT, L } = this.C, eps = this.cfg.eps;
    this.op("embed_f", [this.IDS, this.PRM, this.XE, this.XS[0], this.ST0], { R, eps, tOff: this.poff.table, wOff: this.poff.emb_norm }, R);
    for (let i = 0; i < L; i++) this.#layer(i);
    this.op("ln", [this.XS[L], this.PRM, this.HF, this.STF], { R, eps, wOff: this.poff.final_norm }, R);
    this.op("pool", [this.HF, this.LENS, this.PL], { B, S }, B);
    this.op("mm", [this.PL, this.PRM, this.Y], { M: B, N: OUT, K: H, kChunk: H, aOff: 0, aRS: H, aCS: 1, bOff: this.poff.dense, bRS: 1, bCS: H, cOff: 0, cRS: OUT, cCS: 1, alpha: 1 },
      [Math.ceil(OUT / 64), Math.ceil(B / 64), 1]);
  }

  forwardBackward({ slot = 0 } = {}) {
    const { H, I, L, NH, OUT } = this.C, { R, S, B, nU } = this.b;
    this.clear(this.GRD);
    this.forward();
    this.op("cosloss", [this.Y, this.TGT, this.DY, this.LOSS], { B, slot }, 1);
    // dense: dW = DY^T PL; dPL = DY W
    this.op("mm", [this.DY, this.PL, this.GRD], { M: OUT, N: H, K: B, kChunk: B, aOff: 0, aRS: 1, aCS: OUT, bOff: 0, bRS: H, bCS: 1, cOff: this.poff.dense, cRS: H, cCS: 1, alpha: 1 },
      [Math.ceil(H / 64), Math.ceil(OUT / 64), 1]);
    this.op("mm", [this.DY, this.PRM, this.DPL], { M: B, N: H, K: OUT, kChunk: OUT, aOff: 0, aRS: OUT, aCS: 1, bOff: this.poff.dense, bRS: H, bCS: 1, cOff: 0, cRS: H, cCS: 1, alpha: 1 },
      [Math.ceil(H / 64), Math.ceil(B / 64), 1]);
    this.op("unpool", [this.DPL, this.LENS, this.DHF], { R, S }, R);
    let dxo = this.DXA, dxi = this.DXB;
    this.op("ln_bwd", [this.XS[L], this.STF, this.PRM, this.DHF, dxo], { R, wOff: this.poff.final_norm }, R);
    this.#lngrad(this.XS[L], this.STF, this.DHF, "final_norm", R);
    for (let i = L - 1; i >= 0; i--) {
      const X = this.XS[i], pf = `L${i}.`;
      const { h, att } = this.#layer(i, true);
      // MLP
      this.#linT(dxo, pf + "Wo_mlp", H, I, this.DG, R);
      this.#wgrad(dxo, this.G, pf + "Wo_mlp", H, I, R);
      this.op("geglu_bwd", [this.U, this.DG, this.DU], { n: R * I }, Math.ceil((R * I) / 256));
      this.#linT(this.DU, pf + "Wi", 2 * I, H, this.DH2, R);
      this.#wgrad(this.DU, this.H2, pf + "Wi", 2 * I, H, R);
      this.op("copy", [dxo, this.DXM], { n: R * H }, Math.ceil((R * H) / 256));
      this.op("ln_bwd", [this.XM, this.ST2, this.PRM, this.DH2, this.DXM], { R, wOff: this.poff[pf + "mlp_norm"], acc: 1 }, R);
      this.#lngrad(this.XM, this.ST2, this.DH2, pf + "mlp_norm", R);
      // attention
      this.#linT(this.DXM, pf + "Wo", H, H, this.DO, R);
      this.#wgrad(this.DXM, this.O, pf + "Wo", H, H, R);
      this.op("attn_bq", [this.QR, this.QKV, this.LENS, this.LSE, this.DO, this.O, this.DQR, this.DD], att, R * NH);
      this.op("attn_bkv", [this.QR, this.QKV, this.LENS, this.LSE, this.DO, this.DD, this.DQR, this.DQKV], att, R * NH);
      this.op("rope_bwd", [this.DQR, this.CS, this.DQKV], { R, S, sinOff: SMAX * (this.C.D / 2) }, R);
      this.#linT(this.DQKV, pf + "Wqkv", 3 * H, H, this.DH, R);
      this.#wgrad(this.DQKV, h, pf + "Wqkv", 3 * H, H, R);
      this.op("copy", [this.DXM, dxi], { n: R * H }, Math.ceil((R * H) / 256));
      if (i > 0) {
        this.op("ln_bwd", [X, this.ST1, this.PRM, this.DH, dxi], { R, wOff: this.poff[pf + "attn_norm"], acc: 1 }, R);
        this.#lngrad(X, this.ST1, this.DH, pf + "attn_norm", R);
      } else this.op("copy", [this.DH, dxi], { n: R * H, acc: 1 }, Math.ceil((R * H) / 256));
      [dxo, dxi] = [dxi, dxo];
    }
    // embedding LayerNorm, then the table rows
    this.op("ln_bwd", [this.XE, this.ST0, this.PRM, dxo, this.DXE], { R, wOff: this.poff.emb_norm }, R);
    this.#lngrad(this.XE, this.ST0, dxo, "emb_norm", R);
    this.op("tgrad", [this.UTOK, this.USTART, this.UROWS, this.DXE, this.GRD], { nU, tOff: this.poff.table }, nU);
  }

  adamStep({ lr, wd = 0.01, step, clip = 1.0, b1 = 0.9, b2 = 0.999, eps = 1e-8 }) {
    const n = this.nParams;
    this.op("sqsum", [this.GRD, this.PARTN], { n }, this.NPART);
    this.op("norm", [this.PARTN, this.NRM], { n: this.NPART }, 1);
    this.op("adamw", [this.PRM, this.GRD, this.AM, this.AV, this.NRM], { n, headOff: n, lr, lrHead: lr, wd, b1, b2, eps,
      bc1: 1 - b1 ** step, bc2s: Math.sqrt(1 - b2 ** step), clip }, Math.ceil(n / 256));
  }
}
