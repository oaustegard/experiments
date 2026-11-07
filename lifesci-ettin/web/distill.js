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

import { SpecRunner, grid } from "../../_lib/wgsl/train/spec.js";
import { fullKernels as kernels } from "../../_lib/wgsl/train/kernels.js";
import { recordNce } from "../../_lib/wgsl/train/nce.js";

const SMAX = 1024, KCHUNK = 512, LNCH = 256;

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

export class DistillTrainer extends SpecRunner {
  // opts: { V, maxRows, maxSeqs, window (local attention, 128), theta, eps }
  static async create(device, opts) {
    const t = new DistillTrainer(device, opts);
    await t.#init();
    return t;
  }

  constructor(device, { V, maxRows, maxSeqs = 128, window = 128, theta = 160000, eps = 1e-5, H = 256, NH = 4, I = 384, L = 7, OUT = 768 }) {
    super(device, { maxSlots: 4096 });
    this.C = { H, NH, D: H / NH, I, L, OUT, V, r: 8 };
    this.cfg = { window, theta, eps, global: Array.from({ length: L }, (_, i) => i % 3 === 0) };
    this.max = { R: maxRows, B: maxSeqs };
    const { layout, n } = layoutFor(V, { H, I, L, OUT });
    this.layout = layout;
    this.poff = Object.fromEntries(layout.map((x) => [x.name, x.offset]));
    this.nParams = n;
  }

  async #init() {
    const dev = this.device, { H, I, L, NH, D, OUT } = this.C, R = this.max.R, B = this.max.B;
    const K = kernels(this.C);
    await this.compile(K);
    const half = D / 2, cs = new Float32Array(2 * SMAX * half), fr = Math.fround;
    for (let j = 0; j < half; j++) {
      const inv = fr(1 / fr(this.cfg.theta ** fr((2 * j) / D)));
      for (let p = 0; p < SMAX; p++) { const a = fr(p * inv); cs[p * half + j] = Math.cos(a); cs[SMAX * half + p * half + j] = Math.sin(a); }
    }
    this.CS = this.buf("CS", cs.length, cs);
    for (const nm of ["PRM", "GRD", "AM", "AV"]) this[nm] = this.buf(nm, this.nParams);
    this.NPART = 256;
    this.PARTN = this.buf("PARTN", this.NPART);
    this.NRM = this.buf("NRM", 4);
    this.XS = Array.from({ length: L + 1 }, (_, i) => this.buf(`x${i}`, R * H));
    const A = (name, size) => { this[name] = this.buf(name, size); };
    A("XE", R * H); A("ST0", 2 * R); A("DXE", R * H);
    A("Hn", R * H); A("ST1", 2 * R); A("QKV", R * 3 * H); A("QR", R * 2 * H); A("O", R * H); A("LSE", R * NH);
    A("XM", R * H); A("H2", R * H); A("ST2", 2 * R); A("U", R * 2 * I); A("G", R * I);
    A("DXA", R * H); A("DXB", R * H); A("DG", R * I); A("DU", R * 2 * I); A("DH2", R * H); A("DO", R * H);
    A("DH", R * H); A("DXM", R * H); A("DQR", R * 2 * H); A("DQKV", R * 3 * H); A("DD", R * NH);
    A("HF", R * H); A("STF", 2 * R); A("DHF", R * H);
    A("PL", B * H); A("Y", B * OUT); A("TGT", B * OUT); A("DY", B * OUT); A("DPL", B * H);
    // contrastive loss (web/nce.js): B / 2 pairs
    A("YN", B * OUT); A("NR", B); A("DYN", B * OUT); A("SIM", (B / 2) ** 2); A("DS", (B / 2) ** 2); A("LROW", B / 2);
    this.ZMAX = Math.ceil(R / KCHUNK);
    A("PART", Math.max(this.ZMAX * 3 * H * H, Math.ceil(R / LNCH) * H));
    this.IDS = this.buf("IDS", R);
    this.LENS = this.buf("LENS", B);
    this.UTOK = this.buf("UTOK", R);
    this.USTART = this.buf("USTART", R + 1);
    this.UROWS = this.buf("UROWS", R);
    this.LOSS = this.buf("LOSS", 4096);
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

  // loss "cos": 1 - cos to the targets; "nce": rows 0..B/2-1 are queries and B/2..B-1 their positives
  // (web/nce.js), scale = 1 / temperature
  forwardBackward({ slot = 0, loss = "cos", scale = 20 } = {}) {
    const { H, I, L, NH, OUT } = this.C, { R, S, B, nU } = this.b;
    this.clear(this.GRD);
    this.forward();
    if (loss === "nce") {
      if (B % 2) throw new Error("nce needs an even batch: queries then their positives");
      recordNce(this, { P: B / 2, scale, slot });
    } else this.op("cosloss", [this.Y, this.TGT, this.DY, this.LOSS], { B, slot }, 1);
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
