// A WebGPU inference engine, in hand-written WGSL, for the small query embedders of ettin-native:
// Ettin-17m (ModernBERT) and leaf-ir (BERT). It reads the same single-file model.onnx that
// onnxruntime-web loads, through web/models/<pkg>/manifest.json (each tensor's byte range in the file,
// run/webexport.py), so a page that keeps onnxruntime-web as a fallback downloads the weights once.
// run/ref.py is the NumPy spec it follows.
//
//   const eng = await SmallEngine.create(device, manifest, modelOnnxBytes);
//   const vecs = await eng.embed([ids0, ids1, ...]);   // Float32Array[B * 768], unit rows
//
// Weights stay in their shipped formats: int8 matmul codes with one f32 scale per output column
// (transposed once at init from the file's [K, N] to [N, K]), a 4-bit (packed nibbles, fp16 scale per 32), 3-bit (eg2-layer-student layout)
// or int8 (f32 scale per row, leaf-wasm's file through leaf-wasm/web/leaf-engine.js) token table, f32 norms and biases. Activations are f32 throughout. Several sequences run in one
// pass, packed back to back; attention stays inside each sequence.
// Every kernel is a shared template in _lib/wgsl/kernels.js (matmul, norm, attention, pool, l2);
// so are the token-table gather, RoPE and the GLU. Batching, dispatch recording, weight
// upload and embed / embedTo / probe come from _lib/wgsl/runtime.js.

import { MAX_SEQ, NONE, SeqEngine } from "../../_lib/wgsl/runtime.js";
import { KERNELS, attention, embed, gated, l2, matmul, norm, pool, rope, ropeTable } from "../../_lib/wgsl/kernels.js";

export class SmallEngine extends SeqEngine {
  // manifest: web/models/<pkg>/manifest.json (copied; offsets are rewritten); data: model.onnx bytes.
  // opts.maxTokens bounds one batch (all sequences together); each sequence is at most 128 tokens.
  // embed, embedTo, probe and destroy come from SeqEngine (_lib/wgsl/runtime.js).
  static create(device, manifest, data, opts = {}) {
    return super.create(device, structuredClone(manifest), data, opts);
  }

  constructor(device, manifest, opts) {
    super(device, manifest, opts);
    this.dim = manifest.dim;
  }

  async _build(bytes) {
    const dev = this.device, m = this.m, T = this.maxTokens, B = this.maxSeqs;
    const H = m.hidden, nh = m.heads, D = H / nh;
    if (m.dim !== 768 || D > 64 || H > 512 || H % 32) throw new Error("unexpected manifest shape");
    // The manifest gives each tensor's byte range inside model.onnx, where protobuf leaves them
    // unaligned. Copy every tensor once into a 4-byte-aligned buffer and point the manifest at it.
    bytes = this.uploadPacked(bytes);

    const buf = (n, label, extra) => this.buf(n, label, extra), U = GPUBufferUsage;
    // RoPE tables [global, local][pos][D/2]: cos block, then sin block. Angles formed in float32 like the graph.
    const half = D / 2;
    this.rope = this.upload(m.arch === "modernbert" ? ropeTable([m.theta_global, m.theta_local], D, MAX_SEQ) : new Float32Array(4), "rope");
    const I = m.inter;
    const bert = m.arch === "bert";
    const B_ = {
      tok: buf(T * 4, "tok", U.COPY_DST), seqs: buf(B * 4, "seqs", U.COPY_DST),
      e: buf(T * H, "e"), h: buf(T * H, "h"), a: buf(T * H, "a"), o: buf(T * H, "o"), t: buf(T * H, "t"),
      qkv: buf(T * 3 * H, "qkv"), u: buf(T * (bert ? I : 2 * I), "u"), g: buf(T * I, "g"),
      pooled: buf(B * H, "pooled"), d: buf(B * 768, "d"), out: buf(B * 768, "out"),
    };
    this.b = B_;
    this.shaders = { ...KERNELS };

    const step = (...args) => this.step(...args);
    const word = (t) => { if (t.offset % 4) throw new Error("unaligned f32 tensor"); return t.offset / 4; };
    // Transpose every int8 matmul from the file's [K, N] to [N, K] (4 k per u32) into one buffer, once.
    const lins = [];
    const walk = (o) => { if (o && typeof o === "object") { if (o.q && o.s) lins.push(o); else Object.values(o).forEach(walk); } };
    walk(m);
    let words = 0;
    for (const w of lins) { if (w.K % 32 || w.N % 32) throw new Error("matmul shape"); w._wt = words; words += w.N * w.K / 4; }
    const wt = new Int8Array(words * 4);
    for (const w of lins) {
      const src = new Int8Array(bytes.buffer, bytes.byteOffset + w.q.offset, w.K * w.N), dst = w._wt * 4;
      for (let k = 0; k < w.K; k++) for (let n = 0; n < w.N; n++) wt[dst + n * w.K + k] = src[k * w.N + n];
    }
    this.Wt = this.upload(wt, "int8 weights [N, K]");
    // int8 (repacked [N, K] in Wt) or f32 [N, K] weights, optional bias; mode bit 0 GELU, bit 1 Y +=.
    const mm = (w, X, Y, mode = 0, rowsSel = 0) => {
      let f;
      if (w.q) f = { fmt: "i8", offs: [w._wt, word(w.s)] };
      else if (w.f32_t) f = { fmt: "f32", offs: [word(w.f32_t)] };
      else throw new Error("f32 [K, N] matmul weights are not supported; export them as [N, K]");
      matmul(this, { ...f, N: w.N, K: w.K }, this.W, X, Y,
             { rowsSel, bias: w.b ? word(w.b) : NONE, gelu: !!(mode & 1), accumulate: !!(mode & 2), Wt: this.Wt });
    };
    // LayerNorm (bias optional); mode 1: y = LN(y + x).
    const ln = (w, b, X, Y, mode = 0, rowsSel = 0) =>
      norm(this, "ln", this.W, word(w), X, Y, { D: H, eps: m.eps, bOff: b ? word(b) : NONE, mode, rowsSel });

    const tb = m.table;
    embed(this, this.W, B_.e, { fmt: { 4: "t4", 8: "t8" }[tb.bits] ?? "t3", codes: tb.codes.offset, scales: tb.scales.offset, zp: tb.zp?.offset },
          { H, pos: bert ? word(m.pos) : NONE, type0: bert ? word(m.type0) : 0 });
    if (!bert) {
      ln(m.emb_norm, null, B_.e, B_.h);
      m.layers.forEach((L) => {
        let X = B_.h;
        if (L.attn_norm) { ln(L.attn_norm, null, B_.h, B_.a); X = B_.a; }
        mm(L.Wqkv, X, B_.qkv);
        rope(this, this.rope, B_.qkv, { nh, hd: D, maxSeq: MAX_SEQ, tables: 2, sel: L.global ? 0 : 1, parts: 2, stride: 3 * H });
        attention(this, B_.qkv, B_.qkv, B_.qkv, B_.o, { nh, hd: D, fused: true, window: L.global ? 0 : m.window });
        mm(L.Wo, B_.o, B_.h, 2);
        ln(L.mlp_norm, null, B_.h, B_.a);
        mm(L.Wi, B_.a, B_.u);
        gated(this, B_.u, B_.u, B_.g, { I, sa: 2 * I, sb: 2 * I, ob: I });
        mm(L.Wo_mlp, B_.g, B_.h, 2);
      });
      ln(m.final_norm, null, B_.h, B_.a);
      pool(this, B_.a, B_.pooled, H);
    } else {
      ln(m.emb_ln.w, m.emb_ln.b, B_.e, B_.h);
      m.layers.forEach((L) => {
        mm(L.q, B_.h, B_.a); mm(L.k, B_.h, B_.t); mm(L.v, B_.h, B_.e);
        attention(this, B_.a, B_.t, B_.e, B_.o, { nh, hd: D });
        mm(L.o, B_.o, B_.t);
        ln(L.ln1.w, L.ln1.b, B_.t, B_.h, 1);
        mm(L.up, B_.h, B_.u, 1);
        mm(L.down, B_.u, B_.t);
        ln(L.ln2.w, L.ln2.b, B_.t, B_.h, 1);
      });
      pool(this, B_.h, B_.pooled, H);
    }
    mm(m.dense, B_.pooled, B_.d, 0, 1);
    l2(this, B_.d, B_.out, 768);
  }
}
