// A WebGPU inference engine, in hand-written WGSL, for EmbeddingGemma 2 cut to its first k layers
// (oaustegard/experiments eg2-layer-student). It runs the same graph as model_t3w.onnx and reads the
// same weights file (model_t3w.onnx_data) through a manifest of byte offsets (run/manifest.py), so a
// page that also keeps onnxruntime-web as a fallback downloads the weights once.
//
//   const eng = await Eg2Engine.create(device, manifest, onnxDataArrayBuffer);
//   const vecs = await eng.embed([ids0, ids1, ...]);   // Float32Array[B * 768], unit rows
//   await eng.embedTo([ids], gpuBuffer);                // leaves the vector on the GPU
//
// Graph, per token: 3-bit token table -> x * sqrt(512); PLE = rms(x W_ple * 512^-0.5) per layer;
// k layers of [rms -> q,k,v (q4) -> q/k rms+weight, v rms, RoPE -> bidirectional GQA attention ->
// o -> h += rms(o)*w ; h += rms(mlp(rms(h)*w))*w ; h = (h + rms(W_p(gelu(W_g h) * ple_i))*w) * s];
// then rms(h)*w, mean over the sequence, the 512->768 projection (q4) and L2 normalization. The
// projection is linear, so it runs once per sequence after pooling instead of once per token.
// Several sequences run in one pass, packed back to back; attention stays inside each sequence.
//
// The same engine runs the full 24-layer model (onnx-community/embeddinggemma-2-ONNX model_q4.onnx,
// manifest from run/manifest_full.py): a 4-bit token table, full-attention layers with head_dim 512
// and one KV head on their own RoPE table, and per-norm eps. Its 174 MB weight file is one buffer,
// and each dispatch binds the window of it that holds its tensors.
//
// The matmul, norm, attention, pool and l2 kernels are the shared templates in _lib/wgsl/kernels.js;
// the token-table gathers, the q/k norm + RoPE prep and the PLE gate are this file's. Batching,
// dispatch recording, weight upload and embed / embedTo / probe come from _lib/wgsl/runtime.js.

import { BYTE, HEADER, MAX_SEQ, SeqEngine, split } from "../../_lib/wgsl/runtime.js";
import { KERNELS, attention, l2, matmul, norm, pool } from "../../_lib/wgsl/kernels.js";

const SHADERS = {
  // One workgroup per token, one thread per 8-value group (3 bytes) of the 512-wide row.
  // p.a = (codes, scales, zp) offsets in words; p.f.x = embedding scale.
  embed: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${BYTE}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  if (t >= dn.n.x) { return; }
  let id = tok[t].x;
  let g = lid.x;
  let cb = p.a.x * 4u + id * 192u + g * 3u;
  let word = wbyte(cb) | (wbyte(cb + 1u) << 8u) | (wbyte(cb + 2u) << 16u);
  let blk = g / 4u;
  let z = f32(wbyte(p.a.z * 4u + id * 16u + blk));
  let s = unpack2x16float(W[p.a.y + id * 8u + blk / 2u])[blk % 2u];
  for (var i = 0u; i < 8u; i++) {
    y[t * 512u + g * 8u + i] = (f32((word >> (3u * i)) & 7u) - z) * s * p.f.x;
  }
}`,

  // 4-bit token table (onnxruntime GatherBlockQuantized: 2 values per byte, low nibble first, f32
  // scales [V, 16], packed 4-bit zero points [V, 8]). One workgroup per token, one thread per word.
  // p.a = (codes, scales, zp) offsets in words; p.f.x = embedding scale.
  embed_q4: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${BYTE}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x;
  if (t >= dn.n.x) { return; }
  let id = tok[t].x;
  let g = lid.x;
  let word = W[p.a.x + id * 64u + g];
  let blk = g / 4u;
  let z = f32((wbyte(p.a.z * 4u + id * 8u + blk / 2u) >> (4u * (blk % 2u))) & 15u);
  let s = bitcast<f32>(W[p.a.y + id * 16u + blk]) * p.f.x;
  for (var i = 0u; i < 8u; i++) {
    y[t * 512u + g * 8u + i] = (f32((word >> (4u * i)) & 15u) - z) * s;
  }
}`,

  // rhq token table (eg2-vocab-codebook/run/rhq.py): row = scale * s * H(levels[codes]), H the
  // orthonormal 512-point Walsh-Hadamard transform, s fixed signs. Rows [0, n3) hold 3-bit codes
  // (192 bytes each), the rest 2-bit codes (128 bytes), little-endian bitstreams; one fp16 scale per
  // row. One workgroup of 256 threads per token: each thread loads two levels, then nine butterfly
  // stages run in workgroup memory.
  // p.a = (codes3, codes2, scales, n3) word offsets / count; p.b = (signs, levels3, levels2) word
  // offsets; p.f.x = embedding scale.
  embed_rhq: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> v: array<f32, 512>;
${BYTE}
fn code(base: u32, bits: u32, i: u32) -> u32 {
  let bit = i * bits;
  let b0 = base + (bit >> 3u);
  let two = wbyte(b0) | (wbyte(b0 + 1u) << 8u);
  return (two >> (bit & 7u)) & ((1u << bits) - 1u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let t = wid.x;
  if (t >= dn.n.x) { return; }
  let id = tok[t].x;
  let n3 = p.a.w;
  var bits = 3u; var base = p.a.x * 4u + id * 192u; var lv = p.b.y;
  if (id >= n3) { bits = 2u; base = p.a.y * 4u + (id - n3) * 128u; lv = p.b.z; }
  for (var j = 0u; j < 2u; j++) {
    let i = l + j * 256u;
    v[i] = bitcast<f32>(W[lv + code(base, bits, i)]);
  }
  workgroupBarrier();
  for (var h = 1u; h < 512u; h = h * 2u) {
    let a = (l / h) * 2u * h + (l % h);
    let x = v[a]; let z = v[a + h];
    workgroupBarrier();
    v[a] = x + z; v[a + h] = x - z;
    workgroupBarrier();
  }
  let sc = unpack2x16float(W[p.a.z + id / 2u])[id % 2u] * p.f.x * 0.04419417382415922;  // 1/sqrt(512)
  for (var j = 0u; j < 2u; j++) {
    let i = l + j * 256u;
    let sg = select(-1.0, 1.0, ((W[p.b.x + i / 32u] >> (i % 32u)) & 1u) == 1u);
    y[t * 512u + i] = v[i] * sg * sc;
  }
}`,

  // Per-head RMSNorm of q, k (with weights) and v (without), then RoPE on q and k. One workgroup
  // per (token, head slot): slots [0, nh) are q heads, [nh, nh+nkv) k heads, the rest v heads.
  // Head dim hd = 256 or 512: thread i holds elements i + 128 j and i + 128 j + hd / 2 (the pairs
  // rotate_half mixes) for j < hd / 256.
  // p.a = (nh, nkv, q_norm offset, k_norm offset); p.b = (hd, RoPE table offset in floats); p.f.x = eps.
  prep: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> rope: array<f32>;
@group(0) @binding(4) var<storage, read> tok: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> q: array<f32>;
@group(0) @binding(6) var<storage, read_write> k: array<f32>;
@group(0) @binding(7) var<storage, read_write> v: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
  let t = wid.x; let slot = wid.y;
  if (t >= dn.n.x) { return; }
  let nh = p.a.x; let nkv = p.a.y;
  let hd = p.b.x; let half = hd / 2u; let np = hd / 256u;
  var kind = 0u; var base = 0u;
  if (slot < nh) { base = (t * nh + slot) * hd; }
  else if (slot < nh + nkv) { kind = 1u; base = (t * nkv + slot - nh) * hd; }
  else { kind = 2u; base = (t * nkv + slot - nh - nkv) * hd; }
  var x0 = array<f32, 2>(0.0, 0.0); var x1 = array<f32, 2>(0.0, 0.0);
  var ss = 0.0;
  for (var j = 0u; j < np; j++) {
    let e = base + i + 128u * j;
    if (kind == 0u) { x0[j] = q[e]; x1[j] = q[e + half]; }
    else if (kind == 1u) { x0[j] = k[e]; x1[j] = k[e + half]; }
    else { x0[j] = v[e]; x1[j] = v[e + half]; }
    ss += x0[j] * x0[j] + x1[j] * x1[j];
  }
  red[i] = ss;
  workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) {
    if (i < s) { red[i] += red[i + s]; }
    workgroupBarrier();
  }
  let inv = 1.0 / sqrt(red[0] / f32(hd) + p.f.x);
  let wo = select(p.a.w, p.a.z, kind == 0u);
  let pos = tok[t].y;
  let ro = p.b.y + pos * half;
  for (var j = 0u; j < np; j++) {
    let d = i + 128u * j;
    let e = base + d;
    var a = x0[j] * inv; var b = x1[j] * inv;
    if (kind == 2u) { v[e] = a; v[e + half] = b; continue; }
    a *= bitcast<f32>(W[wo + d]); b *= bitcast<f32>(W[wo + d + half]);
    let c = rope[ro + d]; let sn = rope[ro + ${MAX_SEQ}u * half + d];
    let y0 = a * c - b * sn; let y1 = b * c + a * sn;
    if (kind == 0u) { q[e] = y0; q[e + half] = y1; }
    else { k[e] = y0; k[e + half] = y1; }
  }
}`,

  // a[r, j] = gelu_tanh(a[r, j]) * b[r * bStride + bOff + j] for r < tokens, j < D.
  // p.a = (D, bStride, bOff).
  gelumul: HEADER + /* wgsl */ `
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let D = p.a.x;
  let idx = gid.x + gid.y * 65535u * 256u;
  if (idx >= dn.n.x * D) { return; }
  let r = idx / D; let j = idx % D;
  let x = a[idx];
  let u = clamp(0.7978845608028654 * (x + 0.044715 * x * x * x), -15.0, 15.0);
  a[idx] = 0.5 * x * (1.0 + tanh(u)) * b[r * p.a.y + p.a.z + j];
}`,
};

export class Eg2Engine extends SeqEngine {
  // manifest: web/manifest.json; data: the model_t3w.onnx_data bytes (ArrayBuffer or typed array).
  // opts.maxTokens bounds one batch (all sequences together); each sequence is at most 128 tokens.
  // embed, embedTo, probe and destroy come from SeqEngine (_lib/wgsl/runtime.js).
  async _build(bytes) {
    const dev = this.device, m = this.m, T = this.maxTokens, B = this.maxSeqs;
    // RoPE inverse frequencies per attention kind (the cut has one table, the full model two).
    const freqs = m.rope ?? { default: m.inv_freq };
    const tbl = m.table.kind ?? "t3";
    for (const L of m.layers) {
      const f = freqs[L.rope ?? "default"];
      if (![256, 512].includes(L.head_dim) || L.n_heads % L.n_kv_heads || !f || f.length * 2 !== L.head_dim) {
        throw new Error("engine expects head_dim 256 or 512, grouped heads and a matching RoPE table");
      }
    }
    if (m.hidden !== 512 || !["t3", "rhq", "q4"].includes(tbl) || (tbl === "t3" && m.table.bits !== 3)) throw new Error("unexpected manifest shape");
    // The whole weight file in one storage buffer; every tensor is addressed by its word offset
    // inside the window of the buffer its dispatch binds (the full model's 174 MB exceed the
    // default 128 MiB binding limit; the whole file stays under the 256 MiB buffer limit).
    const win = this.uploadWhole(bytes);

    const ropeOff = {};
    let nRope = 0;
    for (const [k, f] of Object.entries(freqs)) { ropeOff[k] = nRope; nRope += 2 * MAX_SEQ * f.length; }
    const rope = new Float32Array(nRope);
    for (const [k, f] of Object.entries(freqs)) {
      const o = ropeOff[k], half = f.length;
      for (let pos = 0; pos < MAX_SEQ; pos++) {
        for (let i = 0; i < half; i++) {
          const ang = Math.fround(pos * f[i]); // the graph forms the angle in float32
          rope[o + pos * half + i] = Math.cos(ang);
          rope[o + MAX_SEQ * half + pos * half + i] = Math.sin(ang);
        }
      }
    }
    this.rope = this.upload(rope, "rope");
    const buf = (n, label, extra) => this.buf(n, label, extra), U = GPUBufferUsage;
    const most = (f) => Math.max(...m.layers.map(f));
    const qN = most((L) => L.n_heads * L.head_dim), kN = most((L) => L.n_kv_heads * L.head_dim), ff = most((L) => L.gate_proj.N);
    const B_ = {
      tok: buf(T * 4, "tok", U.COPY_DST), seqs: buf(B * 4, "seqs", U.COPY_DST),
      h: buf(T * 512, "h"), a: buf(T * 512, "a"), t: buf(T * 512, "t"),
      q: buf(T * qN, "q"), k: buf(T * kN, "k"), v: buf(T * kN, "v"), o: buf(T * qN, "o"),
      g: buf(T * ff, "g"), u: buf(T * Math.max(ff, m.ple_dim), "u"),
      pleRaw: buf(T * m.ple_layers * m.ple_dim, "pleRaw"), ple: buf(T * m.ple_layers * m.ple_dim, "ple"),
      pooled: buf(B * 512, "pooled"), e: buf(B * 768, "e"), out: buf(B * 768, "out", U.COPY_SRC),
    };
    this.b = B_;
    this.shaders = { ...KERNELS, ...SHADERS };

    // Word offset of a tensor inside the weight binding that starts at byte `base`.
    const W = (t, base) => {
      if (t.offset % 4) throw new Error("tensor offset not word-aligned");
      return (t.offset - base) / 4;
    };
    const step = (...args) => this.step(...args);
    const eps = m.eps;
    // MatMulNBits weights (q4); rowsSel 1 runs over sequences (after pooling) instead of tokens.
    const mm = (w, X, Y, rowsSel = 0, alpha = 1) => {
      if (w.block !== 32 || w.bits !== 4) throw new Error("matmul: expected 4-bit MatMulNBits with block 32");
      const { base, res } = win(w.q, w.scales, w.zp);
      matmul(this, { fmt: "q4", offs: [W(w.q, base), W(w.scales, base), W(w.zp, base)], N: w.N, K: w.K }, res, X, Y, { rowsSel, alpha });
    };
    // RMSNorm; residual: y = (y + rms(x) * w) * scalar.
    const rms = (w, X, Y, D, residual = false, scalar = 1, rowsSel = 0, mult = 1) => {
      const { base, res } = win(w);
      norm(this, "rms", res, W(w, base), X, Y, { D, eps: w.eps ?? eps, mode: residual ? 2 : 0, scalar, rowsSel, mult });
    };
    const gelumul = (A, Bf, D, bStride, bOff) =>
      step("gelumul", { a: [D, bStride, bOff] }, [A, Bf], (n) => split(Math.ceil(n[0] * D / 256)));

    const t = m.table;
    if (tbl === "rhq") {
      const { base, res } = win(t.codes3, t.codes2, t.scales, t.signs, t.levels3, t.levels2);
      step("embed_rhq", { a: [W(t.codes3, base), W(t.codes2, base), W(t.scales, base), t.n3],
                          b: [W(t.signs, base), W(t.levels3, base), W(t.levels2, base)], f: [m.embed_scale] },
           [res, B_.tok, B_.h], (n) => [n[0]]);
    } else {
      const { base, res } = win(t.codes, t.scales, t.zp);
      step(tbl === "q4" ? "embed_q4" : "embed", { a: [W(t.codes, base), W(t.scales, base), W(t.zp, base)], f: [m.embed_scale] },
           [res, B_.tok, B_.h], (n) => [n[0]]);
    }
    mm(m.ple, B_.h, B_.pleRaw, 0, m.ple_scale);
    rms(m.per_layer_projection_norm, B_.pleRaw, B_.ple, m.ple_dim, false, 1, 0, m.ple_layers);
    m.layers.forEach((L, i) => {
      rms(L.input_layernorm, B_.h, B_.a, 512);
      mm(L.q_proj, B_.a, B_.q); mm(L.k_proj, B_.a, B_.k); mm(L.v_proj, B_.a, B_.v);
      const pw = win(L.q_norm, L.k_norm);
      step("prep", { a: [L.n_heads, L.n_kv_heads, W(L.q_norm, pw.base), W(L.k_norm, pw.base)], b: [L.head_dim, ropeOff[L.rope ?? "default"]],
                     f: [L.q_norm.eps ?? eps] },
           [pw.res, this.rope, B_.tok, B_.q, B_.k, B_.v], (n) => [n[0], L.n_heads + 2 * L.n_kv_heads]);
      attention(this, B_.q, B_.k, B_.v, B_.o, { nh: L.n_heads, nkv: L.n_kv_heads, hd: L.head_dim, scale: L.attn_scale });
      mm(L.o_proj, B_.o, B_.t);
      rms(L.post_attention_layernorm, B_.t, B_.h, 512, true);
      rms(L.pre_feedforward_layernorm, B_.h, B_.a, 512);
      mm(L.gate_proj, B_.a, B_.g); mm(L.up_proj, B_.a, B_.u);
      gelumul(B_.g, B_.u, L.gate_proj.N, L.gate_proj.N, 0);
      mm(L.down_proj, B_.g, B_.t);
      rms(L.post_feedforward_layernorm, B_.t, B_.h, 512, true);
      mm(L.per_layer_input_gate, B_.h, B_.u);
      gelumul(B_.u, B_.ple, m.ple_dim, m.ple_layers * m.ple_dim, i * m.ple_dim);
      mm(L.per_layer_projection, B_.u, B_.t);
      rms(L.post_per_layer_input_norm, B_.t, B_.h, 512, true, L.layer_scalar);
    });
    rms(m.norm, B_.h, B_.a, 512);
    pool(this, B_.a, B_.pooled, 512);
    mm(m.embedding_projection, B_.pooled, B_.e, 1);
    l2(this, B_.e, B_.out, 768);
  }
}
