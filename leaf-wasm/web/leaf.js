// mdbr-leaf-ir forward pass in plain JS + one WASM SIMD matmul kernel (kernel*.wat). No runtime.
//
//   const leaf = await createLeaf({ weights, vocab, wasm });   // ArrayBuffer, string, kernel
//   const v = leaf.embed(text, { query: true });                // Float32Array(768), unit length
//
// wasm: bytes or a WebAssembly.Module, or an array of candidates tried in order; the first
// that validates is used, so [kernel-fma.wasm, kernel.wasm] falls back where relaxed SIMD is
// missing. leaf.kernel is the index chosen.
//
// BERT (post-LayerNorm, GELU-erf, 6 layers, hidden 384, 12 heads) -> masked mean pool ->
// Dense 384->768 -> L2 normalize, as sentence-transformers runs it. Weights come from
// run/export_weights.py: f32 or int8 (dequantized into f32 at load, so the math is f32 either way).
// Matmuls run in the kernel; LayerNorm, softmax, GELU, pooling run in JS.

import { WordPiece } from "./tokenizer.js";

export const QUERY_PROMPT = "Represent this sentence for searching relevant passages: ";
const MAX_TOKENS = 128;

function parse(buf) {
  const dv = new DataView(buf);
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== "LEAF") throw new Error("not a LEAF weights file");
  const hlen = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, hlen)));
  const base = Math.ceil((12 + hlen) / 16) * 16;
  const get = (name) => {
    const t = header.tensors[name];
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === "f32") return { ...t, f32: new Float32Array(buf, base + t.offset, n) };
    if (t.dtype === "i4") // [V][H/2] packed, low nibble = even column, value + 8; f32 scales [V][H/group]
      return { ...t, i4: new Uint8Array(buf, base + t.offset, n / 2),
               scaleArr: new Float32Array(buf, base + t.scale, n / t.group) };
    const scale = new Float32Array(buf, base + t.scale, t.shape[t.axis]);
    return { ...t, i8: new Int8Array(buf, base + t.offset, n), scaleArr: scale };
  };
  return { config: header.config, get, names: Object.keys(header.tensors) };
}

// Numerical Recipes erfc, fractional error < 1.2e-7.
function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

export async function createLeaf({ weights, vocab, wasm }) {
  const W = parse(weights);
  const { hidden: H, heads: NH, layers: L, ffn: F, out_dim: D, eps } = W.config;
  const DH = H / NH, SP = MAX_TOKENS; // activation buffers sized for the longest (padded) input
  const tok = new WordPiece(vocab, MAX_TOKENS);

  // Wasm memory: every matrix, bias the kernel reads, then activations.
  const kernelTensors = W.names.filter((n) => /^l\d+\.(qkv|o|up|down)_[wb]$/.test(n));
  let bytes = 0;
  const sizeOf = (n) => W.get(n).shape.reduce((a, b) => a * b, 1) * 4;
  for (const n of kernelTensors) bytes += sizeOf(n);
  const act = { x: SP * H, qkv: SP * 3 * H, h: SP * H, u: SP * F, ctx: SP * H,
                qh: SP * DH, kT: DH * SP, vh: SP * DH, p: SP * SP, ch: SP * DH };
  for (const k in act) bytes += act[k] * 4;
  const mem = new WebAssembly.Memory({ initial: Math.ceil(bytes / 65536) + 1 });
  const candidates = Array.isArray(wasm) ? wasm : [wasm];
  const kernel = candidates.findIndex((c) => c instanceof WebAssembly.Module || WebAssembly.validate(c));
  if (kernel < 0) throw new Error("no kernel validates (WebAssembly SIMD unsupported?)");
  const c = candidates[kernel];
  const mod = c instanceof WebAssembly.Module ? c : await WebAssembly.compile(c);
  const { mm: mm16, mm4 } = (await WebAssembly.instantiate(mod, { env: { mem } })).exports;
  // 4 x 16 tile where N allows (every weight matmul); 4 x 4 for attention, where N = padded length.
  const mm = (x, w, y, b, S, K, N) => (N % 16 ? mm4 : mm16)(x, w, y, b, S, K, N);
  const f32 = new Float32Array(mem.buffer);

  let off = 0;
  const alloc = (n) => { const p = off; off += n * 4; return p; };
  const view = (p, n) => f32.subarray(p / 4, p / 4 + n);
  const ptr = {};
  for (const n of kernelTensors) {
    const t = W.get(n), size = sizeOf(n) / 4;
    ptr[n] = alloc(size);
    const dst = view(ptr[n], size);
    if (t.f32) dst.set(t.f32);
    else { // int8 [K][N], one scale per column
      const N = t.shape[1], s = t.scaleArr;
      const q = t.i8;
      for (let o = 0; o < size; o += N) for (let n = 0; n < N; n++) dst[o + n] = q[o + n] * s[n];
    }
  }
  for (const k in act) ptr[k] = alloc(act[k]);
  const a = Object.fromEntries(Object.keys(act).map((k) => [k, view(ptr[k], act[k])]));

  // Kept in JS: token table (gathered per token), positions, LayerNorm, the output Dense.
  const tokEmb = W.get("tok_emb"), pos = W.get("pos_emb").f32;
  const row = new Float32Array(H);
  const tokRow = (id) => {
    if (tokEmb.f32) return tokEmb.f32.subarray(id * H, id * H + H);
    if (tokEmb.i4) {
      const G = tokEmb.group, sc = tokEmb.scaleArr, o = id * (H / 2), so = id * (H / G);
      for (let j = 0; j < H; j += 2) {
        const b = tokEmb.i4[o + j / 2], s = sc[so + Math.floor(j / G)];
        row[j] = ((b & 15) - 8) * s;
        row[j + 1] = ((b >> 4) - 8) * s;
      }
      return row;
    }
    const s = tokEmb.scaleArr[id];
    for (let j = 0; j < H; j++) row[j] = tokEmb.i8[id * H + j] * s;
    return row;
  };
  const vec = (n) => W.get(n).f32;
  const lnp = { emb: [vec("emb_ln_g"), vec("emb_ln_b")] };
  for (let l = 0; l < L; l++) lnp[l] = [[vec(`l${l}.ln1_g`), vec(`l${l}.ln1_b`)], [vec(`l${l}.ln2_g`), vec(`l${l}.ln2_b`)]];
  const dense = W.get("dense_w"), denseB = vec("dense_b");
  let denseW = dense.f32;
  if (!denseW) {
    denseW = new Float32Array(H * D);
    for (let o = 0; o < H * D; o += D) for (let n = 0; n < D; n++) denseW[o + n] = dense.i8[o + n] * dense.scaleArr[n];
  }

  // y[s] = LayerNorm(x[s] + r[s]) for the first S rows (r optional), in f64 accumulators.
  function layerNorm(x, r, y, S, [g, b]) {
    for (let s = 0; s < S; s++) {
      const o = s * H;
      let mean = 0;
      for (let j = 0; j < H; j++) { const v = r ? x[o + j] + r[o + j] : x[o + j]; y[o + j] = v; mean += v; }
      mean /= H;
      let v2 = 0;
      for (let j = 0; j < H; j++) { const d = y[o + j] - mean; v2 += d * d; }
      const inv = 1 / Math.sqrt(v2 / H + eps);
      for (let j = 0; j < H; j++) y[o + j] = (y[o + j] - mean) * inv * g[j] + b[j];
    }
  }

  function forward(ids) {
    const S = ids.length, Sp = (S + 3) & ~3;
    const { x, qkv, h, u, ctx, qh, kT, vh, p, ch } = a;
    x.fill(0, 0, Sp * H);
    for (let s = 0; s < S; s++) {
      const t = tokRow(ids[s]);
      for (let j = 0; j < H; j++) x[s * H + j] = t[j] + pos[s * H + j];
    }
    layerNorm(x, null, x, S, lnp.emb);
    const scale = 1 / Math.sqrt(DH);
    for (let l = 0; l < L; l++) {
      mm(ptr.x, ptr[`l${l}.qkv_w`], ptr.qkv, ptr[`l${l}.qkv_b`], Sp, H, 3 * H);
      for (let hd = 0; hd < NH; hd++) {
        kT.fill(0, 0, DH * Sp); // padded key columns stay 0 (and are masked below)
        for (let s = 0; s < Sp; s++) {
          const o = s * 3 * H + hd * DH;
          for (let d = 0; d < DH; d++) {
            qh[s * DH + d] = qkv[o + d];
            if (s < S) { kT[d * Sp + s] = qkv[o + H + d]; vh[s * DH + d] = qkv[o + 2 * H + d]; }
            else vh[s * DH + d] = 0;
          }
        }
        mm(ptr.qh, ptr.kT, ptr.p, 0, Sp, DH, Sp);
        for (let s = 0; s < Sp; s++) {
          const o = s * Sp;
          let mx = -Infinity;
          for (let j = 0; j < S; j++) { p[o + j] *= scale; if (p[o + j] > mx) mx = p[o + j]; }
          let sum = 0;
          for (let j = 0; j < S; j++) { const e = Math.exp(p[o + j] - mx); p[o + j] = e; sum += e; }
          for (let j = 0; j < S; j++) p[o + j] /= sum;
          for (let j = S; j < Sp; j++) p[o + j] = 0;
        }
        mm(ptr.p, ptr.vh, ptr.ch, 0, Sp, Sp, DH);
        for (let s = 0; s < Sp; s++)
          for (let d = 0; d < DH; d++) ctx[s * H + hd * DH + d] = ch[s * DH + d];
      }
      mm(ptr.ctx, ptr[`l${l}.o_w`], ptr.h, ptr[`l${l}.o_b`], Sp, H, H);
      layerNorm(h, x, x, S, lnp[l][0]);
      mm(ptr.x, ptr[`l${l}.up_w`], ptr.u, ptr[`l${l}.up_b`], Sp, H, F);
      for (let i = 0, n = S * F; i < n; i++) { const v = u[i]; u[i] = v * (1 - 0.5 * erfc(v * Math.SQRT1_2)); }
      u.fill(0, S * F, Sp * F);
      mm(ptr.u, ptr[`l${l}.down_w`], ptr.h, ptr[`l${l}.down_b`], Sp, F, H);
      layerNorm(h, x, x, S, lnp[l][1]);
    }
    const pooled = new Float64Array(H);
    for (let s = 0; s < S; s++) for (let j = 0; j < H; j++) pooled[j] += x[s * H + j];
    const out = new Float32Array(D);
    for (let n = 0; n < D; n++) out[n] = denseB[n];
    for (let j = 0; j < H; j++) {
      const v = pooled[j] / S, o = j * D;
      for (let n = 0; n < D; n++) out[n] += v * denseW[o + n];
    }
    let norm = 0;
    for (let n = 0; n < D; n++) norm += out[n] * out[n];
    norm = Math.sqrt(norm);
    for (let n = 0; n < D; n++) out[n] /= norm;
    return out;
  }

  return {
    config: W.config,
    kernel,
    tokenize: (text) => tok.encode(text),
    forward,
    embed: (text, { query = false } = {}) => forward(tok.encode((query ? QUERY_PROMPT : "") + text)),
  };
}
