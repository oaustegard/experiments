// leaf-engine.js: mdbr-leaf-ir on WebGPU from leaf.js's int8 weights file, through ettin-native's
// SmallEngine and the shared kernels in _lib/wgsl/ (kernel library step 6). It replaced leaf-gpu.js,
// this experiment's own copy of the BERT kernels; leaf.js stays the reference.
//
//   const gpu = await createLeafEngine({ weights, vocab });          // ArrayBuffer (int8 file), string
//   const [v] = await gpu.embedBatch([text], { query: true });       // Float32Array(768), unit length
//   const vs = await gpu.forwardBatch(idsList);                       // token-id arrays in, vectors out
//
// leafManifest() turns the file into SmallEngine's input: a manifest of tensor byte ranges and the
// bytes. Most tensors pass through untouched. The fused q/k/v matrix [384][1152] is split by column
// into three [384][384] ones (SmallEngine's BERT path runs q, k and v as separate matmuls), and a
// zero row stands in for the token-type embedding, which the file already folds into the positions.

import { SmallEngine } from "../../ettin-native/web/engine.js";
import { WordPiece } from "./tokenizer.js";
import { QUERY_PROMPT } from "./leaf.js";

const MAX_TOKENS = 128;

export function leafManifest(buf) {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  if (String.fromCharCode(...u8.subarray(0, 4)) !== "LEAF") throw new Error("not a LEAF weights file");
  const hlen = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + hlen)));
  const base = Math.ceil((12 + hlen) / 16) * 16, t = header.tensors, c = header.config;
  for (const [name, x] of Object.entries(t))
    if (/_w$|^tok_emb$/.test(name) && x.dtype !== "i8") throw new Error(`${name}: needs the int8 file (export without t4)`);
  const size = (x) => x.shape.reduce((a, b) => a * b, 1) * (x.dtype === "f32" ? 4 : 1);
  // Everything the manifest points at goes into `parts`, appended in order, 4-byte aligned.
  const parts = [];
  let off = 0;
  const put = (bytes) => { const r = { offset: off, length: bytes.byteLength }; parts.push(bytes); off += Math.ceil(bytes.byteLength / 4) * 4; parts.push(new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4 - bytes.byteLength)); return r; };
  const raw = (name) => put(u8.subarray(base + t[name].offset, base + t[name].offset + size(t[name])));
  const scale = (name) => put(u8.subarray(base + t[name].scale, base + t[name].scale + t[name].shape[t[name].axis] * 4));
  // int8 [K][N] with per-column scale and f32 bias; cols = [c0, c1) takes a column slice.
  const lin = (w, b, cols) => {
    const [K, N] = t[w].shape;
    if (!cols) return { q: raw(w), s: scale(w), K, N, b: raw(b) };
    const [c0, c1] = cols, n = c1 - c0, src = u8.subarray(base + t[w].offset), q = new Uint8Array(K * n);
    for (let k = 0; k < K; k++) q.set(src.subarray(k * N + c0, k * N + c1), k * n);
    const sAll = new Float32Array(u8.slice(base + t[w].scale, base + t[w].scale + N * 4).buffer), bAll = new Float32Array(u8.slice(base + t[b].offset, base + t[b].offset + N * 4).buffer);
    return { q: put(q), s: put(new Uint8Array(sAll.slice(c0, c1).buffer)), K, N: n, b: put(new Uint8Array(bAll.slice(c0, c1).buffer)) };
  };
  const H = c.hidden, ln = (g, b) => ({ w: raw(g), b: raw(b) });
  const m = {
    arch: "bert", dim: c.out_dim, hidden: H, heads: c.heads, inter: c.ffn, eps: c.eps,
    table: { bits: 8, V: t.tok_emb.shape[0], H, codes: raw("tok_emb"), scales: scale("tok_emb") },
    pos: raw("pos_emb"), type0: put(new Uint8Array(H * 4)), emb_ln: ln("emb_ln_g", "emb_ln_b"),
    layers: Array.from({ length: c.layers }, (_, l) => ({
      q: lin(`l${l}.qkv_w`, `l${l}.qkv_b`, [0, H]), k: lin(`l${l}.qkv_w`, `l${l}.qkv_b`, [H, 2 * H]), v: lin(`l${l}.qkv_w`, `l${l}.qkv_b`, [2 * H, 3 * H]),
      o: lin(`l${l}.o_w`, `l${l}.o_b`), ln1: ln(`l${l}.ln1_g`, `l${l}.ln1_b`),
      up: lin(`l${l}.up_w`, `l${l}.up_b`), down: lin(`l${l}.down_w`, `l${l}.down_b`), ln2: ln(`l${l}.ln2_g`, `l${l}.ln2_b`),
    })),
    dense: lin("dense_w", "dense_b"),
  };
  const bytes = new Uint8Array(off);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.byteLength; }
  return { manifest: m, bytes, config: c };
}

export async function createLeafEngine({ weights, vocab, device, maxRows = 4096 }) {
  const { manifest, bytes, config } = leafManifest(weights);
  const tok = new WordPiece(vocab, MAX_TOKENS);
  if (!device) {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU unavailable");
    device = await adapter.requestDevice();
  }
  const maxSeqs = Math.min(256, maxRows);
  const eng = await SmallEngine.create(device, manifest, bytes, { maxTokens: maxRows, maxSeqs });
  const D = eng.dim;
  // Any number of sequences: packed back to back in batches of <= maxRows tokens and maxSeqs sequences.
  async function forwardBatch(idsList) {
    const out = [];
    for (let i = 0; i < idsList.length;) {
      let j = i, rows = 0;
      while (j < idsList.length && j - i < maxSeqs && rows + idsList[j].length <= maxRows) rows += idsList[j++].length;
      if (j === i) throw new Error(`sequence of ${idsList[i].length} tokens exceeds maxRows ${maxRows}`);
      const flat = await eng.embed(idsList.slice(i, j));
      for (let b = 0; b < j - i; b++) out.push(flat.slice(b * D, b * D + D));
      i = j;
    }
    return out;
  }
  return {
    device, config, engine: eng,
    tokenize: (text) => tok.encode(text),
    forwardBatch,
    embedBatch: (texts, { query = false } = {}) => forwardBatch(texts.map((t) => tok.encode((query ? QUERY_PROMPT : "") + t))),
    destroy: () => eng.destroy(),
  };
}
