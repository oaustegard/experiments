// linkup-sparseup-embed-v2 in the browser: tokenizer + WGSL engine (engine.js) on a local pack directory
// written by pack.py (manifest.json, weights.bin, tokenizer.json).
//
//   const sp = await loadSparseUp(PACK_URL);   // a directory with manifest.json, weights, tokenizer.json
//   const [q] = await sp.encode(["what is a cat"], "query");       // [{ id: weight }] per text, folded
//   sp.token(id)                                                    // "cat" (word-start marker stripped)
//   sp.score(q, d)                                                  // dot product of two sparse maps
//
// Tokenization follows the model's _tokenize: a leading space on the text, the "[Q]" / "[D]" prefix token
// (one added token, attended but not pooled), CLS ... SEP, truncation to maxSeq tokens keeping SEP.

import { SparseEngine, toSparse } from "./engine.js";

const TOKENIZERS = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";

export async function loadSparseUp(base, opts = {}) {
  base = new URL(base.endsWith("/") ? base : base + "/", location.href);
  const t0 = performance.now();
  const bytesOf = (name) => fetch(new URL(name, base)).then((r) => {
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return r.arrayBuffer();
  });
  const manifest = await fetch(new URL("manifest.json", base)).then((r) => r.json());
  // A pack is weights.bin, or weights.0.bin ... listed in manifest.weights_parts (hosted packs, < 50 MB per file).
  const names = manifest.weights_parts ?? ["weights.bin"];
  const [tj, parts, { Tokenizer }] = await Promise.all([
    fetch(new URL("tokenizer.json", base)).then((r) => r.json()),
    Promise.all(names.map(bytesOf)),
    import(TOKENIZERS)]);
  const data = new Uint8Array(parts.reduce((s, x) => s + x.byteLength, 0));
  parts.reduce((o, x) => { data.set(new Uint8Array(x), o); return o + x.byteLength; }, 0);
  const fetchMs = performance.now() - t0;
  const tok = new Tokenizer(tj, {});
  const vocab = [];
  for (const [s, id] of Object.entries(tj.model.vocab)) vocab[id] = s;
  for (const a of tj.added_tokens) vocab[a.id] = a.content;
  const maxSeq = opts.maxSeq ?? 512;

  const tokenize = (text, kind = "document") => {
    const prefix = kind === "query" ? manifest.query_prefix : kind === "document" ? manifest.document_prefix : "";
    if (manifest.add_prefix_space && !text.startsWith(" ")) text = " " + text;
    let ids = Array.from(tok.encode(prefix + text).ids);
    if (ids.length > maxSeq) ids = ids.slice(0, maxSeq - 1).concat(ids[ids.length - 1]);
    const pool = ids.map((_, j) => (prefix && j === 1 ? 0 : 1));
    return { ids, pool };
  };

  const adapter = opts.adapter ?? await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const L = adapter.limits;
  const device = opts.device ?? await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
  const t1 = performance.now();
  const maxTokens = opts.maxTokens ?? 512, maxSeqs = opts.maxSeqs ?? 16;
  const engine = await SparseEngine.create(device, manifest, data, { maxTokens, maxSeq, maxSeqs });
  const initMs = performance.now() - t1;
  const V = manifest.vpad;

  // Run tokenized inputs in batches that fit the engine; returns Float32Array rows (dense, folded).
  const run = async (items) => {
    const rows = [];
    for (let i = 0; i < items.length;) {
      const batch = [];
      let n = 0;
      while (i < items.length && batch.length < maxSeqs && n + items[i].ids.length <= maxTokens) { n += items[i].ids.length; batch.push(items[i++]); }
      const out = await engine.embed(batch);
      for (let b = 0; b < batch.length; b++) rows.push(out.subarray(b * V, (b + 1) * V));
    }
    return rows;
  };
  const encodeIds = async (items, topK = 0) => (await run(items)).map((r) => toSparse(r, V, 1, topK)[0]);
  const encode = (texts, kind = "document", topK = 0) => encodeIds(texts.map((t) => tokenize(t, kind)), topK);
  const token = (id) => (vocab[id] ?? `#${id}`).replace(/^Ġ/, "");
  const score = (a, b) => { let s = 0; for (const k in a) if (k in b) s += a[k] * b[k]; return s; };
  return { manifest, engine, device, tokenize, run, encode, encodeIds, token, score,
           info: { fetchMs, initMs, bytes: data.byteLength, adapter: adapter.info ?? {} } };
}
