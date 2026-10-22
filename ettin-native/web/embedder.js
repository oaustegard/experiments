// Query embedder for a web/models/<pkg>/ package: the WGSL engine (engine.js) on WebGPU, with
// onnxruntime-web on the same weights file as the fallback (the Bluesky image search pattern).
// Both paths read the same single-file model.onnx: the engine through manifest.json's byte offsets,
// onnxruntime-web directly (single-file, so it keeps its fused int8 matmuls). The engine path adds
// ~25 KB of JS and no runtime; the fallback adds onnxruntime-web (3.07 MB brotli), never the weights again.
//
//   const emb = await loadEmbedder("models/e17c-t4-w8L63/");          // { backend: "webgpu" | "ort-wasm", ... }
//   const vecs = await emb.embed(["claude coding agent costs"]);      // Float32Array[n * 768], unit rows
//
// ?engine=ort on the page URL (or opts.backend = "ort") forces the fallback.

import { SmallEngine } from "./engine.js";

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const TOKENIZERS = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
const MAX_TOKENS = 128;

export async function loadEmbedder(base, opts = {}) {
  base = new URL(base.endsWith("/") ? base : base + "/", location.href);
  const backend = opts.backend ?? new URLSearchParams(location.search).get("engine") ?? "auto";
  const t0 = performance.now();
  const [manifest, meta, tj, data, { Tokenizer }] = await Promise.all([
    fetch(new URL("manifest.json", base)).then((r) => r.json()),
    fetch(new URL("meta.json", base)).then((r) => r.json()),
    fetch(new URL("tokenizer.json", base)).then((r) => r.json()),
    fetch(new URL("model.onnx", base)).then((r) => r.arrayBuffer()),
    import(TOKENIZERS)]);
  const fetchMs = performance.now() - t0;
  const tok = new Tokenizer(tj, {});
  const qp = (meta.prompts || {}).query ?? "";
  // Same truncation as the Python side: keep the first 127 ids and the closing special token.
  const encode = (text, prompt = qp) => {
    const ids = tok.encode(prompt + text).ids;
    return ids.length > MAX_TOKENS ? ids.slice(0, MAX_TOKENS - 1).concat(ids[ids.length - 1]) : ids;
  };
  const info = { manifest, fetchMs, dataBytes: data.byteLength };

  if (backend !== "ort") {
    try {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("no WebGPU adapter");
      const device = opts.device ?? await adapter.requestDevice();
      const t1 = performance.now();
      const engine = await SmallEngine.create(device, manifest, data, { maxTokens: opts.maxTokens ?? 512, maxSeqs: opts.maxSeqs ?? 64 });
      info.initMs = performance.now() - t1;
      return { backend: "webgpu", info, encode, engine,
               embedIds: (batch) => engine.embed(batch),
               embed: (texts) => engine.embed(texts.map((t) => encode(t))) };
    } catch (e) {
      if (backend === "webgpu") throw e;
      console.warn("WGSL engine unavailable, using onnxruntime-web:", e);
      info.engineError = String(e);
    }
  }
  // Fallback: onnxruntime-web (wasm) on the same model.onnx bytes.
  const t1 = performance.now();
  const ort = await import(ORT + "ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = opts.threads ?? 1;
  // graphOptimizationLevel "all" fuses DequantizeLinear+MatMul into int8 kernels: ~4x faster on wasm,
  // vectors within cosine ~0.9998 of the exact graph (the engine computes the exact graph).
  const session = await ort.InferenceSession.create(new Uint8Array(data), { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  info.initMs = performance.now() - t1;
  const out = session.outputNames.includes("sentence_embedding") ? "sentence_embedding" : session.outputNames[0];
  const embedIds = async (batch) => {
    const res = new Float32Array(batch.length * 768);
    for (let b = 0; b < batch.length; b++) {
      const ids = batch[b], S = ids.length;
      const r = await session.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, S]),
                                    attention_mask: new ort.Tensor("int64", new BigInt64Array(S).fill(1n), [1, S]) });
      const v = r[out].data;
      let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
      for (let i = 0; i < 768; i++) res[b * 768 + i] = v[i] / n;
    }
    return res;
  };
  return { backend: "ort-wasm", info, encode, session, embedIds, embed: (texts) => embedIds(texts.map((t) => encode(t))) };
}
