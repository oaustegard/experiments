// Chat on the hand-written WGSL engines: weights streamed (or read from the Cache API) straight to the
// GPU, the tokenizer and chat template from transformers.js, sampling in JS. Two models:
//   lfm25: LFM2.5-350M, onnx-community's q4 file from Hugging Face (lfm-wgsl/web/lfm.js), ~230 MB
//   lfm3:  LFM3-1.7B, rotated-GPTQ 4-bit (lfm-wgsl/web/lfm3.js), ~1 GB in 15 parts on R2
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";
import { LfmEngine, MAX_CTX } from "../../lfm-wgsl/web/lfm.js";
import { Lfm3Engine, partsSource } from "../../lfm-wgsl/web/lfm3.js";

env.allowLocalModels = false;
const LFM25_WEIGHTS = "https://huggingface.co/onnx-community/LFM2.5-350M-ONNX/resolve/main/onnx/model_q4.onnx_data";
const CACHE = "lfm-wgsl-weights-v1";

export const MODELS = {
  lfm25: { label: "LFM2.5 350M", tokenizer: "onnx-community/LFM2.5-350M-ONNX", manifest: "manifest.json",
    // Liquid's recommendation for LFM2: temperature 0.3, min_p 0.15, repetition penalty 1.05
    sampling: { temperature: 0.3, minP: 0.15, repPenalty: 1.05 }, template: {}, eos: [2] },
  lfm3: { label: "LFM3 1.7B", tokenizer: "onnx-community/LFM3-1.7B-ONNX", manifest: "manifest3.json",
    // the release's generation_config: temperature 0.1, top_k 50, repetition penalty 1.1
    sampling: { temperature: 0.1, topK: 50, repPenalty: 1.1 }, template: { enable_thinking: false }, eos: [] },
};

async function openCache() { try { return await caches.open(CACHE); } catch { return null; } }

// A response from the Cache API, or from the network with a copy teed into the cache.
async function cachedFetch(cache, url) {
  const hit = cache && await cache.match(url).catch(() => null);
  if (hit) return { res: hit, cached: true };
  const res = await fetch(url);
  if (!res.ok) throw new Error(`weights: HTTP ${res.status}`);
  if (!cache) return { res, cached: false };
  const [a, b] = res.body.tee();
  cache.put(url, new Response(b, { headers: res.headers })).catch(() => {});
  return { res: new Response(a, { headers: res.headers }), cached: false };
}

export class Chat {
  static async load(model, onProgress) {
    const spec = MODELS[model];
    if (!spec) throw new Error(`unknown model ${model}`);
    if (!navigator.gpu) throw new Error("this browser has no WebGPU");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const L = adapter.limits;
    const device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
    const base = new URL("../../lfm-wgsl/web/", import.meta.url);
    const [manifest, tokenizer] = await Promise.all([
      fetch(new URL(spec.manifest, base)).then((r) => r.json()),
      AutoTokenizer.from_pretrained(spec.tokenizer),
    ]);
    if (model === "lfm3") {
      const largest = Math.max(...manifest.buffers.map((b) => b.size));
      if (L.maxStorageBufferBindingSize < largest) throw new Error(`this GPU binds at most ${(L.maxStorageBufferBindingSize / 1e6).toFixed(0)} MB per buffer; LFM3 needs ${(largest / 1e6).toFixed(0)} MB`);
    }
    try { await navigator.storage?.persist?.(); } catch {}
    const cache = await openCache();
    let eng, cached;
    if (model === "lfm3") {
      let hits = 0;
      const src = partsSource(manifest.parts, async (u) => { const r = await cachedFetch(cache, u); hits += r.cached; return r.res; });
      eng = await Lfm3Engine.create(device, manifest, null, { source: src, onProgress });
      cached = hits === manifest.parts.length;
    } else {
      const r = await cachedFetch(cache, LFM25_WEIGHTS);
      eng = await LfmEngine.create(device, manifest, LFM25_WEIGHTS, { source: r.res, onProgress });
      cached = r.cached;
    }
    const c = new Chat(eng, tokenizer, manifest, spec);
    c.cached = cached;
    device.lost.then((i) => { c.lost = i.message || "GPU device lost"; });
    return c;
  }

  constructor(eng, tok, m, spec) { this.eng = eng; this.tok = tok; this.m = m; this.spec = spec; }

  // Generate a reply to `messages`; onText(text) gets the whole reply so far.
  async generate(messages, { maxTokens = 160, onText, signal } = {}) {
    if (this.lost) throw new Error(this.lost);
    // transformers.js 4 returns { input_ids, attention_mask } here, 3.x a bare array.
    const enc = this.tok.apply_chat_template(messages, { add_generation_prompt: true, tokenize: true, return_tensor: false, ...this.spec.template });
    const ids = Array.from(enc.input_ids ?? enc, Number);
    if (!ids.length || ids.some((x) => !Number.isInteger(x))) throw new Error("chat template produced no token ids");
    if (ids.length > MAX_CTX - maxTokens) throw new Error("conversation too long; tap Stop and start over");
    const eos = new Set([this.m.eos, this.tok.eos_token_id, ...this.spec.eos].filter((x) => x != null));
    const out = [];
    const t0 = performance.now();
    this.eng.reset();
    let logits = await this.eng.forward(ids);
    const tFirst = performance.now();
    let text = "";
    while (out.length < maxTokens && !signal?.aborted) {
      const next = sample(logits, out, this.spec.sampling);
      if (eos.has(next)) break;
      out.push(next);
      const t = this.tok.decode(out, { skip_special_tokens: true });
      if (t !== text) { text = t; onText?.(text); }
      logits = await this.eng.forward([next]);
    }
    const t1 = performance.now();
    return { text, promptTokens: ids.length, tokens: out.length,
      prefillMs: tFirst - t0, tokPerSec: out.length > 1 ? (out.length - 1) / ((t1 - tFirst) / 1000) : null };
  }
}

function sample(logits, prev, { temperature, minP = 0, topK = 0, repPenalty = 1 }) {
  const l = logits;
  for (const id of new Set(prev)) l[id] = l[id] > 0 ? l[id] / repPenalty : l[id] * repPenalty;
  let mx = -Infinity, arg = 0;
  for (let i = 0; i < l.length; i++) if (l[i] > mx) { mx = l[i]; arg = i; }
  if (temperature <= 0) return arg;
  // min_p: keep tokens whose probability is at least minP times the top token's; top_k: the k largest.
  let cut = minP > 0 ? mx + temperature * Math.log(minP) : -Infinity;
  if (topK > 0) {
    // the k-th largest logit, from the candidates within 30 temperatures of the top (below that a
    // token's weight is under e^-30 and cannot matter)
    const near = [];
    for (let i = 0; i < l.length; i++) if (l[i] >= mx - 30 * temperature) near.push(l[i]);
    near.sort((a, b) => b - a);
    cut = Math.max(cut, near[Math.min(topK, near.length) - 1]);
  }
  const keep = [], w = [];
  let sum = 0;
  for (let i = 0; i < l.length; i++) if (l[i] >= cut) { const e = Math.exp((l[i] - mx) / temperature); keep.push(i); w.push(e); sum += e; }
  let r = Math.random() * sum;
  for (let j = 0; j < keep.length; j++) { r -= w[j]; if (r <= 0) return keep[j]; }
  return arg;
}
