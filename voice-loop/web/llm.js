// Chat on the WGSL LFM2.5 engine: weights streamed from Hugging Face (or the browser cache) straight
// to the GPU, the tokenizer and chat template from transformers.js, sampling in JS.
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";
import { LfmEngine, MAX_CTX } from "../../lfm-wgsl/web/lfm.js";

env.allowLocalModels = false;
const REPO = "onnx-community/LFM2.5-350M-ONNX";
const WEIGHTS = `https://huggingface.co/${REPO}/resolve/main/onnx/model_q4.onnx_data`;
const CACHE = "lfm-wgsl-weights-v1";

// The weight response from the Cache API, or from the network with a copy teed into the cache.
async function weights(onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch {}
  const hit = cache && await cache.match(WEIGHTS).catch(() => null);
  if (hit) return { res: hit, cached: true };
  const res = await fetch(WEIGHTS);
  if (!res.ok) throw new Error(`weights: HTTP ${res.status}`);
  if (!cache) return { res, cached: false };
  const [a, b] = res.body.tee();
  cache.put(WEIGHTS, new Response(b, { headers: res.headers })).catch(() => {});
  return { res: new Response(a, { headers: res.headers }), cached: false };
}

export class Chat {
  static async load(onProgress) {
    if (!navigator.gpu) throw new Error("this browser has no WebGPU");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const L = adapter.limits;
    const device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
    const base = new URL("../../lfm-wgsl/web/", import.meta.url);
    const [manifest, tokenizer] = await Promise.all([
      fetch(new URL("manifest.json", base)).then((r) => r.json()),
      AutoTokenizer.from_pretrained(REPO),
    ]);
    try { await navigator.storage?.persist?.(); } catch {}
    const { res, cached } = await weights();
    const eng = await LfmEngine.create(device, manifest, WEIGHTS, { source: res, onProgress });
    const c = new Chat(eng, tokenizer, manifest);
    c.cached = cached;
    device.lost.then((i) => { c.lost = i.message || "GPU device lost"; });
    return c;
  }

  constructor(eng, tok, m) { this.eng = eng; this.tok = tok; this.m = m; }

  // Generate a reply to `messages`; onText(text) gets the whole reply so far. Sampling follows
  // Liquid's recommendation for LFM2: temperature 0.3, min_p 0.15, repetition penalty 1.05.
  async generate(messages, { maxTokens = 160, onText, signal, temperature = 0.3, minP = 0.15, repPenalty = 1.05 } = {}) {
    if (this.lost) throw new Error(this.lost);
    let ids = this.tok.apply_chat_template(messages, { add_generation_prompt: true, tokenize: true, return_tensor: false });
    ids = Array.from(ids, Number);
    if (ids.length > MAX_CTX - maxTokens) throw new Error("conversation too long; tap Stop and start over");
    const eos = new Set([this.m.eos, 2]);
    const out = [];
    const t0 = performance.now();
    this.eng.reset();
    let logits = await this.eng.forward(ids);
    const tFirst = performance.now();
    let text = "";
    while (out.length < maxTokens && !signal?.aborted) {
      const next = sample(logits, out, { temperature, minP, repPenalty });
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

function sample(logits, prev, { temperature, minP, repPenalty }) {
  const l = logits;
  for (const id of new Set(prev)) l[id] = l[id] > 0 ? l[id] / repPenalty : l[id] * repPenalty;
  let mx = -Infinity, arg = 0;
  for (let i = 0; i < l.length; i++) if (l[i] > mx) { mx = l[i]; arg = i; }
  if (temperature <= 0) return arg;
  // min_p: keep tokens whose probability is at least minP times the top token's.
  const cut = mx + temperature * Math.log(minP);
  const keep = [], w = [];
  let sum = 0;
  for (let i = 0; i < l.length; i++) if (l[i] >= cut) { const e = Math.exp((l[i] - mx) / temperature); keep.push(i); w.push(e); sum += e; }
  let r = Math.random() * sum;
  for (let j = 0; j < keep.length; j++) { r -= w[j]; if (r <= 0) return keep[j]; }
  return arg;
}
