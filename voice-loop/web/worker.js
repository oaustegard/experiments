// Runs the language model off the main thread so the page stays responsive while it generates.
import { pipeline, TextStreamer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";

env.allowLocalModels = false;

let generator = null;
let loadedKey = null;

async function pickDevice() {
  if (!("gpu" in navigator)) return { device: "wasm", dtype: "q8", note: "no WebGPU" };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { device: "wasm", dtype: "q8", note: "no WebGPU adapter" };
    return adapter.features.has("shader-f16")
      ? { device: "webgpu", dtype: "q4f16", note: "WebGPU, f16" }
      : { device: "webgpu", dtype: "q4", note: "WebGPU, no f16" };
  } catch (e) {
    return { device: "wasm", dtype: "q8", note: "WebGPU failed: " + e.message };
  }
}

async function load(model, forceWasm) {
  const pick = forceWasm ? { device: "wasm", dtype: "q8", note: "wasm (forced)" } : await pickDevice();
  const key = model + "|" + pick.device + "|" + pick.dtype;
  if (generator && loadedKey === key) return pick;
  if (generator) { try { await generator.dispose(); } catch {} generator = null; }
  const files = {};
  generator = await pipeline("text-generation", model, {
    device: pick.device,
    dtype: pick.dtype,
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        files[p.file] = [p.loaded, p.total];
        let done = 0, total = 0;
        for (const [l, t] of Object.values(files)) { done += l; total += t; }
        postMessage({ type: "progress", done, total, file: p.file });
      }
    },
  });
  loadedKey = key;
  return pick;
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const t0 = performance.now();
      const pick = await load(data.model, data.forceWasm);
      // One token of warm-up compiles the GPU shaders before the first real question.
      await generator([{ role: "user", content: "Hi" }], { max_new_tokens: 1 });
      postMessage({ type: "ready", ...pick, seconds: (performance.now() - t0) / 1000 });
    } else if (data.type === "generate") {
      const t0 = performance.now();
      let first = 0, count = 0;
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text) => {
          if (!first) first = performance.now();
          count++;
          postMessage({ type: "token", text });
        },
      });
      await generator(data.messages, {
        max_new_tokens: data.maxTokens,
        do_sample: true, temperature: 0.6, top_p: 0.9, repetition_penalty: 1.1,
        streamer,
      });
      const end = performance.now();
      postMessage({
        type: "done",
        firstMs: first ? first - t0 : null,
        tokPerSec: first && count > 1 ? (count - 1) / ((end - first) / 1000) : null,
      });
    }
  } catch (e) {
    postMessage({ type: "error", message: e?.message || String(e) });
  }
};
