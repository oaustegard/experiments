// Module worker that speaks the same messages as anything-to-text's transformers.js worker, on the
// packed WGSL engine.
//
//   in:  { type: "load", model: <pack directory URL> }
//        { type: "transcribe", audio: Float32Array (16 kHz mono), language, task }
//   out: { type: "download", payload: { status: "progress", file, loaded, total } }
//        { type: "ready", model, cached }   { type: "partial", text }
//        { type: "progress", frac, windowsDone, totalWindows }
//        { type: "complete", text, chunks: [{ timestamp: [s, e], text }], elapsed, language }
//        { type: "log", message }           { type: "error", message }

import { Whisper } from "./whisper.js";
import { Tokenizer } from "./tokenizer.js";
import { Transcriber } from "./transcribe.js";

let device = null, transcriber = null, loaded = null;

async function gpu() {
  if (device) return device;
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  device = await adapter.requestDevice();
  device.lost.then((i) => { device = null; transcriber = null; loaded = null; self.postMessage({ type: "log", message: "GPU device lost: " + i.message }); });
  device.addEventListener("uncapturederror", (e) => self.postMessage({ type: "log", message: "GPU error: " + e.error.message }));
  return device;
}

self.onmessage = async (event) => {
  const msg = event.data;
  try {
    if (msg.type === "load") {
      if (transcriber && loaded === msg.model) { self.postMessage({ type: "ready", model: msg.model, cached: true }); return; }
      transcriber = null; loaded = null;
      const dev = await gpu();
      const base = msg.model.endsWith("/") ? msg.model : msg.model + "/";
      const t0 = performance.now();
      const [whisper, tok] = await Promise.all([
        Whisper.load(dev, base, { onProgress: (a, b) => self.postMessage({ type: "download", payload: { status: "progress", file: "weights", loaded: a, total: b } }) }),
        Tokenizer.load(base + "tokenizer.json"),
      ]);
      transcriber = new Transcriber(whisper, tok);
      loaded = msg.model;
      self.postMessage({ type: "log", message: `pack ${whisper.m.model} (${whisper.m.scheme}) ready in ${((performance.now() - t0) / 1000).toFixed(1)} s` });
      self.postMessage({ type: "ready", model: msg.model, cached: false });
      return;
    }
    if (msg.type === "transcribe") {
      if (!transcriber) { self.postMessage({ type: "error", message: "Model not loaded" }); return; }
      const t0 = performance.now();
      const multilingual = transcriber.g.multilingual;
      const language = multilingual ? (msg.language || "auto") : "en";
      const out = await transcriber.transcribe(msg.audio, {
        language, task: msg.task === "translate" && multilingual ? "translate" : "transcribe", noRepeatNgram: 3,
        onPartial: (text) => self.postMessage({ type: "partial", text }),
        onProgress: (frac, windows) => self.postMessage({ type: "progress", frac, windowsDone: windows, totalWindows: Math.max(windows, Math.ceil(msg.audio.length / 480000)) }),
      });
      self.postMessage({ type: "complete", text: out.text, chunks: out.chunks, elapsed: (performance.now() - t0) / 1000, language: out.language });
      return;
    }
  } catch (err) {
    self.postMessage({ type: "error", message: err && err.message ? err.message : String(err) });
  }
};
