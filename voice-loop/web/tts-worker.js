// Supertonic 2 text-to-speech (onnx-community/Supertonic-TTS-2-ONNX, ~260 MB) on onnxruntime-web wasm,
// off the main thread. Messages: {type:"load"} -> {type:"ready"}; {type:"say", id, text, voice, steps}
// -> {type:"audio", id, audio: Float32Array, rate, ms}.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js";

env.allowLocalModels = false;
const REPO = "onnx-community/Supertonic-TTS-2-ONNX";
let tts = null;
const voices = {};

async function voice(name) {
  voices[name] ??= `https://huggingface.co/${REPO}/resolve/main/voices/${name}.bin`;
  return voices[name];
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const files = {};
      tts ??= await pipeline("text-to-speech", REPO, {
        device: "wasm",
        progress_callback: (p) => {
          if (p.status === "progress" && p.total) {
            files[p.file] = [p.loaded, p.total];
            let d = 0, t = 0; for (const [a, b] of Object.values(files)) { d += a; t += b; }
            postMessage({ type: "progress", done: d, total: t });
          }
        },
      });
      postMessage({ type: "ready" });
    } else if (data.type === "say") {
      const t0 = performance.now();
      const out = await tts(`<en>${data.text}</en>`, {
        speaker_embeddings: await voice(data.voice || "F1"),
        num_inference_steps: data.steps || 5,
        // 0.9, not the model card's 1.05: the predicted duration is too short at 1.05 and words
        // get squeezed out (Whisper transcripts in voice-loop/README.md).
        speed: data.speed || 0.9,
      });
      postMessage({ type: "audio", id: data.id, audio: out.audio, rate: out.sampling_rate, ms: performance.now() - t0 }, [out.audio.buffer]);
    }
  } catch (e) {
    postMessage({ type: "error", id: data.id, message: e?.message || String(e) });
  }
};
