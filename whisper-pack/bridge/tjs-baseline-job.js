// GPU Bridge job: the page's current setup (transformers.js, onnx-community q4 turbo on WebGPU) on the same clip.
// inputs: pcm (f32, 16 kHz), model, dtype, reps.
const { pcm, model = "onnx-community/whisper-large-v3-turbo", dtype = { encoder_model: "q4", decoder_model_merged: "q4" }, reps = 3, version = "4.2.0" } = ctx.inputs;
const { pipeline } = await ctx.importModule(`https://cdn.jsdelivr.net/npm/@huggingface/transformers@${version}`);
let t = performance.now();
const asr = await pipeline("automatic-speech-recognition", model, { device: "webgpu", dtype,
  progress_callback: (p) => { if (p.status === "progress" && p.total) ctx.progress(Math.min(0.8, 0.8 * p.loaded / p.total), "download " + (p.file || "")); } });
const loadMs = performance.now() - t;
const runs = [];
let text = "";
for (let r = 0; r < reps + 1; r++) {
  t = performance.now();
  const o = await asr(pcm, { language: "en", task: "transcribe", return_timestamps: false });
  runs.push(performance.now() - t);
  text = o.text;
  ctx.progress(0.8 + 0.2 * (r + 1) / (reps + 1), "transcribe");
}
return { model, dtype, version, loadMs, runMs: runs, text, seconds: pcm.length / 16000 };
