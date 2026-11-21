// GPU Bridge job: packed turbo vs transformers.js q4 turbo under the page's own settings (language given,
// timestamps on, 3-gram ban), same clips, both from raw 16 kHz audio. Times our spectrogram separately.
const { reps = 3, pack = "turbo-en-r15" } = ctx.inputs;
const clips = Object.fromEntries(Object.entries(ctx.inputs).filter(([k]) => k.startsWith("clip_")));
// `base` (a commit-pinned URL) gives every module, including the ones they import, a fresh address: the bridge
// worker caches modules by URL for its lifetime, so a plain URL can serve code from hours ago.
const SITE = ctx.inputs.base || "https://austegard.com/experiments/whisper-pack/web/", bust = ctx.inputs.base ? "" : "?t=" + Date.now();
const R2 = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/models/whisper-pack/";
const med = (a) => { const s = a.slice(1).sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const out = { ours: [], tjs: [], errors: [] };
{
  const { Whisper } = await ctx.importModule(SITE + "whisper.js" + bust);
  const { Tokenizer } = await ctx.importModule(SITE + "tokenizer.js" + bust);
  const { Transcriber } = await ctx.importModule(SITE + "transcribe.js" + bust);
  const { logMel } = await ctx.importModule(SITE + "mel.js" + bust);
  const device = await ctx.gpu();
  device.addEventListener("uncapturederror", (e) => out.errors.push(e.error.message));
  const w = await Whisper.load(device, R2 + pack + "/", { onProgress: (a, b) => ctx.progress(0.3 * a / b, "pack") });
  const tr = new Transcriber(w, await Tokenizer.load(R2 + pack + "/tokenizer.json"));
  for (const [name, pcm] of Object.entries(clips)) {
    const total = [], mel = [], enc = [];
    let res = null;
    for (let r = 0; r < reps + 1; r++) {
      let t = performance.now(); const m = logMel(pcm, w.dims.n_mels); mel.push(performance.now() - t);
      t = performance.now(); await w.encode(m); enc.push(performance.now() - t);
      t = performance.now(); res = await tr.transcribe(pcm, { language: "en", noRepeatNgram: 3 }); total.push(performance.now() - t);
    }
    out.ours.push({ clip: name, seconds: pcm.length / 16000, totalMs: med(total), melMs: med(mel), encodeMs: med(enc), text: res.text });
    ctx.log(JSON.stringify(out.ours.at(-1)));
  }
  w.destroy?.();
}
ctx.progress(0.5, "transformers.js");
{
  const { pipeline } = await ctx.importModule("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0");
  const asr = await pipeline("automatic-speech-recognition", "onnx-community/whisper-large-v3-turbo", { device: "webgpu", dtype: { encoder_model: "q4", decoder_model_merged: "q4" } });
  for (const [name, pcm] of Object.entries(clips)) {
    const total = [];
    let res = null;
    for (let r = 0; r < reps + 1; r++) {
      const t = performance.now();
      res = await asr(pcm, { language: "en", task: "transcribe", return_timestamps: true, no_repeat_ngram_size: 3 });
      total.push(performance.now() - t);
    }
    out.tjs.push({ clip: name, seconds: pcm.length / 16000, totalMs: med(total), text: res.text });
    ctx.log(JSON.stringify(out.tjs.at(-1)));
  }
}
return out;
