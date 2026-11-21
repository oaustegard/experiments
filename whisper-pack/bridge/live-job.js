// GPU Bridge job: the deployed modules and hosted packs, end to end from PCM (spectrogram, encode, decode with
// timestamps), on the clip the transformers.js baseline used.
// inputs: pcm (f32, 16 kHz), packs = [names under the R2 prefix], reps.
const { pcm, packs, reps = 3 } = ctx.inputs;
const SITE = "https://austegard.com/experiments/whisper-pack/web/";
const R2 = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/models/whisper-pack/";
const bust = "?t=" + Date.now();
const { Whisper } = await ctx.importModule(SITE + "whisper.js" + bust);
const { Tokenizer } = await ctx.importModule(SITE + "tokenizer.js" + bust);
const { Transcriber } = await ctx.importModule(SITE + "transcribe.js" + bust);
const device = await ctx.gpu();
const errors = [];
device.addEventListener("uncapturederror", (e) => errors.push(e.error.message));
const rows = [];
for (const [i, name] of packs.entries()) {
  let t = performance.now();
  const w = await Whisper.load(device, R2 + name + "/", { onProgress: (a, b) => ctx.progress((i + 0.6 * a / b) / packs.length, name) });
  const tok = await Tokenizer.load(R2 + name + "/tokenizer.json");
  const loadMs = performance.now() - t;
  const tr = new Transcriber(w, tok);
  const runs = [];
  let out = null;
  for (let r = 0; r < reps + 1; r++) { t = performance.now(); out = await tr.transcribe(pcm, { language: "auto" }); runs.push(performance.now() - t); }
  rows.push({ pack: name, loadMs, runMs: runs, text: out.text, chunks: out.chunks, language: out.language });
  ctx.log(JSON.stringify(rows[rows.length - 1]));
}
return { rows, errors, seconds: pcm.length / 16000 };
