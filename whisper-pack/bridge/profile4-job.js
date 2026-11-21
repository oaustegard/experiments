// GPU Bridge job: transcribe()'s own phase timings, plus logMel timed alone right before and right after.
const { clip_7s: pcm, reps = 4 } = ctx.inputs;
const SITE = "https://austegard.com/experiments/whisper-pack/web/", bust = "?t=" + Date.now();
const R2 = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/models/whisper-pack/turbo-en-r15/";
const { Whisper } = await ctx.importModule(SITE + "whisper.js" + bust);
const { Tokenizer } = await ctx.importModule(SITE + "tokenizer.js" + bust);
const { Transcriber } = await ctx.importModule(SITE + "transcribe.js" + bust);
const { logMel } = await ctx.importModule(SITE + "mel.js" + bust);
const device = await ctx.gpu();
const w = await Whisper.load(device, R2);
const tr = new Transcriber(w, await Tokenizer.load(R2 + "tokenizer.json"));
const rows = [], r1 = (x) => Math.round(x);
for (let r = 0; r < reps + 1; r++) {
  let t = performance.now(); logMel(pcm, w.dims.n_mels); const melBefore = performance.now() - t;
  const timings = {};
  await tr.transcribe(pcm, { language: "en", noRepeatNgram: 3, timings });
  t = performance.now(); logMel(pcm, w.dims.n_mels); const melAfter = performance.now() - t;
  const row = { melBefore: r1(melBefore), melAfter: r1(melAfter), ...Object.fromEntries(Object.entries(timings).map(([k, v]) => [k, r1(v)])) };
  row.unaccounted = row.total - (row.mel + row.encode + (row.detect || 0) + row.decode + row.parse);
  if (r) rows.push(row);
  ctx.log(JSON.stringify(row));
}
return { rows };
