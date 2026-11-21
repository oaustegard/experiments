// GPU Bridge job: the full transcribe() call against its phases run by hand, interleaved, same window.
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
const rows = [];
for (let r = 0; r < reps + 1; r++) {
  const p = {};
  let t = performance.now(); const m = logMel(pcm, w.dims.n_mels); p.mel = performance.now() - t;
  t = performance.now(); await w.encode(m); p.encode = performance.now() - t;
  t = performance.now(); const ids = await tr.decodeWindow({ language: "en", noRepeatNgram: 3 }); p.decode = performance.now() - t;
  t = performance.now(); tr.parseWindow(ids, 0, pcm.length / 16000); p.parse = performance.now() - t;
  p.phases = p.mel + p.encode + p.decode + p.parse;
  t = performance.now(); await tr.transcribe(pcm, { language: "en", noRepeatNgram: 3 }); p.transcribe = performance.now() - t;
  // encode alone again, then the same encode with the device idle first
  t = performance.now(); await w.encode(m); p.encodeAgain = performance.now() - t;
  if (r) rows.push(Object.fromEntries(Object.entries(p).map(([k, v]) => [k, Math.round(v)])));
  ctx.log(JSON.stringify(p));
}
return { rows };
