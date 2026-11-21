// GPU Bridge job: time encode() and decodeWindow() from inside a real transcribe() call by wrapping them.
const { clip_7s: pcm, reps = 4 } = ctx.inputs;
const SITE = "https://austegard.com/experiments/whisper-pack/web/", bust = "?t=" + Date.now();
const R2 = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/models/whisper-pack/turbo-en-r15/";
const { Whisper } = await ctx.importModule(SITE + "whisper.js" + bust);
const { Tokenizer } = await ctx.importModule(SITE + "tokenizer.js" + bust);
const { Transcriber } = await ctx.importModule(SITE + "transcribe.js" + bust);
const device = await ctx.gpu();
const w = await Whisper.load(device, R2);
const tr = new Transcriber(w, await Tokenizer.load(R2 + "tokenizer.json"));
let log = [];
const wrap = (obj, name) => { const f = obj[name].bind(obj); obj[name] = async (...a) => {
  const t = performance.now(); const r = await f(...a); log.push([name, Math.round(performance.now() - t), Array.isArray(r) ? r.length : null]); return r; }; };
wrap(w, "encode"); wrap(tr, "decodeWindow"); wrap(tr, "detectLanguage");
const stepCount = { n: 0 }; const st = w.step.bind(w); w.step = (...a) => { stepCount.n++; return st(...a); };
const rows = [];
for (let r = 0; r < reps + 1; r++) {
  log = []; stepCount.n = 0;
  const t = performance.now();
  const out = await tr.transcribe(pcm, { language: "en", noRepeatNgram: 3 });
  const total = Math.round(performance.now() - t);
  if (r) rows.push({ total, calls: log, steps: stepCount.n, windows: out.windows, chunks: out.chunks.length });
  ctx.log(JSON.stringify(rows.at(-1) || { total, calls: log }));
}
return { rows };
