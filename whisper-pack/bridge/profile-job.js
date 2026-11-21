// GPU Bridge job: where the packed turbo's decode time goes. One encoded window, then the decoder alone,
// with each of the page's options switched on in turn. inputs: clip_7s (f32 16 kHz), reps.
const { clip_7s: pcm, reps = 3 } = ctx.inputs;
const SITE = "https://austegard.com/experiments/whisper-pack/web/", bust = "?t=" + Date.now();
const R2 = "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/models/whisper-pack/turbo-en-r15/";
const { Whisper } = await ctx.importModule(SITE + "whisper.js" + bust);
const { Tokenizer } = await ctx.importModule(SITE + "tokenizer.js" + bust);
const { Transcriber } = await ctx.importModule(SITE + "transcribe.js" + bust);
const { logMel } = await ctx.importModule(SITE + "mel.js" + bust);
const device = await ctx.gpu();
const w = await Whisper.load(device, R2);
const tr = new Transcriber(w, await Tokenizer.load(R2 + "tokenizer.json"));
await w.encode(logMel(pcm, w.dims.n_mels));
const g = w.gen, med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
const arms = {
  "bare decode, no timestamps": () => w.decode([g.sot, g.lang_to_id["<|en|>"], g.task_to_id.transcribe, g.no_timestamps]),
  "rules off, no timestamps": () => tr.decodeWindow({ language: "en", timestamps: false, noRepeatNgram: 0 }),
  "timestamps on": () => tr.decodeWindow({ language: "en", timestamps: true, noRepeatNgram: 0 }),
  "timestamps + 3-gram ban (page)": () => tr.decodeWindow({ language: "en", timestamps: true, noRepeatNgram: 3 }),
};
const rows = [];
// raw step cost: one decoder step with and without the logits readback
const stepMs = { withReadback: [], noReadback: [] };
for (let r = 0; r < 8; r++) {
  let t = performance.now(); await w.step(g.sot, 0, true); stepMs.withReadback.push(performance.now() - t);
  t = performance.now(); w.step(g.sot, 0, false); await device.queue.onSubmittedWorkDone(); stepMs.noReadback.push(performance.now() - t);
}
for (const [name, fn] of Object.entries(arms)) {
  const ms = []; let n = 0;
  for (let r = 0; r < reps + 1; r++) { const t = performance.now(); const ids = await fn(); ms.push(performance.now() - t); n = ids.length; }
  ms.shift();
  rows.push({ arm: name, ms: med(ms), tokens: n, msPerStep: med(ms) / (n + 4) });
  ctx.log(JSON.stringify(rows.at(-1)));
}
return { rows, stepWithReadbackMs: med(stepMs.withReadback), stepNoReadbackMs: med(stepMs.noReadback) };
