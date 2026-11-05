// GPU Bridge job body (async function (ctx)): time the spiky-ball system on a real GPU.
// Inputs: cfg = { sha, models, reps }. Code from jsDelivr at commit `sha`; models from the published site.
const { cfg } = ctx.inputs;
const code = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/adapter-library/web/`;
ctx.progress(0.02, "loading code");
const { loadSystem } = await ctx.importModule(code + "system.js");
const device = await ctx.gpu();
const t0 = performance.now();
const sys = await loadSystem(cfg.models, { device });
const setupMs = performance.now() - t0;
const samples = await fetch(cfg.models + "samples.json").then((r) => r.json());
const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const p90 = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length * 0.9)]; };
ctx.progress(0.1, "warm-up");
let t = performance.now();
const r0 = await sys.route(samples.cases[0].state); await sys.answer(r0.spike, samples.cases[0].state);
const firstMs = performance.now() - t;
const routeMs = [], answerSwapMs = [], answerNoSwapMs = [], totalMs = [];
let correctRoute = 0, n = 0;
for (let rep = 0; rep < (cfg.reps ?? 3); rep++) {
  for (const c of samples.cases) {
    const t1 = performance.now();
    const r = await sys.route(c.state);
    const t2 = performance.now();
    await sys.answer(r.spike, c.state);                       // swaps router -> spike
    const t3 = performance.now();
    await sys.answer(r.spike, c.state);                       // same spike again: no swap
    const t4 = performance.now();
    routeMs.push(t2 - t1); answerSwapMs.push(t3 - t2); answerNoSwapMs.push(t4 - t3); totalMs.push(t3 - t1);
    correctRoute += r.spike === c.workflow && !r.gap; n++;
  }
  ctx.progress(0.1 + 0.7 * (rep + 1) / (cfg.reps ?? 3), "timing");
}
ctx.progress(0.85, "memory");
const sid = samples.cases[0].workflow;
t = performance.now(); await sys.seed(sid, samples.seed[sid]); const seed16Ms = performance.now() - t;
const memMs = [];
for (const c of samples.cases.filter((x) => x.workflow === sid)) { t = performance.now(); await sys.answer(sid, c.state); memMs.push(performance.now() - t); }
return {
  describe: ctx.describe ? await ctx.describe() : null, version: sys.info.version, downloadBytes: sys.info.bytes,
  setupMs, firstCaseMs: firstMs, nCases: samples.cases.length, reps: cfg.reps ?? 3, routeCorrect: correctRoute / n,
  routeMsMedian: med(routeMs), routeMsP90: p90(routeMs), answerMsMedian: med(answerSwapMs), answerMsP90: p90(answerSwapMs),
  answerNoSwapMsMedian: med(answerNoSwapMs), caseTotalMsMedian: med(totalMs), caseTotalMsP90: p90(totalMs),
  seed16Ms, answerWithMemoryMsMedian: med(memMs), memoryCases: sys.memCount()[sid],
};
