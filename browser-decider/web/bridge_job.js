// GPU Bridge job body (async function (ctx)): time the decider on a real GPU and check its logits.
// Inputs: weights (u8, weights.bin), manifest / tokenizer / fixtures (u8, the JSON files' bytes),
// cfg = { sha, reps }. Code comes from jsDelivr at the pushed commit `sha`.
const { cfg } = ctx.inputs;
const dec = new TextDecoder();
const json = (k) => JSON.parse(dec.decode(ctx.inputs[k]));
const base = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/browser-decider/web/`;
ctx.progress(0.02, "loading code");
const { deciderFrom } = await ctx.importModule(base + "decider.js");
const manifest = json("manifest"), tj = json("tokenizer"), fx = json("fixtures");
const device = await ctx.gpu();
const t0 = performance.now();
const d = await deciderFrom({ manifest, tj, data: ctx.inputs.weights }, { device });
const setupMs = performance.now() - t0;
ctx.progress(0.1, "parity");

let maxDiff = 0, agree = 0, nq = 0;
for (const c of fx.cases) {
  const z = await d.logitsFor(c.seqs.map((s) => ({ ids: s.ids, pos: s.pos })));
  c.seqs.forEach((s, i) => {
    const a = Array.from(z[i]), b = s.ref_logits;
    for (let k = 0; k < a.length; k++) maxDiff = Math.max(maxDiff, Math.abs(a[k] - b[k]));
    agree += a.indexOf(Math.max(...a)) === b.indexOf(Math.max(...b)); nq++;
  });
}

const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const p90 = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length * 0.9)]; };
const reps = cfg.reps ?? 5;
// whole case: tokenize + 5 questions in one engine pass + head + softmax
const caseMs = [], oneQMs = [], gpuOnlyMs = [];
let t = performance.now();
await d.decide(fx.cases[0].state, fx.cases[0].questions);
const firstMs = performance.now() - t;
for (let r = 0; r < reps; r++) {
  for (const c of fx.cases) {
    t = performance.now(); await d.decide(c.state, c.questions); caseMs.push(performance.now() - t);
    const [n0] = Object.keys(c.questions);
    t = performance.now(); await d.decide(c.state, { [n0]: c.questions[n0] }); oneQMs.push(performance.now() - t);
    t = performance.now(); await d.logitsFor(c.seqs.map((s) => ({ ids: s.ids, pos: s.pos }))); gpuOnlyMs.push(performance.now() - t);
  }
  ctx.progress(0.1 + 0.9 * (r + 1) / reps, "timing");
}
const tokens = fx.cases.map((c) => c.seqs.reduce((s, q) => s + q.ids.length, 0));
return {
  describe: ctx.describe ? await ctx.describe() : null,
  setupMs, firstCaseMs: firstMs, parity: { maxAbsLogitDiff: maxDiff, argmaxAgree: agree, nQuestions: nq },
  caseMsMedian: med(caseMs), caseMsP90: p90(caseMs), oneQuestionMsMedian: med(oneQMs),
  engineOnlyMsMedian: med(gpuOnlyMs), tokensPerCaseMedian: med(tokens), nCases: fx.cases.length, reps,
  sample: await d.decide(fx.cases[0].state, fx.cases[0].questions),
};
