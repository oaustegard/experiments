// GPU Bridge job body (async function (ctx)): which start should the MASSIVE demo default to?
// Inputs: cfg = { sha, starts: { name: [spike keys "set/id"] }, sizes: [n], seeds: [s] }. Code, the demo and the
// ±31 models come from jsDelivr at cfg.sha. Each arm runs the page's own protocol: splitCases (15% validation,
// at most n labelled cases, the demo's 600 marked test cases), 10 epochs, lr 5e-4, batch 16, best epoch by
// validation KL with the start as a candidate, temperatures fitted on validation.
const { cfg } = ctx.inputs;
const root = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`, code = root + "decider-lora-creator/web/";
const models = root + "adapter-library/web/models/";
ctx.progress(0.01, "loading code");
const C = await ctx.importModule(code + "creator.js");
const Bk = await ctx.importModule(code + "backends.js");
const device = await ctx.gpu();
const FB = (u) => ctx.fetchBytes(u), J = async (u) => JSON.parse(new TextDecoder().decode(await FB(u)));
const [manifest, wb, tok, demo] = await Promise.all([J(models + "base31/manifest.json"), FB(models + "base31/weights.bin"),
  C.loadTokenizer(models + "base31/tokenizer.json"), J(code + "demo/massive.json")]);
const W = () => wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.byteLength);
const startSpike = async (keys) => {
  if (!keys.length) return C.scratchSpike(manifest);
  const sp = await Promise.all(keys.map(async (k) => { const [set, id] = k.split("/"); return C.unpackSpike(await J(models + `${set}/spikes/${id}.json`), await FB(models + `${set}/spikes/${id}.bin`)); }));
  return C.packSpike(sp.length === 1 ? sp[0] : Bk.truncateRank(C.mergeSpikes(sp), 8), { id: "start" });
};
const all = C.examples(demo.decision, demo.cases, tok.build);
const maxLen = Math.max(...all.map((e) => e.ids.length));
const lb = await Bk.LoraBackend.create(device, manifest, W(), { module: root + "ettin-wgsl-train/web/train.js", pad: tok.pad, maxSeqs: 16, maxRows: 16 * maxLen });
const out = { describe: ctx.describe ? await ctx.describe() : null, sha: cfg.sha, maxLen, arms: [] };
const plan = [];
for (const [name, keys] of Object.entries(cfg.starts)) for (const n of cfg.sizes) for (const s of cfg.seeds) plan.push({ name, keys, n, seed: s });
let k = 0;
for (const p of plan) {
  const init = await startSpike(p.keys);
  await lb.loadSpike(init.sm, init.bytes);
  const sp = C.splitCases(demo.cases, { valFrac: 0.15, seed: p.seed, maxTrain: p.n });
  const data = { train: C.examples(demo.decision, sp.train, tok.build), val: C.examples(demo.decision, sp.val, tok.build), test: C.examples(demo.decision, sp.test, tok.build) };
  const start = await C.evaluateStart(lb, data), t0 = performance.now();
  const res = await C.train(lb, data, { epochs: 10, lr: 5e-4, bs: 16, seed: p.seed }, (e) => {
    if (e.kind === "step" && e.step % 20 === 0) ctx.progress(0.03 + 0.97 * (k + e.step / e.of) / plan.length, `${p.name} n=${p.n || "all"} seed ${p.seed}: step ${e.step}/${e.of}`);
  });
  out.arms.push({ start: p.name, n: p.n, seed: p.seed, train_cases: sp.train.length, val_cases: sp.val.length, test_cases: sp.test.length,
    start_test: start.test?.accuracy, test: res.test?.accuracy, test_kl: res.test?.kl, questions: res.test?.questions, prior: res.prior?.accuracy,
    best_epoch: res.bestEpoch, ms: performance.now() - t0 });
  ctx.log(`${p.name} n=${p.n || "all"} seed ${p.seed}: start ${start.test?.accuracy?.toFixed(3)} -> ${res.test?.accuracy?.toFixed(3)} (epoch ${res.bestEpoch}, ${((performance.now() - t0) / 1000).toFixed(0)} s)`);
  k++;
}
return out;
