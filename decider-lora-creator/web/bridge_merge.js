// GPU Bridge job body (async function (ctx)): round 4. (1) more seeds of the page's library-start LoRA (`warm`);
// (2) the learned library merge (LearnedMergeBackend) in three modes, after a check of its alpha gradient
// against central differences. Inputs: cfg from run/bridge_inputs.py with seeds 0-7 at n = 16, 64, plus
// { warmSeeds, mergeSeeds: { "alpha+res": n, "uniform+res": n, alpha: n } }.
const { cfg } = ctx.inputs;
const root = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`, code = root + "decider-lora-creator/web/";
ctx.progress(0.01, "loading code");
const C = await ctx.importModule(code + "creator.js");
const Bk = await ctx.importModule(code + "backends.js");
const { Trainer } = await ctx.importModule(root + "ettin-wgsl-train/web/train.js");
const device = await ctx.gpu();
const FB = (u) => ctx.fetchBytes(u), J = async (u) => JSON.parse(new TextDecoder().decode(await FB(u))), models = cfg.models;
const [manifest, wb, tok, demo] = await Promise.all([J(models + "base/manifest.json"), FB(models + "base/weights.bin"), C.loadTokenizer(models + "base/tokenizer.json"), J(code + `demo/${cfg.workflow}.json`)]);
const W = () => wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.byteLength);
const lib = await Promise.all(demo.init.map(async (id) => C.unpackSpike(await J(models + `workflows/spikes/${id}.json`), await FB(models + `workflows/spikes/${id}.bin`))));
const merged = C.mergeSpikes(lib), init = C.packSpike(Bk.truncateRank(merged, 8), { id: "library-merge" });
const test = C.examples(demo.decision, demo.cases.filter((c) => c.split === "test"), tok.build);
const trainAll = C.examples(demo.decision, demo.cases.filter((c) => c.split !== "test"), tok.build);
const maxRows = cfg.bs * Math.max(...[...trainAll, ...test].map((e) => e.ids.length));
const out = { describe: ctx.describe ? await ctx.describe() : null, sha: cfg.sha, maxRows, arms: [] };
const sizes = Object.keys(cfg.reveal).map(Number);
const plan = [];
for (const n of sizes) for (let s = 0; s < cfg.warmSeeds; s++) plan.push({ arm: "warm", n, seed: s });
for (const [mode, k] of Object.entries(cfg.mergeSeeds)) for (const n of sizes) for (let s = 0; s < k; s++) plan.push({ arm: mode, n, seed: s });
let done = 0;
const run = async (b, p, startFn) => {
  await startFn();
  const ids = new Set(cfg.reveal[p.n][p.seed]), tr = trainAll.filter((e) => ids.has(e.caseId)), ta = performance.now();
  await C.train(b, { train: tr, val: [], test }, { epochs: cfg.epochs[p.n], lr: 5e-4, bs: cfg.bs, seed: p.seed }, (e) => {
    if (e.kind === "step" && e.step % 10 === 0) ctx.progress(0.05 + 0.95 * (done + e.step / e.of) / plan.length, `${p.arm} n=${p.n} seed=${p.seed}`);
  });
  const m = C.metrics(test, C.probsOf(test, await b.score(test), {}));
  const rec = { ...p, test_accuracy: m.accuracy, test_kl: m.kl, questions: m.questions, ms: performance.now() - ta };
  if (Array.isArray(b.alpha)) { const a = b.alpha.flatMap((L) => L.flatMap((o) => Object.values(o))); rec.alpha = { mean: a.reduce((x, y) => x + y, 0) / a.length, min: Math.min(...a), max: Math.max(...a),
    perSpike: b.alpha.map((L) => L.flatMap((o) => Object.values(o)).reduce((x, y) => x + y, 0) / (L.length * Object.keys(L[0]).length)) }; }
  out.arms.push(rec); done++;
  ctx.log(`${p.arm} n=${p.n} seed=${p.seed}: ${m.accuracy.toFixed(3)} (${(rec.ms / 1000).toFixed(1)} s)`);
};
// 1. warm (rank-8 SVD-cut merge, everything trained)
const lb = await Bk.LoraBackend.create(device, manifest, W(), { module: root + "ettin-wgsl-train/web/train.js", pad: tok.pad, maxSeqs: cfg.bs, maxRows });
for (const p of plan.filter((x) => x.arm === "warm")) await run(lb, p, () => lb.loadSpike(init.sm, init.bytes));
// 2. learned merge: gradient check first
const mb = await Bk.LearnedMergeBackend.create(device, manifest, W(), { Trainer, pad: tok.pad, lib, head: merged.head, maxSeqs: cfg.bs, maxRows, mode: "alpha+res" });
{
  const batch = trainAll.slice(0, cfg.bs), tg = batch.map((e) => e.target), t = mb.t;
  t.loadBatch(mb.inner.collate(batch, tg)); t.forwardBackward({ T: 1, slot: 0 });
  const ga = mb.alphaGrads(await t.read(t.GRD)), rows = [];
  for (const [k, i, m] of [[0, 0, "Wqkv"], [1, 3, "Wi"], [2, 6, "Wo_mlp"], [1, 5, "Wo"]]) {
    const a0 = mb.alpha[k][i][m], h = 0.05, L = [];
    for (const d of [h, -h]) { mb.alpha[k][i][m] = a0 + d; const prm = await t.read(t.PRM); mb.writeLib(prm); device.queue.writeBuffer(t.PRM, 0, prm); L.push(await mb.lossAt(batch, tg)); }
    mb.alpha[k][i][m] = a0; const prm = await t.read(t.PRM); mb.writeLib(prm); device.queue.writeBuffer(t.PRM, 0, prm);
    const fd = (L[0] - L[1]) / (2 * h);
    rows.push({ k, layer: i, m, analytic: ga[k][i][m], fd, rel: Math.abs(fd - ga[k][i][m]) / (Math.abs(fd) + Math.abs(ga[k][i][m]) + 1e-9) });
  }
  out.alpha_gradcheck = rows;
  ctx.log(`alpha gradcheck worst rel ${Math.max(...rows.map((r) => r.rel)).toExponential(2)}`);
  // the merge with no residual or training equals the uniform merge: score it
  mb.mode = "uniform+res"; mb.reset(); await t.flush();
  out.merge_start_test = C.metrics(test, C.probsOf(test, await mb.score(test), {})).accuracy;
}
for (const p of plan.filter((x) => x.arm !== "warm")) await run(mb, p, async () => { mb.mode = p.arm; mb.seed = p.seed; mb.reset(); await mb.t.flush(); });
return out;
