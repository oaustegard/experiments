// GPU Bridge job body (async function (ctx)): the creator's head-only trainer on a real GPU, on
// adapter-library's growth-arm protocol, so its numbers sit next to the Python LoRA arms.
// Inputs: cfg from run/bridge_inputs.py = { sha, workflow, models, reveal: {n: {seed: [case ids]}}, epochs: {n}, lr, bs }.
// Start: the mean of the other three workflow spikes, SVD-cut to rank 8 (adapter-library's library start).
// Each arm trains on the n revealed cases with no validation, for growth_epochs(n), and scores the
// workflow's 100 test cases at temperature 1, as the G arms do.
const { cfg } = ctx.inputs;
const code = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/decider-lora-creator/web/`;
ctx.progress(0.01, "loading code");
const C = await ctx.importModule(code + "creator.js");
const Bk = await ctx.importModule(code + "backends.js");
const device = await ctx.gpu();
const J = async (u) => JSON.parse(new TextDecoder().decode(await ctx.fetchBytes(u)));
const models = cfg.models;
ctx.progress(0.03, "loading model");
const [manifest, weights, tok, demo] = await Promise.all([J(models + "base/manifest.json"), ctx.fetchBytes(models + "base/weights.bin"),
  C.loadTokenizer(models + "base/tokenizer.json"), J(code + `demo/${cfg.workflow}.json`)]);
const spike = async (id) => ({ sm: await J(models + `workflows/spikes/${id}.json`), bytes: await ctx.fetchBytes(models + `workflows/spikes/${id}.bin`) });
const lib = await Promise.all(demo.init.map(spike));
const init = C.packSpike(Bk.truncateRank(C.mergeSpikes(lib.map((s) => C.unpackSpike(s.sm, s.bytes))), 8), { id: "library-merge" });
const t0 = performance.now();
const be = await Bk.HeadBackend.create(device, manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength), { init });
const setupMs = performance.now() - t0;
const byId = new Map(demo.cases.map((c) => [c.id, c]));
const test = C.examples(demo.decision, demo.cases.filter((c) => c.split === "test"), tok.build);
const trainAll = C.examples(demo.decision, demo.cases.filter((c) => c.split !== "test"), tok.build);
let t = performance.now();
await be.warm([...trainAll, ...test], (d, n) => ctx.progress(0.05 + 0.25 * d / n, `encoding ${d}/${n}`));
const encodeMs = performance.now() - t, nSeq = trainAll.length + test.length;
const tokens = [...trainAll, ...test].reduce((a, e) => a + e.ids.length, 0);
const score = async (exs) => C.metrics(exs, C.probsOf(exs, await be.score(exs), {}));
const start = await score(test);
ctx.log(`start (library merge, T=1): ${start.accuracy.toFixed(3)}`);
const arms = [];
const sizes = Object.keys(cfg.reveal), seeds = Object.keys(cfg.reveal[sizes[0]]);
let k = 0;
for (const n of sizes) for (const s of seeds) {
  await be.loadSpike(init.sm, init.bytes);            // same LoRA: the row cache stays; head and Adam reset
  const ids = new Set(cfg.reveal[n][s]);
  const tr = trainAll.filter((e) => ids.has(e.caseId));
  const ta = performance.now();
  const res = await C.train(be, { train: tr, val: [], test }, { epochs: cfg.epochs[n], lr: cfg.lr, bs: cfg.bs, seed: +s }, () => {});
  const m = await score(test);
  arms.push({ n: +n, seed: +s, sequences: tr.length, epochs: cfg.epochs[n], test_accuracy: m.accuracy, test_kl: m.kl, questions: m.questions, ms: performance.now() - ta, best_epoch: res.bestEpoch });
  ctx.log(`n=${n} seed=${s}: ${m.accuracy.toFixed(3)} in ${((performance.now() - ta) / 1000).toFixed(1)} s`);
  ctx.progress(0.3 + 0.7 * ++k / (sizes.length * seeds.length), `n=${n} seed=${s}`);
}
return { describe: ctx.describe ? await ctx.describe() : null, sha: cfg.sha, workflow: cfg.workflow, setupMs, encodeMs, sequences: nSeq, tokens,
         encodeMsPerSeq: encodeMs / nSeq, start_test: { accuracy: start.accuracy, kl: start.kl }, arms };
