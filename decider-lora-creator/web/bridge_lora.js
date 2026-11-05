// GPU Bridge job body (async function (ctx)): the creator's checks, then LoRA growth arms with
// ettin-wgsl-train's WGSL trainer, on adapter-library's growth protocol (see bridge_job.js for head-only).
// Inputs: cfg from run/bridge_inputs.py plus { checks: bool, arms: bool, sizes?: [n] }.
const { cfg } = ctx.inputs;
const code = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/decider-lora-creator/web/`;
const trainerUrl = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/ettin-wgsl-train/web/train.js`;
ctx.progress(0.01, "loading code");
const C = await ctx.importModule(code + "creator.js");
const Bk = await ctx.importModule(code + "backends.js");
const { runChecks } = await ctx.importModule(code + "checks.js");
const device = await ctx.gpu();
const models = cfg.models, out = { describe: ctx.describe ? await ctx.describe() : null, sha: cfg.sha };
const FB = (u) => ctx.fetchBytes(u);
if (cfg.checks !== false) {
  ctx.progress(0.03, "checks");
  out.checks = await runChecks({ device, models, code, fetchBytes: FB, trainer: trainerUrl, n: 16, test: 20, epochs: 3, log: (k, v) => ctx.log(`${k}: ${v.ok ? "ok" : "FAIL"}`) });
}
if (cfg.arms !== false) {
  const J = async (u) => JSON.parse(new TextDecoder().decode(await FB(u)));
  const [manifest, wb, tok, demo] = await Promise.all([J(models + "base/manifest.json"), FB(models + "base/weights.bin"), C.loadTokenizer(models + "base/tokenizer.json"), J(code + `demo/${cfg.workflow}.json`)]);
  const lib = await Promise.all(demo.init.map(async (id) => ({ sm: await J(models + `workflows/spikes/${id}.json`), bytes: await FB(models + `workflows/spikes/${id}.bin`) })));
  const init = C.packSpike(Bk.truncateRank(C.mergeSpikes(lib.map((s) => C.unpackSpike(s.sm, s.bytes))), 8), { id: "library-merge" });
  const test = C.examples(demo.decision, demo.cases.filter((c) => c.split === "test"), tok.build);
  const trainAll = C.examples(demo.decision, demo.cases.filter((c) => c.split !== "test"), tok.build);
  const maxRows = cfg.bs * Math.max(...[...trainAll, ...test].map((e) => e.ids.length));
  const t0 = performance.now();
  const lb = await Bk.LoraBackend.create(device, manifest, wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.byteLength), { module: trainerUrl, pad: tok.pad, maxSeqs: cfg.bs, maxRows });
  out.trainerInitMs = performance.now() - t0; out.maxRows = maxRows;
  const score = async (exs) => C.metrics(exs, C.probsOf(exs, await lb.score(exs), {}));
  await lb.loadSpike(init.sm, init.bytes);
  out.wo_mlp_dropped = lb.dropped;
  out.start_test = await score(test);
  ctx.log(`start (library merge without Wo_mlp, T=1): ${out.start_test.accuracy.toFixed(3)}`);
  out.arms = [];
  const sizes = cfg.sizes ?? Object.keys(cfg.reveal).map(Number), seeds = Object.keys(cfg.reveal[sizes[0]]);
  let k = 0;
  for (const n of sizes) for (const s of seeds) {
    await lb.loadSpike(init.sm, init.bytes);
    const ids = new Set(cfg.reveal[n][s]), tr = trainAll.filter((e) => ids.has(e.caseId));
    const ta = performance.now();
    const res = await C.train(lb, { train: tr, val: [], test }, { epochs: cfg.epochs[n], lr: cfg.lora_lr ?? 5e-4, bs: cfg.bs, seed: +s }, (e) => {
      if (e.kind === "step" && e.step % 10 === 0) ctx.progress(0.2 + 0.8 * (k + e.step / e.of) / (sizes.length * seeds.length), `n=${n} seed=${s} step ${e.step}/${e.of}`);
    });
    const m = await score(test);
    out.arms.push({ n, seed: +s, sequences: tr.length, epochs: cfg.epochs[n], test_accuracy: m.accuracy, test_kl: m.kl, questions: m.questions, ms: performance.now() - ta });
    ctx.log(`LoRA n=${n} seed=${s}: ${m.accuracy.toFixed(3)} in ${((performance.now() - ta) / 1000).toFixed(1)} s`);
    k++;
  }
}
return out;
