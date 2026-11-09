// Per-op parity and timing for the shared WGSL kernels (_lib/wgsl/tests/ops.js) on a real GPU, then
// the training kernels' finite-difference parity (_lib/wgsl/tests/train_ops.js) and the retrieval ops
// (retrieval_ops.js), parity only, no timing.
// Same cases as the headless check; each inference op is also timed, reps dispatches per pass, median
// of rounds. It is the per-op baseline a kernel-library change is compared against.
// inputs.cfg = { sha, only: "", reps: 200, rounds: 7, train: true, retrieval: true }
const cfg = ctx.inputs.cfg;
const dev = await ctx.gpu();
const base = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/_lib/wgsl/tests`;
const { runOps, cases } = await ctx.importModule(`${base}/ops.js`);
let i = 0;
const total = cases.filter((c) => !cfg.only || c.name.includes(cfg.only) || c.op === cfg.only).length;
const results = await runOps(dev, { only: cfg.only ?? "", reps: cfg.reps ?? 200, rounds: cfg.rounds ?? 7,
  log: (m) => { ctx.log(m); ctx.progress(++i / (total + 1), m.slice(5, 40)); } });
if (cfg.train !== false) {
  const { runTrainOps } = await ctx.importModule(`${base}/train_ops.js`);
  results.push(...await runTrainOps(dev, { only: cfg.only ?? "", log: (m) => ctx.log(m) }));
}
if (cfg.retrieval !== false) {
  const { runRetrievalOps } = await ctx.importModule(`${base}/retrieval_ops.js`);
  results.push(...await runRetrievalOps(dev, { only: cfg.only ?? "", log: (m) => ctx.log(m) }));
}
return { cfg, pass: results.filter((r) => r.pass).length, total: results.length, results };
