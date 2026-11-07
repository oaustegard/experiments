// Per-op parity and timing of the shared WGSL kernels (_lib/wgsl/tests/ops.js) on a real GPU.
// Same cases as the headless check; each op is also timed, reps dispatches per pass, median of rounds.
// It is the per-op baseline a kernel-library change is compared against.
// inputs.cfg = { sha, only: "", reps: 200, rounds: 7 }
const cfg = ctx.inputs.cfg;
const dev = await ctx.gpu();
const { runOps, cases } = await ctx.importModule(`https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/_lib/wgsl/tests/ops.js`);
let i = 0;
const total = cases.filter((c) => !cfg.only || c.name.includes(cfg.only) || c.op === cfg.only).length;
const results = await runOps(dev, { only: cfg.only ?? "", reps: cfg.reps ?? 200, rounds: cfg.rounds ?? 7,
  log: (m) => { ctx.log(m); ctx.progress(++i / total, m.slice(5, 40)); } });
return { cfg, pass: results.filter((r) => r.pass).length, total: results.length, results };
