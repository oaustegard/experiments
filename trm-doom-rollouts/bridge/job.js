// GPU-bridge job: the TRM bot with and without rollout fights, on the tab's GPU. Body of async function (ctx).
// inputs: cfg (JSON: {sha, runs: [{map, seed, arm: "base"|"roll"}], skill, maxTics, opts, sup, minSup, budgetMs})
const cfg = ctx.inputs.cfg;
const X = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`;
const D = "https://austegard.com/games/doom/";
const [{ runLevel }, { RolloutBot }, { Bot }, { TRMPlanner }] = await Promise.all([
  X + "trm-doom-rollouts/web/runner.js", X + "trm-doom-rollouts/web/combat.js", X + "trm-doom/web/bot.js", X + "trm-doom/web/trm.js",
].map((u) => ctx.importModule(u)));
const doom = { engine: await ctx.importModule(D + "engine.js"), info: await ctx.importModule(D + "info.js"), specials: await ctx.importModule(D + "specials.js") };
const device = await ctx.gpu();
const wadBytes = await ctx.fetchBytes(D + "e1.wad.gz");
const w = new Float32Array((await ctx.fetchBytes(D + "bot/trm.bin")).slice().buffer);
const man = JSON.parse(new TextDecoder().decode(await ctx.fetchBytes(D + "bot/trm.json")));
const trm = await TRMPlanner.create(device, w, man);
const cache = new Map();
const globalPlanner = async (n) => { if (!cache.has(n)) cache.set(n, await TRMPlanner.create(device, w, man, { w: n, h: n })); return cache.get(n); };
let plans = 0, ms = 0;
const planner = { async plan(g) { const r = await trm.plan(g, { sup: cfg.sup || 4, minSup: cfg.minSup || 2 }); plans++; ms += r.ms; return r; } };
const out = { runs: [], device: ctx.describe?.()?.device };
const t0 = performance.now(), budget = cfg.budgetMs || 1700000;
for (const [i, run] of cfg.runs.entries()) {
  const left = budget - (performance.now() - t0);
  if (left < 30000) { ctx.log(`budget spent before run ${i}`); break; }
  const r = await runLevel({ device, doom, wadBytes, planner, Bot, globalPlanner, map: run.map, skill: run.skill || cfg.skill || 2,
    maxTics: cfg.maxTics || 35 * 300, seed: run.seed, deadlineMs: left - 15000,
    wrap: run.arm === "roll" ? (bot, fk, c, o) => new RolloutBot(bot, fk, c, o) : null, wrapOpts: { ...(cfg.opts || {}), ...(run.opts || {}) },
    onProgress: (f, d) => ctx.progress(Math.min(0.99, (i + f) / cfg.runs.length), `${run.arm} ${run.map} s${run.seed} hp ${d.health} ${d.move || ""}`) });
  delete r.frames; delete r.palette;
  r.arm = run.arm;
  r.events = r.events.filter((e) => e.ev !== "dump").slice(-200);
  if (r.decisions) r.decisions = r.decisions.slice(-600);
  out.runs.push(r);
  const ro = r.stats.rollout;
  ctx.log(`${run.arm} ${run.map} seed ${run.seed}: ${r.result} in ${r.seconds}s, kills ${r.kills}/${r.totalKills}, hp ${r.health}, wall ${Math.round(r.wallMs / 1000)}s` +
    (ro ? `, ${ro.decisions} decisions at ${ro.meanMs} ms` : ""));
}
out.planner = { plans, meanMs: plans ? ms / plans : 0 };
return out;
