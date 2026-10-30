// GPU-bridge job: play levels with the TRM bot on the tab's GPU. Body of async function (ctx).
// inputs: w (f32 weights), man (u8 JSON manifest), cfg (JSON: {sha, maps, maxTics, skill, frameEvery, planner, sup, carry})
const cfg = ctx.inputs.cfg;
const base = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/trm-doom/web/`;
const D = "https://austegard.com/fun-and-games/doom/";
const [{ runLevel }, { Bot }, { TRMPlanner }, { OraclePlanner }] = await Promise.all(
  ["runner.js", "bot.js", "trm.js", "oracle.js"].map((f) => ctx.importModule(base + f)));
const doom = { engine: await ctx.importModule(D + "engine.js"), info: await ctx.importModule(D + "info.js"), specials: await ctx.importModule(D + "specials.js") };
const device = await ctx.gpu();
const wadBytes = await ctx.fetchBytes(D + "e1.wad.gz");
let planner;
if (cfg.planner === "oracle") planner = new OraclePlanner();
else {
  const man = JSON.parse(new TextDecoder().decode(ctx.inputs.man));
  const trm = await TRMPlanner.create(device, ctx.inputs.w, man);
  let plans = 0, steps = 0, ms = 0, halted = 0;
  planner = { async plan(g) { const r = await trm.plan(g, { sup: cfg.sup || 4 }); plans++; steps += r.steps; ms += r.ms; halted += r.halt > 0; return r; },
    stats: () => ({ plans, meanSteps: steps / Math.max(plans, 1), meanMs: ms / Math.max(plans, 1), halted }) };
}
const out = { levels: [], device: ctx.describe?.()?.device };
const t0 = performance.now();
let carry = cfg.carry || null, frames = [], palette = null;
for (const map of cfg.maps) {
  const left = (cfg.budgetMs || 540000) - (performance.now() - t0);
  if (left < 30000) break;
  const r = await runLevel({ device, doom, wadBytes, planner, Bot, map, maxTics: cfg.maxTics || 35 * 600, skill: cfg.skill || 3,
    frameEvery: cfg.frameEvery || 0, carry, deadlineMs: left - 10000,
    onProgress: (f, d) => ctx.progress(Math.min(0.99, (performance.now() - t0) / (cfg.budgetMs || 540000)), `${map} ${d.target?.kind || ""} hp ${d.health}`) });
  frames.push(...r.frames); palette = r.palette; delete r.frames; delete r.palette;
  r.events = r.events.filter((e) => e.ev !== "target" || true).slice(-400);
  out.levels.push(r);
  ctx.log(`${map}: ${r.result} in ${r.seconds}s, kills ${r.kills}/${r.totalKills}, hp ${r.health}`);
  carry = r.carry;
  if (r.result !== "exit" && r.result !== "secret-exit") break;
}
out.planner = planner.stats ? planner.stats() : "oracle";
out.carry = carry;
if (frames.length) {
  const all = new Uint8Array(frames.length * frames[0].length);
  frames.forEach((f, i) => all.set(f, i * f.length));
  out.frames = all; out.palette = Uint32Array.from(palette); out.frameCount = frames.length;
}
return out;
