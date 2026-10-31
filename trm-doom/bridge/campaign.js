// GPU-bridge job: play levels with the TRM bot on the tab's GPU. Body of async function (ctx).
// inputs: w (f32 weights), man (u8 JSON manifest), cfg (JSON: {sha, maps, maxTics, skill, frameEvery, planner, sup, carry})
const cfg = ctx.inputs.cfg;
const base = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/trm-doom/web/`;
const D = "https://austegard.com/games/doom/";
const [{ runLevel }, { Bot }, { TRMPlanner }, { OraclePlanner }] = await Promise.all(
  ["runner.js", "bot.js", "trm.js", "oracle.js"].map((f) => ctx.importModule(base + f)));
const doom = { engine: await ctx.importModule(D + "engine.js"), info: await ctx.importModule(D + "info.js"), specials: await ctx.importModule(D + "specials.js") };
const device = await ctx.gpu();
const wadBytes = await ctx.fetchBytes(D + "e1.wad.gz");
let planner, globalPlanner = null;
if (cfg.planner === "oracle") { planner = new OraclePlanner(); if (cfg.global) globalPlanner = async () => planner; }
else {
  const man = JSON.parse(new TextDecoder().decode(ctx.inputs.man));
  const trm = await TRMPlanner.create(device, ctx.inputs.w, man);
  // whole-level planning on its own weights when given (the multi-size fine-tune), else the same ones
  const gw = ctx.inputs.gw || ctx.inputs.w, gman = ctx.inputs.gman ? JSON.parse(new TextDecoder().decode(ctx.inputs.gman)) : man;
  const cache = new Map();
  if (cfg.global) globalPlanner = async (n) => { if (!cache.has(n)) cache.set(n, await TRMPlanner.create(device, gw, gman, { w: n, h: n })); return cache.get(n); };
  let plans = 0, steps = 0, ms = 0, halted = 0;
  planner = { async plan(g) { const r = await trm.plan(g, { sup: cfg.sup || 4, minSup: cfg.minSup || 2 }); plans++; steps += r.steps; ms += r.ms; halted += r.halt > 0; return r; },
    stats: () => ({ plans, meanSteps: steps / Math.max(plans, 1), meanMs: ms / Math.max(plans, 1), halted }) };
}
const out = { levels: [], device: ctx.describe?.()?.device };
const t0 = performance.now(), budget = cfg.budgetMs || 540000;
let carry = cfg.carry || null, frames = [], palette = null;
// mode "campaign": maps in order carrying the player over, stopping at a failure;
// mode "each": every map from a pistol start, up to `attempts` tries each
const queue = cfg.mode === "each" ? cfg.maps.flatMap((m) => Array(cfg.attempts || 1).fill(m)) : cfg.maps.slice();
const cleared = new Set();
const tries = {};
for (const map of queue) {
  tries[map] = (tries[map] || 0) + 1;
  if (cfg.mode === "each" && cleared.has(map)) continue;
  const left = budget - (performance.now() - t0);
  if (left < 20000) break;
  const r = await runLevel({ device, doom, wadBytes, planner, Bot, map, maxTics: cfg.maxTics || 35 * 600, skill: cfg.skill || 3,
    globalPlanner, debugGrids: !!cfg.debug, frameEvery: cfg.frameEvery || 0, carry: cfg.mode === "each" ? null : carry, deadlineMs: left - 8000, seed: (cfg.seed || 0) + 37 * (tries[map] - 1),
    onProgress: (f, d) => ctx.progress(Math.min(0.99, (performance.now() - t0) / budget), `${map} ${d.target?.kind || ""} hp ${d.health}`) });
  if (cfg.keepFrames !== false) frames.push(...r.frames);
  palette = r.palette; delete r.frames; delete r.palette;
  r.events = cfg.debug ? r.events.filter((e) => e.ev !== "dump" || e.tic > (cfg.dumpFrom || 0)).slice(-400) : r.events.filter((e) => e.ev !== "dump").slice(-300);
  out.levels.push(r);
  ctx.log(`${map}: ${r.result} in ${r.seconds}s, kills ${r.kills}/${r.totalKills}, hp ${r.health}`);
  const ok = r.result === "exit" || r.result === "secret-exit";
  if (ok) cleared.add(map);
  carry = r.carry;
  if (cfg.mode !== "each" && !ok) break;
}
out.planner = planner.stats ? planner.stats() : "oracle";
out.carry = carry;
if (frames.length) {
  const all = new Uint8Array(frames.length * frames[0].length);
  frames.forEach((f, i) => all.set(f, i * f.length));
  out.frames = all; out.palette = Uint32Array.from(palette); out.frameCount = frames.length;
}
return out;
