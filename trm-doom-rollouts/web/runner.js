// Headless lockstep player, from trm-doom's runner.js with two changes: `wrap` puts a controller around the Bot
// (given the engine and a Forker), and tics are not drawn unless a frame is being kept. Drawing changes no game
// state the Bot reads (it sets the automap's seen-line bit, which the Bot masks off; test/fork-check.html).
import { Forker } from "./fork.js";

export async function runLevel({ device, doom, wadBytes, planner, map = "E1M1", maxTics = 35 * 300, skill = 3,
  frameEvery = 0, frameSize = [320, 200], onProgress = null, deadlineMs = 1e12, seed = 0, globalPlanner = null, Bot, wrap = null, wrapOpts = {} }) {
  const { Engine, unpackWad, freshPlayer, W } = doom.engine;
  const eng = await Engine.create(device, await unpackWad(wadBytes), { width: frameSize[0], height: frameSize[1], skill, maxWidth: frameSize[0], maxHeight: frameSize[1] });
  await eng.loadMap(map, { ...freshPlayer(), rng: seed & 255 });
  const fk = new Forker(eng, device);
  let bot = new Bot({ wad: eng.wad, map: eng.map, MOBJINFO: doom.info.MOBJINFO, MF: doom.info.MF, LINE_SPECIALS: doom.specials.LINE_SPECIALS, TR: doom.specials.TR, W }, planner, { seed });
  if (globalPlanner) bot.globalPlanner = globalPlanner;
  if (wrap) bot = wrap(bot, fk, { W, MF: doom.info.MF, eng }, wrapOpts);
  const sizes = { ls: Math.ceil(eng.map.lines.length * 4 / 4) * 4 };
  const read = async () => {
    const parts = [["world", eng.worldBuf], ["sec", eng.secBuf], ["mobj", eng.mobjBuf], ["ls", eng.lsBuf]];
    const got = await fk.read(parts.map(([k, b]) => [b, 0, sizes[k] || b.size]));
    return Object.fromEntries(parts.map(([k], i) => [k, got[i]]));
  };
  const frames = [], trace = [];
  let result = "timeout", tic = 0, world = null;
  const tStart = performance.now();
  for (tic = 0; tic < maxTics; tic++) {
    const st = await read(); world = st.world;
    bot.readState(st);
    const p = bot.s.p;
    if (p.exit) { result = p.exit === 2 ? "secret-exit" : "exit"; break; }
    if (p.dead) { result = "dead"; break; }
    if (performance.now() - tStart > deadlineMs) { result = "time-limit"; break; }
    if (tic % 35 === 0) { trace.push([tic, Math.round(p.x), Math.round(p.y), p.health]); onProgress?.(tic / maxTics, bot.debug()); }
    const input = await bot.step();
    if (frameEvery && tic % frameEvery === 0) {
      const snap = await eng.snapshot(1, { ...input, frac: 1 });
      const idx = new Uint8Array(snap.indices.length); for (let i = 0; i < idx.length; i++) idx[i] = snap.indices[i] & 255;
      frames.push(idx);
    } else fk.tic(input);
  }
  await device.queue.onSubmittedWorkDone();
  const wi = new Int32Array(world);
  const pal = eng.gfx.words.slice(eng.gfx.layout.PAL, eng.gfx.layout.PAL + 256);
  return {
    map, seed, skill, result, tics: tic, seconds: +(tic / 35).toFixed(1), wallMs: Math.round(performance.now() - tStart),
    kills: wi[W.killcount], totalKills: wi[W.totalkills], items: wi[W.itemcount], totalItems: wi[W.totalitems], health: wi[W.health], armor: wi[W.armor],
    stats: bot.stats, events: bot.events, final: bot.debug(), trace, frames, palette: pal, decisions: bot.decisions || null,
  };
}
