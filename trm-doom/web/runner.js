// Headless lockstep player: loads a level, and each tic reads the game state back (every `every` tics),
// asks the bot for input and runs one tic. Used by the test page here and by the GPU-bridge job.

export async function runLevel({ device, doom, wadBytes, planner, map = "E1M1", maxTics = 35 * 300, every = 2, skill = 3,
  frameEvery = 0, debugGrids = false, frameSize = [320, 200], carry = null, onProgress = null, deadlineMs = 1e12, seed = 0, globalPlanner = null, Bot }) {
  const { Engine, unpackWad, freshPlayer, W } = doom.engine;
  const eng = await Engine.create(device, await unpackWad(wadBytes), { width: frameSize[0], height: frameSize[1], skill, maxWidth: frameSize[0], maxHeight: frameSize[1] });
  const t0 = performance.now();
  await eng.loadMap(map, carry || { ...freshPlayer(), rng: seed & 255 });
  const compileMs = performance.now() - t0;
  const bot = new Bot({ wad: eng.wad, map: eng.map, MOBJINFO: doom.info.MOBJINFO, MF: doom.info.MF, LINE_SPECIALS: doom.specials.LINE_SPECIALS, TR: doom.specials.TR, W }, planner);
  bot.debugGrids = debugGrids;
  if (globalPlanner) bot.globalPlanner = globalPlanner;
  const read = async () => {
    const enc = device.createCommandEncoder(), out = {};
    const parts = { world: eng.worldBuf, sec: eng.secBuf, mobj: eng.mobjBuf, ls: eng.lsBuf };
    const sizes = { ls: eng.map.lines.length * 4 };
    const rbs = Object.entries(parts).map(([k, b]) => {
      const size = Math.ceil((sizes[k] || b.size) / 4) * 4;
      const rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(b, 0, rb, 0, size); return [k, rb];
    });
    device.queue.submit([enc.finish()]);
    for (const [k, rb] of rbs) { await rb.mapAsync(GPUMapMode.READ); out[k] = rb.getMappedRange().slice(0); rb.unmap(); rb.destroy(); }
    return out;
  };
  const frames = [], trace = [];
  let input = {}, result = "timeout", tic = 0, world = null;
  const tStart = performance.now();
  for (tic = 0; tic < maxTics; tic++) {
    if (tic % every === 0) {
      const st = await read(); world = st.world;
      bot.readState(st);
      const p = bot.s.p;
      if (p.exit) { result = p.exit === 2 ? "secret-exit" : "exit"; break; }
      if (p.dead) { result = "dead"; break; }
      if (performance.now() - tStart > deadlineMs) { result = "time-limit"; break; }
      if (tic % 35 === 0) { trace.push([tic, Math.round(p.x), Math.round(p.y), p.health]); onProgress?.(tic / maxTics, bot.debug()); }
    }
    input = await bot.step();
    if (frameEvery && tic % frameEvery === 0) {
      const snap = await eng.snapshot(1, { ...input, frac: 1 });
      const idx = new Uint8Array(snap.indices.length); for (let i = 0; i < idx.length; i++) idx[i] = snap.indices[i] & 255;
      frames.push(idx);
    } else {
      const enc = device.createCommandEncoder(); eng.encodeFrame(enc, 1, { ...input, frac: 1 }); device.queue.submit([enc.finish()]);
    }
  }
  await device.queue.onSubmittedWorkDone();
  const wi = new Int32Array(world), wu = new Uint32Array(world);
  const pal = eng.gfx.words.slice(eng.gfx.layout.PAL + new Uint32Array(world)[W.palette] * 0, eng.gfx.layout.PAL + 256);
  return {
    map, seed, result, tics: tic, seconds: +(tic / 35).toFixed(1), wallMs: Math.round(performance.now() - tStart), compileMs: Math.round(compileMs),
    kills: wi[W.killcount], totalKills: wi[W.totalkills], items: wi[W.itemcount], totalItems: wi[W.totalitems], health: wi[W.health],
    stats: bot.stats, events: bot.events, final: bot.debug(), trace, frames, palette: pal, carry: result === "exit" || result === "secret-exit" ? eng.playerFrom(world) : null,
  };
}
