// GPU Bridge job: the camera-resonator engine on real hardware. Imports engine.js from jsDelivr at the commit in
// inputs.meta.sha, runs the float64 accuracy check, then times sequential blocks (submit to readback, as the
// page's latency path) for plates of growing size and for the bells.
const { sha, block = 512, sr = 48000, counts = [1000, 2000, 4000, 8000, 16000, 32000, 64000], blocks = 60 } = ctx.inputs.meta;
const E = await ctx.importModule(`https://cdn.jsdelivr.net/gh/oaustegard/experiments@${sha}/camera-resonator/web/engine.js`);
const dev = await ctx.gpu();
ctx.progress(0, "accuracy check");
const check = await E.selfTest(dev, { modes: 2000, blocks: 16, block, sr });
ctx.log(`accuracy: ${check.rmsDb.toFixed(1)} dB RMS re peak`);
const blockMs = 1000 * block / sr, rows = [];
const time = async (label, S) => {
  const e = await E.ModalEngine.create(dev, S, { block });
  await e.render([{ x: 0.37, y: 0.41, vel: 1, tau: 3e-4, js: 0 }, { x: 0.2, y: 0.8, vel: 1, tau: 3e-4, js: 0 }]);   // every plate mode rings
  const ms = [];
  for (let b = 0; b < blocks; b++) { const t = performance.now(); await e.render(b % 8 ? [] : [{ x: 0.6, y: 0.3, vel: 0.7, tau: 6e-4, js: 128 }]); ms.push(performance.now() - t); }
  e.destroy(); ms.sort((a, b) => a - b);
  const r = { label, modes: S.n, medianMs: +ms[blocks >> 1].toFixed(2), p90Ms: +ms[Math.floor(blocks * 0.9)].toFixed(2), load: +(ms[blocks >> 1] / blockMs).toFixed(3) };
  rows.push(r); ctx.log(JSON.stringify(r));
};
for (const [i, n] of counts.entries()) { ctx.progress((i + 1) / (counts.length + 2), `plate ${n}`); await time("plate", E.plate(n, sr)); }
await time("bells", E.bells(sr));
return { check, block, sr, blockMs, rows, adapter: await ctx.describe() };
