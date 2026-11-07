// GPU-bridge job: splat-diffusion sampling time, paint time, and a fixed-noise sample for a torch comparison.
// inputs: src (splatdiff.js bytes), man (manifest.json bytes), w (weights.bin u16), noise (1024x10 f32)
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { SplatDiffusion, f16ToF32, paintToBytes } = await import(url);
const device = await ctx.gpu();
const errs = [];
device.addEventListener("uncapturederror", (e) => errs.push(e.error.message));
const m = new SplatDiffusion(device, JSON.parse(td.decode(ctx.inputs.man)), f16ToF32(ctx.inputs.w));
const runs = [];
for (const [steps, cfg] of [[24, 3], [24, 1], [12, 3]]) {
  let best = Infinity;
  for (let r = 0; r < 3; r++) {
    const t0 = performance.now();
    await m.sample({ cls: 23, steps, cfg, seed: 3 });
    await device.queue.onSubmittedWorkDone();
    if (r > 0) best = Math.min(best, performance.now() - t0);
  }
  runs.push({ steps, cfg, total_ms: best });
  ctx.log(JSON.stringify(runs[runs.length - 1]));
}
const paint = [];
for (const size of [512, 1024, 2048]) {
  let best = Infinity;
  for (let r = 0; r < 3; r++) {
    const t0 = performance.now();
    await paintToBytes(device, m.b.splats, m.K, size);
    if (r > 0) best = Math.min(best, performance.now() - t0);
  }
  paint.push({ size, ms_incl_readback: best });
  ctx.log(JSON.stringify(paint[paint.length - 1]));
}
await m.sample({ cls: 23, steps: 24, cfg: 3, noise: ctx.inputs.noise });
const state = await m.read(m.b.S, m.K * 10);
const splats = await m.splats();
const px = await paintToBytes(device, m.b.splats, m.K, 128);
return { runs, paint, state, splats, px, errs };
