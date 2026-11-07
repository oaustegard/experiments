// GPU-bridge job: packed-f16 activations vs f32 — 20-step guided sampling time and memory per size,
// plus a 128x128 sample from fixed noise per mode for a torch comparison.
// inputs: src (diffusion.js bytes), man (manifest.json bytes), w (weights.bin as u16), noise (1x3x128x128 f32)
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { TextureDiffusion, f16ToF32 } = await import(url);
const device = await ctx.gpu();
const man = JSON.parse(td.decode(ctx.inputs.man));
const w = f16ToF32(ctx.inputs.w);
const runs = [], out = {};
let k = 0;
for (const act of ["f32", "f16"]) {
  const m = new TextureDiffusion(device, man, w, { act });
  for (const H of [64, 128, 256, 512]) {
    let best = Infinity;
    for (let rep = 0; rep < 3; rep++) {
      const t0 = performance.now();
      await m.sample({ cls: 23, H, W: H, steps: 20, cfg: 3, seed: 5, circ: true });
      await device.queue.onSubmittedWorkDone();
      if (rep > 0) best = Math.min(best, performance.now() - t0);
    }
    runs.push({ act, H, total_ms: best, bytes: m.bytes });
    ctx.log(JSON.stringify(runs[runs.length - 1]));
  }
  await m.sample({ cls: 23, H: 128, W: 128, steps: 20, cfg: 3, circ: true, noise: ctx.inputs.noise });
  out[`img_${act}`] = await m.read(m.dBuf, 3 * 128 * 128);
  m.freePlan(); m.wbuf.destroy();
  ctx.progress(++k / 2, act);
}
return { runs, ...out };
