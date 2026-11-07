// GPU-bridge job: time the sampler on real hardware and return one image for a torch comparison.
// inputs: src (diffusion.js bytes), man (manifest.json bytes), w (weights.bin as u16), noise (1x3x128x128 f32)
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { TextureDiffusion, f16ToF32 } = await import(url);
const device = await ctx.gpu();
const man = JSON.parse(td.decode(ctx.inputs.man));
const m = new TextureDiffusion(device, man, f16ToF32(ctx.inputs.w));
const info = ctx.describe ? await ctx.describe() : null;
const runs = [];
const configs = [
  { H: 64, cfg: 3, steps: 20 }, { H: 64, cfg: 1, steps: 20 }, { H: 128, cfg: 3, steps: 20 }, { H: 128, cfg: 1, steps: 20 },
  { H: 192, cfg: 3, steps: 20 }, { H: 256, cfg: 3, steps: 20 }];
let k = 0;
for (const c of configs) {
  for (let rep = 0; rep < 2; rep++) {          // first run warms pipelines and buffers
    const t0 = performance.now();
    await m.sample({ cls: 23, H: c.H, W: c.H, steps: c.steps, cfg: c.cfg, seed: 5, circ: true });
    await device.queue.onSubmittedWorkDone();
    if (rep === 1) {
      const total = performance.now() - t0;
      runs.push({ ...c, total_ms: total, step_ms: total / c.steps, bytes: m.bytes });
      ctx.log(JSON.stringify(runs[runs.length - 1]));
    }
  }
  ctx.progress(++k / (configs.length + 1), `size ${c.H}`);
}
// correctness on this GPU: fixed noise, marbled, 20 steps, cfg 3, circular
await m.sample({ cls: 23, H: 128, W: 128, steps: 20, cfg: 3, circ: true, noise: ctx.inputs.noise });
await device.queue.onSubmittedWorkDone();
const img = await m.read(m.dBuf, 3 * 128 * 128);
ctx.progress(1, "done");
return { runs, img, adapter: info && info.adapter };
