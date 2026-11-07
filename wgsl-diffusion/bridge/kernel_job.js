// GPU-bridge job: conv kernel v1 vs v2 (split-K targets) — 20-step guided sampling time per size,
// plus a 128x128 sample from fixed noise per variant for a torch comparison.
// inputs: src (diffusion.js bytes), man (manifest.json bytes), w (weights.bin as u16), noise (1x3x128x128 f32)
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { TextureDiffusion, f16ToF32 } = await import(url);
const device = await ctx.gpu();
const man = JSON.parse(td.decode(ctx.inputs.man));
const w = f16ToF32(ctx.inputs.w);
const variants = [["v1", { kernel: "v1" }], ["v2-s0", { kernel: "v2", splitTarget: 0 }],
  ["v2-s256", { kernel: "v2", splitTarget: 256 }], ["v2-s512", { kernel: "v2", splitTarget: 512 }],
  ["v2-s1024", { kernel: "v2", splitTarget: 1024 }], ["v2-s2048", { kernel: "v2", splitTarget: 2048 }]];
const sizes = [64, 128, 256];
const runs = [], imgs = {};
let k = 0;
for (const [name, opts] of variants) {
  const m = new TextureDiffusion(device, man, w, opts);
  for (const H of sizes) {
    let best = Infinity;
    for (let rep = 0; rep < 3; rep++) {
      const t0 = performance.now();
      await m.sample({ cls: 23, H, W: H, steps: 20, cfg: 3, seed: 5, circ: true });
      await device.queue.onSubmittedWorkDone();
      if (rep > 0) best = Math.min(best, performance.now() - t0);
    }
    runs.push({ variant: name, H, total_ms: best, step_ms: best / 20 });
    ctx.log(JSON.stringify(runs[runs.length - 1]));
  }
  await m.sample({ cls: 23, H: 128, W: 128, steps: 20, cfg: 3, circ: true, noise: ctx.inputs.noise });
  imgs[name] = await m.read(m.dBuf, 3 * 128 * 128);
  m.freePlan(); m.wbuf.destroy();
  ctx.progress(++k / variants.length, name);
}
return { runs, ...Object.fromEntries(Object.entries(imgs).map(([k2, v]) => [`img_${k2.replace(/-/g, "_")}`, v])) };
