// GPU-bridge job for a phone: where does the 1,024-splat page fail? Times each sampling step, then painting
// a cached 1024^2 tile, then painting a phone-sized screen directly (every pixel x every splat).
// inputs: src (splatdiff.js bytes), man (manifest.json bytes), w (weights.bin u16)
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { SplatDiffusion, f16ToF32, paintToBytes } = await import(url);
const device = await ctx.gpu();
const errs = [];
device.addEventListener("uncapturederror", (e) => errs.push(e.error.message));
let lost = null;
device.lost.then((i) => { lost = i.message; });
const m = new SplatDiffusion(device, JSON.parse(td.decode(ctx.inputs.man)), f16ToF32(ctx.inputs.w));
const out = { K: m.K, steps: [], paint: [], errs, lost: null };
let t = performance.now();
await m.sample({ cls: 23, steps: 24, cfg: 3, seed: 3, onStep: async (i) => {
  await device.queue.onSubmittedWorkDone();
  const n = performance.now(); out.steps.push(n - t); t = n;
  ctx.progress((i + 1) / 30, `step ${i + 1}`);
} });
ctx.log(`sampling done: ${out.steps.reduce((a, b) => a + b, 0).toFixed(0)} ms`);
for (const size of [512, 1024, 1600]) {
  const t0 = performance.now();
  await paintToBytes(device, m.b.splats, m.K, size);
  out.paint.push({ size, ms: performance.now() - t0, lost });
  ctx.log(JSON.stringify(out.paint[out.paint.length - 1]));
}
out.lost = lost;
return out;
