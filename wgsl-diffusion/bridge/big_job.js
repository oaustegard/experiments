// GPU-bridge job: can the Mac sample 512x512 (and 384)? Reports limits, buffer bytes, time.
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { TextureDiffusion, f16ToF32 } = await import(url);
const device = await ctx.gpu();
const errs = [];
device.addEventListener("uncapturederror", (e) => errs.push(e.error.message));
const lim = { maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize, maxBufferSize: device.limits.maxBufferSize };
const m = new TextureDiffusion(device, JSON.parse(td.decode(ctx.inputs.man)), f16ToF32(ctx.inputs.w));
const runs = [];
for (const [H, cfg] of [[384, 3], [512, 1], [512, 3]]) {
  try {
    const t0 = performance.now();
    await m.sample({ cls: 23, H, W: H, steps: 20, cfg, seed: 5, circ: true });
    await device.queue.onSubmittedWorkDone();
    const d = await m.read(m.dBuf, 3 * H * W_(H));
    runs.push({ H, cfg, total_ms: performance.now() - t0, bytes: m.bytes, maxBuf: Math.max(...m.allBufs.map((b) => b.n * 4)),
                finite: d.every(Number.isFinite), std: Math.sqrt(d.reduce((s, v) => s + v * v, 0) / d.length) });
  } catch (e) { runs.push({ H, cfg, error: e.message }); }
  ctx.log(JSON.stringify(runs[runs.length - 1]));
}
function W_(h) { return h; }
return { lim, runs, errs: errs.slice(0, 5) };
