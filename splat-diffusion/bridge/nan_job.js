// GPU-bridge job: count NaNs in the x0 estimate after every step, with the old and the GELU-clamped runtime.
// inputs: old, new (splatdiff.js bytes), man, w
const td = new TextDecoder();
const device = await ctx.gpu();
const man = JSON.parse(td.decode(ctx.inputs.man));
const out = {};
for (const name of ["old", "new"]) {
  const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs[name])], { type: "text/javascript" }));
  const { SplatDiffusion, f16ToF32 } = await import(url);
  const m = new SplatDiffusion(device, man, f16ToF32(ctx.inputs.w));
  const rows = [];
  for (const cls of [23, 42, 9, 44]) {
    const nan = [];
    await m.sample({ cls, steps: 24, cfg: 3, seed: 3, onStep: async () => {
      await device.queue.onSubmittedWorkDone();
      const x = await m.read(m.b.X0, m.K * 10);
      nan.push(x.reduce((c, v) => c + (Number.isFinite(v) ? 0 : 1), 0));
    } });
    rows.push({ cls, nan_per_step: nan });
    ctx.log(`${name} cls ${cls}: ${nan.join(",")}`);
  }
  out[name] = rows;
  m.free(); m.wbuf.destroy();
}
return out;
