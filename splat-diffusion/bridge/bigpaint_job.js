// GPU-bridge job: does painting at screen size go wrong on a phone? Per step, paint 96, 1024 and 1600 px
// tiles of the same splats and report each one's pixel mean and spread.
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { SplatDiffusion, f16ToF32, paintToBytes } = await import(url);
const device = await ctx.gpu();
const m = new SplatDiffusion(device, JSON.parse(td.decode(ctx.inputs.man)), f16ToF32(ctx.inputs.w));
const stat = (px) => {
  let s = 0, s2 = 0, n = 0;
  for (let j = 0; j < px.length; j += 4) for (let c = 0; c < 3; c++) { s += px[j + c]; s2 += px[j + c] ** 2; n++; }
  return `${(s / n).toFixed(1)}/${Math.sqrt(s2 / n - (s / n) ** 2).toFixed(1)}`;
};
const rows = [];
await m.sample({ cls: 9, steps: 24, cfg: 3, seed: 3, onStep: async (i) => {
  if (i % 3 !== 2) return;
  await device.queue.onSubmittedWorkDone();
  const r = [i + 1];
  for (const size of [96, 1024, 1600]) {
    const t0 = performance.now();
    const px = await paintToBytes(device, m.b.splats, m.K, size);
    r.push(`${size}:${stat(px)}@${(performance.now() - t0).toFixed(0)}ms`);
  }
  rows.push(r.join(" "));
  ctx.log(rows[rows.length - 1]);
} });
return { rows };
