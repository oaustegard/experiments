// GPU-bridge job: per step, paint a 96x96 tile from the decoded splats and report the pixel spread, plus
// the range of splat precisions and colours. Finds where a texture "disappears" on a phone.
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { SplatDiffusion, f16ToF32, paintToBytes } = await import(url);
const device = await ctx.gpu();
const m = new SplatDiffusion(device, JSON.parse(td.decode(ctx.inputs.man)), f16ToF32(ctx.inputs.w));
const out = {};
for (const cls of [23, 9]) {
  const rows = [];
  await m.sample({ cls, steps: 24, cfg: 3, seed: 3, onStep: async (i) => {
    await device.queue.onSubmittedWorkDone();
    const sp = await m.splats();
    const px = await paintToBytes(device, m.b.splats, m.K, 96);
    let s = 0, s2 = 0, n = 0;
    for (let j = 0; j < px.length; j += 4) for (let c = 0; c < 3; c++) { s += px[j + c]; s2 += px[j + c] ** 2; n++; }
    let amax = 0, cmax = 0, cmin = 1e9, colmax = 0;
    for (let k = 0; k < m.K; k++) {
      const o = k * 8;
      amax = Math.max(amax, sp[o + 2]); cmax = Math.max(cmax, sp[o + 4]); cmin = Math.min(cmin, sp[o + 2], sp[o + 4]);
      colmax = Math.max(colmax, Math.abs(sp[o + 5]), Math.abs(sp[o + 6]), Math.abs(sp[o + 7]));
    }
    rows.push({ step: i + 1, mean: +(s / n).toFixed(1), std: +Math.sqrt(s2 / n - (s / n) ** 2).toFixed(1),
      amax: +amax.toFixed(1), cmax: +cmax.toFixed(1), minPrec: +cmin.toFixed(2), colmax: +colmax.toFixed(3) });
  } });
  out[cls] = rows;
  ctx.log(`cls ${cls}: ` + rows.map((r) => `${r.step}:${r.mean}/${r.std}/a${r.amax}/c${r.cmax}/col${r.colmax}`).join(" "));
}
return out;
