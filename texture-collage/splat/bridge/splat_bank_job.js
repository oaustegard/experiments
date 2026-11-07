// GPU-bridge job: the splat texture bank. SEEDS samples of each of the 47 classes from splat-diffusion
// (model512: 512 splats), 24 guided steps, guidance 3, as splat-diffusion's page samples them.
// inputs: src (splatdiff.js), man (manifest.json), w (weights.bin, f16 as u16), cfg (@json {seeds, steps})
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { SplatDiffusion, f16ToF32 } = await import(url);
const device = await ctx.gpu();
const man = JSON.parse(td.decode(ctx.inputs.man));
const m = new SplatDiffusion(device, man, f16ToF32(ctx.inputs.w));
const { seeds, steps } = ctx.inputs.cfg, C = man.nclass, K = m.K;
const out = new Float32Array(C * seeds * K * 12);
const ms = [];
let n = 0;
for (let c = 0; c < C; c++) for (let s = 0; s < seeds; s++) {
  const t0 = performance.now();
  await m.sample({ cls: c, steps, cfg: 3, seed: 2000 + s });
  const sp = await m.splats();
  ms.push(performance.now() - t0);
  out.set(sp, (c * seeds + s) * K * 12);
  ctx.progress(++n / (C * seeds), `${c} ${s}`);
}
ms.sort((a, b) => a - b);
return { splats: out, K, seeds, steps, C, median_ms: ms[ms.length >> 1], first_ms: ms[0], max_ms: ms[ms.length - 1] };
