// GPU-bridge job: generate the texture bank, SEEDS samples of each of the 47 classes at S x S,
// with the page's sampler settings (q8 weights, 24 steps, cfg 3, circular padding).
// inputs: src (diffusion.js), man (manifest_q8.json), w (weights_q8.bin), cfg (@json {S, seeds, steps})
const td = new TextDecoder();
const url = URL.createObjectURL(new Blob([td.decode(ctx.inputs.src)], { type: "text/javascript" }));
const { TextureDiffusion, dequantQ8 } = await import(url);
const device = await ctx.gpu();
const man = JSON.parse(td.decode(ctx.inputs.man));
const wb = ctx.inputs.w;
const m = new TextureDiffusion(device, man, dequantQ8(man, wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.byteLength)));
const { S, seeds, steps } = ctx.inputs.cfg;
const C = man.names.length;
const out = new Uint8Array(C * seeds * S * S * 3);
const ms = [];
let k = 0;
for (let c = 0; c < C; c++) {
  for (let s = 0; s < seeds; s++) {
    const t0 = performance.now();
    await m.sample({ cls: c, H: S, W: S, steps, cfg: 3, seed: 1000 + s, circ: true });
    const d = await m.read(m.dBuf, 3 * S * S);
    ms.push(performance.now() - t0);
    const o = (c * seeds + s) * S * S * 3;
    for (let i = 0; i < S * S; i++) for (let ch = 0; ch < 3; ch++)
      out[o + 3 * i + ch] = Math.round(Math.min(1, Math.max(0, (d[ch * S * S + i] + 1) / 2)) * 255);
    ctx.progress(++k / (C * seeds), `${man.names[c]} ${s}`);
  }
}
ms.sort((a, b) => a - b);
return { bank: out, names: man.names, S, seeds, steps, median_ms: ms[ms.length >> 1], max_ms: ms[ms.length - 1],
  adapter: (await ctx.describe()).adapter };
