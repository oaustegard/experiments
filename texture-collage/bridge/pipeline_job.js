// GPU-bridge job: time the whole collage pipeline on real hardware, the way the page runs it:
// segment, describe, choose, generate one fresh texture per region, render.
// inputs: csrc (collage.js), dsrc (diffusion.js), man (manifest_q8.json), w (weights_q8.bin),
//         table (classes.json bytes), img (RGBA u8), cfg (@json {W, H, K, sizes, steps})
const td = new TextDecoder();
const mod = (b) => import(URL.createObjectURL(new Blob([td.decode(b)], { type: "text/javascript" })));
const C = await mod(ctx.inputs.csrc);
const { TextureDiffusion, dequantQ8 } = await mod(ctx.inputs.dsrc);
const device = await ctx.gpu();
const man = JSON.parse(td.decode(ctx.inputs.man));
const wb = ctx.inputs.w;
let t = performance.now();
const m = new TextureDiffusion(device, man, dequantQ8(man, wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.byteLength)));
const tModel = performance.now() - t;
const table = JSON.parse(td.decode(ctx.inputs.table));
const { W, H, K, sizes, steps } = ctx.inputs.cfg;
const N = W * H, out = { W, H, K, steps, model_init_ms: tModel, runs: [] };
const lab = C.rgbToLab(ctx.inputs.img, N, 4);
for (let rep = 0; rep < 2; rep++) {            // rep 0 warms the JIT
  const r = {};
  t = performance.now(); const seg = C.segment(lab, W, H, { K }); r.segment_ms = performance.now() - t;
  t = performance.now(); const rdesc = C.regionDescriptors(lab, seg); r.describe_ms = performance.now() - t;
  const regions = rdesc.map((d, k) => ({ desc: d, stats: seg.stats[k], frac: seg.stats[k].frac }));
  t = performance.now(); const pick = C.chooseClasses(regions, table, { mode: "structure" }); r.choose_ms = performance.now() - t;
  r.K = seg.K; r.classes = pick.map((p) => table.names[p.cls]);
  out.seg = r;
  if (rep === 0) continue;
  out.runs.push(r);
  for (const S of sizes) {
    const tiles = [], per = [];
    for (let k = 0; k < seg.K; k++) {
      t = performance.now();
      await m.sample({ cls: pick[k].cls, H: S, W: S, steps, cfg: 3, seed: 77 + k, circ: true });
      const d = await m.read(m.dBuf, 3 * S * S);
      per.push(performance.now() - t);
      const u = new Uint8Array(3 * S * S);
      for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) u[3 * i + c] = Math.round(Math.min(1, Math.max(0, (d[c * S * S + i] + 1) / 2)) * 255);
      tiles.push(C.prepTile(u, S, pick[k].scale));
    }
    t = performance.now();
    const lab2 = C.render(lab, seg, tiles, { scales: pick.map((p) => p.scale), rdesc });
    const rgba = C.labToRgba(lab2, N);
    const render_ms = performance.now() - t;
    out[`S${S}`] = { textures_ms: per.reduce((a, b) => a + b, 0), per_texture_ms: per, render_ms };
    if (S === 128) out.collage128 = new Uint8Array(rgba.buffer);
    ctx.log(`S=${S}: ${per.map((x) => x.toFixed(0)).join(" ")} ms, render ${render_ms.toFixed(0)} ms`);
    ctx.progress(sizes.indexOf(S) / sizes.length + 0.3, `size ${S}`);
  }
}
out.adapter = (await ctx.describe()).adapter;
return out;
