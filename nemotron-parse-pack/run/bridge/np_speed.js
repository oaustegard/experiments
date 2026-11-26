// GPU bridge job: Nemotron Parse 2.0 (pack v1) at full size on a real GPU. Loads the engine from jsDelivr at a
// pinned commit, keeps the loaded model in ctx.cache across jobs, parses each input PNG and returns tokens,
// raw text and timings. Inputs: png0, png1, ... (u8 PNG bytes) and cfg {commit, maxTokens}.
const cfg = ctx.inputs.cfg || {};
const url = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.commit}/nemotron-parse-pack/web/engine/nemotron.js`;
const { NemotronParse } = await ctx.importModule(url);
let m = ctx.cache.get("np-v1");
const out = { device: ctx.describe ? (await ctx.describe()).adapter : null, pages: [] };
if (!m) {
  const t0 = performance.now();
  m = await NemotronParse.load({
    base: "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/nemotron-parse-pack/v1/",
    device: await ctx.gpu(),
    onProgress: ({ phase, loaded, total }) => ctx.progress(total ? loaded / total : null, `load ${phase}`),
  });
  out.loadMs = performance.now() - t0;
  ctx.cache.set("np-v1", m);
  ctx.log("loaded in", Math.round(out.loadMs), "ms");
}
const names = Object.keys(ctx.inputs).filter(k => k.startsWith("png")).sort();
for (const [i, k] of names.entries()) {
  const bmp = await createImageBitmap(new Blob([ctx.inputs[k]], { type: "image/png" }));
  let n = 0;
  const t0 = performance.now();
  const r = await m.parse(bmp, {
    maxTokens: cfg.maxTokens || 4096,
    onStage: (d, t) => ctx.progress((i + d / t * 0.3) / names.length, `${k} encode ${d}/${t}`),
    onToken: () => { if (++n % 50 === 0) ctx.progress(null, `${k} token ${n}`); },
  });
  const rec = { name: k, w: bmp.width, h: bmp.height, ms: r.ms, totalMs: performance.now() - t0,
                ntok: r.tokens.length, tokens: r.tokens, raw: r.raw, blocks: r.blocks.length };
  ctx.log(k, JSON.stringify({ ms: r.ms, ntok: rec.ntok, blocks: rec.blocks }));
  out.pages.push(rec);
}
return out;
