// GPU bridge job: Nemotron 3.5 Parse (pack v35i8, all int8) on a real GPU. Inputs: png0 (u8 PNG bytes) and
// cfg {commit, base?, runs: [{name, size?: {height, width}, maxTokens?, profile?}]}.
// Each run parses the page end to end (ms.encode includes ms.prefill; decode is per token after that) at the
// given maximum canvas (default the pack's 2048 x 1664), or with profile: true times the encoder per kind of
// step. The loaded weights stay in ctx.cache under "np-v35i8" and are adopted by whichever engine commit runs next.
const cfg = ctx.inputs.cfg || {};
const url = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.commit}/nemotron-parse-pack/web/engine/nemotron.js`;
const { NemotronParse } = await ctx.importModule(url);
const out = { commit: cfg.commit, runs: [] };
let old = ctx.cache.get("np-v35i8"), m;
if (old) {
  old.destroy?.();
  m = new NemotronParse(old.dev, old.m, {});
  m.w = old.w; m.cpu = old.cpu; m.tok = old.tok; m.size = { ...old.size };
} else {
  const t0 = performance.now();
  m = await NemotronParse.load({
    base: cfg.base || "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/nemotron-parse-pack/v35i8/",
    device: await ctx.gpu(),
    onProgress: ({ phase, loaded, total }) => ctx.progress(total ? loaded / total : null, `load ${phase}`),
  });
  out.loadMs = performance.now() - t0;
}
ctx.cache.set("np-v35i8", m);
const errors = [];
m.dev.addEventListener("uncapturederror", (e) => { if (errors.length < 5) errors.push(e.error.message.slice(0, 400)); });
const bmp = await createImageBitmap(new Blob([ctx.inputs.png0], { type: "image/png" }));
const full = { ...m.m.image };
const runs = cfg.runs || [{ name: "full", maxTokens: 4096 }];
for (const [i, r] of runs.entries()) {
  const rec = { name: r.name };
  try {
    m.size = r.size ? { height: r.size.height, width: r.size.width } : { height: full.height, width: full.width };
    if (r.profile) {
      const pre = m.preprocess(bmp);
      await m.encode(pre, { profile: true, layers: 1 });   // compile outside the timing
      const t0 = performance.now();
      rec.prof = await m.encode(pre, { profile: true, onStage: (d, t) => ctx.progress((i + d / t) / runs.length, `${r.name} ${d}/${t}`) });
      rec.ms = performance.now() - t0;
      for (const k in rec.prof) rec.prof[k] = Math.round(rec.prof[k]);
      rec.grid = [pre.H, pre.W, pre.ch, pre.cw];
    } else {
      let n = 0;
      const p = await m.parse(bmp, { maxTokens: r.maxTokens || 4096,
        onStage: (d, t) => ctx.progress((i + d / t * 0.5) / runs.length, `${r.name} encode ${d}/${t}`),
        onToken: () => { if (++n % 50 === 0) ctx.progress(null, `${r.name} token ${n}`); } });
      Object.assign(rec, { ms: p.ms, ntok: p.tokens.length, tokens: p.tokens, raw: p.raw, blocks: p.blocks.length,
        tokPerSec: p.tokens.length / (p.ms.decode / 1000), prefixRows: m.plan.Te, encTokens: m.plan.T });
    }
  } catch (e) { rec.error = String(e.stack || e).slice(0, 600); }
  ctx.log(JSON.stringify({ ...rec, tokens: undefined, raw: undefined }));
  out.runs.push(rec);
}
out.errors = errors;
return out;
