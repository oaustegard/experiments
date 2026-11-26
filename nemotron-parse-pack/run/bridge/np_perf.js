// GPU bridge job: encoder timing for Nemotron Parse 2.0 (pack v1) at full size, per kind of step, for a list of
// kernel variants. Inputs: png0 (u8 PNG bytes) and cfg {commit, variants: [{name, tune, layers}], probe}.
// Each variant runs encode({profile: true, layers}) (layers = how many of the 32 blocks to run; 0 = all) and
// reports wall ms per step kind. With cfg.probe the first 4096 rows of the residual stream after the run are
// compared against the first variant's (same layers), so a kernel change is checked on the real GPU too.
// The loaded weights are kept in ctx.cache and adopted by whichever engine commit runs next.
const cfg = ctx.inputs.cfg || {};
const url = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.commit}/nemotron-parse-pack/web/engine/nemotron.js`;
const { NemotronParse } = await ctx.importModule(url);
const out = { device: ctx.describe ? (await ctx.describe()).adapter : null, commit: cfg.commit, variants: [] };
let old = ctx.cache.get("np-v1"), m;
if (old) {   // same weights, this commit's code
  old.destroy?.();
  m = new NemotronParse(old.dev, old.m, {});
  m.w = old.w; m.cpu = old.cpu; m.tok = old.tok; m.size = { ...old.size };
} else {
  const t0 = performance.now();
  m = await NemotronParse.load({
    base: "https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/nemotron-parse-pack/v1/",
    device: await ctx.gpu(),
    onProgress: ({ phase, loaded, total }) => ctx.progress(total ? loaded / total : null, `load ${phase}`),
  });
  out.loadMs = performance.now() - t0;
}
ctx.cache.set("np-v1", m);
const errors = [];
m.dev.addEventListener("uncapturederror", (e) => { if (errors.length < 5) errors.push(e.error.message.slice(0, 400)); });
const bmp = await createImageBitmap(new Blob([ctx.inputs.png0], { type: "image/png" }));
const pre = m.preprocess(bmp);
let ref = null;
const vs = cfg.variants || [{ name: "default", tune: {}, layers: 2 }];
for (const [i, v] of vs.entries()) {
  m.setTune(v.tune || {});
  const rec = { name: v.name, tune: { ...m.tune }, layers: v.layers || 0 };
  try {
    const t0 = performance.now();
    m._plan(m.size.height, m.size.width);
    await m.dev.queue.onSubmittedWorkDone();
    rec.planMs = performance.now() - t0;
    if (v.warm !== false) await m.encode(pre.patches, { profile: true, layers: 1 });   // compile pipelines outside the timing
    const t1 = performance.now();
    rec.prof = await m.encode(pre.patches, { profile: true, layers: v.layers || 0,
      onStage: (d, t) => ctx.progress((i + d / t) / vs.length, `${v.name} ${d}/${t}`) });
    rec.ms = performance.now() - t1;
    for (const k in rec.prof) rec.prof[k] = Math.round(rec.prof[k]);
    if (cfg.probe) {
      const x = await m.probe("x", 4096 * 1280);
      let mx = 0, top = 0, dot = 0, na = 0, nb = 0, bad = 0;
      if (!ref) ref = x;
      for (let k = 0; k < x.length; k++) {
        if (!Number.isFinite(x[k])) bad++;
        mx = Math.max(mx, Math.abs(x[k] - ref[k])); top = Math.max(top, Math.abs(ref[k]));
        dot += x[k] * ref[k]; na += x[k] * x[k]; nb += ref[k] * ref[k];
      }
      rec.vsFirst = { maxRel: mx / top, cos: dot / Math.sqrt(na * nb), absMax: top, nonFinite: bad };
    }
  } catch (e) { rec.error = String(e.stack || e).slice(0, 600); }
  ctx.log(JSON.stringify(rec));
  out.variants.push(rec);
}
out.errors = errors;
return out;
