// GPU Bridge job: gradcheck (inputs = fixtures/gradcheck files as u8) and the training-size step bench.
// cfg = { sha, bench: true|false }
const { cfg } = ctx.inputs;
const gh = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/lifesci-ettin/web/`;
const { gradcheck } = await ctx.importModule(gh + "gradcheck.js");
const get = async (p) => {
  const k = p.replace("fixtures/gradcheck/", "").replace(/\.(bin|json)$/, "");
  if (!ctx.inputs[k]) throw new Error(`missing input ${k}`);
  return ctx.inputs[k];
};
const device = await ctx.gpu();
ctx.progress(0.1, "gradcheck");
const g = await gradcheck(device, get);
let b = null;
if (cfg.bench) {
  ctx.progress(0.5, "bench");
  const { bench } = await ctx.importModule(gh + "bench.js");
  b = await bench(device, null, { log: ctx.log });
}
return { gradcheck: g, bench: b };
