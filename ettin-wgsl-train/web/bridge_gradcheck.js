// GPU Bridge job body: web/gradcheck.js on a real GPU. Inputs: the fixtures/gradcheck files as u8
// (meta, ids, lens, pos, seg, gold, params0, grads, params1), cfg = { sha }.
const { cfg } = ctx.inputs;
const gh = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`;
const { gradcheck } = await ctx.importModule(gh + "ettin-wgsl-train/web/gradcheck.js");
const cache = {};
const get = async (p) => {
  if (p.startsWith("model/")) return (cache[p] ??= await ctx.fetchBytes(gh + "bsky-account-decider/web/models/account/" + p.slice(6)));
  const k = p.replace("fixtures/gradcheck/", "").replace(/\.(bin|json)$/, "");
  if (!ctx.inputs[k]) throw new Error(`missing input ${k}`);
  return ctx.inputs[k];
};
return await gradcheck(await ctx.gpu(), get);
