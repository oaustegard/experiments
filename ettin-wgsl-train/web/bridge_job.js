// GPU Bridge job body: web/traj.js's training run on a real GPU, with the code at a pushed commit.
// Inputs: the fixtures/traj files as u8 (meta, <set>_ids, <set>_lens, <set>_pos, <set>_gold, params0),
// cfg = { sha, smoke, epochs }. The packed model comes from jsDelivr at the same commit.
const { cfg } = ctx.inputs;
const gh = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`;
ctx.progress(0.01, "loading code");
const { trajectory } = await ctx.importModule(gh + "ettin-wgsl-train/web/traj.js");
const cache = {};
const get = async (p) => {
  if (p.startsWith("model/")) return (cache[p] ??= await ctx.fetchBytes(gh + "bsky-account-decider/web/models/account/" + p.slice(6)));
  const k = p.replace("fixtures/traj/", "").replace(/\.(bin|json)$/, "");
  if (!ctx.inputs[k]) throw new Error(`missing input ${k}`);
  return ctx.inputs[k];
};
const device = await ctx.gpu();
const r = await trajectory(device, get, { smoke: !!cfg.smoke, epochs: cfg.epochs || 0, log: (m) => ctx.log(m), progress: (f, l) => ctx.progress(f, l) });
// typed arrays at the top level become result files
const out = { ...r, epochs: r.epochs.map(({ val, test, reply_thread, ...e }) => ({ ...e, val: strip(val), test: strip(test), reply_thread: strip(reply_thread) })) };
for (const e of r.epochs) for (const k of ["val", "test", "reply_thread"]) if (e[k].probs) out[`probs_${k}`] = e[k].probs;
return out;

function strip({ probs, ...m }) { return m; }
