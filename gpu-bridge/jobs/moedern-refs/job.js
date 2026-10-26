// Same-device references for moedern: the shipping EG2 k=4 cut and Ettin-17m t4 WGSL engines, timed
// the way jobs/moedern times MoEdernColBERT (back-to-back embed() loops divided by the count, since
// the worker's clock ticks in ~1 ms). Per engine: its short queries (<= 20 tokens), its queries
// nearest 32 tokens (the MoEdern query length), and the longest.
// inputs.cfg = { sha, reps: 20, rounds: 5 }
const cfg = ctx.inputs.cfg;
const GH = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`;
const SITE = "https://austegard.com/bsky/";
const dev = await ctx.gpu();
const bytes = async (url) => {
  if (ctx.cache.has(url)) return ctx.cache.get(url);
  const b = await ctx.fetchBytes(url);
  ctx.cache.set(url, b);
  return b;
};
const json = async (url) => JSON.parse(new TextDecoder().decode(await bytes(url)));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const d = await ctx.describe();
const out = { adapter: { vendor: d.vendor, architecture: d.architecture, description: d.description }, ua: navigator.userAgent, cfg, models: {} };
const refs = [
  { name: "eg2-cut", file: "eg2-wgsl-engine/web/engine.js", cls: "Eg2Engine",
    manifest: `${GH}eg2-wgsl-engine/web/manifest.json`, data: `${SITE}image-search/model/ort/model_t3w.onnx_data`,
    ids: (await json(`${GH}eg2-wgsl-engine/web/queries.json`)).map((x) => x.ids) },
  { name: "ettin-17m-t4", file: "ettin-native/web/engine.js", cls: "SmallEngine",
    manifest: `${SITE}embedders/models/ettin/manifest.json`, data: `${SITE}embedders/models/ettin/model.onnx`,
    ids: (await json(`${GH}ettin-native/web/models/e17c-t4-w8L63/fixtures.json`)).map((x) => x.ids) },
];
for (const [j, m] of refs.entries()) {
  ctx.progress(j / refs.length, `${m.name}`);
  const [man, data] = await Promise.all([json(m.manifest), bytes(m.data)]);
  const mod = await import(GH + m.file);
  const eng = await mod[m.cls].create(dev, man, data, { maxTokens: 512, maxSeqs: 16 });
  await eng.embed([m.ids[0]]);
  const time = async (idx) => {
    const ts = [];
    for (let r = 0; r < cfg.rounds; r++) {
      const t0 = performance.now(), n = idx.length * cfg.reps;
      for (let k = 0; k < n; k++) await eng.embed([m.ids[idx[k % idx.length]]]);
      ts.push((performance.now() - t0) / n);
    }
    return +median(ts).toFixed(3);
  };
  const all = m.ids.map((x, i) => i);
  const short = all.filter((i) => m.ids[i].length <= 20);
  const near = [...all].sort((a, b) => Math.abs(m.ids[a].length - 32) - Math.abs(m.ids[b].length - 32)).slice(0, 3);
  const longest = all.reduce((b, i) => (m.ids[i].length > m.ids[b].length ? i : b), 0);
  out.models[m.name] = { shortMs: await time(short), nShort: short.length, near32Ms: await time(near),
    near32Tokens: near.map((i) => m.ids[i].length), longMs: await time([longest]), longTokens: m.ids[longest].length };
  ctx.log(m.name, JSON.stringify(out.models[m.name]));
  eng.destroy();
}
return out;
