// MoEdernColBERT 1.5M and 27M on the bridge's GPU, with the EG2 k=4 cut and Ettin-17m engines as
// same-session references. Loads moederncolbert-bench/web/engine.js (and the two reference engines)
// from jsDelivr at cfg.sha, the MoEdern weights straight from Hugging Face.
//   parity: every token row of web/fixtures.json against run/mcb.py (inputs exp15, exp27)
//   query:  one 32-token ColBERT query per embed(), readback included; per query the median of
//           `reps`, then the median over the 16 queries
//   q16:    16 queries in one embed()
//   docs:   the 64 fixture documents packed into batches of <= cfg.docTokens tokens, `rounds` passes
// inputs.cfg = { sha, reps: 30, rounds: 5, docTokens: 8192 }
const cfg = ctx.inputs.cfg;
const GH = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`;
const HF = "https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-";
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
const out = { adapter: { vendor: d.vendor, architecture: d.architecture, description: d.description }, cfg, models: {} };

const fx = await json(GH + "moederncolbert-bench/web/fixtures.json");
const { MoEdernEngine } = await import(GH + "moederncolbert-bench/web/engine.js");
const seqs = fx.queries.concat(fx.docs);
for (const [name, exp] of [["1.5M", ctx.inputs.exp15], ["27M", ctx.inputs.exp27]]) {
  ctx.progress(0, `${name}: loading`);
  const [config, model, dense] = await Promise.all([json(`${HF}${name}/resolve/main/config.json`),
    bytes(`${HF}${name}/resolve/main/model.safetensors`), bytes(`${HF}${name}/resolve/main/1_Dense/model.safetensors`)]);
  let t = performance.now();
  const eng = await MoEdernEngine.create(dev, { config, dense }, model, { maxTokens: cfg.docTokens, maxSeqs: 64 });
  await eng.embed([fx.queries[0]]);
  const res = { initMs: performance.now() - t, weightMB: eng.weightBytes / 1e6 };
  // parity
  let row = 0, minCos = 1;
  for (const s of seqs) {
    const v = await eng.embed([s]);
    for (let r = 0; r < s.length; r++) {
      let dd = 0, x = 0, y = 0;
      for (let i = 0; i < 128; i++) { const a = v[r * 128 + i], b = exp[(row + r) * 128 + i]; dd += a * b; x += a * a; y += b * b; }
      minCos = Math.min(minCos, dd / Math.sqrt(x * y));
    }
    row += s.length;
  }
  res.parityMinCos = minCos;
  // single query latency
  ctx.progress(0.3, `${name}: query latency`);
  const per = fx.queries.map(() => []);
  for (let k = 0; k < cfg.reps; k++) for (let i = 0; i < fx.queries.length; i++) {
    t = performance.now(); await eng.embed([fx.queries[i]]); per[i].push(performance.now() - t);
  }
  res.queryMs = +median(per.map(median)).toFixed(3);
  const q16 = [];
  for (let k = 0; k < cfg.reps; k++) { t = performance.now(); await eng.embed(fx.queries); q16.push(performance.now() - t); }
  res.q16Ms = +median(q16).toFixed(3);
  // document throughput
  ctx.progress(0.6, `${name}: documents`);
  const batches = []; let cur = [], n = 0;
  for (const s of fx.docs) { if (n + s.length > cfg.docTokens || cur.length === 64) { batches.push(cur); cur = []; n = 0; } cur.push(s); n += s.length; }
  if (cur.length) batches.push(cur);
  const docTok = fx.docs.reduce((a, s) => a + s.length, 0);
  for (const bt of batches) await eng.embed(bt);
  const passes = [];
  for (let r = 0; r < cfg.rounds; r++) {
    t = performance.now();
    for (const bt of batches) await eng.embed(bt);
    passes.push(performance.now() - t);
  }
  const ms = median(passes);
  Object.assign(res, { docPassMs: +ms.toFixed(2), docsPerSec: +(fx.docs.length / ms * 1000).toFixed(1),
    tokensPerSec: Math.round(docTok / ms * 1000), nDocs: fx.docs.length, docTokens: docTok, nBatches: batches.length });
  ctx.log(name, JSON.stringify(res));
  out.models[name] = res;
  eng.destroy();
}

// references: the shipping EG2 cut and Ettin-17m engines, short-query latency as in wgsl-ab
const refs = [
  { name: "eg2-cut", file: "eg2-wgsl-engine/web/engine.js", cls: "Eg2Engine",
    manifest: `${GH}eg2-wgsl-engine/web/manifest.json`, data: `${SITE}image-search/model/ort/model_t3w.onnx_data`,
    ids: (await json(`${GH}eg2-wgsl-engine/web/queries.json`)).map((x) => x.ids) },
  { name: "ettin-17m-t4", file: "ettin-native/web/engine.js", cls: "SmallEngine",
    manifest: `${SITE}embedders/models/ettin/manifest.json`, data: `${SITE}embedders/models/ettin/model.onnx`,
    ids: (await json(`${GH}ettin-native/web/models/e17c-t4-w8L63/fixtures.json`)).map((x) => x.ids) },
];
for (const m of refs) {
  ctx.progress(0.9, `${m.name}: reference`);
  const [man, data] = await Promise.all([json(m.manifest), bytes(m.data)]);
  const mod = await import(GH + m.file);
  const eng = await mod[m.cls].create(dev, man, data, { maxTokens: 512, maxSeqs: 16 });
  await eng.embed([m.ids[0]]);
  const per = m.ids.map(() => []);
  for (let k = 0; k < cfg.reps; k++) for (let i = 0; i < m.ids.length; i++) {
    const t = performance.now(); await eng.embed([m.ids[i]]); per[i].push(performance.now() - t);
  }
  const short = m.ids.map((x, i) => [x.length, i]).filter(([n]) => n <= 20).map(([, i]) => i);
  out.models[m.name] = { shortQueryMs: +median(short.map((i) => median(per[i]))).toFixed(3), nShort: short.length };
  // the MoEdern query is always 32 tokens: time the reference on its queries nearest 32 tokens too
  const near = m.ids.map((x, i) => [Math.abs(x.length - 32), i]).sort((a, b) => a[0] - b[0]).slice(0, 5).map(([, i]) => i);
  out.models[m.name].near32Ms = +median(near.map((i) => median(per[i]))).toFixed(3);
  out.models[m.name].near32Tokens = near.map((i) => m.ids[i].length);
  ctx.log(m.name, JSON.stringify(out.models[m.name]));
  eng.destroy();
}
return out;
