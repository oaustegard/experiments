// Query latency of every hand-written WGSL encoder at one commit, for the speed/accuracy chart in
// eg2-wgsl-engine/RESULTS.md. Loads each engine from jsDelivr at cfg.sha and the weights the live
// site serves (Hugging Face for full EG2 and MoEdernColBERT). Two timings per model, both one query
// per embed() with readback, back-to-back loops divided by the count (the worker clock ticks in ~1 ms):
//   own:  the model's real fixture queries of <= 32 tokens (prompt prefix included where the model
//         uses one; ColBERT queries are always 32), cycled
//   t32:  one 32-token sequence made by tiling the first fixture query's ids, for a like-for-like row
// inputs.cfg = { sha, reps: 20, rounds: 5, data: sha the CDN already serves (defaults to sha) }
const cfg = ctx.inputs.cfg;
const GH = "https://cdn.jsdelivr.net/gh/oaustegard/experiments@";
const SITE = "https://austegard.com/bsky/";
const HFG = "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/onnx/model_q4.onnx_data";
const HFM = "https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-";
const dev = await ctx.gpu();
const bytes = async (url) => {
  if (ctx.cache.has(url)) return ctx.cache.get(url);
  const b = await ctx.fetchBytes(url);
  ctx.cache.set(url, b);
  return b;
};
const json = async (url) => JSON.parse(new TextDecoder().decode(await bytes(url)));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const code = `${GH}${cfg.sha}/`, data = `${GH}${cfg.data ?? cfg.sha}/`;
const ids = async (url) => (await json(url)).map((x) => x.ids);

const models = [
  { name: "eg2-cut", file: "eg2-wgsl-engine/web/engine.js", cls: "Eg2Engine",
    load: async () => [await json(data + "eg2-wgsl-engine/web/manifest.json"), await bytes(SITE + "image-search/model/ort/model_t3w.onnx_data")],
    queries: () => ids(data + "eg2-wgsl-engine/web/queries.json") },
  { name: "eg2-full", file: "eg2-wgsl-engine/web/engine.js", cls: "Eg2Engine",
    load: async () => [await json(data + "eg2-wgsl-engine/web/manifest_full.json"), await bytes(HFG)],
    queries: () => ids(data + "eg2-wgsl-engine/web/queries_full.json") },
  { name: "ettin-17m", file: "ettin-native/web/engine.js", cls: "SmallEngine",
    load: async () => [await json(SITE + "embedders/models/ettin/manifest.json"), await bytes(SITE + "embedders/models/ettin/model.onnx")],
    queries: () => ids(data + "ettin-native/web/models/e17c-t4-w8L63/fixtures.json") },
  { name: "leaf-ir", file: "ettin-native/web/engine.js", cls: "SmallEngine",
    load: async () => [await json(SITE + "embedders/models/leaf/manifest.json"), await bytes(SITE + "embedders/models/leaf/model.onnx")],
    queries: () => ids(data + "ettin-native/web/models/leafirc-t4-w8L63/fixtures.json") },
];
for (const size of ["1.5M", "27M"]) models.push({ name: `moedern-${size}`, file: "moederncolbert-bench/web/engine.js", cls: "MoEdernEngine",
  opts: { kernel: "template" },  // the 8-row kernel is the faster one for single queries
  load: async () => [{ config: await json(`${HFM}${size}/resolve/main/config.json`), dense: await bytes(`${HFM}${size}/resolve/main/1_Dense/model.safetensors`) },
    await bytes(`${HFM}${size}/resolve/main/model.safetensors`)],
  queries: async () => (await json(data + "moederncolbert-bench/web/fixtures.json")).queries });

const d = await ctx.describe();
const out = { adapter: { vendor: d.vendor, architecture: d.architecture, description: d.description }, ua: navigator.userAgent, cfg, models: {} };
const loop = async (fn, n) => { const ts = []; for (let r = 0; r < cfg.rounds; r++) { const t0 = performance.now(); for (let k = 0; k < n; k++) await fn(k); ts.push((performance.now() - t0) / n); } return +median(ts).toFixed(3); };
for (const [i, m] of models.entries()) {
  ctx.progress(i / models.length, `${m.name}: loading`);
  const [man, w] = await m.load();
  const qs = await m.queries();
  const mod = await import(code + m.file);
  const eng = await mod[m.cls].create(dev, man, w, { maxTokens: 512, maxSeqs: 16, ...(m.opts ?? {}) });
  const own = qs.filter((q) => q.length <= 32);
  const t32 = Array.from({ length: 32 }, (_, k) => qs[0][k % qs[0].length]);
  await eng.embed([own[0]]); await eng.embed([t32]);
  ctx.progress((i + 0.5) / models.length, `${m.name}: timing`);
  const res = {
    ownMs: await loop((k) => eng.embed([own[k % own.length]]), own.length * cfg.reps),
    ownTokensMedian: median(own.map((q) => q.length)), nOwn: own.length,
    t32Ms: await loop(() => eng.embed([t32]), cfg.reps * 4),
  };
  ctx.log(m.name, JSON.stringify(res));
  out.models[m.name] = res;
  eng.destroy();
}
return out;
