// A/B of the hand-written WGSL encoders at two commits of oaustegard/experiments, on the bridge's GPU.
// Loads eg2-wgsl-engine/web/engine.js and ettin-native/web/engine.js at both shas from jsDelivr (so each
// version gets its own relative imports), the weights from austegard.com and Hugging Face, and times
// embed() per query the way eg2-wgsl-engine/web/bench.js does: per query the median of `reps` runs,
// readback included, then the median over the queries of <= 20 tokens. Rounds alternate A and B so
// clock and thermal drift hit both. Also reports the largest |A - B| over every output vector.
// inputs.cfg = { a: sha, b: sha, reps: 20, rounds: 3, full: false }
const cfg = ctx.inputs.cfg;
const GH = "https://cdn.jsdelivr.net/gh/oaustegard/experiments@";
const SITE = "https://austegard.com/bsky/";
const HF = "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/onnx/model_q4.onnx_data";
const dev = await ctx.gpu();
const bytes = async (url) => {
  if (ctx.cache.has(url)) return ctx.cache.get(url);
  const b = await ctx.fetchBytes(url);   // reports download progress on the page's bar
  ctx.cache.set(url, b);
  return b;
};
const json = async (url) => JSON.parse(new TextDecoder().decode(await bytes(url)));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };

// queries come from the newer sha's fixtures; their ids do not depend on the engine
const q = cfg.data ?? cfg.a; // manifests and fixtures; engine modules load from cfg.a and cfg.b
const models = [
  { name: "eg2-cut", file: "eg2-wgsl-engine/web/engine.js", cls: "Eg2Engine",
    manifest: `${GH}${q}/eg2-wgsl-engine/web/manifest.json`, data: `${SITE}image-search/model/ort/model_t3w.onnx_data`,
    ids: (await json(`${GH}${q}/eg2-wgsl-engine/web/queries.json`)).map((x) => x.ids) },
  { name: "ettin-17m-t4", file: "ettin-native/web/engine.js", cls: "SmallEngine",
    manifest: `${SITE}embedders/models/ettin/manifest.json`, data: `${SITE}embedders/models/ettin/model.onnx`,
    ids: (await json(`${GH}${q}/ettin-native/web/models/e17c-t4-w8L63/fixtures.json`)).map((x) => x.ids) },
];
if (cfg.full) models.push({ name: "eg2-full", file: "eg2-wgsl-engine/web/engine.js", cls: "Eg2Engine",
  manifest: `${GH}${q}/eg2-wgsl-engine/web/manifest_full.json`, data: HF,
  ids: (await json(`${GH}${q}/eg2-wgsl-engine/web/queries_full.json`)).map((x) => x.ids) });

const d = await ctx.describe();
const out = { adapter: { vendor: d.vendor, architecture: d.architecture, description: d.description }, cfg, models: {} };
for (const m of models) {
  const [man, data] = await Promise.all([json(m.manifest), bytes(m.data)]);
  const eng = {};
  for (const v of ["a", "b"]) {
    const mod = await import(`${GH}${cfg[v]}/${m.file}`);
    const t = performance.now();
    eng[v] = await mod[m.cls].create(dev, man, data, { maxTokens: 512, maxSeqs: 16 });
    await eng[v].embed([m.ids[0]]);  // warm-up: first dispatch of each pipeline
    out.models[m.name] = out.models[m.name] ?? {};
    out.models[m.name][`initMs_${v}`] = performance.now() - t;
  }
  let maxDiff = 0;
  for (const ids of m.ids) {
    const [va, vb] = [await eng.a.embed([ids]), await eng.b.embed([ids])];
    for (let i = 0; i < va.length; i++) maxDiff = Math.max(maxDiff, Math.abs(va[i] - vb[i]));
  }
  const per = { a: m.ids.map(() => []), b: m.ids.map(() => []) };
  for (let r = 0; r < cfg.rounds; r++) {
    ctx.progress(r / cfg.rounds, `${m.name}: timing round ${r + 1} of ${cfg.rounds}`);
    for (const v of r % 2 ? ["b", "a"] : ["a", "b"]) {
      for (let i = 0; i < m.ids.length; i++) {
        for (let k = 0; k < cfg.reps; k++) {
          const t = performance.now();
          await eng[v].embed([m.ids[i]]);
          per[v][i].push(performance.now() - t);
        }
      }
    }
  }
  const short = m.ids.map((x, i) => [x.length, i]).filter(([n]) => n <= 20).map(([, i]) => i);
  const longest = m.ids.reduce((b, x, i) => (x.length > m.ids[b].length ? i : b), 0);
  const res = out.models[m.name];
  for (const v of ["a", "b"]) {
    res[`shortMs_${v}`] = +median(short.map((i) => median(per[v][i]))).toFixed(3);
    res[`longMs_${v}`] = +median(per[v][longest]).toFixed(3);
  }
  Object.assign(res, { maxAbsDiff: maxDiff, nQueries: m.ids.length, nShort: short.length, longTokens: m.ids[longest].length });
  ctx.log(m.name, JSON.stringify(res));
  eng.a.destroy(); eng.b.destroy();
}
return out;
