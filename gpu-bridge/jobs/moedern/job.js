// MoEdernColBERT 1.5M and 27M on the bridge's GPU, each with both matmul paths (cfg.kernels:
// "tiled" 64 x 64 batch kernel, "template" the shared 8-row kernels.js matmul). Loads
// moederncolbert-bench/web/engine.js from jsDelivr at cfg.sha, the weights from Hugging Face.
// Timings are back-to-back loops divided by the count, since the worker's clock ticks in ~1 ms.
//   parity: every token row of web/fixtures.json against run/mcb.py (inputs exp15, exp27)
//   query:  one 32-token ColBERT query per embed(), readback included: 16 queries x `reps` calls
//           back to back, total / calls; `rounds` such loops, median
//   q16:    16 queries in one embed()
//   docs:   the 64 fixture documents packed into batches of <= cfg.docTokens tokens, `rounds` passes
// inputs.cfg = { sha, reps: 10, rounds: 5, docTokens: 8192, kernels: ["tiled", "template"] }
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
const out = { adapter: { vendor: d.vendor, architecture: d.architecture, description: d.description }, ua: navigator.userAgent, cfg, models: {} };

const fx = await json(GH + "moederncolbert-bench/web/fixtures.json");
const { MoEdernEngine } = await import(GH + "moederncolbert-bench/web/engine.js");
const seqs = fx.queries.concat(fx.docs);
for (const [name, exp] of [["1.5M", ctx.inputs.exp15], ["27M", ctx.inputs.exp27]]) {
  ctx.progress(0, `${name}: loading`);
  const [config, model, dense] = await Promise.all([json(`${HF}${name}/resolve/main/config.json`),
    bytes(`${HF}${name}/resolve/main/model.safetensors`), bytes(`${HF}${name}/resolve/main/1_Dense/model.safetensors`)]);
  for (const kernel of cfg.kernels) {
    const key = `${name}/${kernel}`;
    let t = performance.now();
    const eng = await MoEdernEngine.create(dev, { config, dense }, model, { maxTokens: cfg.docTokens, maxSeqs: 64, kernel });
    await eng.embed([fx.queries[0]]);
    const res = { initMs: Math.round(performance.now() - t), weightMB: eng.weightBytes / 1e6 };
    // parity, one sequence per call
    let row = 0, minCos = 1, low = 0;
    for (const s of seqs) {
      const v = await eng.embed([s]);
      for (let r = 0; r < s.length; r++) {
        let dd = 0, x = 0, y = 0;
        for (let i = 0; i < 128; i++) { const a = v[r * 128 + i], b = exp[(row + r) * 128 + i]; dd += a * b; x += a * a; y += b * b; }
        const c = dd / Math.sqrt(x * y);
        minCos = Math.min(minCos, c); if (c < 0.9999) low++;
      }
      row += s.length;
    }
    Object.assign(res, { parityMinCos: minCos, rowsBelow09999: low, rows: row });
    ctx.progress(0.3, `${key}: query latency`);
    const loop = async (fn, n) => { const ts = []; for (let r = 0; r < cfg.rounds; r++) { const t0 = performance.now(); for (let k = 0; k < n; k++) await fn(k); ts.push((performance.now() - t0) / n); } return +median(ts).toFixed(3); };
    res.queryMs = await loop((k) => eng.embed([fx.queries[k % fx.queries.length]]), fx.queries.length * cfg.reps);
    res.q16Ms = await loop(() => eng.embed(fx.queries), cfg.reps);
    ctx.progress(0.6, `${key}: documents`);
    const batches = []; let cur = [], n = 0;
    for (const s of fx.docs) { if (n + s.length > cfg.docTokens || cur.length === 64) { batches.push(cur); cur = []; n = 0; } cur.push(s); n += s.length; }
    if (cur.length) batches.push(cur);
    const docTok = fx.docs.reduce((a, s) => a + s.length, 0);
    const pass = async () => { for (const bt of batches) await eng.embed(bt); };
    await pass();
    const ms = await loop(pass, 1);
    Object.assign(res, { docPassMs: ms, docsPerSec: +(fx.docs.length / ms * 1000).toFixed(1),
      tokensPerSec: Math.round(docTok / ms * 1000), nDocs: fx.docs.length, docTokens: docTok, nBatches: batches.length });
    ctx.log(key, JSON.stringify(res));
    out.models[key] = res;
    eng.destroy();
  }
}
return out;
