// GPU Bridge job (gpu-bridge/web): the SparseUp WGSL engine on Oskar's GPU. Body of async function (ctx).
// Inputs: w0..w3 (weights.bin split, u8), manifest, fixture (web/fixtures/parity.json rows), reps.
const BASE = "https://cdn.jsdelivr.net/gh/oaustegard/experiments@0ae4e2df885ec58545db450653c0be5dd93296b3/";
const { manifest, fixture, reps } = ctx.inputs;
const key = "sparseup-v2-" + manifest.quant;
let eng = ctx.cache.get(key);
const t0 = performance.now();
if (!eng) {
  const parts = ["w0", "w1", "w2", "w3"].map((k) => ctx.inputs[k]).filter(Boolean);
  const bytes = new Uint8Array(parts.reduce((s, p) => s + p.byteLength, 0));
  let o = 0;
  for (const p of parts) { bytes.set(new Uint8Array(p.buffer, p.byteOffset, p.byteLength), o); o += p.byteLength; }
  const { SparseEngine } = await ctx.importModule(BASE + "sparseup-v2/pack/web/engine.js");
  const device = await ctx.gpu();
  eng = await SparseEngine.create(device, manifest, bytes, { maxTokens: 512, maxSeq: 512, maxSeqs: 16 });
  ctx.cache.set(key, eng);
}
const initMs = performance.now() - t0;
ctx.log("engine ready", manifest.quant, initMs.toFixed(0), "ms");
const V = manifest.vpad, rows = [];
for (const [k, r] of fixture.entries()) {
  const s = performance.now();
  const out = await eng.embed([{ ids: r.ids, pool: r.pool }]);
  const ms = performance.now() - s;
  const i = [], w = [];
  for (let v = 0; v < V; v++) if (out[v] > 0) { i.push(v); w.push(out[v]); }
  rows.push({ name: r.name, n: r.ids.length, tokOk: true, ms, i, w });
  ctx.progress((k + 1) / (fixture.length + 2), "fixture");
}
const med = (a) => { a = [...a].sort((x, y) => x - y); return a[Math.floor(a.length / 2)]; };
const timing = {};
const long = fixture.find((r) => r.n === 512);
for (const n of [128, 512]) {
  const ids = long.ids.slice(0, n - 1).concat(long.ids[long.ids.length - 1]);
  const item = { ids, pool: ids.map((_, j) => (j === 1 ? 0 : 1)) };
  for (let k = 0; k < 3; k++) await eng.embed([item]);
  const ts = [];
  for (let k = 0; k < reps; k++) { const s = performance.now(); await eng.embed([item]); ts.push(performance.now() - s); }
  timing[n] = { tokens: ids.length, median_ms: med(ts), all_ms: ts };
}
const qs = fixture.filter((r) => r.kind === "query").slice(0, 16).map((r) => ({ ids: r.ids, pool: r.pool }));
await eng.embed(qs);
const ts = [];
for (let k = 0; k < reps; k++) { const s = performance.now(); await eng.embed(qs); ts.push(performance.now() - s); }
timing.batch16q = { texts: qs.length, tokens: qs.reduce((s, x) => s + x.ids.length, 0), median_ms: med(ts) };
ctx.progress(1, "done");
return { pack: manifest.quant, rows: JSON.stringify(rows), timing, initMs, adapter: ctx.describe ? (await ctx.describe()).adapter ?? null : null };
