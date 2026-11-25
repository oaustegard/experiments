// The WGSL engine (web/engine.js) under Node on Dawn (npm "webgpu"), for hardware GPUs without a browser:
//   node run_node.mjs PACK_DIR FIXTURE_JSON OUT_JSON [--start S] [--only N] [--reps R] [--backend vulkan]
// Writes { info, rows: [{ name, n, ms, i, w }], timing } in check.html's format, so check.py --source can score it.
// Token ids come from the fixture (the browser path's tokenizer is checked separately by check.html).
import { readFileSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";

Object.assign(globalThis, globals);
const [packDir, fixPath, outPath, ...rest] = process.argv.slice(2);
const opt = (k, d) => { const i = rest.indexOf(`--${k}`); return i >= 0 ? rest[i + 1] : d; };
const start = +opt("start", 0), only = +opt("only", 0), reps = +opt("reps", 0);
const gpu = create([`backend=${opt("backend", "vulkan")}`]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no adapter");
const info = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, device: adapter.info?.device, description: adapter.info?.description };
console.log("adapter", JSON.stringify(info));
const L = adapter.limits;
const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU error", e.error?.message));
const { SparseEngine } = await import("./web/engine.js");
const manifest = JSON.parse(readFileSync(`${packDir}/manifest.json`, "utf8"));
const bytes = readFileSync(`${packDir}/weights.bin`);
let t0 = performance.now();
const eng = await SparseEngine.create(device, manifest, new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), { maxTokens: 512, maxSeq: 512, maxSeqs: 16 });
const initMs = performance.now() - t0;
const V = manifest.vpad;
const all = JSON.parse(readFileSync(fixPath, "utf8"));
const fx = all.slice(start, only ? start + only : undefined);
const rows = [];
for (const r of fx) {
  t0 = performance.now();
  const out = await eng.embed([{ ids: r.ids, pool: r.pool }]);
  const ms = performance.now() - t0;
  const i = [], w = [];
  for (let v = 0; v < V; v++) if (out[v] > 0) { i.push(v); w.push(out[v]); }
  rows.push({ name: r.name, n: r.ids.length, tokOk: true, ms, i, w });
  console.log(`${r.name} n=${r.ids.length} nnz ${i.length} ${ms.toFixed(1)} ms`);
}
const timing = {};
if (reps) {
  const long = all.find((r) => r.n === 512);
  for (const n of [128, 512]) {
    const ids = long.ids.slice(0, n - 1).concat(long.ids[long.ids.length - 1]);
    const item = { ids, pool: ids.map((_, j) => (j === 1 ? 0 : 1)) };
    for (let k = 0; k < 3; k++) await eng.embed([item]);   // warm
    const ts = [];
    for (let k = 0; k < reps; k++) { t0 = performance.now(); await eng.embed([item]); ts.push(performance.now() - t0); }
    ts.sort((a, b) => a - b);
    timing[n] = { tokens: ids.length, median_ms: ts[Math.floor(ts.length / 2)], all_ms: ts };
    console.log(`timing ${n}: median ${timing[n].median_ms.toFixed(1)} ms of ${reps}`);
  }
  // Throughput: 16 short queries packed in one batch.
  const qs = all.filter((r) => r.kind === "query").slice(0, 16).map((r) => ({ ids: r.ids, pool: r.pool }));
  await eng.embed(qs);
  const ts = [];
  for (let k = 0; k < reps; k++) { t0 = performance.now(); await eng.embed(qs); ts.push(performance.now() - t0); }
  ts.sort((a, b) => a - b);
  timing.batch16q = { texts: qs.length, tokens: qs.reduce((s, x) => s + x.ids.length, 0), median_ms: ts[Math.floor(ts.length / 2)] };
  console.log(`batch of ${qs.length} queries: median ${timing.batch16q.median_ms.toFixed(1)} ms`);
}
writeFileSync(outPath, JSON.stringify({ pack: manifest.quant, rows, timing, info: { adapter: info, initMs }, console_errors: [] }));
console.log("wrote", outPath);
process.exit(0);
