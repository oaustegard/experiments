// Run the site's worker/index.js in Node against a file-backed mock of its KB_INDEX KV namespace.
//   node run/worker_local.mjs <index dir with muninn.kbi + muninn.kbc/> <ettin package> <site checkout> <queries.json> <out.json>
import { readFileSync, writeFileSync } from "node:fs";
const [kbDir, pkg, site, qfile, out] = process.argv.slice(2);
const files = {
  "muninn.kbi": `${kbDir}/muninn.kbi`, "shard-0000.bin": `${kbDir}/muninn.kbc/shard-0000.bin`,
  "ettin/model.onnx": `${pkg}/model.onnx`, "ettin/manifest.json": `${pkg}/manifest.json`,
  "ettin/tokenizer.json": `${pkg}/tokenizer.json`, "ettin/meta.json": `${pkg}/meta.json`,
};
const KB_INDEX = { async get(key, opts) {
  if (key === "index-version") return "eval";
  if (key === "ettin/version") return "local";
  const f = files[key]; if (!f) return null;
  const b = readFileSync(f);
  if (opts?.type === "arrayBuffer") return b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
  if (opts?.type === "json") return JSON.parse(b.toString("utf8"));
  return b.toString("utf8");
} };
let n = 0;
const SEARCH_RL = { async limit() { return { success: ++n <= 1000 }; } };
const worker = (await import(`${site}/worker/index.js`)).default;
const Q = JSON.parse(readFileSync(qfile, "utf8"));
const res = [];
for (const q of Q) {
  const t = performance.now();
  const r = await worker.fetch(new Request("https://x/api/search", { method: "POST", body: JSON.stringify({ query: q.query, k: 5 }) }), { KB_INDEX, SEARCH_RL });
  const j = await r.json();
  if (!r.ok) { console.error(r.status, j); process.exit(1); }
  res.push({ query: q.query, ms: performance.now() - t, ids: j.hits.map((h) => h.chunk_id) });
}
writeFileSync(out, JSON.stringify(res));
const ms = res.map((r) => r.ms).sort((a, b) => a - b);
console.log("queries", res.length, "first ms", res[0].ms.toFixed(0), "median ms", ms[ms.length >> 1].toFixed(1), "keys", Object.keys(JSON.parse(JSON.stringify(res[0]))));
