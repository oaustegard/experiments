// Ranking cost per query in kb-reader.js (dense Hamming scan, BM25, fusion + page collapse, chunk fetch
// with sha256 check), the part of a search that runs after the embedding. Random query vector.
//   node run/rank_bench.mjs <index dir> <site checkout>      -> results/rank_timing.json
import { readFileSync } from "node:fs";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const [dir, site] = process.argv.slice(2);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const W = `${site}/worker`;
const { KBReader } = await import(`${W}/_lib/kb-reader.js`);
const kb = new KBReader(new Uint8Array(readFileSync(`${dir}/muninn.kbi`)), "local:/");
const shard = readFileSync(`${dir}/muninn.kbc/shard-0000.bin`);
const fetchImpl = async (url, o) => { const m = /bytes=(\d+)-(\d+)/.exec(o.headers.range); return new Response(shard.subarray(+m[1], +m[2] + 1), { status: 206 }); };
const v = new Float32Array(768).map(() => Math.random() - 0.5);
const Q = JSON.parse(readFileSync(join(ROOT, "data", "queries.json"))).slice(0, 40).map((q) => q.query);
const T = { dense: [], bm25: [], search: [], fetch: [] };
for (let rep = 0; rep < 3; rep++) for (const q of Q) {
  let t = performance.now(); kb._denseSearch(v); T.dense.push(performance.now() - t);
  t = performance.now(); kb._bm25Search(q); T.bm25.push(performance.now() - t);
  t = performance.now(); const h = kb.search(q, v, { k: 8, alpha: 0.7 }); T.search.push(performance.now() - t);
  t = performance.now(); await kb.fetchChunks(h, fetchImpl); T.fetch.push(performance.now() - t);
}
const out = { node: process.version, queries: Q.length, reps: 3, median_ms: {} };
for (const [k, a] of Object.entries(T)) { a.sort((x, y) => x - y); out.median_ms[k] = +a[a.length >> 1].toFixed(2); }
writeFileSync(join(ROOT, "results", "rank_timing.json"), JSON.stringify(out, null, 1)); console.log(out);
