// CPU time of one query embedding in worker/_lib/ettin-cpu.js at several lengths, plus model unpack.
//   node run/bench.mjs <ettin package> <site checkout>      -> results/timing.json
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const [pkg, site] = process.argv.slice(2);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const { EttinCPU } = await import(`${site}/worker/_lib/ettin-cpu.js`);
const man = JSON.parse(readFileSync(`${pkg}/manifest.json`)), bytes = new Uint8Array(readFileSync(`${pkg}/model.onnx`));
let t = performance.now(); const enc = new EttinCPU(man, bytes); const unpack = performance.now() - t;
const out = { node: process.version, unpack_ms: +unpack.toFixed(1), embed_ms_median: {} };
for (const L of [12, 16, 20, 32, 64, 128]) {
  const ids = Array.from({ length: L }, (_, i) => (i === 0 ? 50281 : i === L - 1 ? 50282 : 1000 + i * 37));
  for (let w = 0; w < 3; w++) enc.embed(ids);
  const ms = [];
  for (let r = 0; r < 15; r++) { t = performance.now(); enc.embed(ids); ms.push(performance.now() - t); }
  ms.sort((a, b) => a - b); out.embed_ms_median[L] = +ms[7].toFixed(1);
}
writeFileSync(join(ROOT, "results", "timing.json"), JSON.stringify(out, null, 1)); console.log(out);
