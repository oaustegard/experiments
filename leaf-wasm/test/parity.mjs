// Gates for the JS engine against HF/ORT ground truth from run/ref.py.
//   1. tokenizer: ids identical to HF tokenizers on every eval text (posts, queries, edge cases)
//   2. embeddings: cosine to ORT fp32 on the fixtures, for the f32 and int8 weight files
//   3. latency (only with TIMING=1, under embed-coord's cpu.lock): median single-thread ms
// Both kernel builds are gated: kernel-fma.wasm (relaxed SIMD) and kernel.wasm (plain).
// Writes results/parity.json (timing runs: results/timing.json); exits non-zero if a gate fails.
import { readFileSync, writeFileSync } from "node:fs";
import { createLeaf } from "../web/leaf.js";
import { WordPiece } from "../web/tokenizer.js";

const root = new URL("../", import.meta.url);
const read = (p) => readFileSync(new URL(p, root));
const vocab = read("out/hf/vocab.txt").toString();
const kernels = { fma: read("web/kernel-fma.wasm"), plain: read("web/kernel.wasm") };
const TIMING = process.env.TIMING === "1";
const res = {};

const tok = new WordPiece(vocab);
const lines = read("out/ref/ids.jsonl").toString().trim().split("\n");
let bad = 0;
const examples = [];
for (const l of lines) {
  const { t, ids } = JSON.parse(l);
  const got = tok.encode(t);
  if (got.length !== ids.length || got.some((v, i) => v !== ids[i])) {
    if (examples.length < 5) examples.push({ t: t.slice(0, 120), hf: ids.slice(0, 30), js: got.slice(0, 30) });
    bad++;
  }
}
res.tokenizer = { texts: lines.length, mismatches: bad, examples };
console.log("tokenizer", lines.length, "texts,", bad, "mismatches");
for (const e of examples) console.log(JSON.stringify(e));

const fx = JSON.parse(read("out/ref/fixtures.json"));
for (const [kname, wasm] of Object.entries(kernels)) for (const wkind of ["f32", "i8"]) {
  const kind = `${wkind}_${kname}`;
  const buf = read(`out/leaf-ir-${wkind}.bin`);
  const leaf = await createLeaf({ weights: buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), vocab, wasm });
  const cos = fx.map(({ ids, emb }) => { const v = leaf.forward(ids); let c = 0; for (let i = 0; i < v.length; i++) c += v[i] * emb[i]; return c; });
  cos.sort((a, b) => a - b);
  res[kind] = { n: cos.length, min_cos: cos[0], p01_cos: cos[Math.floor(cos.length * 0.01)], mean_cos: cos.reduce((a, b) => a + b) / cos.length };
  console.log(kind, JSON.stringify(res[kind]));
  if (!TIMING) continue;
  const lat = {};
  for (const n of [16, 32, 64, 128]) {
    const ids = [101, ...Array(n - 2).fill(2023), 102];
    for (let i = 0; i < 3; i++) leaf.forward(ids);
    const t = [];
    for (let i = 0; i < 15; i++) { const t0 = performance.now(); leaf.forward(ids); t.push(performance.now() - t0); }
    t.sort((a, b) => a - b);
    lat[n] = +t[7].toFixed(2);
  }
  res[kind].median_ms_by_tokens = lat;
  console.log(kind, "ms", JSON.stringify(lat));
}
writeFileSync(new URL(TIMING ? "results/timing.json" : "results/parity.json", root), JSON.stringify(res, null, 1));
const ok = bad === 0 && res.f32_fma.min_cos > 0.9999 && res.f32_plain.min_cos > 0.9999 &&
  res.i8_fma.min_cos > 0.99 && res.i8_plain.min_cos > 0.99;
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
