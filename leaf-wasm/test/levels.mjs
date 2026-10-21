// Wire size vs precision for int8 weight files with fewer quantization levels
// (run/export_weights.py --levels): brotli-11 bytes (what a static host serves) and cosine to
// ORT fp32 on the 460 fixtures, split posts / queries / edge cases. Writes results/levels.json.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { createLeaf } from "../web/leaf.js";

const root = new URL("../", import.meta.url);
const read = (p) => readFileSync(new URL(p, root));
const vocab = read("out/hf/vocab.txt").toString();
const wasm = [read("web/kernel-fma.wasm"), read("web/kernel.wasm")];
const fx = JSON.parse(read("out/ref/fixtures.json"));
const groups = { posts: [0, 250], queries: [250, 400], edge: [400, fx.length] };
const files = ["leaf-ir-i8.bin", "leaf-ir-i8-L63.bin", "leaf-ir-i8-L31.bin", "leaf-ir-i8-L15.bin", "leaf-ir-k16000-t4.bin"];
const res = {};
for (const f of files) {
  const path = new URL(`out/${f}`, root).pathname;
  const br = execFileSync("brotli", ["-q", "11", "-c", path], { maxBuffer: 1 << 30 }).length;
  const b = read(`out/${f}`);
  const leaf = await createLeaf({ weights: b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength),
                                  vocab: f.includes("k16000") ? read("out/vocab-k16000.txt").toString() : vocab, wasm });
  const cos = fx.map((x) => { const v = leaf.embed(x.t); let c = 0; for (let i = 0; i < v.length; i++) c += v[i] * x.emb[i]; return c; });
  const r = { raw_mb: +(b.length / 1e6).toFixed(2), brotli_mb: +(br / 1e6).toFixed(2) };
  for (const [g, [a, z]] of Object.entries(groups)) {
    const c = cos.slice(a, z).sort((p, q) => p - q);
    r[g] = { min: +c[0].toFixed(5), p05: +c[Math.floor(c.length * 0.05)].toFixed(5), median: +c[c.length >> 1].toFixed(6) };
  }
  res[f] = r;
  console.log(f, JSON.stringify(r));
}
writeFileSync(new URL("results/levels.json", root), JSON.stringify(res, null, 1));
