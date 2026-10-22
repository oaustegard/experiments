// Gates for the k16000 (ettin-native pruned vocab) builds.
//   1. tokenizer: ids identical to HF with ettin's rewritten k16000 tokenizer.json, all eval texts
//   2. mapping: where pruned and full tokenizations agree (mapped through kept_old_ids), the
//      t8 build must reproduce the full-vocab i8 build's vectors exactly (same weights)
//   3. cosine of t8 / t4 vs ORT fp32 full vocab on the 460 fixtures
// Writes results/pruned.json.
import { readFileSync, writeFileSync } from "node:fs";
import { createLeaf } from "../web/leaf.js";
import { WordPiece } from "../web/tokenizer.js";

const root = new URL("../", import.meta.url);
const read = (p) => readFileSync(new URL(p, root));
const ab = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const vocabK = read("out/vocab-k16000.txt").toString(), vocabFull = read("out/hf/vocab.txt").toString();
const keep = JSON.parse(read("../ettin-native/out/models/leafir/k16000-vocab/kept_old_ids.json"));
const wasm = read("web/kernel.wasm");
const res = {};

const tok = new WordPiece(vocabK);
const lines = read("out/ref/ids-k16000.jsonl").toString().trim().split("\n");
let bad = 0;
for (const l of lines) { const { t, ids } = JSON.parse(l); const g = tok.encode(t); if (g.length !== ids.length || g.some((v, i) => v !== ids[i])) bad++; }
res.tokenizer = { texts: lines.length, mismatches: bad };
console.log("tokenizer", JSON.stringify(res.tokenizer));

const full = await createLeaf({ weights: ab(read("out/leaf-ir-i8.bin")), vocab: vocabFull, wasm });
const fx = JSON.parse(read("out/ref/fixtures.json"));
for (const kind of ["t8", "t4"]) {
  const leaf = await createLeaf({ weights: ab(read(`out/leaf-ir-k16000-${kind}.bin`)), vocab: vocabK, wasm });
  const cos = [], same = [];
  for (const f of fx) {
    const ids = leaf.tokenize(f.t), v = leaf.forward(ids);
    let c = 0; for (let i = 0; i < v.length; i++) c += v[i] * f.emb[i];
    cos.push(c);
    const mapped = ids.map((i) => keep[i]);
    if (mapped.length === f.ids.length && mapped.every((v2, i) => v2 === f.ids[i])) {
      const u = full.forward(f.ids); let d = 0; for (let i = 0; i < u.length; i++) d = Math.max(d, Math.abs(u[i] - v[i]));
      same.push(d);
    }
  }
  cos.sort((a, b) => a - b);
  res[kind] = { n: cos.length, min_cos: cos[0], p05_cos: cos[Math.floor(cos.length * 0.05)], mean_cos: cos.reduce((a, b) => a + b) / cos.length,
                same_tokenization: same.length, same_tokenization_max_abs_diff_vs_full_i8: Math.max(...same) };
  console.log(kind, JSON.stringify(res[kind]));
}
writeFileSync(new URL("results/pruned.json", root), JSON.stringify(res, null, 1));
