// Chunk-level fused candidates (top 40, 40-candidate pool) from the Worker's own modules, for run/parity.py.
//   node run/fused_js.mjs <index.kbi> <ettin package> <site checkout> <queries.json> <alpha|rrf> <out.json>
import { readFileSync, writeFileSync } from "node:fs";
const [kbi, pkg, site, qfile, alphaS, out] = process.argv.slice(2);
const W = `${site}/worker`;
const { KBReader, fuseRanks } = await import(`${W}/_lib/kb-reader.js`);
const { EttinCPU } = await import(`${W}/_lib/ettin-cpu.js`);
const { Tokenizer } = await import(`${W}/node_modules/@huggingface/tokenizers/dist/tokenizers.mjs`);
const alpha = alphaS === "rrf" ? null : Number(alphaS);
const kb = new KBReader(new Uint8Array(readFileSync(kbi)), "kv:/");
const enc = new EttinCPU(JSON.parse(readFileSync(`${pkg}/manifest.json`)), new Uint8Array(readFileSync(`${pkg}/model.onnx`)));
const tok = new Tokenizer(JSON.parse(readFileSync(`${pkg}/tokenizer.json`)), {});
const prompt = JSON.parse(readFileSync(`${pkg}/meta.json`)).prompts.query;
const res = [];
for (const q of JSON.parse(readFileSync(qfile))) {
  const ids = tok.encode(prompt + q.query).ids;
  const v = enc.embed(ids);
  const fused = fuseRanks(kb._denseSearch(v), kb._bm25Search(q.query), 40, alpha).slice(0, 40).map((h) => kb._withChunkId(h).chunk_id);
  res.push({ query: q.query, ids, fused });
}
writeFileSync(out, JSON.stringify(res));
console.log("done", res.length);
