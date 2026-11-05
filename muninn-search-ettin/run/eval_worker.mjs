// Post-level hit@1 / hit@5 / MRR@10 through the Worker's own search path (worker/_lib: Ettin query
// embedding, kb-reader fusion over a 40-candidate pool, one hit per page with the match-count boost),
// for several fusion settings. Writes results/worker.json.
//   node run/eval_worker.mjs <index.kbi> <ettin package dir> <site checkout>
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const [kbi, pkg, site] = process.argv.slice(2);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const { KBReader } = await import(`${site}/worker/_lib/kb-reader.js`);
const { EttinCPU } = await import(`${site}/worker/_lib/ettin-cpu.js`);
const { Tokenizer } = await import(`${site}/worker/node_modules/@huggingface/tokenizers/dist/tokenizers.mjs`);
const kb = new KBReader(new Uint8Array(readFileSync(kbi)), "kv:/");
const enc = new EttinCPU(JSON.parse(readFileSync(`${pkg}/manifest.json`)), new Uint8Array(readFileSync(`${pkg}/model.onnx`)));
const tok = new Tokenizer(JSON.parse(readFileSync(`${pkg}/tokenizer.json`)), {});
const prompt = JSON.parse(readFileSync(`${pkg}/meta.json`)).prompts.query;
const post = (id) => id.split("#")[0];
const ARMS = { rrf: null, "a0.5": 0.5, "a0.7": 0.7, "a0.85": 0.85, "dense (a1.0)": 1.0, "bm25 (a0.0)": 0.0 };
const out = {};
for (const qs of ["queries.json", "queries_hard.json"]) {
  const Q = JSON.parse(readFileSync(join(ROOT, "data", qs)));
  const vecs = Q.map((q) => enc.embed(tok.encode(prompt + q.query).ids));
  for (const [name, alpha] of Object.entries(ARMS)) {
    let h1 = 0, h5 = 0, rr = 0;
    Q.forEach((q, i) => {
      const pages = kb.search(q.query, vecs[i], { k: 10, alpha }).map((h) => post(h.chunk_id));
      const r = pages.indexOf(post(q.chunk_id)) + 1;
      h1 += r === 1; h5 += r >= 1 && r <= 5; rr += r >= 1 ? 1 / r : 0;
    });
    const n = Q.length;
    out[`${qs}:${name}`] = { n, "post_hit@1": +(h1 / n).toFixed(4), "post_hit@5": +(h5 / n).toFixed(4), "post_mrr@10": +(rr / n).toFixed(4) };
    console.log(qs, name, JSON.stringify(out[`${qs}:${name}`]));
  }
}
writeFileSync(join(ROOT, "results", "worker.json"), JSON.stringify(out, null, 1));
