// Run the JavaScript port of dizione (web/dizione.js) on the corpus and compare it with Python dizione.
//
//   node run/dizione_check.mjs [--corpus run/dizione_corpus.json] [--reference run/dizione_reference.jsonl]
//                              [--bundle web/dizione-*.bin] [--debug python_debug.jsonl] [--out results/dizione_agreement.json]
//
// The reference is what run/dizione_reference.py wrote inside the jail: per sentence "ok" with ids and stress,
// "null" (nothing to pronounce) or "throw". A sentence agrees when the port gives the same kind and, for "ok", the
// same id and stress sequences. Writes counts, the exact-match rate, timings and the first 50 mismatches.
import fs from "fs";
import path from "path";
import { createRequire } from "module";
import { fileURLToPath } from "url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, k, all) => (k % 2 ? pairs : [...pairs, [all[k].replace(/^--/, ""), all[k + 1]]]), []));
const corpusPath = args.corpus || path.join(here, "dizione_corpus.json");
const referencePath = args.reference || path.join(here, "dizione_reference.jsonl");
const outPath = args.out || path.join(root, "results", "dizione_agreement.json");
const bundlePath = args.bundle || path.join(root, "web", fs.readdirSync(path.join(root, "web")).find((f) => /^dizione-[0-9a-f]{12}\.bin$/.test(f)));

createRequire(import.meta.url)(path.join(root, "web", "dizione.js"));
const Dizione = globalThis.Dizione;

const corpus = JSON.parse(fs.readFileSync(corpusPath, "utf8"));
const parts = fs.existsSync(path.join(here, "dizione_corpus_parts.json")) ? JSON.parse(fs.readFileSync(path.join(here, "dizione_corpus_parts.json"), "utf8")) : [];
const reference = fs.readFileSync(referencePath, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
const debug = args.debug ? fs.readFileSync(args.debug, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line)) : [];
if (reference.length !== corpus.length) throw new Error(`${corpus.length} sentences but ${reference.length} reference lines`);

const bytes = fs.readFileSync(bundlePath);
let start = performance.now();
const dizione = await Dizione.load({ bundle: bytes });
const loadMs = performance.now() - start;

const same = (a, b) => a.length === b.length && a.every((x, k) => x === b[k]);
const counts = { sentences: corpus.length, agree: 0, disagree: 0, byKind: {}, byPart: {} };
const mismatches = [];
const times = [];
start = performance.now();
corpus.forEach((text, i) => {
  const ref = reference[i];
  let got;
  const t0 = performance.now();
  try {
    const result = dizione.phonemize(text);
    got = result === null ? { kind: "null" } : { kind: "ok", ids: result.ids, stress: result.stress, text: result.text };
  } catch (error) {
    got = { kind: "throw", error: String(error && error.message || error).slice(0, 300) };
  }
  times.push(performance.now() - t0);
  const agree = got.kind === ref.kind && (got.kind !== "ok" || (same(got.ids, ref.ids) && same(got.stress, ref.stress)));
  const kind = (counts.byKind[ref.kind] ||= { sentences: 0, agree: 0 });
  kind.sentences++;
  const part = (counts.byPart[parts[i] || "all"] ||= { sentences: 0, agree: 0 });
  part.sentences++;
  if (agree) { counts.agree++; kind.agree++; part.agree++; return; }
  counts.disagree++;
  if (mismatches.length < 50) {
    const entry = { i, part: parts[i], text, python: ref, javascript: got };
    if (debug[i]) entry.python_debug = debug[i];
    try {
      entry.javascript_debug = {
        said: dizione.tokens(text).map((t) => (t.punct ? t.mark : (t.attach ? "+" : "") + (t.spell ? `[${t.text}]` : t.text))).join(" "),
        units: dizione.analyse(text).filter(([k]) => k === "clause").flatMap(([, units]) => units.map((u) => [u.word, u.source, u.phones.map(([p, s]) => ({ 1: "ˈ", 2: "ˌ" }[s] || "") + p).join(" ")])),
      };
    } catch (error) { entry.javascript_debug = { error: String(error.message) }; }
    mismatches.push(entry);
  }
});
const totalMs = performance.now() - start;
times.sort((a, b) => a - b);
const quantile = (q) => times[Math.min(times.length - 1, Math.floor(q * times.length))];
const result = {
  bundle: path.basename(bundlePath),
  bundle_bytes: bytes.length,
  sentences: counts.sentences,
  agree: counts.agree,
  disagree: counts.disagree,
  exact_match_rate: counts.agree / counts.sentences,
  by_reference_kind: counts.byKind,
  by_corpus_part: counts.byPart,
  timing_ms: { load: +loadMs.toFixed(1), all_sentences: +totalMs.toFixed(1), mean: +(totalMs / corpus.length).toFixed(3),
               median: +quantile(0.5).toFixed(3), p90: +quantile(0.9).toFixed(3), p99: +quantile(0.99).toFixed(3), max: +times[times.length - 1].toFixed(1) },
  node: process.version,
  mismatches,
};
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(result, null, 1) + "\n");
console.log(`${counts.agree}/${counts.sentences} agree (${(100 * result.exact_match_rate).toFixed(3)}%), load ${loadMs.toFixed(0)} ms, ` +
  `${result.timing_ms.mean} ms per sentence (median ${result.timing_ms.median}, p99 ${result.timing_ms.p99})`);
for (const [name, value] of Object.entries(counts.byPart)) console.log(`  ${name}: ${value.agree}/${value.sentences}`);
process.exitCode = counts.disagree ? 1 : 0;
