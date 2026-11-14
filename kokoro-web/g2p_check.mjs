// Run web/g2p.js on results/g2p_reference.json's sentences and compare with misaki's phonemes: whole sentences,
// and words (aligned by text). Writes results/g2p_check.json; prints the mismatches.
//   node g2p_check.mjs LEXICON.json
import { readFileSync, writeFileSync } from "node:fs";
import { G2P } from "./web/g2p.js";

const lex = JSON.parse(readFileSync(process.argv[2] || "web/lexicon-gb.json"));
const g2p = new G2P(lex.gold, lex.silver);
const ref = JSON.parse(readFileSync("results/g2p_reference.json"));
const stats = {}, misses = [];
for (const r of ref) {
  const s = (stats[r.kind] ??= { sentences: 0, sentences_exact: 0, words: 0, words_exact: 0 });
  const got = g2p.phonemize(r.text);
  s.sentences++;
  if (got.ps.trim() === r.ps.trim()) s.sentences_exact++;
  const want = r.tokens.filter((t) => /\p{L}|\d/u.test(t.text));
  const have = got.tokens.filter((t) => /\p{L}|\d/u.test(t.text));
  const byText = new Map();
  have.forEach((t) => (byText.get(t.text) ?? byText.set(t.text, []).get(t.text)).push(t.phonemes));
  for (const t of want) {
    s.words++;
    const cand = byText.get(t.text);
    const p = cand?.shift();
    if (p === t.phonemes) s.words_exact++;
    else misses.push({ kind: r.kind, word: t.text, misaki: t.phonemes, js: p ?? "(token split differently)", text: r.text });
  }
}
for (const s of Object.values(stats)) { s.word_rate = +(s.words_exact / s.words).toFixed(4); s.sentence_rate = +(s.sentences_exact / s.sentences).toFixed(4); }
writeFileSync("results/g2p_check.json", JSON.stringify({ stats, misses }, null, 1));
console.log(JSON.stringify(stats));
for (const m of misses.slice(0, +(process.env.SHOW || 40))) console.log(m.kind.padEnd(9), JSON.stringify(m.word).padEnd(18), "misaki", JSON.stringify(m.misaki).padEnd(22), "js", JSON.stringify(m.js));
