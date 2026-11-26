// Two LiteParse reference sets compared with run.mjs's metrics: how far LiteParse's WebAssembly
// build (tract) and its native build (ONNX Runtime) already disagree on the same pixels.
//   node test/refdiff.mjs --a /home/user/opb/ref40/ref40-native/pages --b /home/user/opb/ref40/ref40-nolayout/pages
import fs from "node:fs";
import path from "node:path";
import { compare } from "./compare.mjs";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1]]);
  return acc;
}, []));
const A = args.a || "/home/user/opb/ref40/ref40-native/pages";
const B = args.b || "/home/user/opb/ref40/ref40-nolayout/pages";
const ids = fs.readdirSync(B).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5)).sort();
const tot = { ref_items: 0, ref_found: 0, our_items: 0, ours_found: 0, exact: 0, box_ok: 0, edits: 0, ref_chars: 0 };
const rows = [];
for (const id of ids) {
  const load = (d) => (JSON.parse(fs.readFileSync(path.join(d, `${id}.json`), "utf8")).text_items || []);
  // b plays the reference, a plays "ours"
  const c = compare(load(B), load(A));
  for (const k of Object.keys(tot)) tot[k] += c[k];
  rows.push({ id, ...c });
  const pct = (x, y) => (y ? ((100 * x) / y).toFixed(1) : "100.0");
  console.log(`${pct(c.ref_found, c.ref_items).padStart(5)}% b-in-a  ${pct(c.ours_found, c.our_items).padStart(5)}% a-in-b  cer ${(100 * c.cer).toFixed(2).padStart(5)}%  ${id}`);
}
console.log(`\nREFDIFF ${ids.length} pages: b items found verbatim in a ${(100 * tot.ref_found / tot.ref_items).toFixed(2)}% (${tot.ref_found}/${tot.ref_items}), ` +
  `a in b ${(100 * tot.ours_found / tot.our_items).toFixed(2)}%, exact ${(100 * tot.exact / tot.ref_items).toFixed(2)}%, box ${(100 * tot.box_ok / tot.ref_items).toFixed(2)}%, CER ${(100 * tot.edits / tot.ref_chars).toFixed(3)}%`);
if (args.json) fs.writeFileSync(args.json, JSON.stringify({ tot, rows }, null, 1));
