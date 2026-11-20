// Checks web/beam_dict.js (the dictionary path, plain node) against tessgpu/beam_dict.py and
// against Tesseract 5.3.4 itself.
//
// Reads the harness dumps that run/check_beam_dict.py reads (/tmp/tdump/dict/<set>/*.dict.bin:
// Tesseract's softmax plus its dictionary-on ids), decodes each softmax with
// web/beam.js decode(probs, T, C, model, dict) (which hands over to web/beam_dict.js), and
// compares the ids with the Python ids in data/beam_dict_py_ids.json (written by
// run/check_beam_dict.py) and with Tesseract's ids in the dump. The word split and permuters
// (decodeDictFull) are compared with the Python's too. Exit 0 only when every set agrees on
// every line.
//
//   python3 run/check_beam_dict.py --parity-only && node run/check_beam_dict.mjs

import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { decode } from "../web/beam.js";
import { loadDict, decodeDictFull } from "../web/beam_dict.js";
import { parseTraineddata } from "../web/traineddata.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DUMPS = process.env.DICT_DUMPS || "/tmp/tdump/dict";

// tessgpu/tessdump.py read_dict
function readDump(path) {
  const buf = readFileSync(path);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 0;
  const i32 = () => { const v = dv.getInt32(o, true); o += 4; return v; };
  const T = i32(), C = i32();
  const probs = new Float32Array(T * C);
  for (let i = 0; i < T * C; i++) probs[i] = dv.getFloat32(8 + 4 * i, true);
  o = 8 + 4 * T * C;
  const n0 = i32();
  o += 4 * n0;
  const n = i32();
  const ids = [];
  for (let i = 0; i < n; i++) ids.push(i32());
  return { T, C, probs, ids };
}

const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

const pyPath = join(ROOT, "data", "beam_dict_py_ids.json");
if (!existsSync(pyPath)) {
  console.error(`missing ${pyPath}: run python3 run/check_beam_dict.py --parity-only first`);
  process.exit(2);
}
const py = JSON.parse(readFileSync(pyPath, "utf8"));
const models = new Map();
const summary = {};
let allOk = true;
const t0 = Date.now();
for (const [set, entry] of Object.entries(py)) {
  if (!models.has(entry.model)) {
    const bytes = readFileSync(join(ROOT, "models", `${entry.model}.traineddata`));
    const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const { man } = parseTraineddata(ab, { name: entry.model });
    models.set(entry.model, { model: { null_char: man.null_char, codes: man.codes,
      simple_text: man.simple_text }, dict: loadDict(bytes) });
  }
  const { model, dict } = models.get(entry.model);
  let vsPy = 0, vsTess = 0, words = 0, n = 0, ms = 0, ms0 = 0;
  const mism = [];
  for (const [file, ref] of Object.entries(entry.files)) {
    const path = join(DUMPS, set, `${file}.dict.bin`);
    if (!existsSync(path)) continue;
    const d = readDump(path);
    const s = performance.now();
    const ids = decode(d.probs, d.T, d.C, model, dict);
    ms += performance.now() - s;
    const s0 = performance.now();
    decode(d.probs, d.T, d.C, model);
    ms0 += performance.now() - s0;
    const full = decodeDictFull(d.probs, d.T, d.C, model, dict);
    n++;
    const a = same(ids, ref.ids), b = same(ids, d.ids);
    const w = same(full.words.map((x) => x.join(",")), ref.words.map((x) => x.join(",")));
    vsPy += a; vsTess += b; words += w;
    if (!(a && b && w)) mism.push({ file, js: ids.slice(0, 30), py: ref.ids.slice(0, 30) });
  }
  const ok = n > 0 && vsPy === n && vsTess === n && words === n;
  allOk = allOk && ok;
  summary[set] = { model: entry.model, lines: n, js_vs_python_ids: vsPy, js_vs_tesseract_ids: vsTess,
    js_vs_python_words: words, js_decode_ms_per_line_dict: n ? +(ms / n).toFixed(2) : null,
    js_decode_ms_per_line_nodict: n ? +(ms0 / n).toFixed(2) : null,
    mismatches: mism.slice(0, 5) };
  console.log(`${set.padEnd(14)} lines=${String(n).padStart(4)} vs_python=${vsPy} ` +
    `vs_tesseract=${vsTess} words=${words} ms/line dict=${n ? (ms / n).toFixed(1) : "-"} ` +
    `nodict=${n ? (ms0 / n).toFixed(1) : "-"}`);
}
mkdirSync(join(ROOT, "results"), { recursive: true });
writeFileSync(join(ROOT, "results", "beam_dict_js.json"), JSON.stringify({
  what: "web/beam_dict.js (via web/beam.js decode with a dict) vs tessgpu/beam_dict.py and Tesseract 5.3.4",
  wall_seconds: (Date.now() - t0) / 1000, all_identical: allOk, sets: summary }, null, 1));
console.log(`all_identical=${allOk}`);
process.exit(allOk ? 0 : 1);
