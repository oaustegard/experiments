// Checks web/beam.js (plain node, no browser) against tessgpu/beam.py.
//
// Reads the same dumps as run/check_beam.py (Tesseract's softmax probs), decodes them with
// web/beam.js, and compares with the Python ids written to data/beam_py_ids.json by
// run/check_beam.py. Also checks that each web pack manifest's codes and null_char match the
// traineddata recoder. Exits 0 when agreement is at least 99.9% in every model.
//
//   python3 run/check_beam.py && node run/check_beam.mjs

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { decode } from "../web/beam.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const BENCH = "/tmp/tdump/bench";
const BAR = 0.999;

// Same layout as tessgpu/tessdump.py: int32 T, int32 C, float32 probs[T*C], then ids.
function readProbs(path) {
  const buf = readFileSync(path);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const T = view.getInt32(0, true);
  const C = view.getInt32(4, true);
  const probs = new Float32Array(T * C);
  for (let i = 0; i < T * C; i++) probs[i] = view.getFloat32(8 + 4 * i, true);
  return { T, C, probs };
}

const py = JSON.parse(readFileSync(join(ROOT, "data", "beam_py_ids.json"), "utf8"));
const summary = {};
let allOk = true;

for (const key of Object.keys(py.models)) {
  const model = py.models[key];
  const [variant, lang] = key.split("/");
  const files = Object.keys(py.ids[key]).sort();
  let same = 0;
  const mismatches = [];
  for (const name of files) {
    const { T, C, probs } = readProbs(join(BENCH, `${variant}_${lang}`, `${name}.png.bin`));
    const ids = decode(probs, T, C, model);
    const ref = py.ids[key][name];
    const eq = ids.length === ref.length && ids.every((v, i) => v === ref[i]);
    if (eq) same++;
    else mismatches.push({ file: name, js: ids.slice(0, 20), py: ref.slice(0, 20) });
  }
  const share = same / files.length;
  const ok = share >= BAR;
  allOk = allOk && ok;
  summary[key] = { lines: files.length, identical: same, share, meets_bar_99_9: ok,
    first_mismatches: mismatches.slice(0, 5) };
  console.log(`${key.padEnd(10)} lines=${String(files.length).padStart(4)} agree=${same} share=${share.toFixed(4)}`);
}

// Web pack manifests: the recoder in web/models/<variant>-<lang>/manifest.json must match.
const webDir = join(ROOT, "web", "models");
const manifestCheck = {};
for (const dir of existsSync(webDir) ? readdirSync(webDir) : []) {
  const mpath = join(webDir, dir, "manifest.json");
  if (!existsSync(mpath)) continue;
  const [variant, lang] = dir.split("-");
  const model = py.models[`${variant}/${lang}`];
  if (!model) continue;
  const man = JSON.parse(readFileSync(mpath, "utf8"));
  const codesEq = JSON.stringify(man.codes) === JSON.stringify(model.codes);
  const nullEq = man.null_char === model.null_char;
  manifestCheck[dir] = { codes_equal_traineddata: codesEq, null_char_equal: nullEq };
  allOk = allOk && codesEq && nullEq;
}

console.log("manifest recoder check:", JSON.stringify(manifestCheck));
console.log(`all_models_meet_bar=${allOk}`);
process.exit(allOk ? 0 : 1);
