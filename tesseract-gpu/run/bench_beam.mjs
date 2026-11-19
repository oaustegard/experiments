// Times web/beam.js decode() over every dump in /tmp/tdump/bench (both variants, all languages)
// and compares it with the same file at HEAD (the baseline, read with git show and loaded from a
// data: URL). Reports ms per line before and after (median of 3 timed runs after one warm-up,
// the two decoders interleaved so both see the same machine load),
// and checks that every id sequence is unchanged. Exits 0 only when the ids are identical in
// every line and the speed-up is at least 5x.
//
//   python3 run/check_beam.py        # writes data/beam_py_ids.json, the model fields used here
//   node run/bench_beam.mjs

import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { decode as decodeNew } from "../web/beam.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const BENCH = "/tmp/tdump/bench";
const RUNS = 3;
const TARGET = 5;

// Baseline: web/beam.js as committed at HEAD.
const baseSrc = execFileSync("git", ["show", "HEAD:./web/beam.js"], { cwd: ROOT, encoding: "utf8" });
const { decode: decodeOld } = await import("data:text/javascript;base64," +
  Buffer.from(baseSrc).toString("base64"));

// Same layout as tessgpu/tessdump.py: int32 T, int32 C, float32 probs[T*C], then ids.
function readDump(path) {
  const buf = readFileSync(path);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const T = view.getInt32(0, true);
  const C = view.getInt32(4, true);
  const probs = new Float32Array(T * C);
  for (let i = 0; i < T * C; i++) probs[i] = view.getFloat32(8 + 4 * i, true);
  // then int32 n, int32 unichar_ids[n] (tessgpu/tessdump.py)
  const o = 8 + 4 * T * C;
  const n = view.getInt32(o, true);
  const ids = [];
  for (let k = 0; k < n; k++) ids.push(view.getInt32(o + 4 + 4 * k, true));
  return { T, C, probs, ids };
}

const models = JSON.parse(readFileSync(join(ROOT, "data", "beam_py_ids.json"), "utf8")).models;
const jobs = [];
for (const key of Object.keys(models).sort()) {
  const [variant, lang] = key.split("/");
  const dir = join(BENCH, `${variant}_${lang}`);
  for (const f of readdirSync(dir).filter((x) => x.endsWith(".png.bin")).sort()) {
    const d = readDump(join(dir, f));
    jobs.push({ model: models[key], ...d });
  }
}

function runAll(decodeFn) {
  const out = new Array(jobs.length);
  const t0 = performance.now();
  for (let i = 0; i < jobs.length; i++) {
    const j = jobs[i];
    out[i] = decodeFn(j.probs, j.T, j.C, j.model);
  }
  return { ms: performance.now() - t0, out };
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// Runs the two decoders alternately (after one warm-up each), so both see the same machine load.
function timeBoth(fnA, fnB) {
  runAll(fnA);
  runAll(fnB);
  const tA = [], tB = [];
  let outA = null, outB = null;
  for (let r = 0; r < RUNS; r++) {
    const a = runAll(fnA);
    const b = runAll(fnB);
    tA.push(a.ms);
    tB.push(b.ms);
    outA = a.out;
    outB = b.out;
  }
  return { before: { ms: median(tA), out: outA }, after: { ms: median(tB), out: outB } };
}

const { before, after } = timeBoth(decodeOld, decodeNew);

let same = 0;
let vsTesseract = 0;
for (let i = 0; i < jobs.length; i++) {
  const a = before.out[i], b = after.out[i];
  if (a.length === b.length && a.every((v, k) => v === b[k])) same++;
  const t = jobs[i].ids;
  if (t.length === b.length && t.every((v, k) => v === b[k])) vsTesseract++;
}

const lines = jobs.length;
const msBefore = before.ms / lines;
const msAfter = after.ms / lines;
const speedup = before.ms / after.ms;
const report = {
  lines,
  ms_per_line_before: Number(msBefore.toFixed(4)),
  ms_per_line_after: Number(msAfter.toFixed(4)),
  total_ms_before: Number(before.ms.toFixed(1)),
  total_ms_after: Number(after.ms.toFixed(1)),
  speedup: Number(speedup.toFixed(2)),
  ids_identical_to_before: `${same}/${lines}`,
  ids_identical_to_tesseract: `${vsTesseract}/${lines}`,
};
console.log(JSON.stringify(report, null, 1));
const ok = same === lines && speedup >= TARGET;
console.log(ok ? `PASS: identical ids and >= ${TARGET}x` : `FAIL: identical=${same}/${lines} speedup=${speedup.toFixed(2)}`);
process.exit(ok ? 0 : 1);
