// ocr.js against LiteParse's own OCR output on the same pixels.
//
// `infer` is backed by onnxruntime-node on the upstream OCRia-nano-1 models. Each reference page is
// a gray PNG that run/png2pdf.py wrapped as a PDF of px*72/250 points; LiteParse 2.21.2 rendered it
// at 250 DPI (pixel for pixel the PNG: test/lp_raster.mjs), OCR'd it with ocria-nano, layout off,
// and wrote its text items in PDF points. Two reference sets:
//
//   wasm    the WebAssembly build (nets on tract): run/lp_pages.mjs --dump-items     (--ref)
//   native  the Python wheel's native build (nets on ONNX Runtime 1.31.1, the version
//           installed here): test/lp_native.py                                       (--native)
//
// The same PNG goes through ocrPage once; its results go through the part of LiteParse's OCR merge
// that changes item text (clean_ocr_table_artifacts, the confidence floor) and are compared with
// each set:
//
//   ref_found    share of reference item strings found verbatim in our page text
//   +merged      ... or equal to 2-6 of our same-row items joined by " " or "" (LiteParse's
//                projection merges side-by-side OCR items into one text item)
//   ours_found   share of our item strings found verbatim in the reference page text
//   exact        share of reference items with an identical string among ours (multiset)
//   cer          CER of our page text against the reference's, both in the detector's reading
//                order (8 px row bands, then left edge); order-sensitive
//   item_cer     order-free CER: our items assigned to the reference item covering them
//   box          share of reference items with one of ours at IoU >= 0.5
//
//   node test/run.mjs [--ref DIR] [--native DIR] [--models DIR] [--png DIR] [--out DIR] [--only SUBSTR]
//                     [--opt all|disabled] [--threads N]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ort from "onnxruntime-node";
import pngjs from "pngjs";
import { DPI, compare, toItems } from "./compare.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith("--")) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith("--") ? all[i + 1] : true]);
  return acc;
}, []));
const MODELS = path.resolve(args.models || path.join(here, "../../../models"));
const REF = path.resolve(args.ref || "/home/user/opb/ref40/ref40-nolayout/pages");
const NATIVE = path.resolve(args.native || "/home/user/opb/ref40/ref40-native/pages");
const HAVE_NATIVE = fs.existsSync(NATIVE);
const PNG_DIR = path.resolve(args.png || "/home/user/opb/gray600");
const OUT = path.resolve(args.out || path.join(here, "out"));
const { ocrPage } = await import(pathToFileURL(path.join(here, "../ocr.js")).href);

// ---------------------------------------------------------------------------------------------
// infer on onnxruntime-node

const sessions = new Map();
let inferMs = 0, inferCalls = 0;
async function session(model) {
  if (!sessions.has(model)) {
    sessions.set(model, await ort.InferenceSession.create(path.join(MODELS, `${model}.onnx`), {
      intraOpNumThreads: Number(args.threads || 2),
      interOpNumThreads: 1,
      executionMode: "sequential",
      graphOptimizationLevel: args.opt || "all",
    }));
  }
  return sessions.get(model);
}
async function infer(model, shape, data, outputNames) {
  const s = await session(model);
  const t = performance.now();
  const res = await s.run({ x: new ort.Tensor("float32", data, shape) }, outputNames);
  inferMs += performance.now() - t;
  inferCalls++;
  const out = {};
  for (const k of outputNames) out[k] = { shape: res[k].dims.map(Number), data: res[k].data };
  return out;
}

const charsets = {};
for (const g of ["latin", "cjk", "rtl", "world"]) {
  charsets[g] = JSON.parse(fs.readFileSync(path.join(MODELS, `rec_${g}.charset.json`), "utf8"));
}
const metaPath = (f) => path.join(MODELS, f);
const detMeta = fs.existsSync(metaPath("det.meta.json")) ? JSON.parse(fs.readFileSync(metaPath("det.meta.json"), "utf8")) : null;
const rtlMeta = fs.existsSync(metaPath("rec_rtl.meta.json")) ? JSON.parse(fs.readFileSync(metaPath("rec_rtl.meta.json"), "utf8")) : {};
charsets.rtl = { chars: charsets.rtl, visual_rtl: !!rtlMeta.visual_rtl };

// ---------------------------------------------------------------------------------------------

function loadGray(file) {
  const png = pngjs.PNG.sync.read(fs.readFileSync(file));
  const { width: w, height: h, data } = png;
  const gray = new Uint8Array(w * h);
  let isGray = true;
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    gray[i] = data[j];
    if (data[j] !== data[j + 1] || data[j] !== data[j + 2]) isGray = false;
  }
  if (isGray) return { data: gray, w, h, ch: 1 };
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    rgb[i * 3] = data[j];
    rgb[i * 3 + 1] = data[j + 1];
    rgb[i * 3 + 2] = data[j + 2];
  }
  return { data: rgb, w, h, ch: 3 };
}

fs.mkdirSync(path.join(OUT, "pages"), { recursive: true });
let ids = fs.readdirSync(REF).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5)).sort();
if (args.only) ids = ids.filter((i) => i.includes(args.only));
const loadRef = (dir, id) => {
  const f = path.join(dir, `${id}.json`);
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")).text_items || [] : null;
};
const pct = (a, b) => (b ? ((100 * a) / b).toFixed(1) : "100.0");
const rows = [];
const t0 = performance.now();
console.log("  wasm: ref  +merged   ours  item_cer   box | native: ref  +merged  ours | items ref/ours      ms  page");
for (const id of ids) {
  const img = loadGray(path.join(PNG_DIR, `${id}.png`));
  const i0 = inferMs, c0 = inferCalls;
  const t = performance.now();
  const results = await ocrPage(img, infer, charsets, { detMeta });
  const ms = performance.now() - t;
  const ours = toItems(results, (img.w * 72) / DPI, (img.h * 72) / DPI);
  const wasm = compare(loadRef(REF, id), ours);
  const nativeItems = HAVE_NATIVE ? loadRef(NATIVE, id) : null;
  const native = nativeItems ? compare(nativeItems, ours) : null;
  // a wasm item we miss that native LiteParse also misses: the nets' backend (tract vs ONNX Runtime)
  const nativeText = nativeItems ? nativeItems.map((x) => x.text).join("\n") : "";
  const alsoNative = nativeItems ? wasm.missing.filter((t) => !nativeText.includes(t)).length : null;
  const row = {
    id, w: img.w, h: img.h, ms: Math.round(ms), infer_ms: Math.round(inferMs - i0), infer_calls: inferCalls - c0,
    wasm, native, missed_by_native_too: alsoNative,
  };
  rows.push(row);
  fs.writeFileSync(path.join(OUT, "pages", `${id}.json`), JSON.stringify({ ...row, results }, null, 1));
  const n = native;
  console.log(`  ${pct(wasm.ref_found, wasm.ref_items).padStart(5)}% ${pct(wasm.ref_found_merged, wasm.ref_items).padStart(6)}% ` +
    `${pct(wasm.ours_found, wasm.our_items).padStart(6)}% ${(100 * wasm.item_edits / Math.max(wasm.item_chars, 1)).toFixed(2).padStart(6)}% ` +
    `${pct(wasm.box_ok, wasm.ref_items).padStart(6)}% | ` +
    (n ? `${pct(n.ref_found, n.ref_items).padStart(9)}% ${pct(n.ref_found_merged, n.ref_items).padStart(6)}% ${pct(n.ours_found, n.our_items).padStart(6)}%` : "      n/a") +
    ` | ${String(wasm.ref_items).padStart(4)}/${String(wasm.our_items).padStart(4)}  ${String(row.ms).padStart(6)}  ${id}`);
}

function totals(key) {
  const rs = rows.map((r) => r[key]).filter(Boolean);
  if (!rs.length) return null;
  const s = (k) => rs.reduce((a, r) => a + r[k], 0);
  return {
    ref_items: s("ref_items"), our_items: s("our_items"),
    ref_found: s("ref_found"), ref_found_merged: s("ref_found_merged"), ours_found: s("ours_found"),
    ref_found_pct: (100 * s("ref_found")) / s("ref_items"),
    ref_found_merged_pct: (100 * s("ref_found_merged")) / s("ref_items"),
    ours_found_pct: (100 * s("ours_found")) / s("our_items"),
    exact_pct: (100 * s("exact")) / s("ref_items"),
    box_pct: (100 * s("box_ok")) / s("ref_items"),
    cer_pct: (100 * s("edits")) / s("ref_chars"),
    item_cer_pct: (100 * s("item_edits")) / s("item_chars"),
    pages_identical: rs.filter((r) => r.ref_found === r.ref_items && r.ours_found === r.our_items).length,
    pages_under_90: rows.filter((r) => r[key] && r[key].ref_found < 0.9 * r[key].ref_items).map((r) => r.id),
  };
}
const W = totals("wasm"), N = totals("native");
const missed = rows.reduce((a, r) => a + r.wasm.missing.length, 0);
const missedNative = HAVE_NATIVE ? rows.reduce((a, r) => a + (r.missed_by_native_too ?? 0), 0) : null;
const summary = {
  pages: rows.length, wasm: W, native: N,
  wasm_missed: missed, wasm_missed_by_native_too: missedNative,
  ms_total: Math.round(performance.now() - t0),
  ms_js: rows.reduce((a, r) => a + r.ms - r.infer_ms, 0),
  ms_infer: rows.reduce((a, r) => a + r.infer_ms, 0),
  ort_opt: args.opt || "all",
};
fs.writeFileSync(path.join(OUT, "summary.json"), JSON.stringify({ summary, rows }, null, 1));
const line = (name, T) => T &&
  `SUMMARY vs ${name}: ref items found verbatim ${T.ref_found_pct.toFixed(2)}% (${T.ref_found}/${T.ref_items}), ` +
  `+merged ${T.ref_found_merged_pct.toFixed(2)}%, ours found ${T.ours_found_pct.toFixed(2)}% (${T.ours_found}/${T.our_items}), ` +
  `exact ${T.exact_pct.toFixed(2)}%, box IoU>=0.5 ${T.box_pct.toFixed(2)}%, CER ${T.cer_pct.toFixed(3)}% (item CER ${T.item_cer_pct.toFixed(3)}%), ` +
  `identical pages ${T.pages_identical}/${rows.length}, under 90%: ${T.pages_under_90.length ? T.pages_under_90.join(", ") : "none"}`;
console.log("");
console.log(line("LiteParse wasm (tract)", W));
if (N) console.log(line("LiteParse native (ONNX Runtime 1.31.1)", N));
if (HAVE_NATIVE) console.log(`ATTRIBUTION: of ${missed} wasm items not found in our text, native LiteParse misses ${missedNative} too (nets backend), ${missed - missedNative} it finds`);
console.log(`TIME ${rows.length} pages ${(summary.ms_total / 1000).toFixed(1)} s (js ${(summary.ms_js / 1000).toFixed(1)} s, infer ${(summary.ms_infer / 1000).toFixed(1)} s, ORT opt ${summary.ort_opt})`);
