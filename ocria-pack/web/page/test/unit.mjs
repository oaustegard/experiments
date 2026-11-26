// The Rust unit tests of src/ocr/ocria/{img,page,det}.rs, run against ocr.js.
//   node test/unit.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import { cleanLatin, isShortToken, visualToLogical, _internal as I } from "../ocr.js";

let n = 0;
const test = (name, fn) => {
  fn();
  n++;
  console.log("ok", name);
};

test("rot90_matches_numpy", () => {
  const img = { w: 3, h: 2, ch: 1, data: Uint8Array.of(1, 2, 3, 4, 5, 6) };
  assert.deepEqual([...I.rot90(img, 1).data], [3, 6, 2, 5, 1, 4]);
  assert.deepEqual([...I.rot90(img, 2).data], [6, 5, 4, 3, 2, 1]);
});

test("bicubic_weights_sum_to_one", () => {
  const t = I.bicubicTab();
  for (let k = 0; k < t.length; k += 16) assert.equal(t.subarray(k, k + 16).reduce((a, b) => a + b, 0), 1 << 15);
});

test("min_area_rect_of_axis_box", () => {
  const r = I.minAreaRect([[0, 0], [10, 0], [10, 4], [0, 4]]);
  assert.ok(Math.abs(r.cx - 5) < 1e-4 && Math.abs(r.cy - 2) < 1e-4);
  const a = Math.max(r.w, r.h), b = Math.min(r.w, r.h);
  assert.ok(Math.abs(a - 10) < 1e-4 && Math.abs(b - 4) < 1e-4);
});

test("short_tokens", () => {
  for (const t of ["1", "12", " 7 ", "x3", "€1", "1,0"]) assert.ok(isShortToken(t), t);
  for (const t of ["", "|", "--", "1 2", "1234567"]) assert.ok(!isShortToken(t), t);
});

test("homoglyphs_fold_only_inside_latin_words", () => {
  assert.equal(cleanLatin("pt6а-42 Привет"), "pt6a-42 Привет");
});

test("clean_latin_folds_other_scripts_in_latin_lines", () => {
  assert.equal(cleanLatin("REPՍBLQՍE FRANÇAISE"), "REPUBLQUE FRANÇAISE");
  assert.equal(cleanLatin("C๐mmunicatoทร Security"), "Communicatons Security");
  assert.equal(cleanLatin("В - Linear 1- Center"), "B - Linear 1- Center");
  assert.equal(cleanLatin("KLAR 08 (UA०८ SHIP TERM"), "KLAR 08 (UA08 SHIP TERM");
  assert.equal(cleanLatin("GSTIN: 27ADC300६/१ZK"), "GSTIN: 27ADC3006/1ZK");
  assert.equal(cleanLatin("ZQX272003тOKEN OнMS"), "ZQX272003TOKEN OHMS");
  assert.equal(cleanLatin("airborлe пoise to гeduce"), "airborne noise to reduce");
  assert.equal(cleanLatin("aрганізацый ФOPYMA ПяТAГΑ"), "арганізацый ФОРУМА ПяТАГА");
  assert.equal(cleanLatin("significant at α = 0.05 level"), "significant at α = 0.05 level");
});

test("clean_latin_leaves_other_script_lines", () => {
  for (const s of [
    "Привет мир, как дела",
    "ReturnToWorkSA ਦਾ ਨਿਯਮ ਸਮੂਹ (Regulation Group) ਰਿਟਰਨ ਟੂ ਵਰਕ",
    "३ मार्च २०२४ को",
    "การบริหารความเสี่ยง ๒๕๖๗",
    "Հայաստանի Հանրապետություն",
    "ติดตั้ง Windowsนี้ ได้ที่ร้าน",
  ]) assert.equal(cleanLatin(s), s);
});

test("reading_frame_maps_quarter_turn_back", () => {
  const q = [[0, 0], [10, 0], [10, 2], [0, 2]];
  const r = I.readingFrame(q, false, true, 100);
  assert.deepEqual(r[0], [99, 0]);
  assert.deepEqual(r[1], [99, 10]);
});

// det.rs tests: a 60 x 20 map at one pixel per cell, one detected line (cells x 2..20, y 8..12)
function page() {
  const pw = 60, ph = 20;
  const prob = new Float32Array(pw * ph);
  for (let y = 8; y < 12; y++) for (let x = 2; x < 20; x++) prob[y * pw + x] = 0.9;
  const mask = Uint8Array.from(prob, (p) => (p > Math.fround(0.55) ? 1 : 0));
  const line = [[0, 0], [22, 0], [22, 20], [0, 20]];
  const img = { w: pw, h: ph, ch: 3, data: new Uint8Array(pw * ph * 3).fill(255) };
  return { prob, mask, boxes: [line], img };
}
const faint = (prob, mask, boxes, img) =>
  I.faintBoxes(prob, mask, 60, 20, (x, y) => [x, y], boxes, { thresh: 0.1, max: 24 }, img);

test("faint_token_in_a_row_is_a_candidate", () => {
  const { prob, mask, boxes, img } = page();
  for (let y = 8; y < 12; y++) prob[y * 60 + 40] = 0.13;
  const c = faint(prob, mask, boxes, img);
  assert.equal(c.length, 1);
  assert.deepEqual([c[0][0][1], c[0][2][1]], [0, 20]);
  assert.ok(c[0][0][0] < 40 && c[0][1][0] > 41);
});

test("faint_regions_off_row_touching_or_tall_are_not", () => {
  const { prob, mask, boxes, img } = page();
  prob[18 * 60 + 40] = 0.2;
  prob[19 * 60 + 40] = 0.2;
  prob[10 * 60 + 20] = 0.2;
  for (let y = 0; y < 20; y++) prob[y * 60 + 50] = 0.2;
  assert.equal(faint(prob, mask, boxes, img).length, 0);
});

// unicode-bidi reorder_line with an RTL paragraph (python-bidi get_display(s, base_dir="R"))
test("visual_to_logical", () => {
  // a visual-order Hebrew word comes back reversed; digits and Latin keep their order
  assert.equal(visualToLogical("םולש"), "שלום");
  assert.equal(visualToLogical("abc def"), "abc def");
  assert.equal(visualToLogical("2024 הנש"), "שנה 2024");
  // no mirroring (unicode-bidi reorder_line leaves L4 to the renderer)
  assert.equal(visualToLogical("(םולש)"), ")שלום(");
});

// 400 random strings over rec_rtl's charset against python-bidi 0.6.12 (Rust unicode-bidi) get_display(s, base_dir="R")
test("visual_to_logical_matches_python_bidi", () => {
  const cases = JSON.parse(fs.readFileSync(new URL("./bidi_cases.json", import.meta.url), "utf8"));
  for (const [s, want] of cases) assert.equal(visualToLogical(s), want, s);
});

console.log(`${n} tests passed`);
