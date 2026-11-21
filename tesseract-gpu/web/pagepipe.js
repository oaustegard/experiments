// The page pipeline shared by app.js and ocr.js: clean up Tesseract's layout boxes, crop a line, tidy its text.
export const PAD = 3; // px of page around each line's box, clamped to the page

// Tesseract's layout returns some "lines" that are not one text line: table rules (a few px wide, hundreds tall),
// logos and banners, cells holding several lines. Drop boxes too narrow to hold text, and split any box taller than
// 1.8 median line heights at its empty or ruled pixel rows; pieces shorter than 0.4 median heights are dropped; then
// drop duplicates (dedupe).
export function medianOf(v) { const s = v.slice().sort((a, b) => a - b); return s.length ? s[s.length >> 1] : 0; }

export function refineBoxes(gray, boxes) {
  const med = medianOf(boxes.map((b) => b[3] - b[1])) || 20;
  const out = [];
  for (const b of boxes) {
    const [x0, y0, x1, y1] = b.map(Math.round);
    const w = x1 - x0, h = y1 - y0;
    if (w < 1.2 * med) continue;
    if (h <= 1.8 * med) { out.push(b); continue; }
    let lo = 255, hi = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const v = gray.data[y * gray.w + x]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const thr = (lo + hi) / 2;
    const ink = [];
    for (let y = y0; y < y1; y++) {
      let n = 0;
      for (let x = x0; x < x1; x++) if (gray.data[y * gray.w + x] < thr) n++;
      ink.push(n > 0.002 * w && n < 0.6 * w); // a row that is mostly ink is a ruling, not text
    }
    let start = -1;
    for (let i = 0; i <= ink.length; i++) {
      if (i < ink.length && ink[i]) { if (start < 0) start = i; continue; }
      if (start >= 0 && i - start >= 0.4 * med) out.push([x0, Math.max(y0, y0 + start - 2), x1, Math.min(y1, y0 + i + 2)]);
      start = -1;
    }
  }
  return dedupe(out);
}

// Table regions come back twice, as rows and again inside a table-wide block: where two boxes overlap by more than
// half of the smaller, keep the smaller (tighter) one. Reading order is kept.
export function dedupe(boxes) {
  const area = (b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const inter = (a, b) => Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const idx = boxes.map((_, i) => i).sort((i, j) => area(boxes[i]) - area(boxes[j]));
  const kept = [];
  for (const i of idx) {
    if (!kept.some((k) => inter(boxes[i], boxes[k]) > 0.5 * Math.min(area(boxes[i]), area(boxes[k])))) kept.push(i);
  }
  const keep = new Set(kept);
  return boxes.filter((_, i) => keep.has(i));
}

// Table rules inside a line read as runs of =, ~, —, _ or |: drop tokens made only of such characters, and drop a
// line with fewer than three letters or digits (a speck or a fragment of a ruling).
export const MIN_CONF = 0.6;
export const RULE_TOKEN = /^[=~—–_|§«»‘’“”'".,:;·•\-]+$/;
export function tidy(t) {
  const kept = t.split(/\s+/).filter((w) => w && !RULE_TOKEN.test(w)).join(" ");
  return (kept.match(/[\p{L}\p{N}]/gu) || []).length >= 3 ? kept : "";
}

// One crop per line: axis-aligned, padded by PAD px, clamped to the page. null for a box with no area.
export function cropOf(gray, box) {
  const [x0, y0, x1, y1] = box;
  const sx = Math.max(0, Math.floor(x0) - PAD), sy = Math.max(0, Math.floor(y0) - PAD);
  const ex = Math.min(gray.w, Math.ceil(x1) + PAD), ey = Math.min(gray.h, Math.ceil(y1) + PAD);
  const w = ex - sx, h = ey - sy;
  if (w < 2 || h < 2) return null;
  const data = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const o = (sy + y) * gray.w + sx;
    data.set(gray.data.subarray(o, o + w), y * w);
  }
  return { data, width: w, height: h };
}
