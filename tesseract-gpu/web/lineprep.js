// Line clean-up before Tesseract's recognizer: flat-field the lighting, then crop tightly to the ink.
//
// cleanLine({data, width, height}) -> {data, width, height}
//   1. Inverted lines (light text on dark) are flipped, by prep.js's isInverted.
//   2. Flat-field: background = grey closing (max filter then min filter, k x k, k = max(9, 1.5 * height) | 1)
//      smoothed by a k x k box blur; out = gray / background * 255; then a 1st..99th percentile stretch to 0..255.
//   3. Tight crop: bounding box of pixels < 128 in the flat-fielded line, padded by 4 px, clamped.
//
// Port of flatfield() in run/flatfield_check.py and tight() in run/crop_check.py, with scipy.ndimage's default
// 'reflect' border (d c b a | a b c d | d c b a) and numpy's linear-interpolation percentile; test/check_lineprep.mjs
// compares it against the Python on fixture lines. Measured on eng tessdata_best (results/flatfield_check.json,
// run/crop_check.py): clean synthetic lines 1.06% -> 0.28% CER, shaded 1.48% -> 0.52%, real scans 14.2% -> 13.0%.
import { isInverted } from "./prep.js";

const PAD = 4;

function reflect(i, n) {
  // scipy 'reflect' (half-sample symmetric), for |i| < 2n
  if (i < 0) return -i - 1 < n ? -i - 1 : n - 1;
  if (i >= n) return 2 * n - i - 1 >= 0 ? 2 * n - i - 1 : 0;
  return i;
}

// 1-D running filter along rows (axis 1) or columns (axis 0); op is "max", "min" or "mean". Each row or column is
// extended by r = k >> 1 reflected samples on both sides; max/min use van Herk / Gil-Werman (three comparisons a
// sample whatever k is), the mean a float64 prefix sum.
function filter1d(src, w, h, k, axis, op) {
  const out = new Float32Array(w * h);
  const r = k >> 1;
  const n = axis === 1 ? w : h, m = axis === 1 ? h : w;
  const L = n + 2 * r;
  const ext = new Float64Array(L), g = new Float64Array(L), hh = new Float64Array(L), pre = new Float64Array(L + 1);
  const better = op === "max" ? (a, b) => (a > b ? a : b) : (a, b) => (a < b ? a : b);
  for (let j = 0; j < m; j++) {
    for (let e = 0; e < L; e++) {
      const i = reflect(e - r, n);
      ext[e] = axis === 1 ? src[j * w + i] : src[i * w + j];
    }
    if (op === "mean") {
      for (let e = 0; e < L; e++) pre[e + 1] = pre[e] + ext[e];
      for (let i = 0; i < n; i++) {
        const o = (pre[i + k] - pre[i]) / k;
        if (axis === 1) out[j * w + i] = o; else out[i * w + j] = o;
      }
      continue;
    }
    for (let e = 0; e < L; e++) g[e] = e % k === 0 ? ext[e] : better(g[e - 1], ext[e]);
    for (let e = L - 1; e >= 0; e--) hh[e] = (e % k === k - 1 || e === L - 1) ? ext[e] : better(hh[e + 1], ext[e]);
    for (let i = 0; i < n; i++) {
      const o = better(hh[i], g[i + k - 1]);
      if (axis === 1) out[j * w + i] = o; else out[i * w + j] = o;
    }
  }
  return out;
}

function percentile(sorted, q) {
  const pos = (sorted.length - 1) * q / 100;
  const lo = Math.floor(pos), hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function flatfield(gray, w, h) {
  const g = new Float32Array(w * h);
  const inv = isInverted(gray, w, h);
  for (let i = 0; i < g.length; i++) g[i] = inv ? 255 - gray[i] : gray[i];
  const k = Math.max(9, Math.trunc(1.5 * h) | 1);
  let bg = filter1d(filter1d(g, w, h, k, 1, "max"), w, h, k, 0, "max");
  bg = filter1d(filter1d(bg, w, h, k, 1, "min"), w, h, k, 0, "min");
  bg = filter1d(filter1d(bg, w, h, k, 1, "mean"), w, h, k, 0, "mean");
  const out = new Float32Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = g[i] / Math.max(bg[i], 1) * 255;
  const s = Float32Array.from(out).sort();
  const lo = percentile(s, 1), hi = percentile(s, 99);
  const res = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) {
    const v = hi > lo ? (out[i] - lo) * 255 / (hi - lo) : out[i];
    res[i] = v <= 0 ? 0 : v >= 255 ? 255 : Math.trunc(v); // numpy astype(uint8) after clip truncates
  }
  return res;
}

export function cleanLine(line) {
  const { data, width: w, height: h } = line;
  const ff = flatfield(data, w, h);
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (ff[y * w + x] < 128) {
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return { data: ff, width: w, height: h };
  // As measured (crop_check.py, tight+ff): the box comes from the flat-fielded line, the crop is taken from the
  // original and flat-fielded again, so the background estimate sees only the cropped region.
  const sx = Math.max(0, x0 - PAD), sy = Math.max(0, y0 - PAD);
  const ex = Math.min(w, x1 + PAD + 1), ey = Math.min(h, y1 + PAD + 1);
  const cw = ex - sx, ch = ey - sy;
  const crop = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) crop.set(data.subarray((sy + y) * w + sx, (sy + y) * w + sx + cw), y * cw);
  return { data: flatfield(crop, cw, ch), width: cw, height: ch };
}
