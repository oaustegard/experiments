// A PDF page's own text layer, as text lines with boxes in the rendered page's pixels, so that born-digital pages
// skip OCR. pdf.js (Apache-2.0) gives the text runs; this file groups them into lines, checks the layer for unmapped
// glyphs, and says whether the page paints any images (text inside an image is not in the layer and still needs OCR).
// No model is involved: this is the PDF's own text. The idea follows Indra OCR's "auto" mode (read the text layer
// where it passes checks, OCR elsewhere); no Indra code or weights are used.

const ASCENT = 0.8, DESCENT = -0.2; // fallbacks when pdf.js gives no font metrics
const JOIN_GAP = 1.0;               // a run joins the line when the gap before it is under this many font heights
const SPACE_GAP = 0.12;             // ... and gets a space before it when the gap is over this many
const SAME_BASE = 0.35;             // baselines closer than this many font heights are the same line
const BAD_SHARE = 0.05;             // more than this share of unmapped characters fails the layer

// U+FFFD, private-use code points and control characters are what a font without a usable ToUnicode map yields.
const BAD_CHAR = /[�-\u{F0000}-\u{10FFFF}\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/gu;

function boxOfRun(item, style, vpTransform, Util) {
  const tx = Util.transform(vpTransform, item.transform);
  const fh = Math.hypot(tx[2], tx[3]);
  if (!(fh > 0)) return null;
  const angle = Math.atan2(tx[1], tx[0]);
  const scale = Math.hypot(vpTransform[0], vpTransform[1]) || 1; // viewport px per PDF unit
  const len = item.width * scale;
  const asc = style && Number.isFinite(style.ascent) && style.ascent > 0 ? style.ascent : ASCENT;
  const desc = style && Number.isFinite(style.descent) && style.descent < 0 ? style.descent : DESCENT;
  const cos = Math.cos(angle), sin = Math.sin(angle);
  // corners of the run's rectangle: along the baseline by len, up by asc*fh, down by -desc*fh (y grows downward)
  const pts = [[0, -asc * fh], [len, -asc * fh], [len, -desc * fh], [0, -desc * fh]].map(([u, v]) => [tx[4] + u * cos - v * sin, tx[5] + u * sin + v * cos]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return { box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], fh, base: tx[5], x0: tx[4], x1: tx[4] + len * cos, flat: Math.abs(sin) < 0.035 };
}

// Group pdf.js text runs, in content-stream order, into lines. A run joins the open line when both are horizontal,
// share a baseline (or the run is a smaller super- or subscript inside the line's height), and start just after it.
export function groupRuns(items, styles, vpTransform, Util, pageW, pageH) {
  const lines = [];
  let cur = null;
  const close = () => {
    if (cur && cur.text.trim()) lines.push({ box: cur.box, text: cur.text.replace(/\s+/g, " ").trim() });
    cur = null;
  };
  for (const item of items) {
    if (typeof item.str !== "string") continue; // marked-content markers
    if (item.str === "" || !item.str.trim()) {
      if (cur && item.str && !cur.text.endsWith(" ")) cur.text += " ";
      if (item.hasEOL) close();
      continue;
    }
    const r = boxOfRun(item, styles[item.fontName], vpTransform, Util);
    if (!r) continue;
    const [bx0, by0, bx1, by1] = r.box;
    if (bx1 < 0 || by1 < 0 || bx0 > pageW || by0 > pageH) continue; // off the page: never seen, never read
    let join = false;
    if (cur && cur.flat && r.flat) {
      const gap = r.x0 - cur.x1;
      const h = Math.max(cur.fh, r.fh);
      const sameBase = Math.abs(r.base - cur.base) < SAME_BASE * h;
      const inside = r.fh < 0.8 * cur.fh && r.box[1] >= cur.box[1] - 0.2 * cur.fh && r.box[3] <= cur.box[3] + 0.2 * cur.fh;
      join = (sameBase || inside) && gap > -0.5 * h && gap < JOIN_GAP * h;
      if (join && gap > SPACE_GAP * h && !cur.text.endsWith(" ") && !item.str.startsWith(" ")) cur.text += " ";
    }
    if (join) {
      cur.text += item.str;
      cur.box = [Math.min(cur.box[0], bx0), Math.min(cur.box[1], by0), Math.max(cur.box[2], bx1), Math.max(cur.box[3], by1)];
      cur.x1 = Math.max(cur.x1, r.x1);
    } else {
      close();
      cur = { text: item.str, box: r.box.slice(), fh: r.fh, base: r.base, x1: r.x1, flat: r.flat };
    }
    if (item.hasEOL) close();
  }
  close();
  // clamp to the page
  for (const l of lines) l.box = [Math.max(0, l.box[0]), Math.max(0, l.box[1]), Math.min(pageW, l.box[2]), Math.min(pageH, l.box[3])];
  return lines.filter((l) => l.box[2] - l.box[0] >= 1 && l.box[3] - l.box[1] >= 1);
}

// The layer's own verdict, before any comparison with the page image.
export function judgeLayer(lines) {
  const text = lines.map((l) => l.text).join("");
  const chars = text.replace(/\s/g, "").length;
  const bad = (text.match(BAD_CHAR) || []).length;
  if (!chars) return { ok: false, why: "the page has no text layer", chars, bad };
  if (bad > BAD_SHARE * chars) return { ok: false, why: `${bad} of ${chars} characters in the text layer are unmapped glyphs`, chars, bad };
  return { ok: true, why: "", chars, bad };
}

const IMAGE_OPS = ["paintImageXObject", "paintInlineImageXObject", "paintImageMaskXObject", "paintImageXObjectRepeat",
  "paintInlineImageXObjectGroup", "paintImageMaskXObjectGroup", "paintImageMaskXObjectRepeat", "paintSolidColorImageMask"];

// Text lines of one pdf.js page, in the pixels of `viewport`, plus whether the page paints images.
export async function pageTextLayer(pdfjs, page, viewport) {
  const t0 = performance.now();
  const tc = await page.getTextContent();
  const lines = groupRuns(tc.items, tc.styles || {}, viewport.transform, pdfjs.Util, viewport.width, viewport.height);
  const ops = await page.getOperatorList();
  const imageOps = new Set(IMAGE_OPS.map((k) => pdfjs.OPS[k]).filter((v) => v !== undefined));
  const images = ops.fnArray.reduce((n, fn) => n + (imageOps.has(fn) ? 1 : 0), 0);
  return { lines, verdict: judgeLayer(lines), images, runs: tc.items.length, ms: performance.now() - t0 };
}

// Share of box b's area covered by the union of boxes (approximated by the sum of overlaps, capped at 1).
export function coverage(b, boxes) {
  const area = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  if (!area) return 1;
  let s = 0;
  for (const o of boxes) s += Math.max(0, Math.min(b[2], o[2]) - Math.max(b[0], o[0])) * Math.max(0, Math.min(b[3], o[3]) - Math.max(b[1], o[1]));
  return Math.min(1, s / area);
}

// Character error rate between two strings after NFKC, lower case and removing whitespace (Levenshtein / len(ref)).
export function looseCer(ref, hyp) {
  const norm = (s) => s.normalize("NFKC").toLowerCase().replace(/\s+/g, "");
  const a = [...norm(ref)], b = [...norm(hyp)];
  if (!a.length) return b.length ? 1 : 0;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    for (let j = 1; j <= b.length; j++) row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = row;
  }
  return prev[b.length] / a.length;
}
