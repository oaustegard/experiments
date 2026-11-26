// Shared by run.mjs and refdiff.mjs: LiteParse's merge steps that touch OCR item text, and the page
// comparison (see run.mjs for the metrics).
export const DPI = 250;

// LiteParse's merge, the parts that touch an OCR item's text and the frame of its box

/** ocr_merge.rs `clean_ocr_table_artifacts`. */
export function cleanOcrTableArtifacts(text) {
  const trimmed = text.trim();
  if (!trimmed) return "";
  const strip = (s) => s.replace(/^[|[\](){}]+/, "").replace(/[|[\](){}]+$/, "");
  const without = strip(trimmed).trim();
  if (!without) return trimmed;
  const numericIsh = /^[0-9,. %\-+*/]*$/.test(without) || without === "N/A" || without === "Z" || without === "-";
  if (numericIsh) {
    const at = trimmed.indexOf(without);
    const before = trimmed.slice(0, at), after = trimmed.slice(at + without.length);
    if (before.endsWith("(") && after.startsWith(")")) return `(${without})`;
    return without;
  }
  return trimmed;
}

export function canonicalQuarter(rot) {
  const r = ((rot % 360) + 360) % 360;
  return (Math.round(r / 90) % 4) * 90;
}
/** ocr_merge.rs `polygon_rotation_deg`. */
export function polygonRotation(p) {
  const e0 = [p[1][0] - p[0][0], p[1][1] - p[0][1]], e1 = [p[2][0] - p[1][0], p[2][1] - p[1][1]];
  const l0 = Math.hypot(...e0), l1 = Math.hypot(...e1);
  if (Math.max(l0, l1) < 1) return 0;
  const [lg, sh] = l0 >= l1 ? [l0, l1] : [l1, l0];
  if (sh > 0 && lg / sh < 1.3) return 0;
  const r = l0 >= l1 ? e0 : e1;
  return canonicalQuarter((-Math.atan2(r[1], r[0]) * 180) / Math.PI);
}

/** Our results as LiteParse text items (points), with `upright_rotated_scan` applied. */
export function toItems(results, pw, ph) {
  const s = 72 / DPI;
  let items = [];
  for (const r of results) {
    if (r.confidence <= 0.1) continue;
    const text = cleanOcrTableArtifacts(r.text);
    if (!text) continue;
    const xs = r.polygon.map((p) => p[0]), ys = r.polygon.map((p) => p[1]);
    const x0 = Math.min(...xs), y0 = Math.min(...ys);
    items.push({
      text, conf: r.confidence, faint: r.faint, rotation: polygonRotation(r.polygon),
      x: x0 * s, y: y0 * s, w: (Math.max(...xs) - x0) * s, h: (Math.max(...ys) - y0) * s,
    });
  }
  const chars = [0, 0, 0, 0];
  for (const it of items) chars[it.rotation / 90] += [...it.text].length;
  const total = chars.reduce((a, b) => a + b, 0);
  let q = 0;
  for (let i = 1; i < 4; i++) if (chars[i] > chars[q]) q = i;
  if (q !== 0 && total >= 20 && chars[q] >= total * 0.7) {
    items = items.map((it) => {
      const { x, y, w, h } = it;
      const t = q === 1 ? [ph - (y + h), x, h, w] : q === 2 ? [pw - (x + w), ph - (y + h), w, h] : [y, pw - (x + w), h, w];
      return { ...it, x: t[0], y: t[1], w: t[2], h: t[3], rotation: 0 };
    });
  }
  return items;
}

// comparison

export function levenshtein(a, b) {
  const A = [...a], B = [...b];
  if (!A.length) return B.length;
  if (!B.length) return A.length;
  let prev = new Int32Array(B.length + 1), cur = new Int32Array(B.length + 1);
  for (let j = 0; j <= B.length; j++) prev[j] = j;
  for (let i = 1; i <= A.length; i++) {
    cur[0] = i;
    const ai = A[i - 1];
    for (let j = 1; j <= B.length; j++) {
      const sub = prev[j - 1] + (ai === B[j - 1] ? 0 : 1);
      const del = prev[j] + 1, ins = cur[j - 1] + 1;
      cur[j] = sub < del ? (sub < ins ? sub : ins) : del < ins ? del : ins;
    }
    [prev, cur] = [cur, prev];
  }
  return prev[B.length];
}

/** The detector's order: 8 px row bands of the top edge, then the left edge. */
export function readingOrder(items) {
  const band = (8 * 72) / DPI;
  return items.slice().sort((a, b) => (Math.round(a.y / band) - Math.round(b.y / band)) || (a.x - b.x));
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy, uni = a.w * a.h + b.w * b.h - inter;
  return uni > 0 ? inter / uni : 0;
}

/** Runs of 2-6 side-by-side items of ours, left to right, joined by " " and by "": the strings
 * LiteParse's projection makes when it merges neighbouring OCR items on one line (gutter-split
 * pieces of one detector line, a marker and its text, a superscript). Items share a row when their
 * vertical extents overlap by half the smaller height. */
function rowRuns(items) {
  const byX = items.slice().sort((a, b) => a.x - b.x);
  const out = new Set();
  for (const a of byX) {
    const row = byX.filter((b) => b !== a && b.x >= a.x &&
      Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y) >= 0.5 * Math.min(a.h, b.h));
    // every mix of " " and "" separators for up to 3 joins, uniform ones beyond
    let partial = [a.text];
    let s1 = a.text, s2 = a.text;
    for (let k = 0; k < Math.min(5, row.length); k++) {
      s1 += " " + row[k].text;
      s2 += row[k].text;
      out.add(s1);
      out.add(s2);
      if (k < 3) {
        partial = partial.flatMap((p) => [p + " " + row[k].text, p + row[k].text]);
        for (const p of partial) out.add(p);
      }
    }
  }
  return out;
}

/** Order-free CER: each of our items goes to the reference item covering most of it (at least half);
 * a reference item is compared with its items joined by " " in x order; unassigned items of ours
 * count as insertions. */
function itemCer(ref, ours) {
  const assigned = ref.map(() => []);
  let ins = 0;
  for (const o of ours) {
    // among reference items covering at least half of it: one whose text holds ours, else the most covering
    let best = -1, bestKey = -1;
    ref.forEach((r, k) => {
      const ix = Math.max(0, Math.min(r.x + r.w, o.x + o.w) - Math.max(r.x, o.x));
      const iy = Math.max(0, Math.min(r.y + r.h, o.y + o.h) - Math.max(r.y, o.y));
      const frac = (ix * iy) / Math.max(o.w * o.h, 1e-9);
      if (frac < 0.5) return;
      const key = (r.text.includes(o.text) ? 2 : 0) + frac;
      if (key > bestKey) {
        bestKey = key;
        best = k;
      }
    });
    if (best >= 0) assigned[best].push(o);
    else ins += [...o.text].length;
  }
  let edits = ins, chars = 0;
  ref.forEach((r, k) => {
    const got = assigned[k].sort((a, b) => a.x - b.x).map((o) => o.text).join(" ");
    edits += levenshtein(got, r.text);
    chars += [...r.text].length;
  });
  return { item_edits: edits, item_chars: chars };
}

export function compare(ref, ours) {
  const refText = readingOrder(ref).map((i) => i.text).join("\n");
  const ourText = readingOrder(ours).map((i) => i.text).join("\n");
  const runs = rowRuns(ours);
  const found = (t) => ourText.includes(t);
  const refFound = ref.filter((r) => found(r.text)).length;
  const refFoundMerged = ref.filter((r) => found(r.text) || runs.has(r.text)).length;
  const oursFound = ours.filter((o) => refText.includes(o.text)).length;
  const pool = new Map();
  for (const o of ours) pool.set(o.text, (pool.get(o.text) || 0) + 1);
  let exact = 0;
  for (const r of ref) {
    const c = pool.get(r.text) || 0;
    if (c > 0) {
      exact++;
      pool.set(r.text, c - 1);
    }
  }
  let boxOk = 0, iouSum = 0;
  for (const r of ref) {
    let best = 0;
    for (const o of ours) best = Math.max(best, iou(r, o));
    iouSum += best;
    if (best >= 0.5) boxOk++;
  }
  const edits = levenshtein(ourText, refText);
  const refChars = [...refText].length;
  return {
    ref_items: ref.length, our_items: ours.length,
    ref_found: refFound, ref_found_merged: refFoundMerged, ours_found: oursFound, exact, box_ok: boxOk,
    mean_iou: ref.length ? iouSum / ref.length : 1,
    edits, ref_chars: refChars, cer: refChars ? edits / refChars : 0,
    ...itemCer(ref, ours),
    missing: ref.filter((r) => !found(r.text)).map((r) => r.text),
    missing_merged: ref.filter((r) => !found(r.text) && !runs.has(r.text)).map((r) => r.text),
    extra: ours.filter((o) => !refText.includes(o.text)).map((o) => o.text),
  };
}
