// Reading order and markdown, ported from indra-ocr's order_post.py and layout_post.py.
// boxes are arrays of [x0, y0, x1, y1]; N x N logit matrices are flat Float32Arrays.

import { pyRound } from "./imgproc.js";
import { isDigit, isUpper, isWide } from "./uprops.js";
import { htmlEscape, joinLines, pyStrip } from "./text.js";

export const BLOCK_CLASSES = ["Text", "Section-Header", "Page-Header", "Page-Footer", "List-Group", "Table",
  "Caption", "Footnote", "Table-Of-Contents", "Form"];
export const GRID_LABELS = new Set(["Table", "Form", "Table-Of-Contents"]);
const CLASS_ID = Object.fromEntries(BLOCK_CLASSES.map((c, i) => [c, i]));
const HEADER = CLASS_ID["Page-Header"], FOOTER = CLASS_ID["Page-Footer"], SECTION = CLASS_ID["Section-Header"];
const LIST = CLASS_ID["List-Group"], TABLE = CLASS_ID["Table"];

export function median(a) {
  const s = Float64Array.from(a).sort();
  const n = s.length;
  if (!n) return NaN;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
const argsort = (v) => v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
const chars = (s) => Array.from(s);

const SENTENCE_END = new Set(chars(".!?:;。！？：；．"));

export function lineFeatures(boxes, W, H, texts) {
  const n = boxes.length;
  const f = new Float32Array(n * 16);
  if (!n) return f;
  const hs = boxes.map((b) => Math.max(Math.fround(b[3] - b[1]), 1e-3));
  const medH = median(hs);
  for (let i = 0; i < n; i++) {
    const b = boxes[i];
    const w = Math.max(Math.fround(b[2] - b[0]), 1e-3), h = hs[i];
    const g = [b[0] / W, b[1] / H, b[2] / W, b[3] / H, (b[0] + b[2]) / (2 * W), (b[1] + b[3]) / (2 * H), w / W,
      Math.log(h / medH), Math.log(w / h) / 4.0, (medH / H) * 20.0];
    g.forEach((v, k) => { f[i * 16 + k] = v; });
    const t = texts ? pyStrip(texts[i]) : "";
    if (!t) continue;
    const tc = chars(t);
    const disp = tc.reduce((s, c) => s + (isWide(c) ? 2 : 1), 0);
    f[i * 16 + 10] = Math.log1p(disp) / 5.0;
    f[i * 16 + 11] = tc.filter(isDigit).length / tc.length;
    const head = tc.slice(0, 3).join("").replace(/[.)]+$/u, "");
    f[i * 16 + 12] = "•-–▪◦*·→".includes(tc[0]) || (head.length > 0 && chars(head).every(isDigit) && tc.length > 3) ? 1 : 0;
    f[i * 16 + 13] = t.endsWith("-") ? 1 : 0;
    f[i * 16 + 14] = SENTENCE_END.has(tc[tc.length - 1]) ? 1 : 0;
    f[i * 16 + 15] = isUpper(tc[0]) ? 1 : 0;
  }
  return f;
}

const logaddexp0 = (x) => (x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x)));

export function decodeOrder(a, n, sweeps = 4) {
  if (n <= 1) return Array.from({ length: n }, (_, i) => i);
  const lp = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) lp[i] = -logaddexp0(-a[i]);
  const key = Array.from({ length: n }, (_, i) => {
    let r = 0, c = 0;
    for (let j = 0; j < n; j++) { r += lp[i * n + j]; c += lp[j * n + i]; }
    return -r + c;
  });
  let order = argsort(key);
  for (let s = 0; s < sweeps; s++) {
    let improved = false;
    for (const x of order.slice()) {
      const rest = order.filter((v) => v !== x);
      const m = rest.length;
      const gains = new Float64Array(m + 1);
      let acc = 0;
      const pre = new Float64Array(m + 1);
      for (let k = 0; k < m; k++) { acc += lp[rest[k] * n + x]; pre[k + 1] = acc; }
      const suf = new Float64Array(m + 1);
      acc = 0;
      for (let k = m - 1; k >= 0; k--) { acc += lp[x * n + rest[k]]; suf[k] = acc; }
      let best = 0;
      for (let k = 0; k <= m; k++) { gains[k] = pre[k] + suf[k]; if (gains[k] > gains[best]) best = k; }
      const cur = order.indexOf(x);
      if (gains[best] > gains[cur] + 1e-6) {
        order = [...rest.slice(0, best), x, ...rest.slice(best)];
        improved = true;
      }
    }
    if (!improved) break;
  }
  return order;
}

export function orderTopDown(boxes) {
  if (!boxes.length) return [];
  const cy = boxes.map((b) => (b[1] + b[3]) / 2);
  const h = median(boxes.map((b) => b[3] - b[1]));
  const q = Math.max(h * 0.5, 1e-3);
  return boxes.map((_, i) => i).sort((i, j) => {
    const a = pyRound(cy[i] / q), b = pyRound(cy[j] / q);
    return a !== b ? a - b : boxes[i][0] - boxes[j][0];
  });
}

function bestCut(b, axis) {
  const lo = b.map((x) => (axis === 1 ? x[1] : x[0])), hi = b.map((x) => (axis === 1 ? x[3] : x[2]));
  const o = argsort(lo);
  let end = hi[o[0]], bestGap = 0, bestK = -1;
  for (let k = 1; k < o.length; k++) {
    const gap = lo[o[k]] - end;
    if (gap > bestGap) { bestGap = gap; bestK = k; }
    end = Math.max(end, hi[o[k]]);
  }
  return [bestGap, bestK, o];
}

export function orderXycut(boxes) {
  const rec = (ids) => {
    if (ids.length <= 1) return ids.slice();
    const b = ids.map((i) => boxes[i]);
    const [gy, ky, oy] = bestCut(b, 1), [gx, kx, ox] = bestCut(b, 0);
    if (ky < 0 && kx < 0) return orderTopDown(b).map((i) => ids[i]);
    const [k, o] = gy >= gx ? [ky, oy] : [kx, ox];
    return [...rec(o.slice(0, k).map((i) => ids[i])), ...rec(o.slice(k).map((i) => ids[i]))];
  };
  return rec(boxes.map((_, i) => i));
}

export function xycutBlocks(boxes, minGap = 1.0) {
  if (!boxes.length) return [];
  const h = median(boxes.map((b) => b[3] - b[1]));
  const rec = (ids) => {
    const b = ids.map((i) => boxes[i]);
    if (ids.length > 1) {
      const [gy, ky, oy] = bestCut(b, 1), [gx, kx, ox] = bestCut(b, 0);
      const [gap, k, o] = gy >= gx ? [gy, ky, oy] : [gx, kx, ox];
      if (k >= 0 && gap >= minGap * h) return [...rec(o.slice(0, k).map((i) => ids[i])), ...rec(o.slice(k).map((i) => ids[i]))];
    }
    return [orderXycut(b).map((i) => ids[i])];
  };
  return rec(boxes.map((_, i) => i));
}

// average-linkage clustering of "same group" logits [n x n] (order_post._cluster)
export function cluster(L, n) {
  if (n === 0) return [];
  const S = Float64Array.from(L);
  for (let i = 0; i < n; i++) S[i * n + i] = 0;
  const size = new Float64Array(n).fill(1), alive = new Array(n).fill(true);
  const members = Array.from({ length: n }, (_, i) => [i]);
  const M = Float64Array.from(S);
  for (let i = 0; i < n; i++) M[i * n + i] = -Infinity;
  const best = new Int32Array(n), bestV = new Float64Array(n);
  const rowMax = (j) => { let k = 0; for (let c = 1; c < n; c++) if (M[j * n + c] > M[j * n + k]) k = c; return k; };
  for (let i = 0; i < n; i++) { best[i] = rowMax(i); bestV[i] = M[i * n + best[i]]; }
  for (let it = 0; it < n - 1; it++) {
    let a = 0;
    for (let i = 1; i < n; i++) if (bestV[i] > bestV[a]) a = i;
    let b = best[a];
    if (!(bestV[a] > 0)) break;
    if (a > b) [a, b] = [b, a];
    for (let c = 0; c < n; c++) S[a * n + c] += S[b * n + c];
    for (let r = 0; r < n; r++) S[r * n + a] += S[r * n + b];
    S[a * n + a] = 0;
    size[a] += size[b];
    alive[b] = false;
    members[a] = members[a].concat(members[b]);
    const row = new Float64Array(n);
    for (let c = 0; c < n; c++) row[c] = alive[c] && c !== a ? S[a * n + c] / (size[a] * size[c]) : -Infinity;
    for (let c = 0; c < n; c++) { M[a * n + c] = row[c]; M[c * n + a] = row[c]; M[b * n + c] = -Infinity; M[c * n + b] = -Infinity; }
    bestV[b] = -Infinity;
    best[a] = 0;
    for (let c = 1; c < n; c++) if (row[c] > row[best[a]]) best[a] = c;
    bestV[a] = row[best[a]];
    for (let j = 0; j < n; j++) {
      if (!alive[j] || j === a) continue;
      if (best[j] === a || best[j] === b) { best[j] = rowMax(j); bestV[j] = M[j * n + best[j]]; }
    }
    for (let j = 0; j < n; j++) {
      if (!alive[j] || j === a) continue;
      if (row[j] > bestV[j]) { best[j] = a; bestV[j] = row[j]; }
    }
  }
  const out = [];
  for (let i = 0; i < n; i++) if (alive[i]) out.push(members[i].slice().sort((x, y) => x - y));
  return out;
}

const sub = (L, n, idx) => {
  const m = idx.length, out = new Float32Array(m * m);
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) out[i * m + j] = L[idx[i] * n + idx[j]];
  return out;
};

const clsArgmax = (cls, idx) => {
  const s = new Float64Array(10);
  for (const i of idx) for (let c = 0; c < 10; c++) s[c] += cls[i * 10 + c];
  let k = 0;
  for (let c = 1; c < 10; c++) if (s[c] > s[k]) k = c;
  return k;
};

export function decodeBlocksOrder(a, same, boxes, cls, n) {
  if (n <= 1) return Array.from({ length: n }, (_, i) => i);
  const blocks = cluster(same, n);
  const grid = new Set([...GRID_LABELS].map((c) => CLASS_ID[c]));
  const inner = blocks.map((b) => {
    const bx = b.map((i) => boxes[i]);
    const local = cls && grid.has(clsArgmax(cls, b)) ? orderTopDown(bx) : orderXycut(bx);
    return local.map((k) => b[k]);
  });
  const m = blocks.length;
  const ab = new Float32Array(m * m);
  for (let p = 0; p < m; p++)
    for (let q = p + 1; q < m; q++) {
      let s = 0;
      for (const i of blocks[p]) for (const j of blocks[q]) s += a[i * n + j];
      const v = s / (blocks[p].length * blocks[q].length);
      ab[p * m + q] = v;
      ab[q * m + p] = -v;
    }
  return decodeOrder(ab, m).flatMap((k) => inner[k]);
}

// ------------------------------------------------------------------ layout_post

const WHOLE_MATH = /^\$[^$]+\$$/u;
const ITEM = /^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)/u;

export function splitBlocks(order, same, n) {
  const blocks = [];
  for (const i of order) {
    if (blocks.length && same[blocks.at(-1).at(-1) * n + i] > 0) blocks.at(-1).push(i);
    else blocks.push([i]);
  }
  return blocks;
}

function rowsOf(boxes) {
  const cy = boxes.map((b) => (b[1] + b[3]) / 2);
  const h = median(boxes.map((b) => b[3] - b[1]));
  const rows = [];
  for (const i of argsort(cy)) {
    if (rows.length && Math.abs(cy[i] - rows.at(-1).reduce((s, k) => s + cy[k], 0) / rows.at(-1).length) < 0.5 * h) rows.at(-1).push(i);
    else rows.push([i]);
  }
  return rows.map((r) => r.slice().sort((x, y) => boxes[x][0] - boxes[y][0]));
}

function mergeWrapped(rows, boxes) {
  if (rows.length < 3) return rows;
  const h = median(boxes.map((b) => b[3] - b[1]));
  const gaps = rows.slice(0, -1).map((cur, k) => Math.min(...rows[k + 1].map((i) => boxes[i][1])) - Math.max(...cur.map((i) => boxes[i][3])));
  const med = median(gaps);
  const out = [rows[0].slice()];
  let prev = rows[0];
  gaps.forEach((g, k) => {
    const r = rows[k + 1];
    if (g < 0.5 * med && g < 0.6 * h && r.length < prev.length) out.at(-1).push(...r);
    else out.push(r.slice());
    prev = r;
  });
  return out;
}

function searchRight(edges, x) {
  let lo = 0, hi = edges.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (edges[m] <= x) lo = m + 1; else hi = m; }
  return lo;
}

export function tableHtml(boxes, texts) {
  let rows = rowsOf(boxes);
  const spanX = Math.max(...boxes.map((b) => b[2])) - Math.min(...boxes.map((b) => b[0]));
  if (rows.length > 2 && rows[0].length === 1 && Math.max(...rows.map((r) => r.length)) > 1 &&
      boxes[rows[0][0]][2] - boxes[rows[0][0]][0] > 0.6 * spanX) {
    const keep = texts.map((_, k) => k).filter((k) => k !== rows[0][0]);
    return texts[rows[0][0]] + "\n\n" + tableHtml(keep.map((k) => boxes[k]), keep.map((k) => texts[k]));
  }
  const width = Math.max(...rows.map((r) => r.length));
  const full = rows.filter((r) => r.length === width);
  rows = mergeWrapped(rows, boxes);
  const bands = Array.from({ length: width }, (_, c) => [median(full.map((r) => boxes[r[c]][0])), median(full.map((r) => boxes[r[c]][2]))]);
  const edges = [-Infinity, ...Array.from({ length: width - 1 }, (_, c) => (bands[c][1] + bands[c + 1][0]) / 2), Infinity];
  const out = ["<table>"];
  rows.forEach((r, ri) => {
    const tag = ri === 0 && rows.length > 1 ? "th" : "td";
    const cells = new Map();
    for (const k of r.slice().sort((x, y) => boxes[x][1] - boxes[y][1] || boxes[x][0] - boxes[y][0])) {
      const [x0, , x1] = boxes[k];
      const first = searchRight(edges, x0) - 1, last = searchRight(edges, x1) - 1;
      let span = [];
      for (let c = Math.max(first, 0); c <= Math.min(last, width - 1); c++)
        if (Math.min(x1, bands[c][1]) - Math.max(x0, bands[c][0]) > 0.5 * (bands[c][1] - bands[c][0])) span.push(c);
      if (!span.length) span = [Math.min(Math.max(first, 0), width - 1)];
      const c0 = span[0], prev = cells.get(c0);
      const text = prev === undefined ? texts[k] : prev[1] + " " + texts[k];
      cells.set(c0, [prev === undefined ? span.length : Math.max(prev[0], span.length), text]);
    }
    const tds = [];
    let c = 0;
    while (c < width) {
      if (cells.has(c)) {
        const [n, text] = cells.get(c);
        tds.push(`<${tag}${n > 1 ? ` colspan="${n}"` : ""}>${htmlEscape(text)}</${tag}>`);
        c += n;
      } else { tds.push(`<${tag}></${tag}>`); c += 1; }
    }
    out.push("<tr>" + tds.join("") + "</tr>");
  });
  out.push("</table>");
  return out.join("\n");
}

export function pageBlocks(boxes, texts, order, cls, same, n) {
  return splitBlocks(order, same, n).map((blk) => {
    const b = blk.map((i) => boxes[i]);
    return {
      label: BLOCK_CLASSES[clsArgmax(cls, blk)],
      bbox: [Math.min(...b.map((x) => x[0])), Math.min(...b.map((x) => x[1])), Math.max(...b.map((x) => x[2])), Math.max(...b.map((x) => x[3]))].map((v) => pyRound(v * 10) / 10),
      text: blk.map((i) => texts[i]).join("\n"),
    };
  });
}

export function pageMarkdown(boxes, texts, order, cls, same, n, { keepFurniture = false, cells = null, pictures = null } = {}) {
  let parts = [];
  const partBoxes = [];
  for (const blk of splitBlocks(order, same, n)) {
    const before = parts.length;
    blockPart(parts, blk, boxes, texts, cls, keepFurniture, cells);
    if (parts.length > before) {
      const b = blk.map((i) => boxes[i]);
      partBoxes.push([Math.min(...b.map((x) => x[0])), Math.min(...b.map((x) => x[1])), Math.max(...b.map((x) => x[2])), Math.max(...b.map((x) => x[3]))]);
    }
  }
  if (pictures && pictures.length) {
    const slots = new Map();
    pictures.forEach(([x0, y0, x1, y1], k) => {
      const cy = (y0 + y1) / 2;
      let at = partBoxes.findIndex(([bx0, by0, bx1, by1]) => (by0 + by1) / 2 > cy && Math.min(x1, bx1) > Math.max(x0, bx0));
      if (at < 0) at = parts.length;
      if (!slots.has(at)) slots.set(at, []);
      slots.get(at).push(`![](picture-${k})`);
    });
    const merged = [];
    for (let j = 0; j <= parts.length; j++) { merged.push(...(slots.get(j) ?? [])); if (j < parts.length) merged.push(parts[j]); }
    parts = merged;
  }
  return parts.filter((p) => p).join("\n\n");
}

function blockPart(parts, blk, boxes, texts, cls, keepFurniture, cells) {
  const c = clsArgmax(cls, blk);
  const tb = blk.map((i) => texts[i]);
  if ((c === HEADER || c === FOOTER) && !keepFurniture) return;
  if (tb.every((t) => WHOLE_MATH.test(pyStrip(t)))) {
    parts.push(tb.map((t) => "$$" + pyStrip(t).slice(1, -1) + "$$").join("\n"));
    return;
  }
  if (c === SECTION) parts.push("## " + joinLines(tb));
  else if (c === TABLE && blk.length > 1) {
    const cb = [], ct = [];
    for (const i of blk) for (const [bx, tx] of (cells && cells.get(i)) || [[boxes[i], texts[i]]]) { cb.push(bx); ct.push(tx); }
    parts.push(tableHtml(cb, ct));
  } else if (c === LIST) {
    const items = [];
    for (const t of tb) { if (!items.length || ITEM.test(t)) items.push([t]); else items.at(-1).push(t); }
    parts.push(items.map(joinLines).join("\n"));
  } else parts.push(joinLines(tb));
}
