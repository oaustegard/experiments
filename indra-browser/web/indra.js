// Indra OCR in the browser: indra-ocr 1.1.2's OCR path (pipeline.py, det_post.py,
// rec_preproc.py, line_mask.py), ported to JS around the WebGPU networks in nets.js.
//
//   const ocr = await Indra.load("models/s4c/");
//   const page = await ocr.read({ w, h, data });   // 8-bit grayscale page
//   page.markdown, page.lines [{box, text, conf}], page.blocks
//
// The recognizers for other scripts (CJK, right-to-left, world) and their n-gram tables are
// fetched the first time the script classifier sends a line to them.

import { Gpu } from "./gpu.js";
import { components, img, pyRound, resize, rot90, rotationMatrix, warpAffine } from "./imgproc.js";
import { ConvNet, OrderNet, Recognizer } from "./nets.js";
import { loadModel } from "./pack.js";
import { BLOCK_CLASSES, GRID_LABELS, decodeBlocksOrder, lineFeatures, median, orderXycut, pageBlocks, pageMarkdown, xycutBlocks } from "./layout.js";
import { Vocab, flip, hasRtl, pySplit, pyStrip } from "./text.js";
import { isAlpha, isRtl, setExceptions } from "./uprops.js";
import { lmFactory } from "./lm.js";

const ADAPTIVE_MIN_H = 9.0, ADAPTIVE_TARGET_H = 12.0, ADAPTIVE_MAX_SIDE = 2048;
const SPLIT_GAP = 3.0, CELL_GAP = 1.8, GUTTER_GAP = 1.6, GUTTER_EDGES = 3;
const ORIENT_SIDE = 768, ORIENT_MIN = 0.95, VERTICAL = 4;
const MIN_CONF = 0.5, LM_GATE = 0.9, LM_ALPHA = 0.5, LM_BETA = 1.5, LM_BEAM = 8, LM_CAND_FLOOR = 3e-3;
const ROUTE_SURE = 0.95;
const DESKEW_MAX = 45.0, DESKEW_MIN = 0.3, PREVIEW_SIDE = 1600;
const PAD_X = 0.3, PAD_Y = 0.2, WIDTH_MULTIPLE = 8;
const CORE_X_SHRINK = 0.15, OUT_STRIDE = 2;
const GROUPS = ["base", "cjk", "rtl", "other"];

const now = () => performance.now();

// ------------------------------------------------------------------ page geometry

function shrinkTo(gray, side, interp = "area") {
  const s = side / Math.max(gray.h, gray.w);
  return resize(gray, Math.max(1, pyRound(gray.w * s)), Math.max(1, pyRound(gray.h * s)), interp);
}

export function preview(gray, side = PREVIEW_SIDE) {
  const s = side / Math.max(gray.h, gray.w);
  return s >= 1 ? gray : shrinkTo(gray, side);
}

function inkThumb(gray, side) {
  const t = shrinkTo(gray, side);
  for (let i = 0; i < t.data.length; i++) t.data[i] = 255 - t.data[i];
  return t;
}

function rowSharpness(ink, a) {
  const r = warpAffine(ink, rotationMatrix(ink.w / 2, ink.h / 2, a, 1), ink.w, ink.h, 0);
  let prev = null, s = 0;
  for (let y = 0; y < r.h; y++) {
    let p = 0;
    for (let x = 0; x < r.w; x++) p += r.data[y * r.w + x];
    if (prev !== null) s += (p - prev) ** 2;
    prev = p;
  }
  return s;
}

const arange = (a, b, st) => { const out = []; const n = Math.ceil((b - a) / st); for (let i = 0; i < n; i++) out.push(a + i * st); return out; };

export function estimateSkew(gray, side = 800) {
  const fine = inkThumb(gray, side);
  let m = 0;
  for (const v of fine.data) m += v;
  if (m / fine.data.length < 0.003 * 255) return 0.0;
  const coarse = inkThumb(gray, Math.floor(side / 2));
  const grid = arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 2.0);
  const scores = grid.map((a) => rowSharpness(coarse, a));
  const top = scores.map((s, i) => [s, i]).sort((x, y) => x[0] - y[0] || x[1] - y[1]).slice(-2).map((p) => grid[p[1]]);
  const starts = [...new Set([0.0, ...top])];
  const near = starts.flatMap((a0) => arange(a0 - 2.0, a0 + 2.0 + 1e-6, 0.5));
  const seen = new Map();
  for (const a of near) if (!seen.has(a)) seen.set(a, rowSharpness(fine, a));
  let a1 = near[0];
  for (const a of near) if (seen.get(a) > seen.get(a1)) a1 = a;
  const last = arange(a1 - 0.5, a1 + 0.5 + 1e-6, 0.1);
  let best = last[0], bv = -Infinity;
  for (const a of last) { const v = rowSharpness(fine, a); if (v > bv) { bv = v; best = a; } }
  return best;
}

export function rotatePage(gray, deg) {
  return warpAffine(gray, rotationMatrix(gray.w / 2, gray.h / 2, deg, 1), gray.w, gray.h, "replicate");
}

function unrotateBox(box, deg, w, h) {
  const M = rotationMatrix(w / 2, h / 2, -deg, 1);
  const [x0, y0, x1, y1] = box;
  const xs = [], ys = [];
  for (const [x, y] of [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) { xs.push(M[0] * x + M[1] * y + M[2]); ys.push(M[3] * x + M[4] * y + M[5]); }
  const cl = (v, hi) => Math.min(Math.max(v, 0), hi);
  return [cl(Math.min(...xs), w), cl(Math.min(...ys), h), cl(Math.max(...xs), w), cl(Math.max(...ys), h)].map((v) => pyRound(v * 10) / 10);
}

function orientInput(gray, side = ORIENT_SIDE) {
  const small = shrinkTo(gray, side);
  const out = new Float32Array(side * side);
  for (let y = 0; y < small.h; y++) for (let x = 0; x < small.w; x++) out[y * side + x] = Math.fround(1 - Math.fround(small.data[y * small.w + x] / 255));
  return out;
}

// ------------------------------------------------------------------ line crops

function percentiles(gray, x0, y0, x1, y1, qs = [0.02, 0.5, 0.98]) {
  const hist = new Float64Array(256);
  let n = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { hist[gray.data[y * gray.w + x]]++; n++; }
  const cum = new Float64Array(256);
  let c = 0;
  for (let i = 0; i < 256; i++) { c += hist[i]; cum[i] = c; }
  const ss = (v) => { let lo = 0, hi = 256; while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] <= v) lo = m + 1; else hi = m; } return lo; };
  return qs.map((q) => {
    const x = q * (n - 1), i = Math.floor(x), g = x - i;
    const lo = ss(i), hi = ss(Math.min(i + 1, n - 1));
    const d = hi - lo;
    return g >= 0.5 ? hi - d * (1 - g) : lo + d * g;
  });
}

function isDarkLine(gray, box) {
  const [x0, y0, x1, y1] = box;
  const ya = Math.max(0, Math.trunc(y0)), yb = Math.min(gray.h, Math.trunc(y1) + 1);
  const xa = Math.max(0, Math.trunc(x0)), xb = Math.min(gray.w, Math.trunc(x1) + 1);
  if (yb <= ya || xb <= xa) return false;
  const [lo, med, hi] = percentiles(gray, xa, ya, xb, yb);
  return hi - med > 2 * (med - lo) + 8;
}

// rec_preproc.crop_line -> {data: Float32Array [H, W] ink, w, h}
function cropLine(gray, box, height, window, dark) {
  const padY = (window * (1 + 2 * PAD_Y) - 1) / 2;
  height *= window;
  const [x0, y0, x1, y1] = box;
  const h = Math.max(1.0, y1 - y0);
  const ix0 = Math.floor(x0 - PAD_X * h), iy0 = Math.floor(y0 - padY * h);
  const ix1 = Math.ceil(x1 + PAD_X * h), iy1 = Math.ceil(y1 + padY * h);
  const cw = Math.max(1, ix1 - ix0), ch = Math.max(1, iy1 - iy0);
  const sx0 = Math.max(0, ix0), sy0 = Math.max(0, iy0), sx1 = Math.min(gray.w, ix1), sy1 = Math.min(gray.h, iy1);
  const crop = img(cw, ch, new Uint8Array(cw * ch).fill(dark ? 0 : 255));
  for (let y = sy0; y < sy1; y++) for (let x = sx0; x < sx1; x++) crop.data[(y - iy0) * cw + (x - ix0)] = gray.data[y * gray.w + x];
  const outW = Math.max(WIDTH_MULTIPLE, Math.min(4096, pyRound((cw * height) / ch)));
  const r = resize(crop, outW, height, ch > height ? "area" : "linear");
  const ink = new Float32Array(outW * height);
  for (let i = 0; i < ink.length; i++) {
    const v = Math.fround(r.data[i] / 255);
    ink[i] = dark ? v : Math.fround(1 - v);
  }
  return { data: ink, w: outW, h: height };
}

function padBatch(lines) {
  const H = lines[0].h;
  const Wm = Math.ceil(Math.max(...lines.map((l) => l.w)) / WIDTH_MULTIPLE) * WIDTH_MULTIPLE;
  const out = new Float32Array(lines.length * H * Wm);
  lines.forEach((l, b) => { for (let y = 0; y < H; y++) out.set(l.data.subarray(y * l.w, (y + 1) * l.w), (b * H + y) * Wm); });
  return { data: out, B: lines.length, H, W: Wm, widths: lines.map((l) => l.w) };
}

function widthBatches(widths, maxBatch = 64, maxRatio = 1.3, minWidth = 64, maxArea = 64 * 640) {
  const order = widths.map((w, i) => [w, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map((p) => p[1]);
  const batches = [];
  for (const i of order) {
    const last = batches.at(-1);
    if (last && last.length < maxBatch && Math.max(widths[i], minWidth) <= maxRatio * Math.max(widths[last[0]], minWidth) &&
        (last.length + 1) * Math.max(widths[i], minWidth) <= maxArea) last.push(i);
    else batches.push([i]);
  }
  return batches;
}

// ------------------------------------------------------------------ detection decode

function decodeDet(det, thresh = 0.5, minArea = 2) {
  const { data, h, w } = det; // [h, w, 3]: core logit, top, bottom
  const mask = new Uint8Array(h * w);
  const prob = new Float32Array(h * w);
  for (let i = 0; i < h * w; i++) { prob[i] = 1 / (1 + Math.exp(-data[i * 3])); mask[i] = prob[i] > thresh ? 1 : 0; }
  const { n, lab, stats } = components(mask, w, h, 4);
  if (n <= 1) return [];
  const cnt = new Float64Array(n), sy0 = new Float64Array(n), sy1 = new Float64Array(n), ss = new Float64Array(n);
  for (let i = 0; i < h * w; i++) {
    const l = lab[i];
    if (l <= 0) continue;
    const ys = Math.fround((Math.floor(i / w) + 0.5) * OUT_STRIDE);
    cnt[l]++;
    sy0[l] += Math.fround(ys - data[i * 3 + 1]);
    sy1[l] += Math.fround(ys + data[i * 3 + 2]);
    ss[l] += prob[i];
  }
  const boxes = [];
  for (let i = 1; i < n; i++) {
    if (stats[i][4] < minArea) continue;
    const y0 = sy0[i] / Math.max(cnt[i], 1), y1 = sy1[i] / Math.max(cnt[i], 1);
    const lh = Math.max(1.0, y1 - y0);
    const ex = Math.min(CORE_X_SHRINK * lh, 0.5 * lh);
    boxes.push([stats[i][0] * OUT_STRIDE - ex, y0, (stats[i][0] + stats[i][2]) * OUT_STRIDE + ex, y1, ss[i] / Math.max(cnt[i], 1)].map(Math.fround));
  }
  return boxes;
}

// ------------------------------------------------------------------ the reader

export class Indra {
  static async load(base, { fetchBytes, log = () => {}, gpu = null } = {}) {
    const url = new URL(base, location.href);
    const fb = fetchBytes ?? (async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.arrayBuffer(); });
    const json = async (f) => JSON.parse(new TextDecoder().decode(await fb(new URL(f, url).href)));
    const t0 = now();
    const man = await json("manifest.json");
    const o = new Indra();
    o.base = url;
    o.man = man;
    o.fetchBytes = fb;
    o.log = log;
    o.gpu = gpu ?? (await Gpu.create());
    setExceptions(await json("uexc.json"));
    const core = ["det", "rec", "order", "orient", "script"].filter((m) => man.models[m]);
    const models = await Promise.all(core.map((m) => loadModel(url, man, m, fb)));
    const by = Object.fromEntries(core.map((m, i) => [m, models[i]]));
    o.bytes = models.reduce((s, m) => s + m.bytes, 0);
    o.det = new ConvNet(o.gpu, by.det);
    o.rec = new Recognizer(o.gpu, by.rec);
    o.order = by.order ? new OrderNet(by.order) : null;
    o.orient = by.orient ? new ConvNet(o.gpu, by.orient) : null;
    o.script = by.script ? new ConvNet(o.gpu, by.script) : null;
    o.vocab = new Vocab(await json("vocab.json"));
    o.scriptGroups = man.script_groups ?? GROUPS;
    o.groupNames = Object.keys(man.models).filter((m) => m.startsWith("rec_")).map((m) => m.slice(4));
    o.groups = new Map(); // name -> Promise<{rec, vocab, lm}>
    o.lms = new Map(); // "base" | group -> {lm, alpha, beta, gate}
    o.lmFactory = man.lm ? lmFactory() : null; // n-gram tables, fetched on first use
    o.height = man.rec_height;
    o.window = man.rec_window ?? 1;
    o.longSide = man.det_long_side;
    o.loadMs = now() - t0;
    log(`models loaded: ${core.join(", ")} (${(o.bytes / 1e6).toFixed(2)} MB decoded weights file) in ${o.loadMs.toFixed(0)} ms`);
    return o;
  }

  async group(name) {
    if (!this.groups.has(name)) {
      this.groups.set(name, (async () => {
        const t0 = now();
        const m = await loadModel(this.base, this.man, `rec_${name}`, this.fetchBytes);
        const chars = JSON.parse(new TextDecoder().decode(await this.fetchBytes(new URL(`vocab_${name}.json`, this.base).href)));
        this.log(`loaded the ${name} recognizer (${(m.bytes / 1e6).toFixed(2)} MB) in ${(now() - t0).toFixed(0)} ms`);
        return { rec: new Recognizer(this.gpu, m), vocab: new Vocab(chars), name };
      })());
    }
    return this.groups.get(name);
  }

  // ---------------------------------------------------------------- one page

  async read(gray, { turn = true } = {}) {
    const t = {};
    let t0 = now();
    const small = preview(gray);
    const a = estimateSkew(small);
    let k = 0;
    if (turn && this.orient) {
      const p = await this.orientationProbs(Math.abs(a) >= DESKEW_MIN ? rotatePage(small, a) : small);
      k = p.indexOf(Math.max(...p));
      if (!(p[k] > ORIENT_MIN)) k = 0;
    }
    t.orient_ms = now() - t0;
    t0 = now();
    const [g2, k2, skew] = this.turnPage(gray, k, a, small);
    t.orient_ms += now() - t0;
    t0 = now();
    const boxes = await this.detect(g2);
    t.det_ms = now() - t0;
    t0 = now();
    const vertical = k2 === VERTICAL;
    const rec = await this.recognize(g2, boxes, vertical);
    t.rec_ms = now() - t0;
    const page = this.assemble([g2.w, g2.h], boxes, rec, t, vertical);
    page.rotation = vertical ? 0 : 90 * k2;
    page.skew = pyRound(skew * 100) / 100;
    page.vertical = vertical;
    if (skew) {
      for (const ln of page.lines) ln.box = unrotateBox(ln.box, skew, g2.w, g2.h);
      for (const b of page.blocks) b.bbox = unrotateBox(b.bbox, skew, g2.w, g2.h);
    }
    if (vertical) {
      const W = g2.h;
      const back = (b) => [W - b[3], b[0], W - b[1], b[2]];
      for (const ln of page.lines) ln.box = back(ln.box);
      for (const b of page.blocks) b.bbox = back(b.bbox);
    }
    page.size = [g2.w, g2.h];
    return page;
  }

  async orientationProbs(gray) {
    const x = orientInput(gray);
    return (await this.orient.classify(x, 1, ORIENT_SIDE, ORIENT_SIDE))[0];
  }

  turnPage(gray, k, a, small) {
    const q = k === VERTICAL ? 1 : -k;
    if (k === VERTICAL) gray = rot90(gray, 1);
    else if (k) gray = rot90(gray, -k);
    if (k === 1 || k === 3 || k === VERTICAL) a = estimateSkew(small ? rot90(small, q) : gray);
    if (Math.abs(a) >= DESKEW_MIN) gray = rotatePage(gray, a);
    else a = 0.0;
    return [gray, k, a];
  }

  async detectAt(gray, longSide) {
    const scale = longSide / Math.max(gray.h, gray.w);
    const nh = Math.max(1, pyRound(gray.h * scale)), nw = Math.max(1, pyRound(gray.w * scale));
    const g = nh !== gray.h || nw !== gray.w ? resize(gray, nw, nh, scale < 1 ? "area" : "linear") : gray;
    const ph = Math.ceil(nh / 32) * 32, pw = Math.ceil(nw / 32) * 32;
    const x = new Float32Array(ph * pw);
    for (let y = 0; y < nh; y++) for (let xx = 0; xx < nw; xx++) x[y * pw + xx] = Math.fround(1 - Math.fround(g.data[y * nw + xx] / 255));
    const out = await this.det.det(x, ph, pw);
    return [decodeDet(out), scale];
  }

  async detect(gray) {
    let [boxes, s] = await this.detectAt(gray, this.longSide);
    if (boxes.length >= 8) {
      const medH = median(boxes.map((b) => b[3] - b[1]));
      if (medH < ADAPTIVE_MIN_H) {
        const ls = Math.floor(Math.trunc(Math.min(ADAPTIVE_MAX_SIDE, (this.longSide * ADAPTIVE_TARGET_H) / Math.max(medH, 1.0))) / 32) * 32;
        if (ls > this.longSide) [boxes, s] = await this.detectAt(gray, ls);
      }
    }
    return boxes.map((b) => {
      const c = b.slice();
      for (let j = 0; j < 4; j++) c[j] = Math.fround(c[j] / s);
      c[0] = Math.min(Math.max(c[0], 0), gray.w); c[2] = Math.min(Math.max(c[2], 0), gray.w);
      c[1] = Math.min(Math.max(c[1], 0), gray.h); c[3] = Math.min(Math.max(c[3], 0), gray.h);
      return c;
    });
  }

  async lineGroups(crops) {
    const names = crops.map(() => "base"), sure = crops.map(() => true);
    for (const idx of widthBatches(crops.map((c) => c.w))) {
      const b = padBatch(idx.map((j) => crops[j]));
      const p = await this.script.classify(b.data, b.B, b.H, b.W, b.widths);
      idx.forEach((j, k) => {
        const pk = p[k];
        let am = 0;
        for (let c = 1; c < pk.length; c++) if (pk[c] > pk[am]) am = c;
        const g = this.scriptGroups[am];
        names[j] = this.groupNames.includes(g) ? g : "base";
        sure[j] = pk[am] > ROUTE_SURE;
      });
    }
    return [names, sure];
  }

  async recognize(gray, boxes, vertical = false) {
    if (!boxes.length) return [];
    const dark = this.window > 1 ? boxes.map((b) => isDarkLine(gray, b)) : boxes.map(() => null);
    const darkOf = (j) => (dark[j] === null ? isDarkLine(gray, boxes[j]) : dark[j]);
    const crops = boxes.map((b, j) => cropLine(gray, b, this.height, 1, darkOf(j)));
    const tall = this.window > 1 ? boxes.map((b, j) => cropLine(gray, b, this.height, this.window, dark[j])) : crops;
    const out = crops.map(() => ["", 0.0, []]);
    let plan = [{ name: "base", sel: crops.map((_, j) => j), asChars: false }];
    if (vertical && this.groupNames.includes("cjk")) plan = [{ name: "cjk", sel: crops.map((_, j) => j), asChars: true }];
    else if (this.script) {
      const [names, sure] = await this.lineGroups(crops);
      plan = [{ name: "base", sel: names.map((g, j) => (g === "base" || !sure[j] ? j : -1)).filter((j) => j >= 0), asChars: false }];
      for (const g of this.groupNames) plan.push({ name: g, sel: names.map((n, j) => (n === g ? j : -1)).filter((j) => j >= 0), asChars: true });
    }
    const bestConf = crops.map(() => -1.0);
    for (const { name, sel, asChars } of plan) {
      if (!sel.length) continue;
      const grp = name === "base" ? { rec: this.rec, vocab: this.vocab } : await this.group(name);
      const lm = await this.lmFor(name);
      for (const part of widthBatches(sel.map((j) => crops[j].w))) {
        const idx = part.map((i) => sel[i]);
        const src = name === "base" ? tall : crops;
        const b = padBatch(idx.map((j) => src[j]));
        const r = await grp.rec.run(b.data, b.B, b.H, b.W, b.widths);
        const T = r.T, stride = b.W / Math.max(T, 1);
        idx.forEach((j, k) => {
          const n = r.lengths[k];
          const seq = [], pm = [];
          for (let t = 0; t < n; t++) { const o = (k * T + t) * 19; seq.push(r.out[o]); pm.push(r.out[o + 1]); }
          let t = [], ids = [], cs = 0;
          for (let i = 0; i < n; i++) if (seq[i] !== 0 && (i === 0 || seq[i] !== seq[i - 1])) { t.push(i); ids.push(seq[i]); cs += pm[i]; }
          // numpy mean of float32: pairwise summation in float32; a float64 sum differs in the last bits only
          const conf = t.length ? Math.fround(cs / t.length) : 0.0;
          const [x0, , x1] = boxes[j];
          const y0 = boxes[j][1], y1 = boxes[j][3];
          const h = Math.max(1.0, y1 - y0);
          const ix0 = Math.floor(x0 - PAD_X * h);
          const cw = Math.ceil(x1 + PAD_X * h) - ix0;
          const f = cw / crops[j].w;
          if (lm && MIN_CONF <= conf && conf < lm.gate) {
            const sl = (o, m) => { const a = new Float32Array(n * m); for (let s = 0; s < n; s++) for (let q = 0; q < m; q++) a[s * m + q] = r.out[(k * T + s) * 19 + o + q]; return a; };
            [ids, t] = lm.search(sl(2, 1), sl(3, 8), sl(11, 8), n);
          }
          const xs = t.map((s) => ix0 + (s + 0.5) * stride * f);
          const toks = asChars ? ids.map((id, q) => [grp.vocab.chars[id - 1], xs[q]]) : ids.map((id, q) => [id, xs[q]]);
          if (conf <= bestConf[j]) return;
          bestConf[j] = conf;
          out[j] = conf >= MIN_CONF ? [grp.vocab.decode(ids), conf, toks] : ["", conf, []];
        });
      }
    }
    return out;
  }

  async lmFor(name) {
    if (!this.lmFactory || this.noLm) return null;
    return this.lmFactory(this, name);
  }

  // ---------------------------------------------------------------- lines -> page

  isSpace(tok) { return tok === " " || tok === this.vocab.index.get(" "); }

  decodeToks(toks) {
    if (toks.length && typeof toks[0][0] === "string") {
      let text = pySplit(toks.map((tk) => tk[0]).join("")).join(" ");
      if (toks[0].length === 2) {
        text = text.normalize("NFC");
        if (hasRtl(text)) text = flip(text, true, this.pageRtl || null);
      }
      return text;
    }
    return this.vocab.decode(toks.map((tk) => tk[0]));
  }

  splitLine(box, toks, gap, at = null) {
    const h = Math.max(1.0, box[3] - box[1]);
    const segs = [[]];
    let prev = null;
    const cuts = (at ?? []).slice().sort((a, b) => a - b);
    for (const tk of toks) {
      const x = tk[1];
      if (!this.isSpace(tk[0])) {
        if (prev !== null && (x - prev > gap * h || cuts.some((c) => prev < c && c < x))) segs.push([]);
        prev = x;
      }
      segs.at(-1).push(tk);
    }
    if (segs.length === 1) return [[box.slice(0, 4), this.decodeToks(toks), toks]];
    const out = [];
    for (const sg of segs) {
      const text = pyStrip(this.decodeToks(sg));
      const xs = sg.filter((tk) => !this.isSpace(tk[0])).map((tk) => tk[1]);
      if (text && xs.length) out.push([[Math.max(box[0], Math.min(...xs) - 0.35 * h), box[1], Math.min(box[2], Math.max(...xs) + 0.35 * h), box[3]], text, sg]);
    }
    return out;
  }

  gutterCuts(boxes, rec) {
    return boxes.map((b, i) => {
      const h = Math.max(1.0, b[3] - b[1]);
      const xs = rec[i][2].filter((tk) => !this.isSpace(tk[0])).map((tk) => tk[1]);
      const mine = [];
      for (let q = 0; q + 1 < xs.length; q++) {
        const a = xs[q], c = xs[q + 1];
        if (c - a <= GUTTER_GAP * h) continue;
        const gl = a + 0.35 * h, gr = c - 0.35 * h;
        let near = 0;
        boxes.forEach((o, j) => { if (j !== i && (Math.abs(o[2] - gl) < 0.5 * h || Math.abs(o[0] - gr) < 0.5 * h)) near++; });
        if (near >= GUTTER_EDGES) mine.push((a + c) / 2);
      }
      return mine;
    });
  }

  readingOrder(boxes, texts, wh) {
    if (boxes.length < 2 || !this.order) return [boxes.length ? orderXycut(boxes) : [], null];
    const f = lineFeatures(boxes, wh[0], wh[1], texts);
    const n = boxes.length;
    const o = this.order.run(f, n);
    return [decodeBlocksOrder(o.a, o.same, boxes, o.cls, n), { cls: o.cls, same: o.same }];
  }

  assemble(wh, boxes, rec, t, vertical = false) {
    let t0 = now();
    const letters = rec.flatMap((r) => Array.from(r[0]).filter(isAlpha));
    const nRtl = letters.filter(isRtl).length;
    this.pageRtl = nRtl > letters.length - nRtl;
    const width = wh[0];
    const geo = (b) => (this.pageRtl ? [width - b[2], b[1], width - b[0], b[3]] : b);
    let nb = [], nrec = [];
    const cuts = boxes.length ? this.gutterCuts(boxes, rec) : [];
    boxes.forEach((b, i) => {
      for (const [bx, tx, tk] of this.splitLine(b, rec[i][2], SPLIT_GAP, cuts[i])) { nb.push(bx.map(Math.fround)); nrec.push([tx, rec[i][1], tk]); }
    });
    [nb, nrec] = tidyLines(nb, nrec, this.pageRtl, this.vocab.index.get(" "));
    boxes = nb;
    rec = nrec;
    const texts = rec.map((r) => r[0]);
    const gb = boxes.map(geo);
    let [order, layout] = this.readingOrder(gb, texts, wh);
    if (vertical && layout) {
      const blocks = xycutBlocks(boxes);
      order = blocks.flat();
      const n = boxes.length;
      const same = new Float32Array(n * n).fill(-1);
      for (const b of blocks) for (const i of b) for (const j of b) same[i * n + j] = 1;
      const cls = Float32Array.from(layout.cls);
      for (const c of GRID_LABELS) { const k = BLOCK_CLASSES.indexOf(c); for (let i = 0; i < n; i++) cls[i * 10 + k] = -1e4; }
      layout = { cls, same };
    }
    t.order_ms = now() - t0;
    t.total_ms = Object.entries(t).filter(([k]) => k !== "total_ms").reduce((s, [, v]) => s + v, 0);
    const lines = order.map((i) => ({ box: boxes[i].slice(0, 4).map((v) => pyRound(v * 10) / 10), text: rec[i][0], conf: pyRound(rec[i][1] * 1e4) / 1e4 }));
    let md = null, blocks = [];
    if (layout) {
      const n = boxes.length;
      blocks = pageBlocks(gb, texts, order, layout.cls, layout.same, n);
      for (const b of blocks) b.bbox = geo(b.bbox).map((v) => pyRound(v * 10) / 10);
      const cells = new Map(boxes.map((b, i) => [i, this.splitLine(b, rec[i][2], CELL_GAP).map(([bx, tx]) => [geo(bx), tx])]));
      md = pageMarkdown(gb, texts, order, layout.cls, layout.same, n, { keepFurniture: this.furniture ?? true, cells });
    } else if (boxes.length < 2) md = texts.join("\n");
    return { lines, markdown: md, blocks, timings: t, note: vertical ? "order=xycut" : "order=model", text: lines.map((l) => l.text).join("\n") };
  }
}

// ------------------------------------------------------------------ pipeline.tidy_lines

const LEADER = new Set(Array.from(".·…‥_-–—=")), DASHES = new Set(Array.from("-–—")), BULLETS = new Set(Array.from("•◦▪▫●○■□‣⁃∙*"));
const ENUMERATOR = /^(?:(\d{1,3}|[A-Za-z])[.)]|\((\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\))$/u;
const MARKER_GAP = 2.5;

function tidyLines(boxes, rec, rtl = false, space = undefined) {
  const keep = rec.map(() => true);
  rec = rec.slice();
  boxes = boxes.map((b) => b.slice());
  const h = boxes.map((b) => b[3] - b[1]), w = boxes.map((b) => b[2] - b[0]);
  rec.forEach(([text], i) => {
    const core = Array.from(text.replace(/ /g, ""));
    if (core.length && core.every((c) => LEADER.has(c))) {
      const wide = w[i] >= 3 * h[i];
      if (core.length >= 3 || wide || !core.some((c) => DASHES.has(c))) keep[i] = false;
    }
  });
  rec.forEach(([text, , toks], i) => {
    const t = pyStrip(text);
    if (!keep[i] || !(BULLETS.has(t) || ENUMERATOR.test(t))) return;
    let best = null, bestGap = null;
    for (let j = 0; j < rec.length; j++) {
      if (j === i || !keep[j] || !pyStrip(rec[j][0])) continue;
      const ov = Math.min(boxes[i][3], boxes[j][3]) - Math.max(boxes[i][1], boxes[j][1]);
      if (ov < 0.5 * Math.min(h[i], h[j])) continue;
      const gap = rtl ? boxes[i][0] - boxes[j][2] : boxes[j][0] - boxes[i][2];
      if (-0.3 * h[i] <= gap && gap <= MARKER_GAP * Math.max(h[i], h[j]) && (bestGap === null || gap < bestGap)) { best = j; bestGap = gap; }
    }
    if (best === null) return;
    let [tj, cj, kj] = rec[best];
    const chars = kj.length > 0 && typeof kj[0][0] === "string";
    const sp = chars ? " " : space;
    if (!rtl && toks.length && kj.length && sp !== undefined && sp !== null) {
      if ((typeof toks[0][0] === "string") === chars) {
        const x = (boxes[i][2] + boxes[best][0]) / 2;
        kj = [...toks, [sp, x, ...toks.at(-1).slice(2)], ...kj];
      }
    }
    rec[best] = [`${t} ${pyStrip(tj)}`, cj, kj];
    boxes[best] = [Math.min(boxes[i][0], boxes[best][0]), Math.min(boxes[i][1], boxes[best][1]), Math.max(boxes[i][2], boxes[best][2]), Math.max(boxes[i][3], boxes[best][3])];
    keep[i] = false;
  });
  const idx = keep.map((k, i) => (k ? i : -1)).filter((i) => i >= 0);
  return [idx.map((i) => boxes[i]), idx.map((i) => rec[i])];
}
