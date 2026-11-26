// OCRia-nano-1 page OCR in JavaScript: a port of LiteParse's ocria-nano page logic
// (liteparse 2.21.2, src/ocr/ocria/{mod,page,det,img,rec}.rs, Apache-2.0) with the nets left out.
//
// The detector, script router and four recognizers are called through an injected `infer`, with the
// contract of LiteParse's `SessionPool::infer`:
//
//   const out = await infer(model, shape, data, outputNames)
//     "det":    [1,3,H,W] Float32Array          -> { prob: {shape:[1,1,h,w], data: Float32Array} }
//     "router": [N,1,32,512]                    -> { group: {shape:[N,4], data}, flip: {shape:[N,2], data} }
//     "rec_latin" | "rec_world" | "rec_rtl" | "rec_cjk": [N,1,32,W]
//                                               -> { ids: {shape:[N,T], data: Int32Array|BigInt64Array},
//                                                    conf: {shape:[N,T], data: Float32Array} }
//
//   import { ocrPage } from "./ocr.js";
//   const results = await ocrPage({ data, w, h, ch }, infer, { latin, cjk, rtl, world }, opts);
//   // [{ text, bbox: [x1,y1,x2,y2], confidence, polygon: [[x,y]x4], words: [], rotation: null, faint }]
//
// `charsets` holds the parsed rec_<group>.charset.json arrays (or {chars, visual_rtl}); a missing
// group is read by the Latin recognizer, as LiteParse's WebAssembly build does. Every default is
// the benchmarked OcriaOptions; the detector threshold defaults to the 0.55 that ships in
// det.meta.json (pass opts.detMeta to use another det.meta.json).
//
// Arithmetic follows the Rust source type for type: f32 steps go through Math.fround, image ops
// keep OpenCV's fixed-point weights and rounding, so crops match LiteParse's pixel for pixel.
//
// Translated from LiteParse (Copyright LlamaIndex, Inc., Apache License 2.0) into JavaScript. Changed:
// the ONNX session pool became the injected `infer`, model files are passed in by the caller, and
// unicode-bidi is replaced by the bidi routine at the end of this file. See README.md.

const f32 = Math.fround;

/** Pipeline switches and thresholds (mod.rs `OcriaOptions::default`). */
export const DEFAULT_OPTIONS = Object.freeze({
  detLongSide: 1600,
  detThresh: null, // null: detMeta.thresh, else SHIPPED_DET_THRESH
  detMerge: null, // null: detMeta.merge, else DET_MERGE
  detMeta: null, // parsed det.meta.json
  detAdaptive: null, // {minH, targetH, maxSide} for a second pass on small print (off)
  gutter: true,
  orient: true,
  rescue: true,
  junk: true,
  faint: { thresh: 0.1, max: 24 },
});

/** det.meta.json of OCRia-nano-1 ({"thresh": 0.55}); LiteParse's own fallback without it is 0.5. */
const SHIPPED_DET_THRESH = 0.55;
const DET_MERGE = 0.6;
const FAINT_CONF = f32(0.9);
export const GROUPS = ["latin", "cjk", "rtl", "world"];

// ---------------------------------------------------------------------------------------------
// numeric helpers

/** f64 round half to even (`round_ties_even`, cvRound). */
function rint(x) {
  const r = Math.round(x);
  if (r - x === 0.5 && r % 2 !== 0) return r - 1;
  return r;
}
/** Rust `as i64` / `as usize` from a float: truncation, NaN -> 0. */
function trunc(x) {
  return Number.isNaN(x) ? 0 : Math.trunc(x);
}
function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}
/** f32 `clamp`: NaN passes through. */
function clampF(v, lo, hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
/** f32 `min` / `max` (NaN-ignoring). */
function fminF(a, b) {
  if (Number.isNaN(a)) return b;
  if (Number.isNaN(b)) return a;
  return a < b ? a : b;
}
function fmaxF(a, b) {
  if (Number.isNaN(a)) return b;
  if (Number.isNaN(b)) return a;
  return a > b ? a : b;
}
const RAD_PER_DEG = Math.PI / 180.0;
const DEG_PER_RAD = 180.0 / Math.PI;

// ---------------------------------------------------------------------------------------------
// img.rs

function imgNew(w, h, ch) {
  return { w, h, ch, data: new Uint8Array(w * h * ch) };
}

/** `cv2.cvtColor(.., COLOR_RGB2GRAY)`: 14-bit fixed-point BT.601 weights. */
function toGray(img) {
  if (img.ch === 1) return img;
  const out = imgNew(img.w, img.h, 1);
  const d = img.data, o = out.data, ch = img.ch;
  for (let i = 0, j = 0; i < o.length; i++, j += ch) {
    o[i] = (d[j] * 4899 + d[j + 1] * 9617 + d[j + 2] * 1868 + 8192) >>> 14;
  }
  return out;
}

/** `img.mean(axis=2).astype(np.uint8)`: the channel mean, truncated. */
function meanGray(img) {
  if (img.ch === 1) return img;
  const out = imgNew(img.w, img.h, 1);
  const d = img.data, o = out.data, ch = img.ch;
  for (let i = 0, j = 0; i < o.length; i++, j += ch) {
    let s = 0;
    for (let c = 0; c < ch; c++) s += d[j + c];
    o[i] = Math.floor(s / ch);
  }
  return out;
}

/** `np.rot90(img, k)`: k quarter turns counter-clockwise. */
function rot90(img, k) {
  k %= 4;
  const { w, h, ch, data } = img;
  if (k === 0) return { w, h, ch, data: data.slice() };
  if (k === 1) {
    const out = imgNew(h, w, ch);
    const od = out.data, ow = out.w;
    for (let i = 0; i < out.h; i++) {
      const sx = w - 1 - i;
      for (let j = 0; j < ow; j++) {
        const s = (j * w + sx) * ch, o = (i * ow + j) * ch;
        for (let c = 0; c < ch; c++) od[o + c] = data[s + c];
      }
    }
    return out;
  }
  if (k === 2) {
    const out = imgNew(w, h, ch);
    const od = out.data;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = ((h - 1 - y) * w + (w - 1 - x)) * ch, o = (y * w + x) * ch;
        for (let c = 0; c < ch; c++) od[o + c] = data[s + c];
      }
    }
    return out;
  }
  return rot90(rot90(img, 1), 2);
}

const RESIZE_BITS = 11;
const RESIZE_ONE = 1 << RESIZE_BITS;

function satShort(v) {
  return clamp(rint(v), -32768, 32767);
}

/** Two-tap source index + fixed-point weights per output index (cv2.resize linear path). */
function linearTaps(src, dst, area) {
  const invScale = dst / src;
  const scale = 1.0 / invScale;
  const sxs = new Int32Array(dst), a0s = new Int32Array(dst), a1s = new Int32Array(dst);
  for (let d = 0; d < dst; d++) {
    let sx, f;
    if (area) {
      sx = Math.floor(d * scale);
      f = f32((d + 1) - (sx + 1) * invScale);
      f = f <= 0 ? 0 : f32(f - Math.floor(f));
    } else {
      const fx = f32((d + 0.5) * scale - 0.5);
      const fl = Math.floor(fx);
      sx = fl;
      f = f32(fx - fl);
    }
    if (sx < 0) {
      sx = 0;
      f = 0;
    }
    if (sx >= src - 1) {
      sx = src - 1;
      f = 0;
    }
    sxs[d] = sx;
    a0s[d] = satShort(f32(f32(1 - f) * RESIZE_ONE));
    a1s[d] = satShort(f32(f * RESIZE_ONE));
  }
  return { sxs, a0s, a1s };
}

function resizeLinearTaps(img, w, h, area) {
  const xt = linearTaps(img.w, w, area);
  const yt = linearTaps(img.h, h, area);
  const ch = img.ch, iw = img.w, d = img.data;
  const rows = new Array(img.h);
  for (let y = 0; y < img.h; y++) {
    const r = new Int32Array(w * ch);
    const base = y * iw * ch;
    for (let dx = 0; dx < w; dx++) {
      const sx = xt.sxs[dx], a0 = xt.a0s[dx], a1 = xt.a1s[dx];
      for (let c = 0; c < ch; c++) {
        const s0 = d[base + sx * ch + c];
        r[dx * ch + c] = sx + 1 < iw ? s0 * a0 + d[base + (sx + 1) * ch + c] * a1 : s0 * RESIZE_ONE;
      }
    }
    rows[y] = r;
  }
  const out = imgNew(w, h, ch);
  const o = out.data, n = w * ch;
  for (let dy = 0; dy < h; dy++) {
    const sy = yt.sxs[dy], b0 = yt.a0s[dy], b1 = yt.a1s[dy];
    const r0 = rows[sy], r1 = rows[Math.min(sy + 1, img.h - 1)];
    const ob = dy * n;
    for (let i = 0; i < n; i++) {
      // FixedPtCast<int, uchar, 22>
      const v = Math.floor((r0[i] * b0 + r1[i] * b1 + 2097152) / 4194304);
      o[ob + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}

/** `cv2.resize(.., interpolation=INTER_LINEAR)`. */
function resizeLinear(img, w, h) {
  if (w === img.w && h === img.h) return img;
  return resizeLinearTaps(img, w, h, false);
}

/** OpenCV `computeResizeAreaTab`: (source index, weight) runs per output index. */
function areaTab(src, dst, scale) {
  const out = new Array(dst);
  for (let d = 0; d < dst; d++) {
    const fsx1 = d * scale;
    const fsx2 = fsx1 + scale;
    const cell = Math.min(scale, src - fsx1);
    let sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, src - 1);
    const sx1 = Math.min(Math.ceil(fsx1), sx2);
    const t = [];
    if (sx1 - fsx1 > 1e-3) t.push([sx1 - 1, f32((sx1 - fsx1) / cell)]);
    for (let sx = sx1; sx < sx2; sx++) t.push([sx, f32(1.0 / cell)]);
    if (fsx2 - sx2 > 1e-3) t.push([sx2, f32(Math.min(Math.min(fsx2 - sx2, 1.0), cell) / cell)]);
    out[d] = t;
  }
  return out;
}

const F64_EPS = 2.220446049250313e-16;

/** `cv2.resize(.., interpolation=INTER_AREA)`. */
function resizeArea(img, w, h) {
  if (w === img.w && h === img.h) return img;
  const scaleX = img.w / w, scaleY = img.h / h;
  if (scaleX < 1.0 || scaleY < 1.0) return resizeLinearTaps(img, w, h, true);
  const ch = img.ch, src = img.data;
  const isx = Math.round(scaleX), isy = Math.round(scaleY);
  if (Math.abs(scaleX - isx) < F64_EPS && Math.abs(scaleY - isy) < F64_EPS) {
    // integer factors: plain block mean (resizeAreaFast)
    const kx = isx, ky = isy, area = kx * ky;
    const inv = f32(1 / area);
    const out = imgNew(w, h, ch);
    const o = out.data;
    for (let dy = 0; dy < h; dy++) {
      for (let dx = 0; dx < w; dx++) {
        for (let c = 0; c < ch; c++) {
          let s = 0;
          for (let yy = 0; yy < ky; yy++) {
            const rb = ((dy * ky + yy) * img.w + dx * kx) * ch + c;
            for (let xx = 0; xx < kx; xx++) s += src[rb + xx * ch];
          }
          const v = area === 4 ? (s + 2) >> 2 : rint(s * inv);
          o[(dy * w + dx) * ch + c] = Math.min(v, 255);
        }
      }
    }
    return out;
  }
  // flat horizontal taps: (output offset, source offset, weight), in OpenCV's order
  const xtab = areaTab(img.w, w, scaleX);
  let nt = 0;
  for (const t of xtab) nt += t.length;
  const xd = new Int32Array(nt), xs = new Int32Array(nt), xa = new Float64Array(nt);
  let k = 0;
  for (let dx = 0; dx < w; dx++) {
    for (const [sx, a] of xtab[dx]) {
      xd[k] = dx * ch;
      xs[k] = sx * ch;
      xa[k] = a;
      k++;
    }
  }
  const yt = areaTab(img.h, h, scaleY);
  const out = imgNew(w, h, ch);
  const o = out.data;
  const rowLen = img.w * ch;
  const buf = new Float32Array(w * ch);
  let bufRow = -1;
  const sum = new Float32Array(w * ch);
  const hsum = (sy) => {
    buf.fill(0);
    const rb = sy * rowLen;
    for (let t = 0; t < nt; t++) {
      const dd = xd[t], ss = rb + xs[t], a = xa[t];
      for (let c = 0; c < ch; c++) buf[dd + c] = buf[dd + c] + f32(src[ss + c] * a);
    }
  };
  for (let dy = 0; dy < h; dy++) {
    sum.fill(0);
    for (const [sy, beta] of yt[dy]) {
      if (bufRow !== sy) {
        hsum(sy);
        bufRow = sy;
      }
      for (let i = 0; i < sum.length; i++) sum[i] = sum[i] + f32(buf[i] * beta);
    }
    const ob = dy * w * ch;
    for (let i = 0; i < sum.length; i++) {
      const v = rint(sum[i]);
      o[ob + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}

/** `cv2.getPerspectiveTransform(src, dst)`: the 3x3 homography taking `src` corners to `dst`. */
function perspectiveTransform(srcQ, dstQ) {
  const a = [];
  for (let i = 0; i < 8; i++) a.push(new Float64Array(9));
  for (let i = 0; i < 4; i++) {
    const x = srcQ[i][0], y = srcQ[i][1];
    const u = dstQ[i][0], v = dstQ[i][1];
    a[i].set([x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u]);
    a[i + 4].set([0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v]);
  }
  for (let col = 0; col < 8; col++) {
    // Iterator::max_by: the last maximum
    let piv = col;
    for (let p = col + 1; p < 8; p++) if (Math.abs(a[p][col]) >= Math.abs(a[piv][col])) piv = p;
    const tmp = a[col];
    a[col] = a[piv];
    a[piv] = tmp;
    const d = a[col][col];
    if (Math.abs(d) < 1e-300) continue;
    for (let r = 0; r < 8; r++) {
      if (r !== col) {
        const f = a[r][col] / d;
        if (f !== 0) for (let k = col; k < 9; k++) a[r][k] -= f * a[col][k];
      }
    }
  }
  const m = new Float64Array(9);
  for (let i = 0; i < 8; i++) m[i] = a[i][i] !== 0 ? a[i][8] / a[i][i] : 0.0;
  m[8] = 1.0;
  return m;
}

function invert3(m) {
  const det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (det === 0) return new Float64Array(9);
  const d = 1.0 / det;
  return Float64Array.of(
    (m[4] * m[8] - m[5] * m[7]) * d,
    (m[2] * m[7] - m[1] * m[8]) * d,
    (m[1] * m[5] - m[2] * m[4]) * d,
    (m[5] * m[6] - m[3] * m[8]) * d,
    (m[0] * m[8] - m[2] * m[6]) * d,
    (m[2] * m[3] - m[0] * m[5]) * d,
    (m[3] * m[7] - m[4] * m[6]) * d,
    (m[1] * m[6] - m[0] * m[7]) * d,
    (m[0] * m[4] - m[1] * m[3]) * d,
  );
}

const INTER_BITS = 5;
const INTER_TAB = 1 << INTER_BITS;
const REMAP_SCALE = 1 << 15;

/** OpenCV `interpolateCubic` (A = -0.75), in f32. */
function cubicCoeffs(x) {
  const A = f32(-0.75);
  const x1 = f32(x + 1);
  const c0 = f32(f32(f32(f32(f32(f32(A * x1) - f32(5 * A)) * x1) + f32(8 * A)) * x1) - f32(4 * A));
  const c1 = f32(f32(f32(f32(f32(f32(A + 2) * x) - f32(A + 3)) * x) * x) + 1);
  const omx = f32(1 - x);
  const c2 = f32(f32(f32(f32(f32(f32(A + 2) * omx) - f32(A + 3)) * omx) * omx) + 1);
  return [c0, c1, c2, f32(f32(f32(1 - c0) - c1) - c2)];
}

let BICUBIC_TAB = null;
/** OpenCV `initInterTab2D(INTER_CUBIC)`: 32 x 32 sub-pixel positions, 16 weights summing to 2^15. */
function bicubicTab() {
  if (BICUBIC_TAB) return BICUBIC_TAB;
  const oneD = [];
  for (let i = 0; i < INTER_TAB; i++) oneD.push(cubicCoeffs(f32(i / INTER_TAB)));
  const tab = new Int32Array(INTER_TAB * INTER_TAB * 16);
  for (let iy = 0; iy < INTER_TAB; iy++) {
    for (let ix = 0; ix < INTER_TAB; ix++) {
      const vy = oneD[iy], vx = oneD[ix];
      const t = new Int32Array(16);
      let isum = 0;
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          const v = f32(vy[i] * vx[j]);
          t[i * 4 + j] = satShort(f32(v * REMAP_SCALE));
          isum += t[i * 4 + j];
        }
      }
      if (isum !== REMAP_SCALE) {
        const diff = isum - REMAP_SCALE;
        let mk = [2, 2], big = [2, 2];
        for (let k1 = 2; k1 < 4; k1++) {
          for (let k2 = 2; k2 < 4; k2++) {
            if (t[k1 * 4 + k2] < t[mk[0] * 4 + mk[1]]) mk = [k1, k2];
            else if (t[k1 * 4 + k2] > t[big[0] * 4 + big[1]]) big = [k1, k2];
          }
        }
        if (diff < 0) t[big[0] * 4 + big[1]] -= diff;
        else t[mk[0] * 4 + mk[1]] -= diff;
      }
      tab.set(t, (iy * INTER_TAB + ix) * 16);
    }
  }
  BICUBIC_TAB = tab;
  return tab;
}

/** `cv2.warpPerspective(img, m, (w, h), flags=INTER_CUBIC, borderMode=BORDER_REPLICATE)`. */
function warpPerspectiveCubic(img, m, w, h) {
  const inv = invert3(m);
  const tab = bicubicTab();
  const ch = img.ch, src = img.data, iw = img.w;
  const out = imgNew(w, h, ch);
  const o = out.data;
  const sw = img.w, sh = img.h;
  const cols = [0, 0, 0, 0];
  const sum = [0, 0, 0, 0];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const x0 = inv[0] * x + inv[1] * y + inv[2];
      const y0 = inv[3] * x + inv[4] * y + inv[5];
      let ww = inv[6] * x + inv[7] * y + inv[8];
      ww = ww !== 0 ? INTER_TAB / ww : 0.0;
      const fx = clamp(x0 * ww, -2147483648, 2147483647);
      const fy = clamp(y0 * ww, -2147483648, 2147483647);
      const xi = rint(fx), yi = rint(fy);
      const sx = Math.floor(xi / INTER_TAB), sy = Math.floor(yi / INTER_TAB);
      const a = (((yi % INTER_TAB) + INTER_TAB) % INTER_TAB) * INTER_TAB + (((xi % INTER_TAB) + INTER_TAB) % INTER_TAB);
      const wb = a * 16;
      const inside = sx >= 1 && sy >= 1 && sx + 2 < sw && sy + 2 < sh;
      for (let j = 0; j < 4; j++) cols[j] = inside ? sx - 1 + j : clamp(sx - 1 + j, 0, sw - 1);
      for (let c = 0; c < ch; c++) sum[c] = 0;
      for (let i = 0; i < 4; i++) {
        const yy = inside ? sy - 1 + i : clamp(sy - 1 + i, 0, sh - 1);
        const rb = yy * iw * ch;
        for (let j = 0; j < 4; j++) {
          const wv = tab[wb + i * 4 + j];
          const pb = rb + cols[j] * ch;
          for (let c = 0; c < ch; c++) sum[c] += src[pb + c] * wv;
        }
      }
      const ob = (y * w + x) * ch;
      for (let c = 0; c < ch; c++) {
        const v = Math.floor((sum[c] + 16384) / 32768);
        o[ob + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }
  return out;
}

/**
 * 4-connected components of `mask` (row-major, w x h), in raster order of each component's first
 * pixel; `pixels` holds flat indices in the order the Rust flood fill visits them.
 */
function components4(mask, w, h) {
  const n = w * h;
  const seen = new Uint8Array(n);
  const stack = new Int32Array(n);
  const out = [];
  for (let start = 0; start < n; start++) {
    if (!mask[start] || seen[start]) continue;
    seen[start] = 1;
    let sp = 0;
    stack[sp++] = start;
    let x0 = Infinity, y0 = Infinity, x1 = 0, y1 = 0;
    const pixels = [];
    while (sp > 0) {
      const p = stack[--sp];
      const x = p % w, y = (p - x) / w;
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
      pixels.push(p);
      if (x > 0) {
        const q = p - 1;
        if (mask[q] && !seen[q]) { seen[q] = 1; stack[sp++] = q; }
      }
      if (x + 1 < w) {
        const q = p + 1;
        if (mask[q] && !seen[q]) { seen[q] = 1; stack[sp++] = q; }
      }
      if (y > 0) {
        const q = p - w;
        if (mask[q] && !seen[q]) { seen[q] = 1; stack[sp++] = q; }
      }
      if (y + 1 < h) {
        const q = p + w;
        if (mask[q] && !seen[q]) { seen[q] = 1; stack[sp++] = q; }
      }
    }
    out.push({ x0, y0, x1, y1, pixels });
  }
  return out;
}

function cross(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

/** `cv2.minAreaRect` (convex hull + one candidate per hull edge). */
function minAreaRect(pts) {
  const p0 = pts.map((q) => [q[0], q[1]]);
  p0.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const p = [];
  for (const q of p0) {
    const l = p[p.length - 1];
    if (!l || l[0] !== q[0] || l[1] !== q[1]) p.push(q);
  }
  if (p.length === 1) return { cx: f32(p[0][0]), cy: f32(p[0][1]), w: 0, h: 0, angle: 0 };
  const hull = [];
  for (const q of p) {
    while (hull.length >= 2 && cross(hull[hull.length - 2], hull[hull.length - 1], q) <= 0) hull.pop();
    hull.push(q);
  }
  const lower = hull.length + 1;
  for (let i = p.length - 2; i >= 0; i--) {
    const q = p[i];
    while (hull.length >= lower && cross(hull[hull.length - 2], hull[hull.length - 1], q) <= 0) hull.pop();
    hull.push(q);
  }
  hull.pop();
  let best = null, bestArea = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    // libm hypot: exact here (cell corners times an integer stride), so a plain sqrt matches it
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len === 0) continue;
    const ux = dx / len, uy = dy / len;
    const vx = -uy, vy = ux;
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const q of hull) {
      const pu = q[0] * ux + q[1] * uy;
      const pv = q[0] * vx + q[1] * vy;
      if (pu < u0) u0 = pu;
      if (pu > u1) u1 = pu;
      if (pv < v0) v0 = pv;
      if (pv > v1) v1 = pv;
    }
    const area = (u1 - u0) * (v1 - v0);
    if (best === null || area < bestArea) {
      const cu = (u0 + u1) / 2.0, cv = (v0 + v1) / 2.0;
      bestArea = area;
      best = {
        cx: f32(cu * ux + cv * vx),
        cy: f32(cu * uy + cv * vy),
        w: f32(u1 - u0),
        h: f32(v1 - v0),
        angle: f32(Math.atan2(uy, ux) * DEG_PER_RAD),
      };
    }
  }
  return best ?? { cx: f32(p[0][0]), cy: f32(p[0][1]), w: 0, h: 0, angle: 0 };
}

/** `cv2.boxPoints`: the rectangle's four corners. */
function boxPoints(r) {
  const a = r.angle * RAD_PER_DEG;
  const b = f32(Math.cos(a) * 0.5), c = f32(Math.sin(a) * 0.5);
  const p0 = [
    f32(f32(r.cx - f32(c * r.h)) - f32(b * r.w)),
    f32(f32(r.cy + f32(b * r.h)) - f32(c * r.w)),
  ];
  const p1 = [
    f32(f32(r.cx + f32(c * r.h)) - f32(b * r.w)),
    f32(f32(r.cy - f32(b * r.h)) - f32(c * r.w)),
  ];
  const p2 = [f32(f32(2 * r.cx) - p0[0]), f32(f32(2 * r.cy) - p0[1])];
  const p3 = [f32(f32(2 * r.cx) - p1[0]), f32(f32(2 * r.cy) - p1[1])];
  return [p0, p1, p2, p3];
}

/** Otsu's threshold over 8-bit values: the first level maximising the between-class variance. */
function otsuHist(hist, count) {
  const n = Math.max(count, 1);
  let w = 0, mu = 0;
  const cw = new Float64Array(256), cmu = new Float64Array(256);
  for (let i = 0; i < 256; i++) {
    w += hist[i] / n;
    mu += (hist[i] / n) * i;
    cw[i] = w;
    cmu[i] = mu;
  }
  const muT = cmu[255];
  let bestS = -Infinity, best = 0;
  for (let i = 0; i < 256; i++) {
    const wi = cw[i];
    const s = (muT * wi - cmu[i]) ** 2 / (wi * (1.0 - wi));
    if (Number.isFinite(s) && s > bestS) {
      bestS = s;
      best = i;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------------------------
// det.rs

/** Kernel shrink ratio the detector targets were built with. */
const SHRINK = f32(0.3);

function normF(x, y) {
  return f32(Math.sqrt(f32(f32(x * x) + f32(y * y))));
}

function quadMinY(q) {
  let m = Infinity;
  for (const p of q) m = fminF(m, p[1]);
  return m;
}
function quadMinX(q) {
  let m = Infinity;
  for (const p of q) m = fminF(m, p[0]);
  return m;
}

/** Corners as [top-left, top-right, bottom-right, bottom-left] along the line's long axis. */
export function orderQuad(q) {
  const c = [
    f32(f32(f32(f32(q[0][0] + q[1][0]) + q[2][0]) + q[3][0]) / 4),
    f32(f32(f32(f32(q[0][1] + q[1][1]) + q[2][1]) + q[3][1]) / 4),
  ];
  const e1 = [f32(q[1][0] - q[0][0]), f32(q[1][1] - q[0][1])];
  const e2 = [f32(q[2][0] - q[1][0]), f32(q[2][1] - q[1][1])];
  let u = normF(e1[0], e1[1]) >= normF(e2[0], e2[1]) ? e1 : e2;
  const n = f32(normF(u[0], u[1]) + f32(1e-6));
  u = [f32(u[0] / n), f32(u[1] / n)];
  if (u[0] < 0) u = [-u[0], -u[1]];
  const v = [-u[1], u[0]];
  const a = q.map((p) => f32(f32(f32(p[0] - c[0]) * u[0]) + f32(f32(p[1] - c[1]) * u[1])));
  const b = q.map((p) => f32(f32(f32(p[0] - c[0]) * v[0]) + f32(f32(p[1] - c[1]) * v[1])));
  const pick = (f, max) => {
    let best = 0;
    for (let i = 1; i < 4; i++) if ((max && f(i) > f(best)) || (!max && f(i) < f(best))) best = i;
    return best;
  };
  const tl = pick((i) => f32(a[i] + b[i]), false);
  const tr = pick((i) => f32(a[i] - b[i]), true);
  const br = pick((i) => f32(a[i] + b[i]), true);
  const bl = pick((i) => f32(a[i] - b[i]), false);
  return [q[tl], q[tr], q[br], q[bl]];
}

/** Join near-horizontal line quads that continue one another (`lpocr_rec._merge_collinear`). */
export function mergeCollinear(boxes, gap) {
  const MAX_TILT = f32(0.35), MAX_PERP = f32(0.35), MIN_H_RATIO = f32(0.6), MAX_DANGLE = f32(0.07);
  gap = f32(gap);
  const n = boxes.length;
  if (n < 2) return boxes.slice();
  const c = boxes.map((q) => [
    f32(f32(f32(f32(q[0][0] + q[1][0]) + q[2][0]) + q[3][0]) / 4),
    f32(f32(f32(f32(q[0][1] + q[1][1]) + q[2][1]) + q[3][1]) / 4),
  ]);
  const u = [], len = [], hgt = [];
  for (const q of boxes) {
    const d = [f32(q[1][0] - q[0][0]), f32(q[1][1] - q[0][1])];
    const l = fmaxF(normF(d[0], d[1]), f32(1e-3));
    u.push([f32(d[0] / l), f32(d[1] / l)]);
    len.push(l);
    hgt.push(fmaxF(normF(f32(q[3][0] - q[0][0]), f32(q[3][1] - q[0][1])), f32(1e-3)));
  }
  const v = u.map((uu) => [-uu[1], uu[0]]);
  const flat = [];
  for (let i = 0; i < n; i++) flat.push(Math.abs(u[i][1]) <= MAX_TILT && len[i] >= f32(0.5 * hgt[i]));
  const cosMin = f32(1 - f32(f32(MAX_DANGLE * MAX_DANGLE) / 2));
  const right = new Array(n).fill(-1);
  const leftBest = Array.from({ length: n }, () => [Infinity, -1]);
  for (let a = 0; a < n; a++) {
    let best = [Infinity, -1];
    if (flat[a]) {
      for (let b = 0; b < n; b++) {
        if (a === b || !flat[b]) continue;
        const d = [f32(c[b][0] - c[a][0]), f32(c[b][1] - c[a][1])];
        const along = f32(f32(d[0] * u[a][0]) + f32(d[1] * u[a][1]));
        const perp = Math.abs(f32(f32(d[0] * v[a][0]) + f32(d[1] * v[a][1])));
        const hm = f32(f32(hgt[a] + hgt[b]) / 2);
        const g = f32(along - f32(f32(len[a] + len[b]) / 2));
        const ok = along > 0 &&
          f32(fminF(hgt[a], hgt[b]) / fmaxF(hgt[a], hgt[b])) >= MIN_H_RATIO &&
          Math.abs(f32(f32(u[a][0] * u[b][0]) + f32(u[a][1] * u[b][1]))) >= cosMin &&
          perp <= f32(MAX_PERP * hm) &&
          g <= f32(gap * hm) &&
          g >= f32(-0.5 * fminF(len[a], len[b]));
        if (ok) {
          if (g < best[0]) best = [g, b];
          if (g < leftBest[b][0]) leftBest[b] = [g, a];
        }
      }
    }
    right[a] = best[1];
  }
  const nxt = new Array(n).fill(-1);
  for (let a = 0; a < n; a++) {
    const b = right[a];
    if (b >= 0 && leftBest[b][1] === a) nxt[a] = b;
  }
  const hasPrev = new Array(n).fill(false);
  for (const b of nxt) if (b >= 0) hasPrev[b] = true;
  const out = [];
  const seen = new Array(n).fill(false);
  for (let a = 0; a < n; a++) {
    if (hasPrev[a] || seen[a]) continue;
    const chain = [a];
    seen[a] = true;
    for (;;) {
      const last = chain[chain.length - 1];
      const nx = nxt[last];
      if (nx < 0 || seen[nx]) break;
      chain.push(nx);
      seen[nx] = true;
    }
    if (chain.length === 1) {
      out.push(boxes[a]);
      continue;
    }
    const pts = [];
    for (const i of chain) for (const p of boxes[i]) pts.push(p);
    let k = chain[0];
    for (const i of chain.slice(1)) if (len[i] > len[k]) k = i;
    const np = f32(pts.length);
    let sx = 0, sy = 0;
    for (const p of pts) {
      sx = f32(sx + p[0]);
      sy = f32(sy + p[1]);
    }
    const ca = [f32(sx / np), f32(sy / np)];
    const ua = u[k], va = v[k];
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (const p of pts) {
      const dx = f32(p[0] - ca[0]), dy = f32(p[1] - ca[1]);
      const pa = f32(f32(dx * ua[0]) + f32(dy * ua[1]));
      const pb = f32(f32(dx * va[0]) + f32(dy * va[1]));
      a0 = fminF(a0, pa);
      a1 = fmaxF(a1, pa);
      b0 = fminF(b0, pb);
      b1 = fmaxF(b1, pb);
    }
    const at = (s, t) => [
      f32(f32(ca[0] + f32(ua[0] * s)) + f32(va[0] * t)),
      f32(f32(ca[1] + f32(ua[1] * s)) + f32(va[1] * t)),
    ];
    out.push(orderQuad([at(a0, b0), at(a1, b0), at(a1, b1), at(a0, b1)]));
  }
  for (let i = 0; i < n; i++) if (!seen[i]) out.push(boxes[i]);
  return out;
}

/** Candidate boxes for faint short tokens, in image pixels (`det::faint_boxes`). */
function faintBoxes(prob, mask, pw, ph, toImg, boxes, f, img) {
  const PAD = f32(0.35);
  const bbox = (q) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of q) {
      x0 = fminF(x0, p[0]);
      y0 = fminF(y0, p[1]);
      x1 = fmaxF(x1, p[0]);
      y1 = fmaxF(y1, p[1]);
    }
    return [x0, y0, x1, y1];
  };
  const lines = boxes.map(bbox);
  const rows = [];
  for (let i = 0; i < boxes.length; i++) {
    const q = boxes[i], l = lines[i];
    if (Math.abs(f32(q[1][1] - q[0][1])) <= f32(f32(0.1) * f32(l[2] - l[0]))) rows.push(l);
  }
  if (!rows.length) return [];
  const thr = f32(f.thresh);
  const weak = new Uint8Array(prob.length);
  for (let i = 0; i < prob.length; i++) weak[i] = prob[i] > thr ? 1 : 0;
  const cands = [];
  for (const comp of components4(weak, pw, ph)) {
    let touches = false;
    for (const p of comp.pixels) if (mask[p]) { touches = true; break; }
    if (touches) continue;
    const [x0, y0] = toImg(comp.x0, comp.y0);
    const [x1, y1] = toImg(comp.x1 + 1, comp.y1 + 1);
    const cy = f32(f32(y0 + y1) / 2);
    let row = null, rowKey = 0;
    for (const r of rows) {
      const h = f32(r[3] - r[1]);
      if (!(cy >= f32(r[1] + f32(0.25 * h)) && cy <= f32(r[3] - f32(0.25 * h)))) continue;
      const key = Math.abs(f32(f32(r[1] + r[3]) - f32(2 * cy)));
      if (row === null || key < rowKey) {
        row = r;
        rowKey = key;
      }
    }
    if (row === null) continue;
    const ry0 = row[1], ry1 = row[3];
    const h = f32(ry1 - ry0);
    if (f32(y1 - y0) > f32(f32(0.8) * h) || f32(x1 - x0) > f32(1.5 * h)) continue;
    const bx0 = fmaxF(f32(x0 - f32(PAD * h)), 0), bx1 = fminF(f32(x1 + f32(PAD * h)), f32(img.w));
    const by0 = fmaxF(ry0, 0), by1 = fminF(ry1, f32(img.h));
    const area = f32(f32(bx1 - bx0) * f32(by1 - by0));
    let covered = false;
    for (const l of lines) {
      const ix = fmaxF(f32(fminF(bx1, l[2]) - fmaxF(bx0, l[0])), 0);
      const iy = fmaxF(f32(fminF(by1, l[3]) - fmaxF(by0, l[1])), 0);
      if (f32(ix * iy) > f32(f32(0.2) * area)) {
        covered = true;
        break;
      }
    }
    if (covered) continue;
    let peak = 0;
    for (const p of comp.pixels) peak = fmaxF(peak, prob[p]);
    cands.push([peak, [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]]);
  }
  cands.sort((a, b) => b[0] - a[0]);
  return cands.slice(0, f.max).map((c) => c[1]);
}

/** One detection pass with the long side resized to `longSide` (never enlarged). */
async function detectAt(ctx, img, longSide, withFaint) {
  const bigH = img.h, bigW = img.w;
  const s = Math.min(longSide / Math.max(bigH, bigW), 1.0);
  const h = Math.max(32, rint((bigH * s) / 32.0) * 32);
  const w = Math.max(32, rint((bigW * s) / 32.0) * 32);
  const sx = w / bigW, sy = h / bigH;
  const small = resizeArea(img, w, h);
  const hw = h * w;
  const x = new Float32Array(3 * hw);
  const sd = small.data;
  for (let c = 0; c < 3; c++) {
    const off = c * hw;
    for (let i = 0; i < hw; i++) x[off + i] = DET_LUT[sd[i * 3 + c]];
  }
  const res = await ctx.infer("det", [1, 3, h, w], x, ["prob"]);
  const prob = toF32(res.prob.data);
  const ph = res.prob.shape[2], pw = res.prob.shape[3];
  const stride = f32(Math.floor(h / ph));
  const thresh = ctx.detThresh;
  const mask = new Uint8Array(prob.length);
  for (let i = 0; i < prob.length; i++) mask[i] = prob[i] > thresh ? 1 : 0;
  let boxes = [];
  const minScore = f32(0.6);
  for (const comp of components4(mask, pw, ph)) {
    if (comp.pixels.length < 2) continue;
    let sum = 0;
    for (const p of comp.pixels) sum += prob[p];
    const mean = sum / comp.pixels.length;
    if (f32(mean) < minScore) continue;
    const rows = comp.y1 - comp.y0 + 1;
    const sx0 = new Array(rows).fill(Infinity), sx1 = new Array(rows).fill(0);
    for (const p of comp.pixels) {
      const px = p % pw, py = (p - px) / pw;
      const r = py - comp.y0;
      if (px < sx0[r]) sx0[r] = px;
      if (px > sx1[r]) sx1[r] = px;
    }
    const pts = [];
    for (let i = 0; i < rows; i++) {
      if (sx0[i] === Infinity) continue;
      const yy = comp.y0 + i;
      for (const [qx, qy] of [[sx0[i], yy], [sx0[i], yy + 1], [sx1[i] + 1, yy], [sx1[i] + 1, yy + 1]]) {
        pts.push([f32(qx * stride), f32(qy * stride)]);
      }
    }
    const r = minAreaRect(pts);
    // line height is the side nearer vertical, not the shorter side
    let ang = f32(r.angle + 90);
    ang = ang % 180;
    if (ang < 0) ang = f32(ang + 180);
    const wIsHorizontal = Math.abs(f32(ang - 90)) < 45;
    const kernelH = wIsHorizontal ? r.h : r.w;
    const lineH = f32(kernelH / f32(1 - f32(2 * SHRINK)));
    const d = f32(SHRINK * lineH);
    r.w = f32(r.w + f32(2 * d));
    r.h = f32(r.h + f32(2 * d));
    const quad = boxPoints(r);
    for (const p of quad) {
      p[0] = clampF(f32(p[0] / sx), 0, f32(img.w));
      p[1] = clampF(f32(p[1] / sy), 0, f32(img.h));
    }
    boxes.push(orderQuad(quad));
  }
  if (ctx.detMerge > 0) boxes = mergeCollinear(boxes, ctx.detMerge);
  // (row band of 8 px, then left edge); a stable sort
  const keyed = boxes.map((b) => [rint(quadMinY(b) / 8.0), quadMinX(b), b]);
  keyed.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  boxes = keyed.map((k) => k[2]);
  let faint = [];
  if (withFaint && ctx.opts.faint) {
    const toImg = (px, py) => [f32(f32(px * stride) / sx), f32(f32(py * stride) / sy)];
    faint = faintBoxes(prob, mask, pw, ph, toImg, boxes, ctx.opts.faint, img);
  }
  return { boxes, scale: s, faint };
}

/** Line quads and faint candidates (`Detector::detect_with_faint`). */
async function detect(ctx, img, withFaint) {
  const first = await detectAt(ctx, img, ctx.opts.detLongSide, withFaint);
  const a = ctx.opts.detAdaptive;
  if (!a || first.boxes.length < 8) return first;
  const hs = first.boxes.map((q) => f32(normF(f32(q[3][0] - q[0][0]), f32(q[3][1] - q[0][1])) * f32(first.scale)));
  hs.sort((x, y) => x - y);
  const med = fmaxF(hs[hs.length >> 1], 1);
  if (med >= f32(a.minH)) return first;
  const side = Math.floor(Math.min(trunc(f32(f32(f32(ctx.opts.detLongSide) * f32(a.targetH)) / med)), a.maxSide) / 32) * 32;
  if (side <= ctx.opts.detLongSide) return first;
  return detectAt(ctx, img, side, withFaint);
}

// ---------------------------------------------------------------------------------------------
// page.rs

function distF(a, b) {
  const dx = f32(a[0] - b[0]), dy = f32(a[1] - b[1]);
  return f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy))));
}

/** Perspective-crop a (possibly rotated) quad, as PaddleOCR does (`lpocr_rec._crop`). */
function crop(img, q) {
  const w = Math.max(trunc(fmaxF(distF(q[0], q[1]), distF(q[2], q[3]))), 2);
  const h = Math.max(trunc(fmaxF(distF(q[0], q[3]), distF(q[1], q[2]))), 2);
  const dst = [[0, 0], [f32(w), 0], [f32(w), f32(h)], [0, f32(h)]];
  const m = perspectiveTransform(q, dst);
  const c = warpPerspectiveCubic(img, m, w, h);
  return h / w >= 1.5 ? rot90(c, 1) : c;
}

function extent(q) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of q) {
    x0 = Math.min(x0, p[0]);
    y0 = Math.min(y0, p[1]);
    x1 = Math.max(x1, p[0]);
    y1 = Math.max(y1, p[1]);
  }
  return [x0, y0, x1, y1];
}

/** Boxes that are clearly horizontal text lines (at least 3x wider than tall, >= 30 px). */
function longLines(boxes) {
  let n = 0;
  for (const q of boxes) {
    const [x0, y0, x1, y1] = extent(q);
    const dx = x1 - x0, dy = y1 - y0;
    if (dx >= 30.0 && dx >= 3.0 * dy) n++;
  }
  return n;
}

/** Detector boxes of a page scanned on its side: many boxes, almost none horizontal lines. */
function looksSideways(boxes) {
  return boxes.length >= 15 && longLines(boxes) < 0.25 * boxes.length;
}

/** A box quad as a quad in the original image whose corners run in reading order. */
function readingFrame(q, flip, sideways, w0) {
  let p = q.map((pt) => [pt[0], pt[1]]);
  if (flip) p = [p[2], p[3], p[0], p[1]];
  if (sideways) p = p.map((pt) => [f32(w0 - 1.0 - pt[1]), pt[0]]);
  return p;
}

/** Minimum blank-run width, in line heights, for a column gutter. */
const GUTTER_MIN = 0.4;

/** Cut near-horizontal boxes at blank vertical runs that line up with other boxes' edges. */
function splitGutters(img, boxes) {
  if (boxes.length < 3) return boxes.slice();
  const gray = meanGray(img);
  const gw = gray.w, gh = gray.h, gd = gray.data;
  const ext = boxes.map(extent);
  const out = [];
  const hist = new Float64Array(256);
  for (let i = 0; i < boxes.length; i++) {
    const q = boxes[i];
    const [x0, y0, x1, y1] = ext[i];
    const h = y1 - y0;
    const tilt = Math.abs(q[1][1] - q[0][1]);
    if (h < 6.0 || x1 - x0 < 4.0 * h || tilt > 0.25 * h) {
      out.push(q);
      continue;
    }
    const ya = Math.min(Math.max(trunc(y0 + 0.15 * h), 0), gh);
    const yb = clamp(trunc(Math.ceil(y1 - 0.15 * h)), 0, gh);
    const xa = Math.max(trunc(x0), 0);
    const xb = Math.min(trunc(Math.ceil(x1)), gw);
    if (yb <= ya || xb <= xa) {
      out.push(q);
      continue;
    }
    let lo = 255, hi = 0;
    hist.fill(0);
    for (let y = ya; y < yb; y++) {
      const rb = y * gw;
      for (let x = xa; x < xb; x++) {
        const v = gd[rb + x];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        hist[v] += 1;
      }
    }
    if (hi - lo < 60) {
      out.push(q);
      continue;
    }
    const thr = otsuHist(hist, (yb - ya) * (xb - xa));
    const rule = ruleColumns(gd, xa, xb, y0, y1, gw, gh, thr);
    const idx = [];
    for (let x = xa; x < xb; x++) {
      if (rule.has(x)) continue;
      for (let y = ya; y < yb; y++) {
        if (gd[y * gw + x] < thr) {
          idx.push(x - xa);
          break;
        }
      }
    }
    if (idx.length < 2) {
      out.push(q);
      continue;
    }
    const cy = (y0 + y1) / 2.0, tol = 0.6 * h;
    const cuts = [];
    for (let g = 0; g < idx.length - 1; g++) {
      if (idx[g + 1] - idx[g] <= GUTTER_MIN * h) continue;
      const a = xa + idx[g] + 1, b = xa + idx[g + 1];
      let left = 0, right = 0;
      for (let j = 0; j < ext.length; j++) {
        const e = ext[j];
        if (j === i || !(Math.abs((e[1] + e[3]) / 2.0 - cy) < 25.0 * h) || !(0.5 * h < e[3] - e[1]) || !(e[3] - e[1] < 2.0 * h)) continue;
        if (Math.abs(e[0] - b) < tol) left++;
        if (Math.abs(e[2] - a) < tol) right++;
      }
      if (left < 1 || right < 1 || left + right < 3 || a - x0 < 3.0 * h) continue;
      const m = Math.max(trunc((b - a) * 0.25), 1);
      const sa = trunc(a + m), sb = trunc(b - m);
      if (sb <= sa) continue;
      let ok = 0, bad = 0;
      for (let j = 0; j < ext.length; j++) {
        const e = ext[j];
        if (j === i || Math.abs((e[1] + e[3]) / 2.0 - cy) > 4.0 * h || !(0.5 * h < e[3] - e[1] && e[3] - e[1] < 2.0 * h)) continue;
        if (e[2] <= sa || e[0] >= sb) {
          ok++;
          continue;
        }
        const eh = e[3] - e[1];
        const ra = clamp(trunc(e[1] + 0.15 * eh), 0, gh), rb = clamp(trunc(Math.ceil(e[3] - 0.15 * eh)), 0, gh);
        const ca = clamp(sa, 0, gw), cb = clamp(sb, 0, gw);
        let inked = false;
        for (let y = ra; y < rb && !inked; y++) {
          for (let x = ca; x < cb; x++) {
            if (!rule.has(x) && gd[y * gw + x] < thr) {
              inked = true;
              break;
            }
          }
        }
        if (inked) bad++;
        else ok++;
      }
      if (!(ok >= 2 && (bad === 0 || (bad === 1 && ok >= 3)))) continue;
      cuts.push([a, b]);
    }
    if (!cuts.length) {
      out.push(q);
      continue;
    }
    const pad = 0.15 * h;
    const edges = [x0];
    for (const [a, b] of cuts) edges.push(a + pad, b - pad);
    edges.push(x1);
    for (let k = 0; k + 1 < edges.length; k += 2) {
      const s = edges[k], e = edges[k + 1];
      if (e - s > 0.5 * h) {
        const S = f32(s), E = f32(e), YA = f32(y0), YB = f32(y1);
        out.push([[S, YA], [E, YA], [E, YB], [S, YB]]);
      }
    }
  }
  return out;
}

/** Columns of a printed vertical rule crossing the line box `y0..y1` (`page::rule_columns`). */
function ruleColumns(gd, xa, xb, y0, y1, gw, gh, thr) {
  const h = y1 - y0;
  const band = trunc(Math.ceil(0.35 * h));
  const top = trunc(y0), bot = trunc(Math.ceil(y1));
  const inkedShare = (x, ya, yb) => {
    ya = clamp(ya, 0, gh);
    yb = clamp(yb, 0, gh);
    if (yb <= ya || x < 0 || x >= gw) return 0.0;
    let n = 0;
    for (let y = ya; y < yb; y++) if (gd[y * gw + x] < thr) n++;
    return n / (yb - ya);
  };
  const solid = [];
  for (let x = xa; x < xb; x++) {
    solid.push(inkedShare(x, top, bot) >= 0.9 && inkedShare(x, top - band, top) >= 0.8 && inkedShare(x, bot, bot + band) >= 0.8);
  }
  const maxW = Math.max(trunc(Math.ceil(0.15 * h)), 3);
  const out = new Set();
  let i = 0;
  while (i < solid.length) {
    if (!solid[i]) {
      i++;
      continue;
    }
    const start = i;
    while (i < solid.length && solid[i]) i++;
    const flank = Math.max(trunc(Math.ceil(0.25 * h)), 3);
    const blankSide = (from, to) => {
      let n = 0;
      for (let x = from; x < to; x++) if (inkedShare(x, top, bot) < 0.1) n++;
      return n * 2 >= to - from;
    };
    const rs = xa + start, re = xa + i;
    if (i - start <= maxW && blankSide(rs - flank, rs) && blankSide(re, re + flank)) {
      const pad = Math.max(trunc(Math.ceil(0.06 * h)), 2);
      for (let x = xa + start - pad; x < xa + i + pad; x++) out.add(x);
    }
  }
  return out;
}

/** The most common group (np.bincount(..).argmax(): the lowest index on ties). */
function majority(groups) {
  const n = [0, 0, 0, 0];
  for (const g of groups) n[g]++;
  let best = 0;
  for (let i = 1; i < 4; i++) if (n[i] > n[best]) best = i;
  return best;
}

const inR = (u, a, b) => u >= a && u <= b;
const SCRIPT_RANGES = [
  null,
  [[0x3040, 0x30ff], [0x3400, 0x9fff], [0xac00, 0xd7af], [0xf900, 0xfaff]],
  [[0x0590, 0x08ff], [0xfb1d, 0xfdff], [0xfe70, 0xfeff]],
  [[0x0900, 0x0dff], [0x0e00, 0x0eff], [0x0f00, 0x109f], [0x10a0, 0x10ff], [0x0530, 0x058f], [0x1200, 0x139f], [0x1780, 0x17ff]],
];

/** Characters that show a reading really is in `group`'s script (`server.SCRIPT_CHARS`). */
function scriptChars(group, text) {
  let n = 0;
  for (const ch of text) {
    const u = ch.codePointAt(0);
    if (group === 0) {
      if (isAsciiAlpha(u) || inR(u, 0x00c0, 0x024f) || inR(u, 0x0370, 0x04ff)) n++;
    } else {
      const rs = SCRIPT_RANGES[group > 3 ? 3 : group];
      if (rs.some(([a, b]) => inR(u, a, b))) n++;
    }
  }
  return n;
}

// Unicode character classes as Rust's char methods define them
const RE_ALPHA = /^\p{Alphabetic}$/u;
const RE_UPPER = /^\p{Uppercase}$/u;
const RE_LOWER = /^\p{Lowercase}$/u;
const RE_ALNUM = /^[\p{Alphabetic}\p{N}]$/u;
const RE_WS = /^\p{White_Space}$/u;
const charCache = new Map();
/** bit flags per code point: 1 alphabetic, 2 uppercase, 4 lowercase, 8 alphanumeric, 16 whitespace */
function charFlags(ch) {
  let f = charCache.get(ch);
  if (f === undefined) {
    f = (RE_ALPHA.test(ch) ? 1 : 0) | (RE_UPPER.test(ch) ? 2 : 0) | (RE_LOWER.test(ch) ? 4 : 0) |
      (RE_ALNUM.test(ch) ? 8 : 0) | (RE_WS.test(ch) ? 16 : 0);
    charCache.set(ch, f);
  }
  return f;
}
const isAlpha = (ch) => (charFlags(ch) & 1) !== 0;
const isUpper = (ch) => (charFlags(ch) & 2) !== 0;
const isLower = (ch) => (charFlags(ch) & 4) !== 0;
const isAlnum = (ch) => (charFlags(ch) & 8) !== 0;
const isWs = (ch) => (charFlags(ch) & 16) !== 0;
const isAsciiAlpha = (u) => (u >= 0x41 && u <= 0x5a) || (u >= 0x61 && u <= 0x7a);

/** str::trim: White_Space at both ends. */
function trimWs(s) {
  const cs = Array.from(s);
  let a = 0, b = cs.length;
  while (a < b && isWs(cs[a])) a++;
  while (b > a && isWs(cs[b - 1])) b--;
  return cs.slice(a, b).join("");
}

function zipMap(from, to) {
  const f = Array.from(from), t = Array.from(to);
  const m = new Map();
  for (let i = 0; i < Math.min(f.length, t.length); i++) if (!m.has(f[i])) m.set(f[i], t[i]);
  return m;
}
/** Cyrillic / Greek lookalikes of Latin letters (`server._HOMO`), then the other scripts' ones. */
const HOMO = (() => {
  const m = zipMap("аеорсухкіјѕԁԛԝАВЕКМНОРСТХІЈЅΑΒΕΖΗΙΚΜΝΟΡΤΥΧοαν", "aeopcyxkijsdqwABEKMHOPCTXIJSABEZHIKMNOPTYXoav");
  for (const [k, v] of zipMap("ՍսՕօոհցզաԼՏนทรอบ๐ບນຣ໐၀ဝ", "UuOonhgqwLSunsououusooo")) if (!m.has(k)) m.set(k, v);
  return m;
})();
const DIGIT_ZEROS = [0x0660, 0x06f0, 0x0966, 0x09e6, 0x0a66, 0x0ae6, 0x0b66, 0x0be6, 0x0c66, 0x0ce6, 0x0d66, 0x0de6,
  0x0e50, 0x0ed0, 0x0f20, 0x1040, 0x1090, 0x17e0];
const CYR_FROM = "aceopxyiABCEHKMOPTXYIΑΒΓΕΗΚΜΟΠΡΤΧαο";
const CYR_MAP = zipMap(CYR_FROM, "асеорхуіАВСЕНКМОРТХУІАВГЕНКМОПРТХао");
const CYR_FROM_SET = new Set(Array.from(CYR_FROM));

function isLatinLetter(ch) {
  const u = ch.codePointAt(0);
  return isAsciiAlpha(u) || (isAlpha(ch) && inR(u, 0x00c0, 0x024f));
}
function latinLetters(text) {
  let n = 0;
  for (const ch of text) if (isLatinLetter(ch)) n++;
  return n;
}
function latinLookalike(ch, upper) {
  const h = HOMO.get(ch);
  if (h !== undefined) return h;
  switch (ch) {
    case "т": return upper ? "T" : "t";
    case "н": return upper ? "H" : "h";
    case "м": return upper ? "M" : "m";
    case "п": case "л": return "n";
    case "г": return "r";
    default: return null;
  }
}

/** Clean a reading of other scripts' lookalike characters (`page::clean_latin`). */
export function cleanLatin(text) {
  const cs = Array.from(text);
  const lat = cs.filter(isLatinLetter).length;
  const oth = cs.filter((c) => isAlpha(c) && !isLatinLetter(c)).length;
  const latinLine = lat > 0 && lat >= 2 * oth;
  const cyrLine = cs.filter((c) => inR(c.codePointAt(0), 0x400, 0x4ff)).length > lat;
  let out = "";
  // split() on every whitespace character; each separator becomes one space
  const words = [];
  let cur = [];
  for (const c of cs) {
    if (isWs(c)) {
      words.push(cur);
      cur = [];
    } else cur.push(c);
  }
  words.push(cur);
  words.forEach((word, k) => {
    if (k > 0) out += " ";
    const wl = word.filter(isLatinLetter).length;
    const wo = word.filter((c) => isAlpha(c) && !isLatinLetter(c)).length;
    const upper = word.filter(isUpper).length > word.filter(isLower).length;
    const allLookalike = wo > 0 && word.every((c) => !isAlpha(c) || isLatinLetter(c) ||
      (latinLookalike(c, upper) !== null && !inR(c.codePointAt(0), 0x370, 0x3ff)));
    const cg = word.filter((c) => inR(c.codePointAt(0), 0x370, 0x4ff)).length;
    const foldCg = cg > 0 && wl >= cg;
    const fold = latinLine && wo > 0 && (wl >= wo || allLookalike);
    const cyr = word.filter((c) => inR(c.codePointAt(0), 0x400, 0x4ff)).length;
    const lookalikesOnly = word.filter((c) => isAlpha(c) && !inR(c.codePointAt(0), 0x400, 0x4ff))
      .every((c) => CYR_FROM_SET.has(c));
    const toCyr = (cyr >= 2 && cyr > wl + (cg - cyr)) || (cyr >= 1 && cyrLine && lookalikesOnly);
    for (let c of word) {
      if (toCyr) c = CYR_MAP.get(c) ?? c;
      else if (fold || (foldCg && inR(c.codePointAt(0), 0x370, 0x4ff))) c = latinLookalike(c, upper) ?? c;
      const u = c.codePointAt(0);
      const z = DIGIT_ZEROS.find((zz) => u >= zz && u < zz + 10);
      if (z !== undefined && latinLine) out += String.fromCharCode(48 + (u - z));
      else out += c;
    }
  });
  return out;
}

/** A single short token ("1", "12", "x3"): faint candidates are only kept as one. */
export function isShortToken(text) {
  const t = Array.from(trimWs(text));
  return t.length > 0 && t.length <= 6 && !t.some(isWs) && t.some(isAlnum);
}

/** A detector false positive on a picture: no letter or digit, or 1-2 unsure characters. */
export function isJunk(text, conf) {
  let word = 0;
  for (const c of text) if (isAlnum(c)) word++;
  return word === 0 || (word <= 2 && conf < f32(0.8));
}

// ---------------------------------------------------------------------------------------------
// rec.rs

const H = 32;
const STRIDE = 4;
const MAX_W = 1280;
const BUCKET_W = 128;
const BATCH = 32;
const ROUTER_W = 512;
const ROUTER_BATCH = 64;

/** v / 127.5 - 1 in f32 (detector, router), and in f64 cast to f32 (recognizers). */
const DET_LUT = new Float32Array(256);
const REC_LUT = new Float32Array(256);
for (let v = 0; v < 256; v++) {
  DET_LUT[v] = f32(f32(v / 127.5) - 1);
  REC_LUT[v] = f32(v / 127.5 - 1.0);
}

function toF32(a) {
  return a instanceof Float32Array ? a : Float32Array.from(a, Number);
}

/** Grey line image at height H, width scaled with the aspect ratio (`Recognizer._prep`). */
function prep(cropImg, minW, maxW) {
  const g = toGray(cropImg);
  const nw = clamp(rint((g.w * H) / Math.max(g.h, 1)), minW, maxW);
  return g.h > H ? resizeArea(g, nw, H) : resizeLinear(g, nw, H);
}

/** Input pixels per output timestep, from one probe run (`Recognizer::new`). */
async function recStride(ctx, model) {
  let s = ctx.strides.get(model);
  if (s === undefined) {
    s = (async () => {
      const res = await ctx.infer(model, [1, 1, H, 64], new Float32Array(H * 64).fill(1), ["ids"]);
      const steps = res.ids.shape[1];
      return Math.floor(64 / Math.max(steps, 1));
    })();
    ctx.strides.set(model, s);
  }
  return s;
}

/** (text, confidence) per crop (`Recognizer::read`): width-sorted batches of 32 padded to 128 px. */
async function recRead(ctx, group, crops) {
  if (!crops.length) return [];
  const { model, chars, visualRtl } = ctx.recs[group];
  const stride = await recStride(ctx, model);
  const lines = crops.map((c) => prep(c, STRIDE * 2, MAX_W));
  const order = lines.map((_, i) => i).sort((a, b) => lines[a].w - lines[b].w);
  const out = new Array(lines.length);
  for (let s0 = 0; s0 < order.length; s0 += BATCH) {
    const idx = order.slice(s0, s0 + BATCH);
    let maxw = 1;
    for (const i of idx) maxw = Math.max(maxw, lines[i].w);
    const w = Math.ceil(maxw / BUCKET_W) * BUCKET_W;
    const x = new Float32Array(idx.length * H * w).fill(1);
    idx.forEach((i, j) => {
      const l = lines[i];
      for (let y = 0; y < H; y++) {
        const ob = (j * H + y) * w, ib = y * l.w;
        for (let xx = 0; xx < l.w; xx++) x[ob + xx] = REC_LUT[l.data[ib + xx]];
      }
    });
    const res = await ctx.infer(model, [idx.length, 1, H, w], x, ["ids", "conf"]);
    const ids = res.ids.data, conf = res.conf.data;
    const tAll = res.ids.shape[1];
    idx.forEach((i, j) => {
      const t = Math.min(Math.floor(lines[i].w / stride), tAll);
      let text = "";
      const cs = [];
      let prev = 0;
      for (let k = 0; k < t; k++) {
        const c = Number(ids[j * tAll + k]);
        if (c !== 0 && c !== prev) {
          const s = chars[c - 1];
          if (s !== undefined) text += s;
          cs.push(conf[j * tAll + k]);
        }
        prev = c;
      }
      if (visualRtl && Array.from(text).some(isRtlChar)) text = visualToLogical(text);
      let mean = 0;
      if (cs.length) {
        let sum = 0;
        for (const v of cs) sum += v;
        mean = f32(sum / cs.length);
      }
      out[i] = [text, mean];
    });
  }
  return out;
}

function isRtlChar(ch) {
  const u = ch.codePointAt(0);
  return inR(u, 0x0590, 0x08ff) || inR(u, 0xfb1d, 0xfeff);
}

function routerPrep(cropImg, x, off) {
  const g = prep(cropImg, 8, Number.MAX_SAFE_INTEGER);
  for (let y = 0; y < H; y++) {
    const ib = y * g.w, ob = off + y * ROUTER_W;
    for (let xx = 0; xx < ROUTER_W; xx++) x[ob + xx] = DET_LUT[g.data[ib + (xx % g.w)]];
  }
}

function softmaxF(z) {
  let m = -Infinity;
  for (const v of z) m = fmaxF(m, v);
  const e = z.map((v) => f32(Math.exp(f32(v - m))));
  let s = 0;
  for (const v of e) s = f32(s + v);
  return e.map((v) => f32(v / s));
}

/** Page-level routing (`Router::route`): group per crop, turn the page?, each group's share. */
async function route(ctx, crops) {
  const OVERRIDE = f32(0.9), MIN_ASPECT = 4.0, FLIP_MIN_LINES = 3, FLIP_VOTE = 0.7, OTHER_SHARE = 0.65;
  if (!crops.length) return { groups: [], flip: false, share: [0, 0, 0, 0] };
  const gp = [], fp = [];
  for (let s0 = 0; s0 < crops.length; s0 += ROUTER_BATCH) {
    const part = crops.slice(s0, s0 + ROUTER_BATCH);
    const x = new Float32Array(part.length * H * ROUTER_W);
    part.forEach((c, j) => routerPrep(c, x, j * H * ROUTER_W));
    const res = await ctx.infer("router", [part.length, 1, H, ROUTER_W], x, ["group", "flip"]);
    const g = toF32(res.group.data), f = toF32(res.flip.data);
    for (let j = 0; j < part.length; j++) {
      gp.push(softmaxF(Array.from(g.subarray(j * 4, j * 4 + 4))));
      fp.push(softmaxF(Array.from(f.subarray(j * 2, j * 2 + 2)))[1]);
    }
  }
  const w = crops.map((c) => c.w);
  let wsum = 0;
  for (const v of w) wsum += v;
  const share = [0, 0, 0, 0];
  gp.forEach((p, i) => {
    for (let k = 0; k < 4; k++) share[k] += p[k] * w[i];
  });
  for (let k = 0; k < 4; k++) share[k] /= wsum;
  let dom = argmax(share);
  if (dom !== 0 && share[dom] <= OTHER_SHARE) dom = 0;
  const groups = gp.map((p, i) => {
    const own = argmax(p);
    let pmax = -Infinity;
    for (const v of p) pmax = fmaxF(pmax, v);
    return own !== dom && pmax > OVERRIDE && crops[i].w >= MIN_ASPECT * crops[i].h ? own : dom;
  });
  let vote = 0;
  fp.forEach((f, i) => {
    vote += f * w[i];
  });
  const flip = crops.length >= FLIP_MIN_LINES && vote > FLIP_VOTE * wsum;
  return { groups, flip, share };
}

/** np.argmax: the first maximum. */
function argmax(v) {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
  return best;
}

// ---------------------------------------------------------------------------------------------
// mod.rs

/** np.mean over Python floats. */
function mean(vals) {
  if (!vals.length) return NaN;
  let s = 0;
  for (const v of vals) s += v;
  return s / vals.length;
}

/** Readings per box, whether the page was read upside down, and the page's script group. */
async function readPage(ctx, img, boxes) {
  let crops = boxes.map((b) => crop(img, b));
  const r = await route(ctx, crops);
  const groups = r.groups.slice();
  const { flip, share } = r;
  if (flip) crops = crops.map((c) => rot90(c, 2));
  const n = crops.length;
  const out = Array.from({ length: n }, () => ["", 0]);
  const subset = (idx) => idx.map((i) => crops[i]);
  const where = (pred) => {
    const idx = [];
    for (let i = 0; i < n; i++) if (pred(i)) idx.push(i);
    return idx;
  };
  for (let gi = 0; gi < GROUPS.length; gi++) {
    const idx = where((i) => groups[i] === gi);
    if (!idx.length) continue;
    const res = await recRead(ctx, gi, subset(idx));
    idx.forEach((i, k) => (out[i] = res[k]));
  }
  let dom = majority(groups);
  const KEEP_CONF = f32(0.75);
  if (ctx.opts.rescue && dom === 0) {
    const RESCUE_CONF = 0.75, RESCUE_MARGIN = 0.15, RESCUE_SHARE = 0.2;
    const lat = where((i) => groups[i] === dom);
    if (lat.length && share.slice(1).some((s) => s >= RESCUE_SHARE)) {
      const latConf = mean(lat.map((i) => out[i][1]));
      if (latConf < RESCUE_CONF) {
        const sample = lat.slice(0, 40);
        let best = null, bestConf = latConf + RESCUE_MARGIN;
        for (let gi = 1; gi < GROUPS.length; gi++) {
          if (!(share[gi] >= RESCUE_SHARE)) continue;
          const c = mean((await recRead(ctx, gi, subset(sample))).map((x) => x[1]));
          if (c >= bestConf) {
            best = gi;
            bestConf = c;
          }
        }
        if (best !== null) {
          const res = await recRead(ctx, best, subset(lat));
          lat.forEach((i, k) => {
            // per line: a bilingual page keeps its confident Latin lines
            if (!(out[i][1] >= KEEP_CONF && out[i][1] >= res[k][1])) {
              out[i] = res[k];
              groups[i] = best;
            }
          });
          dom = best;
        }
      }
    }
    if (dom === 0) {
      const LINE_MARGIN = f32(0.1), LINE_SHARE = 0.15, LINE_CONF = f32(0.8);
      for (let gi = 1; gi < GROUPS.length; gi++) {
        if (!(share[gi] >= LINE_SHARE)) continue;
        const weak = where((i) => groups[i] === 0);
        if (!weak.length) break;
        const res = await recRead(ctx, gi, subset(weak));
        weak.forEach((i, k) => {
          const [t, c] = res[k];
          const sc = scriptChars(gi, t);
          if (c >= f32(out[i][1] + LINE_MARGIN) && c >= LINE_CONF && sc >= 3 && sc > latinLetters(t)) {
            out[i] = res[k];
            groups[i] = gi;
          }
        });
      }
    }
  }
  // lines routed away from the page's script are read with the page's recognizer too
  const odd = where((i) => groups[i] !== dom);
  if (odd.length) {
    const res = await recRead(ctx, dom, subset(odd));
    odd.forEach((i, k) => {
      const [t, c] = out[i];
      const sc = scriptChars(groups[i], t);
      const routedOk = c >= KEEP_CONF && sc >= 2 && sc > latinLetters(t);
      const replace = !routedOk && (c < f32(res[k][1] + f32(0.1)) || sc < 3);
      if (replace) out[i] = res[k];
    });
  }
  for (const o of out) o[0] = cleanLatin(o[0]);
  return { reads: out, flip, dom };
}

/** Recognizer strides per `infer` (one probe run per model, as `Recognizer::new` does at load). */
const STRIDES = new WeakMap();
function stridesFor(infer) {
  let m = STRIDES.get(infer);
  if (!m) STRIDES.set(infer, (m = new Map()));
  return m;
}

function makeContext(infer, charsets, opts) {
  const o = { ...DEFAULT_OPTIONS, ...(opts || {}) };
  const meta = o.detMeta || {};
  const num = (v) => (typeof v === "number" ? v : null);
  const parseRec = (g) => {
    const c = charsets?.[g];
    if (!c) return null;
    const chars = Array.isArray(c) ? c : c.chars;
    const visualRtl = Array.isArray(c) ? g === "rtl" : (c.visual_rtl ?? c.visualRtl ?? g === "rtl");
    return { model: `rec_${g}`, chars, visualRtl: !!visualRtl };
  };
  const latin = parseRec("latin");
  if (!latin) throw new Error("ocrPage: charsets.latin (rec_latin.charset.json) is required");
  const recs = GROUPS.map((g) => parseRec(g) ?? latin);
  return {
    infer,
    opts: o,
    detThresh: f32(o.detThresh ?? num(meta.thresh) ?? SHIPPED_DET_THRESH),
    detMerge: f32(o.detMerge ?? num(meta.merge) ?? DET_MERGE),
    recs,
    strides: stridesFor(infer),
  };
}

/**
 * OCR one page image (`OcriaEngine::run`). `img` is {data: Uint8Array, w, h, ch} with ch 1 (grey)
 * or 3 (RGB), row-major. Returns LiteParse's `OcrResult`s in page pixels:
 * {text, bbox: [x1, y1, x2, y2] (truncated), confidence, polygon (reading-order quad), words: [],
 * rotation: null, faint}.
 */
export async function ocrPage(img, infer, charsets, opts) {
  const ctx = makeContext(infer, charsets, opts);
  const { w, h } = img;
  let src = img.data;
  if (src.length === w * h * 4) {
    // RGBA (canvas ImageData): drop alpha
    const d = new Uint8Array(w * h * 3);
    for (let i = 0, j = 0; j < d.length; i += 4, j += 3) {
      d[j] = src[i];
      d[j + 1] = src[i + 1];
      d[j + 2] = src[i + 2];
    }
    src = d;
  }
  const ch = src.length === w * h ? 1 : 3;
  let img0 = { w, h, ch, data: src };
  if (ch === 1) {
    // the server decodes every request as RGB
    const d = new Uint8Array(w * h * 3);
    for (let i = 0, j = 0; i < src.length; i++, j += 3) d[j] = d[j + 1] = d[j + 2] = src[i];
    img0 = { w, h, ch: 3, data: d };
  }
  let { boxes, faint } = await detect(ctx, img0, true);
  let sideways = false;
  let pimg = img0;
  if (ctx.opts.orient && looksSideways(boxes)) {
    // maybe a page scanned on its side: detect again a quarter turn counter-clockwise
    const turned = rot90(img0, 1);
    const tboxes = (await detect(ctx, turned, false)).boxes;
    if (longLines(tboxes) > 2 * longLines(boxes) + 5) {
      pimg = turned;
      boxes = tboxes;
      faint = [];
      sideways = true;
    }
  }
  if (ctx.opts.gutter) boxes = splitGutters(pimg, boxes);
  let reads = [], flip = false, dom = 0;
  if (boxes.length) ({ reads, flip, dom } = await readPage(ctx, pimg, boxes));
  const nLines = boxes.length;
  boxes = boxes.slice();
  if (faint.length && !flip) {
    // with the page's recognizer: a lone glyph is too short for the router
    const crops = faint.map((b) => crop(pimg, b));
    const res = await recRead(ctx, dom, crops);
    faint.forEach((quad, k) => {
      const text = cleanLatin(res[k][0]);
      const conf = res[k][1];
      if (conf >= FAINT_CONF && isShortToken(text)) {
        boxes.push(quad);
        reads.push([text, conf]);
      }
    });
  }
  const out = [];
  for (let k = 0; k < Math.min(boxes.length, reads.length); k++) {
    const quad = boxes[k];
    const [text, conf] = reads[k];
    if (trimWs(text) === "" || (ctx.opts.junk && isJunk(text, conf))) continue;
    const pts = ctx.opts.orient ? readingFrame(quad, flip, sideways, img0.w) : quad.map((p) => [p[0], p[1]]);
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const p of pts) {
      x1 = fminF(x1, p[0]);
      y1 = fminF(y1, p[1]);
      x2 = fmaxF(x2, p[0]);
      y2 = fmaxF(y2, p[1]);
    }
    out.push({
      text,
      // the server sends int() of each extent
      bbox: [Math.trunc(x1), Math.trunc(y1), Math.trunc(x2), Math.trunc(y2)],
      confidence: conf,
      polygon: pts,
      words: [],
      rotation: null,
      faint: k >= nLines,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Unicode bidi (unicode-bidi 0.3.18 `BidiInfo::new(s, Some(Level::rtl()))` + `reorder_line`) for
// the RTL recognizer's visual-order readings: one paragraph, RTL base level, rules W1-W7, N0
// (bracket pairs), N1-N2, I1-I2, L1-L2; no mirroring. Explicit embeddings, overrides and isolates
// (which no recognizer charset contains) are treated as removed by X9.

// Bidi_Class runs that are not L, from Python's unicodedata 15.1 (test/gen_bidi.py):
// "<gap from previous run, hex><class><length, hex>" separated by spaces.
const BIDI_RUNS = 
  "0Z9 0G1 0B1 0G1 0W1 0B1 0Ze 0B3 0G1 0W1 0O2 0T3 0O5 0S1 0C1 0S1 0C2 0Ea 0C1 0O6 1aO6 1aO4 0Z6 0B1 0Z1a 0C1 " +
  "0O1 0T4 0O4 1O2 0Z1 0O2 0T2 0E2 0O1 1O3 0E1 1O5 17O1 1fO1 1c1O2 7Oe 2Oe 5O9 1O11 0M70 4O2 8O1 5O2 1O1 6eO1 " +
  "8cM7 100O1 2O2 0T1 1M2d 0R1 0M1 0R1 0M2 0R1 0M2 0R1 0M1 8R1b 4R6 bN6 0O2 0A1 0T2 0A1 0C1 0A1 0O2 0Mb 0A30 " +
  "0M15 0Na 0T1 0N2 0A3 0M1 0A65 0M7 0N1 0O1 0M6 0A2 0M2 0O1 0M4 0A2 0Ea 0A14 1A2 0M1 0A1e 0M1b 2A59 0Mb 0A1 " +
  "eR2b 0M9 0R2 0O4 0R1 2M1 0R18 0M4 0R1 0M9 0R1 0M3 0R1 0M5 2Rf 1R19 0M3 2R1 1Ab 5A1f 1N2 6M8 0A2a 0M18 0N1 " +
  "0M20 37M1 1M1 4M8 4M1 3M7 aM2 1dM1 3aM1 4M4 8M1 14M2 eT2 7T1 2M1 2M2 39M1 4M2 4M2 2M3 3M1 1eM2 3M1 bM2 39M1 " +
  "4M5 1M2 4M1 14M2 dT1 8M6 1M1 3aM1 2M1 1M4 8M1 7M2 bM2 1eM1 3dM1 cM1 25O6 0T1 0O1 5M1 3M1 37M1 1M3 5M3 1M4 " +
  "7M2 bM2 14O7 2M1 3aM1 fM2 14M2 1cM2 39M2 4M4 8M1 14M2 1dM1 48M1 7M3 1M1 5aM1 2M7 4T1 7M8 62M1 2M9 bM7 49M2 " +
  "1bM1 1M1 1M1 0O4 33Me 1M5 1M2 5Mb 1M24 9M1 66M4 1M6 1M2 2M2 19M2 4M3 10M4 dM1 2M2 6M1 fM1 2bfM3 30Oa 66O1 " +
  "27fW1 1aO2 75M3 1dM2 1eM2 1eM2 40M2 1M7 8M1 2Mb 7T1 1M1 12Oa 6Ob 0M3 0Z1 0M1 75M2 22M1 76M3 4M2 9M1 6M3 4O1 " +
  "3O2 98O22 17M2 2M1 3aM1 1M7 1M1 1M1 2M8 6Ma 2M1 30M1f 31M4 30M1 1M5 1M1 5M1 28M9 cM2 20M4 2M2 1M3 38M1 1M2 " +
  "3M1 1M3 3aM8 2M2 98M3 1Md 1M7 4M1 6M1 3M2 c6M40 1bdO1 1O3 bO3 dO3 dO3 dO2 1Wb 0Z3 1R1 0O18 0W1 0B1 0P1 0Q1 " +
  "0U1 0V1 0X1 0C1 0T5 0Of 0C1 0O1a 0W1 0Z5 1H1 0I1 0F1 0D1 0Z6 0E1 3E6 0S2 0O3 1Ea 0S2 0O3 11T21 fM21 fO2 1O4 " +
  "1O2 aO1 1O3 5O6 1O1 1O1 1O1 4T1 bO2 4O5 5O4 2O10 29O3 4O82 0S1 0T1 0O122 45O1a 1O91 19Ob 15O28 0E14 4eO1c2 " +
  "1O153 100O274 2O20 1O69 e5O6 4M3 7O7 7fM1 60M20 0O5e 22O1a 1O59 cOd6 1aO10 0W1 0O4 3O19 9M4 2O1 5O2 5O3 59M2 " +
  "0O2 3O1 5aO1 c4O24 bO1 2dO2 31O10 1cO3 32Of cO4 a7O4 63O2 1fO1 19c0O40 5690O37 146O3 5fM4 0O1 0Ma 0O2 1eM2 " +
  "50M2 eO22 66O1 79M1 3M1 4M1 19M2 1O4 0M1 bT2 3aO4 4cM2 1aM12 dM1 26M8 19Mb 2eM3 30M1 2M4 2M2 27M1 43M6 2M2 " +
  "2M2 cM1 8M1 2fM1 33M1 1M3 2M2 5M2 1M1 2aM2 8M1 73O2 79M1 2M1 4M1 4f2fR1 0M1 0Ra 0S1 0Rd 1R5 1R1 1R2 1R2 1Ra " +
  "0A73 10A16b 0O12 0A40 2A36 7O1 20Ad 0O3 0M10 0Oa 6M10 0O20 0C1 0O1 0C1 1O1 0C1 0O9 0T1 0O2 0S2 0O3 1O1 0T2 " +
  "0O1 4A5 1A87 2Z1 1O2 0T3 0O5 0S1 0C1 0S1 0C2 0Ea 0C1 0O6 1aO6 1aOb 7aT2 0O3 0T2 1O7 aO5 103O1 3eO4d 3Od 3O1 " +
  "5cM1 e2M1 0E1b 7aM5 485R6 2R1 1R2c 1R2 3R1 2R17 1R48 8R9 30R13 1R2 5R21 3O1 0R1a 5R1 40R38 4R14 2R2f 0M3 1M2 " +
  "5M4 0R4 1R3 1R1d 2M3 4M1 0R9 7R9 7R40 20R25 0M2 4Rc 9R36 3O7 0R16 2R1b 5R1a 7R4 cR7 50R49 37R33 dR33 7R6 " +
  "0A24 0M4 8Na 126N1f 1R2a 1M2 0R1 2R2 4bM3 0R28 8A16 0Mb 0A9 16R12 0M4 0R4 26R1c 14R17 aM1 36Mf bO14 aM1 2M2 " +
  "aM3 31M4 2M2 7M1 3dM3 24M5 1M8 3eM1 cM2 34M9 aM4 2M1 5fM3 2M1 1M2 6M1 2M1 9dM1 3M8 15M2 39M2 3M1 25M7 3M5 " +
  "c3M8 2M3 1M1 17M1 54M6 1M1 4M2 1M2 eeM4 6M2 1M2 1bM2 55M8 2M1 1M2 1fOd 3eM1 1M1 2M6 1M1 65M3 2M4 1M5 103M9 " +
  "1M2 100M2 1M1 4M1 90M4 2M2 4M1 20M6 2M2 28M6 2M4 8M1 9M6 2M3 2eMd 1M2 196M7 1M6 54M16 2M7 1M2 1M2 7aM6 3M1 " +
  "1M2 1M7 1M1 48M2 3M1 1M1 15bM2 bM2 34M5 5M1 1M1 92O8 0T4 0O11 144eM1 6Mf 369aM5 3bM7 418M1 3fM4 4fO1 1M1 " +
  "4cb8M2 1Z4 125cM2e 2M17 220M3 9Z8 0M8 2M7 1eM4 3bO2 15O42 0M3 0O1 baO57 384O1 39O1 39O1 39O1 39O1 aE32 " +
  "200M37 4M32 8M1 eM1 16M5 1Mf 550M7 1M11 2M7 1M2 1M5 64M1 a0M7 177M1 3dM4 fT1 1ecM4 310Rc5 2R9 0M7 29R44 0M7 " +
  "0R1 4Ra 4R2 311A44 4cA3d c2A4 1A1b 1A2 1A1 2A1 1Aa 1A4 1A1 1A1 6A1 4A1 1A1 1A1 1A3 1A2 1A1 2A1 1A1 1A1 1A1 " +
  "1A1 1A2 1A1 2A4 1A7 1A4 1A4 1A1 1Aa 1A11 5A3 1A5 1A11 34O2 10eO2c 4O64 cOf 2Of 1Of 1O25 aEb 0O5 1fO1 3aO6 " +
  "3dO1 b2O6 9aO3d8 4O11 3Od 3O77 4O5f 6Oc 4O1 fOc 4O38 8Oa 6O28 8O1e 2O2 4eO154 cOe 2Od 3O9 7O2e 1O7 8Oe 4O9 " +
  "7O9 7O93 1O37 25Ea c0407Z1 1eZ60 80Mf0";
const BIDI_CODE = { R: "R", A: "AL", E: "EN", S: "ES", T: "ET", N: "AN", C: "CS", M: "NSM", Z: "BN", B: "B",
  G: "S", W: "WS", O: "ON", P: "LRE", Q: "RLE", U: "PDF", V: "LRO", X: "RLO", H: "LRI", I: "RLI", F: "FSI", D: "PDI" };
let BIDI_STARTS = null, BIDI_ENDS = null, BIDI_CLS = null;
function bidiClass(u) {
  if (!BIDI_STARTS) {
    const runs = BIDI_RUNS.split(" ");
    BIDI_STARTS = new Int32Array(runs.length);
    BIDI_ENDS = new Int32Array(runs.length);
    BIDI_CLS = new Array(runs.length);
    let pos = 0;
    runs.forEach((r, i) => {
      const m = /^([0-9a-f]+)([A-Z])([0-9a-f]+)$/.exec(r);
      const s = pos + parseInt(m[1], 16), n = parseInt(m[3], 16);
      BIDI_STARTS[i] = s;
      BIDI_ENDS[i] = s + n - 1;
      BIDI_CLS[i] = BIDI_CODE[m[2]];
      pos = s + n;
    });
  }
  let lo = 0, hi = BIDI_STARTS.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (u < BIDI_STARTS[mid]) hi = mid - 1;
    else if (u > BIDI_ENDS[mid]) lo = mid + 1;
    else return BIDI_CLS[mid];
  }
  return "L";
}

/** Bidi_Paired_Bracket pairs (BidiBrackets.txt), opening -> closing. */
const BRACKETS = (() => {
  const pairs = [[0x28, 0x29], [0x5b, 0x5d], [0x7b, 0x7d], [0xf3a, 0xf3b], [0xf3c, 0xf3d], [0x169b, 0x169c],
    [0x2045, 0x2046], [0x207d, 0x207e], [0x208d, 0x208e], [0x2308, 0x2309], [0x230a, 0x230b], [0x2329, 0x232a],
    [0x2768, 0x2769], [0x276a, 0x276b], [0x276c, 0x276d], [0x276e, 0x276f], [0x2770, 0x2771], [0x2772, 0x2773],
    [0x2774, 0x2775], [0x27c5, 0x27c6], [0x27e6, 0x27e7], [0x27e8, 0x27e9], [0x27ea, 0x27eb], [0x27ec, 0x27ed],
    [0x27ee, 0x27ef], [0x2983, 0x2984], [0x2985, 0x2986], [0x2987, 0x2988], [0x2989, 0x298a], [0x298b, 0x298c],
    [0x298d, 0x2990], [0x298f, 0x298e], [0x2991, 0x2992], [0x2993, 0x2994], [0x2995, 0x2996], [0x2997, 0x2998],
    [0x29d8, 0x29d9], [0x29da, 0x29db], [0x29fc, 0x29fd], [0x2e22, 0x2e23], [0x2e24, 0x2e25], [0x2e26, 0x2e27],
    [0x2e28, 0x2e29], [0x2e55, 0x2e56], [0x2e57, 0x2e58], [0x2e59, 0x2e5a], [0x2e5b, 0x2e5c], [0x3008, 0x3009],
    [0x300a, 0x300b], [0x300c, 0x300d], [0x300e, 0x300f], [0x3010, 0x3011], [0x3014, 0x3015], [0x3016, 0x3017],
    [0x3018, 0x3019], [0x301a, 0x301b], [0xfe59, 0xfe5a], [0xfe5b, 0xfe5c], [0xfe5d, 0xfe5e], [0xff08, 0xff09],
    [0xff3b, 0xff3d], [0xff5b, 0xff5d], [0xff5f, 0xff60], [0xff62, 0xff63]];
  const open = new Map(), close = new Map();
  for (const [o, c] of pairs) {
    open.set(o, c);
    close.set(c, o);
  }
  return { open, close };
})();
/** canonical equivalents for bracket matching (U+2329/U+232A ~ U+3008/U+3009) */
const canonBracket = (u) => (u === 0x2329 ? 0x3008 : u === 0x232a ? 0x3009 : u);

const REMOVED = new Set(["BN", "LRE", "RLE", "PDF", "LRO", "RLO", "LRI", "RLI", "FSI", "PDI"]);
const NI = new Set(["B", "S", "WS", "ON"]);

/** Reorder a visual-order reading into logical order with an RTL paragraph level. */
export function visualToLogical(s) {
  const cps = Array.from(s);
  const n = cps.length;
  const orig = cps.map((c) => bidiClass(c.codePointAt(0)));
  const para = 1;
  const levels = new Array(n).fill(para);
  const seq = [];
  for (let i = 0; i < n; i++) if (!REMOVED.has(orig[i])) seq.push(i);
  const m = seq.length;
  const t = seq.map((i) => orig[i]);
  const sos = "R", eos = "R", e = "R";
  // W1
  for (let k = 0; k < m; k++) if (t[k] === "NSM") t[k] = k === 0 ? sos : t[k - 1];
  // W2
  let strong = sos;
  for (let k = 0; k < m; k++) {
    if (t[k] === "R" || t[k] === "L" || t[k] === "AL") strong = t[k];
    else if (t[k] === "EN" && strong === "AL") t[k] = "AN";
  }
  // W3
  for (let k = 0; k < m; k++) if (t[k] === "AL") t[k] = "R";
  // W4
  for (let k = 1; k + 1 < m; k++) {
    if (t[k] === "ES" && t[k - 1] === "EN" && t[k + 1] === "EN") t[k] = "EN";
    else if (t[k] === "CS" && t[k - 1] === t[k + 1] && (t[k - 1] === "EN" || t[k - 1] === "AN")) t[k] = t[k - 1];
  }
  // W5
  for (let k = 0; k < m;) {
    if (t[k] !== "ET") {
      k++;
      continue;
    }
    let j = k;
    while (j < m && t[j] === "ET") j++;
    if ((k > 0 && t[k - 1] === "EN") || (j < m && t[j] === "EN")) for (let x = k; x < j; x++) t[x] = "EN";
    k = j;
  }
  // W6
  for (let k = 0; k < m; k++) if (t[k] === "ES" || t[k] === "ET" || t[k] === "CS") t[k] = "ON";
  // W7
  strong = sos;
  for (let k = 0; k < m; k++) {
    if (t[k] === "L" || t[k] === "R") strong = t[k];
    else if (t[k] === "EN" && strong === "L") t[k] = "L";
  }
  // N0: bracket pairs (BD16, a stack of 63)
  const strongOf = (c) => (c === "L" ? "L" : c === "R" || c === "EN" || c === "AN" ? "R" : null);
  const pairs = [];
  const stack = [];
  for (let k = 0; k < m; k++) {
    if (t[k] !== "ON") continue;
    const u = canonBracket(cps[seq[k]].codePointAt(0));
    if (BRACKETS.open.has(u)) {
      if (stack.length === 63) break;
      stack.push([canonBracket(BRACKETS.open.get(u)), k]);
    } else if (BRACKETS.close.has(u)) {
      for (let sp = stack.length - 1; sp >= 0; sp--) {
        if (stack[sp][0] === u) {
          pairs.push([stack[sp][1], k]);
          stack.length = sp;
          break;
        }
      }
    }
  }
  pairs.sort((a, b) => a[0] - b[0]);
  for (const [o, c] of pairs) {
    let foundE = false, foundOpp = false;
    for (let k = o + 1; k < c; k++) {
      const st = strongOf(t[k]);
      if (st === e) {
        foundE = true;
        break;
      }
      if (st && st !== e) foundOpp = true;
    }
    let set = null;
    if (foundE) set = e;
    else if (foundOpp) {
      let ctxDir = sos;
      for (let k = o - 1; k >= 0; k--) {
        const st = strongOf(t[k]);
        if (st) {
          ctxDir = st;
          break;
        }
      }
      set = ctxDir !== e ? ctxDir : e;
    }
    if (set) {
      t[o] = t[c] = set;
      for (const b of [o, c]) for (let k = b + 1; k < m && orig[seq[k]] === "NSM"; k++) t[k] = set;
    }
  }
  // N1, N2
  for (let k = 0; k < m;) {
    if (!NI.has(t[k])) {
      k++;
      continue;
    }
    let j = k;
    while (j < m && NI.has(t[j])) j++;
    const before = k === 0 ? sos : strongOf(t[k - 1]) ?? e;
    const after = j === m ? eos : strongOf(t[j]) ?? e;
    const dir = before === after ? before : e;
    for (let x = k; x < j; x++) t[x] = dir;
    k = j;
  }
  // I1, I2 (odd base level)
  for (let k = 0; k < m; k++) levels[seq[k]] = t[k] === "R" ? para : para + 1;
  // removed characters take the previous character's level
  for (let i = 0; i < n; i++) if (REMOVED.has(orig[i])) levels[i] = i > 0 ? levels[i - 1] : para;
  // L1: separators, and whitespace before them or at the end, to the paragraph level
  const wsLike = (c) => c === "WS" || REMOVED.has(c);
  for (let i = n - 1, trailing = true; i >= 0; i--) {
    if (orig[i] === "S" || orig[i] === "B") {
      levels[i] = para;
      trailing = true;
    } else if (trailing && wsLike(orig[i])) levels[i] = para;
    else trailing = false;
  }
  // L2
  let maxL = 0;
  for (const l of levels) maxL = Math.max(maxL, l);
  const order = cps.map((_, i) => i);
  for (let lev = maxL; lev >= 1; lev--) {
    for (let i = 0; i < n;) {
      if (levels[order[i]] < lev) {
        i++;
        continue;
      }
      let j = i;
      while (j < n && levels[order[j]] >= lev) j++;
      const seg = order.slice(i, j).reverse();
      for (let x = i; x < j; x++) order[x] = seg[x - i];
      i = j;
    }
  }
  return order.map((i) => cps[i]).join("");
}

// internals for tests
export const _internal = {
  rint, toGray, meanGray, rot90, resizeLinear, resizeArea, perspectiveTransform, warpPerspectiveCubic,
  bicubicTab, components4, minAreaRect, boxPoints, otsuHist, crop, splitGutters, readingFrame, prep,
  scriptChars, latinLetters, bidiClass, makeContext, detect, readPage, route, recRead, faintBoxes, linearTaps,
  cubicCoeffs,
};
