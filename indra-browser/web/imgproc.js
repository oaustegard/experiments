// The OpenCV image operations Indra's pipeline uses, on 8-bit grayscale images
// ({data: Uint8Array, w, h}), following OpenCV's own arithmetic where it is cheap to:
// INTER_AREA decimation by exact area weights (float32 accumulation, round half to even),
// INTER_LINEAR and INTER_AREA upscaling with 11-bit fixed-point coefficients, warpAffine
// in float32 as OpenCV 5 does, and connected components labelled
// in raster order. Results can still differ from cv2 by one grey level where OpenCV takes a
// vectorized or IPP path.

export const img = (w, h, data = new Uint8Array(w * h)) => ({ w, h, data });

// Python's round(): half to even
export function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
const sat8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
const f32 = Math.fround;

// cv2.resize(src, (dw, dh), interpolation) for INTER_AREA ("area") or INTER_LINEAR ("linear")
// whether resize() takes the area-decimation path (the one deskew.js also runs on the GPU)
export function isAreaDecimate(src, dw, dh, interp) {
  const sx = src.w / dw, sy = src.h / dh;
  if (interp !== "area" || sx < 1 || sy < 1 || (dw === src.w && dh === src.h)) return false;
  return !(Math.abs(sx - Math.round(sx)) < 1e-12 && Math.abs(sy - Math.round(sy)) < 1e-12);
}

export function resize(src, dw, dh, interp) {
  if (dw === src.w && dh === src.h) return img(dw, dh, src.data.slice());
  const sx = src.w / dw, sy = src.h / dh; // scale = source / destination
  if (interp === "area" && sx >= 1 && sy >= 1) {
    const ix = Math.round(sx), iy = Math.round(sy);
    if (Math.abs(sx - ix) < 1e-12 && Math.abs(sy - iy) < 1e-12) return areaFast(src, dw, dh, ix, iy);
    return areaDecimate(src, dw, dh, sx, sy);
  }
  return bilinear(src, dw, dh, interp === "area");
}

function areaFast(src, dw, dh, kx, ky) {
  const out = img(dw, dh), s = src.data, W = src.w;
  const area = kx * ky, inv = f32(1 / area);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      let sum = 0;
      for (let j = 0; j < ky; j++) {
        const r = (y * ky + j) * W + x * kx;
        for (let i = 0; i < kx; i++) sum += s[r + i];
      }
      out.data[y * dw + x] = kx === 2 && ky === 2 ? (sum + 2) >> 2 : sat8(pyRound(f32(sum * inv)));
    }
  }
  return out;
}

// computeResizeAreaTab: [{di, si, alpha}] for one axis
export function areaTab(ssize, dsize, scale) {
  const tab = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale, fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1), sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) tab.push([dx, sx1 - 1, f32((sx1 - fsx1) / cellWidth)]);
    for (let sx = sx1; sx < sx2; sx++) tab.push([dx, sx, f32(1.0 / cellWidth)]);
    if (fsx2 - sx2 > 1e-3) tab.push([dx, sx2, f32(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth)]);
  }
  return tab;
}

function areaDecimate(src, dw, dh, sx, sy) {
  const xt = areaTab(src.w, dw, sx), yt = areaTab(src.h, dh, sy);
  const out = img(dw, dh), s = src.data, W = src.w;
  const buf = new Float32Array(dw), sum = new Float32Array(dw);
  let prev = -1;
  for (const [dy, syi, beta] of yt) {
    buf.fill(0);
    const r = syi * W;
    for (const [dx, si, a] of xt) buf[dx] = f32(buf[dx] + f32(s[r + si] * a));
    if (dy !== prev) {
      if (prev >= 0) for (let dx = 0; dx < dw; dx++) out.data[prev * dw + dx] = sat8(pyRound(sum[dx]));
      for (let dx = 0; dx < dw; dx++) sum[dx] = f32(beta * buf[dx]);
      prev = dy;
    } else {
      for (let dx = 0; dx < dw; dx++) sum[dx] = f32(sum[dx] + f32(beta * buf[dx]));
    }
  }
  if (prev >= 0) for (let dx = 0; dx < dw; dx++) out.data[prev * dw + dx] = sat8(pyRound(sum[dx]));
  return out;
}

// separable linear interpolation, OpenCV's coordinate rules for INTER_LINEAR or INTER_AREA
// (upscaling), 11-bit fixed-point coefficients, scalar rounding
function coefs(ssize, dsize, areaMode) {
  const scale = ssize / dsize, inv = dsize / ssize;
  const idx = new Int32Array(dsize), a0 = new Int32Array(dsize), a1 = new Int32Array(dsize);
  for (let d = 0; d < dsize; d++) {
    let s, f;
    if (!areaMode) {
      f = f32((d + 0.5) * scale - 0.5);
      s = Math.floor(f);
      f = f32(f - s);
    } else {
      s = Math.floor(d * scale);
      f = f32(d + 1 - (s + 1) * inv);
      f = f <= 0 ? 0 : f32(f - Math.floor(f));
    }
    if (s < 0) { f = 0; s = 0; }
    if (s >= ssize - 1) { f = 0; s = ssize - 1; }
    idx[d] = s;
    a0[d] = pyRound(f32(1 - f) * 2048);
    a1[d] = pyRound(f * 2048);
  }
  return { idx, a0, a1 };
}

function bilinear(src, dw, dh, areaMode) {
  const X = coefs(src.w, dw, areaMode), Y = coefs(src.h, dh, areaMode);
  const s = src.data, W = src.w, out = img(dw, dh);
  const rows = new Map();
  const hrow = (sy) => {
    let r = rows.get(sy);
    if (!r) {
      r = new Int32Array(dw);
      const o = sy * W;
      for (let x = 0; x < dw; x++) {
        const i = X.idx[x];
        r[x] = s[o + i] * X.a0[x] + (i + 1 < W ? s[o + i + 1] : s[o + i]) * X.a1[x];
      }
      rows.set(sy, r);
      if (rows.size > 4) rows.delete(rows.keys().next().value);
    }
    return r;
  };
  for (let y = 0; y < dh; y++) {
    const sy = Y.idx[y];
    const r0 = hrow(sy), r1 = hrow(Math.min(sy + 1, src.h - 1));
    const b0 = Y.a0[y], b1 = Y.a1[y];
    for (let x = 0; x < dw; x++) {
      // VResizeLinear's vector path: ((S0 >> 4) * b0 >> 16) + ((S1 >> 4) * b1 >> 16), then (+ 2) >> 2
      const v = (((r0[x] >> 4) * b0) >> 16) + (((r1[x] >> 4) * b1) >> 16);
      out.data[y * dw + x] = sat8((v + 2) >> 2);
    }
  }
  return out;
}

// cv2.getRotationMatrix2D(center, angle, scale)
export function rotationMatrix(cx, cy, angle, scale = 1) {
  const a = (angle * Math.PI) / 180;
  const al = scale * Math.cos(a), be = scale * Math.sin(a);
  return [al, be, (1 - al) * cx - be * cy, -be, al, be * cx + (1 - al) * cy];
}

// cv2.warpAffine(src, M, (dw, dh), INTER_LINEAR, border): border "replicate" or a constant value.
// OpenCV 5's 8-bit linear warp maps each destination pixel through the inverse matrix in float32
// and interpolates in float32; this matches cv2 5.1 on all but ~0.01% of pixels (off by 1).
export function warpAffine(src, Mf, dw, dh, border = 0) {
  const D = Mf[0] * Mf[4] - Mf[1] * Mf[3];
  const id = D !== 0 ? 1 / D : 0;
  const a11 = Mf[4] * id, a22 = Mf[0] * id, a12 = -Mf[1] * id, a21 = -Mf[3] * id;
  const M = [a11, a12, -a11 * Mf[2] - a12 * Mf[5], a21, a22, -a21 * Mf[2] - a22 * Mf[5]].map(f32);
  const out = img(dw, dh), s = src.data, W = src.w, H = src.h;
  const rep = border === "replicate";
  const px = (x, y) => {
    if (x >= 0 && x < W && y >= 0 && y < H) return s[y * W + x];
    if (!rep) return border;
    return s[Math.min(Math.max(y, 0), H - 1) * W + Math.min(Math.max(x, 0), W - 1)];
  };
  for (let y = 0; y < dh; y++) {
    const by0 = f32(M[1] * y), by1 = f32(M[4] * y);
    for (let x = 0; x < dw; x++) {
      const sx = f32(f32(f32(M[0] * x) + by0) + M[2]), sy = f32(f32(f32(M[3] * x) + by1) + M[5]);
      const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = f32(sx - x0), fy = f32(sy - y0);
      const p00 = px(x0, y0), p01 = px(x0 + 1, y0), p10 = px(x0, y0 + 1), p11 = px(x0 + 1, y0 + 1);
      const top = f32(p00 + f32(fx * f32(p01 - p00))), bot = f32(p10 + f32(fx * f32(p11 - p10)));
      out.data[y * dw + x] = sat8(pyRound(f32(top + f32(fy * f32(bot - top)))));
    }
  }
  return out;
}

// np.rot90(a, k): k quarter turns counter-clockwise
export function rot90(src, k) {
  k = ((k % 4) + 4) % 4;
  if (k === 0) return img(src.w, src.h, src.data.slice());
  const { w, h, data } = src;
  if (k === 2) {
    const out = img(w, h);
    for (let i = 0; i < w * h; i++) out.data[i] = data[w * h - 1 - i];
    return out;
  }
  const out = img(h, w); // width h, height w
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      // k=1: out[w-1-x, y] = a[y, x]; k=3: out[x, h-1-y] = a[y, x]
      if (k === 1) out.data[(w - 1 - x) * h + y] = data[y * w + x];
      else out.data[x * h + (h - 1 - y)] = data[y * w + x];
    }
  return out;
}

// cv2.connectedComponentsWithStats(mask, connectivity): labels in raster order of each
// component's first pixel; stats [n][left, top, width, height, area] (row 0 = background)
export function components(mask, w, h, conn = 4) {
  const lab = new Int32Array(w * h).fill(-1);
  const stats = [[0, 0, w, h, 0]];
  const stack = new Int32Array(w * h);
  let n = 1;
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) { lab[i] = 0; stats[0][4]++; continue; }
    if (lab[i] >= 0) continue;
    let sp = 0, x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0;
    stack[sp++] = i;
    lab[i] = n;
    while (sp) {
      const p = stack[--sp], x = p % w, y = (p - x) / w;
      area++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if ((dx === 0 && dy === 0) || (conn === 4 && dx !== 0 && dy !== 0)) continue;
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const q = yy * w + xx;
          if (mask[q] && lab[q] < 0) { lab[q] = n; stack[sp++] = q; }
        }
    }
    stats.push([x0, y0, x1 - x0 + 1, y1 - y0 + 1, area]);
    n++;
  }
  return { n, lab, stats };
}
