// The OpenCV image operations Indra's pipeline uses, on 8-bit grayscale images
// ({data: Uint8Array, w, h}), following OpenCV's own arithmetic where it is cheap to:
// INTER_AREA decimation by exact area weights (float32 accumulation, round half to even),
// INTER_LINEAR and INTER_AREA upscaling with 11-bit fixed-point coefficients, warpAffine
// with 1/32-pixel coordinates and 15-bit bilinear weights, and connected components labelled
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
function areaTab(ssize, dsize, scale) {
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

// bilinear weights for the 32 x 32 sub-pixel positions, 15-bit, summing to 32768
const WTAB = (() => {
  const t = new Int32Array(32 * 32 * 4);
  for (let fy = 0; fy < 32; fy++)
    for (let fx = 0; fx < 32; fx++) {
      const vy = fy / 32, vx = fx / 32;
      const w = [f32((1 - vy) * (1 - vx)), f32((1 - vy) * vx), f32(vy * (1 - vx)), f32(vy * vx)].map((v) => pyRound(v * 32768));
      const sum = w[0] + w[1] + w[2] + w[3];
      if (sum !== 32768) {
        const diff = sum - 32768;
        let mi = 0, Mi = 0;
        for (let k = 1; k < 4; k++) { if (w[k] < w[mi]) mi = k; else if (w[k] > w[Mi]) Mi = k; }
        if (diff < 0) w[Mi] -= diff; else w[mi] -= diff;
      }
      t.set(w, (fy * 32 + fx) * 4);
    }
  return t;
})();

// cv2.warpAffine(src, M, (dw, dh), INTER_LINEAR, border): border "replicate" or a constant value
export function warpAffine(src, M, dw, dh, border = 0) {
  const out = img(dw, dh), s = src.data, W = src.w, H = src.h;
  const AB = 1024, rd = 16;
  const ad = new Int32Array(dw), bd = new Int32Array(dw);
  for (let x = 0; x < dw; x++) { ad[x] = pyRound(M[0] * x * AB); bd[x] = pyRound(M[3] * x * AB); }
  const rep = border === "replicate";
  const px = (x, y) => {
    if (x >= 0 && x < W && y >= 0 && y < H) return s[y * W + x];
    if (!rep) return border;
    return s[Math.min(Math.max(y, 0), H - 1) * W + Math.min(Math.max(x, 0), W - 1)];
  };
  for (let y = 0; y < dh; y++) {
    const X0 = pyRound((M[1] * y + M[2]) * AB) + rd, Y0 = pyRound((M[4] * y + M[5]) * AB) + rd;
    for (let x = 0; x < dw; x++) {
      const Xf = (X0 + ad[x]) >> 5, Yf = (Y0 + bd[x]) >> 5;
      const ix = Xf >> 5, iy = Yf >> 5, k = ((Yf & 31) * 32 + (Xf & 31)) * 4;
      const v = px(ix, iy) * WTAB[k] + px(ix + 1, iy) * WTAB[k + 1] + px(ix, iy + 1) * WTAB[k + 2] + px(ix + 1, iy + 1) * WTAB[k + 3];
      out.data[y * dw + x] = sat8((v + 16384) >> 15);
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
