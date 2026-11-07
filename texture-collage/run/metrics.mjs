// Metrics comparing a collage (Lab) with its input (Lab). All take interleaved Lab, W, H.
import { blur, describe, featureDist } from "../web/collage.js";

const Lch = (lab, N) => { const L = new Float32Array(N); for (let i = 0; i < N; i++) L[i] = lab[3 * i]; return L; };

// Mean CIE76 colour difference after a blur of radius r: colour fidelity at the scale of regions,
// blind to texture detail.
export function deltaEBlur(a, b, W, H, r) {
  const A = blur(a, W, H, 3, r), B = blur(b, W, H, 3, r), N = W * H;
  let s = 0;
  for (let i = 0; i < N; i++) s += Math.hypot(A[3 * i] - B[3 * i], A[3 * i + 1] - B[3 * i + 1], A[3 * i + 2] - B[3 * i + 2]);
  return s / N;
}

// SSIM on L with 7x7 box windows (L in 0..100, so C1 = (0.01*100)^2, C2 = (0.03*100)^2)
export function ssimL(a, b, W, H) {
  const N = W * H, x = Lch(a, N), y = Lch(b, N), m = new Float32Array(5 * N);
  for (let i = 0; i < N; i++) { m[5 * i] = x[i]; m[5 * i + 1] = y[i]; m[5 * i + 2] = x[i] * x[i]; m[5 * i + 3] = y[i] * y[i]; m[5 * i + 4] = x[i] * y[i]; }
  const s = box1(m, W, H, 5, 3), C1 = 1, C2 = 9;
  let tot = 0;
  for (let i = 0; i < N; i++) {
    const mx = s[5 * i], my = s[5 * i + 1], vx = s[5 * i + 2] - mx * mx, vy = s[5 * i + 3] - my * my, cxy = s[5 * i + 4] - mx * my;
    tot += ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2));
  }
  return tot / N;
}

function box1(img, W, H, C, r) {      // a single box pass per axis (a true box window)
  const a = Float32Array.from(img), b = new Float32Array(img.length), inv = 1 / (2 * r + 1);
  const pass = (src, dst, horiz) => {
    const len = horiz ? W : H, lines = horiz ? H : W, acc = new Float64Array(C);
    for (let l = 0; l < lines; l++) {
      const at = (k) => (horiz ? l * W + k : k * W + l);
      acc.fill(0);
      for (let k = -r; k <= r; k++) { const p = at(Math.min(len - 1, Math.max(0, k))) * C; for (let c = 0; c < C; c++) acc[c] += src[p + c]; }
      for (let k = 0; k < len; k++) {
        const p = at(k) * C;
        for (let c = 0; c < C; c++) dst[p + c] = acc[c] * inv;
        const q = at(Math.min(len - 1, k + r + 1)) * C, o = at(Math.max(0, k - r)) * C;
        for (let c = 0; c < C; c++) acc[c] += src[q + c] - src[o + c];
      }
    }
  };
  pass(a, b, true); pass(b, a, false);
  return a;
}

// GMSD (Xue et al. 2014) on L with Prewitt gradients; lower is better. c scaled from the 0..255
// convention (170) to L's 0..100 range. mask: optional Uint8Array restricting the pixels counted.
export function gmsd(a, b, W, H, mask = null) {
  const N = W * H, x = Lch(a, N), y = Lch(b, N), c = 170 * (100 / 255) ** 2;
  const gm = (L, i, xx, yy) => {
    let gx = 0, gy = 0;
    for (let d = -1; d <= 1; d++) {
      gx += L[(yy + d) * W + xx + 1] - L[(yy + d) * W + xx - 1];
      gy += L[(yy + 1) * W + xx + d] - L[(yy - 1) * W + xx + d];
    }
    return Math.hypot(gx / 3, gy / 3);
  };
  const v = [];
  for (let yy = 1; yy < H - 1; yy++) for (let xx = 1; xx < W - 1; xx++) {
    const i = yy * W + xx;
    if (mask && !mask[i]) continue;
    const p = gm(x, i, xx, yy), q = gm(y, i, xx, yy);
    v.push((2 * p * q + c) / (p * p + q * q + c));
  }
  const m = v.reduce((s, t) => s + t, 0) / v.length;
  return Math.sqrt(v.reduce((s, t) => s + (t - m) ** 2, 0) / v.length);
}

// Pixels within d px of a region boundary.
export function boundaryBand(labels, W, H, d) {
  const N = W * H, edge = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = i % W;
    if ((x < W - 1 && labels[i + 1] !== labels[i]) || (i < N - W && labels[i + W] !== labels[i])) edge[i] = 1;
  }
  const b = blur(edge, W, H, 1, d), m = new Uint8Array(N);
  for (let i = 0; i < N; i++) m[i] = b[i] > 1e-3 ? 1 : 0;
  return m;
}

// Per region: descriptor distance (normalised feature space) between the input region and the same
// region of the collage, and the orientation error where the input region is oriented. Area-weighted.
export function structure(a, b, seg, norm) {
  const { W, H, K, labels } = seg, N = W * H, La = Lch(a, N), Lb = Lch(b, N);
  let dsum = 0, wsum = 0, osum = 0, ow = 0;
  for (let k = 0; k < K; k++) {
    const m = new Uint8Array(N);
    for (let i = 0; i < N; i++) m[i] = labels[i] === k ? 1 : 0;
    const da = describe(La, m, W, H), db = describe(Lb, m, W, H), w = seg.stats[k].frac;
    const d = featureDist(da.f, db.f, norm);
    if (Number.isFinite(d)) { dsum += w * d; wsum += w; }
    if (da.f[3] > 0.15) {
      const e = Math.abs((((db.theta - da.theta + Math.PI / 2) % Math.PI) + Math.PI) % Math.PI - Math.PI / 2);
      osum += w * da.f[3] * e; ow += w * da.f[3];
    }
  }
  return { dist: dsum / wsum, orientErr: ow ? osum / ow : NaN };
}

// Area-weighted CIE76 distance between each region's mean colour in the input and in the collage:
// whether the tint lands on the region's colour, whatever the texture does inside it.
export function regionColourError(a, b, seg) {
  const { K, labels, W, H } = seg, N = W * H, sa = new Float64Array(3 * K), sb = new Float64Array(3 * K), n = new Float64Array(K);
  for (let i = 0; i < N; i++) { const k = labels[i]; n[k]++; for (let c = 0; c < 3; c++) { sa[3 * k + c] += a[3 * i + c]; sb[3 * k + c] += b[3 * i + c]; } }
  let e = 0;
  for (let k = 0; k < K; k++) e += (n[k] / N) * Math.hypot(...[0, 1, 2].map((c) => (sa[3 * k + c] - sb[3 * k + c]) / n[k]));
  return e;
}
