// Texture collage: decompose an image into colour regions, describe each region's structure, pick a
// texture class for it, and fill it with a tinted, rotated, tiled texture. Pure functions on typed
// arrays: the page, a bridge worker and Node all run the same code.
//
// Conventions: images are W x H, pixel i = y * W + x. Lab images are interleaved Float32Array(3N),
// L in [0, 100]. Texture tiles are seamless S x S RGB Uint8Array(3 * S * S).

// ---------------------------------------------------------------- random
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- colour (sRGB D65 <-> CIELAB)
const DEC = new Float32Array(256);
for (let i = 0; i < 256; i++) { const c = i / 255; DEC[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; }
const XN = 0.95047, ZN = 1.08883;
const fLab = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
const fInv = (t) => (t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787);
const enc = (c) => { c = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055; return c <= 0 ? 0 : c >= 1 ? 255 : Math.round(c * 255); };

// rgb: Uint8 array with `ch` channels per pixel (3 or 4). Returns interleaved Lab.
export function rgbToLab(rgb, n, ch = 4) {
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const r = DEC[rgb[ch * i]], g = DEC[rgb[ch * i + 1]], b = DEC[rgb[ch * i + 2]];
    const x = fLab((0.4124 * r + 0.3576 * g + 0.1805 * b) / XN);
    const y = fLab(0.2126 * r + 0.7152 * g + 0.0722 * b);
    const z = fLab((0.0193 * r + 0.1192 * g + 0.9505 * b) / ZN);
    out[3 * i] = 116 * y - 16; out[3 * i + 1] = 500 * (x - y); out[3 * i + 2] = 200 * (y - z);
  }
  return out;
}

// Lab -> RGBA Uint8ClampedArray (alpha 255)
export function labToRgba(lab, n, out = new Uint8ClampedArray(4 * n)) {
  for (let i = 0; i < n; i++) {
    const fy = (lab[3 * i] + 16) / 116, fx = fy + lab[3 * i + 1] / 500, fz = fy - lab[3 * i + 2] / 200;
    const x = XN * fInv(fx), y = fInv(fy), z = ZN * fInv(fz);
    out[4 * i] = enc(3.2406 * x - 1.5372 * y - 0.4986 * z);
    out[4 * i + 1] = enc(-0.9689 * x + 1.8758 * y + 0.0415 * z);
    out[4 * i + 2] = enc(0.0557 * x - 0.204 * y + 1.057 * z);
    out[4 * i + 3] = 255;
  }
  return out;
}

// ---------------------------------------------------------------- filters
// One box pass of radius r along rows or columns of a C-channel interleaved image, clamped edges.
function boxPass(src, dst, W, H, C, r, horiz) {
  const len = horiz ? W : H, lines = horiz ? H : W, inv = 1 / (2 * r + 1);
  const acc = new Float64Array(C);
  for (let l = 0; l < lines; l++) {
    const at = (k) => (horiz ? l * W + k : k * W + l);
    acc.fill(0);
    for (let k = -r; k <= r; k++) { const p = at(Math.min(len - 1, Math.max(0, k))) * C; for (let c = 0; c < C; c++) acc[c] += src[p + c]; }
    for (let k = 0; k < len; k++) {
      const p = at(k) * C;
      for (let c = 0; c < C; c++) dst[p + c] = acc[c] * inv;
      const a = at(Math.min(len - 1, k + r + 1)) * C, b = at(Math.max(0, k - r)) * C;
      for (let c = 0; c < C; c++) acc[c] += src[a + c] - src[b + c];
    }
  }
}

// Three box passes per axis: close to a Gaussian with sigma ~ r. Returns a new array.
export function blur(img, W, H, C, r) {
  if (r <= 0) return Float32Array.from(img);
  let a = Float32Array.from(img), b = new Float32Array(img.length);
  for (let k = 0; k < 3; k++) { boxPass(a, b, W, H, C, r, true); boxPass(b, a, W, H, C, r, false); }
  return a;
}

// ---------------------------------------------------------------- segmentation
// Colour k-means (k-means++ init, Lab of a blurred copy) then a majority filter and a merge of
// components smaller than minFrac of the image into the neighbour they share most border with.
export function segment(lab, W, H, { K = 8, blurR = null, seed = 1, minFrac = 0.002, iters = 20 } = {}) {
  const N = W * H, rnd = mulberry32(seed);
  const r = blurR ?? Math.max(1, Math.round(Math.max(W, H) / 150));
  const sm = blur(lab, W, H, 3, r);
  // sample
  const M = Math.min(N, 20000), samp = new Float32Array(3 * M);
  for (let j = 0; j < M; j++) { const i = Math.floor(rnd() * N); samp.set(sm.subarray(3 * i, 3 * i + 3), 3 * j); }
  const d2 = (a, ai, b, bi) => { const x = a[ai] - b[bi], y = a[ai + 1] - b[bi + 1], z = a[ai + 2] - b[bi + 2]; return x * x + y * y + z * z; };
  // k-means++
  const cen = new Float32Array(3 * K);
  cen.set(samp.subarray(0, 3));
  const best = new Float32Array(M).fill(Infinity);
  for (let k = 1; k < K; k++) {
    let tot = 0;
    for (let j = 0; j < M; j++) { best[j] = Math.min(best[j], d2(samp, 3 * j, cen, 3 * (k - 1))); tot += best[j]; }
    let t = rnd() * tot, j = 0;
    while (j < M - 1 && (t -= best[j]) > 0) j++;
    cen.set(samp.subarray(3 * j, 3 * j + 3), 3 * k);
  }
  const asg = new Uint8Array(M);
  for (let it = 0; it < iters; it++) {
    const sum = new Float64Array(3 * K), cnt = new Float64Array(K);
    for (let j = 0; j < M; j++) {
      let bk = 0, bd = Infinity;
      for (let k = 0; k < K; k++) { const d = d2(samp, 3 * j, cen, 3 * k); if (d < bd) { bd = d; bk = k; } }
      asg[j] = bk; cnt[bk]++; for (let c = 0; c < 3; c++) sum[3 * bk + c] += samp[3 * j + c];
    }
    for (let k = 0; k < K; k++) if (cnt[k]) for (let c = 0; c < 3; c++) cen[3 * k + c] = sum[3 * k + c] / cnt[k];
  }
  let lab8 = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let bk = 0, bd = Infinity;
    for (let k = 0; k < K; k++) { const d = d2(sm, 3 * i, cen, 3 * k); if (d < bd) { bd = d; bk = k; } }
    lab8[i] = bk;
  }
  // majority filter, 5x5, twice
  const cnt = new Uint16Array(K);
  for (let pass = 0; pass < 2; pass++) {
    const o = new Uint8Array(N);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      cnt.fill(0);
      for (let dy = -2; dy <= 2; dy++) { const yy = Math.min(H - 1, Math.max(0, y + dy)) * W;
        for (let dx = -2; dx <= 2; dx++) cnt[lab8[yy + Math.min(W - 1, Math.max(0, x + dx))]]++; }
      let bk = lab8[y * W + x], bc = cnt[bk];
      for (let k = 0; k < K; k++) if (cnt[k] > bc) { bc = cnt[k]; bk = k; }
      o[y * W + x] = bk;
    }
    lab8 = o;
  }
  // merge small components
  const minA = Math.max(4, Math.round(minFrac * N));
  for (let pass = 0; pass < 4; pass++) {
    const { comp, sizes } = components(lab8, W, H);
    // boundary pixels per component: a component whose area is under 2.5x its boundary is a thin
    // strip (often the blend band k-means puts along a blurred edge) and merges like a small one
    const bnd = new Int32Array(sizes.length);
    for (let i = 0; i < N; i++) {
      const x = i % W, c = comp[i];
      if ((x > 0 && comp[i - 1] !== c) || (x < W - 1 && comp[i + 1] !== c) || (i >= W && comp[i - W] !== c) || (i < N - W && comp[i + W] !== c)) bnd[c]++;
    }
    const small = sizes.map((n, c) => n < minA || n < 2.5 * bnd[c]);
    let changed = 0;
    const border = new Map();     // small comp -> Map(label -> shared border length)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, c = comp[i];
      if (!small[c]) continue;
      let m = border.get(c); if (!m) border.set(c, (m = new Map()));
      for (const j of [x > 0 ? i - 1 : -1, x < W - 1 ? i + 1 : -1, y > 0 ? i - W : -1, y < H - 1 ? i + W : -1]) {
        if (j >= 0 && comp[j] !== c) m.set(lab8[j], (m.get(lab8[j]) || 0) + 1);
      }
    }
    const to = new Map();
    for (const [c, m] of border) { let bl = -1, bn = 0; for (const [l, n] of m) if (n > bn) { bn = n; bl = l; } if (bl >= 0) to.set(c, bl); }
    for (let i = 0; i < N; i++) { const t = to.get(comp[i]); if (t !== undefined) { lab8[i] = t; changed++; } }
    if (!changed) break;
  }
  // drop empty clusters, reindex, region stats from the unblurred image
  const used = new Int16Array(K).fill(-1); let k2 = 0;
  for (let i = 0; i < N; i++) if (used[lab8[i]] < 0) used[lab8[i]] = k2++;
  for (let i = 0; i < N; i++) lab8[i] = used[lab8[i]];
  return { labels: lab8, K: k2, W, H, blurR: r, stats: regionStats(lab, lab8, k2, N) };
}

export function components(labels, W, H) {
  const N = W * H, comp = new Int32Array(N).fill(-1), sizes = [], stack = new Int32Array(N);
  let nc = 0;
  for (let s = 0; s < N; s++) {
    if (comp[s] >= 0) continue;
    let sp = 0, n = 0; stack[sp++] = s; comp[s] = nc;
    while (sp) {
      const i = stack[--sp]; n++;
      const x = i % W, l = labels[i];
      if (x > 0 && comp[i - 1] < 0 && labels[i - 1] === l) { comp[i - 1] = nc; stack[sp++] = i - 1; }
      if (x < W - 1 && comp[i + 1] < 0 && labels[i + 1] === l) { comp[i + 1] = nc; stack[sp++] = i + 1; }
      if (i >= W && comp[i - W] < 0 && labels[i - W] === l) { comp[i - W] = nc; stack[sp++] = i - W; }
      if (i < N - W && comp[i + W] < 0 && labels[i + W] === l) { comp[i + W] = nc; stack[sp++] = i + W; }
    }
    sizes.push(n); nc++;
  }
  return { comp, sizes };
}

// per-region mean and std of each Lab channel, and pixel count
export function regionStats(lab, labels, K, N) {
  const s = new Float64Array(3 * K), q = new Float64Array(3 * K), n = new Float64Array(K);
  for (let i = 0; i < N; i++) { const k = labels[i]; n[k]++; for (let c = 0; c < 3; c++) { const v = lab[3 * i + c]; s[3 * k + c] += v; q[3 * k + c] += v * v; } }
  const out = [];
  for (let k = 0; k < K; k++) {
    const mean = [0, 1, 2].map((c) => s[3 * k + c] / n[k]);
    const std = [0, 1, 2].map((c) => Math.sqrt(Math.max(0, q[3 * k + c] / n[k] - mean[c] ** 2)));
    out.push({ n: n[k], frac: n[k] / N, mean, std });
  }
  return out;
}

// ---------------------------------------------------------------- structure descriptor
// On the L channel of a region (mask) or a whole seamless tile (mask null, wrap true):
//   b1, b2, b4  busyness, mean |grad L| / (std L + 2), at full, half and quarter resolution
//   gcoh        coherence of the region-wide structure tensor (one dominant orientation?)
//   lcoh        energy-weighted mean of the local (r = 3) structure-tensor coherence (lines vs blobs)
//   skew, kurt  of L (dark specks vs bright specks, spiky vs flat histograms; kurt as log)
// theta is the dominant gradient direction in radians, mod pi. A dim with too few pixels is NaN.
export const FEATS = ["b1", "b2", "b4", "gcoh", "lcoh", "skew", "kurt"];

export function describe(L, mask, W, H, wrap = false) {
  const f = new Array(FEATS.length).fill(NaN);
  let Ls = L, Ms = mask, w = W, h = H, theta = 0;
  for (let s = 0; s < 3; s++) {
    const g = grad(Ls, Ms, w, h, wrap);
    if (g.n >= 32) {
      f[s] = g.mag / (g.std + 2);
      if (s === 0) {
        const { jxx, jyy, jxy } = g;
        f[3] = Math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / (jxx + jyy + 1e-9);
        theta = 0.5 * Math.atan2(2 * jxy, jxx - jyy);
        f[4] = localCoherence(g, w, h);
        f[5] = g.skew; f[6] = g.kurt;
      }
    }
    if (w < 8 || h < 8) break;
    [Ls, Ms, w, h] = half(Ls, Ms, w, h);
  }
  return { f, theta, n: mask ? countMask(mask) : W * H };
}

function countMask(m) { let n = 0; for (let i = 0; i < m.length; i++) n += m[i] ? 1 : 0; return n; }

function half(L, M, W, H) {
  const w = W >> 1, h = H >> 1, o = new Float32Array(w * h), m = M ? new Uint8Array(w * h) : null;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const a = 2 * y * W + 2 * x, b = a + 1, c = a + W, d = c + 1;
    o[y * w + x] = (L[a] + L[b] + L[c] + L[d]) / 4;
    if (m) m[y * w + x] = M[a] && M[b] && M[c] && M[d] ? 1 : 0;
  }
  return [o, m, w, h];
}

function grad(L, M, W, H, wrap) {
  const N = W * H, gx = new Float32Array(N), gy = new Float32Array(N), ok = new Uint8Array(N);
  let n = 0, mag = 0, jxx = 0, jyy = 0, jxy = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, m = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (M && !M[i]) continue;
    m++; const v = L[i]; s1 += v; s2 += v * v;
    let xl = x - 1, xr = x + 1, yu = y - 1, yd = y + 1;
    if (wrap) { xl = (xl + W) % W; xr %= W; yu = (yu + H) % H; yd %= H; }
    else if (xl < 0 || xr >= W || yu < 0 || yd >= H) continue;
    const il = y * W + xl, ir = y * W + xr, iu = yu * W + x, id = yd * W + x;
    if (M && !(M[il] && M[ir] && M[iu] && M[id])) continue;
    const a = (L[ir] - L[il]) / 2, b = (L[id] - L[iu]) / 2;
    gx[i] = a; gy[i] = b; ok[i] = 1; n++;
    mag += Math.hypot(a, b); jxx += a * a; jyy += b * b; jxy += a * b;
  }
  const mean = s1 / Math.max(1, m), std = Math.sqrt(Math.max(0, s2 / Math.max(1, m) - mean * mean));
  for (let i = 0; i < N; i++) {
    if (M && !M[i]) continue;
    const z = (L[i] - mean) / (std + 1e-6); s3 += z * z * z; s4 += z * z * z * z;
  }
  return { gx, gy, ok, n, mag: mag / Math.max(1, n), std, jxx, jyy, jxy,
    skew: Math.max(-3, Math.min(3, s3 / Math.max(1, m))), kurt: Math.log(Math.max(1, s4 / Math.max(1, m))) };
}

function localCoherence(g, W, H) {
  const N = W * H, J = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) if (g.ok[i]) { const a = g.gx[i], b = g.gy[i]; J[3 * i] = a * a; J[3 * i + 1] = b * b; J[3 * i + 2] = a * b; }
  const S = blur(J, W, H, 3, 3);
  let num = 0, den = 0;
  for (let i = 0; i < N; i++) {
    if (!g.ok[i]) continue;
    const xx = S[3 * i], yy = S[3 * i + 1], xy = S[3 * i + 2], e = xx + yy;
    if (e <= 1e-9) continue;
    num += Math.sqrt((xx - yy) ** 2 + 4 * xy * xy); den += e;
  }
  return den > 0 ? num / den : NaN;
}

// ---------------------------------------------------------------- tiles
export function tileLab(rgb, S) { return rgbToLab(rgb, S * S, 3); }

// L channel of a tile resampled at `scale` image px per tile px (bilinear, wrapping), as a tile of
// round(S * scale) px: what the descriptor of a region filled at that scale sees.
export function scaledL(tlab, S, scale) {
  const T = Math.max(8, Math.round(S * scale)), o = new Float32Array(T * T);
  for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) o[y * T + x] = sampleWrap(tlab, S, x / scale, y / scale, 0, 3);
  return { L: o, T };
}

// bilinear sample of channel c of a seamless S x S tile with C interleaved channels
export function sampleWrap(t, S, u, v, c, C) {
  const fu = Math.floor(u), fv = Math.floor(v), du = u - fu, dv = v - fv;
  const x0 = ((fu % S) + S) % S, y0 = ((fv % S) + S) % S, x1 = (x0 + 1) % S, y1 = (y0 + 1) % S;
  const a = t[(y0 * S + x0) * C + c], b = t[(y0 * S + x1) * C + c], d = t[(y1 * S + x0) * C + c], e = t[(y1 * S + x1) * C + c];
  return (a * (1 - du) + b * du) * (1 - dv) + (d * (1 - du) + e * du) * dv;
}

export function labStats(tlab, n) {
  const mean = [0, 0, 0], std = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) mean[c] += tlab[3 * i + c];
  for (let c = 0; c < 3; c++) mean[c] /= n;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) std[c] += (tlab[3 * i + c] - mean[c]) ** 2;
  for (let c = 0; c < 3; c++) std[c] = Math.sqrt(std[c] / n);
  return { mean, std };
}

// Descriptor table entry for one tile at the given scales.
export function describeTile(rgb, S, scales = [1, 2]) {
  const tlab = tileLab(rgb, S), st = labStats(tlab, S * S);
  const per = {};
  for (const sc of scales) { const { L, T } = scaledL(tlab, S, sc); per[sc] = describe(L, null, T, T, true); }
  return { mean: st.mean, std: st.std, per };
}

// ---------------------------------------------------------------- class choice
// table: { names, norm: [per-feature scale], classes: [{ mean: Lab, per: { scale: { f, theta } } }] }
// regions: [{ desc: { f }, stats: { mean } , frac }]
// mode: "structure" | "colour" | "random" | "fixed:<name>"; distinct: each class at most once (largest region first)
export function featureDist(fa, fb, norm) {
  let d = 0, n = 0;
  for (let i = 0; i < norm.length; i++) {
    if (!Number.isFinite(fa[i]) || !Number.isFinite(fb[i])) continue;
    d += ((fa[i] - fb[i]) / norm[i]) ** 2; n++;
  }
  return n ? d / n : Infinity;
}

export function chooseClasses(regions, table, { mode = "structure", distinct = false, scales = [1, 2], seed = 1, exclude = [] } = {}) {
  const C = table.classes.length, rnd = mulberry32(seed);
  const order = regions.map((r, i) => i).sort((a, b) => regions[b].frac - regions[a].frac);
  const taken = new Set(), out = new Array(regions.length);
  for (const k of order) {
    const r = regions[k];
    const cands = [];
    for (let c = 0; c < C; c++) {
      if (distinct && taken.has(c) && taken.size < C) continue;
      if (exclude.includes(table.names[c]) && !mode.startsWith("fixed:")) continue;
      for (const sc of scales) {
        let d;
        if (mode === "structure") d = featureDist(r.desc.f, table.classes[c].per[sc].f, table.norm);
        else if (mode === "colour") { const m = table.classes[c].mean, q = r.stats.mean; d = (m[0] - q[0]) ** 2 + (m[1] - q[1]) ** 2 + (m[2] - q[2]) ** 2; }
        else if (mode === "random") d = rnd();
        else if (mode.startsWith("fixed:")) d = table.names[c] === mode.slice(6) ? 0 : 1;
        cands.push({ c, sc, d });
      }
    }
    cands.sort((a, b) => a.d - b.d);
    // colour, random and fixed do not judge scale: take the first scale listed
    const pick = mode === "structure" ? cands[0] : cands.find((x) => x.sc === scales[0]) || cands[0];
    taken.add(pick.c);
    out[k] = { cls: pick.c, scale: pick.sc, dist: pick.d };
  }
  return out;
}

// ---------------------------------------------------------------- render
// seg: from segment(). tex[k]: { lab (tile Lab), S, stats: labStats, theta, gcoh } for region k.
// opts.tint: "region" (region mean colour, texture detail scaled to the region's spread),
//            "local" (blurred input colour plus texture detail), "none" (raw texture colours)
// opts.feather: px of soft blend between regions (0 = hard edges)
// opts.strength: floor on the L spread of the texture detail (L units), so flat regions still show texture
// opts.align: rotate each texture so its dominant orientation matches its region's
// opts.scale[k]: image px per tile px for region k
export function render(lab, seg, tex, opts = {}) {
  const { W, H, K, labels } = seg, N = W * H;
  const { tint = "region", feather = 2, strength = 8, align = true, scales = [], rdesc = [], localR = null,
    maxGainL = 4, maxGainAB = 1.5, stochastic = true, cells = 2 } = opts;
  const out = new Float32Array(3 * N);
  // soft weights per region
  let wts = null;
  if (feather > 0) {
    wts = [];
    for (let k = 0; k < K; k++) {
      const m = new Float32Array(N);
      for (let i = 0; i < N; i++) m[i] = labels[i] === k ? 1 : 0;
      wts.push(blur(m, W, H, 1, Math.max(1, Math.round(feather / 1.7))));
    }
  }
  // local base colour and the residual spread it leaves per region
  let base = null, resStd = null;
  if (tint === "local") {
    base = blur(lab, W, H, 3, localR ?? Math.max(2, 2 * seg.blurR));
    const q = new Float64Array(3 * K), n = new Float64Array(K);
    for (let i = 0; i < N; i++) { const k = labels[i]; n[k]++; for (let c = 0; c < 3; c++) q[3 * k + c] += (lab[3 * i + c] - base[3 * i + c]) ** 2; }
    resStd = []; for (let k = 0; k < K; k++) resStd.push([0, 1, 2].map((c) => Math.sqrt(q[3 * k + c] / Math.max(1, n[k]))));
  }
  // per-region transform constants
  const P = [];
  for (let k = 0; k < K; k++) {
    const t = tex[k], st = seg.stats[k], sc = scales[k] || 1;
    let phi = 0;
    const rd = rdesc[k];
    if (align && rd && t.gcoh > 0.15 && rd.f[3] > 0.15) phi = rd.theta - t.theta;
    const spread = tint === "local" ? resStd[k] : st.std;
    const so = [Math.max(spread[0], strength), spread[1], spread[2]];
    // gain capped: a tile with little chroma spread would otherwise have its noise blown up to the
    // region's spread and clipped out of gamut, shifting the region's colour
    const gain = [0, 1, 2].map((c) => Math.min(c ? maxGainAB : maxGainL, so[c] / Math.max(0.5, t.stats.std[c])));
    P.push({ t, cs: Math.cos(phi) / sc, sn: Math.sin(phi) / sc, gain, mean: st.mean, tm: t.stats.mean });
    if (opts.info) opts.info[k] = { phi, period: t.S * sc };
  }
  const px = new Float32Array(3), tap = new Float32Array(3);
  const one = (k, x, y, i) => {
    const p = P[k], t = p.t;
    const u = x * p.cs + y * p.sn, v = -x * p.sn + y * p.cs;
    if (stochastic) stochasticSample(t, u, v, cells, tap);
    for (let c = 0; c < 3; c++) {
      const s = stochastic ? tap[c] : sampleWrap(t.lab, t.S, u, v, c, 3);
      if (tint === "none") px[c] = s;
      else px[c] = (tint === "local" ? base[3 * i + c] : p.mean[c]) + (s - p.tm[c]) * p.gain[c];
    }
  };
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (!wts) { one(labels[i], x, y, i); out[3 * i] = px[0]; out[3 * i + 1] = px[1]; out[3 * i + 2] = px[2]; continue; }
    let a0 = 0, a1 = 0, a2 = 0, ws = 0;
    for (let k = 0; k < K; k++) {
      const w = wts[k][i];
      if (w < 0.004) continue;
      one(k, x, y, i); a0 += w * px[0]; a1 += w * px[1]; a2 += w * px[2]; ws += w;
    }
    out[3 * i] = a0 / ws; out[3 * i + 1] = a1 / ws; out[3 * i + 2] = a2 / ws;
  }
  for (let i = 0; i < N; i++) out[3 * i] = Math.min(100, Math.max(0, out[3 * i]));
  return out;
}

// Region descriptors for an image: one describe() per region over its pixels.
export function regionDescriptors(lab, seg) {
  const { W, H, K, labels } = seg, N = W * H, L = new Float32Array(N);
  for (let i = 0; i < N; i++) L[i] = lab[3 * i];
  const out = [];
  for (let k = 0; k < K; k++) {
    const m = new Uint8Array(N);
    for (let i = 0; i < N; i++) m[i] = labels[i] === k ? 1 : 0;
    out.push(describe(L, m, W, H));
  }
  return out;
}

// Stochastic tiling (Heitz & Neyret 2018, as splat-diffusion's WGSL_BLIT mode 1): the tile is cut by a
// triangle grid with `cells` triangles' worth of edge per tile; each corner of the triangle around (u, v)
// reads the tile at its own random offset, and the three taps are blended around the tile's mean with
// weights normalised by sqrt(sum w^2), which keeps the texture's variance. The repeat grid disappears.
// u, v in tile pixels; writes Lab into out.
const fract = (x) => x - Math.floor(x);
function hash2(qx, qy) {
  return [fract(Math.sin(qx * 127.1 + qy * 311.7) * 43758.5453), fract(Math.sin(qx * 269.5 + qy * 183.3) * 43758.5453)];
}
export function stochasticSample(t, u, v, cells, out) {
  const S = t.S, m = t.stats.mean;
  const sx = (u / S) * cells, sy = (v / S) * cells;
  const kx = sx - 0.57735027 * sy, ky = 1.15470054 * sy;
  const bx = Math.floor(kx), by = Math.floor(ky), fx = kx - bx, fy = ky - by, z = 1 - fx - fy;
  let w, V;
  if (z > 0) { w = [z, fy, fx]; V = [[bx, by], [bx, by + 1], [bx + 1, by]]; }
  else { w = [-z, 1 - fy, 1 - fx]; V = [[bx + 1, by + 1], [bx + 1, by], [bx, by + 1]]; }
  const norm = 1 / Math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  out[0] = out[1] = out[2] = 0;
  for (let j = 0; j < 3; j++) {
    const [hx, hy] = hash2(V[j][0], V[j][1]), uu = u + hx * S, vv = v + hy * S;
    for (let c = 0; c < 3; c++) out[c] += w[j] * (sampleWrap(t.lab, S, uu, vv, c, 3) - m[c]);
  }
  for (let c = 0; c < 3; c++) out[c] = m[c] + out[c] * norm;
}

// Prepare a tile for render(): Lab, stats and orientation at the scale it will be shown.
export function prepTile(rgb, S, scale = 1) {
  const lab = tileLab(rgb, S), stats = labStats(lab, S * S);
  const { L, T } = scaledL(lab, S, scale), d = describe(L, null, T, T, true);
  return { lab, S, stats, theta: d.theta, gcoh: d.f[3] };
}

// Solid-colour baseline: each region filled with its mean Lab (what a paint-region vectoriser gives).
export function renderSolid(seg) {
  const N = seg.W * seg.H, out = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) out.set(seg.stats[seg.labels[i]].mean, 3 * i);
  return out;
}
