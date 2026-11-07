// Splat collage: the texture-collage regions, each filled with a splat-diffusion texture, painted by one
// WGSL shader. No pixels: a packed low-resolution label map whose region edges are reconstructed as
// anti-aliased contours, a placement transform per region, and each region's splats (recoloured to the
// region before baking). The same scene renders from storage buffers in the page and from const arrays in
// the exported file, and `evalCPU` mirrors the shader for tests and measurements.
//
// Coordinates: p is in reference pixels of the picture the regions were cut from (W0 x H0, the working
// size); px is the size of one output pixel in reference pixels. Splats use splat-diffusion's layout,
// back to front, 12 floats each: mu_x, mu_y, e^a, b, e^c, r, g, b (in [-1, 1]), opacity, sharpness p,
// depth, 0. Only the first 10 reach the shader.
import { rgbToLab, labToRgba, labStats, describe, mulberry32 } from "../web/collage.js";

export const TILE_REF = 128;
// region edges: map cells, bilinear contour, B-spline contour, interpolated distance to the boundary
const MODES = { nearest: 0, contour: 1, spline: 2, distance: 3 };
const DMAX = 2;            // distances are stored up to this many cells, in 8 bits      // reference px per texture repeat at scale 1 (as the pixel collage)

// ---------------------------------------------------------------- CPU painting of one splat tile
// Mirrors splat-diffusion's WGSL_PAINT at uv in [0,1)^2. Writes rgb in [-1, 1] (unclamped) into out.
export function splatColour(sp, K, u, v, out, stride = 12) {
  let r = 0, g = 0, b = 0, T = 1;
  for (let i = 0; i < K; i++) {
    const o = (K - 1 - i) * stride;
    let dx = u - sp[o], dy = v - sp[o + 1];
    dx -= Math.round(dx); dy -= Math.round(dy);
    const a = sp[o + 2] * dx, c = sp[o + 3] * dx + sp[o + 4] * dy, q = a * a + c * c;
    if (q < 60) {
      const w = sp[o + 8] * Math.exp(-0.5 * Math.pow(q + 1e-8, sp[o + 9]));
      r += T * w * sp[o + 5]; g += T * w * sp[o + 6]; b += T * w * sp[o + 7];
      T *= 1 - w;
      if (T < 0.002) break;
    }
  }
  out[0] = r; out[1] = g; out[2] = b;
}

// Paint a tile of S x S pixels as RGB bytes (display colours, as the GPU painter clamps them).
export function paintTile(sp, K, S, stride = 12) {
  const rgb = new Uint8Array(3 * S * S), c = new Float32Array(3);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    splatColour(sp, K, (x + 0.5) / S, (y + 0.5) / S, c, stride);
    for (let k = 0; k < 3; k++) rgb[3 * (y * S + x) + k] = Math.round(Math.min(1, Math.max(0, (c[k] + 1) / 2)) * 255);
  }
  return rgb;
}

// ---------------------------------------------------------------- recolouring on the splats
// The pixel collage recolours each pixel: region mean + (texture - texture mean) * gain, per Lab channel.
// Here the same map is applied to each splat's own colour, so the shader stays pure splats.
// tileStats: labStats of the painted tile; reg: { mean, std } of the region (Lab).
export function recolourSplats(sp, K, tileStats, reg, { tint = "region", strength = 8, maxGainL = 4, maxGainAB = 1.5 } = {}) {
  const out = Float32Array.from(sp.subarray(0, K * 12));
  if (tint === "none") return out;
  const so = [Math.max(reg.std[0], strength), reg.std[1], reg.std[2]];
  const gain = [0, 1, 2].map((c) => Math.min(c ? maxGainAB : maxGainL, so[c] / Math.max(0.5, tileStats.std[c])));
  const rgb = new Uint8Array(3);
  for (let k = 0; k < K; k++) {
    const o = k * 12;
    // splat colour -> display rgb (8-bit, as painted) -> Lab -> transfer -> rgb -> [-1, 1]
    for (let c = 0; c < 3; c++) rgb[c] = Math.round(Math.min(1, Math.max(0, (sp[o + 5 + c] + 1) / 2)) * 255);
    const lab = rgbToLab(rgb, 1, 3);
    for (let c = 0; c < 3; c++) lab[c] = reg.mean[c] + (lab[c] - tileStats.mean[c]) * gain[c];
    lab[0] = Math.min(100, Math.max(0, lab[0]));
    const q = labToRgba(lab, 1);
    for (let c = 0; c < 3; c++) out[o + 5 + c] = (q[c] / 255) * 2 - 1;
  }
  return out;
}

// ---------------------------------------------------------------- label map
// Majority label of the full-resolution segmentation in each cell of an LW x LH grid.
export function labelMap(seg, LW) {
  const { W, H, labels, K } = seg;
  // square cells of W / LW reference px, as the shader assumes on both axes
  const cw = W / LW, LH = Math.max(2, Math.ceil(H / cw));
  const out = new Uint8Array(LW * LH), cnt = new Uint32Array(K);
  for (let j = 0; j < LH; j++) for (let i = 0; i < LW; i++) {
    const x0 = Math.floor(i * cw), x1 = Math.max(x0 + 1, Math.min(W, Math.floor((i + 1) * cw)));
    const y0 = Math.min(H - 1, Math.floor(j * cw)), y1 = Math.max(y0 + 1, Math.min(H, Math.floor((j + 1) * cw)));
    cnt.fill(0);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) cnt[labels[y * W + x]]++;
    let bk = 0; for (let k = 1; k < K; k++) if (cnt[k] > cnt[bk]) bk = k;
    out[j * LW + i] = bk;
  }
  // distance from each cell centre to the nearest pixel of another region, in cells, capped at DMAX
  const dist = new Float32Array(LW * LH), R = Math.ceil(DMAX * cw) + 1;
  for (let j = 0; j < LH; j++) for (let i = 0; i < LW; i++) {
    const cx = (i + 0.5) * cw, cy = (j + 0.5) * cw, l = out[j * LW + i];
    let best = DMAX * cw;
    for (let y = Math.max(0, Math.floor(cy - R)); y < Math.min(H, Math.ceil(cy + R)); y++)
      for (let x = Math.max(0, Math.floor(cx - R)); x < Math.min(W, Math.ceil(cx + R)); x++) {
        if (labels[y * W + x] === l) continue;
        // distance to the edge of that pixel's square, not its centre
        const dx = Math.max(0, Math.abs(x + 0.5 - cx) - 0.5), dy = Math.max(0, Math.abs(y + 0.5 - cy) - 0.5), d = Math.hypot(dx, dy);
        if (d < best) best = d;
      }
    dist[j * LW + i] = best / cw;
  }
  return { LW, LH, labels: out, dist };
}

// 4 bits per label for up to 16 regions, else 8 bits.
// In distance mode each cell is 16 bits: the label in the low 8, the distance (0..DMAX cells) in the high 8.
export function packLabels(lm, NR, mode = "contour") {
  const n = lm.labels.length;
  if (mode === "distance") {
    const words = new Uint32Array(Math.ceil(n / 2 / 4) * 4);
    for (let i = 0; i < n; i++) {
      const q = Math.round((Math.min(DMAX, lm.dist[i]) / DMAX) * 255);
      words[i >> 1] |= ((lm.labels[i] | (q << 8)) << ((i & 1) * 16)) >>> 0;
    }
    return { bits: 16, words };
  }
  const bits = NR <= 16 ? 4 : 8, per = 32 / bits;
  const words = new Uint32Array(Math.ceil(n / per / 4) * 4);
  for (let i = 0; i < n; i++) words[(i / per) | 0] |= lm.labels[i] << ((i % per) * bits);
  return { bits, words };
}

// ---------------------------------------------------------------- scene
// regions[k]: { splats (K x 12, recoloured), phi (radians), scale, off: [u, v] }
// The texture coordinate of reference point p in region k is R(-phi) p / (TILE_REF * scale) + off.
export function makeScene({ W0, H0, lm, regions, K, mode = "contour", blend = 0 }) {
  const NR = regions.length, packed = packLabels(lm, NR, mode);
  const xf = regions.map((r) => {
    const s = 1 / (TILE_REF * r.scale), c = Math.cos(r.phi) * s, n = Math.sin(r.phi) * s;
    return { m: [c, n, -n, c], off: r.off };        // uv = (m0 x + m1 y, m2 x + m3 y) + off
  });
  // the CPU mirror reads the distances as the shader will, after 8-bit quantisation
  const dist = lm.dist ? Float32Array.from(lm.dist, (d) => (Math.round((Math.min(DMAX, d) / DMAX) * 255) / 255) * DMAX) : null;
  return { W0, H0, LW: lm.LW, LH: lm.LH, labels: lm.labels, dist, packed, NR, K, xf, splats: regions.map((r) => r.splats), mode, blend };
}

// The shader's region choice at reference point (px, py): the heaviest label l1, the runner-up l2 and
// l1's blend weight a1 (1 away from edges).
export function chooseLabels(sc, px, py, pxSize) {
  const cell = sc.W0 / sc.LW;
  const lab = (i, j) => sc.labels[Math.min(sc.LH - 1, Math.max(0, j)) * sc.LW + Math.min(sc.LW - 1, Math.max(0, i))];
  if (sc.mode === "nearest") return { l1: lab(Math.floor(px / cell), Math.floor(py / cell)), l2: 0, a1: 1 };
  if (sc.mode === "distance") {
    // per candidate label X among the 4 nearest cells: interpolate +d (cell is X) or -d (it is not); the
    // largest wins. Between two regions this puts the edge where the stored distances say it is.
    const idx = (i, j) => Math.min(sc.LH - 1, Math.max(0, j)) * sc.LW + Math.min(sc.LW - 1, Math.max(0, i));
    const gx = px / cell - 0.5, gy = py / cell - 0.5, ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy;
    const I = [idx(ix, iy), idx(ix + 1, iy), idx(ix, iy + 1), idx(ix + 1, iy + 1)];
    const Wb = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
    let b1 = -1e9, b2 = -1e9, l1 = sc.labels[I[0]], l2 = l1;
    for (let a = 0; a < 4; a++) {
      const X = sc.labels[I[a]];
      let seen = false; for (let b = 0; b < a; b++) if (sc.labels[I[b]] === X) seen = true;
      if (seen) continue;
      let phi = 0;
      for (let b = 0; b < 4; b++) phi += Wb[b] * (sc.labels[I[b]] === X ? sc.dist[I[b]] + 0.02 : -sc.dist[I[b]]);
      if (phi > b1) { b2 = b1; l2 = l1; b1 = phi; l1 = X; } else if (phi > b2) { b2 = phi; l2 = X; }
    }
    if (b2 <= -1e8) return { l1, l2: 0, a1: 1 };
    const e = Math.max(pxSize, sc.blend) / cell;     // phi changes by about 2 per cell across an edge
    return { l1, l2, a1: Math.min(1, 0.5 + (b1 - b2) / Math.max(1e-6, 2 * e)) };
  }
  let L, Wt;
  if (sc.mode === "spline") {
    // quadratic B-spline over the 3 x 3 cells around the nearest cell centre: rounded (C1) boundaries
    const cx = px / cell, cy = py / cell, nx = Math.floor(cx), ny = Math.floor(cy), fx = cx - nx - 0.5, fy = cy - ny - 0.5;
    const bx = [0.5 * (0.5 - fx) ** 2, 0.75 - fx * fx, 0.5 * (0.5 + fx) ** 2], by = [0.5 * (0.5 - fy) ** 2, 0.75 - fy * fy, 0.5 * (0.5 + fy) ** 2];
    L = []; Wt = [];
    for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) { L.push(lab(nx - 1 + i, ny - 1 + j)); Wt.push(bx[i] * by[j]); }
  } else {
    const gx = px / cell - 0.5, gy = py / cell - 0.5;
    const ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy;
    L = [lab(ix, iy), lab(ix + 1, iy), lab(ix, iy + 1), lab(ix + 1, iy + 1)];
    Wt = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
  }
  const n = L.length;
  // merge duplicate labels, then take the two heaviest
  for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) if (L[b] === L[a] && Wt[b] > 0) { Wt[a] += Wt[b]; Wt[b] = 0; }
  let i1 = 0; for (let a = 1; a < n; a++) if (Wt[a] > Wt[i1]) i1 = a;
  let i2 = -1; for (let a = 0; a < n; a++) if (a !== i1 && L[a] !== L[i1] && (i2 < 0 || Wt[a] > Wt[i2])) i2 = a;
  if (i2 < 0 || Wt[i2] <= 0) return { l1: L[i1], l2: 0, a1: 1 };
  const e = (2 * Math.max(pxSize, sc.blend)) / cell;          // transition width in weight units
  return { l1: L[i1], l2: L[i2], a1: Math.min(1, 0.5 + (Wt[i1] - Wt[i2]) / Math.max(1e-6, e)) };
}
export const dominantLabel = (sc, px, py) => chooseLabels(sc, px, py, 0).l1;

// CPU mirror of the shader's collage(): rgb in [0, 1] into out.
export function evalCPU(sc, px, py, pxSize, out, tmp = new Float32Array(3)) {
  if (px < 0 || py < 0 || px >= sc.W0 || py >= sc.H0) { out[0] = out[1] = out[2] = 0; return; }
  const { l1, l2, a1 } = chooseLabels(sc, px, py, pxSize);
  texCPU(sc, l1, px, py, out);
  if (a1 < 1) {
    texCPU(sc, l2, px, py, tmp);
    for (let c = 0; c < 3; c++) out[c] = a1 * out[c] + (1 - a1) * tmp[c];
  }
}

function texCPU(sc, k, px, py, out) {
  const { m, off } = sc.xf[k];
  const u = m[0] * px + m[1] * py + off[0], v = m[2] * px + m[3] * py + off[1];
  splatColour(sc.splats[k], sc.K, u - Math.floor(u), v - Math.floor(v), out);
  for (let c = 0; c < 3; c++) out[c] = Math.min(1, Math.max(0, (out[c] + 1) / 2));
}

// Render the whole scene on the CPU at W x H output pixels (W / W0 output px per reference px).
export function renderCPU(sc, W, H) {
  const s = sc.W0 / W, rgba = new Uint8ClampedArray(4 * W * H), c = new Float32Array(3), t = new Float32Array(3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    evalCPU(sc, (x + 0.5) * s, (y + 0.5) * s, s, c, t);
    const o = 4 * (y * W + x);
    rgba[o] = c[0] * 255; rgba[o + 1] = c[1] * 255; rgba[o + 2] = c[2] * 255; rgba[o + 3] = 255;
  }
  return rgba;
}

// ---------------------------------------------------------------- WGSL
// The splat loop, given accessor expressions for splat i's three vectors.
const LOOP = (n, A, B, C) => `
  var c = vec3f(0.0);
  var T = 1.0;
  for (var i = 0u; i < ${n}; i++) {
    let k = ${n} - 1u - i;
    let s0 = ${A}; let s1 = ${B}; let s2 = ${C};
    var d = uv - s0.xy; d = d - round(d);
    let u = s0.z * d.x; let v = s0.w * d.x + s1.x * d.y;
    let q = u * u + v * v;
    if (q < 60.0) {
      let w = s2.x * exp(-0.5 * pow(q + 1e-8, s2.y));
      c += T * w * s1.yzw;
      T *= 1.0 - w;
      if (T < 0.002) { break; }
    }
  }
  return clamp((c + 1.0) * 0.5, vec3f(0.0), vec3f(1.0));`;

// The region choice and blend, shared by both variants. Needs: LW, LH, CELL, MODE, BLEND, NR, BITS consts,
// fn labelWord(i: u32) -> u32, fn tex(k: u32, p: vec2f) -> vec3f.
const COLLAGE = `
fn label(i: i32, j: i32) -> u32 {
  let x = u32(clamp(i, 0, i32(LW) - 1)); let y = u32(clamp(j, 0, i32(LH) - 1));
  let n = y * LW + x;
  let per = 32u / BITS;
  return (labelWord(n / per) >> ((n % per) * BITS)) & ((1u << BITS) - 1u);
}
// Distance mode: a cell is 16 bits, label | distance-to-boundary (0..2 cells in 8 bits) << 8.
fn cellAt(i: i32, j: i32) -> vec2f {
  let x = u32(clamp(i, 0, i32(LW) - 1)); let y = u32(clamp(j, 0, i32(LH) - 1));
  let n = y * LW + x;
  let v = (labelWord(n / 2u) >> ((n % 2u) * 16u)) & 0xFFFFu;
  return vec2f(f32(v & 0xFFu), f32(v >> 8u) / 255.0 * 2.0);
}
fn distanceEdge(p: vec2f, px: f32) -> vec3f {
  let g = p / CELL - 0.5;
  let i0 = vec2i(floor(g)); let f = g - floor(g);
  var c = array<vec2f, 4>(cellAt(i0.x, i0.y), cellAt(i0.x + 1, i0.y), cellAt(i0.x, i0.y + 1), cellAt(i0.x + 1, i0.y + 1));
  var w = array<f32, 4>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
  var b1 = -1e9; var b2 = -1e9; var l1 = c[0].x; var l2 = l1;
  for (var a = 0; a < 4; a++) {
    let X = c[a].x;
    var seen = false;
    for (var b = 0; b < a; b++) { if (c[b].x == X) { seen = true; } }
    if (seen) { continue; }
    var phi = 0.0;
    for (var b = 0; b < 4; b++) { phi += w[b] * select(-c[b].y, c[b].y + 0.02, c[b].x == X); }
    if (phi > b1) { b2 = b1; l2 = l1; b1 = phi; l1 = X; } else if (phi > b2) { b2 = phi; l2 = X; }
  }
  let c1 = tex(u32(l1), p);
  if (b2 <= -1e8) { return c1; }
  let e = max(px, BLEND) / CELL;
  let a1 = min(1.0, 0.5 + (b1 - b2) / max(1e-6, 2.0 * e));
  if (a1 >= 1.0) { return c1; }
  return mix(tex(u32(l2), p), c1, a1);
}
// p: reference pixels (0..W0, 0..H0); px: size of one output pixel in reference pixels.
fn collage(p: vec2f, px: f32) -> vec3f {
  if (p.x < 0.0 || p.y < 0.0 || p.x >= W0 || p.y >= H0) { return vec3f(0.0); }
  if (MODE == 0u) { return tex(label(i32(floor(p.x / CELL)), i32(floor(p.y / CELL))), p); }
  if (MODE == 3u) { return distanceEdge(p, px); }
  var L: array<u32, 9>;
  var w: array<f32, 9>;
  var n = 4;
  if (MODE == 2u) {
    // quadratic B-spline over the 3 x 3 cells around the nearest cell centre: rounded (C1) boundaries
    let c = p / CELL; let nc = vec2i(floor(c)); let f = c - floor(c) - 0.5;
    let bx = vec3f(0.5 * (0.5 - f.x) * (0.5 - f.x), 0.75 - f.x * f.x, 0.5 * (0.5 + f.x) * (0.5 + f.x));
    let by = vec3f(0.5 * (0.5 - f.y) * (0.5 - f.y), 0.75 - f.y * f.y, 0.5 * (0.5 + f.y) * (0.5 + f.y));
    for (var j = 0; j < 3; j++) { for (var i = 0; i < 3; i++) {
      L[j * 3 + i] = label(nc.x - 1 + i, nc.y - 1 + j); w[j * 3 + i] = bx[i] * by[j]; } }
    n = 9;
  } else {
    let g = p / CELL - 0.5;
    let i0 = vec2i(floor(g)); let f = g - floor(g);
    L[0] = label(i0.x, i0.y); L[1] = label(i0.x + 1, i0.y); L[2] = label(i0.x, i0.y + 1); L[3] = label(i0.x + 1, i0.y + 1);
    w[0] = (1.0 - f.x) * (1.0 - f.y); w[1] = f.x * (1.0 - f.y); w[2] = (1.0 - f.x) * f.y; w[3] = f.x * f.y;
  }
  for (var a = 0; a < n; a++) { for (var b = a + 1; b < n; b++) { if (L[b] == L[a] && w[b] > 0.0) { w[a] += w[b]; w[b] = 0.0; } } }
  var i1 = 0; for (var a = 1; a < n; a++) { if (w[a] > w[i1]) { i1 = a; } }
  var i2 = -1;
  for (var a = 0; a < n; a++) { if (a != i1 && L[a] != L[i1] && (i2 < 0 || w[a] > w[i2])) { i2 = a; } }
  let c1 = tex(L[i1], p);
  if (i2 < 0 || w[i2] <= 0.0) { return c1; }
  let e = 2.0 * max(px, BLEND) / CELL;
  let a1 = min(1.0, 0.5 + (w[i1] - w[i2]) / max(1e-6, e));
  if (a1 >= 1.0) { return c1; }
  return mix(tex(L[i2], p), c1, a1);
}`;

const fx = (v) => (Number.isFinite(v) ? (Math.abs(v) < 1e-30 ? "0.0" : v.toPrecision(5)) : "0.0");
const fxf = (v) => { const s = fx(v); return /[.e]/.test(s) ? s : s + ".0"; };

function header(sc) {
  return `const W0 = ${fxf(sc.W0)}; const H0 = ${fxf(sc.H0)};
const LW = ${sc.LW}u; const LH = ${sc.LH}u; const CELL = ${fxf(sc.W0 / sc.LW)};
const NR = ${sc.NR}u; const BITS = ${sc.packed.bits}u; const MODE = ${MODES[sc.mode]}u; const BLEND = ${fxf(sc.blend)};`;
}

// The standalone file: every number is a const. Arrays are cut at 2,000 elements (WebKit refuses
// const arrays of more than 2,047).
export function bakeCollage(sc, { title = "texture collage" } = {}) {
  const parts = [];
  const words = sc.packed.words, nv = words.length / 4, CH = 2000;
  const chunks = [];
  for (let s = 0; s < nv; s += CH) {
    const n = Math.min(CH, nv - s), items = [];
    for (let i = 0; i < n; i++) { const o = 4 * (s + i); items.push(`vec4u(${words[o]}u,${words[o + 1]}u,${words[o + 2]}u,${words[o + 3]}u)`); }
    chunks.push({ s, n, code: `const LABELS${chunks.length} = array<vec4u, ${n}>(${items.join(",")});` });
  }
  let sel = "";
  chunks.forEach((c, i) => {
    sel += i < chunks.length - 1 ? `  if (v < ${c.s + c.n}u) { q = LABELS${i}[v - ${c.s}u]; } else ` : `  { q = LABELS${i}[v - ${c.s}u]; }\n`;
  });
  parts.push(`// ${title}: ${sc.NR} regions of splat-diffusion textures (${sc.K} splats each), no pixels.
// Generated by texture-collage/splat (oaustegard/experiments). Call collage(p, px) from a fragment shader:
// p in reference pixels (0..W0, 0..H0), px = reference pixels per output pixel (for anti-aliased edges).
${header(sc)}
${chunks.map((c) => c.code).join("\n")}
fn labelWord(i: u32) -> u32 {
  let v = i / 4u;
  var q: vec4u;
${sel}  return q[i % 4u];
}`);
  const xfa = sc.xf.map((x) => `vec4f(${x.m.map(fxf).join(",")})`), ofa = sc.xf.map((x) => `vec2f(${x.off.map(fxf).join(",")})`);
  parts.push(`const XF = array<vec4f, ${sc.NR}>(${xfa.join(",")});
const OFF = array<vec2f, ${sc.NR}>(${ofa.join(",")});`);
  const K = sc.K, cases = [];
  sc.splats.forEach((sp, r) => {
    const a = [], b = [], c = [];
    for (let k = 0; k < K; k++) {
      const o = k * 12;
      a.push(`vec4f(${fxf(sp[o])},${fxf(sp[o + 1])},${fxf(sp[o + 2])},${fxf(sp[o + 3])})`);
      b.push(`vec4f(${fxf(sp[o + 4])},${fxf(sp[o + 5])},${fxf(sp[o + 6])},${fxf(sp[o + 7])})`);
      c.push(`vec2f(${fxf(sp[o + 8])},${fxf(sp[o + 9])})`);
    }
    parts.push(`const R${r}A = array<vec4f, ${K}>(${a.join(",")});
const R${r}B = array<vec4f, ${K}>(${b.join(",")});
const R${r}C = array<vec2f, ${K}>(${c.join(",")});
fn tex${r}(uv: vec2f) -> vec3f {${LOOP(`${K}u`, `R${r}A[k]`, `R${r}B[k]`, `R${r}C[k]`)}
}`);
    cases.push(`    case ${r}u: { return tex${r}(uv); }`);
  });
  parts.push(`fn tex(r: u32, p: vec2f) -> vec3f {
  let x = XF[r];
  let uv = fract(vec2f(x.x * p.x + x.y * p.y, x.z * p.x + x.w * p.y) + OFF[r]);
  switch r {
${cases.join("\n")}
    default: { return vec3f(0.0); }
  }
}`);
  parts.push(COLLAGE);
  return parts.join("\n");
}

// The page's variant: the same functions over storage buffers, so a new texture or recolour is a
// buffer write, not a shader compile. Bindings (group 0): 1 SP (vec4f, 3 per splat: A, B, C.xy),
// 2 LBL (u32 words), 3 XF (vec4f, 2 per region: matrix, offset.xy).
export function storageShaderHeader() {
  return `
struct U { W0: f32, H0: f32, LW: u32, LH: u32, NR: u32, BITS: u32, MODE: u32, K: u32,
           BLEND: f32, x0: f32, y0: f32, scale: f32 }
@group(0) @binding(0) var<uniform> uu: U;
@group(0) @binding(1) var<storage, read> SP: array<vec4f>;
@group(0) @binding(2) var<storage, read> LBL: array<u32>;
@group(0) @binding(3) var<storage, read> XFB: array<vec4f>;`;
}

export function storageShader() {
  const body = `
fn labelWord(i: u32) -> u32 { return LBL[i]; }
fn tex(r: u32, p: vec2f) -> vec3f {
  let x = XFB[2u * r];
  let uv = fract(vec2f(x.x * p.x + x.y * p.y, x.z * p.x + x.w * p.y) + XFB[2u * r + 1u].xy);
  let base = r * uu.K * 3u;${LOOP("uu.K", "SP[base + 3u * k]", "SP[base + 3u * k + 1u]", "SP[base + 3u * k + 2u]")}
}`;
  const collage = COLLAGE.replace(/\bW0\b/g, "uu.W0").replace(/\bH0\b/g, "uu.H0").replace(/\bLW\b/g, "uu.LW")
    .replace(/\bLH\b/g, "uu.LH").replace(/\bBITS\b/g, "uu.BITS").replace(/\bMODE\b/g, "uu.MODE").replace(/\bBLEND\b/g, "uu.BLEND")
    .replace(/\bCELL\b/g, "(uu.W0 / f32(uu.LW))");
  return `${storageShaderHeader()}${body}${collage}
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2f(uu.x0, uu.y0) + pos.xy * uu.scale;
  return vec4f(collage(p, uu.scale), 1.0);
}`;
}

// Buffers for storageShader().
export function sceneBuffers(sc) {
  const K = sc.K, sp = new Float32Array(sc.NR * K * 12);
  sc.splats.forEach((s, r) => {
    for (let k = 0; k < K; k++) {
      const o = k * 12, d = (r * K + k) * 12;
      sp[d] = s[o]; sp[d + 1] = s[o + 1]; sp[d + 2] = s[o + 2]; sp[d + 3] = s[o + 3];
      sp[d + 4] = s[o + 4]; sp[d + 5] = s[o + 5]; sp[d + 6] = s[o + 6]; sp[d + 7] = s[o + 7];
      sp[d + 8] = s[o + 8]; sp[d + 9] = s[o + 9]; sp[d + 10] = 0; sp[d + 11] = 0;
    }
  });
  const xf = new Float32Array(sc.NR * 8);
  sc.xf.forEach((x, r) => { xf.set(x.m, 8 * r); xf[8 * r + 4] = x.off[0]; xf[8 * r + 5] = x.off[1]; });
  return { sp, labels: sc.packed.words, xf };
}

// Uniforms for storageShader(): view maps output pixel (x, y) to reference p = (x0, y0) + (x, y) * scale.
export function sceneUniforms(sc, { x0 = 0, y0 = 0, scale = 1 } = {}) {
  const ab = new ArrayBuffer(48), dv = new DataView(ab);
  dv.setFloat32(0, sc.W0, true); dv.setFloat32(4, sc.H0, true); dv.setUint32(8, sc.LW, true); dv.setUint32(12, sc.LH, true);
  dv.setUint32(16, sc.NR, true); dv.setUint32(20, sc.packed.bits, true); dv.setUint32(24, MODES[sc.mode], true);
  dv.setUint32(28, sc.K, true); dv.setFloat32(32, sc.blend, true); dv.setFloat32(36, x0, true); dv.setFloat32(40, y0, true);
  dv.setFloat32(44, scale, true);
  return ab;
}

// ---------------------------------------------------------------- the whole build, from a segmentation
// seg/rdesc/pick from collage.js; bank(cls) -> { splats (K x 12), stats (labStats of the painted tile),
// theta, gcoh } for the region's texture. Returns a scene.
export function buildScene({ lab, seg, rdesc, pick, tex, K, LW = 192, mode = "distance", blend = 0, tint = "region",
  strength = 8, align = true, seed = 1 }) {
  const rnd = mulberry32(seed);
  const regions = pick.map((p, k) => {
    const t = tex[k], rd = rdesc[k];
    let phi = 0;
    if (align && rd && t.gcoh > 0.15 && rd.f[3] > 0.15) phi = rd.theta - t.theta;
    return { splats: recolourSplats(t.splats, K, t.stats, seg.stats[k], { tint, strength }), phi, scale: p.scale, off: [rnd(), rnd()] };
  });
  return makeScene({ W0: seg.W, H0: seg.H, lm: labelMap(seg, LW), regions, K, mode, blend });
}

// Lab stats, orientation and coherence of a painted splat tile at the scale it will be shown, as
// collage.js prepTile gives for pixel tiles.
export function describeSplatTile(sp, K, scale = 1, S = 128) {
  const rgb = paintTile(sp, K, S), lab = rgbToLab(rgb, S * S, 3), stats = labStats(lab, S * S);
  const T = Math.round(S * scale), L = new Float32Array(T * T);
  // nearest-free resample for the descriptor at this scale: repaint at T px (exact for splats)
  const rgb2 = scale === 1 ? rgb : paintTile(sp, K, T), lab2 = scale === 1 ? lab : rgbToLab(rgb2, T * T, 3);
  for (let i = 0; i < T * T; i++) L[i] = lab2[3 * i];
  const d = describe(L, null, T, T, true);
  return { stats, theta: d.theta, gcoh: d.f[3], f: d.f, rgb };
}

// A single-file viewer: the baked WGSL plus a few lines of WebGPU, with wheel or pinch zoom and drag pan.
export function viewerHTML(src, W0, H0) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Splat collage</title><style>html,body{margin:0;height:100%;background:#0b0b0d;overflow:hidden}canvas{width:100vw;height:100vh;display:block;touch-action:none;cursor:grab}
p{position:fixed;left:12px;bottom:8px;margin:0;font:12px system-ui;color:#999}</style></head><body><canvas></canvas>
<p>A collage of splat textures, drawn by one WGSL shader with no pixels in it. Scroll to zoom, drag to pan. Made with texture-collage/splat (oaustegard/experiments).</p>
<script type="module">
const WGSL = ${JSON.stringify(src)} + \`
struct V { x0: f32, y0: f32, scale: f32, _p: f32 }
@group(0) @binding(0) var<uniform> view: V;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(xy * 2.0 - 1.0, 0.0, 1.0); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(collage(vec2f(view.x0, view.y0) + pos.xy * view.scale, view.scale), 1.0); }\`;
const cv = document.querySelector("canvas");
if (!navigator.gpu) { document.body.innerHTML = "<p style='color:#ccc;padding:20px'>This file needs a browser with WebGPU.</p>"; throw 0; }
const device = await (await navigator.gpu.requestAdapter()).requestDevice();
const ctx = cv.getContext("webgpu"), format = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format, alphaMode: "opaque" });
const mod = device.createShaderModule({ code: WGSL });
const errs = (await mod.getCompilationInfo()).messages.filter((m) => m.type === "error");
if (errs.length) { document.body.innerHTML = "<p style='color:#ccc;padding:20px'>Shader error: " + errs[0].message + "</p>"; throw new Error(errs[0].message); }
const pipe = await device.createRenderPipelineAsync({ layout: "auto", vertex: { module: mod, entryPoint: "vs" }, fragment: { module: mod, entryPoint: "fs", targets: [{ format }] } });
const u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: u } }] });
const W0 = ${W0}, H0 = ${H0}, v = { x0: 0, y0: 0, s: 1 };
function size() { const d = Math.min(devicePixelRatio || 1, 2); cv.width = Math.round(innerWidth * d); cv.height = Math.round(innerHeight * d); }
function fit() { size(); v.s = Math.max(W0 / cv.width, H0 / cv.height) * 1.04; v.x0 = W0 / 2 - cv.width / 2 * v.s; v.y0 = H0 / 2 - cv.height / 2 * v.s; draw(); }
let q = false;
function draw() { if (q) return; q = true; requestAnimationFrame(() => { q = false;
  device.queue.writeBuffer(u, 0, new Float32Array([v.x0, v.y0, v.s, 0]));
  const e = device.createCommandEncoder(), p = e.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0.04, 0.04, 0.05, 1] }] });
  p.setPipeline(pipe); p.setBindGroup(0, bg); p.draw(3); p.end(); device.queue.submit([e.finish()]);
  device.queue.onSubmittedWorkDone().then(() => { window.drawn = (window.drawn || 0) + 1; }); }); }
function zoomAt(x, y, f) { const d = cv.width / innerWidth, px = x * d, py = y * d; v.x0 += px * v.s * (1 - f); v.y0 += py * v.s * (1 - f); v.s *= f; draw(); }
addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY * 0.0015)); }, { passive: false });
const pts = new Map(); let pinch = 0;
cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); });
cv.addEventListener("pointerup", (e) => { pts.delete(e.pointerId); pinch = 0; });
cv.addEventListener("pointermove", (e) => { const o = pts.get(e.pointerId); if (!o) return; const d = cv.width / innerWidth;
  if (pts.size === 2) { pts.set(e.pointerId, [e.clientX, e.clientY]); const [a, b] = [...pts.values()]; const dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (pinch) zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, pinch / dist); pinch = dist; return; }
  v.x0 -= (e.clientX - o[0]) * d * v.s; v.y0 -= (e.clientY - o[1]) * d * v.s; pts.set(e.pointerId, [e.clientX, e.clientY]); draw(); });
addEventListener("resize", fit); fit();
</script></body></html>`;
}
