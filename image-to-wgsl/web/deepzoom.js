// Deep zoom for a splat list: detail tiles trained from a higher-resolution source.
//
// The base list is fitted at the page's working size (512 or 768 px). Zooming past that redraws the same splats,
// smooth but with no new detail. Here the source is kept (up to a cap) and cut into a pyramid, as deep-zoom image
// viewers do: level L has 2^L times the base's pixels per side, and the last level is the source itself. Each level
// is cut into 256-pixel tiles. A tile is a small splat list trained by the WebGPU trainer on its crop of the level,
// widened by a 16-pixel margin, painting over what the base and the coarser levels already paint there (their
// render is the trainer's background). So a tile only has to add the detail its level brings. When drawn, each tile
// is clipped to its 256-pixel core: inside the core the screen shows exactly what the tile was trained to show, and
// the margin only keeps splats near the edge honest. A tile depends on the coarser tiles under its crop, which are
// trained first.
//
// The page asks for the tiles a view needs (want), and they train one at a time in the background, nearest the
// centre of the view first. The renderer draws the detail splats of every level up to the one the view needs.
import { encodeGPU } from "./gputrain.js";
import { LAYOUT } from "./trainer.js";

const B = LAYOUT.bits, MAX = B.map((b) => 2 ** b - 1), SPAN = (LAYOUT.sHi - LAYOUT.sLo) / MAX[2];
export const TILE = 256, MARGIN = 16;
export const TILE_STEPS = { stepsStage: 150, stepsFinal: 500, stepsQat: 200 };

// Levels above a base of W0 x H0 for a source of Ws x Hs: 2, 4, 8 ... times the base, then the source itself. A level
// must add at least 1.25 times the resolution below it; a smaller last step is folded into the source level.
export function pyramid(W0, H0, Ws, Hs, maxLevels = 6) {
  const R = Ws / W0, out = [];
  let prev = 1;
  for (let L = 1; L <= maxLevels; L++) {
    const s = 2 ** L;
    if (s * 1.25 < R && L < maxLevels) { out.push({ L, s, W: Math.round(Ws * s / R), H: Math.round(Hs * s / R) }); prev = s; continue; }
    if (R > prev * 1.25) out.push({ L, s: R, W: Ws, H: Hs });
    break;
  }
  for (const lv of out) { lv.TX = Math.ceil(lv.W / TILE); lv.TY = Math.ceil(lv.H / TILE); }
  return out;
}

// Core and training crop of tile (tx, ty) of a level, in its pixels: [x0, y0, x1, y1), x1 and y1 exclusive.
export function tileGeom(lv, tx, ty, margin = MARGIN) {
  const core = [tx * TILE, ty * TILE, Math.min(lv.W, (tx + 1) * TILE), Math.min(lv.H, (ty + 1) * TILE)];
  const crop = [Math.max(0, core[0] - margin), Math.max(0, core[1] - margin), Math.min(lv.W, core[2] + margin), Math.min(lv.H, core[3] + margin)];
  return { core, crop };
}

// Camera that maps a crop [x0, y0, x1, y1) of a W x H level onto a texture of its own size, exactly, for a base
// list of aspect A (renderer units: x in [0, A], y in [0, 1]).
export function cropCam(lv, crop, A) {
  const w = crop[2] - crop[0], h = crop[3] - crop[1];
  return { z: lv.H / Math.min(w / A, h), x: (crop[0] + w / 2) / lv.W, y: (crop[1] + h / 2) / lv.H, sx: lv.W / (A * lv.H) };
}

// A trained tile's integer fields, decoded into the renderer's image units (12 floats per splat).
export function decodeTile(P, n, lv, crop, A, tileIndex) {
  const w = crop[2] - crop[0], h = crop[3] - crop[1], At = w / h, S = h / lv.H, out = new Float32Array(n * 12);
  for (let i = 0; i < n; i++) {
    const q = P.subarray(i * 10, i * 10 + 10), o = i * 12;
    const cx = q[0] * At / MAX[0], cy = q[1] / MAX[1], th = q[4] * Math.PI / 2 ** B[4];
    out[o] = (crop[0] + cx * h) / lv.W * A; out[o + 1] = (crop[1] + cy * h) / lv.H;
    out[o + 2] = 2 ** (LAYOUT.sLo + q[2] * SPAN) * S; out[o + 3] = 2 ** (LAYOUT.sLo + q[3] * SPAN) * S;
    out[o + 4] = Math.cos(th); out[o + 5] = Math.sin(th); out[o + 6] = q[9]; out[o + 7] = (q[8] + 1) / 2 ** B[8];
    out[o + 8] = q[5] / MAX[5]; out[o + 9] = q[6] / MAX[6]; out[o + 10] = q[7] / MAX[7]; out[o + 11] = tileIndex;
  }
  return out;
}

// Sum of squared 8-bit errors of a float picture against an RGBA byte target, over the core of a crop.
export function coreSSE(img, tgt, crop, core) {
  const w = crop[2] - crop[0]; let s = 0;
  for (let y = core[1]; y < core[3]; y++) for (let x = core[0]; x < core[2]; x++) {
    const i = ((y - crop[1]) * w + (x - crop[0])) * 4;
    for (let c = 0; c < 3; c++) { const d = Math.round(Math.min(1, Math.max(0, img[i + c])) * 255) - tgt[i + c]; s += d * d; }
  }
  return s;
}

export class DeepZoom {
  // base: the base list {P, bg, A, n} as the renderer draws it, fitted at W0 x H0. Ws x Hs: the source.
  // crop(lv, x0, y0, x1, y1): RGBA bytes of that rectangle of level lv (W x H), resampled from the source.
  // nTile: splats for a full tile (edge tiles get fewer, by area); prefix: train every power-of-two prefix.
  constructor(dev, { base, W0, H0, Ws, Hs, crop, nTile = 256, prefix = false, steps = TILE_STEPS, margin = MARGIN, maxLevels = 6 }) {
    Object.assign(this, { dev, base, W0, H0, Ws, Hs, crop, nTile, prefix, steps, margin });
    this.A = base.A;
    this.levels = pyramid(W0, H0, Ws, Hs, maxLevels);
    this.tiles = new Map();          // "L,tx,ty" -> {L, tx, ty, core, crop, n, P, seconds, ...}
    this.queue = []; this.busy = false; this.rend = null; this.onTile = null; this.paused = false;
    this.counts = [0];               // counts[L]: detail splats in levels 1..L (the draw prefix for level L)
    this.bytes = 0;
  }
  get maxLevel() { return this.levels.length; }
  key(L, tx, ty) { return `${L},${tx},${ty}`; }
  attach(rend) { this.rend = rend; this.upload(); }

  // The level a view needs: device pixels per base pixel d means level ceil(log2 d), up to the source.
  levelFor(density) {
    if (density <= 1.05) return 0;
    let L = 0;
    for (const lv of this.levels) { L = lv.L; if (lv.s >= density / 1.05) break; }
    return L;
  }
  // Detail splats to draw for a view at level L: every trained tile up to L.
  countFor(L) { return this.counts[Math.min(L, this.counts.length - 1)] ?? 0; }

  // Tiles of level L under a view rectangle (fractions of the image), nearest its centre first.
  visible(L, fx0, fy0, fx1, fy1) {
    const lv = this.levels[L - 1], out = [];
    const t0 = [Math.max(0, Math.floor(fx0 * lv.W / TILE)), Math.max(0, Math.floor(fy0 * lv.H / TILE))];
    const t1 = [Math.min(lv.TX - 1, Math.floor(fx1 * lv.W / TILE)), Math.min(lv.TY - 1, Math.floor(fy1 * lv.H / TILE))];
    const cx = (fx0 + fx1) / 2 * lv.W / TILE - 0.5, cy = (fy0 + fy1) / 2 * lv.H / TILE - 0.5;
    for (let ty = t0[1]; ty <= t1[1]; ty++) for (let tx = t0[0]; tx <= t1[0]; tx++) out.push([L, tx, ty, (tx - cx) ** 2 + (ty - cy) ** 2]);
    return out.sort((a, b) => a[3] - b[3]).map((t) => t.slice(0, 3));
  }
  // Coarser tiles a tile's crop lies on: they paint its background, so they train first.
  deps(L, tx, ty) {
    if (L < 2) return [];
    const lv = this.levels[L - 1], lo = this.levels[L - 2], { crop } = tileGeom(lv, tx, ty, this.margin), out = [];
    const x0 = Math.floor(crop[0] / lv.W * lo.W / TILE), x1 = Math.min(lo.TX - 1, Math.floor((crop[2] - 1e-6) / lv.W * lo.W / TILE));
    const y0 = Math.floor(crop[1] / lv.H * lo.H / TILE), y1 = Math.min(lo.TY - 1, Math.floor((crop[3] - 1e-6) / lv.H * lo.H / TILE));
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) out.push([L - 1, x, y]);
    return out;
  }
  // Replace the wanted set: these tiles (and what they depend on) train next, in this order.
  want(list) {
    const seen = new Set(), order = [];
    const add = (t) => { const k = this.key(...t); if (seen.has(k) || this.tiles.has(k)) return; seen.add(k); for (const d of this.deps(...t)) add(d); order.push(t); };
    for (const t of list) add(t);
    this.queue = order;
    this.pump();
  }
  pending() { return this.queue.length + (this.busy ? 1 : 0); }
  async pump() {
    if (this.busy || this.paused) return;
    this.busy = true;
    try {
      while (this.queue.length && !this.paused) {
        const t = this.queue.shift();
        if (this.tiles.has(this.key(...t))) continue;
        if (this.deps(...t).some((d) => !this.tiles.has(this.key(...d)))) { this.want([t, ...this.queue]); continue; }
        const rec = await this.trainTile(...t);
        this.onTile?.(rec);
      }
    } catch (e) { this.error = e; this.onTile?.({ error: e }); }
    finally { this.busy = false; }
  }

  // Train one tile over the current render of the levels below it. Its dependencies must be trained.
  async trainTile(L, tx, ty, extra = {}) {
    const lv = this.levels[L - 1], { core, crop } = tileGeom(lv, tx, ty, this.margin);
    const w = crop[2] - crop[0], h = crop[3] - crop[1];
    const t0 = performance.now();
    const rgba = await this.crop(lv, ...crop);
    const t1 = performance.now();
    const bg = await this.rend.readback(w, h, this.base.n, cropCam(lv, crop, this.A), this.counts[L - 1] ?? 0);
    const mask = new Float32Array(w * h);
    for (let y = core[1]; y < core[3]; y++) for (let x = core[0]; x < core[2]; x++) mask[(y - crop[1]) * w + (x - crop[0])] = 1;
    const full = (TILE + 2 * this.margin) ** 2, n = Math.max(16, Math.round(this.nTile * Math.min(1, w * h / full)));
    const t2 = performance.now();
    const r = await encodeGPU(this.dev, { rgba, width: w, height: h }, { n, ...this.steps, prefix: this.prefix, bgImg: bg, focusEval: mask, seed: L * 7919 + ty * 131 + tx, ...extra });
    const rec = { L, tx, ty, core, crop, n, P: r.P, points: r.points, seconds: r.seconds, cropMs: t1 - t0, bgMs: t2 - t1,
      totalMs: performance.now() - t0, coreN: (core[2] - core[0]) * (core[3] - core[1]),
      sseBelow: coreSSE(bg, rgba, crop, core), psnr: r.points[r.points.length - 1].psnrFocus };
    this.tiles.set(this.key(L, tx, ty), rec);
    this.bytes += Math.ceil(n * B.reduce((a, b) => a + b, 0) / 8);
    this.upload();
    return rec;
  }

  // Rebuild the renderer's detail list: tiles by level, coarse first.
  upload() {
    const recs = [...this.tiles.values()].sort((a, b) => a.L - b.L);
    let N = 0; for (const r of recs) N += r.n;
    const spl = new Float32Array(N * 12), rects = new Float32Array(Math.max(1, recs.length) * 4);
    this.counts = [0]; let o = 0;
    recs.forEach((r, i) => {
      const lv = this.levels[r.L - 1];
      spl.set(decodeTile(r.P, r.n, lv, r.crop, this.A, i), o * 12); o += r.n;
      rects.set([r.core[0] / lv.W * this.A, r.core[1] / lv.H, r.core[2] / lv.W * this.A, r.core[3] / lv.H], i * 4);
      while (this.counts.length <= r.L) this.counts.push(this.counts[this.counts.length - 1]);
      this.counts[r.L] = o;
    });
    while (this.counts.length <= this.levels.length) this.counts.push(this.counts[this.counts.length - 1]);
    this.rend?.setDetail(spl, rects);
  }
  stats() {
    const per = this.levels.map((lv) => ({ L: lv.L, W: lv.W, H: lv.H, tiles: lv.TX * lv.TY, trained: 0, splats: 0 }));
    for (const r of this.tiles.values()) { per[r.L - 1].trained++; per[r.L - 1].splats += r.n; }
    return per;
  }
}
