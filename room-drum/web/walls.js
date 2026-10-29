// Camera frame -> one wall bit per grid cell, on the CPU. Runs in a Worker (walls-worker.js) for the page, and
// directly in ../bench/walls-job.js for timing.
//
// The page draws each camera frame into a 2D canvas at twice the grid's resolution (once, for grids over 8k cells;
// the canvas route keeps iOS frames upright, METHODS.md, xray-glasses) and hands it over. Here it becomes luma.
// Edges: Sobel magnitude, max-pooled to the cell. Dark / bright: the cell's mean luma against the threshold.
//
// Walls settle rather than follow the picture. A cell's score is a slow average (about a third of a second at 30
// fps), and a cell where the image is moving keeps its score, so a waving hand strikes the drum without repainting
// it, and a hand held still becomes a wall. A cell turns to wall above the threshold and back to skin only below
// 0.75 of it. Isolated wall specks are dropped and skin islands too small to be a drum are filled in.
//
// The skin is then split into connected regions (4-neighbour), and the largest nine become pads: each pad's strike
// point is its deepest cell, the one farthest from any wall, where a strike excites the lowest mode most.
export const MAX_PADS = 9;

export class Walls {
  constructor(nx, ny) {
    const s = nx * ny > 8192 ? 1 : 2, n = nx * ny;
    Object.assign(this, { nx, ny, s, w: s * nx, h: s * ny, warm: 0 });
    this.canvas = new OffscreenCanvas(this.w, this.h); this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.luma = new Float32Array(this.w * this.h); this.prevL = new Float32Array(this.w * this.h); this.score = new Float32Array(n);
    this.wall = new Uint8Array(n); this.mask = new Float32Array(n).fill(1); this.label = new Int32Array(n); this.sizes = [n];
    this.motion = new Float32Array(n); this.q = new Int32Array(n); this.dist = new Int32Array(n); this.pads = [];
  }
  // src: 'edge' | 'dark' | 'bright'; thr in 0..1; settle in 0..1 is the per-frame weight of a new frame in the
  // score. With freeze the walls hold and only motion is updated. Returns true when the walls were recomputed.
  frame(image, { src = "edge", thr = 0.5, settle = 0.12, freeze = false } = {}) {
    const { nx, ny, w, h, s } = this, n = nx * ny;
    this.ctx.drawImage(image, 0, 0, w, h);
    const px = this.ctx.getImageData(0, 0, w, h).data, L = this.luma, Lp = this.prevL;
    for (let i = 0, j = 0; i < L.length; i++, j += 4) { Lp[i] = L[i]; L[i] = (0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2]) / 255; }
    const raw = this.score, mo = this.motion, edge = src === "edge";
    const at = (x, y) => L[(y < 0 ? 0 : y >= h ? h - 1 : y) * w + (x < 0 ? 0 : x >= w ? w - 1 : x)];
    const eScale = 0.15 + 1.6 * thr, lum = 0.2 + 0.6 * thr;
    for (let cy = 0; cy < ny; cy++) for (let cx = 0; cx < nx; cx++) {
      let g = 0, m = 0, sum = 0;
      for (let dy = 0; dy < s; dy++) for (let dx = 0; dx < s; dx++) {
        const x = s * cx + dx, y = s * cy + dy, i = y * w + x;
        m += Math.abs(L[i] - Lp[i]); sum += L[i];
        if (edge) {
          const gx = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
          const gy = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
          g = Math.max(g, Math.hypot(gx, gy));
        }
      }
      const k = cy * nx + cx, mean = sum / (s * s);
      const v = edge ? g / eScale : src === "dark" ? lum / Math.max(1e-3, mean) : mean / lum;   // > 1 means wall
      mo[k] = m / (s * s);
      if (!this.warm) raw[k] = v;
      else if (mo[k] < 0.04) raw[k] += settle * (v - raw[k]);   // moving cells keep their score
    }
    this.warm++;
    if (freeze) return false;
    const wall = this.wall;
    for (let k = 0; k < n; k++) wall[k] = raw[k] > 1 ? 1 : raw[k] < 0.75 ? 0 : wall[k];
    this.despeckle();
    this.labelRegions();
    const minArea = Math.max(8, Math.round(0.002 * n));   // fill skin islands smaller than this
    if (this.sizes.some((a) => a < minArea)) {
      for (let k = 0; k < n; k++) if (!wall[k] && this.sizes[this.label[k]] < minArea) wall[k] = 1;
      this.labelRegions();
    }
    for (let k = 0; k < n; k++) this.mask[k] = wall[k] ? 0 : 1;
    this.findPads();
    return true;
  }
  despeckle() {   // a wall cell with fewer than two wall neighbours (of eight) goes back to skin
    const { nx, ny, wall } = this;
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const k = y * nx + x; if (!wall[k]) continue;
      let c = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < nx && yy < ny && wall[yy * nx + xx]) c++;
      }
      if (c < 2) wall[k] = 0;
    }
  }
  labelRegions() {
    const { nx, label, wall, q } = this, n = nx * this.ny; label.fill(-1); this.sizes = [];
    for (let s = 0; s < n; s++) {
      if (wall[s] || label[s] >= 0) continue;
      const id = this.sizes.length; let h = 0, t = 0; q[t++] = s; label[s] = id;
      while (h < t) {
        const i = q[h++], x = i % nx;
        if (x > 0 && !wall[i - 1] && label[i - 1] < 0) { label[i - 1] = id; q[t++] = i - 1; }
        if (x + 1 < nx && !wall[i + 1] && label[i + 1] < 0) { label[i + 1] = id; q[t++] = i + 1; }
        if (i >= nx && !wall[i - nx] && label[i - nx] < 0) { label[i - nx] = id; q[t++] = i - nx; }
        if (i + nx < n && !wall[i + nx] && label[i + nx] < 0) { label[i + nx] = id; q[t++] = i + nx; }
      }
      this.sizes.push(t);
    }
  }
  // Distance to the nearest wall or grid edge (4-neighbour steps), then each large region's deepest cell.
  findPads() {
    const { nx, ny, wall, dist, q, label } = this, n = nx * ny; let h = 0, t = 0;
    for (let k = 0; k < n; k++) {
      const x = k % nx, y = (k / nx) | 0;
      if (wall[k]) dist[k] = 0;
      else if (x === 0 || y === 0 || x === nx - 1 || y === ny - 1) { dist[k] = 1; q[t++] = k; }
      else dist[k] = -1;
    }
    for (let k = 0; k < n; k++) if (!wall[k] && dist[k] < 0) {   // skin next to a wall
      const x = k % nx;
      if ((x > 0 && wall[k - 1]) || (x + 1 < nx && wall[k + 1]) || wall[k - nx] || wall[k + nx]) { dist[k] = 1; q[t++] = k; }
    }
    while (h < t) {
      const i = q[h++], x = i % nx;
      for (const j of [x > 0 ? i - 1 : -1, x + 1 < nx ? i + 1 : -1, i - nx, i + nx]) if (j >= 0 && j < n && dist[j] < 0) { dist[j] = dist[i] + 1; q[t++] = j; }
    }
    const deep = new Map();
    for (let k = 0; k < n; k++) if (!wall[k]) { const id = label[k], d = deep.get(id); if (!d || dist[k] > dist[d.i]) deep.set(id, { i: k }); }
    this.pads = [...deep.entries()].map(([id, d]) => ({ id, i: d.i, area: this.sizes[id], depth: dist[d.i] }))
      .sort((a, b) => b.area - a.area).slice(0, MAX_PADS);
  }
  nearestSkin(x, y, rmax) {   // grid coordinates -> the closest skin cell, or null
    return nearestSkin(this, x, y, rmax);
  }
  pickupNear(i0, dx, dy) {
    return pickupNear(this, i0, dx, dy);
  }
}

// The page keeps a plain copy of the worker's result ({nx, ny, wall, label}); these work on either.
export function nearestSkin(W, x, y, rmax) {
  const { nx, ny } = W; let best = null, bd = 1e9;
  for (let dy = -rmax; dy <= rmax; dy++) for (let dx = -rmax; dx <= rmax; dx++) {
    const cx = Math.round(x) + dx, cy = Math.round(y) + dy, d = dx * dx + dy * dy;
    if (cx < 0 || cy < 0 || cx >= nx || cy >= ny || d >= bd || W.wall[cy * nx + cx]) continue;
    best = cy * nx + cx; bd = d;
  }
  return best;
}
export function pickupNear(W, i0, dx, dy) {   // a cell of the same region roughly (dx, dy) away
  const { nx } = W, id = W.label[i0], x0 = i0 % nx, y0 = (i0 / nx) | 0;
  for (let s = 1; s > 0.1; s -= 0.2) {
    const j = nearestSkin(W, x0 + dx * s, y0 + dy * s, 1);
    if (j !== null && W.label[j] === id && j !== i0) return j;
  }
  return i0;
}
