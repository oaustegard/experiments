// Camera frame -> one wall bit per grid cell, on the CPU. Used by the page and timed by ../bench/walls-job.js.
//
// Each frame is drawn into a 2D canvas (the route that keeps iOS frames upright; METHODS.md, xray-glasses) at twice
// the grid's resolution (once, for grids over 8k cells), turned to luma and smoothed over time. Edges: Sobel
// magnitude, max-pooled to the cell. Dark / bright: the cell's mean luma against the threshold. A cell turns to
// wall above the threshold and back to skin only below 0.75 of it, so a flickering edge does not crackle.
// Skin cells are then split into connected regions (4-neighbour), so a strike's pickups sit in the drum it struck.
export class Walls {
  constructor(nx, ny) {
    const s = nx * ny > 8192 ? 1 : 2;
    Object.assign(this, { nx, ny, s, w: s * nx, h: s * ny, warm: 0 });
    this.canvas = new OffscreenCanvas(this.w, this.h); this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.luma = new Float32Array(this.w * this.h); this.prevL = new Float32Array(this.w * this.h); this.score = new Float32Array(nx * ny);
    this.wall = new Uint8Array(nx * ny); this.mask = new Float32Array(nx * ny).fill(1); this.label = new Int32Array(nx * ny); this.sizes = [nx * ny];
    this.motion = new Float32Array(nx * ny); this.q = new Int32Array(nx * ny);
  }
  // src: 'edge' | 'dark' | 'bright'; thr in 0..1. With freeze the walls hold and only motion is updated.
  frame(image, { src = "edge", thr = 0.5, freeze = false } = {}) {
    const { nx, ny, w, h, s } = this;
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
      raw[k] = this.warm ? 0.5 * raw[k] + 0.5 * v : v;
      mo[k] = m / (s * s);
    }
    this.warm++;
    if (freeze) return false;
    for (let k = 0; k < nx * ny; k++) { this.wall[k] = raw[k] > 1 ? 1 : raw[k] < 0.75 ? 0 : this.wall[k]; this.mask[k] = this.wall[k] ? 0 : 1; }
    this.labelRegions();
    return true;
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
  nearestSkin(x, y, rmax) {   // grid coordinates -> the closest skin cell, or null
    const { nx, ny } = this; let best = null, bd = 1e9;
    for (let dy = -rmax; dy <= rmax; dy++) for (let dx = -rmax; dx <= rmax; dx++) {
      const cx = Math.round(x) + dx, cy = Math.round(y) + dy, d = dx * dx + dy * dy;
      if (cx < 0 || cy < 0 || cx >= nx || cy >= ny || d >= bd || this.wall[cy * nx + cx]) continue;
      best = cy * nx + cx; bd = d;
    }
    return best;
  }
  pickupNear(i0, dx, dy) {   // a cell of the same region roughly (dx, dy) away
    const { nx } = this, id = this.label[i0], x0 = i0 % nx, y0 = (i0 / nx) | 0;
    for (let s = 1; s > 0.1; s -= 0.2) {
      const j = this.nearestSkin(x0 + dx * s, y0 + dy * s, 1);
      if (j !== null && this.label[j] === id && j !== i0) return j;
    }
    return i0;
  }
}
