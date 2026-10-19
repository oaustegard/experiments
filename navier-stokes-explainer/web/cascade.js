// Chapter 5: a schematic cascade of ever-smaller, ever-faster vortex layers, and
// the scaling window in which such a cascade is consistent with bounded energy
// and still outruns viscosity. Canvas2D; no GPU needed.

import { fitCanvas, css } from './plots.js';

const LAYER_SECONDS = 2.4; // screen time per layer

export class Cascade {
  constructor(canvas, readout) {
    this.cv = canvas; this.out = readout;
    this.lambda = 0.5; this.mu = 0.2;
    this.re0 = 2000;
    this.t0 = 1;
    this.screen = 0;
    this.playing = true;
    this.setParams(this.lambda, this.mu);
  }

  setParams(lambda, mu) {
    this.lambda = lambda; this.mu = mu;
    this.centers = [[0, 0]];
    for (let n = 1; n < 60; n++) {
      const th = 2.1 * n + 0.4;
      const [x, y] = this.centers[n - 1];
      const L = lambda ** (n - 1);
      this.centers.push([x + (1 - lambda) * 0.85 * L * Math.cos(th), y + (1 - lambda) * 0.85 * L * Math.sin(th)]);
    }
  }

  derived(n) {
    const { lambda: l, mu: m, t0 } = this;
    const Tstar = t0 / (1 - m);
    const Tn = (t0 * (1 - m ** n)) / (1 - m);
    const ratioRe = (l * l) / m; // Re_{n+1} / Re_n
    const ratioE3 = l ** 5 / (m * m);
    const ratioE2 = l ** 4 / (m * m);
    let dies = null;
    if (ratioRe < 1) dies = Math.ceil(Math.log(this.re0) / Math.log(1 / ratioRe));
    return { Tstar, Tn, ratioRe, ratioE3, ratioE2, dies, omega: m ** -n, L: l ** n, bkm: n + 1 };
  }

  frame(dt) {
    if (this.playing) this.screen += dt;
    const total = this.screen / LAYER_SECONDS;
    const maxLayers = 14;
    const nf = total % maxLayers;
    const n = Math.floor(nf), frac = nf - n;
    const dpr = fitCanvas(this.cv);
    const g = this.cv.getContext('2d');
    const W = this.cv.width, H = this.cv.height;
    g.fillStyle = '#090a0f'; g.fillRect(0, 0, W, H);
    const { lambda: l } = this;
    // Camera: zoom so layer n+frac fills the view, centred between c_n and c_{n+1}.
    const zoomL = l ** (n + frac);
    const cA = this.centers[n], cB = this.centers[n + 1];
    const s = frac * frac * (3 - 2 * frac);
    const cx = cA[0] + (cB[0] - cA[0]) * s, cy = cA[1] + (cB[1] - cA[1]) * s;
    const scale = (Math.min(W, H) * 0.42) / zoomL;
    const toS = (x, y) => [W / 2 + (x - cx) * scale, H / 2 + (y - cy) * scale];
    for (let k = Math.max(0, n - 2); k < n + 6; k++) {
      const L = l ** k;
      const [px, py] = toS(...this.centers[k]);
      const R = L * scale;
      if (R < 1.5 || R > 40 * Math.max(W, H)) continue;
      const hue = (200 + k * 37) % 360;
      const active = k === n;
      const grad = g.createRadialGradient(px, py, 0, px, py, R);
      grad.addColorStop(0, `hsla(${hue}, 80%, 55%, ${active ? 0.35 : 0.16})`);
      grad.addColorStop(1, `hsla(${hue}, 80%, 55%, 0)`);
      g.fillStyle = grad; g.beginPath(); g.arc(px, py, R, 0, 2 * Math.PI); g.fill();
      // Swirl arms. Every layer turns once per its own turnover time; in screen
      // time that is the same rate for all, because each layer gets equal screen time.
      const spin = (this.screen / LAYER_SECONDS) * 2 * Math.PI * (k % 2 ? -1 : 1);
      g.strokeStyle = `hsla(${hue}, 90%, ${active ? 70 : 60}%, ${active ? 0.95 : 0.55})`;
      g.lineWidth = Math.max(1, (active ? 2.4 : 1.4) * dpr);
      for (let a = 0; a < 6; a++) {
        g.beginPath();
        for (let q = 0; q <= 24; q++) {
          const u = q / 24;
          const r = R * (0.25 + 0.6 * u);
          const th = spin + (a * Math.PI) / 3 + 1.6 * u;
          const x = px + r * Math.cos(th), y = py + r * Math.sin(th);
          if (q) g.lineTo(x, y); else g.moveTo(x, y);
        }
        g.stroke();
      }
    }
    // Timeline: layer durations t_n laid end to end, ending at T*.
    const d = this.derived(n);
    const tNow = d.Tn + frac * this.t0 * this.mu ** n;
    const bx = 16 * dpr, bw = W - 32 * dpr, by = H - 26 * dpr;
    g.fillStyle = 'rgba(255,255,255,0.08)'; g.fillRect(bx, by, bw, 8 * dpr);
    let acc = 0;
    for (let k = 0; k < 30; k++) {
      const len = this.t0 * this.mu ** k;
      const x0 = bx + (acc / d.Tstar) * bw, x1 = bx + ((acc + len) / d.Tstar) * bw;
      g.fillStyle = `hsla(${(200 + k * 37) % 360}, 80%, 55%, ${k === n ? 1 : 0.5})`;
      g.fillRect(x0, by, Math.max(0.5, x1 - x0 - 1), 8 * dpr);
      acc += len;
    }
    g.fillStyle = '#fff'; g.fillRect(bx + (tNow / d.Tstar) * bw - 1 * dpr, by - 4 * dpr, 2 * dpr, 16 * dpr);
    g.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
    g.fillStyle = 'rgba(255,255,255,0.75)'; g.textAlign = 'right'; g.textBaseline = 'bottom';
    g.fillText('T*', bx + bw, by - 4 * dpr);
    g.textAlign = 'left'; g.fillText('t = 0', bx, by - 4 * dpr);
    if (this.out) {
      const fmt = (v) => (v >= 1e4 || v < 1e-3 ? v.toExponential(1) : v.toPrecision(3));
      this.out.innerHTML =
        `<span>layer <b>${n}</b></span><span>size <b>${fmt(d.L)}</b></span>` +
        `<span>vorticity <b>${fmt(d.omega)}×</b></span><span>t <b>${tNow.toFixed(4)}</b> of T* = ${d.Tstar.toFixed(4)}</span>` +
        `<span>∫ max|ω| dt ≈ <b>${d.bkm}</b> × ω₀t₀</span>`;
    }
  }
}

// The (lambda, mu) plane. A cascade with size ratio lambda and time ratio mu
// keeps its local Reynolds number from collapsing only if mu <= lambda^2, and
// keeps energy bounded only if mu >= lambda^(5/2) in 3D (lambda^2 in 2D).
export class ScalingWindow {
  constructor(canvas, onChange) {
    this.cv = canvas; this.onChange = onChange;
    this.lambda = 0.5; this.mu = 0.2;
    let drag = false;
    const pick = (e) => {
      const r = canvas.getBoundingClientRect();
      const { l, b, w, h } = this._box(r.width, r.height);
      const lam = Math.min(0.95, Math.max(0.05, (e.clientX - r.left - l) / w));
      const mu = Math.min(0.95, Math.max(0.01, 1 - (e.clientY - r.top - b) / h));
      this.set(lam, mu, true);
    };
    canvas.style.touchAction = 'none';
    canvas.addEventListener('pointerdown', (e) => { drag = true; canvas.setPointerCapture(e.pointerId); pick(e); });
    canvas.addEventListener('pointermove', (e) => { if (drag) pick(e); });
    canvas.addEventListener('pointerup', () => { drag = false; });
  }
  _box(W, H) { const l = 44, r = 12, t = 12, bt = 34; return { l, b: t, w: W - l - r, h: H - t - bt }; }
  set(lambda, mu, notify = false) {
    this.lambda = lambda; this.mu = mu; this.draw();
    if (notify) this.onChange?.(lambda, mu);
  }
  draw() {
    const dpr = fitCanvas(this.cv);
    const g = this.cv.getContext('2d');
    const W = this.cv.width / dpr, H = this.cv.height / dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const { l, b, w, h } = this._box(W, H);
    const X = (v) => l + v * w, Y = (v) => b + (1 - v) * h;
    const ink = css('--ink-2'), rule = css('--rule');
    g.strokeStyle = rule; g.lineWidth = 1; g.strokeRect(l, b, w, h);
    // 3D window: lambda^(5/2) <= mu <= lambda^2
    g.beginPath();
    for (let i = 0; i <= 100; i++) { const x = i / 100; g.lineTo(X(x), Y(x * x)); }
    for (let i = 100; i >= 0; i--) { const x = i / 100; g.lineTo(X(x), Y(x ** 2.5)); }
    g.closePath(); g.fillStyle = 'rgba(80, 160, 255, 0.28)'; g.fill();
    const curve = (p, color, dash) => {
      g.beginPath(); g.setLineDash(dash);
      for (let i = 0; i <= 100; i++) { const x = i / 100; g.lineTo(X(x), Y(x ** p)); }
      g.strokeStyle = color; g.lineWidth = 2; g.stroke(); g.setLineDash([]);
    };
    curve(2, '#ff8a3d', []);
    curve(2.5, '#5aa9ff', [5, 4]);
    g.font = '11px ui-sans-serif, system-ui, sans-serif'; g.fillStyle = ink;
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const v of [0, 0.25, 0.5, 0.75, 1]) { g.fillText(v, X(v), b + h + 4); }
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const v of [0, 0.25, 0.5, 0.75, 1]) { g.fillText(v, l - 4, Y(v)); }
    g.textAlign = 'center'; g.textBaseline = 'bottom';
    g.fillText('λ  (each layer’s size ÷ the previous one’s)', l + w / 2, H - 1);
    g.save(); g.translate(11, b + h / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle'; g.fillText('μ  (time ratio)', 0, 0); g.restore();
    g.textAlign = 'left'; g.textBaseline = 'middle';
    g.fillStyle = '#ff8a3d'; g.fillText('μ = λ²  viscosity line; also 2D energy line', X(0.06), Y(0.86));
    g.fillStyle = '#5aa9ff'; g.fillText('μ = λ^5/2  3D energy line', X(0.06), Y(0.78));
    const inside = this.mu <= this.lambda ** 2 && this.mu >= this.lambda ** 2.5;
    g.beginPath(); g.arc(X(this.lambda), Y(this.mu), 6, 0, 2 * Math.PI);
    g.fillStyle = inside ? '#fff' : '#ff4d6a'; g.fill();
    g.strokeStyle = '#000'; g.lineWidth = 1.5; g.stroke();
  }
}
