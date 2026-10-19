// Small Canvas2D plotting helpers and a radix-2 FFT for the energy spectrum.

export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function fitCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  return dpr;
}

// series: [{ pts: [[x, y], ...], color, width, dash, label }]
export function plot(cv, series, { xlim, ylim, logx = false, logy = false, xlabel = '', ylabel = '', vlines = [], hlines = [], title = '' } = {}) {
  const dpr = fitCanvas(cv);
  const g = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const pad = { l: 46 * dpr, r: 10 * dpr, t: (title ? 20 : 10) * dpr, b: 30 * dpr };
  g.clearRect(0, 0, W, H);
  const fx = logx ? Math.log10 : (v) => v, fy = logy ? Math.log10 : (v) => v;
  const all = series.flatMap((s) => s.pts).filter(([x, y]) => isFinite(fx(x)) && isFinite(fy(y)));
  let [x0, x1] = xlim ?? [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
  let [y0, y1] = ylim ?? [Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[1]))];
  if (!isFinite(x0) || !isFinite(x1)) { x0 = logx ? 1 : 0; x1 = logx ? 10 : 1; }
  if (!isFinite(y0) || !isFinite(y1)) { y0 = logy ? 1 : 0; y1 = logy ? 10 : 1; }
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) y1 = y0 + 1;
  const X = (v) => pad.l + ((fx(v) - fx(x0)) / (fx(x1) - fx(x0))) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - ((fy(v) - fy(y0)) / (fy(y1) - fy(y0))) * (H - pad.t - pad.b);
  const ink = css('--ink-2') || '#999', grid = css('--rule') || '#333';
  g.font = `${11 * dpr}px ui-sans-serif, system-ui, sans-serif`;
  g.lineWidth = 1 * dpr;
  // Ticks
  const ticks = (a, b, log) => {
    if (log) {
      const out = [];
      for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++) { const v = 10 ** e; if (v >= a * 0.999 && v <= b * 1.001) out.push(v); }
      return out;
    }
    const span = b - a, step = 10 ** Math.floor(Math.log10(span / 4));
    const m = span / step > 20 ? 5 : span / step > 8 ? 2 : 1;
    const s = step * m, out = [];
    for (let v = Math.ceil(a / s) * s; v <= b + 1e-9; v += s) out.push(+v.toPrecision(6));
    return out;
  };
  const fmt = (v, log) => (log ? `1e${Math.round(Math.log10(v))}` : Math.abs(v) >= 1000 ? v.toExponential(0) : `${+v.toPrecision(3)}`);
  g.strokeStyle = grid; g.fillStyle = ink;
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (const v of ticks(x0, x1, logx)) { g.beginPath(); g.moveTo(X(v), pad.t); g.lineTo(X(v), H - pad.b); g.stroke(); g.fillText(fmt(v, logx), X(v), H - pad.b + 4 * dpr); }
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const v of ticks(y0, y1, logy)) { g.beginPath(); g.moveTo(pad.l, Y(v)); g.lineTo(W - pad.r, Y(v)); g.stroke(); g.fillText(fmt(v, logy), pad.l - 4 * dpr, Y(v)); }
  g.textAlign = 'center'; g.textBaseline = 'bottom';
  g.fillText(xlabel, (pad.l + W - pad.r) / 2, H - 1 * dpr);
  if (title) { g.textBaseline = 'top'; g.textAlign = 'left'; g.fillText(title, pad.l, 3 * dpr); }
  g.save(); g.translate(11 * dpr, (pad.t + H - pad.b) / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle'; g.fillText(ylabel, 0, 0); g.restore();
  g.save();
  g.beginPath(); g.rect(pad.l, pad.t, W - pad.l - pad.r, H - pad.t - pad.b); g.clip();
  for (const v of vlines) { g.strokeStyle = v.color; g.setLineDash((v.dash ?? [4, 4]).map((d) => d * dpr)); g.beginPath(); g.moveTo(X(v.x), pad.t); g.lineTo(X(v.x), H - pad.b); g.stroke(); if (v.label) { g.fillStyle = v.color; g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText(v.label, X(v.x) - 4 * dpr, H - pad.b - 4 * dpr); } }
  for (const v of hlines) { g.strokeStyle = v.color; g.setLineDash((v.dash ?? [4, 4]).map((d) => d * dpr)); g.beginPath(); g.moveTo(pad.l, Y(v.y)); g.lineTo(W - pad.r, Y(v.y)); g.stroke(); }
  for (const s of series) {
    g.strokeStyle = s.color; g.lineWidth = (s.width ?? 1.8) * dpr; g.setLineDash((s.dash ?? []).map((d) => d * dpr));
    g.beginPath(); let on = false;
    for (const [x, y] of s.pts) {
      if (!isFinite(fx(x)) || !isFinite(fy(y))) { on = false; continue; }
      const px = X(x), py = Y(Math.max(Math.min(y, y1 * 10), logy ? y0 / 10 : -1e9));
      if (on) g.lineTo(px, py); else { g.moveTo(px, py); on = true; }
    }
    g.stroke();
  }
  g.restore();
  // Legend
  let ly = pad.t + 4 * dpr;
  g.setLineDash([]); g.textAlign = 'left'; g.textBaseline = 'middle';
  const lw = Math.max(0, ...series.filter((s) => s.label).map((s) => g.measureText(s.label).width));
  for (const s of series) {
    if (!s.label) continue;
    const lx = W - pad.r - lw - 34 * dpr;
    g.strokeStyle = s.color; g.lineWidth = 2 * dpr; g.setLineDash((s.dash ?? []).map((d) => d * dpr));
    g.beginPath(); g.moveTo(lx, ly + 6 * dpr); g.lineTo(lx + 18 * dpr, ly + 6 * dpr); g.stroke();
    g.fillStyle = ink; g.fillText(s.label, lx + 24 * dpr, ly + 6 * dpr);
    ly += 16 * dpr;
  }
  return { X, Y };
}

function fft1(re, im, n, stride, off, rev, cos, sin) {
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      const a = off + i * stride, b = off + j * stride;
      let t = re[a]; re[a] = re[b]; re[b] = t;
      t = im[a]; im[a] = im[b]; im[b] = t;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let k = 0; k < half; k++) {
        const a = off + (i + k) * stride, b = off + (i + k + half) * stride;
        const wr = cos[k * step], wi = -sin[k * step];
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

function fft2(re, im, n) {
  const bits = Math.log2(n);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  const cos = new Float64Array(n), sin = new Float64Array(n);
  for (let i = 0; i < n; i++) { cos[i] = Math.cos((2 * Math.PI * i) / n); sin[i] = Math.sin((2 * Math.PI * i) / n); }
  for (let r = 0; r < n; r++) fft1(re, im, n, 1, r * n, rev, cos, sin);
  for (let c = 0; c < n; c++) fft1(re, im, n, n, c, rev, cos, sin);
}

// Shell-summed kinetic energy spectrum of a periodic n x n velocity field (rgba32 layout).
export function energySpectrum(vel, n) {
  const E = new Float64Array(n);
  for (const comp of [0, 1]) {
    const re = new Float64Array(n * n), im = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) re[i] = vel[i * 4 + comp];
    fft2(re, im, n);
    for (let y = 0; y < n; y++) {
      const ky = y <= n / 2 ? y : y - n;
      for (let x = 0; x < n; x++) {
        const kx = x <= n / 2 ? x : x - n;
        const k = Math.round(Math.hypot(kx, ky));
        if (k < n) { const i = y * n + x; E[k] += (0.5 * (re[i] * re[i] + im[i] * im[i])) / (n ** 4); }
      }
    }
  }
  return Array.from(E.slice(1, n / 2 + 1), (e, i) => [i + 1, e]);
}
