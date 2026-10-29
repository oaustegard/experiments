// Heart rate from the face region's colour, plus measurements of the exposure drift that fakes it.
// Pure JS (runs in the page and in Node). Samples arrive at camera times, are resampled to a uniform grid,
// and the rate is the spectral peak in 42-180 bpm of:
//   pos   the POS projection (Wang, den Brinker, Stuijk, de Haan 2017) of the face's mean RGB
//   evm   the face's mean of the EVM band-passed level (green), the signal the magnified video shows
//   green the face's raw mean green
// A global gain change (auto-exposure) scales all three channels alike; POS projects it out, green does not.
// The background rectangle has no pulse, so its luma measures drift directly.

export const BAND = [0.7, 3.0];   // Hz searched for the peak, 42-180 bpm

export function resample(ts, xs, t0, fs, n) {
  const out = new Float64Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = t0 + i / fs;
    while (j < ts.length - 2 && ts[j + 1] < t) j++;
    const a = ts[j], b = ts[j + 1] ?? a, u = b > a ? Math.min(1, Math.max(0, (t - a) / (b - a))) : 0;
    out[i] = xs[j] + ((xs[j + 1] ?? xs[j]) - xs[j]) * u;
  }
  return out;
}

const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
const std = (x) => { const m = mean(x); return Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / x.length); };

export function detrend(x) {
  const n = x.length, tm = (n - 1) / 2, m = mean(x);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (i - tm) * (x[i] - m); den += (i - tm) ** 2; }
  const s = den ? num / den : 0;
  return Float64Array.from(x, (v, i) => v - m - s * (i - tm));
}

// POS, algorithm 1 of Wang et al. 2017: temporal normalisation over 1.6 s windows, projection onto the plane
// orthogonal to the skin tone, alpha tuning, overlap-add.
export function pos(R, G, B, fs) {
  const n = R.length, l = Math.max(2, Math.round(1.6 * fs)), H = new Float64Array(n);
  for (let m = 0; m + l <= n; m++) {
    const r = R.subarray(m, m + l), g = G.subarray(m, m + l), b = B.subarray(m, m + l);
    const mr = mean(r), mg = mean(g), mb = mean(b);
    if (!(mr > 0 && mg > 0 && mb > 0)) continue;
    const s1 = new Float64Array(l), s2 = new Float64Array(l);
    for (let i = 0; i < l; i++) { const cr = r[i] / mr, cg = g[i] / mg, cb = b[i] / mb; s1[i] = cg - cb; s2[i] = -2 * cr + cg + cb; }
    const sd2 = std(s2), a = sd2 > 0 ? std(s1) / sd2 : 0, h = Float64Array.from(s1, (v, i) => v + a * s2[i]), mh = mean(h);
    for (let i = 0; i < l; i++) H[m + i] += h[i] - mh;
  }
  return H;
}

// Power at frequencies f0..f1 (Hz, step df) of a Hann-windowed signal, by direct DFT.
export function spectrum(x, fs, f0 = 0.5, f1 = 4.0, df = 0.005) {
  const n = x.length, w = Float64Array.from(x, (v, i) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1))));
  const f = [], p = [];
  for (let fr = f0; fr <= f1 + 1e-9; fr += df) {
    const om = 2 * Math.PI * fr / fs; let re = 0, im = 0;
    for (let i = 0; i < n; i++) { re += w[i] * Math.cos(om * i); im -= w[i] * Math.sin(om * i); }
    f.push(fr); p.push(re * re + im * im);
  }
  return { f, p };
}

// Strongest peak in BAND, refined by a parabola through its neighbours. SNR as in de Haan & Jeanne 2013:
// power within 0.1 Hz of the peak and 0.2 Hz of its first harmonic, against the rest of 0.5-4 Hz.
export function peak({ f, p }, band = BAND) {
  let k = -1;
  for (let i = 1; i < f.length - 1; i++) if (f[i] >= band[0] && f[i] <= band[1] && (k < 0 || p[i] > p[k])) k = i;
  if (k < 0) return null;
  const a = p[k - 1], b = p[k], c = p[k + 1], d = a - 2 * b + c, off = d < 0 ? Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / d)) : 0;
  const fr = f[k] + off * (f[1] - f[0]);
  let sig = 0, noise = 0;
  for (let i = 0; i < f.length; i++) (Math.abs(f[i] - fr) <= 0.1 || Math.abs(f[i] - 2 * fr) <= 0.2 ? (sig += p[i]) : (noise += p[i]));
  return { hz: fr, bpm: 60 * fr, snr: 10 * Math.log10(sig / Math.max(noise, 1e-30)), power: b };
}

// Same band-pass as the GPU's (two-pole smoother at fh minus one at fl), on a uniform grid.
export function bandpass(x, fs, fl = 0.7, fh = 3.0) {
  const kh = 1 - Math.exp(-2 * Math.PI * fh / fs), kl = 1 - Math.exp(-2 * Math.PI * fl / fs);
  let a1 = x[0], a2 = x[0], b1 = x[0], b2 = x[0];
  return Float64Array.from(x, (v) => { a1 += kh * (v - a1); a2 += kh * (a1 - a2); b1 += kl * (v - b1); b2 += kl * (b1 - b2); return a2 - b2; });
}

export function corr(a, b) {
  const ma = mean(a), mb = mean(b); let s = 0, sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) { s += (a[i] - ma) * (b[i] - mb); sa += (a[i] - ma) ** 2; sb += (b[i] - mb) ** 2; }
  return sa && sb ? s / Math.sqrt(sa * sb) : null;
}

const luma = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];

export class RateTracker {
  constructor({ win = 10, minWin = 6, fs = 30 } = {}) { Object.assign(this, { win, minWin, fs }); this.t = []; this.m = []; }
  push(t, m) {
    if (this.t.length && t <= this.t[this.t.length - 1]) return;
    this.t.push(t); this.m.push(m);
    while (this.t.length > 2 && this.t[0] < t - this.win - 2) { this.t.shift(); this.m.shift(); }
  }
  span() { return this.t.length ? this.t[this.t.length - 1] - this.t[0] : 0; }
  reset() { this.t = []; this.m = []; }
  estimate() {
    if (this.span() < this.minWin) return null;
    const { fs } = this, t1 = this.t[this.t.length - 1], t0 = Math.max(this.t[0], t1 - this.win), n = Math.floor((t1 - t0) * fs);
    const ch = (get) => resample(this.t, this.m.map(get), t0, fs, n);
    const R = ch((m) => m.face[0]), G = ch((m) => m.face[1]), B = ch((m) => m.face[2]);
    const EVM = ch((m) => m.filt[1]), BG = ch((m) => luma(m.bg)), FL = ch((m) => luma(m.face));
    const spec = (x) => spectrum(detrend(x), fs);
    const gN = Float64Array.from(G, (v) => v / mean(G)), bgN = Float64Array.from(BG, (v) => v / mean(BG));
    const P = peak(spec(pos(R, G, B, fs))), E = peak(spec(EVM)), Gr = peak(spec(gN)), Bg = peak(spec(bgN));
    const range = (x) => (Math.max(...x) - Math.min(...x)) / mean(x);
    const bpG = bandpass(gN, fs), bpB = bandpass(bgN, fs), skip = Math.round(2 * fs);   // skip the filter's settling
    return {
      t: t1, seconds: t1 - t0, fps: (this.t.length - 1) / this.span(),
      pos: P, evm: E, green: Gr, bg: Bg,
      // exposure: how far the background's brightness moved over the window, its in-band ripple, and how much
      // of the face's in-band green moves with it
      bgDrift: range(BG), faceDrift: range(FL), bgBandStd: std(bpB.subarray(skip)), faceBandStd: std(bpG.subarray(skip)),
      expCorr: corr(bpG.subarray(skip), bpB.subarray(skip)),
      // the background shows the same peak at comparable strength: the "pulse" is probably light or exposure
      exposureSuspect: !!(P && Bg && Math.abs(Bg.bpm - P.bpm) < 4 && Bg.snr > P.snr - 3),
    };
  }
}
