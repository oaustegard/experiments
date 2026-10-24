// Line-by-line CPU mirror of the WGSL kernel in engine.js (256 lanes simulated per barrier phase).
// Used to verify the DSP math in a sandbox where GPU dispatches cannot execute.
const N = 1024, HOP = 256, NB = 5, HALF = 512, OLA = 0, ENV = 1024, FRZ = 1040, STATS = 16;
const TWO_PI = 6.283185307179586, SLOPE = 6, OLA_NORM = 1.5;
const hann = i => 0.5 - 0.5 * Math.cos(TWO_PI * i / N);
const LOG10 = x => Math.log2(x) * 0.30102999566;
const rev10 = i => { let r = 0; for (let b = 0; b < 10; b++) r |= ((i >> b) & 1) << (9 - b); return r; };
const pcg = v => { const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0; const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0; return ((w >>> 22) ^ w) >>> 0; };
const colors = [[1, .18, .15], [1, .62, .1], [.35, .95, .3], [.2, .55, 1], [.68, .32, 1]];

class CpuPrism {
  constructor({ sampleRate = 48000 } = {}) {
    this.sr = sampleRate; this.history = new Float32Array(N - HOP); this.st = new Float32Array(1560);
    this.st.fill(-100, ENV, ENV + NB); this.frameCounter = 0; this.setParams({});
  }
  setParams(p) {
    const d = { wet: 1, xover: [120, 500, 2000, 6000], attackMs: 10, releaseMs: 150,
      bands: Array.from({ length: NB }, () => ({ gainDb: 0, thrDb: 0, ratio: 1, gateDb: -120, freeze: 0, mute: 0, solo: 0 })) };
    this.params = Object.assign(d, this.params || {}, p);
  }
  weights(freq) {
    const lf = Math.log2(freq + 1), s = this.params.xover.map(c => 1 / (1 + Math.exp(-(lf - Math.log2(c)) * SLOPE)));
    return [1 - s[0], s[0] - s[1], s[1] - s[2], s[2] - s[3], s[3]];
  }
  fft(re, im, sgn) {
    for (let i = 0; i < N; i++) { const j = rev10(i); if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let s = 1; s <= 10; s++) {
      const m = 1 << s, hlf = m >> 1;
      for (let k = 0; k < 512; k++) {
        const pos = k % hlf, i = Math.floor(k / hlf) * m + pos, j = i + hlf;
        const ang = -sgn * TWO_PI * pos / m, wr = Math.cos(ang), wi = Math.sin(ang);
        const tr = wr * re[j] - wi * im[j], ti = wr * im[j] + wi * re[j];
        re[j] = re[i] - tr; im[j] = im[i] - ti; re[i] += tr; im[i] += ti;
      }
    }
  }
  process(samples) {
    const p = this.params, nFrames = samples.length / HOP, st = this.st, sr = this.sr;
    const win = new Float32Array(N - HOP + samples.length); win.set(this.history); win.set(samples, N - HOP);
    this.history = win.slice(win.length - (N - HOP));
    const hopSec = HOP / sr, att = Math.exp(-hopSec / (p.attackMs / 1000)), rel = Math.exp(-hopSec / (p.releaseMs / 1000));
    const anySolo = p.bands.some(b => b.solo);
    const out = new Float32Array(STATS + samples.length);
    const binHz = sr / N, refE = 3 * N * N / 32;
    const re = new Float32Array(N), im = new Float32Array(N), bandGain = new Float32Array(NB);
    for (let f = 0; f < nFrames; f++) {
      for (let i = 0; i < N; i++) { re[i] = win[f * HOP + i] * hann(i); im[i] = 0; }
      this.fft(re, im, 1);
      const E = new Float64Array(NB);
      for (let k = 0; k <= HALF; k++) { const m2 = re[k] ** 2 + im[k] ** 2, w = this.weights(k * binHz); for (let b = 0; b < NB; b++) E[b] += w[b] * m2; }
      for (let b = 0; b < NB; b++) {
        const B = p.bands[b], level = 10 * LOG10(E[b] / refE + 1e-12);
        let env = st[ENV + b]; const coef = level > env ? att : rel; env = coef * env + (1 - coef) * level; st[ENV + b] = env;
        const gr = Math.max(0, env - B.thrDb) * (1 - 1 / Math.max(B.ratio, 1));
        let gDb = B.gainDb - gr; if (env < B.gateDb) gDb -= 60;
        if (B.mute || (anySolo && !B.solo)) gDb = -120;
        bandGain[b] = 10 ** (gDb / 20); out[b] = level; out[5 + b] = gr; out[10 + b] = env;
      }
      for (let k = 0; k <= HALF; k++) {
        const w = this.weights(k * binHz); let g = 0, F = 0;
        for (let b = 0; b < NB; b++) { g += w[b] * bandGain[b]; F += w[b] * (p.bands[b].freeze ? 1 : 0); }
        let r = re[k], i2 = im[k]; const mag = Math.hypot(r, i2);
        if (F < 0.01) st[FRZ + k] = mag;
        if (F > 0) { const fm = st[FRZ + k] * 1.85, ph = (pcg(k * 9781 + (this.frameCounter + f) * 6271) & 0xffff) / 65536 * TWO_PI; r += (fm * Math.cos(ph) - r) * F; i2 += (fm * Math.sin(ph) - i2) * F; }
        if (k === 0 || k === HALF) i2 = 0;
        re[k] = r * g; im[k] = i2 * g;
      }
      for (let k = 1; k < HALF; k++) { re[N - k] = re[k]; im[N - k] = -im[k]; }
      this.fft(re, im, -1);
      for (let i = 0; i < N; i++) st[OLA + i] += re[i] / N * hann(i) / OLA_NORM;
      for (let i = 0; i < HOP; i++) out[STATS + f * HOP + i] = win[f * HOP + i] + (st[OLA + i] - win[f * HOP + i]) * p.wet;
      for (let i = 0; i < N; i++) st[OLA + i] = i + HOP < N ? st[OLA + i + HOP] : 0;
    }
    this.frameCounter += nFrames;
    return { out: out.subarray(STATS), stats: out.subarray(0, STATS) };
  }
}
module.exports = { CpuPrism };
