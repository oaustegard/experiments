// Tap onsets in the microphone signal. Pure JS, so the AudioWorklet (worklet.js) and the Node check
// (../test/taps.mjs) run the same code.
//
// A finger tapping paper on a table reaches the mic as a short broadband thump. The page's own drum reaches it too,
// from a speaker a few centimetres away, often louder than the tap. Three stages:
//
// 1. Echo cancelling. The detector is handed exactly what the page plays (`out`), so it can learn the path from
//    speaker to mic and subtract it: an NLMS adaptive filter, 120 ms long, on both signals decimated to ~16 kHz.
//    It adapts only while the page is playing and holds still while a tap is being detected, so a tap does not
//    teach it the wrong thing. The page plays a short burst of noise at start-up so it has learned the path before
//    the first drum. Browser echo cancellation is no substitute: it is tuned for speech and removes taps too.
// 2. Onsets. The residual is high-passed at about 100 Hz and squared, and its energy is followed two ways: a fast
//    envelope (2 ms) and a slow baseline (250 ms) that a spike cannot drag up, because the baseline only sees
//    min(fast, 4 x baseline). An onset is the fast envelope crossing `ratio` times the baseline and an absolute floor.
//    A decaying ring never crosses its own baseline, so whatever the canceller leaves of the drum's ring is harmless.
// 3. The drum's own attack. Every sound the page makes starts within ~50 ms of a tap it heard, so a refractory window
//    after each onset covers most attacks. For the rest (a screen tap, a second strike on a ringing drum), for 40 ms
//    after each attack in the played signal an onset must also beat what the speaker alone would leave in the residual:
//    a coupling factor learned from how loud each strike's attack came back after cancelling.
export class TapDetector {
  constructor(sr, { ratio = 8, floorDb = -62, refractoryMs = 150, hpHz = 100, echoMs = 30, aecMs = 120, mu = 0.4 } = {}) {
    this.sr = sr; this.F = Math.max(1, Math.round(sr / 16000)); const fs = (this.fs = sr / this.F);
    this.L = Math.round(aecMs * fs / 1000); this.w = new Float32Array(this.L); this.x = new Float32Array(2 * this.L); this.xi = 0;
    this.px = 0; this.mu = mu; this.aec = aecMs > 0; this.dsum = 0; this.osum = 0; this.k = 0;
    this.dl = new Float32Array(Math.round(0.3 * fs)); this.di = 0;
    this.set({ ratio, floorDb, refractoryMs, echoMs });
    this.hpA = Math.exp(-2 * Math.PI * hpHz / fs); this.hx = 0; this.hy = 0;
    this.af = 1 - Math.exp(-1 / (0.002 * fs)); this.as = 1 - Math.exp(-1 / (0.25 * fs));
    this.ao = 1 - Math.exp(-1 / (0.003 * fs)); this.ro = Math.exp(-1 / (0.06 * fs));
    this.fast = 0; this.base = 1e-9; this.n = 0; this.m = 0; this.last = -1e12; this.outEnv = 0;
    this.cpl = 1;   // residual energy per unit of speaker energy, learned per strike
    this.oBase = 1e-12; this.attackUntil = -1; this.holdUntil = -1;
    this.learn = null; this.peak = null; this.erle = { d: 1e-12, e: 1e-12 };
  }
  set({ ratio, floorDb, refractoryMs, echoMs, mu } = {}) {
    if (echoMs !== undefined) this.delay = Math.min(this.dl.length - 1, Math.max(0, Math.round(echoMs * this.fs / 1000)));
    if (ratio !== undefined) this.ratio = ratio;
    if (floorDb !== undefined) this.floor = 10 ** (floorDb / 10);
    if (refractoryMs !== undefined) this.refr = Math.round(refractoryMs * this.sr / 1000);
    if (mu !== undefined) this.mu = mu;
  }
  // mic: Float32Array of samples; out: what the page played over the same samples (or null). Returns onsets found
  // in this chunk, each { n (sample index at the input rate), level (fast/baseline at the peak, energy ratio), db }.
  // An onset is reported 3 ms after it starts, once its peak is known.
  process(mic, out) {
    const found = [], F = this.F;
    for (let i = 0; i < mic.length; i++, this.n++) {
      this.dsum += mic[i]; this.osum += out ? out[i] : 0;
      if (++this.k < F) continue;
      const d = this.dsum / F, o = this.osum / F; this.dsum = this.osum = 0; this.k = 0;
      const r = this.step(d, o);
      if (r) found.push(r);
    }
    return found;
  }
  step(d, o) {
    const L = this.L, w = this.w, x = this.x, m = this.m++;
    // 1. echo cancelling: x holds the last L played samples twice over, so x[xi .. xi+L) is contiguous, newest first
    const old = x[this.xi + L - 1] || 0;
    this.xi = (this.xi - 1 + L) % L; x[this.xi] = o; x[this.xi + L] = o;
    this.px += o * o - old * old; if (this.px < 0) this.px = 0;
    let e = d;
    if (this.aec && this.px > 1e-10) {   // nothing played in the last 120 ms: nothing to cancel
      let yh = 0; const xs = this.xi;
      for (let j = 0; j < L; j++) yh += w[j] * x[xs + j];
      e = d - yh;
      if (this.px > 1e-7 * L && m >= this.holdUntil) {
        const g = this.mu * e / (this.px + 1e-6);
        for (let j = 0; j < L; j++) w[j] += g * x[xs + j];
      }
      if (this.px > 1e-7 * L) { this.erle.d += 0.001 * (d * d - this.erle.d); this.erle.e += 0.001 * (e * e - this.erle.e); }
    }
    // 2. onsets on the residual
    const y = this.hpA * (this.hy + e - this.hx); this.hx = e; this.hy = y;
    this.fast += this.af * (y * y - this.fast);
    this.base += this.as * (Math.min(this.fast, 4 * this.base) - this.base);
    if (this.base < 1e-12) this.base = 1e-12;
    // 3. the played signal's envelope, delayed to when it reaches the mic, and its attacks
    const D = this.dl.length; this.dl[this.di] = o * o;
    const od = this.dl[(this.di - this.delay + D) % D]; this.di = (this.di + 1) % D;
    this.outEnv = od > this.outEnv ? this.outEnv + this.ao * (od - this.outEnv) : this.outEnv * this.ro;
    this.oBase += this.as * (Math.min(this.outEnv, 4 * this.oBase) - this.oBase); if (this.oBase < 1e-12) this.oBase = 1e-12;
    if (this.outEnv > 1e-8 && this.outEnv > 4 * this.oBase) this.attackUntil = m + Math.round(0.04 * this.fs);
    if (this.learn && this.n >= this.learn.start) {   // a strike's own attack, back through the room: how loud per unit played
      const Lr = this.learn;
      Lr.mic = Math.max(Lr.mic, this.fast); Lr.out = Math.max(Lr.out, this.outEnv);
      if (this.n >= Lr.end) {
        if (Lr.out > 1e-6) this.cpl = 0.7 * this.cpl + 0.3 * Math.min(50, Lr.mic / Lr.out);
        this.learn = null;
      }
    }
    if (this.peak) {
      const P = this.peak;
      if (this.fast > P.e) P.e = this.fast;
      if (m >= P.at) { this.peak = null; return { n: P.n, level: P.e / P.b, db: 10 * Math.log10(P.e) }; }
      return null;
    }
    if (this.n - this.last < this.refr) return null;
    const speaker = m < this.attackUntil ? 2 * this.cpl * this.outEnv : 0;
    if (this.fast > this.ratio * this.base && this.fast > this.floor && this.fast > speaker) {
      this.last = this.n;
      this.peak = { n: this.n, e: this.fast, b: this.base, at: m + Math.round(0.003 * this.fs) };
      this.holdUntil = m + Math.round(0.03 * this.fs);   // the canceller sits out the tap
    }
    return null;
  }
  // The page calls this when it starts a strike for a tap. From 15 ms on (past the tap's own thump, which is over in
  // about 10 ms) to 200 ms, the mic hears mostly that strike's echo.
  learnFrom(n) { this.learn = { start: n + Math.round(0.015 * this.sr), end: n + Math.round(0.2 * this.sr), mic: 0, out: 0 }; }
  // How much of the speaker the canceller removes, in dB, while the page is playing (echo return loss enhancement).
  erleDb() { return 10 * Math.log10(this.erle.d / this.erle.e); }
}

// Tap loudness -> strike velocity in 0..1: 9 dB over the baseline is a soft tap, 30 dB a hard one.
export const tapVelocity = (level) => Math.min(1, Math.max(0.25, (10 * Math.log10(level) - 9) / 21 * 0.75 + 0.25));

// A burst of noise for the canceller to learn the speaker-to-mic path from: `secs` long, faded in and out, at about
// -20 dBFS. The worklet plays it once at start-up.
export function trainingNoise(sr, secs = 0.6, seed = 1) {
  const n = Math.round(secs * sr), out = new Float32Array(2 * n); let a = seed >>> 0, lp = 0;
  for (let i = 0; i < n; i++) {
    a = (a * 1664525 + 1013904223) >>> 0; const wn = a / 2147483648 - 1;
    lp += 0.5 * (wn - lp);   // gently tilted toward the low end, where the drum is
    const env = Math.min(1, i / (0.05 * sr), (n - i) / (0.05 * sr));
    out[2 * i] = out[2 * i + 1] = 0.15 * env * (0.6 * wn + 0.8 * lp);
  }
  return out;
}
