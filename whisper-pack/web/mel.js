// Log-mel spectrogram for OpenAI Whisper. Dependency-free ES module, runs in
// browsers, Web Workers and Node. Matches WhisperFeatureExtractor.
//
// FFT: mixed-radix decimation-in-time complex FFT, N = 400 = 5*5*2*2*2*2,
// float64 throughout, precomputed twiddles. Two real frames are packed into
// one complex transform (x + i*y) and separated by conjugate symmetry.

const SR = 16000, N_FFT = 400, HOP = 160, N_SAMPLES = 480000, N_FRAMES = 3000;
const N_FREQ = N_FFT / 2 + 1;

// ---------- mel filterbank (Slaney scale, Slaney norm) ----------
const hz2mel = (f) => {
  const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, step = Math.log(6.4) / 27;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / step : f / fsp;
};
const mel2hz = (m) => {
  const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, step = Math.log(6.4) / 27;
  return m >= minLogMel ? minLogHz * Math.exp(step * (m - minLogMel)) : m * fsp;
};

const filterCache = new Map();
/** Float64Array [nMels * 201], row-major [mel, freq] (HF's is [freq, mel]). */
export function melFilters(nMels) {
  if (filterCache.has(nMels)) return filterCache.get(nMels);
  const fb = new Float64Array(nMels * N_FREQ);
  const fftFreqs = new Float64Array(N_FREQ);
  for (let i = 0; i < N_FREQ; i++) fftFreqs[i] = (i * SR) / N_FFT;
  const lo = hz2mel(0), hi = hz2mel(SR / 2);
  const pts = new Float64Array(nMels + 2);
  for (let i = 0; i < pts.length; i++) pts[i] = mel2hz(lo + ((hi - lo) * i) / (nMels + 1));
  for (let m = 0; m < nMels; m++) {
    const f0 = pts[m], f1 = pts[m + 1], f2 = pts[m + 2];
    const enorm = 2 / (f2 - f0);
    for (let k = 0; k < N_FREQ; k++) {
      const f = fftFreqs[k];
      const v = Math.min((f - f0) / (f1 - f0), (f2 - f) / (f2 - f1));
      fb[m * N_FREQ + k] = Math.max(0, v) * enorm;
    }
  }
  filterCache.set(nMels, fb);
  return fb;
}

// ---------- mixed-radix FFT, N = 400 ----------
const FACTORS = [5, 5, 2, 2, 2, 2];
const COS = new Float64Array(N_FFT), SIN = new Float64Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  COS[i] = Math.cos((2 * Math.PI * i) / N_FFT);
  SIN[i] = -Math.sin((2 * Math.PI * i) / N_FFT); // e^{-2 pi i k/N}
}
// Scratch buffers for the recursion (ping-pong by level).
const bufRe = new Float64Array(N_FFT), bufIm = new Float64Array(N_FFT);
const tmpRe = new Float64Array(8), tmpIm = new Float64Array(8);

// Recursive DIT: out[0..n) = DFT of in[off + stride*j], j<n.
function fft(inRe, inIm, off, stride, outRe, outIm, oo, n, lvl) {
  if (n === 1) { outRe[oo] = inRe[off]; outIm[oo] = inIm[off]; return; }
  const r = FACTORS[lvl], m = n / r;
  for (let q = 0; q < r; q++) fft(inRe, inIm, off + q * stride, stride * r, outRe, outIm, oo + q * m, m, lvl + 1);
  const tstep = N_FFT / n; // twiddle w_n^k = w_N^{k*tstep}
  for (let k = 0; k < m; k++) {
    for (let q = 0; q < r; q++) {
      const idx = oo + q * m + k, tw = (q * k * tstep) % N_FFT;
      const c = COS[tw], s = SIN[tw], a = outRe[idx], b = outIm[idx];
      tmpRe[q] = a * c - b * s; tmpIm[q] = a * s + b * c;
    }
    for (let p = 0; p < r; p++) {
      let sr = 0, si = 0;
      for (let q = 0; q < r; q++) {
        const w = (p * q * m * tstep) % N_FFT; // w_r^{pq}
        const c = COS[w], s = SIN[w];
        sr += tmpRe[q] * c - tmpIm[q] * s;
        si += tmpRe[q] * s + tmpIm[q] * c;
      }
      // write back after all reads for this k are done: use second buffer trick
      sRe[p] = sr; sIm[p] = si;
    }
    for (let p = 0; p < r; p++) { outRe[oo + p * m + k] = sRe[p]; outIm[oo + p * m + k] = sIm[p]; }
  }
}
const sRe = new Float64Array(8), sIm = new Float64Array(8);

const WINDOW = new Float64Array(N_FFT); // periodic Hann
for (let i = 0; i < N_FFT; i++) WINDOW[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT);

function reflectIndex(i, len) {
  if (len === 1) return 0;
  const period = 2 * (len - 1);
  i = ((i % period) + period) % period;
  return i < len ? i : period - i;
}

function logMelOne(pcm, nMels) {
  // zero-pad / truncate to 30 s
  const x = new Float64Array(N_SAMPLES);
  const n = Math.min(pcm.length, N_SAMPLES);
  for (let i = 0; i < n; i++) x[i] = pcm[i];
  const pad = N_FFT / 2;
  const nFrames = Math.floor(N_SAMPLES / HOP) + 1; // 3001, last dropped
  const padded = new Float64Array(N_SAMPLES + 2 * pad);
  padded.set(x, pad);
  for (let i = 1; i <= pad; i++) {
    padded[pad - i] = x[reflectIndex(-i, N_SAMPLES)];
    padded[pad + N_SAMPLES - 1 + i] = x[reflectIndex(N_SAMPLES - 1 + i, N_SAMPLES)];
  }
  const fb = melFilters(nMels);
  const span = filterSpans(nMels, fb);
  const power = new Float64Array(N_FRAMES * N_FREQ);
  const inRe = new Float64Array(N_FFT), inIm = new Float64Array(N_FFT);
  // Frames that lie wholly in the zero padding have zero power: skip their FFTs (exact).
  let last = padded.length - 1;
  while (last >= 0 && padded[last] === 0) last--;
  const tEnd = Math.min(N_FRAMES, Math.floor(last / HOP) + 1);
  for (let t = 0; t < tEnd; t += 2) {
    const two = t + 1 < N_FRAMES;
    const a = t * HOP, b = (t + 1) * HOP;
    for (let i = 0; i < N_FFT; i++) {
      inRe[i] = padded[a + i] * WINDOW[i];
      inIm[i] = two ? padded[b + i] * WINDOW[i] : 0;
    }
    fft(inRe, inIm, 0, 1, bufRe, bufIm, 0, N_FFT, 0);
    for (let k = 0; k < N_FREQ; k++) {
      const k2 = (N_FFT - k) % N_FFT;
      const zr = bufRe[k], zi = bufIm[k], cr = bufRe[k2], ci = -bufIm[k2];
      // X = (Z + conj(Z[-k]))/2 ; Y = (Z - conj(Z[-k]))/(2i)
      const xr = (zr + cr) / 2, xi = (zi + ci) / 2;
      power[t * N_FREQ + k] = xr * xr + xi * xi;
      if (two) {
        const yr = (zi - ci) / 2, yi = -(zr - cr) / 2;
        power[(t + 1) * N_FREQ + k] = yr * yr + yi * yi;
      }
    }
  }
  const out = new Float64Array(nMels * N_FRAMES);
  let max = -Infinity;
  for (let m = 0; m < nMels; m++) {
    const row = m * N_FREQ;
    for (let t = 0; t < N_FRAMES; t++) {
      let s = 0; const p = t * N_FREQ;
      // each filter is non-zero only on [lo, hi); the zero terms add nothing (exact)
      for (let k = span[2 * m]; k < span[2 * m + 1]; k++) s += fb[row + k] * power[p + k];
      const v = Math.log10(Math.max(s, 1e-10));
      out[m * N_FRAMES + t] = v;
      if (v > max) max = v;
    }
  }
  const res = new Float32Array(nMels * N_FRAMES), floor = max - 8;
  for (let i = 0; i < res.length; i++) res[i] = (Math.max(out[i], floor) + 4) / 4;
  return res;
}

const spanCache = new Map();
/** Per filter, the first and one-past-last frequency bin with a non-zero weight. */
function filterSpans(nMels, fb) {
  if (spanCache.has(nMels)) return spanCache.get(nMels);
  const span = new Int32Array(2 * nMels);
  for (let m = 0; m < nMels; m++) {
    let lo = 0, hi = N_FREQ;
    while (lo < N_FREQ && fb[m * N_FREQ + lo] === 0) lo++;
    while (hi > lo && fb[m * N_FREQ + hi - 1] === 0) hi--;
    span[2 * m] = lo; span[2 * m + 1] = hi;
  }
  spanCache.set(nMels, span);
  return span;
}

/** pcm: Float32Array, 16 kHz mono. Returns Float32Array nMels*3000 [mel, frame]. */
export function logMel(pcm, nMels) {
  if (nMels !== 80 && nMels !== 128) throw new Error("nMels must be 80 or 128");
  return logMelOne(pcm, nMels);
}

/** Split into consecutive 30 s windows; last one zero-padded. */
export function logMelChunks(pcm, nMels) {
  const out = [];
  const n = Math.max(1, Math.ceil(pcm.length / N_SAMPLES));
  for (let i = 0; i < n; i++) out.push(logMel(pcm.subarray(i * N_SAMPLES, (i + 1) * N_SAMPLES), nMels));
  return out;
}
