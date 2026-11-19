// CPU side of a line's preparation, following tessgpu/ref.py: scaling to the network height,
// Tesseract's input normalisation, and the Convolve padding table from Tesseract 5.3.4's
// seeded std::minstd_rand.

// Resize an 8-bit grey line (Uint8Array, row-major) to height H. Area average when shrinking,
// bilinear when growing (like ref.scale_to_height's PIL BOX / BILINEAR; not bit-identical).
export function scaleToHeight(src, w, h, H) {
  if (h === H) return { data: src, width: w };
  const f = H / h;
  const W = Math.max(1, Math.round(w * f));
  const out = new Uint8Array(W * H);
  if (f < 1) {
    const sx = w / W, sy = h / H;
    for (let y = 0; y < H; y++) {
      const y0 = y * sy, y1 = y0 + sy;
      for (let x = 0; x < W; x++) {
        const x0 = x * sx, x1 = x0 + sx;
        let acc = 0, area = 0;
        for (let yy = Math.floor(y0); yy < Math.min(h, Math.ceil(y1)); yy++) {
          const wy = Math.min(y1, yy + 1) - Math.max(y0, yy);
          for (let xx = Math.floor(x0); xx < Math.min(w, Math.ceil(x1)); xx++) {
            const wx = Math.min(x1, xx + 1) - Math.max(x0, xx);
            acc += src[yy * w + xx] * wx * wy;
            area += wx * wy;
          }
        }
        out[y * W + x] = Math.round(acc / area);
      }
    }
  } else {
    for (let y = 0; y < H; y++) {
      const fy = Math.min(h - 1, Math.max(0, (y + 0.5) / f - 0.5));
      const y0 = Math.floor(fy), y1 = Math.min(h - 1, y0 + 1), ay = fy - y0;
      for (let x = 0; x < W; x++) {
        const fx = Math.min(w - 1, Math.max(0, (x + 0.5) / f - 0.5));
        const x0 = Math.floor(fx), x1 = Math.min(w - 1, x0 + 1), ax = fx - x0;
        const top = src[y0 * w + x0] * (1 - ax) + src[y0 * w + x1] * ax;
        const bot = src[y1 * w + x0] * (1 - ax) + src[y1 * w + x1] * ax;
        out[y * W + x] = Math.round(top * (1 - ay) + bot * ay);
      }
    }
  }
  return { data: out, width: W };
}

// STATS(0, 255)::ile
function ile(counts, frac) {
  let total = 0;
  for (let i = 0; i < 256; i++) total += counts[i];
  if (!total) return 0;
  const target = Math.min(Math.max(frac * total, 1), total);
  let s = 0, i = 0;
  while (i <= 255 && s < target) s += counts[i++];
  return i - (s - target) / counts[i - 1];
}

// Light text on a dark ground? Tesseract answers this after the fact (a second forward pass on
// the inverted line when the mean best-path probability is under 0.7); the image says it up
// front: most pixels darker than halfway between Tesseract's own black and white estimates.
// 86/86 inverted and 0/600 normal lines on the synthetic sample, 1/800 real scanned lines.
export function isInverted(gray, W, H) {
  const [black, white] = blackWhite(gray, W, H);
  const mid = (black + white) / 2;
  let dark = 0;
  for (let i = 0; i < W * H; i++) if (gray[i] < mid) dark++;
  return dark > 0.5 * W * H;
}

function blackWhite(gray, W, H) {
  const mins = new Float64Array(256), maxes = new Float64Array(256);
  if (W >= 3) {
    const r = Math.floor(H / 2) * W;
    for (let x = 1; x + 1 < W; x++) {
      const p = gray[r + x - 1], c = gray[r + x], n = gray[r + x + 1];
      if ((c < p && c <= n) || (c <= p && c < n)) mins[c]++;
      if ((c > p && c >= n) || (c >= p && c > n)) maxes[c]++;
    }
  }
  if (!mins.some((v) => v)) mins[0] = 1;
  if (!maxes.some((v) => v)) maxes[255] = 1;
  return [ile(mins, 0.25), ile(maxes, 0.75)];
}

// The normalised value of every possible pixel for one line (NetworkIO::SetPixel with that line's
// black and contrast): normalise() is this table applied to the pixels.
export function normTable(gray, W, H, intMode) {
  const [b0, white] = blackWhite(gray, W, H);
  const black = Math.fround(b0);
  let contrast = Math.fround((white - black) / 2);
  if (contrast <= 0) contrast = 1;
  const t = new Float32Array(256);
  for (let p = 0; p < 256; p++) {
    const v = Math.fround(Math.fround(Math.fround(p - black) / contrast) - 1);
    t[p] = intMode ? Math.max(-127, Math.min(127, roundAway(Math.fround(128 * v)))) : v;
  }
  return t;
}

// NetworkIO::FromPixes + SetPixel. Returns Float32Array H*W (int models: integers -127..127).
export function normalise(gray, W, H, intMode) {
  const [b0, white] = blackWhite(gray, W, H);
  const black = Math.fround(b0);
  let contrast = Math.fround((white - black) / 2);
  if (contrast <= 0) contrast = 1;
  const out = new Float32Array(W * H);
  for (let i = 0; i < out.length; i++) {
    const v = Math.fround(Math.fround(Math.fround(gray[i] - black) / contrast) - 1);
    out[i] = intMode ? Math.max(-127, Math.min(127, roundAway(128 * v))) : v;
  }
  return out;
}

export function roundAway(x) {
  return x >= 0 ? Math.floor(x + 0.5) : -Math.floor(-x + 0.5);
}

// Tesseract 5.3.4 TRand (std::minstd_rand) seeded as LSTMRecognizer::SetRandomSeed does.
export class MinStd {
  constructor(sampleIteration) {
    const m = 2147483647n;
    let s = (BigInt(sampleIteration) * 0x10000001n) & 0xffffffffffffffffn;
    s %= m;
    this.state = Number(s === 0n ? 1n : s);
    this.next();
  }
  next() {
    this.state = (this.state * 48271) % 2147483647; // < 2^47, exact in a double
    return this.state;
  }
  signed(range) {
    return range * 2.0 * this.next() / 2147483647 - range;
  }
}

// Convolve padding as [nb, 9] in border raster order, slot (dx+1)*3 + (dy+1); see
// ref.pad_table. Every line starts from the same seed.
// One random stream per model: every line starts from the same seed, so a line's padding is a
// prefix of the same sequence of draws (already rounded as Tesseract stores them).
const streams = new Map();
function stream(sampleIteration, intMode, n) {
  const key = `${sampleIteration}:${intMode}`;
  let s = streams.get(key);
  if (!s || s.vals.length < n) {
    const rng = new MinStd(sampleIteration);
    const vals = new Float32Array(Math.max(n, 2 * (s ? s.vals.length : 4096)));
    for (let i = 0; i < vals.length; i++) vals[i] = intMode ? roundAway(rng.signed(127)) : Math.fround(rng.signed(1.0));
    s = { vals };
    streams.set(key, s);
  }
  return s.vals;
}

export function padTable(H, W, sampleIteration, intMode) {
  const nb = H <= 2 ? H * W : 2 * W + (H - 2) * Math.min(2, W);
  const vals = stream(sampleIteration, intMode, 12 * (W + H) + 64);
  const out = new Float32Array(nb * 9);
  let r = 0, d = 0;
  const fill = (y, x) => {
    const o = r * 9;
    for (let dx = -1; dx <= 1; dx++) {
      const base = o + (dx + 1) * 3;
      if (x + dx < 0 || x + dx >= W) {
        out[base] = vals[d]; out[base + 1] = vals[d + 1]; out[base + 2] = vals[d + 2]; d += 3;
      } else {
        if (y - 1 < 0) out[base] = vals[d++];
        if (y + 1 >= H) out[base + 2] = vals[d++];
      }
    }
    r++;
  };
  for (let y = 0; y < H; y++) {
    if (y === 0 || y === H - 1) for (let x = 0; x < W; x++) fill(y, x);
    else { fill(y, 0); if (W > 1) fill(y, W - 1); }
  }
  return out;
}
