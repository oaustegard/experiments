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

// NetworkIO::FromPixes + SetPixel. Returns Float32Array H*W (int models: integers -127..127).
export function normalise(gray, W, H, intMode) {
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
  const black = Math.fround(ile(mins, 0.25));
  const white = ile(maxes, 0.75);
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
export function padTable(H, W, sampleIteration, intMode) {
  const rng = new MinStd(sampleIteration);
  const draw = () => (intMode ? roundAway(rng.signed(127)) : Math.fround(rng.signed(1.0)));
  const rows = [];
  for (let y = 0; y < H; y++) {
    const edgeY = y === 0 || y === H - 1;
    for (let x = 0; x < W; x++) {
      if (!edgeY && x !== 0 && x !== W - 1) continue;
      const row = new Float32Array(9);
      for (let dx = -1; dx <= 1; dx++) {
        const base = (dx + 1) * 3;
        if (x + dx < 0 || x + dx >= W) {
          for (let k = 0; k < 3; k++) row[base + k] = draw();
        } else {
          for (let dy = -1; dy <= 1; dy++) if (y + dy < 0 || y + dy >= H) row[base + dy + 1] = draw();
        }
      }
      rows.push(row);
    }
  }
  const out = new Float32Array(rows.length * 9);
  rows.forEach((r, i) => out.set(r, i * 9));
  return out;
}
