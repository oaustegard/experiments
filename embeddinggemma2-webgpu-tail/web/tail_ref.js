// CPU reference for the EG2 fused tail (see SPEC.md).
// Pure functions over typed arrays; no dependencies; runs in browsers and Node.
// Internal arithmetic is Float64. Stored outputs are Float32 for floats,
// Uint8 for codes and packed bytes, Int32 for indices.

export const D = 768;
export const WORDS = 48;
export const BYTES_PER_VEC = 192;

const EPS_NORM = 1e-8;
const EPS_VLEN = 1e-12;

// dst[k] = src[off+k] / max(sqrt(sum src^2), 1e-8), for k < d.
function unitInto(src, off, d, dst) {
  let ss = 0;
  for (let k = 0; k < d; k++) {
    const v = src[off + k];
    ss += v * v;
  }
  const inv = 1 / Math.max(Math.sqrt(ss), EPS_NORM);
  for (let k = 0; k < d; k++) dst[k] = src[off + k] * inv;
}

// dst[k] = src[off+k], for k < d.
function copyInto(src, off, d, dst) {
  for (let k = 0; k < d; k++) dst[k] = src[off + k];
}

// out[j] = sum_k R[j*d+k] * x[k]. R is row-major d x d; x has length d.
function matvecInto(R, x, d, out) {
  for (let j = 0; j < d; j++) {
    const base = j * d;
    let s = 0;
    for (let k = 0; k < d; k++) s += R[base + k] * x[k];
    out[j] = s;
  }
}

// Masked mean over the sequence axis. hidden [B*S*d], mask [B*S] (u32).
// A row counts when its mask is nonzero. A batch row with no counted rows is zeros.
export function poolRef(hidden, mask, B, S, d = D) {
  const out = new Float32Array(B * d);
  const acc = new Float64Array(d);
  for (let b = 0; b < B; b++) {
    acc.fill(0);
    let cnt = 0;
    for (let s = 0; s < S; s++) {
      const row = b * S + s;
      if (mask[row] === 0) continue;
      cnt++;
      const base = row * d;
      for (let k = 0; k < d; k++) acc[k] += hidden[base + k];
    }
    if (cnt > 0) {
      const ob = b * d;
      for (let k = 0; k < d; k++) out[ob + k] = acc[k] / cnt;
    }
  }
  return out;
}

// Column mean of X [n*d] (d = 768). Returns Float32Array(d); zeros when n = 0.
export function meanRef(X, n, d = D) {
  const acc = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let k = 0; k < d; k++) acc[k] += X[base + k];
  }
  const out = new Float32Array(d);
  if (n > 0) for (let k = 0; k < d; k++) out[k] = acc[k] / n;
  return out;
}

// R . v for a row-major square R with side v.length. Returns Float32Array.
export function rotateRef(v, R) {
  const d = v.length;
  const x = Float64Array.from(v);
  const y = new Float64Array(d);
  matvecInto(R, x, d, y);
  return Float32Array.from(y);
}

// Encode n rows of X [n*768] into packed 2-bit codes plus per-doc aux floats.
// params: { R, Rmu, mu, boundaries: [3], centroids: [4], kappa }
// Returns { packed: Uint8Array(n*192), codes: Uint8Array(n*768),
//           scale, bias, len, mdot: Float32Array(n) }.
export function encodeRef(X, n, params, { normalize = true } = {}) {
  const d = D;
  const R = params.R;
  const rmu = Float64Array.from(params.Rmu);
  const b0 = params.boundaries[0];
  const b1 = params.boundaries[1];
  const b2 = params.boundaries[2];
  const c = Float64Array.from(params.centroids);
  const kappa = params.kappa;
  if (R.length !== d * d || rmu.length !== d) throw new Error("encodeRef: bad R or Rmu size");

  const packed = new Uint8Array(n * BYTES_PER_VEC);
  const codes = new Uint8Array(n * d);
  const scale = new Float32Array(n);
  const bias = new Float32Array(n);
  const len = new Float32Array(n);
  const mdot = new Float32Array(n);
  const x = new Float64Array(d);
  const t = new Float64Array(d);

  for (let i = 0; i < n; i++) {
    if (normalize) unitInto(X, i * d, d, x);
    else copyInto(X, i * d, d, x);

    // t = R x - Rmu
    matvecInto(R, x, d, t);
    let ss = 0;
    let md = 0;
    for (let j = 0; j < d; j++) {
      const tj = t[j] - rmu[j];
      t[j] = tj;
      ss += tj * tj;
      md += rmu[j] * tj;
    }
    const ln = Math.sqrt(ss);
    const denom = Math.max(ln, EPS_NORM);

    // codes: strict > against the three boundaries
    const cb = i * d;
    let vlen2 = 0;
    let vmu = 0;
    for (let j = 0; j < d; j++) {
      const u = t[j] / denom;
      const code = (u > b0 ? 1 : 0) + (u > b1 ? 1 : 0) + (u > b2 ? 1 : 0);
      codes[cb + j] = code;
      const cv = c[code];
      vlen2 += cv * cv;
      vmu += cv * rmu[j];
    }

    // pack: byte g holds codes 4g..4g+3, code 4g+m at shift 6-2m
    const pb = i * BYTES_PER_VEC;
    for (let g = 0; g < BYTES_PER_VEC; g++) {
      const q = cb + 4 * g;
      packed[pb + g] = (codes[q] << 6) | (codes[q + 1] << 4) | (codes[q + 2] << 2) | codes[q + 3];
    }

    const s = ln / (Math.max(Math.sqrt(vlen2), EPS_VLEN) * kappa);
    scale[i] = s;
    bias[i] = md - s * vmu;
    len[i] = ln;
    mdot[i] = md;
  }

  return { packed, codes, scale, bias, len, mdot };
}

// Scores [nq*n] for queries Q [nq*768] against an encodeRef result.
// score = scale_i * sum_j lut[j*4 + code_ij] + bias_i + offset,
// lut[j*4+c] = (R q)_j * centroids[c], offset = q . mu.
export function searchRef(Q, nq, enc, params, { normalize = true } = {}) {
  const d = D;
  const R = params.R;
  const mu = Float64Array.from(params.mu);
  const c = Float64Array.from(params.centroids);
  const n = enc.scale.length;
  const codes = enc.codes;
  const scale = enc.scale;
  const bias = enc.bias;
  const out = new Float32Array(nq * n);
  const q = new Float64Array(d);
  const qr = new Float64Array(d);
  const lut = new Float64Array(d * 4);

  for (let qi = 0; qi < nq; qi++) {
    if (normalize) unitInto(Q, qi * d, d, q);
    else copyInto(Q, qi * d, d, q);
    matvecInto(R, q, d, qr);
    let off = 0;
    for (let k = 0; k < d; k++) off += q[k] * mu[k];
    for (let j = 0; j < d; j++) {
      for (let m = 0; m < 4; m++) lut[j * 4 + m] = qr[j] * c[m];
    }

    const ob = qi * n;
    for (let i = 0; i < n; i++) {
      const cb = i * d;
      let acc = 0;
      for (let j = 0; j < d; j++) acc += lut[j * 4 + codes[cb + j]];
      out[ob + i] = scale[i] * acc + bias[i] + off;
    }
  }
  return out;
}

// Top-k per query: indices into [0, n), descending score, ties to the lower index.
// Returns Int32Array [nq*k].
export function topkRef(scores, nq, n, k) {
  const out = new Int32Array(nq * k);
  const idx = new Int32Array(n);
  for (let qi = 0; qi < nq; qi++) {
    const base = qi * n;
    for (let i = 0; i < n; i++) idx[i] = i;
    const cmp = (a, b) => {
      const sa = scores[base + a];
      const sb = scores[base + b];
      if (sa !== sb) return sa > sb ? -1 : 1;
      return a - b;
    };
    idx.sort(cmp);
    out.set(idx.subarray(0, k), qi * k);
  }
  return out;
}
