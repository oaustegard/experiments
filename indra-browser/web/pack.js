// Packed Indra weights (run/export.py) back to float32, once, at load.
//
// A model's .bin holds each tensor as fp16 (byte planes when the record has planes: low bytes,
// then high bytes), or as b-bit codes (a little-endian bit stream) with fp16 scales per row of
// its [O, K] view (and a byte zero point per row for asymmetric schemes); a scheme with groups
// has one scale per G values of a row. Every tensor comes back in its ONNX shape, row-major.

const f16 = (h) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
};
const F16 = new Float32Array(65536);
for (let i = 0; i < 65536; i++) F16[i] = f16(i);

// n fp16 values at off; planes: all n low bytes, then all n high bytes (they gzip better grouped)
function halfs(buf, off, n, planes) {
  const out = new Float32Array(n);
  if (planes) {
    const b = new Uint8Array(buf, off, 2 * n);
    for (let i = 0; i < n; i++) out[i] = F16[b[i] | (b[n + i] << 8)];
    return out;
  }
  const u = new Uint16Array(buf, off, n);
  for (let i = 0; i < n; i++) out[i] = F16[u[i]];
  return out;
}

export function decodeTensor(buf, r) {
  const n = r.shape.reduce((a, b) => a * b, 1);
  if (r.enc === "f16") return halfs(buf, r.data, n, r.planes);
  if (r.enc === "f32") return new Float32Array(buf.slice(r.data, r.data + 4 * n));
  const { O, K, G, bits, sym } = r;
  // b-bit codes, little-endian bit stream
  const raw = new Uint8Array(buf, r.codes, Math.ceil((O * K * bits) / 8));
  const codes = new Uint16Array(O * K);
  for (let i = 0, bit = 0; i < O * K; i++, bit += bits) {
    let v = 0;
    for (let j = 0; j < bits; j++) { const q = bit + j; v |= ((raw[q >> 3] >> (q & 7)) & 1) << j; }
    codes[i] = v;
  }
  const ng = G ? Math.ceil(K / G) : 1;
  const sc = halfs(buf, r.scales, O * ng, r.planes);
  const zp = sym ? null : new Uint8Array(buf, r.zp, O * ng);
  const bias = sym ? 2 ** (bits - 1) - 1 : 0;
  const w2 = new Float32Array(O * K); // [O, K]
  for (let o = 0; o < O; o++) {
    for (let k = 0; k < K; k++) {
      const row = o * ng + (G ? Math.floor(k / G) : 0);
      const c = codes[o * K + k] - (sym ? bias : zp[row]);
      w2[o * K + k] = c * sc[row];
    }
  }
  if (r.view === "conv") return w2; // [O, ...] is the ONNX layout already
  // matmul: ONNX [K, N] = transpose of [O = N, K]
  const out = new Float32Array(O * K);
  for (let o = 0; o < O; o++) for (let k = 0; k < K; k++) out[k * O + o] = w2[o * K + k];
  return out;
}

// { entry, tensors: {name: Float32Array}, bytes } for one model of a pack directory URL.
export async function loadModel(base, manifest, name, fetchBytes = defaultFetch) {
  const entry = manifest.models[name];
  if (!entry) throw new Error(`pack has no model ${name}`);
  const buf = await fetchBytes(new URL(entry.file, base).href);
  const tensors = {};
  for (const [tn, r] of Object.entries(entry.tensors)) tensors[tn] = decodeTensor(buf, r);
  return { entry, tensors, bytes: buf.byteLength };
}

async function defaultFetch(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.arrayBuffer();
}
