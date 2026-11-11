// Plain-JavaScript half of the GLiNER bi-encoder runtime: everything after the text encoder.
//
//   word splitting      splitWords(text)                 Python `\w+(?:[-_]\w+)*|\S` with Python's \w and \s
//   label encoder       new LabelEncoder(pack, m)         MiniLM (BERT, post-LN) over label ids, mean-pooled
//   span head           new SpanHead(pack, ghead, K)      markerV0 in GRef's factorised form
//   decoding            decodeSpans(...)                  sigmoid, threshold, greedy flat / nested search
//
// run/gref.py (GRefBi) is the spec; every function names the GRef lines it follows. No DOM and no
// WebGPU here, so a bridge job worker can use it as is. Matmuls run on typed arrays with Float64
// accumulators, four rows per pass over the weights, and skip input columns that are zero in all
// four rows (most of them after a ReLU). int8 records keep their codes and apply the per-column
// scale after the sum, as GRef's Ref.lin does.

// ---- weight file access --------------------------------------------------------------------------

export function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 5.960464477539063e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// Copies out of the packed weights.bin (bytes: Uint8Array) by manifest record { offset, length }.
export class Pack {
  constructor(bytes) { this.bytes = bytes; }
  raw(r) { return this.bytes.slice(r.offset, r.offset + r.length).buffer; }
  f32(r) { return new Float32Array(this.raw(r)); }
  i8(r) { return new Int8Array(this.raw(r)); }
  u8(r) { return new Uint8Array(this.raw(r)); }
  f16(r) { const u = new Uint16Array(this.raw(r)), o = new Float32Array(u.length); for (let i = 0; i < u.length; i++) o[i] = f16(u[i]); return o; }
  // A matmul record as { K, N, w: [N, K] row-major (torch layout), s: per-column scale or null }. fp32
  // records are stored [N, K] in the file; int8 records are [K, N] there and are transposed once here.
  lin(r) {
    if (r.f32_t) return { K: r.K, N: r.N, w: this.f32(r.f32_t), s: null };
    const K = r.K, N = r.N, src = this.i8(r.q), w = new Int8Array(K * N);
    for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) w[n * K + k] = src[k * N + n];
    return { K, N, w, s: this.f32(r.s) };
  }
  // Input columns [k0, k0 + K) of a lin record, copied (the concat halves of out_project.0).
  static kslice(L, k0, K) {
    const w = new L.w.constructor(L.N * K);
    for (let n = 0; n < L.N; n++) w.set(L.w.subarray(n * L.K + k0, n * L.K + k0 + K), n * K);
    return { K, N: L.N, w, s: L.s };
  }
  // Token table rows for ids, dequantized like gref.Ref.table / bd.Ref.table: bits 32, 8 or 4 (groups of 32, fp16 scale).
  static tableRows(T, ids, out) {
    const H = T.H, G = H / 32;
    ids.forEach((id, r) => {
      const o = r * H;
      if (T.bits === 32) { out.set(T.f32.subarray(id * H, id * H + H), o); return; }
      for (let j = 0; j < H; j++) {
        const s = T.scales[id * G + (j >> 5)];
        const q = T.bits === 8 ? T.codes[id * H + j] : ((T.codes[id * (H >> 1) + (j >> 1)] >> ((j & 1) * 4)) & 15) - 8;
        out[o + j] = Math.fround(q * s);
      }
    });
  }
  table(T) {
    if (T.bits === 32) return { bits: 32, V: T.V, H: T.H, f32: this.f32(T.f32) };
    if (T.bits !== 4 && T.bits !== 8) throw new Error(`token table bits ${T.bits}`);
    return { bits: T.bits, V: T.V, H: T.H, codes: T.bits === 8 ? this.i8(T.codes) : this.u8(T.codes), scales: this.f16(T.scales) };
  }
}

// ---- matmul ---------------------------------------------------------------------------------------
// Y[r, n] = act((sum_k X[r, k] w[n, k]) * s[n] + bias[n]) for r < rows, weights [N, K]. Four rows by two
// columns accumulate in registers (1.6 GMAC/s in Node 22 against 1.1 for a row-axpy loop, K 384 N 1536);
// a short last block recomputes its last row and writes only the real ones. Two copies, one per weight
// array type, so each stays monomorphic in the JIT.

function linI8(X, rows, K, N, w, s, bias, Y, relu) {
  for (let r0 = 0; r0 < rows; r0 += 4) {
    const nr = Math.min(4, rows - r0), b0 = r0 * K;
    const b1 = nr > 1 ? b0 + K : b0, b2 = nr > 2 ? b0 + 2 * K : b1, b3 = nr > 3 ? b0 + 3 * K : b2;
    for (let n = 0; n < N; n += 2) {
      const wa = n * K, wc = wa + K;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, t0 = 0, t1 = 0, t2 = 0, t3 = 0;
      for (let k = 0; k < K; k++) {
        const v = w[wa + k], u = w[wc + k], x0 = X[b0 + k], x1 = X[b1 + k], x2 = X[b2 + k], x3 = X[b3 + k];
        s0 += x0 * v; s1 += x1 * v; s2 += x2 * v; s3 += x3 * v;
        t0 += x0 * u; t1 += x1 * u; t2 += x2 * u; t3 += x3 * u;
      }
      const sa = s[n], sb = s[n + 1], ba = bias ? bias[n] : 0, bb = bias ? bias[n + 1] : 0;
      let o = r0 * N + n, y = s0 * sa + ba, z = t0 * sb + bb;
      Y[o] = relu && y < 0 ? 0 : y; Y[o + 1] = relu && z < 0 ? 0 : z;
      if (nr > 1) { o += N; y = s1 * sa + ba; z = t1 * sb + bb; Y[o] = relu && y < 0 ? 0 : y; Y[o + 1] = relu && z < 0 ? 0 : z; }
      if (nr > 2) { o += N; y = s2 * sa + ba; z = t2 * sb + bb; Y[o] = relu && y < 0 ? 0 : y; Y[o + 1] = relu && z < 0 ? 0 : z; }
      if (nr > 3) { o += N; y = s3 * sa + ba; z = t3 * sb + bb; Y[o] = relu && y < 0 ? 0 : y; Y[o + 1] = relu && z < 0 ? 0 : z; }
    }
  }
}

function linF32(X, rows, K, N, w, bias, Y, relu) {
  for (let r0 = 0; r0 < rows; r0 += 4) {
    const nr = Math.min(4, rows - r0), b0 = r0 * K;
    const b1 = nr > 1 ? b0 + K : b0, b2 = nr > 2 ? b0 + 2 * K : b1, b3 = nr > 3 ? b0 + 3 * K : b2;
    for (let n = 0; n < N; n += 2) {
      const wa = n * K, wc = wa + K;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, t0 = 0, t1 = 0, t2 = 0, t3 = 0;
      for (let k = 0; k < K; k++) {
        const v = w[wa + k], u = w[wc + k], x0 = X[b0 + k], x1 = X[b1 + k], x2 = X[b2 + k], x3 = X[b3 + k];
        s0 += x0 * v; s1 += x1 * v; s2 += x2 * v; s3 += x3 * v;
        t0 += x0 * u; t1 += x1 * u; t2 += x2 * u; t3 += x3 * u;
      }
      const ba = bias ? bias[n] : 0, bb = bias ? bias[n + 1] : 0;
      let o = r0 * N + n, y = s0 + ba, z = t0 + bb;
      Y[o] = relu && y < 0 ? 0 : y; Y[o + 1] = relu && z < 0 ? 0 : z;
      if (nr > 1) { o += N; y = s1 + ba; z = t1 + bb; Y[o] = relu && y < 0 ? 0 : y; Y[o + 1] = relu && z < 0 ? 0 : z; }
      if (nr > 2) { o += N; y = s2 + ba; z = t2 + bb; Y[o] = relu && y < 0 ? 0 : y; Y[o + 1] = relu && z < 0 ? 0 : z; }
      if (nr > 3) { o += N; y = s3 + ba; z = t3 + bb; Y[o] = relu && y < 0 ? 0 : y; Y[o + 1] = relu && z < 0 ? 0 : z; }
    }
  }
}

export function linear(X, rows, L, bias, Y, relu = false) {
  if (L.N % 2) throw new Error("linear: N must be even");
  if (L.s) linI8(X, rows, L.K, L.N, L.w, L.s, bias, Y, relu);
  else linF32(X, rows, L.K, L.N, L.w, bias, Y, relu);
  return Y;
}

// Scratch Float32Arrays that grow and are reused (no per-call allocation in steady state).
class Scratch {
  constructor() { this.m = new Map(); }
  get(name, n) {
    let a = this.m.get(name);
    if (!a || a.length < n) { a = new Float32Array(Math.max(n, a ? a.length * 1.5 : 0) | 0); this.m.set(name, a); }
    return a;
  }
}

// ---- exact-erf GELU (float64), as scipy.special.erf in browser-decider's pack.gelu ----------------
export function erf(x) {
  const a = Math.abs(x);
  if (a >= 6) return Math.sign(x);
  if (a < 3) {
    const x2 = x * x;
    let term = x, sum = x;
    for (let n = 1; n < 100; n++) {
      term *= -x2 / n;
      const d = term / (2 * n + 1);
      sum += d;
      if (Math.abs(d) < 1e-17 * Math.abs(sum)) break;
    }
    return 1.1283791670955126 * sum;
  }
  let f = a;   // erfc continued fraction, evaluated from depth 80 back
  for (let k = 80; k >= 1; k--) f = a + (k / 2) / f;
  const erfc = Math.exp(-a * a) / (1.7724538509055159 * f);
  return Math.sign(x) * (1 - erfc);
}
const gelu = (x) => 0.5 * x * (1 + erf(x * 0.7071067811865476));

function layerNorm(X, rows, H, w, b, eps) {
  for (let r = 0; r < rows; r++) {
    const o = r * H;
    let mu = 0;
    for (let j = 0; j < H; j++) mu += X[o + j];
    mu /= H;
    let v = 0;
    for (let j = 0; j < H; j++) { const d = X[o + j] - mu; v += d * d; }
    const inv = 1 / Math.sqrt(v / H + eps);
    for (let j = 0; j < H; j++) X[o + j] = (X[o + j] - mu) * inv * w[j] + (b ? b[j] : 0);
  }
}

// ---- label encoder: gref.BertRef ---------------------------------------------------------------------
// embeddings (word[ids] + token_type[0]) + position[0..S-1] -> LayerNorm; per layer fused qkv
// attention, x = LN(x + Wo a + bo), x = LN(x + Wo2 gelu(Wi x + bi) + bo2); mean over all tokens.
// All new labels go through each layer together (one pass over each weight matrix), attention per label.
export class LabelEncoder {
  constructor(pack, m) {
    this.H = m.hidden; this.nh = m.heads; this.D = m.hidden / m.heads; this.I = m.inter; this.eps = m.eps; this.maxPos = m.max_pos;
    this.table = pack.table(m.table);
    this.pos = pack.f32(m.pos); this.type0 = pack.f32(m.type0);
    this.embW = pack.f32(m.emb_ln_w); this.embB = pack.f32(m.emb_ln_b);
    this.layers = m.layers.map((L) => ({
      Wqkv: pack.lin(L.Wqkv), bqkv: pack.f32(L.bqkv), Wo: pack.lin(L.Wo), bo: pack.f32(L.bo),
      ln1w: pack.f32(L.ln1_w), ln1b: pack.f32(L.ln1_b), Wi: pack.lin(L.Wi), bi: pack.f32(L.bi),
      Wo2: pack.lin(L.Wo2), bo2: pack.f32(L.bo2), ln2w: pack.f32(L.ln2_w), ln2b: pack.f32(L.ln2_b) }));
    this.sc = new Scratch();
  }

  // idsList: one id array per label ([CLS] ... [SEP]). Returns { pooled: Float32Array(L * H), hidden }
  // with hidden the full last_hidden_state of the first label when keepFirst.
  encode(idsList, keepFirst = false) {
    const { H, nh, D, I, eps } = this, L = idsList.length;
    const seg = [];
    let rows = 0;
    for (const ids of idsList) {
      if (ids.length > this.maxPos) throw new Error(`label of ${ids.length} tokens exceeds ${this.maxPos} positions`);
      seg.push([rows, ids.length]); rows += ids.length;
    }
    const X = this.sc.get("x", rows * H), QKV = this.sc.get("qkv", rows * 3 * H), O = this.sc.get("o", rows * H);
    const T = this.sc.get("t", rows * H), U = this.sc.get("u", rows * I), P = this.sc.get("p", 512);
    idsList.forEach((ids, i) => {
      const o = seg[i][0] * H;
      Pack.tableRows(this.table, ids, X.subarray(o, o + ids.length * H));
      for (let t = 0; t < ids.length; t++) for (let j = 0; j < H; j++) {
        const q = o + t * H + j;
        X[q] = Math.fround(X[q] + this.type0[j]) + this.pos[t * H + j];
      }
    });
    layerNorm(X, rows, H, this.embW, this.embB, eps);
    const scale = 1 / Math.fround(Math.sqrt(D)), H3 = 3 * H;
    for (const Ly of this.layers) {
      linear(X, rows, Ly.Wqkv, Ly.bqkv, QKV);
      for (const [s0, S] of seg) {
        for (let h = 0; h < nh; h++) {
          for (let i = 0; i < S; i++) {
            const qo = (s0 + i) * H3 + h * D;
            let mx = -Infinity;
            for (let j = 0; j < S; j++) {
              const ko = (s0 + j) * H3 + H + h * D;
              let d = 0;
              for (let c = 0; c < D; c++) d += QKV[qo + c] * QKV[ko + c];
              P[j] = d * scale;
              if (P[j] > mx) mx = P[j];
            }
            let sum = 0;
            for (let j = 0; j < S; j++) { P[j] = Math.exp(P[j] - mx); sum += P[j]; }
            const oo = (s0 + i) * H + h * D;
            for (let c = 0; c < D; c++) {
              let a = 0;
              for (let j = 0; j < S; j++) a += P[j] * QKV[(s0 + j) * H3 + 2 * H + h * D + c];
              O[oo + c] = a / sum;
            }
          }
        }
      }
      linear(O, rows, Ly.Wo, Ly.bo, T);
      for (let q = 0; q < rows * H; q++) X[q] += T[q];
      layerNorm(X, rows, H, Ly.ln1w, Ly.ln1b, eps);
      linear(X, rows, Ly.Wi, Ly.bi, U);
      for (let q = 0; q < rows * I; q++) U[q] = gelu(U[q]);
      linear(U, rows, Ly.Wo2, Ly.bo2, T);
      for (let q = 0; q < rows * H; q++) X[q] += T[q];
      layerNorm(X, rows, H, Ly.ln2w, Ly.ln2b, eps);
    }
    const pooled = new Float32Array(L * H);
    seg.forEach(([s0, S], i) => {
      for (let j = 0; j < H; j++) {
        let a = 0;
        for (let t = 0; t < S; t++) a += X[(s0 + t) * H + j];
        pooled[i * H + j] = a / S;
      }
    });
    return { pooled, hidden: keepFirst && L ? X.slice(0, seg[0][1] * H) : null };
  }
}

// ---- span head: gref.GRef.span_logits, factorised ------------------------------------------------------
// S = mlp_start(W), E = mlp_end(W); a_i = W1a relu(S_i) + b1, b_j = W1b relu(E_j) (out_project.0 split
// at H); per label q_c = W2^T p_c, r_c = b2 . p_c with p_c = mlp_prompt(label) (out_project.3 folded
// into the label); logit(s, k, c) = relu(a_s + b_{s+k}) . q_c + r_c.
export class SpanHead {
  constructor(pack, gh, maxWidth) {
    const mlp = (r) => ({ l0: pack.lin(r.l0), b0: pack.f32(r.b0), l1: pack.lin(r.l1), b1: pack.f32(r.b1) });
    this.start = mlp(gh.start); this.end = mlp(gh.end); this.prompt = mlp(gh.prompt);
    const out = mlp(gh.out);
    this.H = this.start.l1.N; this.D = out.l0.N; this.K = maxWidth;
    if (out.l0.K !== 2 * this.H) throw new Error("out_project.0 must take concat(start, end)");
    this.W1a = Pack.kslice(out.l0, 0, this.H); this.W1b = Pack.kslice(out.l0, this.H, this.H);
    this.b1 = out.b0; this.W2 = out.l1; this.b2 = out.b1;
    this.sc = new Scratch();
  }

  mlp(r, X, rows, Y) {
    const h = this.sc.get("mlp_h", rows * r.l0.N);
    linear(X, rows, r.l0, r.b0, h, true);
    return linear(h, rows, r.l1, r.b1, Y);
  }

  // Pooled label vectors [L, H] -> { q: Float32Array(L * D), r: Float64Array(L) }, cacheable per label.
  labels(pooled, L) {
    const { H, D } = this, Pm = new Float32Array(L * H);
    this.mlp(this.prompt, pooled, L, Pm);
    const q = new Float32Array(L * D), r = new Float64Array(L);
    const { w, s, K } = this.W2;   // [N = H, K = D]: W2[n, k] = w[n * D + k] * s[n]
    const acc = new Float64Array(D);
    for (let c = 0; c < L; c++) {
      acc.fill(0);
      for (let n = 0; n < H; n++) {
        const pn = Pm[c * H + n] * (s ? s[n] : 1), wb = n * K;
        if (pn !== 0) for (let k = 0; k < D; k++) acc[k] += pn * w[wb + k];
      }
      q.set(acc, c * D);
      let b = 0;
      for (let n = 0; n < H; n++) b += this.b2[n] * Pm[c * H + n];
      r[c] = b;
    }
    return { q, r, Pm };
  }

  // Word vectors W [n, H] -> A, B [n, D] (views into scratch, valid until the next call).
  words(W, n) {
    const { H, D } = this;
    const S = this.sc.get("S", n * H), E = this.sc.get("E", n * H);
    this.mlp(this.start, W, n, S);
    this.mlp(this.end, W, n, E);
    for (let i = 0; i < n * H; i++) { if (S[i] < 0) S[i] = 0; if (E[i] < 0) E[i] = 0; }
    const A = this.sc.get("A", n * D), B = this.sc.get("B", n * D);
    linear(S, n, this.W1a, this.b1, A);
    linear(E, n, this.W1b, null, B);
    return { A, B };
  }

  // Logits [n, K, C] into out (Float32Array, -Infinity where s + k >= n). q: [C, D], r: [C].
  scores(A, B, n, q, r, C, out) {
    const { D, K } = this;
    const idx = this.idx ??= new Int32Array(D), val = this.val ??= new Float64Array(D);
    out.fill(-Infinity, 0, n * K * C);
    for (let s = 0; s < n; s++) {
      const kmax = Math.min(K, n - s), ab = s * D;
      for (let k = 0; k < kmax; k++) {
        const bb = (s + k) * D;
        let m = 0;
        for (let j = 0; j < D; j++) { const v = A[ab + j] + B[bb + j]; if (v > 0) { idx[m] = j; val[m] = v; m++; } }
        const ob = (s * K + k) * C;
        for (let c = 0; c < C; c++) {
          const qb = c * D;
          let a = r[c];
          for (let t = 0; t < m; t++) a += val[t] * q[qb + idx[t]];
          out[ob + c] = a;
        }
      }
    }
    return out;
  }
}

// ---- decoding: gref.GRef._decode + greedy_search --------------------------------------------------------
// Candidates with sigmoid(z / T + b) > threshold in (start, width, class) order; stable sort on -score;
// keep a span unless it overlaps a kept one (flat: any overlap, nested: partial overlap only; the same
// (start, end) is an overlap unless multiLabel); kept spans sorted by start (stable).
const fr = Math.fround;
export function decodeSpans(z, n, K, C, threshold, { flat = true, multiLabel = false, T = 1, b = 0 } = {}) {
  const thr = fr(threshold);
  // sigmoid is monotone: skip logits clearly below the threshold, decide the rest exactly in float32
  const zcut = thr <= 0 ? -Infinity : thr >= 1 ? Infinity : (Math.log(thr / (1 - thr)) - b) * T - 1e-3 * (1 + Math.abs(T));
  const cands = [];
  for (let s = 0; s < n; s++) for (let k = 0; k < K && s + k < n; k++) for (let c = 0; c < C; c++) {
    const zz = z[(s * K + k) * C + c];
    if (!(zz > zcut)) continue;
    const x = T === 1 && b === 0 ? zz : fr(fr(zz / T) + b);
    const p = fr(1 / (1 + Math.exp(-x)));
    if (p > thr) cands.push({ start: s, end: s + k, cls: c, score: p });
  }
  cands.sort((x, y) => y.score - x.score);
  const kept = [];
  for (const sp of cands) {
    let ok = true;
    for (const k of kept) {
      if (sp.start === k.start && sp.end === k.end) { if (!multiLabel) { ok = false; break; } continue; }
      const disjoint = sp.start > k.end || k.start > sp.end;
      if (flat ? !disjoint : !(disjoint || (sp.start <= k.start && sp.end >= k.end) || (k.start <= sp.start && k.end >= sp.end))) { ok = false; break; }
    }
    if (ok) kept.push(sp);
  }
  kept.sort((x, y) => x.start - y.start);
  return kept;
}

// ---- word splitting: gpack.WORD_REGEX with Python's Unicode \w and \s -------------------------------------
// Python's \w is exactly [\p{L}\p{N}_] (checked over all code points, Unicode 15.1); Python's \s is the
// str.isspace() set below, which differs from JavaScript's \s (\x1c-\x1f and U+0085 are spaces in
// Python, U+FEFF is not). Offsets are returned both in code points (GRef's) and UTF-16 units.
const WCH = "[\\p{L}\\p{N}_]";
const PYSPACE = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
export const WORD_RE = new RegExp(`${WCH}+(?:[-_]${WCH}+)*|[^${PYSPACE}]`, "gu");

// -> [{ w, cs, ce, us, ue }]: word, code-point start / end, UTF-16 start / end.
export function splitWords(text, max = Infinity) {
  const out = [];
  let u = 0, cp = 0;
  WORD_RE.lastIndex = 0;
  for (let m; out.length < max && (m = WORD_RE.exec(text)) !== null;) {
    for (; u < m.index; u++) { const c = text.charCodeAt(u); if (c < 0xdc00 || c > 0xdfff || u === 0 || (text.charCodeAt(u - 1) & 0xfc00) !== 0xd800) cp++; }
    const w = m[0], cs = cp;
    for (let i = 0; i < w.length; i++) { const c = w.charCodeAt(i); if ((c & 0xfc00) !== 0xdc00 || i === 0 || (w.charCodeAt(i - 1) & 0xfc00) !== 0xd800) cp++; }
    u = m.index + w.length;
    out.push({ w, cs, ce: cp, us: m.index, ue: u });
  }
  return out;
}
