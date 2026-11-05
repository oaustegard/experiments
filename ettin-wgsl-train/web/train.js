// LoRA training of a packed Ettin-17m (ModernBERT) decision model in WebGPU compute shaders.
//
// The frozen base is the browser-decider page format (int8 matmuls with a scale per output column, a
// 4-bit token table), dequantized to f32 once on upload. Trainable: a rank-r LoRA on every Wqkv, Wo and
// Wi, and the decision head (Linear, GELU, LayerNorm, Linear). One flat parameter vector holds them, in
// run/reference.py's layout: per layer Wqkv A, Wqkv B, Wo A, Wo B, Wi A, Wi B; then head w0, b0, ln_w,
// ln_b, w1, b1.
//
// One step: forward through the 7 layers keeping only each layer's input (checkpointing), the head and
// the segment-softmax cross-entropy loss, then for each layer from the top: recompute its forward from
// the saved input, and differentiate it. Attention backward runs in two passes without atomics: per
// query row (dQ and D = dO.O) and per key row (dK, dV), both recomputing the probabilities from the
// saved log-sum-exp. Layer 0's input needs no gradient, so its dX is skipped. Weight gradients that sum
// over every token (LoRA A and B) use a split-K matmul and a reduction. AdamW (PyTorch's update order),
// gradient clipping by global norm, all on the GPU; nothing is read back unless asked.
//
// Buffers are separate GPUBuffers per tensor, so no dispatch binds one buffer as both read and write.
// Uniforms live in one buffer with a 256-byte slot per dispatch (dynamic offsets); bind groups are cached.

const SMAX = 1024;          // longest sequence (the attention kernels' workgroup arrays)
const SLOT = 256;           // uniform slot bytes (minUniformBufferOffsetAlignment)
const MAX_SLOTS = 2048;
const KCHUNK = 256;         // rows per split-K slice

// Abramowitz & Stegun 7.1.26 erf (|error| < 1.5e-7): exact-erf GELU as in PyTorch, and its derivative.
const GELU = /* wgsl */ `
fn erf_(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return sign(x) * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.7071067811865476)); }
fn gelu_d(x: f32) -> f32 { return 0.5 * (1.0 + erf_(x * 0.7071067811865476)) + x * exp(-0.5 * x * x) * 0.3989422804014327; }
`;

const REDUCE = (n) => /* wgsl */ `
var<workgroup> red: array<f32, ${n}>;
fn wsum(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) {
    if (t < s) { red[t] = red[t] + red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn wmax(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) {
    if (t < s) { red[t] = max(red[t], red[t + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`;

const ENTRY = (wg) => `@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let gi = wid.x + wid.y * nwg.x;`;

// Kernel specs: u = uniform names (suffix ":f" for f32), b = [name, "r" | "rw", type], wg, src (body + helpers).
function kernels(C) {
  const { H, NH, D, I } = C;
  const K = {};
  // ---- embedding: 4-bit table row, LayerNorm (no bias) ---------------------------------------
  K.embed = { u: ["R", "eps:f", "wOff"], b: [["IDS", "r", "u32"], ["CODES", "r", "u32"], ["SCALES", "r", "f32"], ["W", "r", "f32"], ["X", "rw", "f32"]],
    wg: H, pre: REDUCE(H), body: `
  if (gi >= p.R) { return; }
  let j = lid.x;
  let tok = IDS[gi];
  let e = tok * ${H}u + j;
  let q = f32((CODES[e / 8u] >> ((e % 8u) * 4u)) & 15u) - 8.0;
  let v = q * SCALES[tok * ${H / 32}u + j / 32u];
  let mu = wsum(v, j) / ${H}.0;
  let d = v - mu;
  let va = wsum(d * d, j) / ${H}.0;
  X[gi * ${H}u + j] = d * inverseSqrt(va + p.eps) * W[p.wOff + j];` };
  // ---- LayerNorm forward (optional bias), saves mean and rstd --------------------------------
  K.ln = { u: ["R", "eps:f", "wOff", "bOff", "hasB"], b: [["X", "r", "f32"], ["W", "r", "f32"], ["Y", "rw", "f32"], ["ST", "rw", "f32"]],
    wg: H, pre: REDUCE(H), body: `
  if (gi >= p.R) { return; }
  let j = lid.x;
  let v = X[gi * ${H}u + j];
  let mu = wsum(v, j) / ${H}.0;
  let d = v - mu;
  let rs = inverseSqrt(wsum(d * d, j) / ${H}.0 + p.eps);
  var y = d * rs * W[p.wOff + j];
  if (p.hasB == 1u) { y = y + W[p.bOff + j]; }
  Y[gi * ${H}u + j] = y;
  if (j == 0u) { ST[2u * gi] = mu; ST[2u * gi + 1u] = rs; }` };
  // ---- LayerNorm backward to the input (frozen or trainable weight; acc adds into DX) --------
  K.ln_bwd = { u: ["R", "wOff", "acc"], b: [["X", "r", "f32"], ["ST", "r", "f32"], ["W", "r", "f32"], ["DY", "r", "f32"], ["DX", "rw", "f32"]],
    wg: H, pre: REDUCE(H), body: `
  if (gi >= p.R) { return; }
  let j = lid.x;
  let i = gi * ${H}u + j;
  let rs = ST[2u * gi + 1u];
  let xh = (X[i] - ST[2u * gi]) * rs;
  let g = DY[i] * W[p.wOff + j];
  let mg = wsum(g, j) / ${H}.0;
  let mgx = wsum(g * xh, j) / ${H}.0;
  let dx = rs * (g - mg - xh * mgx);
  if (p.acc == 1u) { DX[i] = DX[i] + dx; } else { DX[i] = dx; }` };
  // ---- LayerNorm weight and bias gradients (head LayerNorm, few rows) -------------------------
  K.ln_pgrad = { u: ["R", "gwOff", "gbOff"], b: [["X", "r", "f32"], ["ST", "r", "f32"], ["DY", "r", "f32"], ["G", "rw", "f32"]],
    wg: H, body: `
  let c = lid.x;
  var sw = 0.0;
  var sb = 0.0;
  for (var r = 0u; r < p.R; r = r + 1u) {
    let dy = DY[r * ${H}u + c];
    sw = sw + dy * (X[r * ${H}u + c] - ST[2u * r]) * ST[2u * r + 1u];
    sb = sb + dy;
  }
  G[p.gwOff + c] = sw;
  G[p.gbOff + c] = sb;` };
  // ---- tiled matmul: C[z] (=|+=) alpha * op(A) op(B), arbitrary strides, split-K over wid.z -----
  // A(m,k) = A[aOff + m*aRS + k*aCS], B(k,n) = B[bOff + k*bRS + n*bCS], C(m,n) = C[cOff + z*cZS + m*cRS + n*cCS]
  K.mm = { u: ["M", "N", "K", "kChunk", "aOff", "aRS", "aCS", "bOff", "bRS", "bCS", "cOff", "cRS", "cCS", "cZS", "acc", "alpha:f"],
    b: [["A", "r", "f32"], ["B", "r", "f32"], ["C", "rw", "f32"]], wg: "16, 16", pre: `
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;`, body: `
  let tid = lid.y * 16u + lid.x;
  let m0 = wid.y * 64u;
  let n0 = wid.x * 64u;
  let kb = wid.z * p.kChunk;
  let ke = min(p.K, kb + p.kChunk);
  var acc = array<f32, 16>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var k0 = kb; k0 < ke; k0 = k0 + 16u) {
    for (var s = 0u; s < 4u; s = s + 1u) {
      let e = tid + s * 256u;
      let mm = e / 16u;
      let kk = e % 16u;
      let gm = m0 + mm;
      let gk = k0 + kk;
      var v = 0.0;
      if (gm < p.M && gk < ke) { v = A[p.aOff + gm * p.aRS + gk * p.aCS]; }
      As[kk * 64u + mm] = v;
      let kk2 = e / 64u;
      let nn = e % 64u;
      let gk2 = k0 + kk2;
      let gn = n0 + nn;
      var w = 0.0;
      if (gn < p.N && gk2 < ke) { w = B[p.bOff + gk2 * p.bRS + gn * p.bCS]; }
      Bs[kk2 * 64u + nn] = w;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk = kk + 1u) {
      let a = vec4<f32>(As[kk * 64u + lid.y * 4u], As[kk * 64u + lid.y * 4u + 1u], As[kk * 64u + lid.y * 4u + 2u], As[kk * 64u + lid.y * 4u + 3u]);
      let b = vec4<f32>(Bs[kk * 64u + lid.x * 4u], Bs[kk * 64u + lid.x * 4u + 1u], Bs[kk * 64u + lid.x * 4u + 2u], Bs[kk * 64u + lid.x * 4u + 3u]);
      for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i * 4u] = acc[i * 4u] + a[i] * b.x;
        acc[i * 4u + 1u] = acc[i * 4u + 1u] + a[i] * b.y;
        acc[i * 4u + 2u] = acc[i * 4u + 2u] + a[i] * b.z;
        acc[i * 4u + 3u] = acc[i * 4u + 3u] + a[i] * b.w;
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i = i + 1u) {
    for (var j = 0u; j < 4u; j = j + 1u) {
      let m = m0 + lid.y * 4u + i;
      let n = n0 + lid.x * 4u + j;
      if (m < p.M && n < p.N) {
        let ix = p.cOff + wid.z * p.cZS + m * p.cRS + n * p.cCS;
        var v = p.alpha * acc[i * 4u + j];
        if (p.acc == 1u) { v = v + C[ix]; }
        C[ix] = v;
      }
    }
  }` };
  // ---- sum split-K partials [Z, M, N] into C ---------------------------------------------------
  K.sumz = { u: ["M", "N", "Z", "cOff", "cRS", "cCS", "acc", "alpha:f"], b: [["P", "r", "f32"], ["C", "rw", "f32"]], wg: 256, body: `
  let e = gi * 256u + lid.x;
  if (e >= p.M * p.N) { return; }
  let m = e / p.N;
  let n = e % p.N;
  var s = 0.0;
  for (var z = 0u; z < p.Z; z = z + 1u) { s = s + P[z * p.M * p.N + e]; }
  let ix = p.cOff + m * p.cRS + n * p.cCS;
  var v = p.alpha * s;
  if (p.acc == 1u) { v = v + C[ix]; }
  C[ix] = v;` };
  K.copy = { u: ["n", "acc"], b: [["X", "r", "f32"], ["Y", "rw", "f32"]], wg: 256, body: `
  let e = gi * 256u + lid.x;
  if (e >= p.n) { return; }
  if (p.acc == 1u) { Y[e] = Y[e] + X[e]; } else { Y[e] = X[e]; }` };
  // ---- RoPE on q and k: one workgroup per token, one thread per (q|k, head, pair) ----------------
  K.rope = { u: ["R", "S", "sinOff"], b: [["QKV", "r", "f32"], ["CS", "r", "f32"], ["QR", "rw", "f32"]], wg: H, body: `
  if (gi >= p.R) { return; }
  let t = lid.x;
  let sel = t / ${H / 2}u;
  let h = (t % ${H / 2}u) / ${D / 2}u;
  let d = t % ${D / 2}u;
  let pos = gi % p.S;
  let c = CS[pos * ${D / 2}u + d];
  let s = CS[p.sinOff + pos * ${D / 2}u + d];
  let bi = gi * ${3 * H}u + sel * ${H}u + h * ${D}u;
  let x1 = QKV[bi + d];
  let x2 = QKV[bi + d + ${D / 2}u];
  let bo = gi * ${2 * H}u + sel * ${H}u + h * ${D}u;
  QR[bo + d] = x1 * c - x2 * s;
  QR[bo + d + ${D / 2}u] = x2 * c + x1 * s;` };
  K.rope_bwd = { u: ["R", "S", "sinOff"], b: [["DQR", "r", "f32"], ["CS", "r", "f32"], ["DQKV", "rw", "f32"]], wg: H, body: `
  if (gi >= p.R) { return; }
  let t = lid.x;
  let sel = t / ${H / 2}u;
  let h = (t % ${H / 2}u) / ${D / 2}u;
  let d = t % ${D / 2}u;
  let pos = gi % p.S;
  let c = CS[pos * ${D / 2}u + d];
  let s = CS[p.sinOff + pos * ${D / 2}u + d];
  let bo = gi * ${2 * H}u + sel * ${H}u + h * ${D}u;
  let g1 = DQR[bo + d];
  let g2 = DQR[bo + d + ${D / 2}u];
  let bi = gi * ${3 * H}u + sel * ${H}u + h * ${D}u;
  DQKV[bi + d] = g1 * c + g2 * s;
  DQKV[bi + d + ${D / 2}u] = g2 * c - g1 * s;` };
  // ---- attention: one workgroup of D threads per (token row, head) --------------------------------
  const ATT_HEAD = `
  if (gi >= p.R * ${NH}u) { return; }
  let r = gi / ${NH}u;
  let h = gi % ${NH}u;
  let bq = r / p.S;
  let i = r % p.S;
  let t = lid.x;
  if (t == 0u) { lenS = LENS[bq]; }
  let len = workgroupUniformLoad(&lenS);
  var lo = 0u;
  var hi = len;
  if (p.isLocal == 1u) {
    lo = select(0u, i - p.win, i > p.win);
    hi = min(len, i + p.win + 1u);
  }
  let base = bq * p.S;
  let scale = ${1 / Math.sqrt(D)};`;
  const ATT_PRE = `${REDUCE(D)}
var<workgroup> lenS: u32;
var<workgroup> qs: array<f32, ${D}>;
var<workgroup> ws: array<f32, ${D}>;
var<workgroup> sc: array<f32, ${SMAX}>;
var<workgroup> sd: array<f32, ${SMAX}>;`;
  K.attn = { u: ["R", "S", "isLocal", "win"], b: [["QR", "r", "f32"], ["QKV", "r", "f32"], ["LENS", "r", "u32"], ["O", "rw", "f32"], ["LSE", "rw", "f32"]],
    wg: D, pre: ATT_PRE, body: `${ATT_HEAD}
  if (i >= len) { O[r * ${H}u + h * ${D}u + t] = 0.0; if (t == 0u) { LSE[gi] = 0.0; } return; }
  qs[t] = QR[r * ${2 * H}u + h * ${D}u + t];
  workgroupBarrier();
  let n = hi - lo;
  var mx = -3.0e38;
  for (var jj = t; jj < n; jj = jj + ${D}u) {
    let kr = (base + lo + jj) * ${2 * H}u + ${H}u + h * ${D}u;
    var s = 0.0;
    for (var d = 0u; d < ${D}u; d = d + 1u) { s = s + qs[d] * QR[kr + d]; }
    s = s * scale;
    sc[jj] = s;
    mx = max(mx, s);
  }
  mx = wmax(mx, t);
  var l = 0.0;
  for (var jj = t; jj < n; jj = jj + ${D}u) { let e = exp(sc[jj] - mx); sc[jj] = e; l = l + e; }
  l = wsum(l, t);
  var o = 0.0;
  for (var jj = 0u; jj < n; jj = jj + 1u) { o = o + sc[jj] * QKV[(base + lo + jj) * ${3 * H}u + ${2 * H}u + h * ${D}u + t]; }
  O[r * ${H}u + h * ${D}u + t] = o / l;
  if (t == 0u) { LSE[gi] = mx + log(l); }` };
  // pass 1: per query row: D_i = dO_i . O_i, dS_ij = P_ij (dO_i . v_j - D_i), dQ_i = scale sum_j dS_ij k_j
  K.attn_bq = { u: ["R", "S", "isLocal", "win"], b: [["QR", "r", "f32"], ["QKV", "r", "f32"], ["LENS", "r", "u32"], ["LSE", "r", "f32"], ["DO", "r", "f32"], ["O", "r", "f32"], ["DQR", "rw", "f32"], ["DD", "rw", "f32"]],
    wg: D, pre: ATT_PRE, body: `${ATT_HEAD}
  if (i >= len) { DQR[r * ${2 * H}u + h * ${D}u + t] = 0.0; if (t == 0u) { DD[gi] = 0.0; } return; }
  let oi = r * ${H}u + h * ${D}u + t;
  qs[t] = QR[r * ${2 * H}u + h * ${D}u + t];
  ws[t] = DO[oi];
  let Di = wsum(DO[oi] * O[oi], t);
  let lse = LSE[gi];
  let n = hi - lo;
  for (var jj = t; jj < n; jj = jj + ${D}u) {
    let kr = (base + lo + jj) * ${2 * H}u + ${H}u + h * ${D}u;
    let vr = (base + lo + jj) * ${3 * H}u + ${2 * H}u + h * ${D}u;
    var s = 0.0;
    var dp = 0.0;
    for (var d = 0u; d < ${D}u; d = d + 1u) { s = s + qs[d] * QR[kr + d]; dp = dp + ws[d] * QKV[vr + d]; }
    let pr = exp(s * scale - lse);
    sd[jj] = pr * (dp - Di);
  }
  workgroupBarrier();
  var dq = 0.0;
  for (var jj = 0u; jj < n; jj = jj + 1u) { dq = dq + sd[jj] * QR[(base + lo + jj) * ${2 * H}u + ${H}u + h * ${D}u + t]; }
  DQR[r * ${2 * H}u + h * ${D}u + t] = dq * scale;
  if (t == 0u) { DD[gi] = Di; }` };
  // pass 2: per key row j: dV_j = sum_i P_ij dO_i, dK_j = scale sum_i dS_ij q_i (queries i < len that see j)
  K.attn_bkv = { u: ["R", "S", "isLocal", "win"], b: [["QR", "r", "f32"], ["QKV", "r", "f32"], ["LENS", "r", "u32"], ["LSE", "r", "f32"], ["DO", "r", "f32"], ["DD", "r", "f32"], ["DQR", "rw", "f32"], ["DQKV", "rw", "f32"]],
    wg: D, pre: ATT_PRE, body: `${ATT_HEAD}
  if (i >= len) { DQR[r * ${2 * H}u + ${H}u + h * ${D}u + t] = 0.0; DQKV[r * ${3 * H}u + ${2 * H}u + h * ${D}u + t] = 0.0; return; }
  qs[t] = QR[r * ${2 * H}u + ${H}u + h * ${D}u + t];
  ws[t] = QKV[r * ${3 * H}u + ${2 * H}u + h * ${D}u + t];
  workgroupBarrier();
  let n = hi - lo;
  for (var ii = t; ii < n; ii = ii + ${D}u) {
    let q = base + lo + ii;
    var s = 0.0;
    var dp = 0.0;
    for (var d = 0u; d < ${D}u; d = d + 1u) { s = s + qs[d] * QR[q * ${2 * H}u + h * ${D}u + d]; dp = dp + ws[d] * DO[q * ${H}u + h * ${D}u + d]; }
    let pr = exp(s * scale - LSE[q * ${NH}u + h]);
    sc[ii] = pr;
    sd[ii] = pr * (dp - DD[q * ${NH}u + h]);
  }
  workgroupBarrier();
  var dv = 0.0;
  var dk = 0.0;
  for (var ii = 0u; ii < n; ii = ii + 1u) {
    let q = base + lo + ii;
    dv = dv + sc[ii] * DO[q * ${H}u + h * ${D}u + t];
    dk = dk + sd[ii] * QR[q * ${2 * H}u + h * ${D}u + t];
  }
  DQR[r * ${2 * H}u + ${H}u + h * ${D}u + t] = dk * scale;
  DQKV[r * ${3 * H}u + ${2 * H}u + h * ${D}u + t] = dv;` };
  // ---- GeGLU: g = gelu(u[:, :I]) * u[:, I:] -------------------------------------------------------
  K.geglu = { u: ["n"], b: [["U", "r", "f32"], ["G", "rw", "f32"]], wg: 256, pre: GELU, body: `
  let e = gi * 256u + lid.x;
  if (e >= p.n) { return; }
  let r = e / ${I}u;
  let c = e % ${I}u;
  G[e] = gelu(U[r * ${2 * I}u + c]) * U[r * ${2 * I}u + ${I}u + c];` };
  K.geglu_bwd = { u: ["n"], b: [["U", "r", "f32"], ["DG", "r", "f32"], ["DU", "rw", "f32"]], wg: 256, pre: GELU, body: `
  let e = gi * 256u + lid.x;
  if (e >= p.n) { return; }
  let r = e / ${I}u;
  let c = e % ${I}u;
  let a = U[r * ${2 * I}u + c];
  let b = U[r * ${2 * I}u + ${I}u + c];
  let dg = DG[e];
  DU[r * ${2 * I}u + c] = dg * b * gelu_d(a);
  DU[r * ${2 * I}u + ${I}u + c] = dg * gelu(a);` };
  // ---- head ---------------------------------------------------------------------------------------
  K.gather = { u: ["R"], b: [["POS", "r", "u32"], ["X", "r", "f32"], ["Y", "rw", "f32"]], wg: H, body: `
  if (gi >= p.R) { return; }
  Y[gi * ${H}u + lid.x] = X[POS[gi] * ${H}u + lid.x];` };
  K.scatter = { u: ["R"], b: [["POS", "r", "u32"], ["X", "r", "f32"], ["Y", "rw", "f32"]], wg: H, body: `
  if (gi >= p.R) { return; }
  Y[POS[gi] * ${H}u + lid.x] = X[gi * ${H}u + lid.x];` };
  K.bias_gelu = { u: ["R", "bOff"], b: [["PRM", "r", "f32"], ["Z", "rw", "f32"], ["A", "rw", "f32"]], wg: H, pre: GELU, body: `
  if (gi >= p.R) { return; }
  let i = gi * ${H}u + lid.x;
  let z = Z[i] + PRM[p.bOff + lid.x];
  Z[i] = z;
  A[i] = gelu(z);` };
  K.gelu_bwd = { u: ["n"], b: [["Z", "r", "f32"], ["DA", "r", "f32"], ["DZ", "rw", "f32"]], wg: 256, pre: GELU, body: `
  let e = gi * 256u + lid.x;
  if (e >= p.n) { return; }
  DZ[e] = DA[e] * gelu_d(Z[e]);` };
  K.colsum = { u: ["R", "gOff"], b: [["X", "r", "f32"], ["G", "rw", "f32"]], wg: H, body: `
  var s = 0.0;
  for (var r = 0u; r < p.R; r = r + 1u) { s = s + X[r * ${H}u + lid.x]; }
  G[p.gOff + lid.x] = s;` };
  K.head_out = { u: ["R", "wOff", "bOff"], b: [["X", "r", "f32"], ["PRM", "r", "f32"], ["Y", "rw", "f32"]], wg: H, pre: REDUCE(H), body: `
  if (gi >= p.R) { return; }
  let s = wsum(X[gi * ${H}u + lid.x] * PRM[p.wOff + lid.x], lid.x);
  if (lid.x == 0u) { Y[gi] = s + PRM[p.bOff]; }` };
  K.head_out_bwd = { u: ["R", "wOff", "gwOff", "gbOff"], b: [["DL", "r", "f32"], ["X", "r", "f32"], ["PRM", "r", "f32"], ["DX", "rw", "f32"], ["G", "rw", "f32"]], wg: H, body: `
  let c = lid.x;
  var sw = 0.0;
  var sb = 0.0;
  for (var r = 0u; r < p.R; r = r + 1u) {
    sw = sw + DL[r] * X[r * ${H}u + c];
    sb = sb + DL[r];
    DX[r * ${H}u + c] = DL[r] * PRM[p.wOff + c];
  }
  G[p.gwOff + c] = sw;
  if (c == 0u) { G[p.gbOff] = sb; }` };
  // segment-softmax cross-entropy on logits / T; one thread (a batch has a few dozen options)
  K.loss = { u: ["nSeg", "T:f", "slot"], b: [["Z", "r", "f32"], ["SEG", "r", "u32"], ["GOLD", "r", "f32"], ["DZ", "rw", "f32"], ["LOSS", "rw", "f32"]], wg: 1, body: `
  var o = 0u;
  var tot = 0.0;
  for (var s = 0u; s < p.nSeg; s = s + 1u) {
    let k = SEG[s];
    var mx = -3.0e38;
    for (var j = 0u; j < k; j = j + 1u) { mx = max(mx, Z[o + j] / p.T); }
    var l = 0.0;
    var g = 0.0;
    for (var j = 0u; j < k; j = j + 1u) { l = l + exp(Z[o + j] / p.T - mx); g = g + GOLD[o + j]; }
    let lse = mx + log(l);
    for (var j = 0u; j < k; j = j + 1u) {
      let lp = Z[o + j] / p.T - lse;
      tot = tot - GOLD[o + j] * lp;
      DZ[o + j] = (exp(lp) * g - GOLD[o + j]) / (p.T * f32(p.nSeg));
    }
    o = o + k;
  }
  LOSS[p.slot] = tot / f32(p.nSeg);` };
  // ---- optimizer ----------------------------------------------------------------------------------
  K.sqsum = { u: ["n"], b: [["G", "r", "f32"], ["PART", "rw", "f32"]], wg: 256, pre: REDUCE(256), body: `
  var s = 0.0;
  for (var e = gi * 256u + lid.x; e < p.n; e = e + nwg.x * 256u) { let g = G[e]; s = s + g * g; }
  s = wsum(s, lid.x);
  if (lid.x == 0u) { PART[gi] = s; }` };
  K.norm = { u: ["n"], b: [["PART", "r", "f32"], ["NRM", "rw", "f32"]], wg: 256, pre: REDUCE(256), body: `
  var s = 0.0;
  for (var e = lid.x; e < p.n; e = e + 256u) { s = s + PART[e]; }
  s = wsum(s, lid.x);
  if (lid.x == 0u) { NRM[0] = sqrt(s); }` };
  K.adamw = { u: ["n", "headOff", "lr:f", "lrHead:f", "wd:f", "b1:f", "b2:f", "eps:f", "bc1:f", "bc2s:f", "clip:f"],
    b: [["P", "rw", "f32"], ["G", "r", "f32"], ["M", "rw", "f32"], ["V", "rw", "f32"], ["NRM", "r", "f32"]], wg: 256, body: `
  let e = gi * 256u + lid.x;
  if (e >= p.n) { return; }
  let cf = min(1.0, p.clip / (NRM[0] + 1e-6));
  let g = G[e] * cf;
  let lr = select(p.lr, p.lrHead, e >= p.headOff);
  let w = P[e] * (1.0 - lr * p.wd);
  let m = p.b1 * M[e] + (1.0 - p.b1) * g;
  let v = p.b2 * V[e] + (1.0 - p.b2) * g * g;
  M[e] = m;
  V[e] = v;
  P[e] = w - (lr / p.bc1) * m / (sqrt(v) / p.bc2s + p.eps);` };
  return K;
}

function wgsl(name, k) {
  const fields = k.u.map((s) => { const [n, t] = s.split(":"); return `  ${n}: ${t === "f" ? "f32" : "u32"},`; });
  const st = `struct Uni {\n${fields.join("\n")}\n}\n@group(0) @binding(0) var<uniform> p: Uni;\n`;
  const binds = k.b.map(([n, m, t], i) => `@group(0) @binding(${i + 1}) var<storage, ${m === "rw" ? "read_write" : "read"}> ${n}: array<${t}>;`).join("\n");
  return `// ${name}\n${st}${binds}\n${k.pre || ""}\n${ENTRY(k.wg)}\n${k.body}\n}\n`;
}

const f16 = (h) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
};

export class Trainer {
  // opts: { manifest, weights (ArrayBuffer | Uint8Array of weights.bin), maxRows, maxSeqs, maxPool, maxSeg, rank, alpha }
  static async create(device, opts) {
    const t = new Trainer(device, opts);
    await t.#init();
    return t;
  }

  constructor(device, { manifest, weights, maxRows, maxSeqs = 64, maxPool = 256, maxSeg = 128, rank = 8, alpha = 16 }) {
    this.device = device;
    this.man = manifest;
    this.wbytes = weights instanceof Uint8Array ? weights : new Uint8Array(weights);
    const H = manifest.hidden, NH = manifest.heads;
    this.C = { H, NH, D: H / NH, I: manifest.inter, L: manifest.layers.length, r: rank, s: alpha / rank };
    if (H !== 256 || this.C.D !== 64) throw new Error(`kernels assume hidden 256, head dim 64 (got ${H}, ${this.C.D})`);
    this.max = { R: maxRows, B: maxSeqs, P: maxPool, G: maxSeg };
    this.ops = [];
    this.bgCache = new Map();
    this.nid = 0;
  }

  async #init() {
    const dev = this.device, { H, I, L, r, NH, D } = this.C, m = this.man, R = this.max.R;
    const K = kernels(this.C);
    this.K = {};
    await Promise.all(Object.entries(K).map(async ([name, k]) => {
      const code = wgsl(name, k);
      const mod = dev.createShaderModule({ code, label: name });
      const entries = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
        ...k.b.map(([, mode], i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: mode === "rw" ? "storage" : "read-only-storage" } }))];
      const bgl = dev.createBindGroupLayout({ entries, label: name });
      let pipe;
      try {
        pipe = await dev.createComputePipelineAsync({ layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: mod, entryPoint: "main" }, label: name });
      } catch (e) {
        const info = await mod.getCompilationInfo();
        throw new Error(`${name}: ${e.message}\n${info.messages.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("\n")}`);
      }
      this.K[name] = { pipe, bgl, u: k.u.map((s) => s.split(":")), nb: k.b.length };
    }));
    this.ubo = dev.createBuffer({ size: SLOT * MAX_SLOTS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "uniforms" });
    this.udata = new ArrayBuffer(SLOT * MAX_SLOTS);
    this.uview = new DataView(this.udata);

    // frozen weights, dequantized: per layer attn_norm, Wqkv [3H,H], Wo [H,H], mlp_norm, Wi [2I,H], Wo_mlp [H,I] (PyTorch [out, in])
    const tf = (rec) => new Float32Array(this.wbytes.buffer.slice(this.wbytes.byteOffset + rec.offset, this.wbytes.byteOffset + rec.offset + rec.length));
    const tu8 = (rec) => this.wbytes.slice(rec.offset, rec.offset + rec.length);
    const lin = (w) => {   // file: q [K, N] int8, s [N] -> f32 [N, K]
      const q = new Int8Array(tu8(w.q).buffer), s = tf(w.s), out = new Float32Array(w.N * w.K);
      for (let k = 0; k < w.K; k++) for (let n = 0; n < w.N; n++) out[n * w.K + k] = Math.fround(q[k * w.N + n] * s[n]);
      return out;
    };
    const parts = [], off = {};
    let n = 0;
    const put = (name, arr) => { off[name] = n; parts.push(arr); n += arr.length; };
    put("emb_norm", tf(m.emb_norm));
    put("final_norm", tf(m.final_norm));
    m.layers.forEach((Ly, i) => {
      if (Ly.attn_norm) put(`L${i}.attn_norm`, tf(Ly.attn_norm));
      put(`L${i}.Wqkv`, lin(Ly.Wqkv));
      put(`L${i}.Wo`, lin(Ly.Wo));
      put(`L${i}.mlp_norm`, tf(Ly.mlp_norm));
      put(`L${i}.Wi`, lin(Ly.Wi));
      put(`L${i}.Wo_mlp`, lin(Ly.Wo_mlp));
    });
    const wb = new Float32Array(n);
    let o = 0;
    for (const a of parts) { wb.set(a, o); o += a.length; }
    this.woff = off;
    this.WB = this.#buf("WB", wb.length, wb);
    const T = m.table;
    const codes = tu8(T.codes);
    this.CODES = this.#buf("CODES", codes.length / 4, new Uint32Array(codes.buffer, codes.byteOffset, codes.length / 4));
    const sh = new Uint16Array(tu8(T.scales).buffer), sc = new Float32Array(sh.length);
    for (let i = 0; i < sh.length; i++) sc[i] = f16(sh[i]);
    this.SCALES = this.#buf("SCALES", sc.length, sc);
    // RoPE cos / sin [SMAX, D/2], computed the way PyTorch does it in float32
    const half = D / 2, cs = new Float32Array(2 * SMAX * half), fr = Math.fround;
    const rope = (theta) => {
      for (let j = 0; j < half; j++) {
        const inv = fr(1 / fr(theta ** fr((2 * j) / D)));
        for (let p = 0; p < SMAX; p++) { const a = fr(p * inv); cs[p * half + j] = Math.cos(a); cs[SMAX * half + p * half + j] = Math.sin(a); }
      }
    };
    if (m.theta_global !== m.theta_local) throw new Error("separate local / global RoPE tables not implemented");
    rope(m.theta_global);
    this.CS = this.#buf("CS", cs.length, cs);

    // trainable parameters: per layer [Wqkv A, B, Wo A, B, Wi A, B], then head
    const tg = [["Wqkv", 3 * H, H], ["Wo", H, H], ["Wi", 2 * I, H]];
    const lay = [];
    let pn = 0;
    const add = (name, size, shape) => { lay.push({ name, offset: pn, shape }); pn += size; };
    for (let i = 0; i < L; i++) for (const [t, N, Kin] of tg) { add(`L${i}.${t}.A`, r * Kin, [r, Kin]); add(`L${i}.${t}.B`, N * r, [N, r]); }
    add("head.w0", H * H, [H, H]); add("head.b0", H, [H]); add("head.ln_w", H, [H]); add("head.ln_b", H, [H]); add("head.w1", H, [1, H]); add("head.b1", 1, [1]);
    this.layout = lay;
    this.poff = Object.fromEntries(lay.map((x) => [x.name, x.offset]));
    this.nParams = pn;
    this.headOff = this.poff["head.w0"];
    for (const nm of ["PRM", "GRD", "AM", "AV"]) this[nm] = this.#buf(nm, pn);
    this.NPART = 64;
    this.PARTN = this.#buf("PARTN", this.NPART);
    this.NRM = this.#buf("NRM", 4);

    // activations (sized for maxRows tokens) and gradients
    const P = this.max.P;
    this.XS = Array.from({ length: L + 1 }, (_, i) => this.#buf(`x${i}`, R * H));
    const A = (name, size) => { this[name] = this.#buf(name, size); };
    A("Hn", R * H); A("ST1", 2 * R); A("QKV", R * 3 * H); A("QR", R * 2 * H); A("O", R * H); A("LSE", R * NH);
    A("XM", R * H); A("H2", R * H); A("ST2", 2 * R); A("U", R * 2 * I); A("G", R * I);
    A("T1", R * r); A("T2", R * r); A("T3", R * r);
    A("DXA", R * H); A("DXB", R * H); A("DG", R * I); A("DU", R * 2 * I); A("DT", R * r); A("DH2", R * H); A("DO", R * H);
    A("DH", R * H); A("DXM", R * H); A("DQR", R * 2 * H); A("DQKV", R * 3 * H); A("DD", R * NH);
    this.ZMAX = Math.ceil(R / KCHUNK);
    A("PART", this.ZMAX * 3 * H * r);
    A("HP", P * H); A("STF", 2 * P); A("HF", P * H); A("Z1", P * H); A("A1", P * H); A("STH", 2 * P); A("N1", P * H);
    A("LOGIT", P); A("DL", P); A("DN1", P * H); A("DA1", P * H); A("DZ1", P * H); A("DHF", P * H); A("DHP", P * H);
    this.IDS = this.#buf("IDS", R, null, "u32");
    this.LENS = this.#buf("LENS", this.max.B, null, "u32");
    this.POS = this.#buf("POS", P, null, "u32");
    this.SEG = this.#buf("SEG", this.max.G, null, "u32");
    this.GOLD = this.#buf("GOLD", P);
    this.LOSS = this.#buf("LOSS", 4096);
  }

  #buf(label, n, init = null) {
    const size = Math.max(16, Math.ceil((n * 4) / 16) * 16);
    const b = this.device.createBuffer({ size, label, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: !!init });
    if (init) {
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(init.buffer, init.byteOffset, init.byteLength));
      b.unmap();
    }
    b._id = this.nid++;
    b._n = n;
    return b;
  }

  // record one dispatch
  op(name, bufs, vals, groups) {
    const k = this.K[name];
    if (!k) throw new Error(`no kernel ${name}`);
    if (bufs.length !== k.nb || bufs.some((b) => !b)) throw new Error(`${name}: expected ${k.nb} buffers`);
    this.ops.push({ k, name, bufs, vals, groups: Array.isArray(groups) ? groups : grid(groups) });
  }

  clear(buf) { this.ops.push({ clear: buf }); }

  // submit the recorded dispatches; wait = await the GPU (queue order already keeps later uploads behind it)
  async flush(wait = true) {
    if (!this.ops.length) { if (wait) await this.device.queue.onSubmittedWorkDone(); return; }
    const dev = this.device;
    let slots = 0;
    for (const o of this.ops) {
      if (o.clear) continue;
      if (slots >= MAX_SLOTS) throw new Error("too many dispatches in one flush");
      o.slot = slots++;
      o.k.u.forEach(([n, t], i) => {
        const v = o.vals[n] ?? 0;
        if (t === "f") this.uview.setFloat32(o.slot * SLOT + 4 * i, v, true);
        else this.uview.setUint32(o.slot * SLOT + 4 * i, v, true);
      });
    }
    dev.queue.writeBuffer(this.ubo, 0, this.udata, 0, slots * SLOT);
    const enc = dev.createCommandEncoder();
    let pass = null;
    for (const o of this.ops) {
      if (o.clear) {
        if (pass) { pass.end(); pass = null; }
        enc.clearBuffer(o.clear);
        continue;
      }
      if (!pass) pass = enc.beginComputePass();
      const key = o.name + "|" + o.bufs.map((b) => b._id).join(",");
      let bg = this.bgCache.get(key);
      if (!bg) {
        bg = dev.createBindGroup({ layout: o.k.bgl, entries: [{ binding: 0, resource: { buffer: this.ubo, offset: 0, size: SLOT } },
          ...o.bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))] });
        this.bgCache.set(key, bg);
      }
      pass.setPipeline(o.k.pipe);
      pass.setBindGroup(0, bg, [o.slot * SLOT]);
      pass.dispatchWorkgroups(...o.groups);
    }
    if (pass) pass.end();
    dev.queue.submit([enc.finish()]);
    this.ops = [];
    if (wait) await dev.queue.onSubmittedWorkDone();
  }

  async read(buf, n = buf._n, type = Float32Array) {
    await this.flush();
    const bytes = Math.ceil((n * 4) / 4) * 4;
    const st = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, st, 0, Math.ceil(bytes / 4) * 4);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new type(st.getMappedRange(0, bytes).slice(0));
    st.unmap();
    st.destroy();
    return out;
  }

  setParams(flat) {
    if (flat.length !== this.nParams) throw new Error(`params: ${flat.length} != ${this.nParams}`);
    this.device.queue.writeBuffer(this.PRM, 0, flat);
    this.clear(this.AM);
    this.clear(this.AV);
  }

  // batch: { ids Int32Array [B*S], B, S, lens [B], pos [nP] (flat row b*S+t), seg [nSeg], gold [nP] }
  loadBatch(bt) {
    const q = this.device.queue;
    if (bt.B * bt.S > this.max.R || bt.B > this.max.B || bt.pos.length > this.max.P || bt.seg.length > this.max.G || bt.S > SMAX)
      throw new Error(`batch ${bt.B}x${bt.S} (${bt.pos.length} options) exceeds the trainer's sizes`);
    q.writeBuffer(this.IDS, 0, Uint32Array.from(bt.ids));
    q.writeBuffer(this.LENS, 0, Uint32Array.from(bt.lens));
    q.writeBuffer(this.POS, 0, Uint32Array.from(bt.pos));
    q.writeBuffer(this.SEG, 0, Uint32Array.from(bt.seg));
    if (bt.gold) q.writeBuffer(this.GOLD, 0, Float32Array.from(bt.gold));
    this.b = { R: bt.B * bt.S, B: bt.B, S: bt.S, nP: bt.pos.length, nSeg: bt.seg.length };
  }

  // C (=|+=) alpha * X W^T for a frozen W [N, K] at WB offset (the forward linear)
  #lin(X, wname, N, Kin, Y, R, acc = 0) {
    this.op("mm", [X, this.WB, Y], { M: R, N, K: Kin, kChunk: Kin, aOff: 0, aRS: Kin, aCS: 1, bOff: this.woff[wname], bRS: 1, bCS: Kin,
      cOff: 0, cRS: N, cCS: 1, acc, alpha: 1 }, [Math.ceil(N / 64), Math.ceil(R / 64), 1]);
  }

  // dX (=|+=) dY W for a frozen W [N, K]
  #linT(DY, wname, N, Kin, DX, R, acc = 0) {
    this.op("mm", [DY, this.WB, DX], { M: R, N: Kin, K: N, kChunk: N, aOff: 0, aRS: N, aCS: 1, bOff: this.woff[wname], bRS: Kin, bCS: 1,
      cOff: 0, cRS: Kin, cCS: 1, acc, alpha: 1 }, [Math.ceil(Kin / 64), Math.ceil(R / 64), 1]);
  }

  // LoRA forward: T = X A^T, Y += s T B^T
  #loraF(X, pfx, N, Kin, T, Y, R) {
    const { r, s } = this.C;
    this.op("mm", [X, this.PRM, T], { M: R, N: r, K: Kin, kChunk: Kin, aRS: Kin, aCS: 1, bOff: this.poff[pfx + ".A"], bRS: 1, bCS: Kin, cRS: r, cCS: 1, alpha: 1 },
      [1, Math.ceil(R / 64), 1]);
    this.op("mm", [T, this.PRM, Y], { M: R, N, K: r, kChunk: r, aRS: r, aCS: 1, bOff: this.poff[pfx + ".B"], bRS: 1, bCS: r, cRS: N, cCS: 1, acc: 1, alpha: s },
      [Math.ceil(N / 64), Math.ceil(R / 64), 1]);
  }

  // split-K product into GRD: G[name] (layout [M, N], row-major) = alpha * op(A) op(B), K = rows R
  #gradK(A, B, name, M, N, R, a, b, alpha) {
    const Z = Math.ceil(R / KCHUNK);
    this.op("mm", [A, B, this.PART], { M, N, K: R, kChunk: KCHUNK, ...a, ...b, cOff: 0, cRS: N, cCS: 1, cZS: M * N, alpha: 1 },
      [Math.ceil(N / 64), Math.ceil(M / 64), Z]);
    this.op("sumz", [this.PART, this.GRD], { M, N, Z, cOff: this.poff[name], cRS: N, cCS: 1, alpha }, Math.ceil((M * N) / 256));
  }

  // LoRA backward. DY [R, N] is the gradient of the LoRA'd output, X [R, K] its input, T = X A^T.
  // Writes dA, dB into GRD; DT = s DY B; DX += DT A (when DX given).
  #loraB(DY, X, T, pfx, N, Kin, R, DX) {
    const { r, s } = this.C;
    this.op("mm", [DY, this.PRM, this.DT], { M: R, N: r, K: N, kChunk: N, aRS: N, aCS: 1, bOff: this.poff[pfx + ".B"], bRS: r, bCS: 1, cRS: r, cCS: 1, alpha: s },
      [1, Math.ceil(R / 64), 1]);
    // dB [N, r] = s DY^T T
    this.#gradK(DY, T, pfx + ".B", N, r, R, { aOff: 0, aRS: 1, aCS: N }, { bOff: 0, bRS: r, bCS: 1 }, s);
    // dA [r, K] = DT^T X
    this.#gradK(this.DT, X, pfx + ".A", r, Kin, R, { aOff: 0, aRS: 1, aCS: r }, { bOff: 0, bRS: Kin, bCS: 1 }, 1);
    if (DX) this.op("mm", [this.DT, this.PRM, DX], { M: R, N: Kin, K: r, kChunk: r, aRS: r, aCS: 1, bOff: this.poff[pfx + ".A"], bRS: Kin, bCS: 1, cRS: Kin, cCS: 1, acc: 1, alpha: 1 },
      [Math.ceil(Kin / 64), Math.ceil(R / 64), 1]);
  }

  // forward of layer i from XS[i]; writes XS[i+1] unless recompute
  #layer(i, recompute = false) {
    const { H, I } = this.C, { R, S } = this.b, Ly = this.man.layers[i], X = this.XS[i], eps = this.man.eps;
    let h = X;
    if (Ly.attn_norm) { this.op("ln", [X, this.WB, this.Hn, this.ST1], { R, eps, wOff: this.woff[`L${i}.attn_norm`] }, R); h = this.Hn; }
    this.#lin(h, `L${i}.Wqkv`, 3 * H, H, this.QKV, R);
    this.#loraF(h, `L${i}.Wqkv`, 3 * H, H, this.T1, this.QKV, R);
    this.op("rope", [this.QKV, this.CS, this.QR], { R, S, sinOff: SMAX * (this.C.D / 2) }, R);
    const att = { R, S, isLocal: Ly.global ? 0 : 1, win: this.man.window >> 1 };
    this.op("attn", [this.QR, this.QKV, this.LENS, this.O, this.LSE], att, R * this.C.NH);
    this.op("copy", [X, this.XM], { n: R * H }, Math.ceil((R * H) / 256));
    this.#lin(this.O, `L${i}.Wo`, H, H, this.XM, R, 1);
    this.#loraF(this.O, `L${i}.Wo`, H, H, this.T2, this.XM, R);
    this.op("ln", [this.XM, this.WB, this.H2, this.ST2], { R, eps, wOff: this.woff[`L${i}.mlp_norm`] }, R);
    this.#lin(this.H2, `L${i}.Wi`, 2 * I, H, this.U, R);
    this.#loraF(this.H2, `L${i}.Wi`, 2 * I, H, this.T3, this.U, R);
    this.op("geglu", [this.U, this.G], { n: R * I }, Math.ceil((R * I) / 256));
    if (recompute) return { h, att };
    this.op("copy", [this.XM, this.XS[i + 1]], { n: R * H }, Math.ceil((R * H) / 256));
    this.#lin(this.G, `L${i}.Wo_mlp`, H, I, this.XS[i + 1], R, 1);
    return { h, att };
  }

  #headForward() {
    const { H } = this.C, { nP } = this.b, eps = this.man.eps, L = this.C.L;
    this.op("gather", [this.POS, this.XS[L], this.HP], { R: nP }, nP);
    this.op("ln", [this.HP, this.WB, this.HF, this.STF], { R: nP, eps, wOff: this.woff.final_norm }, nP);
    this.op("mm", [this.HF, this.PRM, this.Z1], { M: nP, N: H, K: H, kChunk: H, aRS: H, aCS: 1, bOff: this.poff["head.w0"], bRS: 1, bCS: H, cRS: H, cCS: 1, alpha: 1 },
      [Math.ceil(H / 64), Math.ceil(nP / 64), 1]);
    this.op("bias_gelu", [this.PRM, this.Z1, this.A1], { R: nP, bOff: this.poff["head.b0"] }, nP);
    this.op("ln", [this.A1, this.PRM, this.N1, this.STH], { R: nP, eps: 1e-5, wOff: this.poff["head.ln_w"], bOff: this.poff["head.ln_b"], hasB: 1 }, nP);
    this.op("head_out", [this.N1, this.PRM, this.LOGIT], { R: nP, wOff: this.poff["head.w1"], bOff: this.poff["head.b1"] }, nP);
  }

  forward() {
    const { R } = this.b;
    this.op("embed", [this.IDS, this.CODES, this.SCALES, this.WB, this.XS[0]], { R, eps: this.man.eps, wOff: this.woff.emb_norm }, R);
    for (let i = 0; i < this.C.L; i++) this.#layer(i);
    this.#headForward();
  }

  // forward, loss (written to LOSS[slot]) and every gradient into GRD
  forwardBackward({ T, slot = 0 }) {
    const { H, I, L, NH } = this.C, { R, S, nP, nSeg } = this.b;
    this.forward();
    this.op("loss", [this.LOGIT, this.SEG, this.GOLD, this.DL, this.LOSS], { nSeg, T, slot }, 1);
    // head backward
    this.op("head_out_bwd", [this.DL, this.N1, this.PRM, this.DN1, this.GRD], { R: nP, wOff: this.poff["head.w1"], gwOff: this.poff["head.w1"], gbOff: this.poff["head.b1"] }, 1);
    this.op("ln_bwd", [this.A1, this.STH, this.PRM, this.DN1, this.DA1], { R: nP, wOff: this.poff["head.ln_w"] }, nP);
    this.op("ln_pgrad", [this.A1, this.STH, this.DN1, this.GRD], { R: nP, gwOff: this.poff["head.ln_w"], gbOff: this.poff["head.ln_b"] }, 1);
    this.op("gelu_bwd", [this.Z1, this.DA1, this.DZ1], { n: nP * H }, Math.ceil((nP * H) / 256));
    this.op("mm", [this.DZ1, this.HF, this.GRD], { M: H, N: H, K: nP, kChunk: nP, aRS: 1, aCS: H, bRS: H, bCS: 1, cOff: this.poff["head.w0"], cRS: H, cCS: 1, alpha: 1 },
      [Math.ceil(H / 64), Math.ceil(H / 64), 1]);
    this.op("colsum", [this.DZ1, this.GRD], { R: nP, gOff: this.poff["head.b0"] }, 1);
    this.op("mm", [this.DZ1, this.PRM, this.DHF], { M: nP, N: H, K: H, kChunk: H, aRS: H, aCS: 1, bOff: this.poff["head.w0"], bRS: H, bCS: 1, cRS: H, cCS: 1, alpha: 1 },
      [Math.ceil(H / 64), Math.ceil(nP / 64), 1]);
    this.op("ln_bwd", [this.HP, this.STF, this.WB, this.DHF, this.DHP], { R: nP, wOff: this.woff.final_norm }, nP);
    let dxo = this.DXA, dxi = this.DXB;
    this.clear(dxo);
    this.op("scatter", [this.POS, this.DHP, dxo], { R: nP }, nP);
    // layers, top down
    for (let i = L - 1; i >= 0; i--) {
      const Ly = this.man.layers[i], X = this.XS[i], pf = `L${i}.`;
      const { h, att } = this.#layer(i, true);
      // MLP
      this.#linT(dxo, pf + "Wo_mlp", H, I, this.DG, R);
      this.op("geglu_bwd", [this.U, this.DG, this.DU], { n: R * I }, Math.ceil((R * I) / 256));
      this.#linT(this.DU, pf + "Wi", 2 * I, H, this.DH2, R);
      this.#loraB(this.DU, this.H2, this.T3, pf + "Wi", 2 * I, H, R, this.DH2);
      this.op("copy", [dxo, this.DXM], { n: R * H }, Math.ceil((R * H) / 256));
      this.op("ln_bwd", [this.XM, this.ST2, this.WB, this.DH2, this.DXM], { R, wOff: this.woff[pf + "mlp_norm"], acc: 1 }, R);
      // attention output projection
      this.#linT(this.DXM, pf + "Wo", H, H, this.DO, R);
      this.#loraB(this.DXM, this.O, this.T2, pf + "Wo", H, H, R, this.DO);
      this.op("attn_bq", [this.QR, this.QKV, this.LENS, this.LSE, this.DO, this.O, this.DQR, this.DD], att, R * NH);
      this.op("attn_bkv", [this.QR, this.QKV, this.LENS, this.LSE, this.DO, this.DD, this.DQR, this.DQKV], att, R * NH);
      this.op("rope_bwd", [this.DQR, this.CS, this.DQKV], { R, S, sinOff: SMAX * (this.C.D / 2) }, R);
      const needDX = i > 0;
      if (needDX) this.#linT(this.DQKV, pf + "Wqkv", 3 * H, H, this.DH, R);
      this.#loraB(this.DQKV, h, this.T1, pf + "Wqkv", 3 * H, H, R, needDX ? this.DH : null);
      if (needDX) {
        this.op("copy", [this.DXM, dxi], { n: R * H }, Math.ceil((R * H) / 256));
        if (Ly.attn_norm) this.op("ln_bwd", [X, this.ST1, this.WB, this.DH, dxi], { R, wOff: this.woff[pf + "attn_norm"], acc: 1 }, R);
        else this.op("copy", [this.DH, dxi], { n: R * H, acc: 1 }, Math.ceil((R * H) / 256));
        [dxo, dxi] = [dxi, dxo];
      }
    }
  }

  // AdamW with global-norm clipping, PyTorch's update order. step counts from 1.
  adamStep({ lr, lrHead = lr, wd = 0.01, step, clip = 1.0, b1 = 0.9, b2 = 0.999, eps = 1e-8 }) {
    const n = this.nParams;
    this.op("sqsum", [this.GRD, this.PARTN], { n }, this.NPART);
    this.op("norm", [this.PARTN, this.NRM], { n: this.NPART }, 1);
    this.op("adamw", [this.PRM, this.GRD, this.AM, this.AV, this.NRM], { n, headOff: this.headOff, lr, lrHead, wd, b1, b2, eps,
      bc1: 1 - b1 ** step, bc2s: Math.sqrt(1 - b2 ** step), clip }, Math.ceil(n / 256));
  }
}

function grid(n) {
  if (n <= 65535) return [n, 1, 1];
  return [65535, Math.ceil(n / 65535), 1];
}
