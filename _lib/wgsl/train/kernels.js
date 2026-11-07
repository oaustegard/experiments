// Forward and backward kernels for ModernBERT-style training in WebGPU (kernel library step 3): the
// LoRA trainer's set (ettin-wgsl-train) and, in fullKernels, the full-weight additions (lifesci-ettin's
// distillation trainer) and the in-batch contrastive loss (nce.js). Each is a spec in spec.js's format,
// specialized on C = { H, NH, D, I[, OUT] } at compile time. The trainers own buffers and recording;
// see their headers for the step each kernel belongs to.

import { GELU, REDUCE, SMAX } from "./spec.js";
import { nceKernels } from "./nce.js";

// Kernel specs: u = uniform names (suffix ":f" for f32), b = [name, "r" | "rw", type], wg, src (body + helpers).
export function kernels(C) {
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

// Full-weight training: every linear's weight gradient, LayerNorm weight gradients, the token-table
// gradient, mean pooling, the cosine loss, plus the contrastive loss.
export function fullKernels(C) {
  const { H, OUT } = C;
  const K = kernels(C);
  K.embed_f = { u: ["R", "eps:f", "tOff", "wOff"], b: [["IDS", "r", "u32"], ["PRM", "r", "f32"], ["XE", "rw", "f32"], ["X", "rw", "f32"], ["ST", "rw", "f32"]],
    wg: H, pre: REDUCE(H), body: `
  if (gi >= p.R) { return; }
  let j = lid.x;
  let v = PRM[p.tOff + IDS[gi] * ${H}u + j];
  XE[gi * ${H}u + j] = v;
  let mu = wsum(v, j) / ${H}.0;
  let d = v - mu;
  let rs = inverseSqrt(wsum(d * d, j) / ${H}.0 + p.eps);
  X[gi * ${H}u + j] = d * rs * PRM[p.wOff + j];
  if (j == 0u) { ST[2u * gi] = mu; ST[2u * gi + 1u] = rs; }` };
  // LayerNorm weight gradient, partial sums over chunks of rows: PART[z, c] = sum_r DY[r, c] * xhat[r, c]
  K.ln_wpart = { u: ["R", "chunk"], b: [["X", "r", "f32"], ["ST", "r", "f32"], ["DY", "r", "f32"], ["PART", "rw", "f32"]], wg: H, body: `
  let c = lid.x;
  let r0 = gi * p.chunk;
  let r1 = min(p.R, r0 + p.chunk);
  var s = 0.0;
  for (var r = r0; r < r1; r = r + 1u) { s = s + DY[r * ${H}u + c] * (X[r * ${H}u + c] - ST[2u * r]) * ST[2u * r + 1u]; }
  PART[gi * ${H}u + c] = s;` };
  K.pool = { u: ["B", "S"], b: [["X", "r", "f32"], ["LENS", "r", "u32"], ["PL", "rw", "f32"]], wg: H, body: `
  if (gi >= p.B) { return; }
  let n = LENS[gi];
  var s = 0.0;
  for (var t = 0u; t < n; t = t + 1u) { s = s + X[(gi * p.S + t) * ${H}u + lid.x]; }
  PL[gi * ${H}u + lid.x] = s / f32(n);` };
  K.unpool = { u: ["R", "S"], b: [["DPL", "r", "f32"], ["LENS", "r", "u32"], ["DX", "rw", "f32"]], wg: H, body: `
  if (gi >= p.R) { return; }
  let b = gi / p.S;
  let t = gi % p.S;
  let n = LENS[b];
  var v = 0.0;
  if (t < n) { v = DPL[b * ${H}u + lid.x] / f32(n); }
  DX[gi * ${H}u + lid.x] = v;` };
  // loss = mean_b (1 - cos(y_b, t_b)); one workgroup walks the batch
  K.cosloss = { u: ["B", "slot"], b: [["Y", "r", "f32"], ["TGT", "r", "f32"], ["DY", "rw", "f32"], ["LOSS", "rw", "f32"]], wg: 256, pre: REDUCE(256), body: `
  let t = lid.x;
  var tot = 0.0;
  for (var b = 0u; b < p.B; b = b + 1u) {
    var yt = 0.0;
    var yy = 0.0;
    var tt = 0.0;
    for (var j = t; j < ${OUT}u; j = j + 256u) { let y = Y[b * ${OUT}u + j]; let g = TGT[b * ${OUT}u + j]; yt = yt + y * g; yy = yy + y * y; tt = tt + g * g; }
    yt = wsum(yt, t);
    yy = wsum(yy, t);
    tt = wsum(tt, t);
    let ny = sqrt(yy);
    let nt = sqrt(tt);
    let cs = yt / (ny * nt);
    tot = tot + 1.0 - cs;
    let k = -1.0 / f32(p.B);
    for (var j = t; j < ${OUT}u; j = j + 256u) {
      let y = Y[b * ${OUT}u + j];
      DY[b * ${OUT}u + j] = k * (TGT[b * ${OUT}u + j] / (ny * nt) - cs * y / yy);
    }
  }
  if (t == 0u) { LOSS[p.slot] = tot / f32(p.B); }` };
  // token-table gradient: one workgroup per unique token u, summing the rows that hold it
  K.tgrad = { u: ["nU", "tOff"], b: [["UTOK", "r", "u32"], ["USTART", "r", "u32"], ["UROWS", "r", "u32"], ["DXE", "r", "f32"], ["G", "rw", "f32"]], wg: H, body: `
  if (gi >= p.nU) { return; }
  var s = 0.0;
  for (var k = USTART[gi]; k < USTART[gi + 1u]; k = k + 1u) { s = s + DXE[UROWS[k] * ${H}u + lid.x]; }
  G[p.tOff + UTOK[gi] * ${H}u + lid.x] = s;` };
  Object.assign(K, nceKernels(C));
  return K;
}
