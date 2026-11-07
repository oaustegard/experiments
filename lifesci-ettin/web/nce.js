// In-batch-negatives contrastive loss (InfoNCE / MultipleNegativesRankingLoss) for the WGSL trainer.
//
// The batch holds P queries in rows 0..P-1 and their P positive passages in rows P..2P-1. With
// yn = y / |y| and S = scale * Qn Dn^T, loss = mean_i -log softmax(S_i)_i (query -> passage only, as
// sentence-transformers' MNRL). Backward: dS = (softmax(S) - I) / P, dQn = scale dS Dn,
// dDn = scale dS^T Qn (both through the trainer's tiled `mm`), then dy = (dyn - yn (yn . dyn)) / |y|.
//
// Kernels are in the trainer's spec format (named uniforms, [name, mode, type] bindings, wg, pre,
// body), self-contained and free of model state, so they can move to _lib/wgsl/train/ with the rest of
// the training ops (kernel library step 3). Each has a float64 case through run/reference.py nce.

const RED = (n, name, op, init) => /* wgsl */ `
var<workgroup> ${name}_a: array<f32, ${n}>;
fn ${name}(v: f32, t: u32) -> f32 {
  ${name}_a[t] = v;
  workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) {
    if (t < s) { ${name}_a[t] = ${op(`${name}_a[t]`, `${name}_a[t + s]`)}; }
    workgroupBarrier();
  }
  let r = ${name}_a[0];
  workgroupBarrier();
  return r;
}`;

export function nceKernels({ OUT }) {
  const K = {};
  // per row: NR[b] = |Y_b|, YN_b = Y_b / |Y_b|
  K.l2n = { u: ["B"], b: [["Y", "r", "f32"], ["YN", "rw", "f32"], ["NR", "rw", "f32"]], wg: 256, pre: RED(256, "rsum", (a, b) => `${a} + ${b}`), body: `
  if (gi >= p.B) { return; }
  let t = lid.x;
  var s = 0.0;
  for (var j = t; j < ${OUT}u; j = j + 256u) { let y = Y[gi * ${OUT}u + j]; s = s + y * y; }
  let n = sqrt(rsum(s, t));
  for (var j = t; j < ${OUT}u; j = j + 256u) { YN[gi * ${OUT}u + j] = Y[gi * ${OUT}u + j] / n; }
  if (t == 0u) { NR[gi] = n; }` };
  // one workgroup per query row i of SIM [P, P]: DS[i, :] = (softmax - onehot(i)) / P, LROW[i] = -log softmax_ii
  K.nce = { u: ["P"], b: [["SIM", "r", "f32"], ["DS", "rw", "f32"], ["LROW", "rw", "f32"]], wg: 64,
    pre: RED(64, "rmax", (a, b) => `max(${a}, ${b})`) + RED(64, "rsum", (a, b) => `${a} + ${b}`), body: `
  if (gi >= p.P) { return; }
  let t = lid.x;
  let o = gi * p.P;
  var m = -3.0e38;
  for (var j = t; j < p.P; j = j + 64u) { m = max(m, SIM[o + j]); }
  m = rmax(m, t);
  var z = 0.0;
  for (var j = t; j < p.P; j = j + 64u) { z = z + exp(SIM[o + j] - m); }
  z = rsum(z, t);
  let inv = 1.0 / f32(p.P);
  for (var j = t; j < p.P; j = j + 64u) {
    var g = exp(SIM[o + j] - m) / z;
    if (j == gi) { g = g - 1.0; }
    DS[o + j] = g * inv;
  }
  if (t == 0u) { LROW[gi] = log(z) + m - SIM[o + gi]; }` };
  // LOSS[slot] = mean_i LROW[i]
  K.nce_sum = { u: ["P", "slot"], b: [["LROW", "r", "f32"], ["LOSS", "rw", "f32"]], wg: 64, pre: RED(64, "rsum", (a, b) => `${a} + ${b}`), body: `
  let t = lid.x;
  var s = 0.0;
  for (var j = t; j < p.P; j = j + 64u) { s = s + LROW[j]; }
  s = rsum(s, t);
  if (t == 0u) { LOSS[p.slot] = s / f32(p.P); }` };
  // per row: DY_b = (DYN_b - YN_b (YN_b . DYN_b)) / NR[b]
  K.l2n_bwd = { u: ["B"], b: [["YN", "r", "f32"], ["NR", "r", "f32"], ["DYN", "r", "f32"], ["DY", "rw", "f32"]], wg: 256, pre: RED(256, "rsum", (a, b) => `${a} + ${b}`), body: `
  if (gi >= p.B) { return; }
  let t = lid.x;
  var s = 0.0;
  for (var j = t; j < ${OUT}u; j = j + 256u) { s = s + YN[gi * ${OUT}u + j] * DYN[gi * ${OUT}u + j]; }
  s = rsum(s, t);
  let n = NR[gi];
  for (var j = t; j < ${OUT}u; j = j + 256u) { let k = gi * ${OUT}u + j; DY[k] = (DYN[k] - YN[k] * s) / n; }` };
  return K;
}

// Records the loss and dY for a trainer `tr` whose forward pass has written Y [2P, OUT]. Needs the
// buffers YN, NR, SIM, DS, LROW, DYN, DY, LOSS on `tr` and its `mm` kernel.
export function recordNce(tr, { P, scale, slot }) {
  const { OUT } = tr.C, B = 2 * P;
  tr.op("l2n", [tr.Y, tr.YN, tr.NR], { B }, B);
  // SIM = scale * Qn Dn^T: A(m,k) = YN[m*OUT + k], B(k,n) = YN[(P+n)*OUT + k]
  tr.op("mm", [tr.YN, tr.YN, tr.SIM], { M: P, N: P, K: OUT, kChunk: OUT, aOff: 0, aRS: OUT, aCS: 1, bOff: P * OUT, bRS: 1, bCS: OUT,
    cOff: 0, cRS: P, cCS: 1, alpha: scale }, [Math.ceil(P / 64), Math.ceil(P / 64), 1]);
  tr.op("nce", [tr.SIM, tr.DS, tr.LROW], { P }, P);
  tr.op("nce_sum", [tr.LROW, tr.LOSS], { P, slot }, 1);
  // dQn = scale * DS Dn: A(m,k) = DS[m*P + k], B(k,n) = YN[(P+k)*OUT + n] -> DYN rows 0..P-1
  tr.op("mm", [tr.DS, tr.YN, tr.DYN], { M: P, N: OUT, K: P, kChunk: P, aOff: 0, aRS: P, aCS: 1, bOff: P * OUT, bRS: OUT, bCS: 1,
    cOff: 0, cRS: OUT, cCS: 1, alpha: scale }, [Math.ceil(OUT / 64), Math.ceil(P / 64), 1]);
  // dDn = scale * DS^T Qn: A(m,k) = DS[k*P + m], B(k,n) = YN[k*OUT + n] -> DYN rows P..2P-1
  tr.op("mm", [tr.DS, tr.YN, tr.DYN], { M: P, N: OUT, K: P, kChunk: P, aOff: 0, aRS: 1, aCS: P, bOff: 0, bRS: OUT, bCS: 1,
    cOff: P * OUT, cRS: OUT, cCS: 1, alpha: scale }, [Math.ceil(OUT / 64), Math.ceil(P / 64), 1]);
  tr.op("l2n_bwd", [tr.YN, tr.NR, tr.DYN, tr.DY], { B }, B);
}
