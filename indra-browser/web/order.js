// The reading-order transformer on the GPU: OrderNet.run (nets.js) as one pass of dispatches.
// Line features in, the pairwise order and same-block scores and the per-line class logits out.
// Matmuls use gpu.js's tiled MM (act 4 is GELU); layer norm, attention and the pair heads are
// kernels here. Only the outputs are read back: a and same [N, N] and the heads [N, 522].

import { MM, registerKernel } from "./gpu.js";
import { orderInput } from "./nets.js";

// one thread per row of D = 128: (x - mean) / sqrt(var + 1e-5) * w + b
const LN = /* wgsl */ `
struct P { N: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  if (n >= p.N) { return; }
  let o = n * 128u;
  var m = 0.0;
  for (var d = 0u; d < 128u; d++) { m += x[o + d]; }
  m = m / 128.0;
  var v = 0.0;
  for (var d = 0u; d < 128u; d++) { let e = x[o + d] - m; v += e * e; }
  let r = 1.0 / sqrt(v / 128.0 + 1e-5);
  for (var d = 0u; d < 128u; d++) { y[o + d] = (x[o + d] - m) * r * w[d] + b[d]; }
}`;
registerKernel(LN, "order.ln");

// self-attention, 4 heads of 32: one thread per (line, head) over all N lines.
// qkv [N, 384] (q | k | v) -> o [N, 128]
const ATTN = /* wgsl */ `
struct P { N: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.N * 4u) { return; }
  let a = idx / 4u; let h = idx % 4u;
  let qo = a * 384u + h * 32u;
  let sc = 1.0 / sqrt(32.0);
  var q: array<f32, 32>;
  for (var d = 0u; d < 32u; d++) { q[d] = qkv[qo + d]; }
  var mx = -3.4e38;
  for (var b = 0u; b < p.N; b++) {
    let ko = b * 384u + 128u + h * 32u;
    var s = 0.0;
    for (var d = 0u; d < 32u; d++) { s += q[d] * qkv[ko + d]; }
    mx = max(mx, s * sc);
  }
  var acc: array<f32, 32>;
  for (var d = 0u; d < 32u; d++) { acc[d] = 0.0; }
  var den = 0.0;
  for (var b = 0u; b < p.N; b++) {
    let ko = b * 384u + 128u + h * 32u;
    var s = 0.0;
    for (var d = 0u; d < 32u; d++) { s += q[d] * qkv[ko + d]; }
    let e = exp(s * sc - mx);
    den += e;
    let vo = b * 384u + 256u + h * 32u;
    for (var d = 0u; d < 32u; d++) { acc[d] += e * qkv[vo + d]; }
  }
  for (var d = 0u; d < 32u; d++) { o[a * 128u + h * 32u + d] = acc[d] / den; }
}`;
registerKernel(ATTN, "order.attn");

// the pair heads from hd [N, 522] = q | k | bu | bw | cls:
// a[i, j] = q_i.k_j / r - q_j.k_i / r, same[i, j] = bu_i.bw_j / r + bu_j.bw_i / r, r = sqrt(128)
const PAIR = /* wgsl */ `
struct P { N: u32, nx: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> hd: array<f32>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;
@group(0) @binding(3) var<storage, read_write> same: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.N * p.N) { return; }
  let i = idx / p.N; let j = idx % p.N;
  let oi = i * 522u; let oj = j * 522u;
  var sij = 0.0; var sji = 0.0; var uij = 0.0; var uji = 0.0;
  for (var d = 0u; d < 128u; d++) {
    sij += hd[oi + d] * hd[oj + 128u + d];
    sji += hd[oj + d] * hd[oi + 128u + d];
    uij += hd[oi + 256u + d] * hd[oj + 384u + d];
    uji += hd[oj + 256u + d] * hd[oi + 384u + d];
  }
  let r = sqrt(128.0);
  a[idx] = sij / r - sji / r;
  same[idx] = uij / r + uji / r;
}`;
registerKernel(PAIR, "order.pair");

// W [K, M] (x @ W) -> [M, K], the layout MM reads
function transpose(W, K, M) {
  const out = new Float32Array(M * K);
  for (let k = 0; k < K; k++) for (let m = 0; m < M; m++) out[m * K + k] = W[k * M + m];
  return out;
}

const HEADS = [["q", 128], ["k", 128], ["bu", 128], ["bw", 128], ["cls", 10]];

export class GpuOrderNet {
  constructor(gpu, model) {
    this.gpu = gpu;
    const W = model.tensors;
    const lin = (w, b, K, M) => ({ w: gpu.upload(transpose(W[w], K, M)), b: gpu.upload(W[b]), K, M });
    const vec = (n) => gpu.upload(W[n]);
    this.inp0 = lin("inp.inp.0.MatMul.weight", "inp.0.bias", 144, 128);
    this.inp2 = lin("inp.inp.2.MatMul.weight", "inp.2.bias", 128, 128);
    this.layers = Array.from({ length: 4 }, (_, i) => {
      const L = `layers.${i}`;
      return {
        n1: [vec(`${L}.n1.weight`), vec(`${L}.n1.bias`)],
        qkv: lin(`${L}.qkv.MatMul.weight`, `${L}.qkv.bias`, 128, 384),
        proj: lin(`${L}.proj.MatMul.weight`, `${L}.proj.bias`, 128, 128),
        n2: [vec(`${L}.n2.weight`), vec(`${L}.n2.bias`)],
        ff0: lin(`${L}.ff.ff.0.MatMul.weight`, `${L}.ff.0.bias`, 128, 256),
        ff2: lin(`${L}.ff.ff.2.MatMul.weight`, `${L}.ff.2.bias`, 256, 128),
      };
    });
    this.norm = [vec("norm.weight"), vec("norm.bias")];
    // the five heads as one [128 -> 522] matmul
    const M = HEADS.reduce((s, [, m]) => s + m, 0);
    const w = new Float32Array(M * 128), b = new Float32Array(M);
    let off = 0;
    for (const [n, m] of HEADS) {
      w.set(transpose(W[`${n}.MatMul.weight`], 128, m), off * 128);
      b.set(W[`${n}.bias`], off);
      off += m;
    }
    this.heads = { w: gpu.upload(w), b: gpu.upload(b), K: 128, M };
  }

  mm(x, N, l, act = 0, res = null) {
    const gpu = this.gpu;
    const y = gpu.buffer(4 * N * l.M);
    gpu.dispatch(MM, [x, l.w, l.b, res, null, y], [N, l.K, l.M, act, res ? 1 : 0, 0, l.M, 0], [Math.ceil(l.M / 64), Math.ceil(N / 64)], "order.mm");
    return y;
  }

  ln(x, N, [w, b]) {
    const y = this.gpu.buffer(4 * N * 128);
    this.gpu.dispatch(LN, [x, w, b, y], [N, 0, 0, 0], [Math.ceil(N / 64)]);
    return y;
  }

  // feats [N, 16] -> {a, cls, same} as OrderNet.run gives them (no table heads)
  async run(feats, N) {
    const gpu = this.gpu;
    gpu.scope = "order";
    const inBuf = gpu.upload(orderInput(feats, N));
    gpu.begin();
    let x = this.mm(this.mm(inBuf, N, this.inp0, 4), N, this.inp2);
    for (const L of this.layers) {
      const qkv = this.mm(this.ln(x, N, L.n1), N, L.qkv);
      const o = gpu.buffer(4 * N * 128);
      gpu.dispatch(ATTN, [qkv, o], [N, 0, 0, 0], [Math.ceil((N * 4) / 64)]);
      x = this.mm(o, N, L.proj, 0, x);
      x = this.mm(this.mm(this.ln(x, N, L.n2), N, L.ff0, 4), N, L.ff2, 0, x);
    }
    const hd = this.mm(this.ln(x, N, this.norm), N, this.heads);
    const a = gpu.buffer(4 * N * N), same = gpu.buffer(4 * N * N);
    const grid = gpu.grid(N * N);
    gpu.dispatch(PAIR, [hd, a, same], [N, grid[0], 0, 0], grid);
    const [ra, rs, rh] = await gpu.end([[a, 4 * N * N], [same, 4 * N * N], [hd, 4 * N * 522]]);
    inBuf.destroy();
    const h = new Float32Array(rh, 0, N * 522);
    const cls = new Float32Array(N * 10);
    for (let n = 0; n < N; n++) cls.set(h.subarray(n * 522 + 512, n * 522 + 522), n * 10);
    return { a: new Float32Array(ra, 0, N * N), cls, same: new Float32Array(rs, 0, N * N) };
  }
}
