// Output layer of the tesseract-gpu engine, tiled: fully connected softmax over the LSTM
// stack's output, plus the per-timestep argmax that the CTC decoder reads.
//
//   recordOutput(gpu, { x, w, s, out, best, bestp, T, ni, C, intMode })
//
// Same contract and outputs as output.js (see there for the layouts and formulas):
//   x [T, ni], w [C, ni+1] (bias last), s [C] (int mode only), out [T, C] probs,
//   best u32 [T], bestp f32 [T].
//
// The logits are a tiled matmul. A workgroup covers TT = 32 timesteps and walks the classes
// in chunks of CC = 32 and the inputs in tiles of KK = 32. Each tile of x and of w is staged
// once in workgroup memory, so each weight row is read once per workgroup (about T/32 times
// in total) instead of once per timestep. Thread (tx, ty) keeps the four logits for timestep
// tx and classes ty, ty+8, ty+16, ty+24 of the chunk in registers.
//
// Logits go to out (global memory) for the whole tile, then a second phase runs the softmax
// on the same tile: eight threads per timestep, a tree reduction over each group of eight for
// the max, the sum and the argmax. Float logit summation order differs from output.js (tile
// order instead of a per-class serial sum); int mode sums the same integer products and
// applies (sum + bias*127) * scale, exact in f32 as before.
// Only records dispatches; the caller submits.

const TILE = 32; // timesteps per workgroup

const WGSL = `
struct Par { ni: u32, nc: u32, intMode: u32, T: u32 }
@group(0) @binding(0) var<uniform> par: Par;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> best: array<u32>;
@group(0) @binding(6) var<storage, read_write> bestp: array<f32>;

const TT: u32 = 32u;   // timesteps per workgroup
const CC: u32 = 32u;   // classes per chunk
const KK: u32 = 32u;   // inputs per k tile
const WG: u32 = 256u;
const PAD: u32 = 33u;  // row stride in workgroup memory (odd, so column reads do not conflict)

var<workgroup> xs: array<f32, 1056>;  // [TT][PAD]
var<workgroup> ws: array<f32, 1056>;  // [CC][PAD]
var<workgroup> red: array<f32, 256>;
var<workgroup> redi: array<u32, 256>;

// Logit from the accumulated dot product: bias and, in int mode, the row scale.
fn finish_logit(acc: f32, c: u32) -> f32 {
  let wb = c * (par.ni + 1u);
  if (par.intMode == 1u) {
    return (acc + w[wb + par.ni] * 127.0) * s[c];
  }
  return acc + w[wb + par.ni];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tile = wid.x + wid.y * 65535u;
  let t0 = tile * TT;
  let T = par.T;
  if (t0 >= T) { return; }  // whole workgroup exits together
  let tid = lid.x;
  let ni = par.ni;
  let nc = par.nc;
  let tx = tid % TT;   // timestep within the tile (matmul phase)
  let ty = tid / TT;   // class lane 0..7 (matmul phase)
  let nk = (ni + KK - 1u) / KK;
  let ncc = (nc + CC - 1u) / CC;

  // Phase 1: logits for the tile, class chunk by class chunk.
  for (var cc: u32 = 0u; cc < ncc; cc = cc + 1u) {
    let c0 = cc * CC;
    var a0: f32 = 0.0;
    var a1: f32 = 0.0;
    var a2: f32 = 0.0;
    var a3: f32 = 0.0;
    for (var kt: u32 = 0u; kt < nk; kt = kt + 1u) {
      let k0 = kt * KK;
      for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        let i = tid + r * WG;
        let row = i / KK;
        let col = i % KK;
        let t = t0 + row;
        let k = k0 + col;
        var xv: f32 = 0.0;
        if (t < T && k < ni) { xv = x[t * ni + k]; }
        xs[row * PAD + col] = xv;
        let c = c0 + row;
        var wv: f32 = 0.0;
        if (c < nc && k < ni) { wv = w[c * (ni + 1u) + k]; }
        ws[row * PAD + col] = wv;
      }
      workgroupBarrier();
      for (var k: u32 = 0u; k < KK; k = k + 1u) {
        let xv = xs[tx * PAD + k];
        a0 = a0 + xv * ws[ty * PAD + k];
        a1 = a1 + xv * ws[(ty + 8u) * PAD + k];
        a2 = a2 + xv * ws[(ty + 16u) * PAD + k];
        a3 = a3 + xv * ws[(ty + 24u) * PAD + k];
      }
      workgroupBarrier();
    }
    let tv = t0 + tx;
    if (tv < T) {
      let c_0 = c0 + ty;
      let c_1 = c_0 + 8u;
      let c_2 = c_0 + 16u;
      let c_3 = c_0 + 24u;
      if (c_0 < nc) { out[tv * nc + c_0] = finish_logit(a0, c_0); }
      if (c_1 < nc) { out[tv * nc + c_1] = finish_logit(a1, c_1); }
      if (c_2 < nc) { out[tv * nc + c_2] = finish_logit(a2, c_2); }
      if (c_3 < nc) { out[tv * nc + c_3] = finish_logit(a3, c_3); }
    }
  }
  storageBarrier();
  workgroupBarrier();

  // Phase 2: softmax and argmax. Eight threads per timestep (r = row, ln = lane).
  let r = tid / 8u;
  let ln = tid % 8u;
  let tv = t0 + r;
  let valid = tv < T;

  var mx: f32 = -3.4e38;
  if (valid) {
    for (var c = ln; c < nc; c = c + 8u) { mx = max(mx, out[tv * nc + c]); }
  }
  red[tid] = mx;
  workgroupBarrier();
  for (var st: u32 = 4u; st > 0u; st = st / 2u) {
    if (ln < st) { red[tid] = max(red[tid], red[tid + st]); }
    workgroupBarrier();
  }
  let m = red[r * 8u];
  workgroupBarrier();

  var sum: f32 = 0.0;
  if (valid) {
    for (var c = ln; c < nc; c = c + 8u) {
      let e = exp(clamp(out[tv * nc + c] - m, -86.0, 0.0));
      out[tv * nc + c] = e;
      sum = sum + e;
    }
  }
  red[tid] = sum;
  workgroupBarrier();
  for (var st: u32 = 4u; st > 0u; st = st / 2u) {
    if (ln < st) { red[tid] = red[tid] + red[tid + st]; }
    workgroupBarrier();
  }
  let total = red[r * 8u];
  workgroupBarrier();

  var bv: f32 = -1.0;
  var bi: u32 = 0u;
  if (valid) {
    for (var c = ln; c < nc; c = c + 8u) {
      let p = out[tv * nc + c] / total;
      out[tv * nc + c] = p;
      if (p > bv) { bv = p; bi = c; }
    }
  }
  red[tid] = bv;
  redi[tid] = bi;
  workgroupBarrier();
  for (var st: u32 = 4u; st > 0u; st = st / 2u) {
    if (ln < st) {
      let v2 = red[tid + st];
      let i2 = redi[tid + st];
      if (v2 > red[tid] || (v2 == red[tid] && i2 < redi[tid])) {
        red[tid] = v2;
        redi[tid] = i2;
      }
    }
    workgroupBarrier();
  }
  if (valid && ln == 0u) {
    best[tv] = redi[tid];
    bestp[tv] = red[tid];
  }
}
`;

export function recordOutput(gpu, { x, w, s, out, best, bestp, T, ni, C, intMode }) {
  if (!(T >= 1 && ni >= 1 && C >= 1)) throw new Error(`recordOutput: bad shape T=${T} ni=${ni} C=${C}`);
  const nt = Math.ceil(T / TILE);
  const par = gpu.uniform(new Uint32Array([ni, C, intMode ? 1 : 0, T]));
  // Float mode never reads s; the binding still needs a valid buffer.
  const sb = s ?? gpu.buffer(16);
  gpu.dispatch(WGSL, [par, x, w, sb, out, best, bestp],
    [Math.min(nt, 65535), Math.ceil(nt / 65535), 1]);
}
