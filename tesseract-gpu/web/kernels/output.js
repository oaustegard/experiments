// Output layer of the tesseract-gpu engine: fully connected softmax over the LSTM stack's
// output, plus the per-timestep argmax that the CTC decoder reads.
//
//   recordOutput(gpu, { x, w, s, out, best, bestp, T, ni, C, intMode })
//
//   x      f32 [T, ni]     x3, the last LSTM's output (int mode: integer codes)
//   w      f32 [C, ni+1]   weights, bias last column (int mode: int8 codes stored as f32)
//   s      f32 [C]         per-row scale, int mode only (null in float mode)
//   out    f32 [T, C]      probabilities; the logits are staged here during the pass
//   best   u32 [T]         argmax code per timestep, first index on ties (numpy order)
//   bestp  f32 [T]         probability of that code
//
// Float:  v = sum w*x + w_bias
// Int:    v = (sum q*x + q_bias*127) * s
// probs = exp(clamp(v - max(v), -86, 0)) / sum(...), as in tessgpu/ref.py and Tesseract's
// SoftmaxInPlace.
//
// One workgroup per timestep, 128 threads striding over the classes. Three passes, each
// closed by a tree reduction: (1) logits and their max, (2) exp and its sum, (3) normalise
// and argmax over the normalised values. The grid folds timesteps past 65535 into y.
// Only records dispatches; the caller submits.

const WGSL = `
struct Par { ni: u32, nc: u32, intMode: u32, pad: u32 }
@group(0) @binding(0) var<uniform> par: Par;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> s: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> best: array<u32>;
@group(0) @binding(6) var<storage, read_write> bestp: array<f32>;

const WG: u32 = 128u;
var<workgroup> rmax: array<f32, 128>;
var<workgroup> rsum: array<f32, 128>;
var<workgroup> rv: array<f32, 128>;
var<workgroup> ri: array<u32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x + wid.y * 65535u;
  let tid = lid.x;
  let ni = par.ni;
  let nc = par.nc;
  let xb = t * ni;
  let ob = t * nc;

  // Pass 1: logits into out, running max per thread.
  var mx: f32 = -3.4e38;
  for (var c = tid; c < nc; c = c + WG) {
    let wb = c * (ni + 1u);
    var acc: f32 = 0.0;
    for (var k = 0u; k < ni; k = k + 1u) {
      acc = acc + w[wb + k] * x[xb + k];
    }
    var v: f32;
    if (par.intMode == 1u) {
      v = (acc + w[wb + ni] * 127.0) * s[c];
    } else {
      v = acc + w[wb + ni];
    }
    out[ob + c] = v;
    mx = max(mx, v);
  }
  rmax[tid] = mx;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st / 2u) {
    if (tid < st) { rmax[tid] = max(rmax[tid], rmax[tid + st]); }
    workgroupBarrier();
  }
  let m = rmax[0];

  // Pass 2: exp(clamp(v - max, -86, 0)) in place, and the sum.
  var sum: f32 = 0.0;
  for (var c = tid; c < nc; c = c + WG) {
    let e = exp(clamp(out[ob + c] - m, -86.0, 0.0));
    out[ob + c] = e;
    sum = sum + e;
  }
  rsum[tid] = sum;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st / 2u) {
    if (tid < st) { rsum[tid] = rsum[tid] + rsum[tid + st]; }
    workgroupBarrier();
  }
  let total = rsum[0];

  // Pass 3: normalise, then argmax over the normalised probabilities.
  var bv: f32 = -1.0;
  var bi: u32 = 0u;
  for (var c = tid; c < nc; c = c + WG) {
    let p = out[ob + c] / total;
    out[ob + c] = p;
    if (p > bv) { bv = p; bi = c; }
  }
  rv[tid] = bv;
  ri[tid] = bi;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st / 2u) {
    if (tid < st) {
      let v2 = rv[tid + st];
      let i2 = ri[tid + st];
      if (v2 > rv[tid] || (v2 == rv[tid] && i2 < ri[tid])) {
        rv[tid] = v2;
        ri[tid] = i2;
      }
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    best[t] = ri[0];
    bestp[t] = rv[0];
  }
}
`;

export function recordOutput(gpu, { x, w, s, out, best, bestp, T, ni, C, intMode }) {
  if (!(T >= 1 && ni >= 1 && C >= 1)) throw new Error(`recordOutput: bad shape T=${T} ni=${ni} C=${C}`);
  const par = gpu.uniform(new Uint32Array([ni, C, intMode ? 1 : 0, 0]));
  // Float mode never reads s; the binding still needs a valid buffer.
  const sb = s ?? gpu.buffer(16);
  gpu.dispatch(WGSL, [par, x, w, sb, out, best, bestp],
    [Math.min(T, 65535), Math.ceil(T / 65535), 1]);
}
