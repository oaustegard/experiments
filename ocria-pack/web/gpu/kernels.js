// WGSL kernels for the OCRia-nano op program (run/export_gpu.py writes it; engine.js runs it).
//
// Activations are float32 NCHW. Every kernel takes the same uniform block at binding 0:
//   struct P { a, b, c, d: vec4<u32>, e: vec4<u32>, f: vec4<f32> }
// a..d hold the kernel's own sizes and offsets (listed above each kernel); e and f hold the epilogue:
//   e = (act1, act2, has_res, fq_mask)   act: 0 none, 1 relu, 2 sigmoid; fq_mask bit 0 fq1, bit 1 fq2
//   f = (s1, z1, s2, z2)                 fake-quant scale and zero point of fq1 and fq2
// and applies, after the accumulator (bias included):
//   v = act1(v); v = fq1(v); v = v + res; v = act2(v); v = fq2(v)
// fq(v) = (clamp(round(v / s) + z, 0, 255) - z) * s is an activation QuantizeLinear -> DequantizeLinear
// pair (uint8, per tensor); WGSL round() is round-half-to-even, as onnxruntime's nearbyint.
// Weights live in one float32 buffer per model (engine.js dequantizes the int8 codes at load);
// offsets into it (wo, bo) are in floats.

const HEAD = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32>, e: vec4<u32>, f: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
fn act(v: f32, a: u32) -> f32 {
  if (a == 1u) { return max(v, 0.0); }
  if (a == 2u) { return 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }
  return v;
}
fn fq(v: f32, s: f32, z: f32) -> f32 {
  return (clamp(round(v / s) + z, 0.0, 255.0) - z) * s;
}
fn epi(acc: f32, res: f32) -> f32 {
  var v = act(acc, p.e.x);
  if ((p.e.w & 1u) != 0u) { v = fq(v, p.f.x, p.f.y); }
  if (p.e.z != 0u) { v = v + res; }
  v = act(v, p.e.y);
  if ((p.e.w & 2u) != 0u) { v = fq(v, p.f.z, p.f.w); }
  return v;
}
`;

// Direct convolution, any group count; one output pixel per invocation. Weights [O, icg, kh, kw].
// a = (icg, ih, iw, oc), b = (oh, ow, ocg, kh), c = (kw, sh, sw, pt), d = (pl, wo, bo, cin)
// grid (ceil(ow / 8), ceil(oh / 8), N * oc). Out-of-image taps read a clamped index times 0.
export const CONV = HEAD + /* wgsl */ `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let icg = p.a.x; let ih = p.a.y; let iw = p.a.z; let oc = p.a.w;
  let oh = p.b.x; let ow = p.b.y; let ocg = p.b.z; let kh = p.b.w;
  let kw = p.c.x; let sh = p.c.y; let sw = p.c.z; let pt = p.c.w;
  let pl = p.d.x; let wo = p.d.y; let bo = p.d.z; let cin = p.d.w;
  let ox = g.x; let oy = g.y; let z = g.z;
  if (ox >= ow || oy >= oh) { return; }
  let o = z % oc; let n = z / oc;
  let grp = o / ocg;
  var acc = w[bo + o];
  let y0 = i32(oy * sh) - i32(pt);
  let x0 = i32(ox * sw) - i32(pl);
  for (var ic: u32 = 0u; ic < icg; ic = ic + 1u) {
    let cb = (n * cin + grp * icg + ic) * ih * iw;
    let wb = wo + (o * icg + ic) * kh * kw;
    for (var ky: u32 = 0u; ky < kh; ky = ky + 1u) {
      let iy = y0 + i32(ky);
      let cy = u32(clamp(iy, 0, i32(ih) - 1));
      let rowok = select(0.0, 1.0, iy >= 0 && iy < i32(ih));
      for (var kx: u32 = 0u; kx < kw; kx = kx + 1u) {
        let ix = x0 + i32(kx);
        let cx = u32(clamp(ix, 0, i32(iw) - 1));
        let m = rowok * select(0.0, 1.0, ix >= 0 && ix < i32(iw));
        acc = acc + m * x[cb + cy * iw + cx] * w[wb + ky * kw + kx];
      }
    }
  }
  let idx = (z * oh + oy) * ow + ox;
  var rv = 0.0;
  if (p.e.z != 0u) { rv = r[idx]; }
  y[idx] = epi(acc, rv);
}
`;

// 1x1 convolution (and Gemm, with P = 1) as a tiled GEMM: Y[n, o, p] = sum_k Wt[k, o] X[n, k, p] + b[o].
// Weights transposed to [cin, cout] at load. A workgroup of 16 x 16 computes a 64 (o) x 64 (p) tile, each
// thread 4 x 4, over k in steps of 16 through workgroup memory.
// a = (cin, cout, P, wo), b = (bo, 0, 0, 0); grid (ceil(P / 64), ceil(cout / 64), N).
export const PW = HEAD + /* wgsl */ `
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> ws: array<vec4<f32>, 256>;
var<workgroup> xs: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let cin = p.a.x; let cout = p.a.y; let P = p.a.z; let wo = p.a.w; let bo = p.b.x;
  let n = wg.z; let p0 = wg.x * 64u; let o0 = wg.y * 64u;
  let tid = l.y * 16u + l.x;
  let lk = tid / 16u; let l4 = (tid % 16u) * 4u;
  let xb = n * cin * P;
  var acc: array<vec4<f32>, 4>;
  acc[0] = vec4<f32>(0.0); acc[1] = vec4<f32>(0.0); acc[2] = vec4<f32>(0.0); acc[3] = vec4<f32>(0.0);
  for (var k0: u32 = 0u; k0 < cin; k0 = k0 + 16u) {
    let k = k0 + lk;
    let kin = k < cin;
    let kc = min(k, cin - 1u);
    var xv = vec4<f32>(0.0);
    var wv = vec4<f32>(0.0);
    for (var c: u32 = 0u; c < 4u; c = c + 1u) {
      let pp = p0 + l4 + c;
      let oo = o0 + l4 + c;
      xv[c] = select(0.0, x[xb + kc * P + min(pp, P - 1u)], kin && pp < P);
      wv[c] = select(0.0, w[wo + kc * cout + min(oo, cout - 1u)], kin && oo < cout);
    }
    xs[tid] = xv;
    ws[tid] = wv;
    workgroupBarrier();
    for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
      let a = ws[kk * 16u + l.y];
      let b = xs[kk * 16u + l.x];
      acc[0] = acc[0] + a.x * b;
      acc[1] = acc[1] + a.y * b;
      acc[2] = acc[2] + a.z * b;
      acc[3] = acc[3] + a.w * b;
    }
    workgroupBarrier();
  }
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let o = o0 + l.y * 4u + i;
    if (o < cout) {
      let bias = w[bo + o];
      for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let pp = p0 + l.x * 4u + j;
        if (pp < P) {
          let idx = (n * cout + o) * P + pp;
          var rv = 0.0;
          if (p.e.z != 0u) { rv = r[idx]; }
          y[idx] = epi(acc[i][j] + bias, rv);
        }
      }
    }
  }
}
`;

// 1x1 convolution for few output channels (cout <= CO): one pixel per invocation, all outputs in
// registers. Same weight layout as PW. a = (cin, cout, P, wo), b = (bo, total = N * P, nx, 0);
// pixel index i = gid.x + gid.y * nx, 64-wide workgroups.
export function PWS(co) {
  return HEAD + /* wgsl */ `
const CO: u32 = ${co}u;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let cin = p.a.x; let cout = p.a.y; let P = p.a.z; let wo = p.a.w;
  let bo = p.b.x; let total = p.b.y;
  let i = g.x + g.y * p.b.z;
  if (i >= total) { return; }
  let n = i / P; let pp = i - n * P;
  var acc: array<f32, CO>;
  for (var o: u32 = 0u; o < CO; o = o + 1u) { acc[o] = 0.0; }
  let xb = n * cin * P + pp;
  for (var k: u32 = 0u; k < cin; k = k + 1u) {
    let v = x[xb + k * P];
    let wb = wo + k * cout;
    for (var o: u32 = 0u; o < CO; o = o + 1u) {
      if (o < cout) { acc[o] = acc[o] + v * w[wb + o]; }
    }
  }
  for (var o: u32 = 0u; o < CO; o = o + 1u) {
    if (o < cout) {
      let idx = (n * cout + o) * P + pp;
      var rv = 0.0;
      if (p.e.z != 0u) { rv = r[idx]; }
      y[idx] = epi(acc[o] + w[bo + o], rv);
    }
  }
}
`;
}

// CTC head, fused: logits L[n, v, t] = sum_k Wt[k, v] X[n, k, t] + b[v] (optionally fake-quantized,
// rec_latin), then per timestep ids = argmax_v (first maximum) and conf = exp(max - logsumexp). The logits
// never reach memory. A workgroup owns 64 timesteps of one line and walks the vocabulary in tiles of 64
// (the PW GEMM tile); each thread keeps a running (max, first argmax, sum of exp) for its 4 timesteps over
// the 4 vocabulary rows it computes in every tile; the 16 partial states per timestep are merged at the end.
// a = (cin, V, T, wo), b = (bo, has_fq, 0, 0), f = (s, z, 0, 0); grid (ceil(T / 64), N).
export const HEADK = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32>, e: vec4<u32>, f: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> conf: array<f32>;
var<workgroup> sh: array<vec4<f32>, 512>;
var<workgroup> shi: array<vec4<u32>, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let cin = p.a.x; let V = p.a.y; let T = p.a.z; let wo = p.a.w;
  let bo = p.b.x; let hasfq = p.b.y != 0u; let s = p.f.x; let zp = p.f.y;
  let n = wg.y; let t0 = wg.x * 64u;
  let tid = l.y * 16u + l.x;
  let lk = tid / 16u; let l4 = (tid % 16u) * 4u;
  let xb = n * cin * T;
  var m = vec4<f32>(-3.0e38);
  var se = vec4<f32>(0.0);
  var am = vec4<u32>(0u);
  for (var vc: u32 = 0u; vc < V; vc = vc + 64u) {
    var acc: array<vec4<f32>, 4>;
    acc[0] = vec4<f32>(0.0); acc[1] = vec4<f32>(0.0); acc[2] = vec4<f32>(0.0); acc[3] = vec4<f32>(0.0);
    for (var k0: u32 = 0u; k0 < cin; k0 = k0 + 16u) {
      let k = k0 + lk;
      let kin = k < cin;
      let kc = min(k, cin - 1u);
      var xv = vec4<f32>(0.0);
      var wv = vec4<f32>(0.0);
      for (var c: u32 = 0u; c < 4u; c = c + 1u) {
        let tt = t0 + l4 + c;
        let vv = vc + l4 + c;
        xv[c] = select(0.0, x[xb + kc * T + min(tt, T - 1u)], kin && tt < T);
        wv[c] = select(0.0, w[wo + kc * V + min(vv, V - 1u)], kin && vv < V);
      }
      sh[256u + tid] = xv;
      sh[tid] = wv;
      workgroupBarrier();
      for (var kk: u32 = 0u; kk < 16u; kk = kk + 1u) {
        let a = sh[kk * 16u + l.y];
        let b = sh[256u + kk * 16u + l.x];
        acc[0] = acc[0] + a.x * b;
        acc[1] = acc[1] + a.y * b;
        acc[2] = acc[2] + a.z * b;
        acc[3] = acc[3] + a.w * b;
      }
      workgroupBarrier();
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let v = vc + l.y * 4u + i;
      if (v < V) {
        var lg = acc[i] + vec4<f32>(w[bo + v]);
        if (hasfq) {
          lg = (clamp(round(lg / s) + zp, vec4<f32>(0.0), vec4<f32>(255.0)) - zp) * s;
        }
        let gt = lg > m;
        se = select(se + exp(lg - m), se * exp(m - lg) + 1.0, gt);
        am = select(am, vec4<u32>(v), gt);
        m = select(m, lg, gt);
      }
    }
  }
  sh[tid] = m;
  sh[256u + tid] = se;
  shi[tid] = am;
  workgroupBarrier();
  if (l.y == 0u) {
    var M = sh[l.x];
    var I = shi[l.x];
    for (var yy: u32 = 1u; yy < 16u; yy = yy + 1u) {
      let mm = sh[yy * 16u + l.x];
      let ii = shi[yy * 16u + l.x];
      let better = (mm > M) | ((mm == M) & (ii < I));
      M = select(M, mm, better);
      I = select(I, ii, better);
    }
    var S = vec4<f32>(0.0);
    for (var yy: u32 = 0u; yy < 16u; yy = yy + 1u) {
      S = S + sh[256u + yy * 16u + l.x] * exp(sh[yy * 16u + l.x] - M);
    }
    let cf = exp(M - (log(S) + M));
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let t = t0 + l.x * 4u + j;
      if (t < T) {
        ids[n * T + t] = I[j];
        conf[n * T + t] = cf[j];
      }
    }
  }
}
`;

// Gray weighting (det): y[n, 0, q] = x[n,0,q] * w0 + x[n,1,q] * w1 + x[n,2,q] * w2 (Mul + ReduceSum).
// a = (HW, total = N * HW, nx, 0), f = (w0, w1, w2, 0).
export const GRAY = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32>, e: vec4<u32>, f: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.a.z;
  if (i >= p.a.y) { return; }
  let hw = p.a.x;
  let n = i / hw; let q = i - n * hw;
  let b = n * 3u * hw + q;
  y[i] = x[b] * p.f.x + x[b + hw] * p.f.y + x[b + 2u * hw] * p.f.z;
}
`;

// Elementwise y = epi(a, b): Add (has_res = 1, b the residual) or a lone fake-quant (b unused).
// a = (total, nx, 0, 0).
export const EW = HEAD + /* wgsl */ `
@group(0) @binding(1) var<storage, read> xa: array<f32>;
@group(0) @binding(2) var<storage, read> xb: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.a.y;
  if (i >= p.a.x) { return; }
  var rv = 0.0;
  if (p.e.z != 0u) { rv = xb[i]; }
  y[i] = epi(xa[i], rv);
}
`;

// Nearest upsample by an integer factor, asymmetric coordinates, floor.
// a = (ih, iw, oh, ow), b = (scale, planes = N * C, 0, 0); grid (ceil(ow / 8), ceil(oh / 8), planes).
export const RESIZE = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32>, e: vec4<u32>, f: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let ih = p.a.x; let iw = p.a.y; let oh = p.a.z; let ow = p.a.w; let s = p.b.x;
  if (g.x >= ow || g.y >= oh) { return; }
  y[(g.z * oh + g.y) * ow + g.x] = x[(g.z * ih + g.y / s) * iw + g.x / s];
}
`;

// Concat along channels: copies x [N, cs, h, w] into y [N, ctot, h, w] at channel offset coff.
// a = (h, w, cs, ctot), b = (coff, 0, 0, 0); grid (ceil(w / 8), ceil(h / 8), N * cs).
export const COPY = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32>, e: vec4<u32>, f: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let h = p.a.x; let wd = p.a.y; let cs = p.a.z; let ct = p.a.w; let coff = p.b.x;
  if (g.x >= wd || g.y >= h) { return; }
  let n = g.z / cs; let c = g.z - n * cs;
  y[((n * ct + coff + c) * h + g.y) * wd + g.x] = x[(g.z * h + g.y) * wd + g.x];
}
`;

// Router pooling: x [N, C, 1, T] -> y [N, 2C] = (mean over T, max over T) (ReduceMean, ReduceMax, Concat).
// a = (C, T, total = N * C, 0); 64-wide workgroups over n * C + c.
export const MEANMAX = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, d: vec4<u32>, e: vec4<u32>, f: vec4<f32> };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let C = p.a.x; let T = p.a.y;
  let i = g.x;
  if (i >= p.a.z) { return; }
  let n = i / C; let c = i - n * C;
  var s = 0.0;
  var m = -3.0e38;
  for (var t: u32 = 0u; t < T; t = t + 1u) {
    let v = x[i * T + t];
    s = s + v;
    m = max(m, v);
  }
  y[n * 2u * C + c] = s / f32(T);
  y[n * 2u * C + C + c] = m;
}
`;
