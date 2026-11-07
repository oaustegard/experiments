// Rectified-flow image U-Net (conv + self-attention) in hand-written WGSL compute.
// Layout: activations are NCHW in storage buffers, stored as f32 or as f16 pairs packed into u32
// (pack2x16float, core WGSL, so no shader-f16 feature is needed); arithmetic is always f32.
// Attention outputs are always stored f32 (one thread owns one query, so packing pairs would race).
// All weights live in one f32 buffer. The sampler state stays f32.
// The network is read from manifest.layout (train/model.py layout()), so this file and the model
// share one op list. Each step: embeddings + FiLM in JS (tiny), then ~150 dispatches in one pass.
// Conv xform modes: 0 plain input, 1 silu(a * x + b) (GroupNorm + FiLM + SiLU), 2 a * x + b
// (GroupNorm alone, before attention).

const G = 16;            // GroupNorm groups
const TDIM = 128, EMB = 256;
const HEAD_DIM = 64;

// Storage helpers. half = true: element i of a packed buffer is half i & 1 of word i >> 1, and
// every writer stores two adjacent elements at once (all widths are even).
const AT = (half) => (half ? "u32" : "f32");
const LDFN = (half, name, buf) => half
  ? `fn ${name}(i: u32) -> f32 { return unpack2x16float(${buf}[i >> 1u])[i & 1u]; }`
  : `fn ${name}(i: u32) -> f32 { return ${buf}[i]; }`;
const ST2 = (half, buf, i, a, b) => half
  ? `${buf}[(${i}) >> 1u] = pack2x16float(vec2f(${a}, ${b}));`
  : `${buf}[${i}] = ${a}; ${buf}[(${i}) + 1u] = ${b};`;

// 3x3 / 1x1 conv, register-tiled (from wgsl-diffusion). A workgroup of 64 threads computes an 8x8 pixel tile x 32 output channels;
// each thread owns 4 adjacent pixels x 8 channels (8 vec4 accumulators), so one input read feeds 8 FMAs
// and one vec4 weight read feeds 4 pixels. Weights are laid out [Cin][K*K][Cout] (see relayout()) so the
// shared-memory load is contiguous. K and STRIDE are pipeline constants, so the tap loops unroll.
// Split-K: with S > 1 each workgroup covers Cin / S input channels and writes a partial sum; a
// second pass (WGSL_SPLITSUM) adds the partials, bias and residual. That gives the 8x8 and 16x16
// layers enough workgroups to fill the GPU.
const WGSL_CONV2 = (half, xHalf = half) => /* wgsl */`
override K: u32 = 3u;
override STRIDE: u32 = 1u;
struct P { N:u32, Cin:u32, Hin:u32, Win:u32, Cout:u32, Hout:u32, Wout:u32, circ:u32,
           xform:u32, res:u32, wOff:u32, bOff:u32, S:u32, cinPer:u32, partial:u32, _a:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<${AT(xHalf)}>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> AB: array<f32>;
@group(0) @binding(4) var<storage, read> R: array<${AT(half)}>;
@group(0) @binding(5) var<storage, read_write> Y: array<${AT(half)}>;
@group(0) @binding(6) var<storage, read_write> Pt: array<f32>;     // split-K partial sums
${LDFN(xHalf, "ldX", "X")}
${LDFN(half, "ldR", "R")}
const CIB = 4u;
var<workgroup> sIn: array<f32, 1156>;     // CIB * 17 * 17
var<workgroup> sW: array<vec4f, 288>;     // CIB * 9 * 32 / 4
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let cob = (p.Cout + 31u) / 32u;
  let s = wg.z % p.S;
  let rest = wg.z / p.S;
  let co0 = (rest % cob) * 32u;
  let n = rest / cob;
  let T = 7u * STRIDE + K;
  let KK = K * K;
  let hk = i32(K / 2u);
  let ix0 = i32(wg.x * 8u * STRIDE) - hk;
  let iy0 = i32(wg.y * 8u * STRIDE) - hk;
  let H = i32(p.Hin); let W = i32(p.Win);
  let pg = li % 16u; let cog = li / 16u;
  let prow = pg / 2u; let pcol0 = (pg % 2u) * 4u;
  var acc = array<vec4f, 8>(vec4f(0.), vec4f(0.), vec4f(0.), vec4f(0.), vec4f(0.), vec4f(0.), vec4f(0.), vec4f(0.));
  let cBeg = s * p.cinPer;
  let cEnd = min(cBeg + p.cinPer, p.Cin);
  for (var ci0 = cBeg; ci0 < cEnd; ci0 += CIB) {
    let tileN = CIB * T * T;
    for (var i = li; i < tileN; i += 64u) {
      let c = i / (T * T); let r = i % (T * T);
      let ci = ci0 + c;
      var v = 0.0;
      if (ci < cEnd) {
        var iy = iy0 + i32(r / T); var ix = ix0 + i32(r % T);
        var inb = iy >= 0 && iy < H && ix >= 0 && ix < W;
        if (p.circ == 1u) { iy = ((iy % H) + H) % H; ix = ((ix % W) + W) % W; inb = true; }
        if (inb) {
          v = ldX(((n * p.Cin + ci) * p.Hin + u32(iy)) * p.Win + u32(ix));
          if (p.xform != 0u) {
            let ab = 2u * (n * p.Cin + ci);
            let z = AB[ab] * v + AB[ab + 1u];
            v = select(z, z / (1.0 + exp(-z)), p.xform == 1u);
          }
        }
      }
      sIn[i] = v;
    }
    let wN = CIB * KK * 8u;
    for (var i = li; i < wN; i += 64u) {
      let q4 = i % 8u; let r = i / 8u;
      let k = r % KK; let c = r / KK;
      let ci = ci0 + c; let co = co0 + q4 * 4u;
      var w = vec4f(0.);
      if (ci < cEnd) {
        let base = p.wOff + (ci * KK + k) * p.Cout + co;
        if (co + 3u < p.Cout) {
          w = vec4f(Wt[base], Wt[base + 1u], Wt[base + 2u], Wt[base + 3u]);
        } else {
          for (var j = 0u; j < 4u; j++) { if (co + j < p.Cout) { w[j] = Wt[base + j]; } }
        }
      }
      sW[i] = w;
    }
    workgroupBarrier();
    for (var c = 0u; c < CIB; c++) {
      for (var ky = 0u; ky < K; ky++) {
        let rb = (c * T + prow * STRIDE + ky) * T + pcol0 * STRIDE;
        var seg = array<f32, 9>(0., 0., 0., 0., 0., 0., 0., 0., 0.);
        for (var j = 0u; j < 3u * STRIDE + K; j++) { seg[j] = sIn[rb + j]; }
        for (var kx = 0u; kx < K; kx++) {
          let wb = (c * KK + ky * K + kx) * 8u + cog * 2u;
          let w0 = sW[wb]; let w1 = sW[wb + 1u];
          for (var q = 0u; q < 4u; q++) {
            let v = seg[q * STRIDE + kx];
            acc[2u * q] += w0 * v;
            acc[2u * q + 1u] += w1 * v;
          }
        }
      }
    }
    workgroupBarrier();
  }
  // write pixel pairs (q, q + 1): ox is even and Wout is even, so a pair never straddles a row
  let HW = p.Hout * p.Wout;
  let oy = wg.y * 8u + prow;
  for (var q = 0u; q < 4u; q += 2u) {
    let ox = wg.x * 8u + pcol0 + q;
    if (ox < p.Wout && oy < p.Hout) {
      for (var h2 = 0u; h2 < 2u; h2++) {
        let a0 = acc[2u * q + h2];
        let a1 = acc[2u * q + 2u + h2];
        for (var j = 0u; j < 4u; j++) {
          let c = co0 + cog * 8u + h2 * 4u + j;
          if (c < p.Cout) {
            if (p.partial == 1u) {
              let pi = ((s * p.N + n) * p.Cout + c) * HW + oy * p.Wout + ox;
              Pt[pi] = a0[j]; Pt[pi + 1u] = a1[j];
            } else {
              let idx = ((n * p.Cout + c) * p.Hout + oy) * p.Wout + ox;
              let b = Wt[p.bOff + c];
              var y0 = a0[j] + b; var y1 = a1[j] + b;
              if (p.res == 1u) { y0 += ldR(idx); y1 += ldR(idx + 1u); }
              ${ST2(half, "Y", "idx", "y0", "y1")}
            }
          }
        }
      }
    }
  }
}`;

const WGSL_SPLITSUM = (half) => /* wgsl */`
struct P { total:u32, S:u32, HW:u32, Cout:u32, bOff:u32, res:u32, _a:u32, _b:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pt: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<${AT(half)}>;
@group(0) @binding(4) var<storage, read_write> Y: array<${AT(half)}>;
${LDFN(half, "ldR", "R")}
fn one(i: u32) -> f32 {
  var y = Wt[p.bOff + (i / p.HW) % p.Cout];
  for (var s = 0u; s < p.S; s++) { y += Pt[s * p.total + i]; }
  if (p.res == 1u) { y += ldR(i); }
  return y;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = 2u * (gid.x + gid.y * nw.x * 256u);
  if (i >= p.total) { return; }
  let y0 = one(i); let y1 = one(i + 1u);
  ${ST2(half, "Y", "i", "y0", "y1")}
}`;

// GroupNorm statistics fused with the affine + FiLM that follow it:
// writes a, b per (n, c) so the consumer computes silu(a * x + b).
const WGSL_GNORM = (half) => /* wgsl */`
struct P { N:u32, C:u32, HW:u32, gOff:u32, bOff:u32, film:u32, fOff:u32, _a:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<${AT(half)}>;
${LDFN(half, "ldX", "X")}
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> F: array<f32>;
@group(0) @binding(4) var<storage, read_write> AB: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> mean_s: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let n = wg.x / ${G}u; let g = wg.x % ${G}u;
  let cpg = p.C / ${G}u;
  let cnt = cpg * p.HW;
  let base = (n * p.C + g * cpg) * p.HW;
  var s = 0.0;
  for (var i = li; i < cnt; i += 256u) { s += ldX(base + i); }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  if (li == 0u) { mean_s = red[0] / f32(cnt); }
  workgroupBarrier();
  let mean = mean_s;
  var q = 0.0;
  for (var i = li; i < cnt; i += 256u) { let d = ldX(base + i) - mean; q += d * d; }
  workgroupBarrier();
  red[li] = q;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let rstd = inverseSqrt(red[0] / f32(cnt) + 1e-5);
  if (li < cpg) {
    let c = g * cpg + li;
    let gam = Wt[p.gOff + c]; let bet = Wt[p.bOff + c];
    var a = rstd * gam; var b = bet - mean * rstd * gam;
    if (p.film == 1u) {
      let sc = F[p.fOff + n * 2u * p.C + c];
      let sh = F[p.fOff + n * 2u * p.C + p.C + c];
      a = a * (1.0 + sc); b = b * (1.0 + sc) + sh;
    }
    AB[2u * (n * p.C + c)] = a;
    AB[2u * (n * p.C + c) + 1u] = b;
  }
}`;

// Optional nearest 2x upsample of A, concatenated with B along channels.
const WGSL_UPCAT = (half) => /* wgsl */`
struct P { N:u32, C1:u32, C2:u32, H:u32, W:u32, up:u32, total:u32, _a:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<${AT(half)}>;
@group(0) @binding(2) var<storage, read> B: array<${AT(half)}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${AT(half)}>;
${LDFN(half, "ldA", "A")}
${LDFN(half, "ldB", "B")}
fn one(i: u32) -> f32 {
  let C = p.C1 + p.C2;
  let x = i % p.W; let y = (i / p.W) % p.H; let c = (i / (p.W * p.H)) % C; let n = i / (p.W * p.H * C);
  if (c < p.C1) {
    let h = p.H >> p.up; let w = p.W >> p.up;
    return ldA(((n * p.C1 + c) * h + (y >> p.up)) * w + (x >> p.up));
  }
  return ldB(((n * p.C2 + c - p.C1) * p.H + y) * p.W + x);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = 2u * (gid.x + gid.y * nw.x * 256u);
  if (i >= p.total) { return; }
  let y0 = one(i); let y1 = one(i + 1u);
  ${ST2(half, "Y", "i", "y0", "y1")}
}`;

// Multi-head self-attention, flash style: one thread per query keeps q (pre-scaled by 1/sqrt(D)) and
// its output accumulator in registers and streams keys and values through shared memory, 16 at a
// time, with an online softmax (running max m, running sum l). QKV is the 1x1 conv output, channels
// [q | k | v], each head-major (channel = part * C + head * D + d), positions L = H * W. The output
// O is NCHW f32, channel = head * D + d, ready for the projection conv. Dispatch (ceil(L / 64), heads, N).
const WGSL_ATTN = (half) => /* wgsl */`
const D4 = ${HEAD_DIM / 4}u;
const KT = 16u;
struct P { N:u32, C:u32, L:u32, heads:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> QKV: array<${AT(half)}>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
${LDFN(half, "ld", "QKV")}
var<workgroup> sK: array<vec4f, ${16 * HEAD_DIM / 4}>;
var<workgroup> sV: array<vec4f, ${16 * HEAD_DIM / 4}>;
fn ld4(b: u32, L: u32) -> vec4f { return vec4f(ld(b), ld(b + L), ld(b + 2u * L), ld(b + 3u * L)); }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let n = wg.z; let h = wg.y; let i = wg.x * 64u + li;
  let L = p.L;
  let qb = (n * 3u * p.C + h * 4u * D4) * L;
  let kb = qb + p.C * L;
  let vb = kb + p.C * L;
  let live = i < L;
  let scale = inverseSqrt(f32(4u * D4));
  var q: array<vec4f, D4>;
  var acc: array<vec4f, D4>;
  for (var e = 0u; e < D4; e++) {
    acc[e] = vec4f(0.);
    q[e] = vec4f(0.);
    if (live) { q[e] = ld4(qb + 4u * e * L + i, L) * scale; }
  }
  var m = -1e30;
  var l = 0.0;
  var s = array<f32, 16>(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);
  for (var j0 = 0u; j0 < L; j0 += KT) {
    for (var t = li; t < KT * D4; t += 64u) {
      let j = t % KT; let e = t / KT; let jj = j0 + j;
      var kv = vec4f(0.); var vv = vec4f(0.);
      if (jj < L) { kv = ld4(kb + 4u * e * L + jj, L); vv = ld4(vb + 4u * e * L + jj, L); }
      sK[j * D4 + e] = kv;
      sV[j * D4 + e] = vv;
    }
    workgroupBarrier();
    if (live) {
      let nk = min(KT, L - j0);
      var mt = m;
      for (var j = 0u; j < nk; j++) {
        var d = 0.0;
        for (var e = 0u; e < D4; e++) { d += dot(q[e], sK[j * D4 + e]); }
        s[j] = d;
        mt = max(mt, d);
      }
      let c = exp(m - mt);
      l *= c;
      for (var e = 0u; e < D4; e++) { acc[e] *= c; }
      for (var j = 0u; j < nk; j++) {
        let pj = exp(s[j] - mt);
        l += pj;
        for (var e = 0u; e < D4; e++) { acc[e] += pj * sV[j * D4 + e]; }
      }
      m = mt;
    }
    workgroupBarrier();
  }
  if (live) {
    let inv = 1.0 / l;
    let ob = (n * p.C + h * 4u * D4) * L + i;
    for (var e = 0u; e < D4; e++) {
      let a = acc[e] * inv;
      O[ob + 4u * e * L] = a.x; O[ob + (4u * e + 1u) * L] = a.y;
      O[ob + (4u * e + 2u) * L] = a.z; O[ob + (4u * e + 3u) * L] = a.w;
    }
  }
}`;

// Euler step with classifier-free guidance. S is the f32 state; X receives N copies of it as the
// next network input; V holds the model outputs (cond first, unconditional second when N = 2).
// Also writes the x0 estimate D for display.
const WGSL_STEP = (half) => /* wgsl */`
struct P { N:u32, HW3:u32, t:f32, dt:f32, cfg:f32, _a:u32, _b:u32, _c:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<${AT(half)}>;
@group(0) @binding(3) var<storage, read_write> D: array<f32>;
@group(0) @binding(4) var<storage, read_write> X: array<${AT(half)}>;
${LDFN(half, "ldV", "V")}
fn one(i: u32) -> f32 {
  var v = ldV(i);
  if (p.N == 2u) { let vu = ldV(p.HW3 + i); v = vu + p.cfg * (v - vu); }
  let x = S[i];
  D[i] = x - p.t * v;
  let xn = x + p.dt * v;
  S[i] = xn;
  return xn;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let i = 2u * (gid.x + gid.y * nw.x * 256u);
  if (i >= p.HW3) { return; }
  let x0 = one(i); let x1 = one(i + 1u);
  ${ST2(half, "X", "i", "x0", "x1")}
  if (p.N == 2u) { ${ST2(half, "X", "p.HW3 + i", "x0", "x1")} }
}`;

// Draws the 3 x H x W image in D to the canvas, scaled to fit: nearest (interp = 0) or bilinear.
const WGSL_SHOW = /* wgsl */`
struct P { H:u32, W:u32, scale:f32, interp:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> D: array<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}
fn texel(x: i32, y: i32) -> vec3f {
  let o = u32(clamp(y, 0, i32(p.H) - 1)) * p.W + u32(clamp(x, 0, i32(p.W) - 1));
  let hw = p.H * p.W;
  return clamp((vec3f(D[o], D[hw + o], D[2u * hw + o]) + 1.0) * 0.5, vec3f(0.0), vec3f(1.0));
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  if (p.interp == 0u) {
    let q = vec2i(floor(pos.xy / p.scale));
    return vec4f(texel(q.x, q.y), 1.0);
  }
  let f = pos.xy / p.scale - 0.5;
  let q = vec2i(floor(f)); let w = fract(f);
  let c = mix(mix(texel(q.x, q.y), texel(q.x + 1, q.y), w.x),
              mix(texel(q.x, q.y + 1), texel(q.x + 1, q.y + 1), w.x), w.y);
  return vec4f(c, 1.0);
}`;

export function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
// f32 -> f16 bits, round to nearest (ties away), flush below the f16 subnormal range
export function toHalf(v) {
  _f[0] = v;
  const x = _u[0], sgn = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff;
  if (e >= 31) return sgn | 0x7c00;
  if (e <= 0) return e < -10 ? sgn : sgn | (((m | 0x800000) >>> (1 - e)) + 0x1000) >>> 13;
  return sgn | ((e << 10) + ((m + 0x1000) >>> 13));
}

// Float32Array -> bytes in a storage layout: f32 as is, or f16 pairs packed into u32
export function packAct(x, half) {
  if (!half) return x;
  const u = new Uint16Array(x.length + (x.length & 1));
  for (let i = 0; i < x.length; i++) u[i] = toHalf(x[i]);
  return u;
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(n, seed) {
  const r = mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(r(), 1e-12), v = r();
    const m = Math.sqrt(-2 * Math.log(u));
    out[i] = m * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = m * Math.sin(2 * Math.PI * v);
  }
  return out;
}

export class ImageDiffusion {
  static async create(device, base, opts = {}) {
    const man = await (await fetch(`${base}/manifest.json`)).json();
    const raw = new Uint16Array(await (await fetch(`${base}/weights.bin`)).arrayBuffer());
    return new ImageDiffusion(device, man, f16ToF32(raw), opts);
  }

  // opts.act: "f16" (activations stored as packed f16, default) or "f32"; opts.splitTarget: split-K
  // until a conv has this many workgroups (0 turns split-K off).
  constructor(device, man, w, { splitTarget = 256, act = "f16" } = {}) {
    this.device = device;
    this.man = man;
    this.w = w;
    this.half = act === "f16";
    this.splitTarget = splitTarget;
    this.nclass = man.nclass;
    this.layout = man.layout;                       // [kind, name, cin, cout], see train/model.py
    this.blocks = this.layout.filter((o) => o[0] === "res").map(([, name, cin, cout]) => [name, cin, cout]);
    const gw = this.relayout(w);
    this.wbuf = device.createBuffer({ size: gw.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.wbuf, 0, gw);
    const half = this.half;
    const mk = (code, constants) => device.createComputePipeline({
      layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main", constants } });
    this.pipe = { gnorm: mk(WGSL_GNORM(half)), upcat: mk(WGSL_UPCAT(half)), step: mk(WGSL_STEP(half)),
                  splitsum: mk(WGSL_SPLITSUM(half)), attn: mk(WGSL_ATTN(half)) };
    this.conv2 = new Map();
    for (const [K, S, xf32] of [[3, 1, 0], [3, 2, 0], [1, 1, 0], [1, 1, 1]]) {
      this.conv2.set(`${K}_${S}_${xf32}`, mk(WGSL_CONV2(half, half && !xf32), { K, STRIDE: S }));
    }
    this.dummy = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });     // read-only slots
    this.dummyW = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });    // an unused writable slot
    // FiLM offsets: per res block, 2 * cout floats per batch item (batch is at most 2)
    this.filmOff = {};
    let off = 0;
    for (const [name, , cout] of this.blocks) { this.filmOff[name] = off; off += 2 * cout * 2; }
    this.filmLen = off;
    this.film = device.createBuffer({ size: off * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.planKey = null;
  }

  // Copy of the weights with every conv kernel moved from [Cout][Cin][K][K] to [Cin][K*K][Cout].
  relayout(w) {
    const out = new Float32Array(w);
    for (const [, [o, s]] of Object.entries(this.man.tensors)) {
      if (s.length !== 4) continue;
      const [co, ci, kh, kw] = s, KK = kh * kw;
      for (let a = 0; a < co; a++)
        for (let b = 0; b < ci; b++)
          for (let k = 0; k < KK; k++) out[o + (b * KK + k) * co + a] = w[o + (a * ci + b) * KK + k];
    }
    return out;
  }

  T(name) { return this.man.tensors[name]; }
  get(name) { const [o, s] = this.T(name); return this.w.subarray(o, o + s.reduce((a, b) => a * b, 1)); }

  linear(name, x) {
    const W = this.get(`${name}.weight`), b = this.get(`${name}.bias`);
    const [, [out, inn]] = this.T(`${name}.weight`);
    const y = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = b[o];
      for (let i = 0; i < inn; i++) s += W[o * inn + i] * x[i];
      y[o] = s;
    }
    return y;
  }

  // silu(t2(silu(t1(temb(t)))) + cls[y]) for one item, then every block's FiLM projection
  filmFor(t, y) {
    const half = TDIM / 2, te = new Float32Array(TDIM);
    for (let i = 0; i < half; i++) {
      const a = t * 1000 * Math.exp(-Math.log(10000) * i / half);
      te[i] = Math.cos(a); te[half + i] = Math.sin(a);
    }
    const silu = (v) => v.map((z) => z / (1 + Math.exp(-z)));
    const h = this.linear("t2", silu(this.linear("t1", te)));
    const ce = this.get("cls.weight").subarray(y * EMB, (y + 1) * EMB);
    const e = silu(h.map((v, i) => v + ce[i]));
    const out = {};
    for (const [name] of this.blocks) out[name] = this.linear(`${name}.emb`, e);
    return out;
  }

  filmData(t, ys, into = new Float32Array(this.filmLen), at = 0) {
    ys.forEach((y, n) => {
      const f = this.filmFor(t, y);
      for (const [name, , cout] of this.blocks) into.set(f[name], at + this.filmOff[name] + n * 2 * cout);
    });
    return into;
  }

  // Build buffers and the dispatch list for one (N, H, W) configuration.
  plan(N, H, W) {
    const key = `${N}:${H}:${W}`;
    if (this.planKey === key) return;
    if (H % 16 || W % 16) throw new Error("H and W must be multiples of 16");
    this.freePlan();
    const dev = this.device, ops = [];
    let tid = 0;
    const tensors = [];
    const actBytes = (n, f32) => (this.half && !f32 ? 4 * Math.ceil(n / 2) : 4 * n);
    const tensor = (C, h, w, f32 = false) => {
      const size = N * C * h * w, t = { id: tid++, size, bytes: actBytes(size, f32), C, h, w, f32 };
      tensors.push(t); return t;
    };
    const abT = (C) => { const t = { id: tid++, size: N * C * 2, bytes: N * C * 2 * 4, ab: true }; tensors.push(t); return t; };
    const off = (name) => this.T(name)[0];
    const conv = (name, x, cout, { stride = 1, K = 3, ab = null, xform = 1, res = null } = {}) => {
      const ho = stride === 2 ? x.h / 2 : x.h, wo = stride === 2 ? x.w / 2 : x.w;
      const y = tensor(cout, ho, wo);
      const cob = Math.ceil(cout / 32), tiles = Math.ceil(wo / 8) * Math.ceil(ho / 8), base = tiles * N * cob;
      let S = 1;
      while (base * S < this.splitTarget && x.C % (8 * S) === 0 && x.C / (2 * S) >= 16) S *= 2;
      const u = [N, x.C, x.h, x.w, cout, ho, wo, 0, ab ? xform : 0, res ? 1 : 0,
                 off(`${name}.weight`), off(`${name}.bias`), S, x.C / S, S > 1 ? 1 : 0, 0];
      const wg = [Math.ceil(wo / 8), Math.ceil(ho / 8), N * cob * S];
      const pk = `${K}_${stride}_${x.f32 ? 1 : 0}`;
      if (S === 1) {
        ops.push({ k: "conv2", pk, ins: [x, ab, res], out: y, u, wg });
      } else {
        const part = { id: tid++, size: S * N * cout * ho * wo, bytes: 4 * S * N * cout * ho * wo };
        tensors.push(part);
        ops.push({ k: "conv2", pk, partial: true, ins: [x, ab, null], out: part, u, wg });
        ops.push({ k: "splitsum", ins: [part, res], out: y,
                   u: [y.size, S, ho * wo, cout, off(`${name}.bias`), res ? 1 : 0, 0, 0], wg: this.flat(y.size / 2) });
      }
      return y;
    };
    const gnorm = (name, x, film) => {
      const ab = abT(x.C);
      ops.push({ k: "gnorm", ins: [x], out: ab,
        u: [N, x.C, x.h * x.w, off(`${name}.weight`), off(`${name}.bias`), film ? 1 : 0, film ? this.filmOff[film] : 0, 0],
        wg: [N * G, 1, 1] });
      return ab;
    };
    const block = (name, x, cout) => {
      const h = conv(`${name}.c1`, x, cout, { ab: gnorm(`${name}.n1`, x, null) });
      const skip = this.man.tensors[`${name}.skip.weight`] ? conv(`${name}.skip`, x, cout, { K: 1 }) : x;
      return conv(`${name}.c2`, h, cout, { ab: gnorm(`${name}.n2`, h, name), res: skip });
    };
    const attn = (name, x) => {
      const qkv = conv(`${name}.qkv`, x, 3 * x.C, { K: 1, ab: gnorm(`${name}.norm`, x, null), xform: 2 });
      const o = tensor(x.C, x.h, x.w, true);
      const heads = x.C / HEAD_DIM, L = x.h * x.w;
      ops.push({ k: "attn", ins: [qkv], out: o, u: [N, x.C, L, heads], wg: [Math.ceil(L / 64), heads, N] });
      return conv(`${name}.proj`, o, x.C, { K: 1, res: x });
    };
    const upcat = (a, b) => {
      const y = tensor(a.C + b.C, b.h, b.w);
      ops.push({ k: "upcat", ins: [a, b], out: y,
                 u: [N, a.C, b.C, b.h, b.w, a.h === b.h ? 0 : 1, y.size, 0], wg: this.flat(y.size / 2) });
      return y;
    };
    const x = tensor(3, H, W);
    const skips = {};
    let h = x;
    for (const [kind, name, , cout] of this.layout) {
      if (kind === "conv_in") h = conv("conv_in", h, cout);
      else if (kind === "res") h = block(name, h, cout);
      else if (kind === "attn") h = attn(name, h);
      else if (kind === "skip") skips[name] = h;
      else if (kind === "down") h = conv(name, h, cout, { stride: 2 });
      else if (kind === "upcat") h = upcat(h, skips[name]);
      else if (kind === "out") h = conv("conv_out", h, cout, { ab: gnorm("n_out", h, null) });
      else throw new Error(`unknown op ${kind}`);
    }
    const v = h;

    // liveness-based buffer reuse: a tensor's buffer returns to the pool after its last reader
    const last = new Map();
    ops.forEach((op, i) => op.ins.forEach((t) => t && last.set(t.id, i)));
    const pool = [], all = [];
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const take = (bytes) => {       // best fit by byte size
      let bi = -1;
      for (let i = 0; i < pool.length; i++) if (pool[i].n >= bytes && (bi < 0 || pool[i].n < pool[bi].n)) bi = i;
      if (bi >= 0) return pool.splice(bi, 1)[0];
      const b = { n: bytes, buf: dev.createBuffer({ size: Math.max(16, bytes), usage }) };
      all.push(b);
      return b;
    };
    const bufs = new Map();
    const pinned = new Set([x.id, v.id]);
    bufs.set(x.id, take(x.bytes));
    ops.forEach((op, i) => {
      bufs.set(op.out.id, take(op.out.bytes));
      for (const t of op.ins) if (t && !pinned.has(t.id) && last.get(t.id) === i) pool.push(bufs.get(t.id));
      if (!pinned.has(op.out.id) && !last.has(op.out.id)) pool.push(bufs.get(op.out.id));
    });
    const B = (t) => (t ? bufs.get(t.id).buf : this.dummy);
    this.ops = ops.map((op) => {
      const ub = dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      dev.queue.writeBuffer(ub, 0, new Uint32Array([...op.u, ...new Array(16 - op.u.length).fill(0)]));
      const p = op.k === "conv2" ? this.conv2.get(op.pk) : this.pipe[op.k];
      let entries;
      if (op.k === "conv2") {
        entries = [ub, B(op.ins[0]), this.wbuf, B(op.ins[1]), B(op.ins[2]),
                   op.partial ? this.dummyW : B(op.out), op.partial ? B(op.out) : this.dummyW];
      } else if (op.k === "splitsum") entries = [ub, B(op.ins[0]), this.wbuf, B(op.ins[1]), B(op.out)];
      else if (op.k === "gnorm") entries = [ub, B(op.ins[0]), this.wbuf, this.film, B(op.out)];
      else if (op.k === "attn") entries = [ub, B(op.ins[0]), B(op.out)];
      else entries = [ub, B(op.ins[0]), B(op.ins[1]), B(op.out)];
      const bg = dev.createBindGroup({ layout: p.getBindGroupLayout(0),
        entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })) });
      return { p, bg, wg: op.wg, ub, k: op.k };
    });
    this.allBufs = all;
    this.xBuf = B(x);
    this.vBuf = B(v);
    this.N = N; this.H = H; this.W = W;
    const su = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.dBuf = dev.createBuffer({ size: 3 * H * W * 4, usage: su });
    this.sBuf = dev.createBuffer({ size: 3 * H * W * 4, usage: su });
    this.stepU = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.stepBG = dev.createBindGroup({ layout: this.pipe.step.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.stepU } }, { binding: 1, resource: { buffer: this.sBuf } },
      { binding: 2, resource: { buffer: this.vBuf } }, { binding: 3, resource: { buffer: this.dBuf } },
      { binding: 4, resource: { buffer: this.xBuf } }] });
    this.planKey = key;
    this.bytes = all.reduce((s, b) => s + b.n, 0);
  }

  flat(n) { const g = Math.ceil(n / 256); return g <= 65535 ? [g, 1, 1] : [16384, Math.ceil(g / 16384), 1]; }

  freePlan() {
    if (!this.allBufs) return;
    for (const b of this.allBufs) b.buf.destroy();
    for (const o of this.ops) o.ub.destroy();
    this.dBuf.destroy(); this.sBuf.destroy(); this.stepU.destroy();
    this.allBufs = null; this.planKey = null;
  }

  encodeForward(enc) {
    const pass = enc.beginComputePass();
    for (const o of this.ops) {
      pass.setPipeline(o.p); pass.setBindGroup(0, o.bg); pass.dispatchWorkgroups(...o.wg);
    }
    pass.end();
  }

  // One model evaluation on an explicit batch (testing).
  async forward(x, t, ys, H, W) {
    this.plan(ys.length, H, W);
    this.device.queue.writeBuffer(this.xBuf, 0, packAct(x, this.half));
    this.device.queue.writeBuffer(this.film, 0, this.filmData(t, ys));
    const enc = this.device.createCommandEncoder();
    this.encodeForward(enc);
    this.device.queue.submit([enc.finish()]);
    return this.readAct(this.vBuf, ys.length * 3 * H * W);
  }

  // read n elements of an activation buffer as f32
  async readAct(buf, n) {
    if (!this.half) return this.read(buf, n);
    const words = await this.read(buf, Math.ceil(n / 2));
    return f16ToF32(new Uint16Array(words.buffer)).subarray(0, n);
  }

  async read(buf, n) {
    const rb = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return out;
  }

  // Euler sampler, t: 1 -> 0. FiLM vectors and step uniforms for every step are uploaded up front,
  // and one step stays queued ahead of the one the CPU waits on, so the GPU never idles between steps.
  // onStep(i, t) runs right after step i is submitted: a draw issued there shows that step's x0 estimate.
  async sample({ cls, H = 64, W = 64, steps = 20, cfg = 2, seed = 0, noise = null, onStep = null, shouldStop = null }) {
    const N = cfg === 1 ? 1 : 2;
    this.plan(N, H, W);
    const dev = this.device;
    const x0 = noise || gaussian(3 * H * W, seed);
    const xs = new Float32Array(N * 3 * H * W);
    for (let n = 0; n < N; n++) xs.set(x0, n * 3 * H * W);
    dev.queue.writeBuffer(this.sBuf, 0, x0);                       // f32 state
    dev.queue.writeBuffer(this.xBuf, 0, packAct(xs, this.half));   // network input, N copies
    const ys = N === 2 ? [cls, this.nclass] : [cls];
    const L = this.filmLen, films = new Float32Array(steps * L), us = new ArrayBuffer(steps * 32);
    for (let i = 0; i < steps; i++) {
      const t = 1 - i / steps, tn = 1 - (i + 1) / steps;
      this.filmData(t, ys, films, i * L);
      const dv = new DataView(us, i * 32, 32);
      dv.setUint32(0, N, true); dv.setUint32(4, 3 * H * W, true);
      dv.setFloat32(8, t, true); dv.setFloat32(12, tn - t, true); dv.setFloat32(16, cfg, true);
    }
    const fAll = dev.createBuffer({ size: films.byteLength, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const uAll = dev.createBuffer({ size: us.byteLength, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(fAll, 0, films);
    dev.queue.writeBuffer(uAll, 0, us);
    let prev = null, stopped = false;
    try {
      for (let i = 0; i < steps; i++) {
        const enc = dev.createCommandEncoder();
        enc.copyBufferToBuffer(fAll, i * L * 4, this.film, 0, L * 4);
        enc.copyBufferToBuffer(uAll, i * 32, this.stepU, 0, 32);
        this.encodeForward(enc);
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pipe.step); pass.setBindGroup(0, this.stepBG);
        pass.dispatchWorkgroups(...this.flat(3 * H * W / 2));
        pass.end();
        dev.queue.submit([enc.finish()]);
        if (onStep) await onStep(i, 1 - (i + 1) / steps);
        const done = dev.queue.onSubmittedWorkDone();
        if (prev) await prev;
        prev = done;
        if (shouldStop && shouldStop()) { stopped = true; break; }
      }
      if (prev) await prev;
    } finally {
      fAll.destroy(); uAll.destroy();
    }
    if (stopped) return false;
    // final state into the display buffer
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.sBuf, 0, this.dBuf, 0, 3 * H * W * 4);
    this.device.queue.submit([enc.finish()]);
    return true;
  }
}

export class Viewer {
  constructor(device, canvas, format) {
    this.device = device;
    this.ctx = canvas.getContext("webgpu");
    this.ctx.configure({ device, format, alphaMode: "opaque" });
    const module = device.createShaderModule({ code: WGSL_SHOW });
    this.pipe = device.createRenderPipeline({ layout: "auto",
      vertex: { module, entryPoint: "vs" }, fragment: { module, entryPoint: "fs", targets: [{ format }] } });
    this.u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  draw(dBuf, H, W, scale, smooth) {
    const u = new ArrayBuffer(16), dv = new DataView(u);
    dv.setUint32(0, H, true); dv.setUint32(4, W, true); dv.setFloat32(8, scale, true); dv.setUint32(12, smooth ? 1 : 0, true);
    this.device.queue.writeBuffer(this.u, 0, u);
    if (this.bgBuf !== dBuf) {
      this.bg = this.device.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.u } }, { binding: 1, resource: { buffer: dBuf } }] });
      this.bgBuf = dBuf;
    }
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(),
      loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(this.pipe); pass.setBindGroup(0, this.bg); pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
