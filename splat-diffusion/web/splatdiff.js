// Splat-list diffusion in hand-written WGSL: a set transformer (DiT-style adaLN-Zero blocks, no
// positional encoding) denoises K splats x 10 features with rectified flow; a fragment shader paints
// the splats additively on a torus, so the tile repeats without seams at any zoom.
//
// Shapes: N = batch items (2 with guidance: chosen class + null class), K = splats (tokens),
// d = 256 width, H = 4 heads of 64. Activations f32. Weights f32 in one buffer, Linear weights
// transposed to [in][out] at load so the shared matmul reads B row-major.

const D = 10, WIDTH = 256, DEPTH = 8, HEADS = 4, TDIM = 128, NCLASS = 47;

// C[b][m][n] = epi(alpha * sum_k A[b][m][k] * B[b][k][n] + bias[n])
// batch b = b1 * nb2 + b2 with separate strides for both parts (item x head for attention).
// epi: act 1 = GELU(tanh); acc 1 = C += gate[item][n] * value (gated residual), 2 = C += value.
const WGSL_MATMUL = /* wgsl */`
struct P { M:u32, N:u32, K:u32, nb2:u32,
           aOff:u32, aM:u32, aK:u32, aB1:u32, aB2:u32,
           bOff:u32, bK:u32, bN:u32, bB1:u32, bB2:u32,
           cOff:u32, cM:u32, cN:u32, cB1:u32, cB2:u32,
           biasOff:u32, hasBias:u32, act:u32, acc:u32, gOff:u32, gItemStride:u32, rowsPerItem:u32,
           alpha:f32, _p:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<storage, read> Wt: array<f32>;     // bias source
@group(0) @binding(5) var<storage, read> G: array<f32>;      // gate source (modulation vectors)
var<workgroup> sA: array<f32, 1024>;   // 64 x 16
var<workgroup> sB: array<f32, 1024>;   // 16 x 64
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u,
        @builtin(local_invocation_index) li: u32) {
  let b = wg.z; let b1 = b / p.nb2; let b2 = b % p.nb2;
  let aBase = p.aOff + b1 * p.aB1 + b2 * p.aB2;
  let bBase = p.bOff + b1 * p.bB1 + b2 * p.bB2;
  let cBase = p.cOff + b1 * p.cB1 + b2 * p.cB2;
  let m0 = wg.y * 64u; let n0 = wg.x * 64u;
  var acc = array<vec4f, 4>(vec4f(0.), vec4f(0.), vec4f(0.), vec4f(0.));
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    for (var i = li; i < 1024u; i += 256u) {
      let mm = i / 16u; let kk = i % 16u;          // A tile [64][16]
      let gm = m0 + mm; let gk = k0 + kk;
      var v = 0.0;
      if (gm < p.M && gk < p.K) { v = A[aBase + gm * p.aM + gk * p.aK]; }
      sA[i] = v;
      let kb = i / 64u; let nb = i % 64u;          // B tile [16][64]
      let hk = k0 + kb; let hn = n0 + nb;
      var w = 0.0;
      if (hk < p.K && hn < p.N) { w = B[bBase + hk * p.bK + hn * p.bN]; }
      sB[i] = w;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let bv = vec4f(sB[kk * 64u + l.x * 4u], sB[kk * 64u + l.x * 4u + 1u],
                     sB[kk * 64u + l.x * 4u + 2u], sB[kk * 64u + l.x * 4u + 3u]);
      for (var r = 0u; r < 4u; r++) { acc[r] += sA[(l.y * 4u + r) * 16u + kk] * bv; }
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < 4u; r++) {
    let gm = m0 + l.y * 4u + r;
    if (gm >= p.M) { continue; }
    let item = gm / p.rowsPerItem;
    for (var j = 0u; j < 4u; j++) {
      let gn = n0 + l.x * 4u + j;
      if (gn >= p.N) { continue; }
      var v = p.alpha * acc[r][j];
      if (p.hasBias == 1u) { v += Wt[p.biasOff + gn]; }
      if (p.act == 1u) { v = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v))); }
      let ci = cBase + gm * p.cM + gn * p.cN;
      if (p.acc == 1u) { C[ci] = C[ci] + G[p.gOff + item * p.gItemStride + gn] * v; }
      else if (p.acc == 2u) { C[ci] = C[ci] + v; }
      else { C[ci] = v; }
    }
  }
}`;

// Row softmax in place over rows of length L (attention scores).
const WGSL_SOFTMAX = /* wgsl */`
struct P { rows:u32, L:u32, _a:u32, _b:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.rows) { return; }
  let base = row * p.L;
  var mx = -3.4e38;
  for (var i = li; i < p.L; i += 64u) { mx = max(mx, S[base + i]); }
  red[li] = mx;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (li < k) { red[li] = max(red[li], red[li + k]); } workgroupBarrier(); }
  let m = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var i = li; i < p.L; i += 64u) { let e = exp(S[base + i] - m); S[base + i] = e; s += e; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var i = li; i < p.L; i += 64u) { S[base + i] = S[base + i] * inv; }
}`;

// y = LayerNorm(x) * (1 + scale[item]) + shift[item], width 256, one workgroup per row.
const WGSL_LNMOD = /* wgsl */`
struct P { rows:u32, rowsPerItem:u32, modOff:u32, modStride:u32, shOff:u32, scOff:u32, _a:u32, _b:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Mo: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let row = wg.x + wg.y * 65535u;
  if (row >= p.rows) { return; }
  let base = row * 256u;
  var v: array<f32, 4>;
  var s = 0.0;
  for (var j = 0u; j < 4u; j++) { v[j] = X[base + li + 64u * j]; s += v[j]; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let mean = red[0] / 256.0;
  workgroupBarrier();
  var q = 0.0;
  for (var j = 0u; j < 4u; j++) { let dd = v[j] - mean; q += dd * dd; }
  red[li] = q;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let rstd = inverseSqrt(red[0] / 256.0 + 1e-5);
  let mb = p.modOff + (row / p.rowsPerItem) * p.modStride;
  for (var j = 0u; j < 4u; j++) {
    let c = li + 64u * j;
    Y[base + c] = (v[j] - mean) * rstd * (1.0 + Mo[mb + p.scOff + c]) + Mo[mb + p.shOff + c];
  }
}`;

// Euler step with guidance on the f32 state S [K*D]; V [N][K*D]; X0 gets the x0 estimate.
const WGSL_STEP = /* wgsl */`
struct P { N:u32, n:u32, t:f32, dt:f32, cfg:f32, _a:u32, _b:u32, _c:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> X0: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  var v = V[i];
  if (p.N == 2u) { let vu = V[p.n + i]; v = vu + p.cfg * (v - vu); }
  let x = S[i];
  X0[i] = x - p.t * v;
  S[i] = x + p.dt * v;
}`;

// Normalised features [K][10] -> splats [K][8]: mu_x, mu_y, a (=e^.), b, c (=e^.), r, g, b.
const WGSL_DECODE = /* wgsl */`
struct P { K:u32, _a:u32, _b:u32, _c:u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> St: array<f32>;     // mean[10], std[10]
@group(0) @binding(3) var<storage, read_write> Sp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= p.K) { return; }
  var f: array<f32, 10>;
  for (var j = 0u; j < 10u; j++) { f[j] = F[k * 10u + j] * St[10u + j] + St[j]; }
  let tau = 6.283185307;
  let mx = fract(atan2(f[1], f[0]) / tau + 1.0);
  let my = fract(atan2(f[3], f[2]) / tau + 1.0);
  let o = k * 8u;
  Sp[o] = mx; Sp[o + 1u] = my;
  Sp[o + 2u] = exp(min(f[4], 10.0)); Sp[o + 3u] = f[5]; Sp[o + 4u] = exp(min(f[6], 10.0));
  Sp[o + 5u] = f[7]; Sp[o + 6u] = f[8]; Sp[o + 7u] = f[9];
}`;

// Paints the splats additively on the unit torus; uv = fragment position / tilePx + offset.
export const WGSL_PAINT = /* wgsl */`
struct P { K:u32, tilePx:f32, offX:f32, offY:f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Sp: array<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = fract(pos.xy / p.tilePx + vec2f(p.offX, p.offY));
  var c = vec3f(0.0);
  for (var k = 0u; k < p.K; k++) {
    let o = k * 8u;
    var d = uv - vec2f(Sp[o], Sp[o + 1u]);
    d = d - round(d);
    let u = Sp[o + 2u] * d.x;
    let v = Sp[o + 3u] * d.x + Sp[o + 4u] * d.y;
    let q = u * u + v * v;
    if (q < 30.0) { c += exp(-0.5 * q) * vec3f(Sp[o + 5u], Sp[o + 6u], Sp[o + 7u]); }
  }
  return vec4f(clamp((c + 1.0) * 0.5, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

export function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

export function gaussian(n, seed) {
  let a = seed >>> 0;
  const r = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(r(), 1e-12), v = r(), m = Math.sqrt(-2 * Math.log(u));
    out[i] = m * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = m * Math.sin(2 * Math.PI * v);
  }
  return out;
}

export class SplatDiffusion {
  static async create(device, base) {
    const man = await (await fetch(`${base}/manifest.json`)).json();
    const raw = new Uint16Array(await (await fetch(`${base}/weights.bin`)).arrayBuffer());
    return new SplatDiffusion(device, man, f16ToF32(raw));
  }

  constructor(device, man, w) {
    this.device = device;
    this.man = man;
    this.w = w;                      // torch layout, used by the JS embedding MLP
    this.K = man.K;
    this.nclass = man.nclass ?? NCLASS;
    // GPU copy with every Linear weight transposed to [in][out]
    const gw = new Float32Array(w);
    for (const [name, [o, s]] of Object.entries(man.tensors)) {
      if (s.length !== 2 || !name.endsWith(".weight") || name === "cls.weight") continue;
      const [out, inn] = s;
      for (let i = 0; i < out; i++) for (let j = 0; j < inn; j++) gw[o + j * out + i] = w[o + i * inn + j];
    }
    const U = GPUBufferUsage;
    this.wbuf = device.createBuffer({ size: gw.byteLength, usage: U.STORAGE | U.COPY_DST });
    device.queue.writeBuffer(this.wbuf, 0, gw);
    this.stats = device.createBuffer({ size: 80, usage: U.STORAGE | U.COPY_DST });
    device.queue.writeBuffer(this.stats, 0, new Float32Array([...man.stats.mean, ...man.stats.std]));
    const mk = (code) => device.createComputePipeline({ layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    this.pipe = { mm: mk(WGSL_MATMUL), sm: mk(WGSL_SOFTMAX), ln: mk(WGSL_LNMOD), step: mk(WGSL_STEP), dec: mk(WGSL_DECODE) };
    this.dummy = device.createBuffer({ size: 16, usage: U.STORAGE });
    this.N = 0;
  }

  T(name) { return this.man.tensors[name]; }
  off(name) { return this.T(name)[0]; }
  get(name) { const [o, s] = this.T(name); return this.w.subarray(o, o + s.reduce((a, b) => a * b, 1)); }

  linearJS(name, x) {
    const W = this.get(`${name}.weight`), b = this.get(`${name}.bias`);
    const [, [out, inn]] = this.T(`${name}.weight`);
    const y = new Float32Array(out);
    for (let o = 0; o < out; o++) { let s = b[o]; for (let i = 0; i < inn; i++) s += W[o * inn + i] * x[i]; y[o] = s; }
    return y;
  }

  // silu(t2(silu(t1(temb(t)))) + cls[y]), the conditioning vector every adaLN layer reads
  embed(t, y) {
    const half = TDIM / 2, te = new Float32Array(TDIM);
    for (let i = 0; i < half; i++) {
      const a = t * 1000 * Math.exp(-Math.log(10000) * i / half);
      te[i] = Math.cos(a); te[half + i] = Math.sin(a);
    }
    const silu = (v) => v.map((z) => z / (1 + Math.exp(-z)));
    const h = this.linearJS("t2", silu(this.linearJS("t1", te)));
    const ce = this.get("cls.weight").subarray(y * WIDTH, (y + 1) * WIDTH);
    return silu(h.map((v, i) => v + ce[i]));
  }

  plan(N) {
    if (this.N === N) return;
    this.free();
    const dev = this.device, U = GPUBufferUsage, K = this.K, R = N * K, d = WIDTH;
    const buf = (n) => dev.createBuffer({ size: Math.max(16, n * 4), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    const b = this.b = {
      S: buf(K * D), X0: buf(K * D), splats: buf(K * 8), V: buf(R * D), emb: buf(N * d),
      mod: buf((DEPTH * 6 + 2) * N * d), h: buf(R * d), hn: buf(R * d), qkv: buf(R * 3 * d),
      sc: buf(N * HEADS * K * K), att: buf(R * d), mlp: buf(R * 4 * d),
    };
    const ops = [];
    const uni = (arr) => {
      const ub = dev.createBuffer({ size: 256, usage: U.UNIFORM | U.COPY_DST });
      dev.queue.writeBuffer(ub, 0, arr);
      return ub;
    };
    const flat2 = (n) => (n <= 65535 ? [n, 1, 1] : [65535, Math.ceil(n / 65535), 1]);
    // generic matmul op
    const mm = (o) => {
      const u = new Uint32Array(28);
      const f = new Float32Array(u.buffer);
      const v = [o.M, o.N, o.K, o.nb2 ?? 1, o.aOff ?? 0, o.aM, o.aK, o.aB1 ?? 0, o.aB2 ?? 0,
        o.bOff ?? 0, o.bK, o.bN, o.bB1 ?? 0, o.bB2 ?? 0, o.cOff ?? 0, o.cM, o.cN, o.cB1 ?? 0, o.cB2 ?? 0,
        o.biasOff ?? 0, o.bias ? 1 : 0, o.act ?? 0, o.acc ?? 0, o.gOff ?? 0, o.gItemStride ?? 0, o.rowsPerItem ?? 1];
      v.forEach((x, i) => { u[i] = x; });
      f[26] = o.alpha ?? 1;
      const ub = uni(u);
      const bg = dev.createBindGroup({ layout: this.pipe.mm.getBindGroupLayout(0), entries: [
        ub, o.A, o.B, o.C, this.wbuf, o.G ?? this.dummy].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      ops.push({ p: this.pipe.mm, bg, wg: [Math.ceil(o.N / 64), Math.ceil(o.M / 64), o.batch ?? 1], ub });
    };
    // linear layer on activations: Y[R][out] = X[R][in] W + b (W stored [in][out] in wbuf)
    const linear = (name, X, Y, inn, out, extra = {}) => mm({ M: extra.M ?? R, N: out, K: inn,
      A: X, aM: inn, aK: 1, B: this.wbuf, bOff: this.off(`${name}.weight`), bK: out, bN: 1,
      C: Y, cM: out, cN: 1, bias: true, biasOff: this.off(`${name}.bias`), rowsPerItem: K, ...extra });
    const lnmod = (blockMod, shIdx, scIdx) => {
      const ub = uni(new Uint32Array([R, K, blockMod, this.modStride(blockMod, N), shIdx * d, scIdx * d, 0, 0]));
      const bg = dev.createBindGroup({ layout: this.pipe.ln.getBindGroupLayout(0), entries: [ub, b.h, b.mod, b.hn]
        .map((buffer, binding) => ({ binding, resource: { buffer } })) });
      ops.push({ p: this.pipe.ln, bg, wg: flat2(R), ub });
    };
    this.modOps = [];
    // modulation vectors for every block (M = N rows of the conditioning vector)
    for (let i = 0; i < DEPTH; i++) {
      linear(`blocks.${i}.ada`, b.emb, b.mod, d, 6 * d, { M: N, cOff: i * N * 6 * d, rowsPerItem: 1 });
    }
    linear("ada_out", b.emb, b.mod, d, 2 * d, { M: N, cOff: DEPTH * N * 6 * d, rowsPerItem: 1 });
    // input projection: the same state for every item (A batch stride 0 via rows: use M = K per item)
    for (let n = 0; n < N; n++) linear("inp", b.S, b.h, D, d, { M: K, cOff: n * K * d });
    const hd = d / HEADS;
    for (let i = 0; i < DEPTH; i++) {
      const mo = i * N * 6 * d;            // this block's modulation rows: [N][6d] = sh1 sc1 g1 sh2 sc2 g2
      lnmod(mo, 0, 1);
      linear(`blocks.${i}.qkv`, b.hn, b.qkv, d, 3 * d);
      // scores[item, head] = Q K^T / sqrt(hd)
      mm({ M: K, N: K, K: hd, batch: N * HEADS, nb2: HEADS,
        A: b.qkv, aM: 3 * d, aK: 1, aB1: K * 3 * d, aB2: hd,
        B: b.qkv, bOff: d, bK: 1, bN: 3 * d, bB1: K * 3 * d, bB2: hd,
        C: b.sc, cM: K, cN: 1, cB1: HEADS * K * K, cB2: K * K, alpha: 1 / Math.sqrt(hd), rowsPerItem: K });
      {
        const ub = uni(new Uint32Array([N * HEADS * K, K, 0, 0]));
        const bg = dev.createBindGroup({ layout: this.pipe.sm.getBindGroupLayout(0), entries: [ub, b.sc]
          .map((buffer, binding) => ({ binding, resource: { buffer } })) });
        ops.push({ p: this.pipe.sm, bg, wg: flat2(N * HEADS * K), ub });
      }
      // att[item, :, head] = P V
      mm({ M: K, N: hd, K: K, batch: N * HEADS, nb2: HEADS,
        A: b.sc, aM: K, aK: 1, aB1: HEADS * K * K, aB2: K * K,
        B: b.qkv, bOff: 2 * d, bK: 3 * d, bN: 1, bB1: K * 3 * d, bB2: hd,
        C: b.att, cM: d, cN: 1, cB1: K * d, cB2: hd, rowsPerItem: K });
      linear(`blocks.${i}.proj`, b.att, b.h, d, d, { acc: 1, G: b.mod, gOff: mo + 2 * d, gItemStride: 6 * d });
      lnmod(mo, 3, 4);
      linear(`blocks.${i}.fc1`, b.hn, b.mlp, d, 4 * d, { act: 1 });
      linear(`blocks.${i}.fc2`, b.mlp, b.h, 4 * d, d, { acc: 1, G: b.mod, gOff: mo + 5 * d, gItemStride: 6 * d });
    }
    lnmod(DEPTH * N * 6 * d, 0, 1);
    linear("out", b.hn, b.V, d, D);
    this.ops = ops;
    // step + decode
    this.stepU = dev.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
    this.stepBG = dev.createBindGroup({ layout: this.pipe.step.getBindGroupLayout(0), entries: [this.stepU, b.S, b.V, b.X0]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const du = uni(new Uint32Array([K, 0, 0, 0]));
    this.decX0 = dev.createBindGroup({ layout: this.pipe.dec.getBindGroupLayout(0), entries: [du, b.X0, this.stats, b.splats]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.decS = dev.createBindGroup({ layout: this.pipe.dec.getBindGroupLayout(0), entries: [du, b.S, this.stats, b.splats]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.extraU = [du];
    this.N = N;
  }

  // the stride between items' rows in the modulation buffer: [block][item][6d] (final block [item][2d])
  modStride(blockMod, N) { return blockMod === DEPTH * N * 6 * WIDTH ? 2 * WIDTH : 6 * WIDTH; }

  free() {
    if (!this.b) return;
    for (const x of Object.values(this.b)) x.destroy();
    for (const o of this.ops) o.ub.destroy();
    this.stepU.destroy();
    for (const u of this.extraU) u.destroy();
    this.b = null; this.N = 0;
  }

  encodeForward(pass) {
    for (const o of this.ops) { pass.setPipeline(o.p); pass.setBindGroup(0, o.bg); pass.dispatchWorkgroups(...o.wg); }
  }

  writeEmb(t, ys) {
    const e = new Float32Array(ys.length * WIDTH);
    ys.forEach((y, n) => e.set(this.embed(t, y), n * WIDTH));
    this.device.queue.writeBuffer(this.b.emb, 0, e);
  }

  // One model evaluation (testing): x [K*D] shared by all items.
  async forward(x, t, ys) {
    this.plan(ys.length);
    this.device.queue.writeBuffer(this.b.S, 0, x);
    this.writeEmb(t, ys);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodeForward(pass);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    return this.read(this.b.V, ys.length * this.K * D);
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

  // Euler sampler. After each step the x0 estimate is decoded into this.b.splats (drawn by onStep);
  // at the end the final state is.
  async sample({ cls, steps = 24, cfg = 3, seed = 0, noise = null, onStep = null, shouldStop = null }) {
    const N = cfg === 1 ? 1 : 2, K = this.K, dev = this.device;
    this.plan(N);
    dev.queue.writeBuffer(this.b.S, 0, noise || gaussian(K * D, seed));
    const ys = N === 2 ? [cls, this.nclass] : [cls];
    const embs = new Float32Array(steps * N * WIDTH), us = new ArrayBuffer(steps * 32);
    for (let i = 0; i < steps; i++) {
      const t = 1 - i / steps, tn = 1 - (i + 1) / steps;
      ys.forEach((y, n) => embs.set(this.embed(t, y), (i * N + n) * WIDTH));
      const dv = new DataView(us, i * 32, 32);
      dv.setUint32(0, N, true); dv.setUint32(4, K * D, true);
      dv.setFloat32(8, t, true); dv.setFloat32(12, tn - t, true); dv.setFloat32(16, cfg, true);
    }
    const U = GPUBufferUsage;
    const eAll = dev.createBuffer({ size: embs.byteLength, usage: U.COPY_SRC | U.COPY_DST });
    const uAll = dev.createBuffer({ size: us.byteLength, usage: U.COPY_SRC | U.COPY_DST });
    dev.queue.writeBuffer(eAll, 0, embs);
    dev.queue.writeBuffer(uAll, 0, us);
    let prev = null, stopped = false;
    try {
      for (let i = 0; i < steps; i++) {
        const enc = dev.createCommandEncoder();
        enc.copyBufferToBuffer(eAll, i * N * WIDTH * 4, this.b.emb, 0, N * WIDTH * 4);
        enc.copyBufferToBuffer(uAll, i * 32, this.stepU, 0, 32);
        const pass = enc.beginComputePass();
        this.encodeForward(pass);
        pass.setPipeline(this.pipe.step); pass.setBindGroup(0, this.stepBG);
        pass.dispatchWorkgroups(Math.ceil(K * D / 256));
        pass.setPipeline(this.pipe.dec); pass.setBindGroup(0, i === steps - 1 ? this.decS : this.decX0);
        pass.dispatchWorkgroups(Math.ceil(K / 64));
        pass.end();
        dev.queue.submit([enc.finish()]);
        if (onStep) await onStep(i);
        const done = dev.queue.onSubmittedWorkDone();
        if (prev) await prev;
        prev = done;
        if (shouldStop && shouldStop()) { stopped = true; break; }
      }
      if (prev) await prev;
    } finally {
      eAll.destroy(); uAll.destroy();
    }
    return !stopped;
  }

  // The current splats as plain numbers [K][8] (mu_x, mu_y, a, b, c, r, g, b)
  async splats() { return this.read(this.b.splats, this.K * 8); }
}

// Blits a cached tile texture across the canvas with a repeating sampler.
const WGSL_BLIT = /* wgsl */`
struct P { tilePx:f32, offX:f32, offY:f32, _a:f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(textureSample(tex, smp, pos.xy / p.tilePx + vec2f(p.offX, p.offY)).rgb, 1.0);
}`;

// Paints the splats. While a tile is at most MAX_TILE device pixels it is painted once into a texture
// (cost tile^2 x K, only when the splats change) and repeated with a sampler; zoomed in further, every
// pixel evaluates the splats directly so edges stay smooth. Phones paint far fewer splat-pixel pairs.
const MAX_TILE = 1024;

export class Painter {
  constructor(device, canvas, format) {
    this.device = device;
    this.format = format;
    this.ctx = canvas.getContext("webgpu");
    this.ctx.configure({ device, format, alphaMode: "opaque" });
    const mod = device.createShaderModule({ code: WGSL_PAINT });
    const mk = (fmt) => device.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" },
      fragment: { module: mod, entryPoint: "fs", targets: [{ format: fmt }] } });
    this.pipe = mk(format);
    this.tilePipe = mk("rgba8unorm");
    const bm = device.createShaderModule({ code: WGSL_BLIT });
    this.blitPipe = device.createRenderPipeline({ layout: "auto", vertex: { module: bm, entryPoint: "vs" },
      fragment: { module: bm, entryPoint: "fs", targets: [{ format }] } });
    this.sampler = device.createSampler({ addressModeU: "repeat", addressModeV: "repeat", magFilter: "linear", minFilter: "linear" });
    this.u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tu = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dirty = true;
  }

  // call whenever the splat buffer's contents change
  setSplats(splatBuf, K) { this.splatBuf = splatBuf; this.K = K; this.dirty = true; }

  uni(buf, a, b, c, d, firstU32 = false) {
    const ab = new ArrayBuffer(16), dv = new DataView(ab);
    if (firstU32) dv.setUint32(0, a, true); else dv.setFloat32(0, a, true);
    dv.setFloat32(4, b, true); dv.setFloat32(8, c, true); dv.setFloat32(12, d, true);
    this.device.queue.writeBuffer(buf, 0, ab);
  }

  draw(tilePx, offX = 0, offY = 0) {
    if (!this.splatBuf) return;
    const dev = this.device, enc = dev.createCommandEncoder();
    const view = this.ctx.getCurrentTexture().createView();
    if (tilePx <= MAX_TILE) {
      const res = Math.min(MAX_TILE, 2 ** Math.ceil(Math.log2(Math.max(64, tilePx))));
      if (!this.tex || this.tex.width !== res) {
        if (this.tex) this.tex.destroy();
        this.tex = dev.createTexture({ size: [res, res], format: "rgba8unorm",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        this.dirty = true;
        this.blitBG = dev.createBindGroup({ layout: this.blitPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: this.tu } }, { binding: 1, resource: this.tex.createView() },
          { binding: 2, resource: this.sampler }] });
      }
      if (this.dirty) {
        this.uni(this.u, this.K, res, 0, 0, true);
        const bg = dev.createBindGroup({ layout: this.tilePipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: this.u } }, { binding: 1, resource: { buffer: this.splatBuf } }] });
        const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
        tp.setPipeline(this.tilePipe); tp.setBindGroup(0, bg); tp.draw(3); tp.end();
        this.dirty = false;
      }
      this.uni(this.tu, tilePx, offX, offY, 0);
      const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(this.blitPipe); pass.setBindGroup(0, this.blitBG); pass.draw(3); pass.end();
    } else {
      this.uni(this.u, this.K, tilePx, offX, offY, true);
      const bg = dev.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.u } }, { binding: 1, resource: { buffer: this.splatBuf } }] });
      const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
      pass.setPipeline(this.pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
    }
    dev.queue.submit([enc.finish()]);
  }
}

// Paint one tile of size x size pixels into an offscreen rgba8unorm texture and return its bytes
// (tests and PNG export; a WebGPU canvas cannot be read back reliably in headless Chromium).
export async function paintToBytes(device, splatBuf, K, size) {
  const mod = device.createShaderModule({ code: WGSL_PAINT });
  const pipe = device.createRenderPipeline({ layout: "auto", vertex: { module: mod, entryPoint: "vs" },
    fragment: { module: mod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] } });
  const u = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const ab = new ArrayBuffer(16), dv = new DataView(ab);
  dv.setUint32(0, K, true); dv.setFloat32(4, size, true);
  device.queue.writeBuffer(u, 0, ab);
  const tex = device.createTexture({ size: [size, size], format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const bpr = Math.ceil(size * 4 / 256) * 256;
  const rb = device.createBuffer({ size: bpr * size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: u } }, { binding: 1, resource: { buffer: splatBuf } }] });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr }, [size, size]);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(rb.getMappedRange()), out = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) out.set(src.subarray(y * bpr, y * bpr + size * 4), y * size * 4);
  rb.unmap(); rb.destroy(); tex.destroy(); u.destroy();
  return out;
}

// A standalone WGSL fragment shader with the splat list baked in (vec4 arrays stay under WebKit's
// 2,047-element const-array limit up to 1,023 splats per array).
export function bakeShader(sp, K) {
  const f = (v) => (Number.isFinite(v) ? v.toPrecision(5) : "0.0");
  const rows = [];
  for (let k = 0; k < K; k++) {
    const o = k * 8;
    rows.push(`vec4f(${f(sp[o])},${f(sp[o + 1])},${f(sp[o + 2])},${f(sp[o + 3])}),vec4f(${f(sp[o + 4])},${f(sp[o + 5])},${f(sp[o + 6])},${f(sp[o + 7])})`);
  }
  return `// ${K} additive Gaussian splats on a torus; uv in [0,1)^2 repeats seamlessly
const S = array<vec4f, ${2 * K}>(${rows.join(",\n")});
fn texture(uv0: vec2f) -> vec3f {
  let uv = fract(uv0);
  var c = vec3f(0.0);
  for (var k = 0u; k < ${K}u; k++) {
    let p = S[2u * k]; let q = S[2u * k + 1u];
    var d = uv - p.xy; d = d - round(d);
    let u = p.z * d.x; let v = p.w * d.x + q.x * d.y;
    c += exp(-0.5 * (u * u + v * v)) * q.yzw;
  }
  return clamp((c + 1.0) * 0.5, vec3f(0.0), vec3f(1.0));
}
`;
}
