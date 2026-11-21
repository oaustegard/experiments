// Whisper on hand-written WebGPU kernels, reading a whisper-pack/1 pack (run/pack.py).
//
// Weights stay packed on the GPU: every matrix is int8 codes [N, K] (4 per u32) with one f32 scale
// per output row, and the token table is int8 with a scale per token or 4-bit with a scale per 32
// values. The table is uploaded once and serves both the embedding and the output projection.
// Activations are f32. One thread computes one output value; shapes arrive in a uniform.
//
//   const w = await Whisper.load(device, "models/base-en/");
//   const ids = await w.transcribeWindow(mel);          // mel: Float32Array [n_mels * 3000] from mel.js
//
// Steps of the forward pass are recorded once at load as (pipeline, bind group, dispatch size).
// The decoder runs one token per submit and reads the logits back; suppression and argmax are in JS.

const WG = 64;

const HEAD = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, c: vec4<u32>, f: vec4<f32> }
struct D { n: vec4<u32> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> d: D;
fn erf(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return sign(x) * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf(x * 0.7071067811865476)); }
fn i8x4(w: u32) -> vec4<f32> {
  let i = bitcast<i32>(w);
  return vec4<f32>(vec4<i32>((i << 24u) >> 24u, (i << 16u) >> 24u, (i << 8u) >> 24u, i >> 24u));
}
`;

// Y[t, n] = act(S[n] * sum_k W[n, k] X[t, k] + B[n]).
// a = (T, K, N, flags)  flags: 1 gelu, 2 add into Y, 4 output row offset += d.n.x
// b = (x row offset, y row offset, 0, 0)   c = (dispatch row width, thread count, 0, 0)
const LINEAR = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read> X: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let K = p.a.y; let N = p.a.z; let fl = p.a.w;
  let t = i / N; let n = i % N;
  let K4 = K / 4u;
  let xo = (p.b.x + t) * K;
  let wo = n * K4;
  var acc = vec4<f32>(0.0);
  for (var k = 0u; k < K4; k++) {
    let xb = xo + 4u * k;
    acc += i8x4(W[wo + k]) * vec4<f32>(X[xb], X[xb + 1u], X[xb + 2u], X[xb + 3u]);
  }
  var y = (acc.x + acc.y + acc.z + acc.w) * S[n] + B[n];
  if ((fl & 1u) != 0u) { y = gelu(y); }
  var row = p.b.y + t;
  if ((fl & 4u) != 0u) { row += d.n.x; }
  let yi = row * N + n;
  if ((fl & 2u) != 0u) { Y[yi] = Y[yi] + y; } else { Y[yi] = y; }
}`;

// Conv1d, kernel 3, padding 1, f32 weights [Cout, 3, Cin], then GELU, then optionally + POS.
// a = (Tout, Cin, Cout, stride)  b = (Tin, flags, 0, 0)  flags: 1 add POS[t, n]
const CONV = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read> POS: array<f32>;
@group(0) @binding(5) var<storage, read> X: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let Cin = p.a.y; let Cout = p.a.z; let stride = p.a.w; let Tin = p.b.x;
  let t = i / Cout; let n = i % Cout;
  var acc = vec4<f32>(0.0);
  for (var j = 0u; j < 3u; j++) {
    let ti = i32(t * stride + j) - 1;
    if (ti < 0 || ti >= i32(Tin)) { continue; }
    let xo = u32(ti) * Cin;
    let wo = (n * 3u + j) * Cin;
    for (var c = 0u; c < Cin; c += 4u) {
      acc += vec4<f32>(W[wo + c], W[wo + c + 1u], W[wo + c + 2u], W[wo + c + 3u])
           * vec4<f32>(X[xo + c], X[xo + c + 1u], X[xo + c + 2u], X[xo + c + 3u]);
    }
  }
  var y = gelu(acc.x + acc.y + acc.z + acc.w + B[n]);
  if ((p.b.y & 1u) != 0u) { y += POS[i]; } else { y += POS[0] * 0.0; }
  Y[i] = y;
  _ = d.n.x;
}`;

// LayerNorm over each row. a = (T, D, 0, 0)  b = (x row offset, y row offset, 0, 0)  f.x = eps
const NORM = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read> X: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x + g.y * p.c.x;
  if (t >= p.c.y) { return; }
  let Dm = p.a.y;
  let xo = (p.b.x + t) * Dm;
  var mu = 0.0;
  for (var k = 0u; k < Dm; k++) { mu += X[xo + k]; }
  mu = mu / f32(Dm);
  var v = 0.0;
  for (var k = 0u; k < Dm; k++) { let e = X[xo + k] - mu; v += e * e; }
  let r = inverseSqrt(v / f32(Dm) + p.f.x);
  let yo = (p.b.y + t) * Dm;
  for (var k = 0u; k < Dm; k++) { Y[yo + k] = (X[xo + k] - mu) * r * G[k] + B[k]; }
  _ = d.n.x;
}`;

// Multi-head attention, head dim 64. One thread per (query row, head): two passes over the keys
// (max score, then the softmax-weighted sum of values), so no per-key storage is needed.
// a = (Tq, Tk, H, D)  b = (flags, 0, 0, 0)  flags: 1 Tk = d.n.x + 1 (decoder self-attention over the cache)
const ATTN = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> KK: array<f32>;
@group(0) @binding(4) var<storage, read> V: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
fn score(qo: u32, ko: u32) -> f32 {
  var s = vec4<f32>(0.0);
  for (var e = 0u; e < 64u; e += 4u) {
    s += vec4<f32>(Q[qo + e], Q[qo + e + 1u], Q[qo + e + 2u], Q[qo + e + 3u])
       * vec4<f32>(KK[ko + e], KK[ko + e + 1u], KK[ko + e + 2u], KK[ko + e + 3u]);
  }
  return (s.x + s.y + s.z + s.w) * 0.125;
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let H = p.a.z; let Dm = p.a.w;
  var Tk = p.a.y;
  if ((p.b.x & 1u) != 0u) { Tk = d.n.x + 1u; }
  let t = i / H; let h = i % H;
  let qo = t * Dm + h * 64u;
  var mx = -1e30;
  for (var j = 0u; j < Tk; j++) { mx = max(mx, score(qo, j * Dm + h * 64u)); }
  var acc: array<f32, 64>;
  for (var e = 0u; e < 64u; e++) { acc[e] = 0.0; }
  var z = 0.0;
  for (var j = 0u; j < Tk; j++) {
    let ko = j * Dm + h * 64u;
    let w = exp(score(qo, ko) - mx);
    z += w;
    for (var e = 0u; e < 64u; e++) { acc[e] += w * V[ko + e]; }
  }
  for (var e = 0u; e < 64u; e++) { O[qo + e] = acc[e] / z; }
}`;

// Token embedding plus position: Y[n] = table[id, n] + POS[pos, n], id = d.n.y, pos = d.n.x.
// a = (D, bits, 0, 0). bits 8: int8 codes, TS[id]. bits 4: nibbles (low = even column), TS[id, n / 32].
const EMBED = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> TB: array<u32>;
@group(0) @binding(3) var<storage, read> TS: array<f32>;
@group(0) @binding(4) var<storage, read> POS: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x;
  let Dm = p.a.x;
  if (n >= Dm) { return; }
  let id = d.n.y;
  var v = 0.0;
  if (p.a.y == 8u) {
    let bi = id * Dm + n;
    let w = bitcast<i32>(TB[bi / 4u]);
    v = f32((w << (24u - 8u * (bi % 4u))) >> 24u) * TS[id];
  } else {
    let bi = id * (Dm / 2u) + n / 2u;
    let byte = (TB[bi / 4u] >> (8u * (bi % 4u))) & 255u;
    let nib = (byte >> (4u * (n & 1u))) & 15u;
    v = (f32(nib) - 8.0) * TS[id * (Dm / 32u) + n / 32u];
  }
  Y[n] = v + POS[d.n.x * Dm + n];
}`;

// Output projection against a 4-bit table: Y[v] = sum_g TS[v, g] * sum_{c < 32} X[32 g + c] * (code - 8).
// a = (D, V, 0, 0). The 8-bit table goes through LINEAR instead.
const LOGITS4 = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> TB: array<u32>;
@group(0) @binding(3) var<storage, read> TS: array<f32>;
@group(0) @binding(4) var<storage, read> X: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let v = g.x + g.y * p.c.x;
  if (v >= p.c.y) { return; }
  let Dm = p.a.x;
  let G = Dm / 32u;
  let wo = v * (Dm / 8u);
  var y = 0.0;
  for (var gi = 0u; gi < G; gi++) {
    var a = 0.0;
    for (var q = 0u; q < 4u; q++) {
      let w = TB[wo + gi * 4u + q];
      let xb = gi * 32u + q * 8u;
      for (var e = 0u; e < 8u; e++) { a += X[xb + e] * (f32((w >> (4u * e)) & 15u) - 8.0); }
    }
    y += a * TS[v * G + gi];
  }
  Y[v] = y;
  _ = d.n.x;
}`;

// Attention for one query row (the decoder), split so the keys run in parallel.
// ASCORE: SC[h * Tk + j] = q_h . k_{j,h} / 8, one thread per (key, head).
// AMIX:   O[h * 64 + e] = sum_j softmax(SC[h])_j V[j, h, e], one thread per output value.
// a = (Tk, H, D, flags)  flags: 1 Tk = d.n.x + 1
const ASCORE = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> KK: array<f32>;
@group(0) @binding(4) var<storage, read_write> SC: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  let H = p.a.y; let Dm = p.a.z;
  var Tk = p.a.x;
  if ((p.a.w & 1u) != 0u) { Tk = d.n.x + 1u; }
  if (i >= Tk * H) { return; }
  let j = i / H; let h = i % H;
  let qo = h * 64u; let ko = j * Dm + h * 64u;
  var s = vec4<f32>(0.0);
  for (var e = 0u; e < 64u; e += 4u) {
    s += vec4<f32>(Q[qo + e], Q[qo + e + 1u], Q[qo + e + 2u], Q[qo + e + 3u])
       * vec4<f32>(KK[ko + e], KK[ko + e + 1u], KK[ko + e + 2u], KK[ko + e + 3u]);
  }
  SC[h * Tk + j] = (s.x + s.y + s.z + s.w) * 0.125;
}`;
const AMIX = HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> SC: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let H = p.a.y; let Dm = p.a.z;
  if (i >= Dm) { return; }
  var Tk = p.a.x;
  if ((p.a.w & 1u) != 0u) { Tk = d.n.x + 1u; }
  let h = i / 64u;
  let so = h * Tk;
  var mx = -1e30;
  for (var j = 0u; j < Tk; j++) { mx = max(mx, SC[so + j]); }
  var z = 0.0;
  var acc = 0.0;
  for (var j = 0u; j < Tk; j++) {
    let w = exp(SC[so + j] - mx);
    z += w;
    acc += w * V[j * Dm + i];
  }
  O[i] = acc / z + f32(H) * 0.0;
}`;

// Encoder-sized linear: one thread per (output column, block of 8 rows), so each weight word is
// decoded once and applied to 8 rows; X is read as vec4. K, N, T and the flags are constants.
const linear8 = (K, N, T, flags) => HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let n = i % ${N}u;
  let r0 = (i / ${N}u) * 8u;
  var rr: array<u32, 8>;
  for (var t = 0u; t < 8u; t++) { rr[t] = min(r0 + t, ${T - 1}u) * ${K / 4}u; }
  var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  let wo = n * ${K / 4}u;
  for (var b = 0u; b < ${K / 4}u; b++) {
    let wv = i8x4(W[wo + b]);
    for (var t = 0u; t < 8u; t++) { acc[t] += dot(wv, X[rr[t] + b]); }
  }
  let sc = S[n]; let bias = B[n];
  for (var t = 0u; t < 8u; t++) {
    let r = r0 + t;
    if (r < ${T}u) {
      var y = acc[t] * sc + bias;
      ${flags & 1 ? "y = gelu(y);" : ""}
      let yi = r * ${N}u + n;
      ${flags & 2 ? "Y[yi] = Y[yi] + y;" : "Y[yi] = y;"}
    }
  }
  _ = d.n.x;
}`;

// Encoder self-attention: one thread per (query row, head) keeps its scores in a local array, so
// each key is scored once. T, H and D are constants; head dim 64.
const attnE = (T, H, Dm) => HEAD + /* wgsl */ `
@group(0) @binding(2) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> KK: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> O: array<vec4<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x + g.y * p.c.x;
  if (i >= p.c.y) { return; }
  let t = i / ${H}u; let h = i % ${H}u;
  let qo = t * ${Dm / 4}u + h * 16u;
  var q: array<vec4<f32>, 16>;
  for (var e = 0u; e < 16u; e++) { q[e] = Q[qo + e] * 0.125; }
  var sc: array<f32, ${T}>;
  var mx = -1e30;
  for (var j = 0u; j < ${T}u; j++) {
    let ko = j * ${Dm / 4}u + h * 16u;
    var s = 0.0;
    for (var e = 0u; e < 16u; e++) { s += dot(q[e], KK[ko + e]); }
    sc[j] = s;
    mx = max(mx, s);
  }
  var acc: array<vec4<f32>, 16>;
  for (var e = 0u; e < 16u; e++) { acc[e] = vec4<f32>(0.0); }
  var z = 0.0;
  for (var j = 0u; j < ${T}u; j++) {
    let w = exp(sc[j] - mx);
    z += w;
    let vo = j * ${Dm / 4}u + h * 16u;
    for (var e = 0u; e < 16u; e++) { acc[e] += w * V[vo + e]; }
  }
  let inv = 1.0 / z;
  for (var e = 0u; e < 16u; e++) { O[qo + e] = acc[e] * inv; }
  _ = d.n.x;
}`;

const SHADERS = { linear: LINEAR, conv: CONV, norm: NORM, attn: ATTN, ascore: ASCORE, amix: AMIX, embed: EMBED, logits4: LOGITS4 };

function halfToFloat(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    out[i] = e === 0 ? s * f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

// Nearest fp16 (ties to even) of a number, returned as a number.
function round16(x) {
  F32[0] = x;
  const bits = U32[0], sign = bits >>> 31, exp = (bits >>> 23) & 255, man = bits & 0x7fffff;
  let h;
  if (exp === 255) h = 0x7c00 | (man ? 0x200 : 0);
  else {
    const e = exp - 127 + 15;
    if (e >= 31) h = 0x7c00;
    else if (e <= 0) {
      if (e < -10) h = 0;
      else {
        const m = man | 0x800000, shift = 14 - e;
        h = m >>> shift;
        const rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
        if (rem > half || (rem === half && (h & 1))) h++;
      }
    } else {
      h = (e << 10) | (man >>> 13);
      const rem = man & 0x1fff;
      if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
    }
  }
  U16[0] = h | (sign << 15);
  return halfToFloat(U16)[0];
}
const F32 = new Float32Array(1), U32 = new Uint32Array(F32.buffer), U16 = new Uint16Array(1);

// Whisper's encoder positions: float64 sinusoids rounded to float32 and then fp16, as the checkpoints store them.
function sinusoids(length, channels) {
  const half = channels / 2, inc = Math.log(10000) / (half - 1);
  const out = new Float32Array(length * channels);
  for (let t = 0; t < length; t++) for (let k = 0; k < half; k++) {
    const a = t * Math.exp(-inc * k);
    out[t * channels + k] = round16(Math.fround(Math.sin(a)));
    out[t * channels + half + k] = round16(Math.fround(Math.cos(a)));
  }
  return out;
}

export class Whisper {
  // `base` is the pack directory URL (manifest.json, weights.bin). opts.onProgress(loadedBytes, totalBytes).
  static async load(device, base, opts = {}) {
    const manifest = await (await fetch(base + "manifest.json")).json();
    if (manifest.format !== "whisper-pack/1") throw new Error("not a whisper-pack/1 manifest");
    let bytes;
    if (manifest.shards) {   // hosted packs: gzip shards the browser inflates, fetched in order
      const total = manifest.shards.reduce((n, s) => n + s.bytes, 0);
      const all = new Uint8Array(total);
      let at = 0;
      for (const s of manifest.shards) {
        const res = await fetch(base + s.file);
        if (!res.ok) throw new Error(s.file + ": HTTP " + res.status);
        const part = new Uint8Array(await res.arrayBuffer());
        if (part.length !== s.bytes) throw new Error(`${s.file}: ${part.length} bytes, expected ${s.bytes}`);
        all.set(part, at);
        at += part.length;
        if (opts.onProgress) opts.onProgress(at, total);
      }
      bytes = all.buffer;
    } else {
      const res = await fetch(base + (manifest.weights || "weights.bin"));
      if (!res.ok) throw new Error("weights: HTTP " + res.status);
      bytes = await res.arrayBuffer();
    }
    return Whisper.fromBytes(device, manifest, bytes, opts);
  }

  // manifest: the parsed manifest.json; bytes: weights.bin as an ArrayBuffer.
  static fromBytes(device, manifest, bytes, opts = {}) {
    const w = new Whisper(device, manifest, opts);
    w._build(bytes);
    return w;
  }

  constructor(device, manifest, opts = {}) {
    this.dev = device;
    this.opts = opts;
    this.m = manifest;
    this.dims = manifest.dims;
    this.gen = manifest.generation;
    this.pipes = {};
    this.zero = {};
    this.dyn = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  _pipe(kind, code) {
    if (!this.pipes[kind]) {
      const module = this.dev.createShaderModule({ code: code || SHADERS[kind] });
      this.pipes[kind] = this.dev.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    }
    return this.pipes[kind];
  }

  _buf(floats, label) {
    return this.dev.createBuffer({ label, size: Math.max(16, floats * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  _upload(view, label) {
    const size = Math.max(16, Math.ceil(view.byteLength / 4) * 4);
    const b = this.dev.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
    b.unmap();
    return b;
  }

  _zeros(n) {
    if (!this.zero[n]) this.zero[n] = this._buf(n, "zero" + n);
    return this.zero[n];
  }

  // One recorded dispatch. threads = number of outputs; params = { a, b, f } arrays.
  _step(list, kind, params, bufs, threads, code) {
    const pipe = this._pipe(kind, code);
    const groups = Math.ceil(threads / WG);
    const gx = Math.min(groups, 16384), gy = Math.ceil(groups / gx);
    const u = new ArrayBuffer(64), u32 = new Uint32Array(u), f32 = new Float32Array(u);
    u32.set(params.a || [], 0);
    u32.set(params.b || [], 4);
    u32.set([gx * WG, threads], 8);
    f32.set(params.f || [], 12);
    const ub = this.dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    new Uint8Array(ub.getMappedRange()).set(new Uint8Array(u));
    ub.unmap();
    const entries = [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: this.dyn } },
      ...bufs.map((buffer, i) => ({ binding: i + 2, resource: { buffer } }))];
    list.push({ pipe, group: this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries }), gx, gy });
  }

  _build(bytes) {
    const { d_model: D, heads: H, n_mels: M, ffn: F, vocab: V, enc_layers, dec_layers, n_audio_ctx: TA, n_text_ctx: TT } = this.dims;
    const eps = this.dims.ln_eps;
    const raw = (r) => new Uint8Array(bytes, r.offset, r.length);
    const f32 = (r) => r.dtype === "float16" ? halfToFloat(new Uint16Array(bytes, r.offset, r.length / 2))
      : new Float32Array(bytes.slice(r.offset, r.offset + r.length));
    const up = (r, label) => this._upload(r.dtype === "float16" ? f32(r) : raw(r), label);
    const lin = (r, label) => {
      if (!r.q) throw new Error(label + ": this engine reads int8 matrices only (scheme " + this.m.scheme + ")");
      return { W: up(r.q, label + ".q"), S: up(r.s, label + ".s"), B: r.b ? up(r.b, label + ".b") : this._zeros(r.N), K: r.K, N: r.N };
    };
    const norm = (r, label) => ({ G: up(r.w, label + ".w"), B: up(r.b, label + ".b") });
    const conv = (r, label) => {
      if (r.kind === "int8") throw new Error("conv stem must be f16 or f32");
      return { W: up(r.w, label + ".w"), B: up(r.b, label + ".b") };
    };
    const b = this.b = {};
    const fast = this.opts.fast !== false;   // opts.fast = false keeps the plain kernels, for A/B checks
    const linear = (list, w, X, Y, T, flags = 0, xrow = 0, yrow = 0) => {
      if (fast && T >= 8 && !xrow && !yrow && !(flags & 4))
        return this._step(list, `linear8_${w.K}_${w.N}_${T}_${flags}`, {}, [w.W, w.S, w.B, X, Y], Math.ceil(T / 8) * w.N, linear8(w.K, w.N, T, flags));
      return this._step(list, "linear", { a: [T, w.K, w.N, flags], b: [xrow, yrow, 0, 0] }, [w.W, w.S, w.B, X, Y], T * w.N);
    };
    const ln = (list, n, X, Y, T) =>
      this._step(list, "norm", { a: [T, D, 0, 0], b: [0, 0, 0, 0], f: [eps] }, [n.G, n.B, X, Y], T);
    const attn = (list, Q, K, Vb, O, Tq, Tk, flags) =>
      this._step(list, "attn", { a: [Tq, Tk, H, D], b: [flags, 0, 0, 0] }, [Q, K, Vb, O], Tq * H);

    // ---- encoder: one step list per layer, submitted separately ----
    b.mel = this._buf(3000 * M, "mel");
    b.c1 = this._buf(3000 * D, "c1");
    b.x = this._buf(TA * D, "x");
    b.xn = this._buf(TA * D, "xn");
    b.q = this._buf(TA * D, "q");
    b.k = this._buf(TA * D, "k");
    b.v = this._buf(TA * D, "v");
    b.att = this._buf(TA * D, "att");
    b.h = this._buf(TA * F, "h");
    b.enc = this._buf(TA * D, "enc");
    const pos = this._upload(sinusoids(TA, D), "enc.pos");
    const c1 = conv(this.m.conv1, "conv1"), c2 = conv(this.m.conv2, "conv2");
    this.encSteps = [];
    let s = [];
    this._step(s, "conv", { a: [3000, M, D, 1], b: [3000, 0, 0, 0] }, [c1.W, c1.B, pos, b.mel, b.c1], 3000 * D);
    this._step(s, "conv", { a: [TA, D, D, 2], b: [3000, 1, 0, 0] }, [c2.W, c2.B, pos, b.c1, b.x], TA * D);
    this.encSteps.push(s);
    this.m.enc.layers.forEach((L, i) => {
      s = [];
      const n1 = norm(L.ln1, `e${i}.ln1`), n2 = norm(L.ln2, `e${i}.ln2`);
      ln(s, n1, b.x, b.xn, TA);
      linear(s, lin(L.q, `e${i}.q`), b.xn, b.q, TA);
      linear(s, lin(L.k, `e${i}.k`), b.xn, b.k, TA);
      linear(s, lin(L.v, `e${i}.v`), b.xn, b.v, TA);
      if (fast) this._step(s, `attnE_${TA}_${H}_${D}`, {}, [b.q, b.k, b.v, b.att], TA * H, attnE(TA, H, D));
      else attn(s, b.q, b.k, b.v, b.att, TA, TA, 0);
      linear(s, lin(L.o, `e${i}.o`), b.att, b.x, TA, 2);
      ln(s, n2, b.x, b.xn, TA);
      linear(s, lin(L.fc1, `e${i}.fc1`), b.xn, b.h, TA, 1);
      linear(s, lin(L.fc2, `e${i}.fc2`), b.h, b.x, TA, 2);
      this.encSteps.push(s);
    });
    s = [];
    ln(s, norm(this.m.enc.ln_post, "ln_post"), b.x, b.enc, TA);
    // cross-attention keys and values for every decoder layer, once per window
    const dl = this.m.dec.layers.map((L, i) => {
      const o = {};
      for (const k of ["sq", "sk", "sv", "so", "cq", "ck", "cv", "co", "fc1", "fc2"]) o[k] = lin(L[k], `d${i}.${k}`);
      for (const k of ["ln1", "ln2", "ln3"]) o[k] = norm(L[k], `d${i}.${k}`);
      o.kc = this._buf(TT * D, `d${i}.kc`); o.vc = this._buf(TT * D, `d${i}.vc`);
      o.xk = this._buf(TA * D, `d${i}.xk`); o.xv = this._buf(TA * D, `d${i}.xv`);
      linear(s, o.ck, b.enc, o.xk, TA);
      linear(s, o.cv, b.enc, o.xv, TA);
      return o;
    });
    this.encSteps.push(s);

    // ---- decoder: one token per run; position in d.n.x, token id in d.n.y ----
    const T = this.m.dec.table;
    const table = { bits: T.bits, codes: up(T.codes, "table.codes"), scales: up(T.scales, "table.scales") };
    const dpos = up(this.m.dec.pos, "dec.pos");
    b.dx = this._buf(D, "dx"); b.dn = this._buf(D, "dn"); b.dq = this._buf(D, "dq");
    b.da = this._buf(D, "da"); b.dh = this._buf(F, "dh"); b.logits = this._buf(V, "logits");
    b.sc = this._buf(H * TA, "sc");
    const attn1 = (list, K, Vb, Tk, flags) => {   // one query row: scores over the keys, then the mix
      this._step(list, "ascore", { a: [Tk, H, D, flags] }, [b.dq, K, b.sc], Math.max(Tk, TT) * H);
      this._step(list, "amix", { a: [Tk, H, D, flags] }, [b.sc, Vb, b.da], D);
    };
    const dnorm = norm(this.m.dec.ln, "dec.ln");
    s = this.decSteps = [];
    this._step(s, "embed", { a: [D, T.bits, 0, 0] }, [table.codes, table.scales, dpos, b.dx], D);
    for (const o of dl) {
      ln(s, o.ln1, b.dx, b.dn, 1);
      linear(s, o.sq, b.dn, b.dq, 1);
      linear(s, o.sk, b.dn, o.kc, 1, 4);
      linear(s, o.sv, b.dn, o.vc, 1, 4);
      attn1(s, o.kc, o.vc, 0, 1);
      linear(s, o.so, b.da, b.dx, 1, 2);
      ln(s, o.ln2, b.dx, b.dn, 1);
      linear(s, o.cq, b.dn, b.dq, 1);
      attn1(s, o.xk, o.xv, TA, 0);
      linear(s, o.co, b.da, b.dx, 1, 2);
      ln(s, o.ln3, b.dx, b.dn, 1);
      linear(s, o.fc1, b.dn, b.dh, 1, 1);
      linear(s, o.fc2, b.dh, b.dx, 1, 2);
    }
    ln(s, dnorm, b.dx, b.dn, 1);
    this.headSteps = [];
    if (T.bits === 8) {
      const w = { W: table.codes, S: table.scales, B: this._zeros(V), K: D, N: V };
      linear(this.headSteps, w, b.dn, b.logits, 1);
    } else if (T.bits === 4) {
      this._step(this.headSteps, "logits4", { a: [D, V, 0, 0] }, [table.codes, table.scales, b.dn, b.logits], V);
    } else throw new Error("token table must be 8-bit or 4-bit");
    this.read = [0, 1].map(() => this.dev.createBuffer({ size: V * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
  }

  _run(enc, steps) {
    const pass = enc.beginComputePass();
    for (const st of steps) {
      pass.setPipeline(st.pipe);
      pass.setBindGroup(0, st.group);
      pass.dispatchWorkgroups(st.gx, st.gy);
    }
    pass.end();
  }

  // mel: Float32Array [n_mels * 3000], mel-major as mel.js returns it.
  async encode(mel) {
    const M = this.dims.n_mels;
    const tm = new Float32Array(3000 * M);
    for (let c = 0; c < M; c++) for (let t = 0; t < 3000; t++) tm[t * M + c] = mel[c * 3000 + t];
    this.dev.queue.writeBuffer(this.b.mel, 0, tm);
    this.dev.queue.writeBuffer(this.dyn, 0, new Uint32Array([0, 0, 0, 0]));
    for (const steps of this.encSteps) {
      const enc = this.dev.createCommandEncoder();
      this._run(enc, steps);
      this.dev.queue.submit([enc.finish()]);
      await this.dev.queue.onSubmittedWorkDone();
    }
  }

  // Feed one token at position `pos`. Returns its logits when `want` is true.
  async step(id, pos, want = true) {
    this.dev.queue.writeBuffer(this.dyn, 0, new Uint32Array([pos, id, 0, 0]));
    const enc = this.dev.createCommandEncoder();
    this._run(enc, this.decSteps);
    if (!want) { this.dev.queue.submit([enc.finish()]); return null; }
    this._run(enc, this.headSteps);
    const rb = this.read[pos & 1];
    enc.copyBufferToBuffer(this.b.logits, 0, rb, 0, this.dims.vocab * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    return out;
  }

  // The forced prompt for a language (multilingual packs) without timestamps.
  prompt(language = "en", task = "transcribe") {
    const g = this.gen;
    if (!g.multilingual) return [g.sot, g.no_timestamps];
    const lang = g.lang_to_id[`<|${language}|>`];
    if (lang === undefined) throw new Error("unknown language " + language);
    return [g.sot, lang, g.task_to_id[task], g.no_timestamps];
  }

  // Greedy decode of the window last given to encode(). Returns generated ids, end-of-text excluded.
  async decode(promptIds, opts = {}) {
    const g = this.gen, maxNew = opts.maxNew ?? 224;
    const suppress = g.suppress_tokens || [], begin = g.begin_suppress_tokens || [];
    let pos = 0, logits = null;
    for (let i = 0; i < promptIds.length; i++) logits = await this.step(promptIds[i], pos++, i === promptIds.length - 1);
    const out = [];
    while (out.length < maxNew && pos < this.dims.n_text_ctx) {
      if (opts.onLogits) opts.onLogits(logits, out.length);
      for (const t of suppress) logits[t] = -Infinity;
      if (out.length === 0) for (const t of begin) logits[t] = -Infinity;
      let best = 0;
      for (let v = 1; v < logits.length; v++) if (logits[v] > logits[best]) best = v;
      if (best === g.eot) break;
      out.push(best);
      if (opts.onToken) opts.onToken(best);
      logits = await this.step(best, pos++);
    }
    return out;
  }

  async transcribeWindow(mel, opts = {}) {
    await this.encode(mel);
    return this.decode(opts.prompt || this.prompt(opts.language, opts.task), opts);
  }

  // Copy a buffer back, for parity checks.
  async probe(name, floats) {
    const src = this.b[name];
    const rb = this.dev.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, rb, 0, floats * 4);
    this.dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy();
    return out;
  }
}
