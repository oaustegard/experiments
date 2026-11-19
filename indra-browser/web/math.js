// Indra's math reader (indra-ocr math.py) on the GPU: a line classifier finds lines that hold
// math, and an encoder-decoder reads them as text with inline LaTeX ("we have $x _ { i } \leq 1$").
//
// The classifier (mathcls) is a ConvNet from nets.js. The encoder follows math_enc.onnx: three
// stride-2 convs with GELU, a fold of the 8 rows left into one sequence, batch norm and GELU,
// six blocks like the recognizer's (depthwise 9, expand, squeeze-excite, project), two
// pre-norm attention layers whose scores fall off with distance (ALiBi, a slope per head), and a
// sinusoidal position code added at the end. The decoder follows math_dec_init / math_dec_step:
// a 3-layer pre-norm transformer decoder (PyTorch's, with GELU), greedy, one token per step, its
// keys and values cached on the GPU. Steps run in chunks of one pass each; only every step's
// argmax and its softmax probability are read back, and the stopping rules of math.py
// (end token, or a mean probability under 0.8 after 8 tokens) run here on the CPU.

import { MM, SEQ, registerKernel } from "./gpu.js";
import { conv, Weights } from "./nets.js";

export const MATH_MIN = 0.5; // classifier probability above which a line is read as math
export const MATH_MIN_CONF = 0.9; // a reading replaces the line at this mean token probability or more
const EARLY_STOP_MARGIN = 0.1;
const BOS = 1, EOS = 2, N_SPECIAL = 4;
const D = 192, HEADS = 8, HD = 24;
const SCALE = Math.fround(24 ** -0.5);
const CHUNK = 16; // decoder steps per pass (one readback each)

const GELU = /* wgsl */ `
fn gelu(x: f32) -> f32 {
  let z = x / sqrt(2.0);
  let a = abs(z); let t = 1.0 / (1.0 + 0.3275911 * a);
  let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return 0.5 * x * (1.0 + select(e, -e, z < 0.0));
}`;

// layer norm over rows of D: (x - mean) / sqrt(var + 1e-5) * w + b, one thread per row
const LN = /* wgsl */ `
struct P { N: u32, D: u32, nx: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = g.x + g.y * p.nx * 64u;
  if (n >= p.N) { return; }
  let o = n * p.D;
  var m = 0.0;
  for (var d = 0u; d < p.D; d++) { m += x[o + d]; }
  m = m / f32(p.D);
  var v = 0.0;
  for (var d = 0u; d < p.D; d++) { let e = x[o + d] - m; v += e * e; }
  let r = 1.0 / sqrt(v / f32(p.D) + 1e-5);
  for (var d = 0u; d < p.D; d++) { y[o + d] = (x[o + d] - m) * r * w[d] + b[d]; }
}`;
registerKernel(LN, "math.ln");

// encoder self-attention, 8 heads of 24, one thread per (line, position, head):
// score(i, j) = q_i.k_j / sqrt(24) - slope_h |i - j| + (mask_j - 1) 1e4, slope_h = 2^-(h+1).
// qkv [B, T, 576] (q | k | v, head h at h * 24) -> o [B, T, 192]
const EATTN = /* wgsl */ `
struct P { B: u32, T: u32, nx: u32, pad: u32, sc: f32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.B * p.T * 8u) { return; }
  let h = idx % 8u; let bt = idx / 8u; let b = bt / p.T; let i = bt % p.T;
  let slope = 0.5 / f32(1u << h);
  var q: array<f32, 24>;
  for (var d = 0u; d < 24u; d++) { q[d] = qkv[bt * 576u + h * 24u + d]; }
  var mx = -3.4e38;
  for (var j = 0u; j < p.T; j++) {
    let ko = (b * p.T + j) * 576u + 192u + h * 24u;
    var s = 0.0;
    for (var d = 0u; d < 24u; d++) { s += q[d] * qkv[ko + d]; }
    s = s * p.sc - slope * abs(f32(i) - f32(j)) + (m[b * p.T + j] - 1.0) * 10000.0;
    mx = max(mx, s);
  }
  var acc: array<f32, 24>;
  for (var d = 0u; d < 24u; d++) { acc[d] = 0.0; }
  var den = 0.0;
  for (var j = 0u; j < p.T; j++) {
    let ko = (b * p.T + j) * 576u + 192u + h * 24u;
    var s = 0.0;
    for (var d = 0u; d < 24u; d++) { s += q[d] * qkv[ko + d]; }
    s = s * p.sc - slope * abs(f32(i) - f32(j)) + (m[b * p.T + j] - 1.0) * 10000.0;
    let e = exp(s - mx);
    den += e;
    for (var d = 0u; d < 24u; d++) { acc[d] += e * qkv[ko + 192u + d]; }
  }
  for (var d = 0u; d < 24u; d++) { o[bt * 192u + h * 24u + d] = acc[d] / den; }
}`;
registerKernel(EATTN, "math.eattn");

// y[b, t, c] = x[b, t, c] + pe[t, c] (the encoder's position code)
const ADDPE = /* wgsl */ `
struct P { total: u32, T: u32, nx: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> pe: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.total) { return; }
  let c = idx % 192u; let t = (idx / 192u) % p.T;
  y[idx] = x[idx] + pe[t * 192u + c];
}`;
registerKernel(ADDPE, "math.pe");

// decoder input: x[b] = emb[token[b]] + pos[t]; tokens are floats (ids < 2^24)
const EMB = /* wgsl */ `
struct P { B: u32, t: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> tok: array<f32>;
@group(0) @binding(2) var<storage, read> emb: array<f32>;
@group(0) @binding(3) var<storage, read> pos: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.B * 192u) { return; }
  let b = idx / 192u; let c = idx % 192u;
  y[idx] = emb[u32(tok[b]) * 192u + c] + pos[p.t * 192u + c];
}`;
registerKernel(EMB, "math.emb");

// decoder self-attention at step t, one thread per (line, head): the cached keys and values of
// steps 0..t-1 and this step's (from qkv [B, 576]); this step's are then written to the cache
// [B, maxT, 192].
const SATTN = /* wgsl */ `
struct P { B: u32, t: u32, maxT: u32, pad: u32, sc: f32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.B * 8u) { return; }
  let b = idx / 8u; let h = idx % 8u;
  var q: array<f32, 24>; var kn: array<f32, 24>; var vn: array<f32, 24>;
  for (var d = 0u; d < 24u; d++) {
    q[d] = qkv[b * 576u + h * 24u + d];
    kn[d] = qkv[b * 576u + 192u + h * 24u + d];
    vn[d] = qkv[b * 576u + 384u + h * 24u + d];
  }
  var sn = 0.0;
  for (var d = 0u; d < 24u; d++) { sn += q[d] * kn[d]; }
  sn = sn * p.sc;
  var mx = sn;
  for (var j = 0u; j < p.t; j++) {
    let ko = (b * p.maxT + j) * 192u + h * 24u;
    var s = 0.0;
    for (var d = 0u; d < 24u; d++) { s += q[d] * kc[ko + d]; }
    mx = max(mx, s * p.sc);
  }
  var acc: array<f32, 24>;
  for (var d = 0u; d < 24u; d++) { acc[d] = 0.0; }
  var den = 0.0;
  for (var j = 0u; j < p.t; j++) {
    let ko = (b * p.maxT + j) * 192u + h * 24u;
    var s = 0.0;
    for (var d = 0u; d < 24u; d++) { s += q[d] * kc[ko + d]; }
    let e = exp(s * p.sc - mx);
    den += e;
    for (var d = 0u; d < 24u; d++) { acc[d] += e * vc[ko + d]; }
  }
  let en = exp(sn - mx);
  den += en;
  for (var d = 0u; d < 24u; d++) { acc[d] += en * vn[d]; }
  for (var d = 0u; d < 24u; d++) { o[b * 192u + h * 24u + d] = acc[d] / den; }
  let ko = (b * p.maxT + p.t) * 192u + h * 24u;
  for (var d = 0u; d < 24u; d++) { kc[ko + d] = kn[d]; vc[ko + d] = vn[d]; }
}`;
registerKernel(SATTN, "math.sattn");

// decoder cross-attention over the encoder's memory, one thread per (line, head):
// q [B, 192], kv [B, T, 384] (k | v), mask [B, T] -> o [B, 192]; masked keys get -1e9
const XATTN = /* wgsl */ `
struct P { B: u32, T: u32, pad0: u32, pad1: u32, sc: f32, pad2: u32, pad3: u32, pad4: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.B * 8u) { return; }
  let b = idx / 8u; let h = idx % 8u;
  var qq: array<f32, 24>;
  for (var d = 0u; d < 24u; d++) { qq[d] = q[b * 192u + h * 24u + d]; }
  var mx = -3.4e38;
  for (var j = 0u; j < p.T; j++) {
    let ko = (b * p.T + j) * 384u + h * 24u;
    var s = 0.0;
    for (var d = 0u; d < 24u; d++) { s += qq[d] * kv[ko + d]; }
    mx = max(mx, s * p.sc + (m[b * p.T + j] - 1.0) * 1e9);
  }
  var acc: array<f32, 24>;
  for (var d = 0u; d < 24u; d++) { acc[d] = 0.0; }
  var den = 0.0;
  for (var j = 0u; j < p.T; j++) {
    let ko = (b * p.T + j) * 384u + h * 24u;
    var s = 0.0;
    for (var d = 0u; d < 24u; d++) { s += qq[d] * kv[ko + d]; }
    let e = exp(s * p.sc + (m[b * p.T + j] - 1.0) * 1e9 - mx);
    den += e;
    for (var d = 0u; d < 24u; d++) { acc[d] += e * kv[ko + 192u + d]; }
  }
  for (var d = 0u; d < 24u; d++) { o[b * 192u + h * 24u + d] = acc[d] / den; }
}`;
registerKernel(XATTN, "math.xattn");

// greedy step: per line the argmax of the logits [B, V] (first of equals) and its softmax
// probability 1 / sum exp(l - max); the argmax is the next step's token and both are kept
// for the readback at slot s of the chunk
const ARGMAX = /* wgsl */ `
struct P { B: u32, V: u32, s: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> l: array<f32>;
@group(0) @binding(2) var<storage, read_write> tok: array<f32>;
@group(0) @binding(3) var<storage, read_write> toks: array<f32>;
@group(0) @binding(4) var<storage, read_write> tops: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let b = g.x;
  if (b >= p.B) { return; }
  let o = b * p.V;
  var mx = l[o]; var am = 0u;
  for (var v = 1u; v < p.V; v++) { if (l[o + v] > mx) { mx = l[o + v]; am = v; } }
  var sum = 0.0;
  for (var v = 0u; v < p.V; v++) { sum += exp(l[o + v] - mx); }
  tok[b] = f32(am);
  toks[p.s * p.B + b] = f32(am);
  tops[p.s * p.B + b] = 1.0 / sum;
}`;
registerKernel(ARGMAX, "math.argmax");

// W [K, N] (x @ W, ONNX MatMul) -> [N, K], the layout MM reads
function transpose(W, K, N) {
  const out = new Float32Array(N * K);
  for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) out[n * K + k] = W[k * N + n];
  return out;
}

// Python's str.split() whitespace
const WS_CODES = [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
  0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000];
const PY_WS = new RegExp(`[${WS_CODES.map((c) => `\\u{${c.toString(16)}}`).join("")}]+`, "u");

// tokens -> text; inside math (between "$" tokens) they are joined by spaces (math.py detokenize)
export function detokenize(tokens) {
  const out = [];
  let math = [], inMath = false;
  for (const t of tokens) {
    if (t === "$") {
      if (inMath) { out.push("$" + math.join(" ") + "$"); math = []; }
      inMath = !inMath;
    } else if (inMath) math.push(t);
    else out.push(t);
  }
  if (inMath && math.length) out.push("$" + math.join(" ") + "$");
  return out.join("").split(PY_WS).filter(Boolean).join(" ");
}

// the encoder's position code [T, 192]: sin at even channels, cos at odd, of t / 10000^(2k/192),
// in float32 as the ONNX graph computes it
function positionCode(T) {
  const pe = new Float32Array(T * D);
  const div = Float32Array.from({ length: D / 2 }, (_, k) => Math.fround(10000 ** Math.fround((2 * k) / D)));
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < D / 2; k++) {
      const a = Math.fround(t / div[k]);
      pe[t * D + 2 * k] = Math.sin(a);
      pe[t * D + 2 * k + 1] = Math.cos(a);
    }
  }
  return pe;
}

export class MathNet {
  // model: {entry, tensors} of math.bin (pack.js loadModel)
  constructor(gpu, model) {
    this.gpu = gpu;
    const W = model.tensors;
    this.W = W;
    const cfg = model.entry.config;
    this.height = cfg.height;
    this.maxTokens = cfg.max_tokens;
    this.layers = cfg.dec_layers;
    this.V = cfg.n_tokens;
    this.itos = ["", "", "", "", ...model.entry.tokens];
    const shapes = Object.fromEntries(Object.entries(model.entry.tensors).map(([k, r]) => [k, r.shape]));
    this.w = new Weights(gpu, W, shapes);
    const up = (a) => gpu.upload(a);
    const lin = (w, b, K, N) => ({ w: up(transpose(W[w], K, N)), b: up(W[b]), K, N });
    // fold batch norm: y = x * s + t
    const g = W["enc.fold_bn.weight"], bb = W["enc.fold_bn.bias"], mu = W["enc.fold_bn.running_mean"], vr = W["enc.fold_bn.running_var"];
    const s = new Float32Array(D), t = new Float32Array(D);
    for (let c = 0; c < D; c++) { s[c] = g[c] / Math.sqrt(vr[c] + 1e-5); t[c] = bb[c] - mu[c] * s[c]; }
    this.bnS = up(s);
    this.bnT = up(t);
    this.attn = [0, 1].map((i) => {
      const A = `enc.attn.${i}`;
      return {
        ln1: [up(W[`${A}.ln1.weight`]), up(W[`${A}.ln1.bias`])],
        qkv: lin(`${A}.qkv.weight`, `${A}.qkv.bias`, D, 3 * D),
        out: lin(`${A}.out.weight`, `${A}.out.bias`, D, D),
        ln2: [up(W[`${A}.ln2.weight`]), up(W[`${A}.ln2.bias`])],
        ff1: lin(`${A}.ff1.weight`, `${A}.ff1.bias`, D, 4 * D),
        ff2: lin(`${A}.ff2.weight`, `${A}.ff2.bias`, 4 * D, D),
      };
    });
    this.dec = Array.from({ length: this.layers }, (_, i) => {
      const L = `dec.layers.layers.${i}`;
      const ip = W[`${L}.multihead_attn.in_proj_weight`], ib = W[`${L}.multihead_attn.in_proj_bias`];
      return {
        n1: [up(W[`${L}.norm1.weight`]), up(W[`${L}.norm1.bias`])],
        n2: [up(W[`${L}.norm2.weight`]), up(W[`${L}.norm2.bias`])],
        n3: [up(W[`${L}.norm3.weight`]), up(W[`${L}.norm3.bias`])],
        sin: lin(`${L}.self_attn.in_proj_weight`, `${L}.self_attn.in_proj_bias`, D, 3 * D),
        sout: lin(`${L}.self_attn.out_proj.weight`, `${L}.self_attn.out_proj.bias`, D, D),
        // the cross-attention's in_proj [576, 192] is [out, in] already: q rows, then k and v rows
        q: { w: up(ip.slice(0, D * D)), b: up(ib.slice(0, D)), K: D, N: D },
        kv: { w: up(ip.slice(D * D)), b: up(ib.slice(D)), K: D, N: 2 * D },
        xout: lin(`${L}.multihead_attn.out_proj.weight`, `${L}.multihead_attn.out_proj.bias`, D, D),
        l1: lin(`${L}.linear1.weight`, `${L}.linear1.bias`, D, 4 * D),
        l2: lin(`${L}.linear2.weight`, `${L}.linear2.bias`, 4 * D, D),
      };
    });
    this.norm = [up(W["dec.norm.weight"]), up(W["dec.norm.bias"])];
    this.out = lin("dec.out.weight", "dec.out.bias", D, this.V);
    this.emb = up(W["dec.emb.weight"]);
    this.pos = up(W["dec.pos.weight"]);
    this.pe = new Map();
  }

  mm(x, R, l, { act = 0, res = null, mask = null, out = null } = {}) {
    const y = out ?? this.gpu.buffer(4 * R * l.N);
    this.gpu.dispatch(MM, [x, l.w, l.b, res, mask, y], [R, l.K, l.N, act, res ? 1 : 0, mask ? 1 : 0, l.N, 0],
      [Math.ceil(l.N / 64), Math.ceil(R / 64)], "math.mm");
    return y;
  }

  ln(x, R, [w, b], out = null) {
    const y = out ?? this.gpu.buffer(4 * R * D);
    const grid = this.gpu.grid(R);
    this.gpu.dispatch(LN, [x, w, b, y], [R, D, grid[0], 0], grid);
    return y;
  }

  posCode(T) {
    if (!this.pe.has(T)) this.pe.set(T, this.gpu.upload(positionCode(T)));
    return this.pe.get(T);
  }

  // Record the encoder for one batch into the open pass. x: Float32Array [B, 64, W] or a
  // recorder (crops.js). Returns {T, mask (Float32Array), mbuf, kv: [3 buffers [B, T, 384]],
  // input}; mbuf, kv and input (an uploaded x) are not pooled: the caller destroys them.
  encode(x, B, H, Wd, widths) {
    const gpu = this.gpu, w = this.w;
    const free = (...ts) => { for (const t of ts) gpu.free(t.buf ?? t); };
    const onGpu = typeof x === "function";
    const xin = { buf: onGpu ? x(gpu) : gpu.upload(x), N: B, H, W: Wd, C: 1 };
    const c1 = conv(gpu, xin, w.conv("enc.front.front.0.Conv.weight"), w.vec("enc.front.front.0.Conv.bias"), { s: [2, 2], p: [1, 1, 1, 1], act: 4 });
    const c2 = conv(gpu, c1, w.conv("enc.front.front.3.Conv.weight"), w.vec("enc.front.front.3.Conv.bias"), { s: [2, 2], p: [1, 1, 1, 1], act: 4 });
    free(c1);
    const c3 = conv(gpu, c2, w.conv("enc.front.front.6.Conv.weight"), w.vec("enc.front.front.6.Conv.bias"), { s: [2, 2], p: [1, 1, 1, 1], act: 4 });
    free(c2);
    const f = conv(gpu, c3, w.conv("enc.fold.weight", 8, 1), w.vec("enc.fold.bias"), {}); // [B, 1, T, 192]
    free(c3);
    const T = f.W;
    const mask = new Float32Array(B * T);
    for (let b = 0; b < B; b++) {
      const n = Math.min(Math.max(Math.floor((widths[b] + 7) / 8), 1), T);
      for (let t = 0; t < n; t++) mask[b * T + t] = 1;
    }
    const mbuf = keep(gpu, B * T); // the decoder reads it in later passes
    gpu.device.queue.writeBuffer(mbuf, 0, mask);
    let h = this.seq(0, f, this.bnS, this.bnT, mbuf, B, T, D, 4);
    free(f);
    for (let i = 0; i < 6; i++) {
      const p = `enc.blocks.${i}`;
      const y0 = conv(gpu, h, w.conv(`${p}.dw.Conv.weight`, 1, 9), w.vec(`${p}.dw.Conv.bias`), { p: [0, 4, 0, 4], g: D });
      const y1 = conv(gpu, y0, w.conv(`${p}.expand.weight`, 1, 1), w.vec(`${p}.expand.bias`), { act: 1 }); // [B, 1, T, 384]
      free(y0);
      const pooled = this.seq(1, y1, null, null, mbuf, B, T, 2 * D, 0, B * 2 * D);
      const pv = { buf: pooled.buf, N: B, H: 1, W: 1, C: 2 * D };
      const s1 = conv(gpu, pv, w.conv(`${p}.se.fc1.weight`, 1, 1), w.vec(`${p}.se.fc1.bias`), { act: 2 });
      const s2 = conv(gpu, s1, w.conv(`${p}.se.fc2.weight`, 1, 1), w.vec(`${p}.se.fc2.bias`), { act: 3 });
      free(pooled, s1);
      const y2 = this.seq(2, y1, s2.buf, null, null, B, T, 2 * D, 0);
      free(y1, s2);
      const hn = conv(gpu, y2, w.conv(`${p}.project.weight`, 1, 1), w.vec(`${p}.project.bias`), { res: h, resAt: 1, mask: mbuf });
      free(y2, h);
      h = hn;
    }
    const R = B * T;
    let x0 = h.buf;
    for (const A of this.attn) {
      const a = this.ln(x0, R, A.ln1);
      const qkv = this.mm(a, R, A.qkv);
      gpu.free(a);
      const o = gpu.buffer(4 * R * D);
      const grid = gpu.grid(R * HEADS);
      gpu.dispatch(EATTN, [qkv, mbuf, o], [B, T, grid[0], 0, { f: SCALE }, 0, 0, 0], grid);
      gpu.free(qkv);
      const x1 = this.mm(o, R, A.out, { res: x0 });
      gpu.free(o);
      gpu.free(x0);
      const a2 = this.ln(x1, R, A.ln2);
      const ff = this.mm(a2, R, A.ff1, { act: 4 });
      gpu.free(a2);
      x0 = this.mm(ff, R, A.ff2, { res: x1, mask: mbuf });
      gpu.free(ff);
      gpu.free(x1);
    }
    const mem = gpu.buffer(4 * R * D);
    const grid = gpu.grid(R * D);
    gpu.dispatch(ADDPE, [x0, this.posCode(T), mem], [R * D, T, grid[0], 0], grid);
    gpu.free(x0);
    const kv = this.dec.map((L) => this.mm(mem, R, L.kv, { out: keep(gpu, R * 2 * D) }));
    gpu.free(mem);
    return { T, mask, mbuf, kv, input: onGpu ? null : xin.buf };
  }

  seq(kind, x, s, t, m, B, T, C, act, total = B * T * C) {
    const gpu = this.gpu;
    const y = { buf: gpu.buffer(4 * total), N: B, H: 1, W: kind === 1 ? 1 : T, C };
    const grid = gpu.grid(total);
    gpu.dispatch(SEQ, [x.buf, s, t, m, y.buf], [kind, B, T, C, total, grid[0], act, 0], grid);
    return y;
  }

  // One batch: {data, B, H, W, widths} -> [[text, mean token probability], ...] per line, as
  // MathReader.read reads it.
  async readBatch(batch, rt = null) {
    const gpu = this.gpu;
    const { B, W: Wd, widths } = batch;
    let t0 = performance.now();
    gpu.scope = "math";
    gpu.begin();
    const enc = this.encode(batch.data, B, batch.H, Wd, widths);
    await gpu.end([]);
    if (rt) rt.math_enc_ms = (rt.math_enc_ms ?? 0) + performance.now() - t0;
    t0 = performance.now();
    const T = enc.T, maxT = this.maxTokens;
    const mk = (n) => keep(gpu, n);
    const kc = this.dec.map(() => mk(B * maxT * D)), vc = this.dec.map(() => mk(B * maxT * D));
    const tok = mk(B), toks = mk(CHUNK * B), tops = mk(CHUNK * B);
    gpu.device.queue.writeBuffer(tok, 0, new Float32Array(B).fill(BOS));
    const xa = mk(B * D), xb = mk(B * D), hb = mk(B * D), qkv = mk(B * 3 * D), ob = mk(B * D), qb = mk(B * D), fb = mk(B * 4 * D), lg = mk(B * this.V);
    const seqs = Array.from({ length: B }, () => []), probs = Array.from({ length: B }, () => []);
    const done = new Array(B).fill(false);
    let steps = 0;
    for (let t0s = 0; t0s < maxT - 1; t0s += CHUNK) {
      const n = Math.min(CHUNK, maxT - 1 - t0s);
      gpu.begin();
      for (let s = 0; s < n; s++) this.step(t0s + s, s, B, T, enc, { kc, vc, tok, toks, tops, xa, xb, hb, qkv, ob, qb, fb, lg });
      const [rt1, rt2] = await gpu.end([[toks, 4 * n * B], [tops, 4 * n * B]]);
      const tk = new Float32Array(rt1), tp = new Float32Array(rt2);
      let all = false;
      for (let s = 0; s < n && !all; s++) {
        steps++;
        for (let b = 0; b < B; b++) {
          if (done[b]) continue;
          const id = tk[s * B + b];
          probs[b].push(tp[s * B + b]);
          if (id === EOS) done[b] = true;
          else if (probs[b].length >= 8 && mean(probs[b]) < MATH_MIN_CONF - EARLY_STOP_MARGIN) done[b] = true;
          else if (id >= N_SPECIAL) seqs[b].push(id);
        }
        all = done.every(Boolean);
      }
      if (all) break;
    }
    for (const b of [...kc, ...vc, tok, toks, tops, xa, xb, hb, qkv, ob, qb, fb, lg, ...enc.kv, enc.mbuf, enc.input]) b?.destroy();
    if (rt) {
      rt.math_dec_ms = (rt.math_dec_ms ?? 0) + performance.now() - t0;
      rt.math_steps = (rt.math_steps ?? 0) + steps;
    }
    return seqs.map((sq, b) => [detokenize(sq.map((i) => this.itos[i])), probs[b].length ? mean(probs[b]) : 0.0]);
  }

  // For web/test/check_math.mjs: the decoder's cross keys | values per layer [B, T, 384], and
  // over nSteps greedy steps every step's argmax and probability and the logits of steps `at`.
  async probe(batch, nSteps, at) {
    const gpu = this.gpu;
    const { B, W: Wd, widths } = batch;
    gpu.begin();
    const enc = this.encode(batch.data, B, batch.H, Wd, widths);
    const T = enc.T;
    const kvs = (await gpu.end(enc.kv.map((b) => [b, 4 * B * T * 2 * D]))).map((a) => new Float32Array(a));
    const mk = (n) => keep(gpu, n);
    const kc = this.dec.map(() => mk(B * this.maxTokens * D)), vc = this.dec.map(() => mk(B * this.maxTokens * D));
    const buf = { kc, vc, tok: mk(B), toks: mk(B), tops: mk(B), xa: mk(B * D), xb: mk(B * D), hb: mk(B * D), qkv: mk(B * 3 * D),
      ob: mk(B * D), qb: mk(B * D), fb: mk(B * 4 * D), lg: mk(B * this.V) };
    gpu.device.queue.writeBuffer(buf.tok, 0, new Float32Array(B).fill(BOS));
    const toks = [], tops = [], logits = {};
    for (let t = 0; t < nSteps; t++) {
      gpu.begin();
      this.step(t, 0, B, T, enc, buf);
      const r = await gpu.end([[buf.toks, 4 * B], [buf.tops, 4 * B], [buf.lg, 4 * B * this.V]]);
      toks.push(Array.from(new Float32Array(r[0])));
      tops.push(Array.from(new Float32Array(r[1])));
      if (at.includes(t)) logits[t] = new Float32Array(r[2]);
    }
    for (const b of [...Object.values(buf).flat(), ...enc.kv, enc.mbuf, enc.input]) b?.destroy();
    return { T, mask: enc.mask, kv: kvs, toks, tops, logits };
  }

  // record one greedy step at position t (slot s of the chunk)
  step(t, s, B, T, enc, buf) {
    const gpu = this.gpu;
    gpu.dispatch(EMB, [buf.tok, this.emb, this.pos, buf.xa], [B, t, 0, 0], [Math.ceil((B * D) / 64)]);
    let x = buf.xa, alt = buf.xb;
    const heads = [Math.ceil((B * HEADS) / 64)];
    this.dec.forEach((L, l) => {
      this.ln(x, B, L.n1, buf.hb);
      this.mm(buf.hb, B, L.sin, { out: buf.qkv });
      gpu.dispatch(SATTN, [buf.qkv, buf.kc[l], buf.vc[l], buf.ob], [B, t, this.maxTokens, 0, { f: SCALE }, 0, 0, 0], heads);
      this.mm(buf.ob, B, L.sout, { res: x, out: alt });
      [x, alt] = [alt, x];
      this.ln(x, B, L.n2, buf.hb);
      this.mm(buf.hb, B, L.q, { out: buf.qb });
      gpu.dispatch(XATTN, [buf.qb, enc.kv[l], enc.mbuf, buf.ob], [B, T, 0, 0, { f: SCALE }, 0, 0, 0], heads);
      this.mm(buf.ob, B, L.xout, { res: x, out: alt });
      [x, alt] = [alt, x];
      this.ln(x, B, L.n3, buf.hb);
      this.mm(buf.hb, B, L.l1, { act: 4, out: buf.fb });
      this.mm(buf.fb, B, L.l2, { res: x, out: alt });
      [x, alt] = [alt, x];
    });
    this.ln(x, B, this.norm, buf.hb);
    this.mm(buf.hb, B, this.out, { out: buf.lg });
    gpu.dispatch(ARGMAX, [buf.lg, buf.tok, buf.toks, buf.tops], [B, this.V, s, 0], [Math.ceil(B / 64)]);
  }
}

// a storage buffer of n floats outside the pass's pool (it outlives the pass)
function keep(gpu, n) {
  return gpu.device.createBuffer({ size: Math.max(16, 4 * n), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
}

function mean(a) {
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}
