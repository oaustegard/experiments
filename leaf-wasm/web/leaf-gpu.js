// leaf-gpu.js: mdbr-leaf-ir forward pass in WebGPU compute shaders (WGSL). No onnxruntime-web, no
// wasm kernel. Same int8 weights file as leaf.js (run/export_weights.py, int8: per-column matmul
// scales, per-row token-table scales) and the same tokenizer.js. leaf.js stays the reference.
//
//   const gpu = await createLeafGPU({ weights, vocab });          // ArrayBuffer (int8 file), string
//   const [v] = await gpu.embedBatch([text], { query: true });   // Float32Array(768), unit length
//   const vs = await gpu.forwardBatch(idsList);                   // token-id arrays in, vectors out
//
// A batch is B sequences padded to one length S (the longest in the batch), rows R = B*S. Per
// batch, one command encoder runs 38 dispatches (embed, 6 x [qkv, attention, out-proj, add+LN,
// up+GELU, down, add+LN], pool) and one readback copies [B, 768]. embedBatch sorts texts by
// token count and fills batches up to maxRows, so similar lengths share a batch.
//
// The int8 bytes are uploaded untouched: four consecutive int8 values per u32, little-endian,
// read with extractBits (sign-extending). Matmuls accumulate x * q in f32 and apply the column
// scale and bias once in the epilogue: y[r,n] = s[n] * sum_k x[r,k] q[k,n] + b[n].

import { WordPiece } from "./tokenizer.js";
import { QUERY_PROMPT } from "./leaf.js";

const MAX_TOKENS = 128;

function parse(buf) {
  const dv = new DataView(buf);
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== "LEAF") throw new Error("not a LEAF weights file");
  const hlen = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, hlen)));
  const base = Math.ceil((12 + hlen) / 16) * 16;
  const t = header.tensors;
  const bytes = (name) => {
    const x = t[name], n = x.shape.reduce((a, b) => a * b, 1);
    if (x.dtype === "i4") throw new Error("leaf-gpu reads int8 tables only (export without t4)");
    return new Uint8Array(buf, base + x.offset, x.dtype === "f32" ? n * 4 : n);
  };
  const scale = (name) => new Uint8Array(buf, base + t[name].scale, t[name].shape[t[name].axis] * 4);
  for (const name of Object.keys(t))
    if (/_w$|^tok_emb$/.test(name) && t[name].dtype !== "i8") throw new Error(`${name}: leaf-gpu needs the int8 file`);
  return { config: header.config, bytes, scale };
}

// ------------------------------------------------------------------------------------------ WGSL

const DIMS = /* wgsl */ `
struct Dims { R: u32, S: u32, B: u32, pad: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
`;

// Workgroup tree reduction over 128 lanes. Every invocation must call it.
const REDUCE128 = /* wgsl */ `
var<workgroup> red: array<f32, 128>;
fn reduce_sum(tid: u32, v: f32) -> f32 {
  red[tid] = v;
  workgroupBarrier();
  for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
    if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn i8_at(w: u32, j: u32) -> f32 { return f32(extractBits(bitcast<i32>(w), j * 8u, 8u)); }
`;

// One workgroup per row: x[r] = LayerNorm(table[id] * scale[id] + pos[s]). H = 384 = 3 x 128 lanes.
const EMBED = DIMS + REDUCE128 + /* wgsl */ `
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> tab: array<u32>;      // [V][H] int8, 4 per u32
@group(0) @binding(3) var<storage, read> tsc: array<f32>;      // [V]
@group(0) @binding(4) var<storage, read> pos: array<f32>;      // [512][H]
@group(0) @binding(5) var<storage, read> g: array<f32>;
@group(0) @binding(6) var<storage, read> b: array<f32>;
@group(0) @binding(7) var<storage, read_write> x: array<f32>;  // [R][H]
const H: u32 = 384u;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let r = wg.x;
  let s = r % dims.S;
  let id = ids[r];
  var v: array<f32, 3>;
  var sum = 0.0;
  for (var i = 0u; i < 3u; i++) {
    let j = tid + i * 128u;
    let e = id * H + j;
    v[i] = i8_at(tab[e / 4u], e % 4u) * tsc[id] + pos[s * H + j];
    sum += v[i];
  }
  let mean = reduce_sum(tid, sum) / f32(H);
  var sq = 0.0;
  for (var i = 0u; i < 3u; i++) { let d = v[i] - mean; sq += d * d; }
  let inv = inverseSqrt(reduce_sum(tid, sq) / f32(H) + EPS);
  for (var i = 0u; i < 3u; i++) {
    let j = tid + i * 128u;
    x[r * H + j] = (v[i] - mean) * inv * g[j] + b[j];
  }
}
`;

// Y[R,N] = act(s[n] * X[R,K] q[K,N] + b[n]). 16x16 lanes, each a 4x4 block: 64x64 tile, K step 16.
// As holds the X tile transposed as vec4 of 4 rows, Bs the dequantized-free int8 tile as vec4 of
// 4 columns, so the inner product is 16 vec4 FMAs per k per lane. Needs K % 16 == 0, N % 64 == 0.
const MATMUL = DIMS + /* wgsl */ `
struct Op { K: u32, N: u32, act: u32, pad: u32 };
@group(0) @binding(1) var<uniform> op: Op;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> W: array<u32>;
@group(0) @binding(4) var<storage, read> sc: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> Y: array<vec4<f32>>;
var<workgroup> As: array<vec4<f32>, 256>;   // [16 k][16 row quads]
var<workgroup> Bs: array<vec4<f32>, 256>;   // [16 k][16 col quads]

// Numerical Recipes erfc, fractional error < 1.2e-7 (same as leaf.js).
fn erfc_(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return select(2.0 - r, r, x >= 0.0);
}
fn gelu(v: vec4<f32>) -> vec4<f32> {
  return v * (vec4<f32>(1.0) - 0.5 * vec4<f32>(erfc_(v.x * 0.70710678), erfc_(v.y * 0.70710678),
                                                 erfc_(v.z * 0.70710678), erfc_(v.w * 0.70710678)));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) t: u32) {
  let K = op.K; let N = op.N; let R = dims.R;
  let row0 = wg.y * 64u; let col0 = wg.x * 64u;
  let tx = lid.x; let ty = lid.y;
  var acc: array<vec4<f32>, 4>;
  for (var i = 0u; i < 4u; i++) { acc[i] = vec4<f32>(0.0); }
  // Tile loads. Lane t writes one whole vec4 of each tile: As[k][row quad] = X rows 4q..4q+3 at
  // column k (k = t / 16, q = t % 16), and Bs[k][col quad] = 4 int8 of W row k. Writing single
  // components of a shared vec4 from different lanes races (read-modify-write on Metal).
  let ak = t / 16u; let aq = t % 16u;
  let bk = t / 16u; let bcq = t % 16u;
  for (var k0 = 0u; k0 < K; k0 += 16u) {
    var xv = vec4<f32>(0.0);
    let kc = k0 + ak;
    for (var c = 0u; c < 4u; c++) {
      let r = row0 + aq * 4u + c;
      if (r < R) { xv[c] = X[(r * K + kc) / 4u][kc % 4u]; }
    }
    As[ak * 16u + aq] = xv;
    let w = W[((k0 + bk) * N + col0) / 4u + bcq];
    Bs[bk * 16u + bcq] = vec4<f32>(f32(extractBits(bitcast<i32>(w), 0u, 8u)), f32(extractBits(bitcast<i32>(w), 8u, 8u)),
                                   f32(extractBits(bitcast<i32>(w), 16u, 8u)), f32(extractBits(bitcast<i32>(w), 24u, 8u)));
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let a = As[kk * 16u + ty];
      let bv = Bs[kk * 16u + tx];
      acc[0] += a.x * bv; acc[1] += a.y * bv; acc[2] += a.z * bv; acc[3] += a.w * bv;
    }
    workgroupBarrier();
  }
  let cq = (col0 / 4u) + tx;
  let s = sc[cq]; let bb = bias[cq];
  for (var i = 0u; i < 4u; i++) {
    let r = row0 + ty * 4u + i;
    if (r < R) {
      var y = acc[i] * s + bb;
      if (op.act == 1u) { y = gelu(y); }
      Y[r * (N / 4u) + cq] = y;
    }
  }
}
`;

// One lane per (sequence, head, query row). Scores over the sequence's real length only;
// rows at or past the length write zeros (never pooled). S <= 128, head dim 32.
const ATTN = DIMS + /* wgsl */ `
@group(0) @binding(1) var<storage, read> lens: array<u32>;
@group(0) @binding(2) var<storage, read> qkv: array<vec4<f32>>;   // [R][288]: Q 0..95, K 96..191, V 192..287
@group(0) @binding(3) var<storage, read_write> ctx: array<vec4<f32>>; // [R][96]
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let S = dims.S;
  let total = dims.B * 12u * S;
  if (gid.x >= total) { return; }
  let i = gid.x % S;
  let h = (gid.x / S) % 12u;
  let bq = gid.x / (S * 12u);
  let L = lens[bq];
  let ri = bq * S + i;
  if (i >= L) {
    for (var d = 0u; d < 8u; d++) { ctx[ri * 96u + h * 8u + d] = vec4<f32>(0.0); }
    return;
  }
  var q: array<vec4<f32>, 8>;
  for (var d = 0u; d < 8u; d++) { q[d] = qkv[ri * 288u + h * 8u + d] * 0.17677669529663687; }
  var sc: array<f32, 128>;
  var mx = -3.0e38;
  for (var j = 0u; j < L; j++) {
    let rj = (bq * S + j) * 288u + 96u + h * 8u;
    var a = 0.0;
    for (var d = 0u; d < 8u; d++) { a += dot(q[d], qkv[rj + d]); }
    sc[j] = a;
    mx = max(mx, a);
  }
  var sum = 0.0;
  for (var j = 0u; j < L; j++) { let e = exp(sc[j] - mx); sc[j] = e; sum += e; }
  var o: array<vec4<f32>, 8>;
  for (var d = 0u; d < 8u; d++) { o[d] = vec4<f32>(0.0); }
  for (var j = 0u; j < L; j++) {
    let p = sc[j] / sum;
    let rj = (bq * S + j) * 288u + 192u + h * 8u;
    for (var d = 0u; d < 8u; d++) { o[d] += p * qkv[rj + d]; }
  }
  for (var d = 0u; d < 8u; d++) { ctx[ri * 96u + h * 8u + d] = o[d]; }
}
`;

// One workgroup per row: x[r] = LayerNorm(y[r] + x[r]).
const ADDLN = DIMS + REDUCE128 + /* wgsl */ `
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
const H: u32 = 384u;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let r = wg.x;
  if (r >= dims.R) { return; } // uniform per workgroup; also keeps binding 0 in the auto layout
  var v: array<f32, 3>;
  var sum = 0.0;
  for (var i = 0u; i < 3u; i++) { let j = tid + i * 128u; v[i] = y[r * H + j] + x[r * H + j]; sum += v[i]; }
  let mean = reduce_sum(tid, sum) / f32(H);
  var sq = 0.0;
  for (var i = 0u; i < 3u; i++) { let d = v[i] - mean; sq += d * d; }
  let inv = inverseSqrt(reduce_sum(tid, sq) / f32(H) + EPS);
  for (var i = 0u; i < 3u; i++) { let j = tid + i * 128u; x[r * H + j] = (v[i] - mean) * inv * g[j] + b[j]; }
}
`;

// One workgroup per sequence: mean over its real rows, Dense 384 -> 768 (int8, per-column
// scale), L2 normalize. 128 lanes x 6 outputs each.
const POOL = DIMS + REDUCE128 + /* wgsl */ `
@group(0) @binding(1) var<storage, read> lens: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<u32>;   // [384][768] int8
@group(0) @binding(4) var<storage, read> sc: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [B][768]
const H: u32 = 384u;
const D: u32 = 768u;
var<workgroup> pooled: array<f32, 384>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let bq = wg.x;
  let L = lens[bq];
  for (var i = 0u; i < 3u; i++) {
    let j = tid + i * 128u;
    var a = 0.0;
    for (var s = 0u; s < L; s++) { a += x[(bq * dims.S + s) * H + j]; }
    pooled[j] = a / f32(L);
  }
  workgroupBarrier();
  var y: array<f32, 6>;
  var sq = 0.0;
  for (var i = 0u; i < 6u; i++) {
    let n = tid + i * 128u;
    var a = 0.0;
    for (var k = 0u; k < H; k++) { let e = k * D + n; a += pooled[k] * i8_at(W[e / 4u], e % 4u); }
    y[i] = a * sc[n] + bias[n];
    sq += y[i] * y[i];
  }
  let inv = inverseSqrt(reduce_sum(tid, sq));
  for (var i = 0u; i < 6u; i++) { out[bq * D + tid + i * 128u] = y[i] * inv; }
}
`;

// --------------------------------------------------------------------------------------- engine

export async function createLeafGPU({ weights, vocab, device, maxRows = 4096 }) {
  const W = parse(weights);
  const { hidden: H, layers: L, ffn: F, out_dim: D, eps } = W.config;
  if (H !== 384 || D !== 768 || W.config.heads !== 12) throw new Error("shaders are written for leaf-ir's 384/12/768 shape");
  const tok = new WordPiece(vocab, MAX_TOKENS);
  if (!device) {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error("WebGPU unavailable");
    device = await adapter.requestDevice();
  }
  const EPS = `const EPS: f32 = ${eps.toExponential()};\n`;
  const S_ = GPUBufferUsage.STORAGE, C_ = GPUBufferUsage.COPY_DST, U_ = GPUBufferUsage.UNIFORM;
  const upload = (u8) => {
    const size = Math.ceil(u8.byteLength / 4) * 4;
    const buf = device.createBuffer({ size, usage: S_ | C_, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(u8);
    buf.unmap();
    return buf;
  };
  const storage = (bytes) => device.createBuffer({ size: bytes, usage: S_ | C_ | GPUBufferUsage.COPY_SRC });
  const uniform = (u32s) => { const b = device.createBuffer({ size: 16, usage: U_ | C_ }); device.queue.writeBuffer(b, 0, new Uint32Array(u32s)); return b; };

  const w = {};
  for (const n of ["tok_emb", "pos_emb", "emb_ln_g", "emb_ln_b", "dense_w", "dense_b"]) w[n] = upload(W.bytes(n));
  w.tok_sc = upload(W.scale("tok_emb"));
  w.dense_sc = upload(W.scale("dense_w"));
  for (let l = 0; l < L; l++)
    for (const p of ["qkv", "o", "up", "down"]) {
      w[`l${l}.${p}_w`] = upload(W.bytes(`l${l}.${p}_w`));
      w[`l${l}.${p}_sc`] = upload(W.scale(`l${l}.${p}_w`));
      w[`l${l}.${p}_b`] = upload(W.bytes(`l${l}.${p}_b`));
      for (const g of ["ln1_g", "ln1_b", "ln2_g", "ln2_b"]) w[`l${l}.${g}`] ||= upload(W.bytes(`l${l}.${g}`));
    }

  const maxB = maxRows; // a batch never has more sequences than rows
  const a = {
    dims: device.createBuffer({ size: 16, usage: U_ | C_ }),
    ids: storage(maxRows * 4), lens: storage(maxB * 4),
    x: storage(maxRows * H * 4), qkv: storage(maxRows * 3 * H * 4), h: storage(maxRows * H * 4),
    u: storage(maxRows * F * 4), ctx: storage(maxRows * H * 4), out: storage(maxB * D * 4),
  };
  const pipe = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: EPS + code }), entryPoint: "main" } });
  const P = { embed: pipe(EMBED), mm: pipe(MATMUL), attn: pipe(ATTN), addln: pipe(ADDLN), pool: pipe(POOL) };
  const bind = (p, bufs) => device.createBindGroup({ layout: p.getBindGroupLayout(0),
    entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });

  // Every dispatch of a forward pass, with its bind group, built once.
  const steps = [];
  steps.push({ p: P.embed, g: bind(P.embed, [a.dims, a.ids, w.tok_emb, w.tok_sc, w.pos_emb, w.emb_ln_g, w.emb_ln_b, a.x]), rows: true });
  const mm = (l, name, K, N, act, X, Y) => ({ p: P.mm, g: bind(P.mm, [a.dims, uniform([K, N, act, 0]), X, w[`l${l}.${name}_w`],
    w[`l${l}.${name}_sc`], w[`l${l}.${name}_b`], Y]), tiles: N / 64 });
  for (let l = 0; l < L; l++) {
    steps.push(mm(l, "qkv", H, 3 * H, 0, a.x, a.qkv));
    steps.push({ p: P.attn, g: bind(P.attn, [a.dims, a.lens, a.qkv, a.ctx]), attn: true });
    steps.push(mm(l, "o", H, H, 0, a.ctx, a.h));
    steps.push({ p: P.addln, g: bind(P.addln, [a.dims, a.h, w[`l${l}.ln1_g`], w[`l${l}.ln1_b`], a.x]), rows: true });
    steps.push(mm(l, "up", H, F, 1, a.x, a.u));
    steps.push(mm(l, "down", F, H, 0, a.u, a.h));
    steps.push({ p: P.addln, g: bind(P.addln, [a.dims, a.h, w[`l${l}.ln2_g`], w[`l${l}.ln2_b`], a.x]), rows: true });
  }
  steps.push({ p: P.pool, g: bind(P.pool, [a.dims, a.lens, a.x, w.dense_w, w.dense_sc, w.dense_b, a.out]), seqs: true });

  const staging = device.createBuffer({ size: maxB * D * 4, usage: GPUBufferUsage.MAP_READ | C_ });

  // One padded batch: idsList.length sequences, all <= MAX_TOKENS ids. debug = {steps, buf}
  // stops after that many dispatches and returns the named activation buffer (rows R x width).
  async function runBatch(idsList, debug) {
    const B = idsList.length, S = Math.max(...idsList.map((x) => x.length)), R = B * S;
    if (R > maxRows) throw new Error(`batch of ${R} rows exceeds maxRows ${maxRows}`);
    const ids = new Uint32Array(R);
    idsList.forEach((x, b) => ids.set(x, b * S)); // pad id 0 = [PAD]; padded rows are never read back
    device.queue.writeBuffer(a.ids, 0, ids);
    device.queue.writeBuffer(a.lens, 0, Uint32Array.from(idsList, (x) => x.length));
    device.queue.writeBuffer(a.dims, 0, new Uint32Array([R, S, B, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (const s of debug ? steps.slice(0, debug.steps) : steps) {
      pass.setPipeline(s.p);
      pass.setBindGroup(0, s.g);
      if (s.rows) pass.dispatchWorkgroups(R);
      else if (s.attn) pass.dispatchWorkgroups(Math.ceil((B * 12 * S) / 64));
      else if (s.seqs) pass.dispatchWorkgroups(B);
      else pass.dispatchWorkgroups(s.tiles, Math.ceil(R / 64));
    }
    pass.end();
    if (debug) {
      const size = a[debug.buf].size, rb = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | C_ });
      enc.copyBufferToBuffer(a[debug.buf], 0, rb, 0, size);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(rb.getMappedRange().slice(0));
      rb.destroy();
      return out;
    }
    enc.copyBufferToBuffer(a.out, 0, staging, 0, B * D * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, B * D * 4);
    const flat = new Float32Array(staging.getMappedRange(0, B * D * 4).slice(0));
    staging.unmap();
    return idsList.map((_, b) => flat.subarray(b * D, b * D + D));
  }

  // Any number of sequences: sorted by length, packed into batches of <= maxRows padded rows.
  async function forwardBatch(idsList) {
    const order = idsList.map((_, i) => i).sort((p, q) => idsList[p].length - idsList[q].length);
    const out = new Array(idsList.length);
    for (let i = 0; i < order.length;) {
      let j = i;
      while (j < order.length && (j - i + 1) * idsList[order[j]].length <= maxRows) j++;
      const chunk = order.slice(i, j);
      const vs = await runBatch(chunk.map((k) => idsList[k]));
      chunk.forEach((k, c) => (out[k] = vs[c]));
      i = j;
    }
    return out;
  }

  return {
    device, config: W.config,
    tokenize: (text) => tok.encode(text),
    forwardBatch,
    debugRun: (idsList, steps, buf) => runBatch(idsList, { steps, buf }),
    embedBatch: (texts, { query = false } = {}) => forwardBatch(texts.map((t) => tok.encode((query ? QUERY_PROMPT : "") + t))),
    destroy: () => { for (const b of [...Object.values(w), ...Object.values(a), staging]) b.destroy(); },
  };
}
