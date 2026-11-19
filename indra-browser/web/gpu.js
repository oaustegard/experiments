// A small WebGPU runner: storage buffers, cached pipelines, and passes that record many
// dispatches into one command buffer. Every kernel takes a uniform block of u32/f32 words at
// binding 0 and its storage buffers at bindings 1.., in order.

export class Gpu {
  static async create() {
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const L = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has("timestamp-query") ? ["timestamp-query"] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(L.maxStorageBuffersPerShaderStage, 10),
      },
    });
    const info = adapter.info ?? {};
    return new Gpu(device, `${info.vendor ?? ""} ${info.architecture ?? ""} ${info.description ?? ""}`.trim());
  }

  constructor(device, label = "") {
    this.device = device;
    this.label = label;
    this.pipes = new Map();
    this.dummy = this.buffer(16);
    this.enc = null;
    this.uniforms = [];
    this.pool = new Map();
    this.uchunks = []; // uniform buffers of 256 slots x 256 bytes, reused across passes
    this.lost = null;
    device.lost?.then((i) => { this.lost = i; });
  }

  // A temporary (no data) asked for while a pass is open comes from a pool of power-of-two sizes;
  // free() hands it back for later dispatches of the same pass (WebGPU orders a pass's dispatches,
  // so a buffer can be rewritten once its readers are recorded). Pooled buffers ignore destroy().
  buffer(bytes, data) {
    if (!data && this.enc) {
      const size = 2 ** Math.max(8, Math.ceil(Math.log2(Math.max(bytes, 4))));
      const pool = this.pool.get(size);
      let b = pool && pool.length ? pool.pop() : null;
      if (!b) {
        b = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        b.pooled = size;
        b.destroy = () => {};
      }
      this.out.add(b);
      return b;
    }
    const size = Math.max(16, Math.ceil(bytes / 4) * 4);
    const b = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (data) this.device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }

  upload(typed) { return this.buffer(typed.byteLength, typed); }

  free(b) {
    if (!b || !b.pooled || !this.out.delete(b)) return;
    if (!this.pool.has(b.pooled)) this.pool.set(b.pooled, []);
    this.pool.get(b.pooled).push(b);
  }

  pipeline(code) {
    let p = this.pipes.get(code);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipes.set(code, p);
    }
    return p;
  }

  // Per-kernel GPU time: startProfile(), run, then profileReport() gives ms per kernel name
  // (one compute pass per dispatch while profiling, so totals run a little high).
  startProfile() {
    if (!this.device.features.has("timestamp-query")) return false;
    this.prof = { ms: {}, calls: {} };
    return true;
  }
  profileReport() { const r = this.prof; this.prof = null; return r; }

  begin() {
    if (this.enc) throw new Error("pass already open");
    this.enc = this.device.createCommandEncoder();
    this.uslot = 0;
    this.out = new Set();
    if (this.prof) {
      this.pq = { set: this.device.createQuerySet({ type: "timestamp", count: 2 * 2048 }), labels: [] };
      this.pass = null;
    } else this.pass = this.enc.beginComputePass();
  }

  // words: array of numbers; ints go as u32, a {f: x} entry as f32.
  dispatch(code, bufs, words, groups, label = null) {
    const slot = this.uslot++;
    const ci = slot >> 8, off = (slot & 255) * 256;
    while (this.uchunks.length <= ci) this.uchunks.push({ buf: this.device.createBuffer({ size: 65536, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), data: new ArrayBuffer(65536) });
    const dv = new DataView(this.uchunks[ci].data, off, 256);
    words.forEach((w, i) => {
      if (typeof w === "object") dv.setFloat32(4 * i, w.f, true);
      else dv.setUint32(4 * i, w >>> 0, true);
    });
    const pipe = this.pipeline(code);
    if (this.prof) {
      const i = this.pq.labels.length;
      if (i < 2048) {
        this.pass = this.enc.beginComputePass({ timestampWrites: { querySet: this.pq.set, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        this.pq.labels.push((this.scope ? this.scope + "." : "") + (label ?? KERNEL_NAMES.get(code) ?? "other"));
      } else this.pass = this.enc.beginComputePass();
    }
    const entries = [{ binding: 0, resource: { buffer: this.uchunks[ci].buf, offset: off, size: Math.max(16, Math.ceil(words.length / 4) * 16) } }, ...bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b ?? this.dummy } }))];
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    this.pass.setPipeline(pipe);
    this.pass.setBindGroup(0, bg);
    const [x, y = 1, z = 1] = groups;
    this.pass.dispatchWorkgroups(x, y, z);
    if (this.prof) { this.pass.end(); this.pass = null; }
  }

  // 1-D work of n threads in workgroups of 64, split over x and y past the 65,535 limit.
  grid(n, wg = 64) {
    const g = Math.ceil(n / wg);
    if (g <= 65535) return [g, 1];
    return [65535, Math.ceil(g / 65535)];
  }

  async end(reads = []) {
    let pbuf = null, pn = 0;
    if (this.prof) {
      pn = this.pq.labels.length;
      if (pn) {
        const rb = this.device.createBuffer({ size: 16 * pn, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        this.enc.resolveQuerySet(this.pq.set, 0, 2 * pn, rb, 0);
        pbuf = this.device.createBuffer({ size: 16 * pn, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.enc.copyBufferToBuffer(rb, 0, pbuf, 0, 16 * pn);
      }
    } else this.pass.end();
    const stage = reads.map(([b, bytes]) => {
      const s = this.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.enc.copyBufferToBuffer(b, 0, s, 0, Math.ceil(bytes / 4) * 4);
      return s;
    });
    for (let c = 0; c * 256 < this.uslot; c++) this.device.queue.writeBuffer(this.uchunks[c].buf, 0, this.uchunks[c].data, 0, Math.min(256, this.uslot - c * 256) * 256);
    this.device.queue.submit([this.enc.finish()]);
    // every temporary of this pass goes back to the pool for later passes
    for (const b of this.out) { if (!this.pool.has(b.pooled)) this.pool.set(b.pooled, []); this.pool.get(b.pooled).push(b); }
    this.out = new Set();
    this.enc = null;
    const out = [];
    for (const s of stage) {
      await s.mapAsync(GPUMapMode.READ);
      out.push(s.getMappedRange().slice(0));
      s.destroy();
    }
    if (!reads.length) await this.device.queue.onSubmittedWorkDone();
    if (pbuf) {
      await pbuf.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(pbuf.getMappedRange());
      this.pq.labels.forEach((l, i) => {
        this.prof.ms[l] = (this.prof.ms[l] ?? 0) + Number(t[2 * i + 1] - t[2 * i]) / 1e6;
        this.prof.calls[l] = (this.prof.calls[l] ?? 0) + 1;
      });
      pbuf.destroy();
      this.pq.set.destroy();
    }
    if (this.lost) throw new Error(`GPU device lost: ${this.lost.message}`);
    return out;
  }
}

// ---------------------------------------------------------------------------------- kernels

const IDX = /* wgsl */ `
fn gid(g: vec3<u32>, n: vec3<u32>) -> u32 { return g.x + g.y * n.x * 64u; }`;

// NHWC convolution, any kernel, stride, padding and group count. One thread per output value.
// Weights are [O][kh][kw][C/g]. Epilogue: + bias, + residual (mode 1, before the activation),
// activation (0 none, 1 hardswish, 2 relu, 3 sigmoid, 4 GELU), + residual (mode 2, after it), times a
// per-pixel mask (sequences: mask[n * W + x]). The input and output may be channel slices of
// wider tensors (stride inCs / outCs, offset inCo / outCo).
export const CONV = /* wgsl */ `
struct P { N: u32, H: u32, W: u32, C: u32, Ho: u32, Wo: u32, O: u32, kh: u32,
           kw: u32, sh: u32, sw: u32, pt: u32, pl: u32, g: u32, act: u32, res: u32,
           inCs: u32, inCo: u32, outCs: u32, outCo: u32, mask: u32, total: u32, nx: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> r: array<f32>;
@group(0) @binding(5) var<storage, read> m: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
// GELU with erf from Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), as nets.js computes it
fn gelu(x: f32) -> f32 {
  let z = x / sqrt(2.0);
  let a = abs(z); let t = 1.0 / (1.0 + 0.3275911 * a);
  let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return 0.5 * x * (1.0 + select(e, -e, z < 0.0));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.total) { return; }
  let o = idx % p.O;
  let pix = idx / p.O;
  let ox = pix % p.Wo;
  let oy = (pix / p.Wo) % p.Ho;
  let n = pix / (p.Wo * p.Ho);
  let cg = p.C / p.g;
  let og = p.O / p.g;
  let gi = o / og;
  var acc = b[o];
  for (var ky = 0u; ky < p.kh; ky++) {
    let iy = i32(oy * p.sh + ky) - i32(p.pt);
    if (iy < 0 || iy >= i32(p.H)) { continue; }
    for (var kx = 0u; kx < p.kw; kx++) {
      let ix = i32(ox * p.sw + kx) - i32(p.pl);
      if (ix < 0 || ix >= i32(p.W)) { continue; }
      let xb = ((n * p.H + u32(iy)) * p.W + u32(ix)) * p.inCs + p.inCo + gi * cg;
      let wb = ((o * p.kh + ky) * p.kw + kx) * cg;
      for (var c = 0u; c < cg; c++) { acc += x[xb + c] * w[wb + c]; }
    }
  }
  let oi = pix * p.O + o;
  if (p.res == 1u) { acc += r[oi]; }
  if (p.act == 1u) { acc = acc * clamp(acc + 3.0, 0.0, 6.0) / 6.0; }
  else if (p.act == 2u) { acc = max(acc, 0.0); }
  else if (p.act == 3u) { acc = 1.0 / (1.0 + exp(-acc)); }
  else if (p.act == 4u) { acc = gelu(acc); }
  if (p.res == 2u) { acc += r[oi]; }
  if (p.mask == 1u) { acc *= m[n * p.Wo + ox]; }
  y[pix * p.outCs + p.outCo + o] = acc;
}`;

// Shape moves over NHWC tensors. kind 0: up2 (nearest, x2). 1: s2d (out channel c*4+dy*2+dx).
// 2: d2s CRD (in channel c*4+dy*2+dx). 3: channel slice copy into a wider tensor (concat).
// 4: 2x2 average pool. 5: 2x2 max pool. Shapes are the output's (N, H, W, C); Hi, Wi, Ci the input's.
export const MOVE = /* wgsl */ `
struct P { kind: u32, N: u32, H: u32, W: u32, C: u32, Hi: u32, Wi: u32, Ci: u32,
           outCs: u32, outCo: u32, total: u32, nx: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.total) { return; }
  let c = idx % p.C;
  let pix = idx / p.C;
  let ox = pix % p.W;
  let oy = (pix / p.W) % p.H;
  let n = pix / (p.W * p.H);
  var v = 0.0;
  if (p.kind == 0u) {
    v = x[((n * p.Hi + oy / 2u) * p.Wi + ox / 2u) * p.Ci + c];
  } else if (p.kind == 1u) {
    let ci = c / 4u; let dy = (c / 2u) % 2u; let dx = c % 2u;
    v = x[((n * p.Hi + 2u * oy + dy) * p.Wi + 2u * ox + dx) * p.Ci + ci];
  } else if (p.kind == 2u) {
    let dy = oy % 2u; let dx = ox % 2u;
    v = x[((n * p.Hi + oy / 2u) * p.Wi + ox / 2u) * p.Ci + c * 4u + dy * 2u + dx];
  } else if (p.kind == 3u) {
    v = x[pix * p.Ci + c];
  } else {
    let b0 = ((n * p.Hi + 2u * oy) * p.Wi + 2u * ox) * p.Ci + c;
    let a = x[b0]; let bb = x[b0 + p.Ci]; let cc = x[b0 + p.Wi * p.Ci]; let d = x[b0 + p.Wi * p.Ci + p.Ci];
    if (p.kind == 4u) { v = (a + bb + cc + d) * 0.25; } else { v = max(max(a, bb), max(cc, d)); }
  }
  y[pix * p.outCs + p.outCo + c] = v;
}`;

// Sequence ops over [B, T, C]. kind 0: y = act(x * s[c] + t[c]) * mask (batch norm, hardswish or GELU).
// 1: pooled[b, c] = sum_t x[b, t, c] mask[b, t] / max(1, sum_t mask[b, t]) (one thread per (b, c)).
// 2: y[b, t, c] = x[b, t, c] * s[b, c] (squeeze-excite scale).
export const SEQ = /* wgsl */ `
struct P { kind: u32, B: u32, T: u32, C: u32, total: u32, nx: u32, act: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read> t: array<f32>;
@group(0) @binding(4) var<storage, read> m: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
// GELU with erf from Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), as nets.js computes it
fn gelu(x: f32) -> f32 {
  let z = x / sqrt(2.0);
  let a = abs(z); let t = 1.0 / (1.0 + 0.3275911 * a);
  let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return 0.5 * x * (1.0 + select(e, -e, z < 0.0));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.total) { return; }
  if (p.kind == 0u) {
    let c = idx % p.C; let bt = idx / p.C;
    var v = x[idx] * s[c] + t[c];
    if (p.act == 1u) { v = v * clamp(v + 3.0, 0.0, 6.0) / 6.0; }
    else if (p.act == 4u) { v = gelu(v); }
    y[idx] = v * m[bt];
  } else if (p.kind == 1u) {
    let c = idx % p.C; let b = idx / p.C;
    var acc = 0.0; var cnt = 0.0;
    for (var i = 0u; i < p.T; i++) {
      let mk = m[b * p.T + i];
      acc += x[(b * p.T + i) * p.C + c] * mk;
      cnt += mk;
    }
    y[idx] = acc / max(cnt, 1.0);
  } else {
    let c = idx % p.C; let b = idx / (p.C * p.T);
    y[idx] = x[idx] * s[b * p.C + c];
  }
}`;

// The recognizer's context read: for each (b, t) and each of 4 heads of 16 channels, a softmax
// over the 24 rows of the head's score (plus its row prior), and the rows' features summed under
// it. feat [B, 24, T, 64], score [B, 24, T, 4], prior [4, 24] -> out [B, T, 64].
export const CTX = /* wgsl */ `
struct P { B: u32, T: u32, total: u32, nx: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read> s: array<f32>;
@group(0) @binding(3) var<storage, read> pr: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x + g.y * p.nx * 64u;
  if (idx >= p.total) { return; }
  let c = idx % 64u; let bt = idx / 64u;
  let t = bt % p.T; let b = bt / p.T;
  let h = c / 16u;
  var mx = -1e30;
  for (var r = 0u; r < 24u; r++) { mx = max(mx, s[((b * 24u + r) * p.T + t) * 4u + h] + pr[h * 24u + r]); }
  var den = 0.0; var acc = 0.0;
  for (var r = 0u; r < 24u; r++) {
    let e = exp(s[((b * 24u + r) * p.T + t) * 4u + h] + pr[h * 24u + r] - mx);
    den += e;
    acc += e * f[((b * 24u + r) * p.T + t) * 64u + c];
  }
  y[idx] = acc / den;
}`;

// CTC outputs per time step of logits [R, V] (one workgroup per row): the best class, its
// probability (1 / sum exp(l - max)), the blank's log-probability, and the 8 best classes'
// log-probabilities. out [R, 19] = id, conf, blank, topv[8], topi[8].
export const CTC = /* wgsl */ `
struct P { R: u32, V: u32, nx: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> l: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> cv: array<f32, 512>;
var<workgroup> ci: array<u32, 512>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wg.x + wg.y * p.nx;
  if (row >= p.R) { return; }
  let base = row * p.V;
  // local top 8, sorted descending
  var tv = array<f32, 8>(-1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30, -1e30);
  var ti = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  for (var k = li; k < p.V; k += 64u) {
    let v = l[base + k];
    if (v > tv[7]) {
      var j = 7u;
      while (j > 0u && tv[j - 1u] < v) { tv[j] = tv[j - 1u]; ti[j] = ti[j - 1u]; j--; }
      tv[j] = v; ti[j] = k;
    }
  }
  for (var j = 0u; j < 8u; j++) { cv[li * 8u + j] = tv[j]; ci[li * 8u + j] = ti[j]; }
  workgroupBarrier();
  let mx = select(-1e30, cv[0], true);
  // global max: reduce the local maxima
  red[li] = tv[0];
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  let gmax = red[0];
  workgroupBarrier();
  var se = 0.0;
  for (var k = li; k < p.V; k += 64u) { se += exp(l[base + k] - gmax); }
  red[li] = se;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let sum = red[0];
    let lse = gmax + log(sum);
    let o = row * 19u;
    // merge: 8 passes picking the largest remaining candidate (ties: lower class id)
    for (var j = 0u; j < 8u; j++) {
      var bv = -1e31; var bi = 0u; var bk = 0u;
      for (var k = 0u; k < 512u; k++) {
        let v = cv[k];
        if (v > bv || (v == bv && ci[k] < bi)) { bv = v; bi = ci[k]; bk = k; }
      }
      cv[bk] = -1e32;
      y[o + 3u + j] = bv - lse;
      y[o + 11u + j] = f32(bi);
      if (j == 0u) { y[o] = f32(bi); }
    }
    y[o + 1u] = 1.0 / sum;
    y[o + 2u] = l[base] - lse;
  }
}`;

// 1x1 convolution as a tiled matmul: Y[r, n] = epilogue(sum_k X[r, k] W[n, k] + b[n]) for rows r
// (pixels) of X [R, K] and weights W [N, K]. 64 x 64 outputs per workgroup of 256 threads, each
// computing 4 x 4, K in tiles of 16 through workgroup memory. Same epilogue as CONV (residual
// before / after the activation, mask per row; activation 4 is GELU, for the order model), output
// channel slice by outCs / outCo.
export const MM = /* wgsl */ `
struct P { R: u32, K: u32, N: u32, act: u32, res: u32, mask: u32, outCs: u32, outCo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read> r: array<f32>;
@group(0) @binding(5) var<storage, read> m: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
// GELU with erf from Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), as nets.js computes it
fn gelu(x: f32) -> f32 {
  let z = x / sqrt(2.0);
  let a = abs(z); let t = 1.0 / (1.0 + 0.3275911 * a);
  let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return 0.5 * x * (1.0 + select(e, -e, z < 0.0));
}
var<workgroup> xs: array<f32, 1024>; // [16 k][64 rows]
var<workgroup> ws: array<f32, 1024>; // [16 k][64 cols]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let r0 = wg.y * 64u; let n0 = wg.x * 64u;
  let tr = (l / 16u) * 4u; let tn = (l % 16u) * 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    for (var q = 0u; q < 4u; q++) {
      let e = l + q * 256u;        // 0..1023
      let kk = e % 16u; let rr = e / 16u;
      let row = r0 + rr; let k = k0 + kk;
      xs[kk * 64u + rr] = select(0.0, x[row * p.K + k], row < p.R && k < p.K);
      let col = n0 + rr;
      ws[kk * 64u + rr] = select(0.0, w[col * p.K + k], col < p.N && k < p.K);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let a = vec4<f32>(xs[kk * 64u + tr], xs[kk * 64u + tr + 1u], xs[kk * 64u + tr + 2u], xs[kk * 64u + tr + 3u]);
      let c = vec4<f32>(ws[kk * 64u + tn], ws[kk * 64u + tn + 1u], ws[kk * 64u + tn + 2u], ws[kk * 64u + tn + 3u]);
      for (var i = 0u; i < 4u; i++) {
        acc[i * 4u] += a[i] * c.x; acc[i * 4u + 1u] += a[i] * c.y;
        acc[i * 4u + 2u] += a[i] * c.z; acc[i * 4u + 3u] += a[i] * c.w;
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let row = r0 + tr + i;
    if (row >= p.R) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + tn + j;
      if (n >= p.N) { continue; }
      var v = acc[i * 4u + j] + b[n];
      let oi = row * p.N + n;
      if (p.res == 1u) { v += r[oi]; }
      if (p.act == 1u) { v = v * clamp(v + 3.0, 0.0, 6.0) / 6.0; }
      else if (p.act == 2u) { v = max(v, 0.0); }
      else if (p.act == 3u) { v = 1.0 / (1.0 + exp(-v)); }
      else if (p.act == 4u) { v = gelu(v); }
      if (p.res == 2u) { v += r[oi]; }
      if (p.mask == 1u) { v *= m[row]; }
      y[row * p.outCs + p.outCo + n] = v;
    }
  }
}`;

export const KERNEL_NAMES = new Map([[CONV, "conv"], [MOVE, "move"], [SEQ, "seq"], [CTX, "ctx"], [CTC, "ctc"], [MM, "mm"]]);
export const registerKernel = (code, name) => KERNEL_NAMES.set(code, name);
