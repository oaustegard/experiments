// tail.js: WebGPU fused tail for EG2 / remex 2-bit codes. ES module, no dependencies.
// Contract: SPEC.md (same directory tree)

const D = 768;
const WORDS = 48;              // u32 words per packed vector (192 bytes)
const LUT_STRIDE = 3073;       // 768*4 LUT entries, then the query offset
const MAX_DIM = 65535;         // max workgroups per dispatch dimension

// ---------------------------------------------------------------- WGSL

// Workgroup tree reduction over 256 lanes. Every invocation must call it.
// Barriers before the first write and after the read keep `red` reusable.
const REDUCE = /* wgsl */ `
var<workgroup> red: array<f32, 256>;

fn reduce_sum(tid: u32, v: f32) -> f32 {
  red[tid] = v;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      red[tid] = red[tid] + red[tid + s];
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`;

// Rotated row dot product: rotated_j = sum_k R[j,k] * xs[k].
// rmat holds R TRANSPOSED (rmat[k*D + j] = R[j,k]): at each k the 256 lanes
// (j = tid + 256*i) read 256 adjacent floats, which coalesces on real GPUs.
// Row-major R had lanes 768 floats apart. Same products, same order.
const DOT_ROW = /* wgsl */ `
const D: u32 = 768u;
var<workgroup> xs: array<f32, 768>;

fn dot_row(j: u32) -> f32 {
  var a: f32 = 0.0;
  for (var k: u32 = 0u; k < D; k += 1u) {
    a += rmat[k * D + j] * xs[k];
  }
  return a;
}
`;

const ENCODE = /* wgsl */ `
struct EncP {
  in_row: u32,
  out_row: u32,
  capacity: u32,
  normalize: u32,
}
@group(0) @binding(0) var<uniform> ep: EncP;
@group(0) @binding(1) var<storage, read> xin: array<f32>;
@group(0) @binding(2) var<storage, read> rmat: array<f32>;
@group(0) @binding(3) var<storage, read> rmu: array<f32>;
@group(0) @binding(4) var<storage, read> cst: array<f32>;
@group(0) @binding(5) var<storage, read_write> wout: array<u32>;
@group(0) @binding(6) var<storage, read_write> aux: array<f32>;

var<workgroup> cds: array<u32, 768>;

fn code_of(u: f32) -> u32 {
  var c: u32 = 0u;
  if (u > cst[0]) { c += 1u; }
  if (u > cst[1]) { c += 1u; }
  if (u > cst[2]) { c += 1u; }
  return c;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let doc = ep.out_row + wg.x;
  let rowb = (ep.in_row + wg.x) * D;

  var x0 = xin[rowb + tid];
  var x1 = xin[rowb + tid + 256u];
  var x2 = xin[rowb + tid + 512u];
  let ss = reduce_sum(tid, x0 * x0 + x1 * x1 + x2 * x2);
  if (ep.normalize != 0u) {
    let nrm = max(sqrt(ss), 1e-8);
    x0 = x0 / nrm;
    x1 = x1 / nrm;
    x2 = x2 / nrm;
  }
  xs[tid] = x0;
  xs[tid + 256u] = x1;
  xs[tid + 512u] = x2;
  workgroupBarrier();

  // t = R x - Rmu, for the three rows this thread owns
  let t0 = dot_row(tid) - rmu[tid];
  let t1 = dot_row(tid + 256u) - rmu[tid + 256u];
  let t2 = dot_row(tid + 512u) - rmu[tid + 512u];
  let ln = sqrt(reduce_sum(tid, t0 * t0 + t1 * t1 + t2 * t2));
  let md = reduce_sum(tid, rmu[tid] * t0 + rmu[tid + 256u] * t1 + rmu[tid + 512u] * t2);

  // codes from u = t / len, strict > against the boundaries
  let den = max(ln, 1e-8);
  let c0 = code_of(t0 / den);
  let c1 = code_of(t1 / den);
  let c2 = code_of(t2 / den);
  cds[tid] = c0;
  cds[tid + 256u] = c1;
  cds[tid + 512u] = c2;

  // vlen = |c[code]|, vmu = c[code] . Rmu
  let v0 = cst[4u + c0];
  let v1 = cst[4u + c1];
  let v2 = cst[4u + c2];
  let vl = sqrt(reduce_sum(tid, v0 * v0 + v1 * v1 + v2 * v2));
  let vm = reduce_sum(tid, v0 * rmu[tid] + v1 * rmu[tid + 256u] + v2 * rmu[tid + 512u]);
  let scale = ln / (max(vl, 1e-12) * cst[8]);
  let bias = md - scale * vm;

  // word w holds codes 16w..16w+15; code 16w+4bb+m at shift 8bb+6-2m
  if (tid < 48u) {
    var word: u32 = 0u;
    for (var bb: u32 = 0u; bb < 4u; bb += 1u) {
      for (var m: u32 = 0u; m < 4u; m += 1u) {
        word = word | (cds[16u * tid + 4u * bb + m] << (8u * bb + 6u - 2u * m));
      }
    }
    wout[tid * ep.capacity + doc] = word;
  }
  if (tid == 0u) {
    aux[doc * 4u] = scale;
    aux[doc * 4u + 1u] = bias;
    aux[doc * 4u + 2u] = ln;
    aux[doc * 4u + 3u] = md;
  }
}
`;

const POOL = /* wgsl */ `
struct PoolP {
  s: u32,
  b0: u32,
  pad0: u32,
  pad1: u32,
}
const D: u32 = 768u;
@group(0) @binding(0) var<uniform> pp: PoolP;
@group(0) @binding(1) var<storage, read> hid: array<f32>;
@group(0) @binding(2) var<storage, read> msk: array<u32>;
@group(0) @binding(3) var<storage, read_write> pout: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let d = wg.x * 256u + lid.x;
  if (d >= D) {
    return;
  }
  let b = pp.b0 + wg.y;
  var sum: f32 = 0.0;
  var cnt: u32 = 0u;
  for (var s: u32 = 0u; s < pp.s; s += 1u) {
    let m = msk[b * pp.s + s];
    if (m != 0u) {
      sum += hid[(b * pp.s + s) * D + d];
      cnt += 1u;
    }
  }
  var r: f32 = 0.0;
  if (cnt > 0u) {
    r = sum / f32(cnt);
  }
  pout[b * D + d] = r;
}
`;

const PREP = /* wgsl */ `
struct PrepP {
  normalize: u32,
  q_base: u32,
  stride: u32,
  pad: u32,
}
@group(0) @binding(0) var<uniform> pq: PrepP;
@group(0) @binding(1) var<storage, read> qin: array<f32>;
@group(0) @binding(2) var<storage, read> rmat: array<f32>;
@group(0) @binding(3) var<storage, read> mu: array<f32>;
@group(0) @binding(4) var<storage, read> cst: array<f32>;
@group(0) @binding(5) var<storage, read_write> lut: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let q = pq.q_base + wg.x;
  let qb = q * D;

  var a0 = qin[qb + tid];
  var a1 = qin[qb + tid + 256u];
  var a2 = qin[qb + tid + 512u];
  let ss = reduce_sum(tid, a0 * a0 + a1 * a1 + a2 * a2);
  if (pq.normalize != 0u) {
    let nrm = max(sqrt(ss), 1e-8);
    a0 = a0 / nrm;
    a1 = a1 / nrm;
    a2 = a2 / nrm;
  }
  xs[tid] = a0;
  xs[tid + 256u] = a1;
  xs[tid + 512u] = a2;
  workgroupBarrier();

  let qoff = reduce_sum(tid, a0 * mu[tid] + a1 * mu[tid + 256u] + a2 * mu[tid + 512u]);

  // lut[j*4 + c] = (R q)_j * centroid[c]
  let lb = q * pq.stride;
  for (var i: u32 = 0u; i < 3u; i += 1u) {
    let j = tid + 256u * i;
    let qr = dot_row(j);
    let jb = lb + j * 4u;
    lut[jb] = qr * cst[4];
    lut[jb + 1u] = qr * cst[5];
    lut[jb + 2u] = qr * cst[6];
    lut[jb + 3u] = qr * cst[7];
  }
  if (tid == 0u) {
    lut[lb + 3072u] = qoff;
  }
}
`;

const SCAN = /* wgsl */ `
struct ScanP {
  n: u32,
  capacity: u32,
  q_base: u32,
  stride: u32,
}
@group(0) @binding(0) var<uniform> ps: ScanP;
@group(0) @binding(1) var<storage, read> wds: array<u32>;
@group(0) @binding(2) var<storage, read> aux: array<f32>;
@group(0) @binding(3) var<storage, read> lut: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

var<workgroup> lsh: array<f32, 3072>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let q = ps.q_base + wg.y;
  let lb = q * ps.stride;
  for (var t: u32 = tid; t < 3072u; t += 256u) {
    lsh[t] = lut[lb + t];
  }
  workgroupBarrier();

  let i = wg.x * 256u + tid;
  if (i >= ps.n) {
    return;
  }
  var acc: f32 = 0.0;
  for (var w: u32 = 0u; w < 48u; w += 1u) {
    let word = wds[w * ps.capacity + i];
    for (var bb: u32 = 0u; bb < 4u; bb += 1u) {
      for (var m: u32 = 0u; m < 4u; m += 1u) {
        let code = (word >> (8u * bb + 6u - 2u * m)) & 3u;
        acc += lsh[(w * 16u + bb * 4u + m) * 4u + code];
      }
    }
  }
  scores[q * ps.capacity + i] = aux[i * 4u] * acc + aux[i * 4u + 1u] + lut[lb + 3072u];
}
`;

const COLSUM = /* wgsl */ `
struct ColP {
  count: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}
@group(0) @binding(0) var<uniform> pc: ColP;
@group(0) @binding(1) var<storage, read> cin: array<f32>;
@group(0) @binding(2) var<storage, read_write> cacc: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let j = wg.x * 256u + lid.x;
  if (j >= 768u) {
    return;
  }
  var s: f32 = 0.0;
  for (var b: u32 = 0u; b < pc.count; b += 1u) {
    s += cin[b * 768u + j];
  }
  cacc[j] = cacc[j] + s;
}
`;

const SHADERS = {
  encode: ENCODE.replace('var<workgroup> cds', REDUCE + '\n' + DOT_ROW + '\nvar<workgroup> cds'),
  pool: POOL,
  prep: PREP.replace('@group(0) @binding(0) var<uniform> pq', REDUCE + '\n' + DOT_ROW + '\n@group(0) @binding(0) var<uniform> pq'),
  scan: SCAN,
  colsum: COLSUM,
};

// ---------------------------------------------------------------- host helpers

// Usage and map flags: the browser globals when present, spec values otherwise.
const USAGE = globalThis.GPUBufferUsage ?? {
  MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80,
};
const MAP = globalThis.GPUMapMode ?? { READ: 0x1 };

function toF32(a, n, name) {
  const f = a instanceof Float32Array ? a : Float32Array.from(a);
  if (f.length !== n) throw new Error(`${name}: ${f.length} values, expected ${n}`);
  return f;
}

// Per-dispatch uniform. Each call returns a fresh buffer that is filled at
// creation, so no queue.writeBuffer ordering can leak values between
// dispatches recorded into the same submit.
function uniform(device, typed) {
  const size = Math.max(16, Math.ceil(typed.byteLength / 16) * 16);
  const buf = device.createBuffer({ size, usage: USAGE.UNIFORM, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
  buf.unmap();
  return buf;
}

function storageBuf(device, bytes, extra = 0) {
  return device.createBuffer({ size: Math.max(4, bytes), usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST | extra });
}

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

// ---------------------------------------------------------------- class

export class RemexTail {
  constructor(device) {
    this.device = device;
  }

  static async create(device, params) {
    const t = new RemexTail(device);
    await t._init(params);
    return t;
  }

  async _init(params) {
    if (!LITTLE_ENDIAN) throw new Error('RemexTail requires a little-endian host');
    const dev = this.device;
    this.R = toF32(params.R, D * D, 'R');
    this.rmat = storageBuf(dev, D * D * 4);
    const RT = new Float32Array(D * D);           // upload R^T, see DOT_ROW
    for (let j = 0; j < D; j++) for (let k = 0; k < D; k++) RT[k * D + j] = this.R[j * D + k];
    dev.queue.writeBuffer(this.rmat, 0, RT);
    this.rmu = storageBuf(dev, D * 4);
    this.mu = storageBuf(dev, D * 4);

    // cst: [b0, b1, b2, _, c0, c1, c2, c3, kappa, _ ...]
    const b = toF32(params.boundaries, 3, 'boundaries');
    const c = toF32(params.centroids, 4, 'centroids');
    const cst = new Float32Array(16);
    cst.set(b, 0);
    cst.set(c, 4);
    cst[8] = params.kappa;
    this.cst = storageBuf(dev, 64);
    dev.queue.writeBuffer(this.cst, 0, cst);
    this.kappa = params.kappa;

    const mk = async (label, code) => {
      const module = dev.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === 'error');
      if (errs.length) {
        throw new Error(`${label} WGSL: ` + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; '));
      }
      return dev.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
    };
    this.pipes = {
      encode: await mk('encode', SHADERS.encode),
      pool: await mk('pool', SHADERS.pool),
      prep: await mk('prep', SHADERS.prep),
      scan: await mk('scan', SHADERS.scan),
      colsum: await mk('colsum', SHADERS.colsum),
    };

    if (params.Rmu) {
      dev.queue.writeBuffer(this.rmu, 0, toF32(params.Rmu, D, 'Rmu'));
      dev.queue.writeBuffer(this.mu, 0, toF32(params.mu, D, 'mu'));
      this.mu_host = toF32(params.mu, D, 'mu');
    } else {
      this.setMean(params.mu);
    }
  }

  // Recomputes Rmu = R mu on the host (float64 accumulate) and uploads mu and Rmu.
  setMean(mu) {
    const m = toF32(mu, D, 'mu');
    const rmu = new Float32Array(D);
    for (let j = 0; j < D; j++) {
      let a = 0;
      const base = j * D;
      for (let k = 0; k < D; k++) a += this.R[base + k] * m[k];
      rmu[j] = a;
    }
    this.device.queue.writeBuffer(this.mu, 0, m);
    this.device.queue.writeBuffer(this.rmu, 0, rmu);
    this.mu_host = m;
    this.rmu_host = rmu;
  }

  allocCorpus(capacity) {
    const dev = this.device;
    return {
      capacity,
      filled: 0,
      words: storageBuf(dev, WORDS * capacity * 4),   // words[w * capacity + i]
      aux: storageBuf(dev, capacity * 16),            // aux[i*4 + {scale, bias, len, mdot}]
    };
  }

  _bind(pipe, buffers) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }

  _dispatch(enc, pipe, buffers, dims) {
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, this._bind(pipe, buffers));
    pass.dispatchWorkgroups(dims[0], dims[1] ?? 1, dims[2] ?? 1);
    pass.end();
  }

  // Encodes rows inputOffsetRows.. (count rows of inputBuf) into corpus rows start..start+count-1.
  encode(inputBuf, count, corpus, start, opts = {}) {
    const normalize = (opts.normalize ?? true) ? 1 : 0;
    const inRow0 = opts.inputOffsetRows ?? 0;
    if (start + count > corpus.capacity) throw new Error(`encode: ${start}+${count} exceeds capacity ${corpus.capacity}`);
    const enc = this.device.createCommandEncoder();
    for (let base = 0; base < count; base += MAX_DIM) {
      const n = Math.min(MAX_DIM, count - base);
      const u = uniform(this.device, new Uint32Array([inRow0 + base, start + base, corpus.capacity, normalize]));
      this._dispatch(enc, this.pipes.encode,
        [u, inputBuf, this.rmat, this.rmu, this.cst, corpus.words, corpus.aux], [n, 1, 1]);
    }
    this.device.queue.submit([enc.finish()]);
    corpus.filled = Math.max(corpus.filled, start + count);
  }

  // hidden [B,S,D] f32 + mask [B,S] u32 -> [B,D] f32 masked mean.
  pool(hiddenBuf, maskBuf, B, S) {
    const out = storageBuf(this.device, B * D * 4);
    const enc = this.device.createCommandEncoder();
    for (let base = 0; base < B; base += MAX_DIM) {
      const n = Math.min(MAX_DIM, B - base);
      const u = uniform(this.device, new Uint32Array([S, base, 0, 0]));
      this._dispatch(enc, this.pipes.pool, [u, hiddenBuf, maskBuf, out], [3, n, 1]);
    }
    this.device.queue.submit([enc.finish()]);
    return out;
  }

  // Scores nq queries (f32 [nq, D] in queryBuf) against the first n corpus rows
  // (default: corpus.filled). Returns scores GPUBuffer, layout [nq, capacity].
  search(queryBuf, nq, corpus, opts = {}) {
    const n = opts.n ?? corpus.filled;
    const normalize = (opts.normalize ?? true) ? 1 : 0;
    if (n > corpus.capacity) throw new Error(`search: n=${n} exceeds capacity`);
    const dev = this.device;
    const lut = storageBuf(dev, nq * LUT_STRIDE * 4);
    const scores = storageBuf(dev, nq * corpus.capacity * 4);
    const enc = dev.createCommandEncoder();
    for (let qb = 0; qb < nq; qb += MAX_DIM) {
      const nqq = Math.min(MAX_DIM, nq - qb);
      const u = uniform(dev, new Uint32Array([normalize, qb, LUT_STRIDE, 0]));
      this._dispatch(enc, this.pipes.prep, [u, queryBuf, this.rmat, this.mu, this.cst, lut], [nqq, 1, 1]);
    }
    if (n > 0) {
      for (let qb = 0; qb < nq; qb += MAX_DIM) {
        const nqq = Math.min(MAX_DIM, nq - qb);
        const u = uniform(dev, new Uint32Array([n, corpus.capacity, qb, LUT_STRIDE]));
        this._dispatch(enc, this.pipes.scan,
          [u, corpus.words, corpus.aux, lut, scores], [Math.ceil(n / 256), nqq, 1]);
      }
    }
    dev.queue.submit([enc.finish()]);
    lut.destroy();   // destruction is deferred until the submitted work finishes
    return scores;
  }

  // Sums a [count, D] f32 batch column-wise and adds it into accBuf [D] f32.
  colsum(inputBuf, count, accBuf) {
    const enc = this.device.createCommandEncoder();
    const u = uniform(this.device, new Uint32Array([count, 0, 0, 0]));
    this._dispatch(enc, this.pipes.colsum, [u, inputBuf, accBuf], [3, 1, 1]);
    this.device.queue.submit([enc.finish()]);
  }

  // Copies a GPU buffer of `bytes` bytes to an ArrayBuffer.
  async readBuffer(buf, bytes) {
    const dev = this.device;
    const stage = dev.createBuffer({ size: bytes, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, stage, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await stage.mapAsync(MAP.READ);
    const out = stage.getMappedRange().slice(0);
    stage.unmap();
    stage.destroy();
    return out;
  }

  // Doc-major remex bytes plus per-doc floats for the first n docs.
  async readCorpus(corpus, n) {
    const cap = corpus.capacity;
    const w = new Uint32Array(await this.readBuffer(corpus.words, WORDS * cap * 4));
    const aux = new Float32Array(await this.readBuffer(corpus.aux, cap * 16));
    const packed = new Uint8Array(n * 192);
    const p32 = new Uint32Array(packed.buffer);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < WORDS; k++) p32[i * WORDS + k] = w[k * cap + i];
    }
    const scale = new Float32Array(n), bias = new Float32Array(n);
    const len = new Float32Array(n), mdot = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      scale[i] = aux[i * 4];
      bias[i] = aux[i * 4 + 1];
      len[i] = aux[i * 4 + 2];
      mdot[i] = aux[i * 4 + 3];
    }
    return { packed, scale, bias, len, mdot };
  }

  // scores GPUBuffer [nq, capacity] -> Float32Array [nq, n].
  async readScores(buf, nq, corpus, n) {
    const full = new Float32Array(await this.readBuffer(buf, nq * corpus.capacity * 4));
    const out = new Float32Array(nq * n);
    for (let q = 0; q < nq; q++) {
      out.set(full.subarray(q * corpus.capacity, q * corpus.capacity + n), q * n);
    }
    return out;
  }

  // Top-k on the host from GPU scores. Ties broken by lower doc index.
  async topk(queryBuf, nq, corpus, n, k, opts = {}) {
    const scoresBuf = this.search(queryBuf, nq, corpus, { ...opts, n });
    const s = await this.readScores(scoresBuf, nq, corpus, n);
    scoresBuf.destroy();
    const res = [];
    for (let q = 0; q < nq; q++) {
      const row = s.subarray(q * n, (q + 1) * n);
      const order = Array.from({ length: n }, (_, i) => i);
      order.sort((a, b) => (row[b] - row[a]) || (a - b));
      const top = order.slice(0, k);
      res.push({ idx: Int32Array.from(top), score: Float32Array.from(top, (i) => row[i]) });
    }
    return res;
  }
}
