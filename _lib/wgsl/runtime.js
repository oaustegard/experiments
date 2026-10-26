// Shared runtime for the hand-written WGSL encoders (eg2-wgsl-engine, ettin-native). An engine
// subclasses SeqEngine and supplies its shaders and its forward pass as a list of steps; this file
// holds everything else they had copied from each other: the uniform layout, shader compilation, one
// static uniform and bind group per dispatch, packed variable-length batches, weight upload, and the
// embed / embedTo / probe / destroy surface.
//
//   class MyEngine extends SeqEngine {
//     async _build(bytes) {
//       this.shaders = { ...KERNELS, ...SHADERS };     // kernels.js templates plus the engine's own
//       ...                                            // buffers, then this.step(kind, params, bufs, grid)
//     }                                                // or the kernels.js helpers (matmul, norm, ...)
//   }
//   const eng = await MyEngine.create(device, manifest, bytes, { maxTokens, maxSeqs });
//   const vecs = await eng.embed([ids0, ids1, ...]);   // Float32Array[B * dim]
//
// A step's grid is a function of n = [tokens, sequences] in the batch, evaluated when it is recorded.
// Only the shaders a model's steps use are compiled, after _build, so unused kernel variants cost
// no start-up time. See _lib/wgsl/README.md for the generic-engine plan.

export const MAX_SEQ = 128;
export const NONE = 0xffffffff;

// Every kernel's bindings 0 and 1: the step's static parameters and the batch's sizes.
export const HEADER = /* wgsl */ `
struct P { a: vec4<u32>, b: vec4<u32>, f: vec4<f32> };
struct Dyn { n: vec4<u32> };            // x: tokens in the batch, y: sequences
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> dn: Dyn;
`;

// Readers of the weight file, bound as `W: array<u32>`. wbyte takes a byte index, wf16 a byte offset,
// wf32 a word index.
export const BYTE = /* wgsl */ `
fn wbyte(i: u32) -> u32 { return (W[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
fn wf16(byteOff: u32) -> f32 { return unpack2x16float(W[byteOff >> 2u])[(byteOff >> 1u) & 1u]; }
fn wf32(word: u32) -> f32 { return bitcast<f32>(W[word]); }
`;

const U = GPUBufferUsage;

// Compile { name: wgsl } into { name: pipeline }; a WGSL error names the shader, line and column.
export async function compile(device, shaders) {
  const pipes = {};
  await Promise.all(Object.entries(shaders).map(async ([name, code]) => {
    const module = device.createShaderModule({ code, label: name });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((x) => x.type === "error");
    if (errs.length) throw new Error(`${name} WGSL: ` + errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("; "));
    pipes[name] = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" }, label: name });
  }));
  return pipes;
}

// Split a workgroup count over y when it exceeds the 65535 per-dimension limit.
export function split(n) { return n <= 65535 ? [n] : [65535, Math.ceil(n / 65535)]; }

// Every { offset, length } object inside a manifest, in walk order.
export function tensorsOf(manifest) {
  const out = [];
  const walk = (o) => {
    if (o && typeof o === "object") {
      if ("offset" in o && "length" in o) out.push(o);
      else Object.values(o).forEach(walk);
    }
  };
  walk(manifest);
  return out;
}

export class SeqEngine {
  // data: the weight bytes (ArrayBuffer or typed array). opts.maxTokens bounds one batch (all
  // sequences together), opts.maxSeqs its sequence count; each sequence is at most MAX_SEQ tokens.
  static async create(device, manifest, data, opts = {}) {
    const e = new this(device, manifest, opts);
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    e._owned = [];
    e.dyn = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST, label: "dyn" });
    e._owned.push(e.dyn);
    e.steps = [];
    await e._build(bytes);
    if (!e.b?.out) throw new Error("engine must put its output buffer at this.b.out");
    await e._finish();
    e.read = device.createBuffer({ size: e.maxSeqs * e.dim * 4, usage: U.MAP_READ | U.COPY_DST, label: "readback" });
    e._owned.push(e.read);
    await device.queue.onSubmittedWorkDone();
    return e;
  }

  constructor(device, manifest, opts) {
    this.device = device;
    this.m = manifest;
    this.maxTokens = opts.maxTokens ?? 512;
    this.maxSeqs = opts.maxSeqs ?? 64;
    this.dim = 768;
  }

  // A storage buffer of n f32, freed by destroy().
  buf(n, label, extra = 0) {
    const b = this.device.createBuffer({ size: Math.max(16, n * 4), usage: U.STORAGE | U.COPY_SRC | extra, label });
    this._owned.push(b);
    return b;
  }

  // A storage buffer holding `data` (a typed array), freed by destroy(); padded to a multiple of 4 bytes.
  upload(data, label) {
    const u8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const size = Math.max(16, Math.ceil(u8.byteLength / 4) * 4);
    const b = this.device.createBuffer({ size, usage: U.STORAGE | U.COPY_DST, label });
    const padded = u8.byteLength === size ? u8 : (() => { const x = new Uint8Array(size); x.set(u8); return x; })();
    this.device.queue.writeBuffer(b, 0, padded);
    this._owned.push(b);
    return b;
  }

  // Upload the whole weight file as one buffer and return win(...tensors) -> { base, res }: the
  // binding that covers those tensors and the byte offset it starts at. Below the device's binding
  // limit that is the whole buffer; above it (the full EG2 model's 174 MB), the aligned window
  // holding the tensors, so tensor word offsets are taken relative to `base`.
  uploadWhole(bytes) {
    const dev = this.device;
    const size = Math.ceil(bytes.byteLength / 4) * 4;
    if (size > dev.limits.maxBufferSize) throw new Error(`weights (${size} B) exceed this device's maxBufferSize`);
    this.W = this.upload(bytes, "weights");
    const lim = dev.limits.maxStorageBufferBindingSize, align = dev.limits.minStorageBufferOffsetAlignment;
    return (...ts) => {
      if (size <= lim) return { base: 0, res: this.W };
      const lo = Math.min(...ts.map((x) => x.offset)), hi = Math.max(...ts.map((x) => x.offset + x.length));
      const base = Math.floor(lo / align) * align, n = Math.ceil((hi - base) / 4) * 4;
      if (n > lim) throw new Error(`a dispatch needs ${n} B of weights in one binding, over the ${lim} B limit`);
      return { base, res: { buffer: this.W, offset: base, size: n } };
    };
  }

  // Copy every manifest tensor into one 4-byte-aligned buffer (tensors inside a single-file
  // model.onnx sit at unaligned protobuf offsets) and point the manifest's offsets at it. Returns the
  // packed bytes, for host-side repacking of particular tensors.
  uploadPacked(bytes) {
    const tensors = tensorsOf(this.m);
    let total = 0;
    for (const t of tensors) { t._src = t.offset; t.offset = total; total += Math.ceil(t.length / 4) * 4; }
    const packed = new Uint8Array(Math.max(16, total));
    for (const t of tensors) packed.set(bytes.subarray(t._src, t._src + t.length), t.offset);
    this.W = this.upload(packed, "weights");
    return packed;
  }

  // Append a dispatch: params { a: u32[<=4], b: u32[<=4], f: f32[<=4] } go into its own uniform;
  // bufs bind from 2 up (a GPUBuffer, or a { buffer, offset, size } window); grid(n) -> workgroups.
  // The bind group is made in _finish, once the pipelines exist.
  step(kind, params, bufs, grid) {
    if (!this.shaders?.[kind]) throw new Error(`no shader named ${kind}`);
    const dev = this.device;
    const pad = (x) => (x ?? []).concat(Array(4 - (x ?? []).length).fill(0));
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 8).set(pad(params.a).concat(pad(params.b)));
    new Float32Array(u, 32, 4).set(pad(params.f));
    const ub = dev.createBuffer({ size: 48, usage: U.UNIFORM | U.COPY_DST, label: kind });
    dev.queue.writeBuffer(ub, 0, u);
    this._owned.push(ub);
    const entries = [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: this.dyn } }]
      .concat(bufs.map((b, i) => ({ binding: i + 2, resource: b.buffer ? b : { buffer: b } })));
    this.steps.push({ kind, entries, grid });
  }

  // Compile the shaders the steps use and make their bind groups.
  async _finish() {
    const used = [...new Set(this.steps.map((s) => s.kind))];
    this.pipes = await compile(this.device, Object.fromEntries(used.map((k) => [k, this.shaders[k]])));
    for (const s of this.steps) {
      s.bg = this.device.createBindGroup({ layout: this.pipes[s.kind].getBindGroupLayout(0), entries: s.entries, label: s.kind });
      delete s.entries;
    }
  }

  // Pack the batch (lists of token ids) into this.b.tok ([id, position, sequence, 0] per token) and
  // this.b.seqs ([start, length, 0, 0] per sequence), then record the first nSteps dispatches.
  _record(enc, batch, nSteps = Infinity) {
    const T = batch.reduce((s, x) => s + x.length, 0), B = batch.length;
    if (!B || T > this.maxTokens || B > this.maxSeqs) throw new Error(`batch of ${B} sequences / ${T} tokens exceeds the engine's limits`);
    const tok = new Uint32Array(T * 4), seqs = new Uint32Array(B * 4);
    let o = 0;
    batch.forEach((ids, b) => {
      if (!ids.length || ids.length > MAX_SEQ) throw new Error(`sequence length ${ids.length} not in 1..${MAX_SEQ}`);
      seqs.set([o, ids.length], b * 4);
      ids.forEach((id, j) => tok.set([id, j, b], (o + j) * 4));
      o += ids.length;
    });
    const q = this.device.queue;
    q.writeBuffer(this.b.tok, 0, tok);
    q.writeBuffer(this.b.seqs, 0, seqs);
    q.writeBuffer(this.dyn, 0, new Uint32Array([T, B, 0, 0]));
    const pass = enc.beginComputePass();
    const n = [T, B];
    for (const s of this.steps.slice(0, nSteps)) {
      pass.setPipeline(this.pipes[s.kind]);
      pass.setBindGroup(0, s.bg);
      pass.dispatchWorkgroups(...s.grid(n));
    }
    pass.end();
    return B;
  }

  // Embed sequences of token ids; returns a Float32Array of B * dim values.
  async embed(batch) {
    const enc = this.device.createCommandEncoder();
    const B = this._record(enc, batch), n = B * this.dim * 4;
    enc.copyBufferToBuffer(this.b.out, 0, this.read, 0, n);
    this.device.queue.submit([enc.finish()]);
    await this.read.mapAsync(GPUMapMode.READ, 0, n);
    const out = new Float32Array(this.read.getMappedRange(0, n).slice(0));
    this.read.unmap();
    return out;
  }

  // Embed and copy the vectors into `target` (same device) without reading them back.
  // sync "none": return once submitted (later submits on this queue see the vectors);
  // "map" (default): wait by mapping 16 bytes, which Chrome resolves promptly;
  // "done": wait on queue.onSubmittedWorkDone().
  async embedTo(batch, target, offset = 0, sync = "map") {
    const enc = this.device.createCommandEncoder();
    const B = this._record(enc, batch);
    enc.copyBufferToBuffer(this.b.out, 0, target, offset, B * this.dim * 4);
    if (sync === "map") enc.copyBufferToBuffer(this.b.out, 0, this.read, 0, 16);
    this.device.queue.submit([enc.finish()]);
    if (sync === "done") await this.device.queue.onSubmittedWorkDone();
    if (sync === "map") { await this.read.mapAsync(GPUMapMode.READ, 0, 16); this.read.unmap(); }
  }

  // Debugging: run the first nSteps dispatches and read `count` floats of the activation buffer this.b[name].
  async probe(batch, nSteps, name, count) {
    const enc = this.device.createCommandEncoder();
    this._record(enc, batch, nSteps);
    const rb = this.device.createBuffer({ size: count * 4, usage: U.MAP_READ | U.COPY_DST });
    enc.copyBufferToBuffer(this.b[name], 0, rb, 0, count * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.destroy();
    return out;
  }

  destroy() {
    for (const b of this._owned) b.destroy();
    this._owned = [];
  }
}
