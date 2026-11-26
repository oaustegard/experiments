// OCRia-nano-1's OCR nets (detector, script router, recognizers) on WebGPU with hand-written WGSL: no
// onnxruntime, no wasm. Runs the op programs run/export_gpu.py writes (models/<model>/program.json +
// weights.bin). The call mirrors LiteParse's SessionPool::infer:
//
//   import { OcriaGpu } from "./engine.js";
//   const eng = await OcriaGpu.load(baseUrl, { models: ["det", "router", "rec_latin"] });  // opts.device optional
//   const out = await eng.infer("det", [1, 3, H, W], x, ["prob"]);     // { prob: { shape: [1, 1, H/2, W/2], data } }
//   const r = await eng.infer("router", [N, 1, 32, 512], x);           // { group: {shape: [N, 4]}, flip: {shape: [N, 2]} }
//   const t = await eng.infer("rec_latin", [N, 1, 32, W], x);          // { ids: {shape: [N, T], data: Int32Array}, conf }
//
// baseUrl is the folder holding <model>/program.json (default: models/ beside this file). Inputs are
// float32 NCHW; see README.md for each model's normalisation. Calls are queued, one forward at a time:
// every dispatch of a forward goes into one command encoder, activations come from a pool of buffers
// reused across calls, and only the requested outputs are read back. eng.stats[model] counts calls and
// wall milliseconds (upload to readback).

import * as K from "./kernels.js";

export const FORMAT = "ocria-gpu/program v1";
const ACT = { relu: 1, sigmoid: 2 };
const SLOT = 256;   // uniform slot stride (minUniformBufferOffsetAlignment)
const U32 = 24;     // words per uniform block (6 x vec4)

async function makeDevice() {
  if (!globalThis.navigator || !navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const L = adapter.limits;
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
    },
  });
}

// One weight tensor as float32: i8 codes * per-row scale (dim 0), or f32 as stored.
function tensorF32(raw, e) {
  const n = e.shape.reduce((a, b) => a * b, 1);
  if (e.kind === "f32") return new Float32Array(raw, e.off, n);
  if (e.kind !== "i8") throw new Error(`unknown tensor kind ${e.kind}`);
  const rows = e.shape[0], cols = n / rows;
  const q = new Int8Array(raw, e.off, n), s = new Float32Array(raw, e.soff, rows);
  const out = new Float32Array(n);
  for (let r = 0; r < rows; r++) {
    const sc = s[r], b = r * cols;
    for (let c = 0; c < cols; c++) out[b + c] = q[b + c] * sc;
  }
  return out;
}

// [rows, cols] -> [cols, rows]
function transpose(a, rows, cols) {
  const t = new Float32Array(a.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) t[c * rows + r] = a[r * cols + c];
  return t;
}

class Model {
  constructor(name, prog, wbuf) {
    this.name = name;
    this.prog = prog;
    this.w = wbuf;
  }
}

export class OcriaGpu {
  static async load(baseUrl, opts = {}) {
    const base = baseUrl ? String(baseUrl).replace(/\/?$/, "/") : new URL("./models/", import.meta.url).href;
    const device = opts.device || await makeDevice();
    const eng = new OcriaGpu(device, opts);
    const names = opts.models || ["det", "router", "rec_latin"];
    const f = opts.fetch || fetch;
    await Promise.all(names.map((m) => eng._loadModel(m, base + m + "/", f)));
    await eng._compile();
    return eng;
  }

  constructor(device, opts = {}) {
    this.device = device;
    this.models = {};
    this.pipes = new Map();      // WGSL source -> pipeline
    this.free = new Map();       // size class -> GPUBuffer[]
    this.queue = Promise.resolve();
    this.stats = {};
    this.lost = null;
    device.lost?.then((info) => { this.lost = info; });
    this.dummy = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  }

  async _loadModel(name, url, f) {
    const prog = await (await f(url + "program.json")).json();
    if (prog.format !== FORMAT) throw new Error(`${name}: program is ${prog.format}; this engine reads ${FORMAT}`);
    const raw = await (await f(url + prog.weights.file)).arrayBuffer();
    if (raw.byteLength !== prog.weights.bytes) throw new Error(`${name}: weights.bin is ${raw.byteLength} bytes, program says ${prog.weights.bytes}`);
    const T = prog.weights.tensors;
    // One float32 buffer: per op its weights (1x1 / Gemm / head transposed to [cin, cout]) and bias.
    const parts = [];
    let total = 0;
    const put = (a) => { const at = total; parts.push([at, a]); total += a.length; total += (-total) & 3; return at; };
    for (const op of prog.ops) {
      if (!op.w) continue;
      const w = tensorF32(raw, T[op.w]);
      if (op.op === "conv") op.wo = put(w);
      else op.wo = put(transpose(w, op.cout, op.cin));
      op.bo = put(tensorF32(raw, T[op.b]));
    }
    const all = new Float32Array(Math.max(4, total));
    for (const [at, a] of parts) all.set(a, at);
    const buf = this.device.createBuffer({ size: all.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, all);
    this.models[name] = new Model(name, prog, buf);
    this.stats[name] = { calls: 0, ms: 0 };
  }

  _kernel(op) {
    if (op.op === "conv") return K.CONV;
    if (op.op === "pw" || op.op === "gemm") return op.cout <= 32 ? K.PWS(op.cout <= 4 ? 4 : op.cout <= 16 ? 16 : 32) : K.PW;
    if (op.op === "head") return K.HEADK;
    if (op.op === "gray") return K.GRAY;
    if (op.op === "add" || op.op === "fq") return K.EW;
    if (op.op === "resize") return K.RESIZE;
    if (op.op === "concat") return K.COPY;
    if (op.op === "meanmax") return K.MEANMAX;
    throw new Error(`unknown op ${op.op}`);
  }

  async _compile() {
    const srcs = new Set();
    for (const m of Object.values(this.models)) for (const op of m.prog.ops) srcs.add(this._kernel(op));
    await Promise.all([...srcs].filter((s) => !this.pipes.has(s)).map(async (code) => {
      const module = this.device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errs = info.messages.filter((x) => x.type === "error");
      if (errs.length) throw new Error("WGSL: " + errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("; "));
      this.pipes.set(code, await this.device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } }));
    }));
  }

  _get(bytes) {
    let s = 256;
    while (s < bytes) s *= 2;
    const list = this.free.get(s);
    if (list && list.length) return list.pop();
    return this.device.createBuffer({ size: s, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  _put(b) {
    const list = this.free.get(b.size);
    if (list) list.push(b); else this.free.set(b.size, [b]);
  }

  /** Drop pooled activation buffers (they are kept between calls for reuse). */
  trim() {
    for (const list of this.free.values()) for (const b of list) b.destroy();
    this.free.clear();
  }

  infer(model, shape, data, outputNames) {
    const run = this.queue.then(() => this._infer(model, shape, data, outputNames));
    this.queue = run.catch(() => {});
    return run;
  }

  async _infer(model, shape, data, outputNames) {
    if (this.lost) throw new Error(`WebGPU device lost: ${this.lost.message || this.lost.reason}`);
    const m = this.models[model];
    if (!m) throw new Error(`model ${model} not loaded`);
    const t0 = performance.now();
    const { prog } = m;
    const dev = this.device;
    if (shape.length !== 4) throw new Error(`${model}: input shape must be [N, C, H, W]`);
    if (!(data instanceof Float32Array)) data = Float32Array.from(data);
    const [N, C0, H0, W0] = shape;
    if (C0 !== prog.tensors[0].c) throw new Error(`${model}: input has ${C0} channels, model takes ${prog.tensors[0].c}`);
    if (data.length !== N * C0 * H0 * W0) throw new Error(`${model}: data has ${data.length} values, shape needs ${N * C0 * H0 * W0}`);
    const fixed = prog.input.shape;
    for (let i = 1; i < 4; i++) if (typeof fixed[i] === "number" && fixed[i] !== shape[i]) throw new Error(`${model}: input dim ${i} must be ${fixed[i]}`);
    const want = outputNames && outputNames.length ? outputNames : Object.keys(prog.outputs);
    for (const o of want) if (!prog.outputs[o]) throw new Error(`${model}: no output ${o}`);

    // shapes: dims[t] = [c, h, w]
    const tens = prog.tensors;
    const dims = new Array(tens.length);
    dims[0] = [C0, H0, W0];
    const opIns = (op) => op.op === "add" || op.op === "concat" ? op.ins : (op.res !== undefined ? [op.in, op.res] : [op.in]);
    const opOuts = (op) => Array.isArray(op.out) ? op.out : [op.out];
    for (const op of prog.ops) {
      const [c, h, w] = dims[op.op === "add" || op.op === "concat" ? op.ins[0] : op.in];
      let d;
      switch (op.op) {
        case "conv": {
          const oh = Math.floor((h + op.p[0] + op.p[2] - op.k[0]) / op.s[0]) + 1;
          const ow = Math.floor((w + op.p[1] + op.p[3] - op.k[1]) / op.s[1]) + 1;
          d = [op.cout, oh, ow]; break;
        }
        case "pw": d = [op.cout, h, w]; break;
        case "gemm": if (h !== 1 || w !== 1) throw new Error("gemm input must be [N, C]"); d = [op.cout, 1, 1]; break;
        case "gray": d = [1, h, w]; break;
        case "fq": case "add": d = [c, h, w]; break;
        case "resize": d = [c, h * op.scale, w * op.scale]; break;
        case "concat": d = [op.ins.reduce((a, t) => a + dims[t][0], 0), h, w]; break;
        case "meanmax": if (h !== 1) throw new Error("meanmax input must have H = 1"); d = [2 * c, 1, 1]; break;
        case "head": if (h !== 1) throw new Error("head input must have H = 1"); d = [1, 1, w]; break;
        default: throw new Error(`unknown op ${op.op}`);
      }
      for (const t of opOuts(op)) dims[t] = d;
      if (tens[op.in]?.h1 && dims[op.in][1] !== 1) throw new Error(`${model}: squeezed tensor has H = ${dims[op.in][1]}`);
    }
    const last = new Array(tens.length).fill(-1);
    prog.ops.forEach((op, i) => { for (const t of opIns(op)) last[t] = i; });
    for (const o of Object.values(prog.outputs)) last[o.tensor] = Infinity;
    const elems = (t) => N * dims[t][0] * dims[t][1] * dims[t][2];

    // uniforms: one slot per dispatch
    let nd = 0;
    for (const op of prog.ops) nd += op.op === "concat" ? op.ins.length : 1;
    const ubuf = dev.createBuffer({ size: nd * SLOT, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const uhost = new ArrayBuffer(nd * SLOT);
    let slot = 0;
    const uni = (a, b = [0, 0, 0, 0], c = [0, 0, 0, 0], d = [0, 0, 0, 0], op = null, f = null) => {
      const u = new Uint32Array(uhost, slot * SLOT, U32);
      const fv = new Float32Array(uhost, slot * SLOT, U32);
      u.set(a, 0); u.set(b, 4); u.set(c, 8); u.set(d, 12);
      if (op) {
        u[16] = ACT[op.act1] || 0; u[17] = ACT[op.act2] || 0; u[18] = op.res !== undefined || op.op === "add" ? 1 : 0;
        u[19] = (op.fq1 ? 1 : 0) | (op.fq2 ? 2 : 0);
        if (op.fq1) { fv[20] = op.fq1[0]; fv[21] = op.fq1[1]; }
        if (op.fq2) { fv[22] = op.fq2[0]; fv[23] = op.fq2[1]; }
      }
      if (f) fv.set(f, 20);
      return { buffer: ubuf, offset: (slot++) * SLOT, size: U32 * 4 };
    };

    // eng.profile = true: each op is submitted on its own and timed (eng.lastProfile); for tuning only
    const prof = this.profile ? [] : null;
    let enc = dev.createCommandEncoder();
    let pass = enc.beginComputePass();
    const dispatch = (code, binds, grid) => {
      const pipe = this.pipes.get(code);
      if (!pipe) throw new Error("pipeline not compiled");
      for (const g of grid) if (g > 65535) throw new Error(`${model}: dispatch of ${g} workgroups exceeds 65535`);
      const bg = dev.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: binds.map((b, i) => ({ binding: i, resource: b.buffer && b.size !== undefined && b.offset !== undefined ? b : { buffer: b } })),
      });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(grid[0], grid[1] || 1, grid[2] || 1);
    };
    const grid1 = (n) => { const nb = Math.ceil(n / 64); const gx = Math.min(nb, 32768); return [gx, Math.ceil(nb / gx), gx * 64]; };

    const live = new Map();
    const xb = this._get(data.byteLength);
    dev.queue.writeBuffer(xb, 0, data.buffer, data.byteOffset, data.byteLength);
    live.set(0, xb);
    const alloc = (t) => { const b = this._get(elems(t) * 4); live.set(t, b); return b; };
    const W = m.w;

    for (let i = 0; i < prog.ops.length; i++) {
      const op = prog.ops[i];
      const code = this._kernel(op);
      const slot0 = slot;
      const [c, h, w] = dims[op.op === "add" || op.op === "concat" ? op.ins[0] : op.in];
      const res = op.res !== undefined ? live.get(op.res) : this.dummy;
      if (op.op === "conv") {
        const [oc, oh, ow] = dims[op.out];
        const icg = op.cin / op.g;
        if (N * oc > 65535) throw new Error(`${model}: batch too large for conv (N * channels > 65535)`);
        const out = alloc(op.out);
        dispatch(code, [uni([icg, h, w, oc], [oh, ow, oc / op.g, op.k[0]], [op.k[1], op.s[0], op.s[1], op.p[0]], [op.p[1], op.wo, op.bo, op.cin], op),
          live.get(op.in), W, res, out], [Math.ceil(ow / 8), Math.ceil(oh / 8), N * oc]);
      } else if (op.op === "pw" || op.op === "gemm") {
        const P = h * w;
        const out = alloc(op.out);
        if (op.cout <= 32) {
          const [gx, gy, nx] = grid1(N * P);
          dispatch(code, [uni([op.cin, op.cout, P, op.wo], [op.bo, N * P, nx, 0], undefined, undefined, op), live.get(op.in), W, res, out], [gx, gy, 1]);
        } else {
          dispatch(code, [uni([op.cin, op.cout, P, op.wo], [op.bo, 0, 0, 0], undefined, undefined, op), live.get(op.in), W, res, out],
            [Math.ceil(P / 64), Math.ceil(op.cout / 64), N]);
        }
      } else if (op.op === "head") {
        const T = w;
        const ids = alloc(op.out[0]), cf = alloc(op.out[1]);
        dispatch(code, [uni([op.cin, op.cout, T, op.wo], [op.bo, op.fq1 ? 1 : 0, 0, 0], undefined, undefined, null, op.fq1 ? [op.fq1[0], op.fq1[1], 0, 0] : null),
          live.get(op.in), W, ids, cf], [Math.ceil(T / 64), N, 1]);
      } else if (op.op === "gray") {
        const out = alloc(op.out);
        const [gx, gy, nx] = grid1(N * h * w);
        dispatch(code, [uni([h * w, N * h * w, nx, 0], undefined, undefined, undefined, null, [...op.coef, 0]), live.get(op.in), out], [gx, gy, 1]);
      } else if (op.op === "add" || op.op === "fq") {
        const out = alloc(op.out);
        const n = elems(op.out);
        const [gx, gy, nx] = grid1(n);
        const e = op.op === "fq" ? { fq1: op.fq } : op;
        dispatch(code, [uni([n, nx, 0, 0], undefined, undefined, undefined, e),
          live.get(op.op === "add" ? op.ins[0] : op.in), op.op === "add" ? live.get(op.ins[1]) : this.dummy, out], [gx, gy, 1]);
      } else if (op.op === "resize") {
        const [, oh, ow] = dims[op.out];
        const out = alloc(op.out);
        dispatch(code, [uni([h, w, oh, ow], [op.scale, N * c, 0, 0]), live.get(op.in), out], [Math.ceil(ow / 8), Math.ceil(oh / 8), N * c]);
      } else if (op.op === "concat") {
        const out = alloc(op.out);
        const ct = dims[op.out][0];
        let coff = 0;
        for (const t of op.ins) {
          const cs = dims[t][0];
          dispatch(code, [uni([h, w, cs, ct], [coff, 0, 0, 0]), live.get(t), out], [Math.ceil(w / 8), Math.ceil(h / 8), N * cs]);
          coff += cs;
        }
      } else if (op.op === "meanmax") {
        const out = alloc(op.out);
        dispatch(code, [uni([c, w, N * c, 0]), live.get(op.in), out], [Math.ceil(N * c / 64), 1, 1]);
      }
      for (const t of opIns(op)) {
        if (last[t] === i && live.has(t)) { this._put(live.get(t)); live.delete(t); }
      }
      if (prof) {
        pass.end();
        dev.queue.writeBuffer(ubuf, slot0 * SLOT, uhost, slot0 * SLOT, (slot - slot0) * SLOT);
        const tp = performance.now();
        dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
        const d = dims[Array.isArray(op.out) ? op.out[0] : op.out];
        prof.push({ i, op: op.op, cin: op.cin, cout: op.cout ?? d[0], out: [N, ...d], ms: performance.now() - tp });
        enc = dev.createCommandEncoder();
        pass = enc.beginComputePass();
      }
    }
    pass.end();
    if (!prof) dev.queue.writeBuffer(ubuf, 0, uhost);
    if (prof) this.lastProfile = prof;

    const reads = [];
    for (const o of want) {
      const t = prog.outputs[o].tensor;
      const bytes = elems(t) * 4;
      const st = dev.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(live.get(t), 0, st, 0, Math.ceil(bytes / 4) * 4);
      reads.push([o, t, st, bytes]);
    }
    dev.queue.submit([enc.finish()]);
    const out = {};
    await Promise.all(reads.map(async ([o, t, st, bytes]) => {
      await st.mapAsync(GPUMapMode.READ);
      const copy = st.getMappedRange(0, Math.ceil(bytes / 16) * 16).slice(0, bytes);
      st.unmap();
      st.destroy();
      const isI = tens[t].dtype === "i32";
      const data = isI ? new Int32Array(copy) : new Float32Array(copy);
      const [c, h, w] = dims[t];
      const rank = prog.outputs[o].rank;
      const shp = rank === 4 ? [N, c, h, w] : [N, c * h * w];
      out[o] = { shape: shp, data };
    }));
    for (const b of live.values()) this._put(b);
    live.clear();
    ubuf.destroy();
    const ms = performance.now() - t0;
    this.stats[model].calls++;
    this.stats[model].ms += ms;
    this.lastMs = ms;
    return out;
  }
}

export default OcriaGpu;
