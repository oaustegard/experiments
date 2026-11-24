// PP-OCRv6 text-line recognizer (tiny or small) on WebGPU, no onnxruntime. Runs the op program that
// run/export_rec.py writes (web/rec/models/<name>/{manifest.json, weights.bin, dict.txt}); the spec is
// run/ref_rec.py.
//
//   const rec = await RecNet.load(gpu, "models/tiny/");   // gpu: a GPUDevice, an object with .device, or null
//   const out = await rec.run(x, B, W);   // x: Float32Array [B, 3, 48, W], preprocessed (see below)
//   out.probs                             // Float32Array [B, T, V], softmax over V per step
//   rec.decode(out)                       // greedy CTC text per line: string[B]
//
// Input, as PaddleOCR's RecResizeImg: resize the line to height 48 keeping its aspect ratio, BGR,
// x / 255, (x - 0.5) / 0.5; lines of a batch are zero-padded on the right to the widest (PaddleOCR pads
// to at least 320 wide). T follows from W through the stride-2 convs and the final pool (about W / 8).
//
// Weights are dequantised once at load (f32, f16, int8 with a scale per output channel, 4-bit groups)
// into one f32 buffer, so every kernel reads f32. Every dispatch of one forward goes into one command
// encoder; activation buffers come from a size-class pool and are reused in order within the pass.

import * as K from "./kernels.js";

export const FORMAT = "ppocr-pack/rec-program v1";
const ACT = { none: 0, relu: 1, hsig: 2, gelu: 3, sigmoid: 4, hswish: 5, swish: 6 };
const TILE = 8;

// Uniform block from ints (u32) and {f} entries (f32), padded to 16 bytes.
function uniformBytes(vals) {
  const dv = new DataView(new ArrayBuffer(Math.max(16, Math.ceil(vals.length / 4) * 16)));
  vals.forEach((v, i) => {
    if (typeof v === "object") dv.setFloat32(4 * i, v.f, true);
    else dv.setUint32(4 * i, v >>> 0, true);
  });
  return dv.buffer;
}

// IEEE half to double (exact).
function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// Dequantises weights.bin into one Float32Array and sets tensor.at (its f32 element offset) on every
// manifest tensor. The value of a quantised weight is code * scale rounded once to f32, as in
// run/export_rec.py dequant() (the double product is exact, the Float32Array store rounds it).
export function unpackParams(man, raw) {
  let total = 0;
  for (const t of man.tensors) {
    t.n = t.shape.reduce((a, b) => a * b, 1);
    t.at = total;
    total += t.n;
  }
  const out = new Float32Array(total);
  const u8 = new Uint8Array(raw);
  for (const t of man.tensors) {
    const { n, at } = t;
    if (t.fmt === "f32") {
      out.set(new Float32Array(raw, t.off, n), at);
    } else if (t.fmt === "f16") {
      const h = new Uint16Array(raw, t.off, n);
      for (let i = 0; i < n; i++) out[at + i] = f16(h[i]);
    } else if (t.fmt === "i8") {
      const q = new Int8Array(raw, t.off, n);
      const s = Array.from(new Uint16Array(raw, t.sc, t.nsc), f16);
      if (t.axis === 0) {
        const per = n / t.nsc;
        for (let r = 0; r < t.nsc; r++) for (let j = 0; j < per; j++) out[at + r * per + j] = q[r * per + j] * s[r];
      } else {
        const N = t.nsc;
        for (let i = 0; i < n; i++) out[at + i] = q[i] * s[i % N];
      }
    } else if (t.fmt === "t4") {
      // codes row-major [K, N], two per byte (low nibble first, + 8); scales [ceil(K / group), N]
      const N = t.shape[1], G = t.group;
      const s = Array.from(new Uint16Array(raw, t.sc, t.nsc), f16);
      for (let i = 0; i < n; i++) {
        const nib = (u8[t.off + (i >> 1)] >> (4 * (i & 1))) & 15;
        const k = Math.floor(i / N);
        out[at + i] = (nib - 8) * s[Math.floor(k / G) * N + (i - k * N)];
      }
    } else {
      throw new Error(`tensor ${t.name}: unknown format ${t.fmt}`);
    }
  }
  return out;
}

// A device with the adapter's buffer limits (the [B, T, V] output of a long batch is large).
export async function requestDevice() {
  if (!navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const L = adapter.limits;
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
    },
  });
}

// Resampling coefficients of PIL's bilinear filter along one axis (Pillow precompute_coeffs: a triangle
// of support 1, widened by the downscale factor, so a downscale averages instead of skipping pixels).
function pilBilinear(inSize, outSize) {
  const scale = inSize / outSize, fs = Math.max(scale, 1), support = fs;
  const taps = [];
  for (let o = 0; o < outSize; o++) {
    const center = (o + 0.5) * scale;
    const lo = Math.max(0, Math.trunc(center - support + 0.5)), hi = Math.min(inSize, Math.trunc(center + support + 0.5));
    const w = [];
    let sum = 0;
    for (let i = lo; i < hi; i++) { const v = Math.max(0, 1 - Math.abs((i - center + 0.5) / fs)); w.push(v); sum += v; }
    taps.push({ lo, w: w.map((v) => (sum ? v / sum : 0)) });
  }
  return taps;
}

// Line crops to the recognizer's input, as PaddleOCR prepares a batch: each crop resized to height 48
// keeping its aspect ratio (width ceil(48 w / h), at most maxWidth; PIL-style bilinear, rounded to 8 bits
// after each pass as PIL does), BGR, x / 255, (x - 0.5) / 0.5, zero-padded on the right to
// W = max(minWidth, widest). crops: [{ data: RGBA bytes, width, height }].
// Returns { x, B, W, widths } for run(x, B, W).
export function lineInput(crops, { minWidth = 320, maxWidth = 3200, H = 48 } = {}) {
  const widths = crops.map((c) => Math.max(1, Math.min(maxWidth, Math.ceil(H * c.width / c.height))));
  const W = Math.max(minWidth, ...widths);
  const x = new Float32Array(crops.length * 3 * H * W);
  crops.forEach((c, b) => {
    const rw = widths[b], { width: iw, height: ih, data } = c;
    const hx = pilBilinear(iw, rw), hy = pilBilinear(ih, H);
    const tmp = new Float32Array(ih * rw * 3);   // horizontal pass, rounded to 8 bits
    for (let y = 0; y < ih; y++) {
      for (let o = 0; o < rw; o++) {
        const { lo, w } = hx[o];
        for (let ch = 0; ch < 3; ch++) {
          let s = 0;
          for (let k = 0; k < w.length; k++) s += w[k] * data[(y * iw + lo + k) * 4 + ch];
          tmp[(y * rw + o) * 3 + ch] = Math.min(255, Math.max(0, Math.round(s)));
        }
      }
    }
    const plane = H * W;
    for (let o = 0; o < H; o++) {
      const { lo, w } = hy[o];
      for (let xx = 0; xx < rw; xx++) {
        for (let ch = 0; ch < 3; ch++) {
          let s = 0;
          for (let k = 0; k < w.length; k++) s += w[k] * tmp[((lo + k) * rw + xx) * 3 + ch];
          const v = Math.min(255, Math.max(0, Math.round(s)));
          x[(b * 3 + (2 - ch)) * plane + o * W + xx] = (v / 255 - 0.5) / 0.5;   // RGB channel ch -> BGR plane 2 - ch
        }
      }
    }
  });
  return { x, B: crops.length, W, widths };
}

// 1-D dispatch of n threads in 64-wide workgroups: [gx, gy, nx] with index i = gid.x + gid.y * nx.
function grid1(n) {
  const nb = Math.max(1, Math.ceil(n / 64));
  const gx = Math.min(nb, 65535);
  const gy = Math.ceil(nb / gx);
  return [gx, gy, gx * 64];
}

export class RecNet {
  // gpu: a GPUDevice, an object holding one as .device (e.g. a Gpu helper), or null for a new device.
  static async load(gpu, base) {
    const device = gpu ? (gpu.device || gpu) : await requestDevice();
    if (!base.endsWith("/")) base += "/";
    const get = async (f) => {
      const r = await fetch(base + f);
      if (!r.ok) throw new Error(`${base}${f}: HTTP ${r.status}`);
      return r;
    };
    const man = await (await get("manifest.json")).json();
    if (man.format !== FORMAT) throw new Error(`rec program is ${man.format}; this engine reads ${FORMAT}`);
    const [raw, text] = await Promise.all([get(man.weights.file).then((r) => r.arrayBuffer()), get(man.dict.file).then((r) => r.text())]);
    if (raw.byteLength !== man.weights.bytes) throw new Error(`weights.bin is ${raw.byteLength} bytes, manifest says ${man.weights.bytes}`);
    const lines = text.split("\n");
    if (lines[lines.length - 1] === "") lines.pop();
    if (lines.length !== man.dict.entries) throw new Error(`dict.txt has ${lines.length} entries, manifest says ${man.dict.entries}`);
    const chars = [""].concat(lines, man.dict.space_appended ? [" "] : []);
    if (chars.length !== man.V) throw new Error(`dictionary gives ${chars.length} classes, the model ${man.V}`);
    return new RecNet(device, man, unpackParams(man, raw), chars);
  }

  constructor(device, man, params, chars) {
    this.device = device;
    this.man = man;
    this.chars = chars;   // class index -> text; 0 is the CTC blank
    this.V = man.V;
    this.H = man.input.h;
    this.w = device.createBuffer({ size: Math.ceil(params.byteLength / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.w, 0, params);
    this.pipelines = new Map();
    this.free = new Map();   // size class (power of 2 bytes) -> GPUBuffer[]
    for (const op of man.ops) if (op.op === "attn" && op.hd > 32) throw new Error(`attention head dim ${op.hd} > 32`);
  }

  pipeline(code) {
    let p = this.pipelines.get(code);
    if (!p) {
      p = this.device.createComputePipeline({ layout: "auto", compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" } });
      this.pipelines.set(code, p);
    }
    return p;
  }

  get(bytes) {
    let s = 256;
    while (s < bytes) s *= 2;
    if (s > this.device.limits.maxStorageBufferBindingSize) s = Math.ceil(bytes / 256) * 256;   // exact size near the limit
    const list = this.free.get(s);
    if (list && list.length) return list.pop();
    return this.device.createBuffer({ size: s, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  put(b) {
    const list = this.free.get(b.size);
    if (list) list.push(b); else this.free.set(b.size, [b]);
  }

  // Records one dispatch into the open pass. vals: the uniform block; bufs: bindings 1.. in order.
  dispatch(code, vals, bufs, grid) {
    const data = uniformBytes(vals);
    const u = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(u, 0, data);
    this.uniforms.push(u);
    const p = this.pipeline(code);
    const entries = [u, ...bufs].map((b, i) => ({ binding: i, resource: { buffer: b } }));
    this.pass.setPipeline(p);
    this.pass.setBindGroup(0, this.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries }));
    this.pass.dispatchWorkgroups(grid[0], grid[1] || 1, grid[2] || 1);
  }

  // Conv entry of the manifest (also a squeeze-excite fc) over B lines of ih x iw.
  conv(e, inB, outB, B, ih, iw, oh, ow) {
    const [O, icg, kh, kw] = e.wshape;
    const g = e.g;
    const blocked = (O / g) % 4 === 0;   // four output channels per invocation when they share a group
    const t = this.man.tensors;
    this.dispatch(blocked ? K.CONV4 : K.CONV, [icg * g, ih, iw, O, oh, ow, icg, O / g, kh, kw, e.s[0], e.s[1], e.pads[0], e.pads[1],
      e.d[0], e.d[1], ACT[e.act], t[e.w].at, t[e.b].at, B, { f: e.ha }, { f: e.hb }, 0, 0],
    [inB, outB, this.w], [Math.ceil(ow / TILE), Math.ceil(oh / TILE), B * (blocked ? O / 4 : O)]);
  }

  // Forward pass. x: Float32Array [B, 3, 48, W]. Returns { B, T, V, probs: Float32Array [B, T, V] }.
  // trace(i, op, data, dims), if given, receives every op's output (read back; for debugging only).
  async run(x, B, W, { trace = null } = {}) {
    const { device, man } = this;
    const H = this.H, bufs = man.buffers, ops = man.ops, T_ = man.tensors;
    if (x.length !== B * 3 * H * W) throw new Error(`x has ${x.length} values; [${B}, 3, ${H}, ${W}] needs ${B * 3 * H * W}`);
    const inputs = (op) => op.ins || [op.in];
    const last = new Array(bufs.length).fill(-1);
    ops.forEach((op, i) => { for (const b of inputs(op)) last[b] = i; });
    last[man.output.buf] = Infinity;
    const dims = [];   // buffer id -> [h, w] per line (tc buffers: [1, T])
    const live = new Map();
    const alloc = (id, h, w) => {
      dims[id] = [h, w];
      const bytes = B * bufs[id].c * h * w * 4;
      if (bytes > device.limits.maxStorageBufferBindingSize) throw new Error(`buffer ${id} needs ${bytes} bytes; split the batch`);
      const b = this.get(bytes);
      live.set(id, b);
      return b;
    };
    const xb = alloc(0, H, W);
    device.queue.writeBuffer(xb, 0, x);
    this.uniforms = [];
    let enc = device.createCommandEncoder();
    this.pass = enc.beginComputePass();
    const flush = () => { this.pass.end(); device.queue.submit([enc.finish()]); enc = device.createCommandEncoder(); this.pass = enc.beginComputePass(); };

    for (let i = 0; i < ops.length; i++) {
      const op = ops[i];
      const c = bufs[op.out].c;
      if (op.op === "conv" || op.op === "maxpool" || op.op === "avgpool") {
        const [ih, iw] = dims[op.in];
        const [pt, pl, pb, pr] = op.pads;
        const span = (n, k, s, d, a, b) => Math.floor((n + a + b - d * (k - 1) - 1) / s) + 1;
        const d = op.d || [1, 1];
        const oh = span(ih, op.k[0], op.s[0], d[0], pt, pb), ow = span(iw, op.k[1], op.s[1], d[1], pl, pr);
        if (oh < 1 || ow < 1) throw new Error(`line too narrow: op ${i} (${op.op}) gets ${ih}x${iw}`);
        const out = alloc(op.out, oh, ow);
        const inB = live.get(op.in);
        if (op.op === "conv") {
          this.conv(op, inB, out, B, ih, iw, oh, ow);
        } else {
          const cin = bufs[op.in].c;
          this.dispatch(K.POOL, [cin, ih, iw, oh, ow, op.k[0], op.k[1], op.s[0], op.s[1], pt, pl, op.op === "avgpool" ? 1 : 0, op.cip ? 1 : 0, B, 0, 0],
            [inB, out], [Math.ceil(ow / TILE), Math.ceil(oh / TILE), B * cin]);
        }
      } else if (op.op === "se") {
        const [h, w] = dims[op.in], hw = h * w;
        const out = alloc(op.out, h, w);
        const inB = live.get(op.in);
        const mean = this.get(B * c * 4);
        this.dispatch(K.GAP, [B * c, hw, 0, 0], [inB, mean], [Math.ceil(B * c / 64), 1, 1]);
        const t1 = this.get(B * op.fc1.wshape[0] * 4);
        this.conv(op.fc1, mean, t1, B, 1, 1, 1, 1);
        const gate = this.get(B * c * 4);
        this.conv(op.fc2, t1, gate, B, 1, 1, 1, 1);
        const n = B * c * hw;
        const [gx, gy, nx] = grid1(n);
        this.dispatch(K.SCALE, [n, nx, hw, op.res ? 1 : 0], [inB, gate, out], [gx, gy, 1]);
        this.put(mean); this.put(t1); this.put(gate);
      } else if (op.op === "add") {
        const [h, w] = dims[op.ins[0]];
        const out = alloc(op.out, h, w);
        const n = B * c * h * w;
        const [gx, gy, nx] = grid1(n);
        this.dispatch(K.ADD, [n, nx, 0, 0], [live.get(op.ins[0]), live.get(op.ins[1]), out], [gx, gy, 1]);
      } else if (op.op === "concat") {
        const [h, w] = dims[op.ins[0]];
        const out = alloc(op.out, h, w);
        let coff = 0;
        for (const b of op.ins) {
          const cs = bufs[b].c;
          this.dispatch(K.COPY, [cs, h, w, coff, c, B, 0, 0], [live.get(b), out], [Math.ceil(w / TILE), Math.ceil(h / TILE), B * cs]);
          coff += cs;
        }
      } else if (op.op === "img2seq" || op.op === "seq2img") {
        const [h, w] = dims[op.in];
        const T = h * w;
        const out = alloc(op.out, 1, T);
        const [r, s] = op.op === "img2seq" ? [c, T] : [T, c];
        this.dispatch(K.TRANSPOSE, [r, s, B, 0], [live.get(op.in), out], [Math.ceil(s / TILE), Math.ceil(r / TILE), B]);
      } else if (op.op === "linear") {
        const T = dims[op.in][1], M = B * T;
        const out = alloc(op.out, 1, T);
        this.dispatch(K.LINEAR, [M, op.K, op.N, T_[op.w].at, T_[op.b].at, ACT[op.act], { f: op.ha }, { f: op.hb }],
          [live.get(op.in), out, this.w], [Math.ceil(op.N / 64), Math.ceil(M / 64), 1]);
      } else if (op.op === "ln") {
        const T = dims[op.in][1], M = B * T;
        const out = alloc(op.out, 1, T);
        this.dispatch(K.LN, [M, op.D, T_[op.g].at, T_[op.b].at, { f: op.eps }, 0, 0, 0], [live.get(op.in), out, this.w], [Math.ceil(M / 64), 1, 1]);
      } else if (op.op === "attn") {
        const T = dims[op.in][1];
        const out = alloc(op.out, 1, T);
        this.dispatch(K.ATTN, [T, op.nh, op.hd, B, { f: op.scale }, 0, 0, 0], [live.get(op.in), out], [Math.ceil(T / 64), op.nh, B]);
      } else if (op.op === "softmax") {
        const T = dims[op.in][1], M = B * T;
        const out = alloc(op.out, 1, T);
        this.dispatch(K.SOFTMAX, [M, c, 0, 0], [live.get(op.in), out], [Math.min(M, 65535), Math.ceil(M / 65535), 1]);
      } else {
        throw new Error(`unknown op ${op.op}`);
      }
      if (trace) {
        flush();
        const [h, w] = dims[op.out];
        await trace(i, op, await this.read(live.get(op.out), B * c * h * w), [B, c, h, w]);
      }
      // Buffers whose last reader was this op go back to the pool (reuse is ordered within the pass).
      for (const b of inputs(op)) {
        if (last[b] === i && live.has(b)) { this.put(live.get(b)); live.delete(b); }
      }
    }
    const T = dims[man.output.buf][1];
    const outB = live.get(man.output.buf);
    this.pass.end();
    const n = B * T * this.V;
    const staging = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(outB, 0, staging, 0, n * 4);
    device.queue.submit([enc.finish()]);
    this.pass = null;
    await staging.mapAsync(GPUMapMode.READ);
    const probs = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    this.put(outB);
    for (const u of this.uniforms) u.destroy();
    this.uniforms = [];
    return { B, T, V: this.V, probs };
  }

  async read(buf, n) {
    const staging = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  // Greedy CTC as PaddleOCR's CTCLabelDecode: argmax per step (first index on ties), merge repeats,
  // drop the blank (class 0). out: run()'s result, or probs with T given. Returns one string per line.
  decode(out, T = out.T) {
    const probs = out.probs || out, V = this.V;
    const B = probs.length / (T * V);
    const texts = [];
    for (let b = 0; b < B; b++) {
      let s = "", prev = -1;
      for (let t = 0; t < T; t++) {
        const base = (b * T + t) * V;
        let k = 0, best = probs[base];
        for (let v = 1; v < V; v++) if (probs[base + v] > best) { best = probs[base + v]; k = v; }
        if (k !== prev && k !== 0) s += this.chars[k];
        prev = k;
      }
      texts.push(s);
    }
    return texts;
  }

  destroy() {
    this.w.destroy();
    for (const list of this.free.values()) for (const b of list) b.destroy();
    this.free.clear();
  }
}
