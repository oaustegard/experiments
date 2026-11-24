// PP-OCRv6 text-line detector on WebGPU. Runs the op program that run/export_det.py writes
// (web/det/models/<tiny|small>/manifest.json + weights.bin); the spec is run/ref_det.py.
//
//   const gpu = await Gpu.create();
//   const det = await DetNet.load(gpu, "models/tiny/");   // a URL relative to the page, or absolute
//   const { prob, H, W, scale } = await det.run(rgbaOrGray, width, height, 1280);
//   det.post   // DB postprocess settings from the model's inference.yml (thresh, box_thresh, unclip_ratio, ...)
//
// prob[y * W + x] is the text probability of padded-map pixel (x, y); the original image pixel is
// (x / scale, y / scale). validH and validW give the part of the map that is image (the rest is pad).
// Preprocessing is on the CPU: bilinear resize (half-pixel centres), BGR, /255, per-channel
// mean/std, zero pad to multiples of 32. Every dispatch of one forward goes into one command
// encoder; buffers come from a size-class pool and are reused in order within that pass.

import { Gpu } from "./gpu.js";
import * as K from "./kernels.js";

export const FORMAT = "ppocr-pack/det-program v1";
const ACT = { none: 0, relu: 1, hsig: 2, gelu: 3, sigmoid: 4 };
const TILE = 8;

// Uniform block from ints (u32) and {f} entries (f32), padded to 16 bytes.
function uniform(vals) {
  const dv = new DataView(new ArrayBuffer(Math.ceil(vals.length / 4) * 16));
  vals.forEach((v, i) => {
    if (typeof v === "object") dv.setFloat32(4 * i, v.f, true);
    else dv.setUint32(4 * i, v >>> 0, true);
  });
  return dv.buffer;
}

// Image pixels as RGB bytes. Accepts RGBA (4 bytes a pixel) or gray (1 byte a pixel).
export function toRGB(img, width, height) {
  const n = width * height;
  const rgb = new Uint8Array(n * 3);
  if (img.length === 4 * n) {
    for (let i = 0; i < n; i++) { rgb[3 * i] = img[4 * i]; rgb[3 * i + 1] = img[4 * i + 1]; rgb[3 * i + 2] = img[4 * i + 2]; }
  } else if (img.length === n) {
    for (let i = 0; i < n; i++) { rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = img[i]; }
  } else {
    throw new Error(`image is ${img.length} bytes; ${width}x${height} needs ${4 * n} (RGBA) or ${n} (gray)`);
  }
  return rgb;
}

// Same arithmetic as run/ref_det.py preprocess(): returns { x: CHW BGR float32, scale, h1, w1, H, W }.
export function preprocess(rgb, width, height, longSide, input) {
  const s = longSide / Math.max(width, height);
  const h1 = Math.max(1, Math.floor(height * s + 0.5)), w1 = Math.max(1, Math.floor(width * s + 0.5));
  const H = Math.ceil(h1 / 32) * 32, W = Math.ceil(w1 / 32) * 32;
  const ys = new Float64Array(h1), y0 = new Int32Array(h1), y1 = new Int32Array(h1), wy = new Float64Array(h1);
  for (let i = 0; i < h1; i++) {
    const v = Math.min(Math.max((i + 0.5) * height / h1 - 0.5, 0), height - 1);
    ys[i] = v; y0[i] = Math.floor(v); y1[i] = Math.min(y0[i] + 1, height - 1); wy[i] = v - y0[i];
  }
  const xs = new Float64Array(w1), x0 = new Int32Array(w1), x1 = new Int32Array(w1), wx = new Float64Array(w1);
  for (let j = 0; j < w1; j++) {
    const v = Math.min(Math.max((j + 0.5) * width / w1 - 0.5, 0), width - 1);
    xs[j] = v; x0[j] = Math.floor(v); x1[j] = Math.min(x0[j] + 1, width - 1); wx[j] = v - x0[j];
  }
  const x = new Float32Array(3 * H * W);
  const mean = input.mean, std = input.std;
  for (let c = 0; c < 3; c++) {
    const src = 2 - c; // BGR channel c is RGB channel 2 - c
    for (let i = 0; i < h1; i++) {
      const r0 = y0[i] * width, r1 = y1[i] * width;
      for (let j = 0; j < w1; j++) {
        const top = (rgb[(r0 + x0[j]) * 3 + src] * (1 - wx[j]) + rgb[(r0 + x1[j]) * 3 + src] * wx[j]);
        const bot = (rgb[(r1 + x0[j]) * 3 + src] * (1 - wx[j]) + rgb[(r1 + x1[j]) * 3 + src] * wx[j]);
        const v = (top * (1 - wy[i]) + bot * wy[i]) / 255;
        x[c * H * W + i * W + j] = (v - mean[c]) / std[c];
      }
    }
  }
  return { x, scale: s, h1, w1, H, W };
}

// IEEE half to double (exact).
function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// One weight tensor to f32 at params[at ...]. wq is the manifest's {kind, off[, soff, axis]} (byte
// offsets into raw, all 4-byte aligned); shape is the ONNX weight shape.
//   f32  copied
//   f16  half -> f32
//   i8   code * fp16 scale of the tensor's output channel (axis 0: conv; axis 1: ConvTranspose)
// The product of a 7-bit code and an 11-bit significand is exact in f32, so this is the same value
// run/ref_det.py computes.
function unpackWeight(wq, shape, raw, params, at) {
  const n = shape.reduce((a, b) => a * b, 1);
  if (wq.kind === "f32") {
    params.set(new Float32Array(raw, wq.off, n), at);
  } else if (wq.kind === "f16") {
    const h = new Uint16Array(raw, wq.off, n);
    for (let i = 0; i < n; i++) params[at + i] = f16(h[i]);
  } else if (wq.kind === "i8") {
    const codes = new Int8Array(raw, wq.off, n);
    const rows = shape[wq.axis];
    const scale = new Float64Array(rows);
    const sh = new Uint16Array(raw, wq.soff, rows + (rows & 1));
    for (let r = 0; r < rows; r++) scale[r] = f16(sh[r]);
    // element (a, b, ...) of a row-major tensor: the channel index is (i / inner) % rows, inner = prod(shape[axis+1:])
    let inner = 1;
    for (let d = wq.axis + 1; d < shape.length; d++) inner *= shape[d];
    for (let i = 0; i < n; i++) params[at + i] = codes[i] * scale[Math.floor(i / inner) % rows];
  } else {
    throw new Error(`unknown tensor kind ${wq.kind}`);
  }
}

// Dequantises weights.bin into one f32 array: every conv, ConvTranspose and squeeze-excite fc
// weight, each followed by its bias (f32 in the file). Sets entry.wf and entry.bf (f32 element
// offsets) and entry.n (weight elements) on every such entry of the manifest.
function unpackParams(man, raw) {
  const entries = [];
  for (const op of man.ops) {
    if (op.op === "conv") entries.push([op, op.wshape[0], 0]);
    else if (op.op === "convt") entries.push([op, op.wshape[1], 1]);
    else if (op.op === "se") entries.push([op.fc1, op.fc1.wshape[0], 0], [op.fc2, op.fc2.wshape[0], 0]);
  }
  let total = 0;
  for (const [e, nb] of entries) {
    e.n = e.wshape.reduce((a, b) => a * b, 1);
    e.wf = total; total += e.n;
    e.bf = total; total += nb;
  }
  const params = new Float32Array(total);
  for (const [e, nb, axis] of entries) {
    if (e.wq.kind === "i8" && e.wq.axis !== axis) throw new Error("scale axis does not match the op type");
    unpackWeight(e.wq, e.wshape, raw, params, e.wf);
    params.set(new Float32Array(raw, e.b, nb), e.bf);
  }
  return params;
}

export class DetNet {
  static async load(gpu, base) {
    const man = await (await fetch(base + "manifest.json")).json();
    if (man.format !== FORMAT) throw new Error(`det program is ${man.format}; this engine reads ${FORMAT}`);
    const raw = await (await fetch(base + man.weights.file)).arrayBuffer();
    if (raw.byteLength !== man.weights.bytes) throw new Error("weights.bin length does not match the manifest");
    return new DetNet(gpu, man, gpu.upload(unpackParams(man, raw)));
  }

  constructor(gpu, man, wbuf) {
    this.gpu = gpu;
    this.man = man;
    this.post = man.postprocess;   // DB postprocess settings (inference.yml)
    this.w = wbuf;   // f32 weights and biases
    this.free = new Map();   // size class (power of 2 bytes) -> GPUBuffer[]
    this.uniforms = [];      // destroyed once the forward's submission has been read back
  }

  get(bytes) {
    let s = 16;
    while (s < bytes) s *= 2;
    const list = this.free.get(s);
    return list && list.length ? list.pop() : this.gpu.buffer(s);
  }

  put(b) {
    const list = this.free.get(b.size);
    if (list) list.push(b); else this.free.set(b.size, [b]);
  }

  uni(vals) {
    const u = this.gpu.uniform(uniform(vals));
    this.uniforms.push(u);
    return u;
  }

  // Conv or 1x1 fc entry from the manifest; in/out are buffers with spatial sizes ih x iw and oh x ow.
  conv(e, inB, outB, ih, iw, oh, ow) {
    const [O, icg, kh, kw] = e.wshape;
    const [pt, pl] = e.pads;
    const g = e.g;
    const blocked = (O / g) % 4 === 0;   // four output channels per invocation when they share a group
    this.gpu.dispatch(blocked ? K.CONV4 : K.CONV, [
      this.uni([icg * g, ih, iw, O, oh, ow, icg, O / g, kh, kw, e.s[0], e.s[1], pt, pl, e.d[0], e.d[1],
        ACT[e.act], e.wf, e.bf, 0, { f: e.ha }, { f: e.hb }, 0, 0]),
      inB, outB, this.w,
    ], [Math.ceil(ow / TILE), Math.ceil(oh / TILE), blocked ? O / 4 : O]);
  }

  // Forward pass on a CHW BGR normalised input (3, H, W). Returns the probability map, H * W.
  // trace(i, op, out, h, w, c), if given, receives every op's output (read back; debugging only).
  async forward(x, H, W, trace = null) {
    const { gpu, man } = this;
    const bufs = man.buffers, ops = man.ops;
    const dims = [[H, W]];
    const inputs = (op) => (op.op === "concat" || op.op === "add") ? op.ins : [op.in];
    const last = new Array(bufs.length).fill(-1);
    ops.forEach((op, i) => { for (const b of inputs(op)) last[b] = i; });
    last[man.output.buf] = Infinity;
    const xb = this.get(x.byteLength);
    gpu.device.queue.writeBuffer(xb, 0, x);
    const live = new Map([[0, xb]]);
    const bytesOf = (id) => bufs[id].c * dims[id][0] * dims[id][1] * 4;
    const alloc = (id, h, w) => { dims[id] = [h, w]; const b = this.get(bufs[id].c * h * w * 4); live.set(id, b); return b; };

    for (let i = 0; i < ops.length; i++) {
      const op = ops[i];
      if (op.op === "conv" || op.op === "convt" || op.op === "maxpool" || op.op === "resize") {
        const [ih, iw] = dims[op.in];
        const [pt, pl, pb, pr] = op.pads || [0, 0, 0, 0];
        let oh, ow;
        if (op.op === "conv" || op.op === "maxpool") {
          oh = Math.floor((ih + pt + pb - op.d[0] * (op.k[0] - 1) - 1) / op.s[0]) + 1;
          ow = Math.floor((iw + pl + pr - op.d[1] * (op.k[1] - 1) - 1) / op.s[1]) + 1;
        } else if (op.op === "convt") {
          oh = 2 * ih; ow = 2 * iw;
        } else {
          oh = ih * op.scale; ow = iw * op.scale;
        }
        const out = alloc(op.out, oh, ow);
        const inB = live.get(op.in);
        if (op.op === "conv") {
          this.conv(op, inB, out, ih, iw, oh, ow);
        } else if (op.op === "convt") {
          const [Cin, Cout] = op.wshape;
          gpu.dispatch(K.CONVT, [this.uni([Cin, ih, iw, Cout, oh, ow, ACT[op.act], op.wf, op.bf, 0,
            { f: op.ha }, { f: op.hb }]), inB, out, this.w], [Math.ceil(ow / TILE), Math.ceil(oh / TILE), Cout]);
        } else if (op.op === "maxpool") {
          const c = bufs[op.in].c;
          gpu.dispatch(K.MAXPOOL, [this.uni([c, ih, iw, oh, ow, op.k[0], op.k[1], op.s[0], op.s[1], pt, pl, 0]),
            inB, out], [Math.ceil(ow / TILE), Math.ceil(oh / TILE), c]);
        } else {
          const c = bufs[op.in].c;
          gpu.dispatch(K.RESIZE, [this.uni([c, ih, iw, oh, ow, op.scale, 0, 0]), inB, out],
            [Math.ceil(ow / TILE), Math.ceil(oh / TILE), c]);
        }
      } else if (op.op === "concat") {
        const [h, w] = dims[op.ins[0]];
        const out = alloc(op.out, h, w);
        let coff = 0;
        for (const b of op.ins) {
          const cs = bufs[b].c;
          gpu.dispatch(K.COPY, [this.uni([cs, h, w, coff]), live.get(b), out], [Math.ceil(w / TILE), Math.ceil(h / TILE), cs]);
          coff += cs;
        }
      } else if (op.op === "add") {
        const [h, w] = dims[op.ins[0]];
        const out = alloc(op.out, h, w);
        const n = bufs[op.out].c * h * w;
        const [gx, gy, nx] = grid1(n);
        gpu.dispatch(K.ADD, [this.uni([n, nx, 0, 0]), live.get(op.ins[0]), live.get(op.ins[1]), out], [gx, gy, 1]);
      } else if (op.op === "se") {
        const c = bufs[op.in].c, [h, w] = dims[op.in], hw = h * w;
        const out = alloc(op.out, h, w);
        const inB = live.get(op.in);
        const mean = this.get(c * 4);
        gpu.dispatch(K.GAP, [this.uni([c, hw, 0, 0]), inB, mean], [Math.ceil(c / 64), 1, 1]);
        const c1 = op.fc1.wshape[0];
        const t1 = this.get(c1 * 4);
        this.conv(op.fc1, mean, t1, 1, 1, 1, 1);
        const gate = this.get(c * 4);
        this.conv(op.fc2, t1, gate, 1, 1, 1, 1);
        const n = c * hw;
        const [gx, gy, nx] = grid1(n);
        gpu.dispatch(K.SCALE, [this.uni([n, nx, hw, op.res ? 1 : 0]), inB, gate, out], [gx, gy, 1]);
        this.put(mean); this.put(t1); this.put(gate);
      } else {
        throw new Error(`unknown op ${op.op}`);
      }
      if (trace) {
        const [oh, ow] = dims[op.out];
        const c = bufs[op.out].c;
        await trace(i, op, await gpu.read(live.get(op.out), Float32Array, c * oh * ow * 4), oh, ow, c);
      }
      // Buffers whose last reader was this op go back to the pool (reuse is ordered within the pass).
      for (const b of inputs(op)) {
        if (last[b] === i && live.has(b)) { this.put(live.get(b)); live.delete(b); }
      }
    }
    const outB = live.get(man.output.buf);
    const prob = await gpu.read(outB, Float32Array, H * W * 4);
    this.put(outB);
    for (const u of this.uniforms) u.destroy();
    this.uniforms = [];
    return prob;
  }

  // Full pipeline from pixels. img: RGBA or gray bytes, width x height.
  async run(img, width, height, longSide = 1280) {
    const pre = preprocess(toRGB(img, width, height), width, height, longSide, this.man.input);
    const prob = await this.forward(pre.x, pre.H, pre.W);
    return { prob, H: pre.H, W: pre.W, scale: pre.scale, validH: pre.h1, validW: pre.w1 };
  }
}

// 1-D dispatch of n threads in 64-wide workgroups: [gx, gy] with index i = gid.x + gid.y * nx.
function grid1(n) {
  const nb = Math.ceil(n / 64);
  const gx = Math.min(nb, 65535);
  const gy = Math.ceil(nb / gx);
  return [gx, gy, gx * 64];
}
