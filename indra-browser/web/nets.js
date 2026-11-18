// Indra's networks on the GPU (gpu.js kernels) and the order model on the CPU, following
// run/ref.py op for op. Tensors are NHWC float32.

import { CONV, CTC, CTX, MM, MOVE, SEQ } from "./gpu.js";

const ACT = { null: 0, undefined: 0, hardswish: 1, relu: 2, sigmoid: 3 };

// ONNX conv weight [O, C/g, kh, kw] -> [O, kh, kw, C/g], the order CONV reads.
function convWeight(w, O, cg, kh, kw) {
  const out = new Float32Array(O * kh * kw * cg);
  for (let o = 0; o < O; o++)
    for (let c = 0; c < cg; c++)
      for (let y = 0; y < kh; y++)
        for (let x = 0; x < kw; x++) out[((o * kh + y) * kw + x) * cg + c] = w[((o * cg + c) * kh + y) * kw + x];
  return out;
}

class Weights {
  constructor(gpu, tensors, shapes) {
    this.gpu = gpu;
    this.t = tensors;
    this.shapes = shapes;
    this.cache = new Map();
  }
  // a conv weight on the GPU in CONV's layout, with its shape
  conv(name, kh, kw) {
    const key = `conv:${name}`;
    if (!this.cache.has(key)) {
      const sh = this.shapes[name];
      const O = sh[0], cg = sh[1], KH = kh ?? sh[2] ?? 1, KW = kw ?? sh[3] ?? 1;
      this.cache.set(key, { buf: this.gpu.upload(convWeight(this.t[name], O, cg, KH, KW)), O, cg, kh: KH, kw: KW });
    }
    return this.cache.get(key);
  }
  vec(name) {
    const key = `vec:${name}`;
    if (!this.cache.has(key)) this.cache.set(key, this.gpu.upload(this.t[name]));
    return this.cache.get(key);
  }
}

// One conv dispatch. x: {buf, N, H, W, C, cs?, co?}. Returns the output tensor descriptor.
function conv(gpu, x, w, b, { s = [1, 1], p = [0, 0, 0, 0], g = 1, act = 0, res = null, resAt = 1, mask = null, out = null }) {
  const Ho = Math.floor((x.H + p[0] + p[2] - w.kh) / s[0]) + 1;
  const Wo = Math.floor((x.W + p[1] + p[3] - w.kw) / s[1]) + 1;
  const O = w.O;
  const y = out ?? { buf: gpu.buffer(4 * x.N * Ho * Wo * O), N: x.N, H: Ho, W: Wo, C: O };
  // a plain 1x1 convolution over a contiguous input is a matmul over its pixels
  if (w.kh === 1 && w.kw === 1 && s[0] === 1 && s[1] === 1 && !p.some((v) => v) && g === 1 && !(x.cs && x.cs !== x.C) && !gpu.naiveConv &&
      (!mask || Ho === 1)) {
    const R = x.N * Ho * Wo;
    gpu.dispatch(MM, [x.buf, w.buf, b, res?.buf, mask, y.buf],
      [R, x.C, O, act, res ? resAt : 0, mask ? 1 : 0, y.cs ?? O, y.co ?? 0], [Math.ceil(O / 64), Math.ceil(R / 64)], `mm${O > 1000 ? ".head" : ""}`);
    return y;
  }
  const total = x.N * Ho * Wo * O;
  const grid = gpu.grid(total);
  gpu.dispatch(CONV, [x.buf, w.buf, b, res?.buf, mask, y.buf], [
    x.N, x.H, x.W, x.C, Ho, Wo, O, w.kh, w.kw, s[0], s[1], p[0], p[1], g, act,
    res ? resAt : 0, x.cs ?? x.C, x.co ?? 0, y.cs ?? O, y.co ?? 0, mask ? 1 : 0, total, grid[0], 0], grid,
    g > 1 ? "conv.dw" : `conv.${w.kh}x${w.kw}`);
  return y;
}

function move(gpu, kind, x, shape, out = null) {
  const [N, H, W, C] = shape;
  const y = out ?? { buf: gpu.buffer(4 * N * H * W * C), N, H, W, C };
  const total = N * H * W * C;
  const grid = gpu.grid(total);
  gpu.dispatch(MOVE, [x.buf, y.buf], [kind, N, H, W, C, x.H, x.W, x.C, y.cs ?? C, y.co ?? 0, total, grid[0]], grid);
  return y;
}

// ------------------------------------------------------------------ conv nets (det, orient, script)

export class ConvNet {
  constructor(gpu, model) {
    this.gpu = gpu;
    this.e = model.entry;
    const shapes = Object.fromEntries(Object.entries(this.e.tensors).map(([k, r]) => [k, r.shape]));
    this.w = new Weights(gpu, model.tensors, shapes);
    this.t = model.tensors;
    this.zero = gpu.upload(new Float32Array(256));
  }

  // x: Float32Array [N, H, W, C]. Returns {name: Float32Array} for `want` tensors (default:
  // the last GPU op's output) with their shapes.
  async run(x, N, H, W, C, want = null) {
    const gpu = this.gpu;
    gpu.scope = this.e.post ? "det" : "cls";
    const T = { [this.e.input]: { buf: gpu.upload(x), N, H, W, C } };
    const ops = this.e.ops.filter((o) => o.op !== "gemm" && o.op !== "gmean" && o.op !== "hmean");
    gpu.begin();
    for (const op of ops) {
      if (op.op === "conv") {
        const w = this.w.conv(op.w);
        const b = op.b ? this.w.vec(op.b) : this.zero;
        T[op.out] = conv(gpu, T[op.in], w, b, {
          s: op.s, p: op.p, g: op.g, act: ACT[op.act],
          res: op.res ? T[op.res] : null, resAt: op.res_at === "post" ? 2 : 1,
        });
      } else if (op.op === "up2") {
        const a = T[op.in];
        T[op.out] = move(gpu, 0, a, [a.N, a.H * 2, a.W * 2, a.C]);
      } else if (op.op === "s2d") {
        const a = T[op.in];
        T[op.out] = move(gpu, 1, a, [a.N, a.H / 2, a.W / 2, a.C * 4]);
      } else if (op.op === "d2s") {
        const a = T[op.in];
        T[op.out] = move(gpu, 2, a, [a.N, a.H * 2, a.W * 2, a.C / 4]);
      } else if (op.op === "concat") {
        const ins = op.ins.map((n) => T[n]);
        const Cc = ins.reduce((s, a) => s + a.C, 0);
        const a0 = ins[0];
        const y = { buf: gpu.buffer(4 * a0.N * a0.H * a0.W * Cc), N: a0.N, H: a0.H, W: a0.W, C: Cc };
        let co = 0;
        for (const a of ins) {
          move(gpu, 3, a, [a.N, a.H, a.W, a.C], { ...y, cs: Cc, co });
          co += a.C;
        }
        T[op.out] = y;
      } else if (op.op === "pool") {
        const a = T[op.in];
        T[op.out] = move(gpu, op.kind === "avg" ? 4 : 5, a, [a.N, Math.floor(a.H / 2), Math.floor(a.W / 2), a.C]);
      } else if (op.op === "act") {
        throw new Error("unfused activation");
      }
    }
    const names = want ?? [ops[ops.length - 1].out];
    const outs = await gpu.end(names.map((n) => [T[n].buf, 4 * T[n].N * T[n].H * T[n].W * T[n].C]));
    const res = {};
    names.forEach((n, i) => { res[n] = { data: new Float32Array(outs[i]), shape: [T[n].N, T[n].H, T[n].W, T[n].C] }; });
    for (const t of Object.values(T)) t.buf.destroy?.();
    return res;
  }

  // det: ink map [H, W] -> [H/2, W/2, 3] (core logit, top, bottom distances in input pixels)
  async det(ink, H, W) {
    const r = await this.run(ink, 1, H, W, 1);
    const { data, shape } = Object.values(r)[0];
    const k = this.e.post.softplus_scale;
    for (let i = 0; i < data.length; i += 3) {
      for (const j of [1, 2]) {
        const v = data[i + j];
        data[i + j] = k * (Math.max(v, 0) + Math.log1p(Math.exp(-Math.abs(v))));
      }
    }
    return { data, h: shape[1], w: shape[2] };
  }

  // orient / script: inputs [N, H, W] -> class probabilities [N, k]
  async classify(x, N, H, W, widths = null) {
    const gemm = this.e.ops.find((o) => o.op === "gemm");
    const last = this.e.ops.filter((o) => o.op !== "gemm" && o.op !== "gmean" && o.op !== "hmean").at(-1);
    const r = await this.run(x, N, H, W, 1, [last.out]);
    const { data, shape } = r[last.out];
    const [, h, w, C] = shape;
    const feat = new Float32Array(N * C);
    for (let n = 0; n < N; n++) {
      if (gemm.wdiv) {
        // mean over rows, then over the valid columns
        const cols = Math.max(1, Math.floor(widths[n] / gemm.wdiv));
        const nc = Math.min(cols, w);
        for (let x0 = 0; x0 < nc; x0++)
          for (let y0 = 0; y0 < h; y0++)
            for (let c = 0; c < C; c++) feat[n * C + c] += data[((n * h + y0) * w + x0) * C + c] / (h * cols);
      } else {
        for (let i = 0; i < h * w; i++) for (let c = 0; c < C; c++) feat[n * C + c] += data[(n * h * w + i) * C + c] / (h * w);
      }
    }
    const Wg = this.t[gemm.w], bg = this.t[gemm.b];
    const K = C, O = bg.length;
    const out = [];
    for (let n = 0; n < N; n++) {
      const z = new Float64Array(O);
      for (let o = 0; o < O; o++) {
        let s = bg[o];
        for (let k = 0; k < K; k++) s += feat[n * C + k] * Wg[o * K + k];
        z[o] = s;
      }
      const mx = Math.max(...z);
      let den = 0;
      for (let o = 0; o < O; o++) { z[o] = Math.exp(z[o] - mx); den += z[o]; }
      out.push(Array.from(z, (v) => v / den));
    }
    return out;
  }
}

// ------------------------------------------------------------------ recognizers

export class Recognizer {
  constructor(gpu, model) {
    this.gpu = gpu;
    this.t = model.tensors;
    const shapes = Object.fromEntries(Object.entries(model.entry.tensors).map(([k, r]) => [k, r.shape]));
    this.w = new Weights(gpu, model.tensors, shapes);
    this.window = "rec.context.row_prior" in this.t;
    this.V = shapes["rec.head.weight"][0];
    const g = this.t["rec.stem_bn.weight"], bb = this.t["rec.stem_bn.bias"];
    const mu = this.t["rec.stem_bn.running_mean"], vr = this.t["rec.stem_bn.running_var"];
    const s = new Float32Array(g.length), t = new Float32Array(g.length);
    for (let c = 0; c < g.length; c++) { s[c] = g[c] / Math.sqrt(vr[c] + 1e-5); t[c] = bb[c] - mu[c] * s[c]; }
    this.bnS = gpu.upload(s);
    this.bnT = gpu.upload(t);
    if (this.window) {
      const pr = this.t["rec.context.row_prior"]; // [1, 4, 24, 1]
      this.prior = gpu.upload(new Float32Array(pr));
    }
    this.dw = [];
    for (let i = 0; i < 10; i++) {
      const n = `rec.blocks.${i}.dw.Conv.weight` in this.t ? `rec.blocks.${i}.dw.Conv` : `rec.blocks.${i}.dw`;
      this.dw.push(n);
    }
  }

  // x: Float32Array [B, H, Wd] ink (H = 96 windowed, 32 otherwise), widths [B]
  // -> {T, lengths, out: Float32Array [B, T, 19]} (see CTC)
  async run(x, B, H, Wd, widths) {
    const gpu = this.gpu, w = this.w;
    gpu.scope = "rec";
    const T = Math.floor(Wd / 4);
    const lengths = widths.map((v) => Math.min(Math.ceil(v / 4), T));
    const mask = new Float32Array(B * T);
    for (let b = 0; b < B; b++) for (let t = 0; t < lengths[b]; t++) mask[b * T + t] = 1;
    const mbuf = gpu.upload(mask);
    const xin = { buf: gpu.upload(x), N: B, H, W: Wd, C: 1 };
    const tmp = [mbuf, xin.buf];
    const keep = (t) => { tmp.push(t.buf); return t; };
    gpu.begin();
    // the line band: rows 32..63 of the window, as its own tensor
    let band = xin;
    if (this.window) {
      const bx = new Float32Array(B * 32 * Wd);
      for (let b = 0; b < B; b++) bx.set(x.subarray((b * H + 32) * Wd, (b * H + 64) * Wd), b * 32 * Wd);
      band = { buf: gpu.upload(bx), N: B, H: 32, W: Wd, C: 1 };
      tmp.push(band.buf);
    }
    const stemW = w.conv("rec.stem.weight");
    let h = keep(conv(gpu, band, stemW, w.vec("rec.stem.bias"), { s: [32, 4] })); // [B, 1, T, 128]
    if (this.window) {
      let c = keep(conv(gpu, xin, w.conv("rec.context.patch.Conv.weight"), w.vec("rec.context.patch.Conv.bias"), { s: [4, 4], act: 1 }));
      c = keep(conv(gpu, c, w.conv("rec.context.dw.weight"), this.zeroVec(32), { p: [1, 1, 1, 1], g: 32 }));
      c = keep(conv(gpu, c, w.conv("rec.context.pw.Conv.weight"), w.vec("rec.context.pw.Conv.bias"), { act: 1 })); // [B, 24, T, 64]
      const sc = keep(conv(gpu, c, w.conv("rec.context.score.weight"), w.vec("rec.context.score.bias"), {})); // [B, 24, T, 4]
      const ctx = { buf: gpu.buffer(4 * B * T * 64), N: B, H: 1, W: T, C: 64 };
      keep(ctx);
      const total = B * T * 64, grid = gpu.grid(total);
      gpu.dispatch(CTX, [c.buf, sc.buf, this.prior, ctx.buf], [B, T, total, grid[0]], grid);
      h = keep(conv(gpu, ctx, w.conv("rec.context.proj.weight", 1, 1), w.vec("rec.context.proj.bias"), { res: h, resAt: 1 }));
    }
    // batch norm, hardswish, mask
    h = keep(this.seq(0, h, this.bnS, this.bnT, mbuf, B, T, 128, 1));
    for (let i = 0; i < 10; i++) {
      const p = `rec.blocks.${i}`;
      const y0 = keep(conv(gpu, h, w.conv(`${this.dw[i]}.weight`, 1, 9), w.vec(`${this.dw[i]}.bias`), { p: [0, 4, 0, 4], g: 128 }));
      const y1 = keep(conv(gpu, y0, w.conv(`${p}.expand.weight`, 1, 1), w.vec(`${p}.expand.bias`), { act: 1 })); // [B, 1, T, 256]
      const pooled = keep(this.seq(1, y1, null, null, mbuf, B, T, 256, 0, B * 256));
      const pv = { buf: pooled.buf, N: B, H: 1, W: 1, C: 256 };
      const s1 = keep(conv(gpu, pv, w.conv(`${p}.se.fc1.weight`, 1, 1), w.vec(`${p}.se.fc1.bias`), { act: 2 }));
      const s2 = keep(conv(gpu, s1, w.conv(`${p}.se.fc2.weight`, 1, 1), w.vec(`${p}.se.fc2.bias`), { act: 3 }));
      const y2 = keep(this.seq(2, y1, s2.buf, null, null, B, T, 256, 0));
      h = keep(conv(gpu, y2, w.conv(`${p}.project.weight`, 1, 1), w.vec(`${p}.project.bias`), { res: h, resAt: 1, mask: mbuf }));
    }
    const lg = keep(conv(gpu, h, w.conv("rec.head.weight", 1, 1), w.vec("rec.head.bias"), {})); // [B, 1, T, V]
    const out = gpu.buffer(4 * B * T * 19);
    tmp.push(out);
    const R = B * T, nx = Math.min(R, 65535);
    gpu.dispatch(CTC, [lg.buf, out], [R, this.V, nx, 0], [nx, Math.ceil(R / nx)]);
    const [res] = await gpu.end([[out, 4 * B * T * 19]]);
    for (const b of tmp) b.destroy();
    return { T, lengths, out: new Float32Array(res) };
  }

  seq(kind, x, s, t, m, B, T, C, act, total = B * T * C) {
    const gpu = this.gpu;
    const y = { buf: gpu.buffer(4 * total), N: B, H: 1, W: kind === 1 ? 1 : T, C };
    const grid = gpu.grid(total);
    gpu.dispatch(SEQ, [x.buf, s, t, m, y.buf], [kind, B, T, C, total, grid[0], act, 0], grid);
    return y;
  }

  zeroVec(n) {
    if (!this._z || this._z.n < n) this._z = { buf: this.gpu.upload(new Float32Array(n)), n };
    return this._z.buf;
  }
}

// ------------------------------------------------------------------ order model (CPU)

const FREQS = Float32Array.from({ length: 16 }, (_, i) => (Math.PI / 2) * 2 ** i);

function erf(x) {
  // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return x < 0 ? -y : y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

// y [N, M] = x [N, K] @ W [K, M] + b
function mm(x, N, K, W, M, b) {
  const y = new Float32Array(N * M);
  for (let n = 0; n < N; n++) {
    const yo = n * M, xo = n * K;
    if (b) y.set(b, yo);
    for (let k = 0; k < K; k++) {
      const v = x[xo + k];
      if (v === 0) continue;
      const wo = k * M;
      for (let m = 0; m < M; m++) y[yo + m] += v * W[wo + m];
    }
  }
  return y;
}

function ln(x, N, D, w, b) {
  const y = new Float32Array(N * D);
  for (let n = 0; n < N; n++) {
    let m = 0;
    for (let d = 0; d < D; d++) m += x[n * D + d];
    m /= D;
    let v = 0;
    for (let d = 0; d < D; d++) v += (x[n * D + d] - m) ** 2;
    const r = 1 / Math.sqrt(v / D + 1e-5);
    for (let d = 0; d < D; d++) y[n * D + d] = (x[n * D + d] - m) * r * w[d] + b[d];
  }
  return y;
}

export class OrderNet {
  constructor(model) { this.W = model.tensors; }

  // feats [N, 16] -> {a, cls, same, tab} (Float32Arrays [N*N], [N*10], [N*N], [3*N*N])
  run(feats, N) {
    const W = this.W;
    const x0 = new Float32Array(N * 144);
    for (let n = 0; n < N; n++) {
      for (let d = 0; d < 16; d++) x0[n * 144 + d] = feats[n * 16 + d];
      for (let f = 0; f < 4; f++) {
        for (let k = 0; k < 16; k++) {
          const v = Math.fround(feats[n * 16 + f] * FREQS[k]);
          x0[n * 144 + 16 + f * 32 + k] = Math.sin(v);
          x0[n * 144 + 16 + f * 32 + 16 + k] = Math.cos(v);
        }
      }
    }
    let x = mm(x0, N, 144, W["inp.inp.0.MatMul.weight"], 128, W["inp.0.bias"]).map(gelu);
    x = mm(x, N, 128, W["inp.inp.2.MatMul.weight"], 128, W["inp.2.bias"]);
    for (let i = 0; i < 4; i++) {
      const L = `layers.${i}`;
      const h = ln(x, N, 128, W[`${L}.n1.weight`], W[`${L}.n1.bias`]);
      const qkv = mm(h, N, 128, W[`${L}.qkv.MatMul.weight`], 384, W[`${L}.qkv.bias`]);
      const o = new Float32Array(N * 128);
      const sc = new Float64Array(N);
      for (let hd = 0; hd < 4; hd++) {
        for (let a = 0; a < N; a++) {
          let mx = -Infinity;
          for (let b = 0; b < N; b++) {
            let s = 0;
            for (let d = 0; d < 32; d++) s += qkv[a * 384 + hd * 32 + d] * qkv[b * 384 + 128 + hd * 32 + d];
            sc[b] = s / Math.sqrt(32);
            if (sc[b] > mx) mx = sc[b];
          }
          let den = 0;
          for (let b = 0; b < N; b++) { sc[b] = Math.exp(sc[b] - mx); den += sc[b]; }
          for (let b = 0; b < N; b++) {
            const p = sc[b] / den;
            for (let d = 0; d < 32; d++) o[a * 128 + hd * 32 + d] += p * qkv[b * 384 + 256 + hd * 32 + d];
          }
        }
      }
      const pr = mm(o, N, 128, W[`${L}.proj.MatMul.weight`], 128, W[`${L}.proj.bias`]);
      for (let j = 0; j < x.length; j++) x[j] += pr[j];
      const h2 = ln(x, N, 128, W[`${L}.n2.weight`], W[`${L}.n2.bias`]);
      const f1 = mm(h2, N, 128, W[`${L}.ff.ff.0.MatMul.weight`], 256, W[`${L}.ff.0.bias`]).map(gelu);
      const f2 = mm(f1, N, 256, W[`${L}.ff.ff.2.MatMul.weight`], 128, W[`${L}.ff.2.bias`]);
      for (let j = 0; j < x.length; j++) x[j] += f2[j];
    }
    x = ln(x, N, 128, W["norm.weight"], W["norm.bias"]);
    const lin = (n, M) => mm(x, N, 128, W[`${n}.MatMul.weight`], M, W[`${n}.bias`]);
    const r = Math.sqrt(128);
    const pair = (u, v, D, ou, ov, su, sv) => {
      const s = new Float32Array(N * N);
      for (let a = 0; a < N; a++)
        for (let b = 0; b < N; b++) {
          let acc = 0;
          for (let d = 0; d < D; d++) acc += u[a * su + ou + d] * v[b * sv + ov + d];
          s[a * N + b] = acc / r;
        }
      return s;
    };
    const q = lin("q", 128), k = lin("k", 128);
    const s1 = pair(q, k, 128, 0, 0, 128, 128);
    const a = new Float32Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) a[i * N + j] = s1[i * N + j] - s1[j * N + i];
    const s2 = pair(lin("bu", 128), lin("bw", 128), 128, 0, 0, 128, 128);
    const same = new Float32Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) same[i * N + j] = s2[i * N + j] + s2[j * N + i];
    const tu = lin("tu", 384), tw = lin("tw", 384);
    const tab = new Float32Array(3 * N * N);
    for (let c = 0; c < 3; c++) {
      const t = pair(tu, tw, 128, c * 128, c * 128, 384, 384);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) tab[c * N * N + i * N + j] = t[i * N + j] + t[j * N + i];
    }
    return { a, cls: lin("cls", 10), same, tab };
  }
}
