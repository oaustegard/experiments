// Tesseract's LSTM line recognizer on WebGPU: every line of a batch in one submit.
//
//   const gpu = await Gpu.create();
//   const model = await TessModel.load(gpu, "models/best-eng/");
//   const out = await recognize(gpu, model, [{ data: Uint8Array, width, height }, ...]);
//   out[i].text
//
// Stages (web/kernels/*): Convolve 3x3 + FC + tanh, Maxpool 3x3, y-summarising LSTM,
// three x LSTMs (forward, reversed, forward), softmax + argmax. Lines are packed longest
// first so the lines still running at step s of an x LSTM are a prefix of the batch.

import { Gpu } from "./gpu.js";
import { recordConv, recordPool } from "./kernels/conv.js";
import { recordYLstm } from "./kernels/ylstm.js";
import { recordXLstm } from "./kernels/xlstm.js";
import { recordOutput } from "./kernels/output.js";
import { normalise, padTable, scaleToHeight } from "./prep.js";

const f32 = Float32Array.BYTES_PER_ELEMENT;

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export class TessModel {
  static async load(gpu, base) {
    const man = await (await fetch(base + "manifest.json")).json();
    const bin = await (await fetch(base + "weights.bin")).arrayBuffer();
    return new TessModel(gpu, man, bin);
  }

  constructor(gpu, man, bin) {
    this.man = man;
    this.H = man.H;
    this.C = man.C;
    this.intMode = man.int_mode;
    this.ns = man.ns;
    this.ni = man.ni;
    this.w = {};
    this.s = {};
    for (const [name, t] of Object.entries(man.tensors)) {
      const n = t.shape[0] * t.shape[1];
      let w;
      if (t.dtype === "f32") w = new Float32Array(bin, t.offset, n);
      else if (t.dtype === "f16") {
        const h = new Uint16Array(bin, t.offset, n);
        w = new Float32Array(n);
        for (let i = 0; i < n; i++) w[i] = halfToFloat(h[i]);
      } else {
        const q = new Int8Array(bin, t.offset, n);
        const sc = new Float32Array(bin, t.scale_offset, t.shape[0]);
        if (t.int_mode) {
          // Tesseract's own int8 model: the kernels take the codes and the row scales
          w = Float32Array.from(q);
          this.s[name] = gpu.upload(sc);
        } else {
          // a float model packed to per-row codes: expand to f32 weights
          w = new Float32Array(n);
          const cols = t.shape[1];
          for (let r = 0; r < t.shape[0]; r++) for (let c = 0; c < cols; c++) w[r * cols + c] = q[r * cols + c] * sc[r];
        }
      }
      this.w[name] = gpu.upload(w);
    }
    const tanh = new Float32Array(4096), logi = new Float32Array(4096);
    for (let i = 0; i < 4096; i++) {
      tanh[i] = Math.tanh(i / 256);
      logi[i] = 1 / (1 + Math.exp(-i / 256));
    }
    this.tanhT = gpu.upload(tanh);
    this.logT = gpu.upload(logi);
    // code sequence -> unichar for greedy decoding (ref decode.Codec)
    this.byCodes = new Map();
    man.codes.forEach((c, uid) => {
      if (uid > 0 && uid < 3) return;
      const k = c.join(",");
      if (!this.byCodes.has(k)) this.byCodes.set(k, uid);
    });
    this.maxCodeLen = Math.max(...man.codes.map((c) => c.length));
  }

  decode(best, from, T) {
    const codes = [];
    let prev = -1;
    for (let t = from; t < from + T; t++) {
      const c = best[t];
      if (c !== prev && c !== this.man.null_char) codes.push(c);
      prev = c;
    }
    let s = "";
    for (let i = 0; i < codes.length;) {
      let hit = false;
      for (let n = Math.min(this.maxCodeLen, codes.length - i); n > 0; n--) {
        const uid = this.byCodes.get(codes.slice(i, i + n).join(","));
        if (uid !== undefined) { s += this.man.unichars[uid]; i += n; hit = true; break; }
      }
      if (!hit) i++;
    }
    return s.split(/\s+/).filter(Boolean).join(" ");
  }
}

// lines: [{ data: Uint8Array grey, width, height }]. Returns [{ text, T, best?, probs? }]
// in input order, plus timings on the array (out.ms).
export async function recognize(gpu, model, lines, { wantProbs = false } = {}) {
  const t0 = performance.now();
  const H = model.H, Hp = Math.floor(H / 3);
  const prepared = lines.map((l, i) => {
    const { data, width } = scaleToHeight(l.data, l.width, l.height, H);
    return { i, W: width, gray: data };
  }).filter((l) => l.W >= 3);
  prepared.sort((a, b) => b.W - a.W);
  const meta = [];
  let inOff = 0, padOff = 0, poolOff = 0, tOff = 0;
  const xs = [], pads = [];
  for (const p of prepared) {
    const T = Math.floor(p.W / 3);
    const x = normalise(p.gray, p.W, H, model.intMode);
    const pt = padTable(H, p.W, model.man.sample_iteration, model.intMode);
    const nb = pt.length / 9;
    meta.push({ src: p.i, W: p.W, T, nb, in_off: inOff, pad_off: padOff, pool_off: poolOff, t_off: tOff });
    xs.push(x);
    pads.push(pt);
    inOff += H * p.W; padOff += nb; poolOff += Hp * T; tOff += T;
  }
  const cat = (arrs, n) => { const o = new Float32Array(n); let k = 0; for (const a of arrs) { o.set(a, k); k += a.length; } return o; };
  const tPrep = performance.now();
  const ns = model.ns, ni = model.ni, intMode = model.intMode, w = model.w, s = model.s;
  const buf = (n) => gpu.buffer(Math.max(1, n) * f32);
  const xBuf = gpu.upload(cat(xs, inOff));
  const padBuf = gpu.upload(cat(pads, padOff * 9));
  const conv = buf(inOff * 16), pool = buf(poolOff * 16);
  const y = buf(tOff * ns.y), x1 = buf(tOff * ns.x1), x2 = buf(tOff * ns.x2), x3 = buf(tOff * ns.x3);
  const state = buf(meta.length * Math.max(ns.x1, ns.x2, ns.x3));
  const probs = buf(tOff * model.C), best = buf(tOff), bestp = buf(tOff);
  const common = { tanhT: model.tanhT, logT: model.logT, lines: meta, intMode };
  gpu.begin();
  recordConv(gpu, { x: xBuf, pads: padBuf, w: w.w_conv, s: s.w_conv, tanhT: model.tanhT, out: conv, lines: meta, H, intMode });
  recordPool(gpu, { conv, out: pool, lines: meta, H });
  recordYLstm(gpu, { ...common, pool, w: w.w_y, s: s.w_y, out: y, H, ns: ns.y });
  recordXLstm(gpu, { ...common, x: y, w: w.w_x1, s: s.w_x1, out: x1, state, ni: ni.x1, ns: ns.x1, reverse: false });
  recordXLstm(gpu, { ...common, x: x1, w: w.w_x2, s: s.w_x2, out: x2, state, ni: ni.x2, ns: ns.x2, reverse: true });
  recordXLstm(gpu, { ...common, x: x2, w: w.w_x3, s: s.w_x3, out: x3, state, ni: ni.x3, ns: ns.x3, reverse: false });
  recordOutput(gpu, { x: x3, w: w.w_out, s: s.w_out, out: probs, best, bestp, T: tOff, ni: ns.x3, C: model.C, intMode });
  gpu.submit();
  const bestHost = await gpu.read(best, Uint32Array, Math.max(16, tOff * 4));
  const tGpu = performance.now();
  const probsHost = wantProbs ? await gpu.read(probs, Float32Array, Math.max(16, tOff * model.C * 4)) : null;
  const out = new Array(lines.length).fill(null).map(() => ({ text: "", T: 0 }));
  for (const m of meta) {
    const r = { text: model.decode(bestHost, m.t_off, m.T), T: m.T };
    if (probsHost) r.probs = probsHost.subarray(m.t_off * model.C, (m.t_off + m.T) * model.C);
    out[m.src] = r;
  }
  for (const b of [xBuf, padBuf, conv, pool, y, x1, x2, x3, state, probs, best, bestp]) b.destroy();
  out.ms = { prep: tPrep - t0, gpu: tGpu - tPrep, total: performance.now() - t0, lines: meta.length, steps: meta.length ? meta[0].T : 0 };
  return out;
}

export { Gpu };
