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
import { recordConvPoolU8 } from "./kernels/conv.js";
import { recordYLstm } from "./kernels/ylstm2.js";
import { recordXLstm as recordXLstm1 } from "./kernels/xlstm.js";
import { recordXLstm as recordXLstm2, interleaveGates } from "./kernels/xlstm2.js";
import { recordOutput } from "./kernels/output2.js";
import { isInverted, normTable, padTable, scaleToHeight } from "./prep.js";
import { decode as beamDecode } from "./beam.js";

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
      if (name.startsWith("w_x") || name === "w_y") this.w[name + "_4"] = gpu.upload(interleaveGates(w, t.shape[0] / 4, t.shape[1]));
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
// opts.decoder: "beam" (Tesseract's RecodeBeamSearch without dictionary, web/beam.js; reads
// back every softmax row) or "greedy" (argmax per step, reads back one code per step).
// opts.xlstm: "v2" (coalesced, default) or "v1"; opts.stages: run only the first n of the
// stages (conv+pool, ylstm, x1, x2, x3, output) for per-stage timing (no text then).
// opts.invert: "auto" (default) flips a line whose pixels are mostly darker than halfway between
// its black and white levels (light text on a dark ground) before normalising; "off" never does.
export async function recognize(gpu, model, lines, { wantProbs = false, decoder = "beam", xlstm = "v2", stages = 7, invert = "auto", dict = true } = {}) {
  const dawgs = dict && model.dict ? model.dict : null; // Tesseract's word lists in the beam search (web/beam_dict.js)
  const t0 = performance.now();
  const H = model.H, Hp = Math.floor(H / 3);
  const prepared = lines.map((l, i) => {
    let { data, width } = scaleToHeight(l.data, l.width, l.height, H);
    const inverted = invert === "auto" && width >= 3 && isInverted(data, width, H);
    if (inverted) data = data.map((v) => 255 - v);
    return { i, W: width, gray: data, inverted };
  }).filter((l) => l.W >= 3);
  prepared.sort((a, b) => b.W - a.W);
  const meta = [];
  let inOff = 0, padOff = 0, poolOff = 0, tOff = 0;
  const luts = [], pads = [];
  for (const p of prepared) {
    const T = Math.floor(p.W / 3);
    luts.push(normTable(p.gray, p.W, H, model.intMode));
    const pt = padTable(H, p.W, model.man.sample_iteration, model.intMode);
    const nb = pt.length / 9;
    meta.push({ src: p.i, W: p.W, T, nb, in_off: inOff, pad_off: padOff, pool_off: poolOff, t_off: tOff, inverted: p.inverted });
    pads.push(pt);
    inOff += H * p.W; padOff += nb; poolOff += Hp * T; tOff += T;
  }
  const cat = (arrs, n) => { const o = new Float32Array(n); let k = 0; for (const a of arrs) { o.set(a, k); k += a.length; } return o; };
  const tPrep = performance.now();
  const ns = model.ns, ni = model.ni, intMode = model.intMode, w = model.w, s = model.s;
  const buf = (n) => gpu.buffer(Math.max(1, n) * f32);
  const px = new Uint8Array(Math.ceil(inOff / 4) * 4);
  for (const [k, p] of prepared.entries()) px.set(p.gray, meta[k].in_off);
  const xBuf = gpu.upload(px);
  const lutBuf = gpu.upload(cat(luts, 256 * luts.length));
  const padBuf = gpu.upload(cat(pads, padOff * 9));
  const pool = buf(poolOff * 16);
  const y = buf(tOff * ns.y), x1 = buf(tOff * ns.x1), x2 = buf(tOff * ns.x2), x3 = buf(tOff * ns.x3);
  const state = buf(meta.length * Math.max(ns.x1, ns.x2, ns.x3));
  const probs = buf(tOff * model.C), best = buf(tOff), bestp = buf(tOff);
  const common = { tanhT: model.tanhT, logT: model.logT, lines: meta, intMode };
  const rx = xlstm === "v1" ? recordXLstm1 : recordXLstm2;
  const wx = (n) => (xlstm === "v1" ? w[n] : w[n + "_4"]);
  gpu.begin();
  const steps = [
    () => recordConvPoolU8(gpu, { x: xBuf, lut: lutBuf, pads: padBuf, w: w.w_conv, s: s.w_conv, tanhT: model.tanhT, out: pool, lines: meta, H, intMode }),
    () => recordYLstm(gpu, { ...common, pool, w: w.w_y_4, s: s.w_y, out: y, H, ns: ns.y }),
    () => rx(gpu, { ...common, x: y, w: wx("w_x1"), s: s.w_x1, out: x1, state, ni: ni.x1, ns: ns.x1, reverse: false }),
    () => rx(gpu, { ...common, x: x1, w: wx("w_x2"), s: s.w_x2, out: x2, state, ni: ni.x2, ns: ns.x2, reverse: true }),
    () => rx(gpu, { ...common, x: x2, w: wx("w_x3"), s: s.w_x3, out: x3, state, ni: ni.x3, ns: ns.x3, reverse: false }),
    () => recordOutput(gpu, { x: x3, w: w.w_out, s: s.w_out, out: probs, best, bestp, T: tOff, ni: ns.x3, C: model.C, intMode }),
  ];
  steps.slice(0, Math.min(stages, steps.length)).forEach((f) => f());
  gpu.submit();
  const bestHost = await gpu.read(stages >= steps.length ? best : pool, Uint32Array, 16).then(() =>
    stages >= steps.length ? gpu.read(best, Uint32Array, Math.max(16, tOff * 4)) : new Uint32Array(tOff));
  const tGpu = performance.now();
  const full = stages >= 6;
  const probsHost = full && (wantProbs || decoder === "beam") ? await gpu.read(probs, Float32Array, Math.max(16, tOff * model.C * 4)) : null;
  const tRead = performance.now();
  const out = new Array(lines.length).fill(null).map(() => ({ text: "", T: 0 }));
  for (const m of meta) {
    const r = { T: m.T, inverted: !!m.inverted };
    if (probsHost) r.probs = probsHost.subarray(m.t_off * model.C, (m.t_off + m.T) * model.C);
    if (!full) r.text = "";
    else if (decoder === "beam") {
      const ids = beamDecode(r.probs, m.T, model.C, model.man, dawgs);
      r.text = ids.map((u) => model.man.unichars[u]).join("").split(/\s+/).filter(Boolean).join(" ");
    } else r.text = model.decode(bestHost, m.t_off, m.T);
    if (!wantProbs) delete r.probs;
    out[m.src] = r;
  }
  for (const b of [xBuf, lutBuf, padBuf, pool, y, x1, x2, x3, state, probs, best, bestp]) b.destroy();
  const tEnd = performance.now();
  out.ms = { prep: tPrep - t0, gpu: tGpu - tPrep, readback: tRead - tGpu, decode: tEnd - tRead, total: tEnd - t0, lines: meta.length, steps: meta.length ? meta[0].T : 0 };
  return out;
}

export { Gpu };
