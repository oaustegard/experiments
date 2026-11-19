// Parse a Tesseract 4/5 .traineddata file in the browser (or node) into the pack manifest and
// weights.bin that TessModel consumes, with no pre-packing step. Mirrors tessgpu/traineddata.py
// (the reader) and run/pack.py (the layout) byte for byte; run/check_traineddata.mjs checks it.
//
//   const { man, bin } = parseTraineddata(arrayBuffer);
//   const model = await loadTraineddata(gpu, "models/best/eng.traineddata");

import { TessModel } from "./engine.js";

const TYPE_NAMES = [
  "Invalid", "Input", "Convolve", "Maxpool", "Parallel", "Replicated", "ParBidiLSTM",
  "DepParUDLSTM", "Par2dLSTM", "Series", "Reconfig", "RTLReversed", "TTBReversed",
  "XYTranspose", "LSTM", "SummLSTM", "Logistic", "LinLogistic", "LinTanh", "Tanh", "Relu",
  "Linear", "Softmax", "SoftmaxNoCTC", "LSTMSoftmax", "LSTMBinarySoftmax", "TensorFlow",
  "Dropout",
];
const PLUMBING = new Set(["Parallel", "Replicated", "ParBidiLSTM", "DepParUDLSTM", "Par2dLSTM",
  "Series", "RTLReversed", "TTBReversed", "XYTranspose"]);
const FULLY = new Set(["Logistic", "LinLogistic", "LinTanh", "Tanh", "Relu", "Linear", "Softmax",
  "SoftmaxNoCTC"]);
const LSTMS = new Set(["LSTM", "SummLSTM", "LSTMSoftmax", "LSTMBinarySoftmax"]);
const NF_LAYER_SPECIFIC_LR = 64;
const TS_ENABLED = 1;
const TF_COMPRESS_UNICHARSET = 64;
const COMP_LSTM = 17, COMP_LSTM_UNICHARSET = 21, COMP_LSTM_RECODER = 22, COMP_VERSION = 23;

const utf8 = new TextDecoder("utf-8", { fatal: true });
const utf8Lossy = new TextDecoder("utf-8");

// Sequential little-endian reader over a Uint8Array view. Every read is bounds-checked.
class Reader {
  constructor(u8) {
    this.u8 = u8;
    this.dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    this.o = 0;
  }
  need(n) {
    if (this.o + n > this.u8.byteLength) {
      throw new Error(`truncated .traineddata: need ${n} bytes at offset ${this.o} of ${this.u8.byteLength}`);
    }
  }
  i8() { this.need(1); const v = this.dv.getInt8(this.o); this.o += 1; return v; }
  u8v() { this.need(1); const v = this.dv.getUint8(this.o); this.o += 1; return v; }
  i32() { this.need(4); const v = this.dv.getInt32(this.o, true); this.o += 4; return v; }
  u32() { this.need(4); const v = this.dv.getUint32(this.o, true); this.o += 4; return v; }
  f32() { this.need(4); const v = this.dv.getFloat32(this.o, true); this.o += 4; return v; }
  f64() { this.need(8); const v = this.dv.getFloat64(this.o, true); this.o += 8; return v; }
  i64() { this.need(8); const v = Number(this.dv.getBigInt64(this.o, true)); this.o += 8; return v; }
  string() {
    const n = this.u32();
    this.need(n);
    const s = utf8.decode(this.u8.subarray(this.o, this.o + n));
    this.o += n;
    return s;
  }
  i32s(n) {
    this.need(4 * n);
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) out[i] = this.dv.getInt32(this.o + 4 * i, true);
    this.o += 4 * n;
    return out;
  }
  f32s(n) {
    this.need(4 * n);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = this.dv.getFloat32(this.o + 4 * i, true);
    this.o += 4 * n;
    return out;
  }
  f64s(n) {
    this.need(8 * n);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.dv.getFloat64(this.o + 8 * i, true);
    this.o += 8 * n;
    return out;
  }
  i8s(n) {
    this.need(n);
    const out = new Int8Array(this.u8.buffer.slice(this.u8.byteOffset + this.o, this.u8.byteOffset + this.o + n));
    this.o += n;
    return out;
  }
  remaining() { return this.u8.byteLength - this.o; }
}

// One WeightMatrix. Rows = outputs, cols = inputs + 1 (bias last).
// float: w Float64Array [rows*cols]. int8: q Int8Array [rows*cols], scale Float64Array [rows]
// already divided by 127 as Tesseract does at load.
function readWeights(r, training) {
  const mode = r.u8v();
  const intMode = (mode & 1) !== 0;
  const adam = (mode & 4) !== 0;
  if ((mode & 128) === 0) throw new Error("old-format weight matrix (pre-double) not supported");
  if (intMode) {
    const rows = r.u32(), cols = r.u32();
    r.i8();                       // empty_ value
    const q = r.i8s(rows * cols);
    const n = r.u32();
    const raw = r.f64s(n);
    const scale = new Float64Array(rows);
    for (let i = 0; i < rows; i++) scale[i] = raw[i] / 127;
    return { intMode, rows, cols, q, scale };
  }
  const rows = r.u32(), cols = r.u32();
  r.f64();                        // empty_ value
  const w = r.f64s(rows * cols);
  if (training) {                 // updates_ (and Adam's dw_sq_sum_) follow in a training dump
    for (let k = 0; k < (adam ? 2 : 1); k++) {
      const e1 = r.u32(), e2 = r.u32();
      r.f64();
      r.f64s(e1 * e2);
    }
  }
  return { intMode, rows, cols, w };
}

function ceilLog2(n) {
  let c = 0;
  while (2 ** c < n) c++;
  return c;
}

function readNetwork(r) {
  const t = r.i8();
  let type;
  if (t === 0) type = r.string();
  else if (t > 0 && t < TYPE_NAMES.length) type = TYPE_NAMES[t];
  else throw new Error(`unsupported layer type id ${t} in .traineddata`);
  const training = r.i8() === TS_ENABLED;
  r.i8();                         // needs_to_backprop
  const flags = r.i32(), ni = r.i32(), no0 = r.i32(), numWeights = r.i32();
  const name = r.string();
  const n = { type, name, ni, no: no0, flags, numWeights, training, children: [],
    shape: null, half: null, scale: null, gates: null, na: 0, ns: 0, isTwoD: false,
    softmax: null, weights: null };

  if (type === "Input") {
    n.shape = [r.i32(), r.i32(), r.i32(), r.i32(), r.i32()];
  } else if (type === "Convolve") {
    n.half = [r.i32(), r.i32()];
    n.no = ni * (2 * n.half[0] + 1) * (2 * n.half[1] + 1);
  } else if (type === "Maxpool" || type === "Reconfig") {
    n.scale = [r.i32(), r.i32()];
    n.no = type === "Maxpool" ? ni : ni * n.scale[0] * n.scale[1];
  } else if (PLUMBING.has(type)) {
    const size = r.u32();
    for (let i = 0; i < size; i++) n.children.push(readNetwork(r));
    if (flags & NF_LAYER_SPECIFIC_LR) {
      const k = r.u32();
      r.f32s(k);
    }
    if (type === "Series") n.no = n.children[n.children.length - 1].no;
    else if (type === "RTLReversed" || type === "TTBReversed" || type === "XYTranspose") n.no = n.children[0].no;
    else n.no = n.children.reduce((a, c) => a + c.no, 0);
  } else if (LSTMS.has(type)) {
    n.na = r.i32();
    const nf = type === "LSTMSoftmax" ? no0 : type === "LSTMBinarySoftmax" ? ceilLog2(no0) : 0;
    n.gates = {};
    for (const g of ["CI", "GI", "GF1", "GO", "GFS"]) {
      if (g === "GFS" && !n.isTwoD) continue;
      n.gates[g] = readWeights(r, training);
      if (g === "CI") {
        n.ns = n.gates.CI.rows;
        n.isTwoD = n.na - nf === ni + 2 * n.ns;
      }
    }
    if (type === "LSTMSoftmax" || type === "LSTMBinarySoftmax") n.softmax = readNetwork(r);
  } else if (FULLY.has(type)) {
    n.weights = readWeights(r, training);
  } else if (type === "Dropout") {
    // no payload
  } else {
    throw new Error(`unsupported layer type ${type} in .traineddata`);
  }
  return n;
}

function* walk(node) {
  yield node;
  for (const c of node.children) yield* walk(c);
  if (node.softmax) yield* walk(node.softmax);
}

// lstm-unicharset text: count line, then one entry per line; "NULL" is the space character.
function readUnicharset(text) {
  const lines = text.split("\n");
  const n = parseInt(lines[0].trim(), 10);
  if (!Number.isFinite(n)) throw new Error("bad lstm-unicharset header");
  const chars = [];
  for (const line of lines.slice(1, 1 + n)) {
    const tok = line.split(" ")[0];
    chars.push(tok === "NULL" ? " " : tok);
  }
  return chars;
}

function readRecoder(u8) {
  const r = new Reader(u8);
  const n = r.u32();
  const codes = [];
  for (let i = 0; i < n; i++) {
    r.i8();                       // self_normalized
    const len = r.u32();
    codes.push(Array.from(r.i32s(len)));
  }
  if (r.remaining() !== 0) throw new Error(`lstm-recoder: ${r.remaining()} trailing bytes`);
  return codes;
}

// Container: uint32 count, count int64 offsets (-1 = absent), entry bytes. Entry i runs to the
// next present offset (or EOF). Returns Map<componentId, Uint8Array>.
function readComponents(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (u8.byteLength < 4) throw new Error("not a .traineddata file (shorter than its header)");
  const n = dv.getUint32(0, true);
  if (4 + 8 * n > u8.byteLength) throw new Error("not a .traineddata file (truncated offset table)");
  const offs = [];
  for (let i = 0; i < n; i++) offs.push(Number(dv.getBigInt64(4 + 8 * i, true)));
  const out = new Map();
  for (let i = 0; i < n; i++) {
    const o = offs[i];
    if (o < 0) continue;
    let end = u8.byteLength;
    for (let j = i + 1; j < n; j++) if (offs[j] >= 0) { end = offs[j]; break; }
    if (o > end || end > u8.byteLength) throw new Error(`corrupt .traineddata entry ${i}`);
    out.set(i, u8.subarray(o, end));
  }
  return out;
}

// Pack-layout tensors, in the order run/pack.py emits them.
function tensorGroups(kids) {
  const gates = (node) => ["CI", "GI", "GF1", "GO"].map((g) => node.gates[g]);
  return [
    ["w_conv", [kids[1].children[1].weights]],
    ["w_y", gates(kids[3].children[0])],
    ["w_x1", gates(kids[4])],
    ["w_x2", gates(kids[5].children[0])],
    ["w_x3", gates(kids[6])],
    ["w_out", [kids[7].weights]],
  ];
}

function checkTopology(net, path) {
  const k = net.children;
  const ok = net.type === "Series" && k.length === 8 && k[0].type === "Input" &&
    k[1].type === "Series" && k[1].children.length === 2 && k[1].children[1].weights &&
    k[2].type === "Maxpool" && k[3].type === "XYTranspose" && k[3].children[0] &&
    LSTMS.has(k[3].children[0].type) && LSTMS.has(k[4].type) && k[5].type === "RTLReversed" &&
    k[5].children[0] && LSTMS.has(k[5].children[0].type) && LSTMS.has(k[6].type) &&
    k[7].type === "Softmax" && k[7].weights;
  if (!ok) {
    throw new Error(`${path}: unsupported network; expected the tessdata_fast/best eng-style ` +
      "Series(Input, ConvSeries(Convolve, Tanh), Maxpool, XYTranspose(LSTM), LSTM, " +
      "RTLReversed(LSTM), LSTM, Softmax) topology");
  }
}

// Parse a .traineddata ArrayBuffer into the manifest and weights.bin pack.py writes.
export function parseTraineddata(arrayBuffer, { name = "" } = {}) {
  const u8 = new Uint8Array(arrayBuffer);
  const comps = readComponents(u8);
  if (!comps.has(COMP_LSTM)) {
    throw new Error(`${name || "traineddata"}: no LSTM component (equ/osd files cannot be recognized)`);
  }
  if (!comps.has(COMP_LSTM_UNICHARSET) || !comps.has(COMP_LSTM_RECODER)) {
    throw new Error(`${name || "traineddata"}: charsets embedded in the LSTM component are not supported; ` +
      "use a tessdata_fast or tessdata_best model");
  }
  const r = new Reader(comps.get(COMP_LSTM));
  const net = readNetwork(r);
  const networkStr = r.string();
  const trainingFlags = r.i32(), trainingIteration = r.i32(), sampleIteration = r.i32(), nullChar = r.i32();
  void trainingIteration;
  r.f32(); r.f32(); r.f32();     // learning_rate, momentum, adam_beta
  if (r.remaining() !== 0) throw new Error(`${name}: ${r.remaining()} trailing bytes in the LSTM component`);

  checkTopology(net, name || "traineddata");
  const intMode = [...walk(net)].some((n) => LSTMS.has(n.type) && n.gates.CI.intMode);
  const unichars = readUnicharset(utf8.decode(comps.get(COMP_LSTM_UNICHARSET)));
  const codes = readRecoder(comps.get(COMP_LSTM_RECODER));
  const version = comps.has(COMP_VERSION) ? utf8Lossy.decode(comps.get(COMP_VERSION)) : "";

  const kids = net.children;
  const ns = { y: kids[3].children[0].ns, x1: kids[4].ns, x2: kids[5].children[0].ns, x3: kids[6].ns };
  const ni = { y: 16, x1: ns.y, x2: ns.x1, x3: ns.x2 };

  // blob layout identical to run/pack.py put(): 4-byte aligned, in tensor order
  const parts = [];
  let size = 0;
  const tensors = {};
  const put = (arr) => {
    while (size % 4) size++;
    const off = size;
    parts.push({ off, arr });
    size += arr.byteLength;
    return off;
  };
  for (const [tname, mats] of tensorGroups(kids)) {
    const cols = mats[0].cols;
    const rows = mats.reduce((a, m) => a + m.rows, 0);
    if (intMode) {
      const q = new Int8Array(rows * cols);
      const s = new Float32Array(rows);
      let qo = 0, so = 0;
      for (const m of mats) {
        if (!m.intMode) throw new Error(`${tname}: mixed int8 and float matrices`);
        q.set(m.q, qo); qo += m.q.length;
        s.set(m.scale, so); so += m.rows;
      }
      tensors[tname] = { dtype: "i8", shape: [rows, cols], offset: put(q),
        scale_offset: put(s), int_mode: true };
    } else {
      const w = new Float32Array(rows * cols);
      let wo = 0;
      for (const m of mats) {
        if (m.intMode) throw new Error(`${tname}: mixed int8 and float matrices`);
        w.set(m.w, wo); wo += m.w.length;
      }
      tensors[tname] = { dtype: "f32", shape: [rows, cols], offset: put(w) };
    }
  }

  const bin = new Uint8Array(size);
  for (const { off, arr } of parts) bin.set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength), off);

  const man = {
    format: "tesseract-gpu/1",
    source: "",
    variant: "",
    weights: intMode ? "int8-native" : "f32",
    version,
    spec: networkStr,
    H: net.children[0].shape[1],
    C: net.no,
    int_mode: intMode,
    simple_text: (trainingFlags & TF_COMPRESS_UNICHARSET) === 0,
    null_char: nullChar,
    sample_iteration: sampleIteration,
    ns,
    ni,
    unichars,
    codes,
    tensors,
    bytes: size,
  };
  return { man, bin: bin.buffer };
}

async function defaultFetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetching ${url}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

// Fetch a .traineddata file and build the TessModel from it.
export async function loadTraineddata(gpu, url, { fetchBytes = defaultFetchBytes } = {}) {
  const buf = await fetchBytes(url);
  const { man, bin } = parseTraineddata(buf, { name: url });
  return new TessModel(gpu, man, bin);
}
