// Port of the dictionary path of Tesseract 5.3.4's LSTM beam search (RecodeBeamSearch with its
// Dict), the twin of tessgpu/beam_dict.py, which this mirrors step for step. See that file for
// the sources and what LSTMRecognizer::LoadDictionary builds. web/beam.js keeps the
// no-dictionary search; its decode(probs, T, C, model, dict) hands over to this file when a
// dictionary is given.
//
//   loadDict(traineddataBytes) -> dict or null   (null: the file has no lstm-*-dawg, and
//                                                  Tesseract then runs without a dictionary)
//   decodeDict(probs, T, C, model, dict) -> unichar ids (ExtractBestPathAsUnicharIds)
//   decodeDictFull(probs, T, C, model, dict) -> { ids, xcoords, perm, words }
//
//   probs : Float32Array T*C, row-major (Tesseract's softmax)
//   model : { null_char, codes, simple_text }, as for web/beam.js
//
// This is the plain object version (nodes are objects, heaps are arrays of [score, node]); it
// is about as fast as the Python, not as the typed-array no-dict search in web/beam.js.
// Float32 points follow the C++ types with Math.fround, as in web/beam.js.

const INVALID = -1;
const MIN_CERT = -20;
const KMIN_PROB = Math.fround(Math.exp(-20));
const DICT_RATIO = 2.25;
const CERT_OFFSET = -0.085;
const BEAM_WIDTHS = [5, 10, 16, 16, 16, 16, 16, 16, 16, 16];
const BW0 = 5;
const NUM_LENGTHS = 10;
const MAX_CODE_LEN = 9;
const NC_ANY = 0, NC_ONLY_DUP = 1, NC_NO_DUP = 2;
const TN_TOP2 = 0, TN_TOPN = 1, TN_ALSO_RAN = 2;
const NUM_BEAMS = 2 * 3 * NUM_LENGTHS;
const TWO32 = 4294967296;
const f32 = Math.fround;

const UNICHAR_SPACE = 0;
const NO_EDGE = -1;
const PATTERN_UNICHAR_ID = 0;
const NO_PERM = 0, PUNC_PERM = 1, TOP_CHOICE_PERM = 2, NUMBER_PERM = 6, SYSTEM_DAWG_PERM = 8;
const DAWG_PUNC = 0, DAWG_WORD = 1, DAWG_NUMBER = 2, DAWG_PATTERN = 3;
const DAWG_SUCCESSORS = [
  [false, true, true, false],
  [true, false, false, false],
  [true, false, false, false],
  [false, false, false, false],
];
// traineddata component -> [dawg type, permuter] (dawg_cache.cpp)
const LSTM_DAWGS = [[18, DAWG_PUNC, PUNC_PERM], [19, DAWG_WORD, SYSTEM_DAWG_PERM],
  [20, DAWG_NUMBER, NUMBER_PERM]];
const WORST_DICT_CERT = f32(f32(-25) / f32(7));   // kWorstDictCertainty / kCertaintyScale

function certOf(p) {
  const pc = p > KMIN_PROB ? f32(Math.log(p)) : MIN_CERT;
  return f32(pc + CERT_OFFSET);
}

// ---------------------------------------------------------------- traineddata components

function readComponents(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const n = dv.getUint32(0, true);
  const offs = [];
  for (let i = 0; i < n; i++) offs.push(Number(dv.getBigInt64(4 + 8 * i, true)));
  const out = new Map();
  for (let i = 0; i < n; i++) {
    const o = offs[i];
    if (o < 0) continue;
    let end = u8.byteLength;
    for (let j = i + 1; j < n; j++) if (offs[j] >= 0) { end = offs[j]; break; }
    out.set(i, u8.subarray(o, end));
  }
  return out;
}

// ---------------------------------------------------------------- SquishedDawg

class SquishedDawg {
  constructor(u8, type, perm) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const magic = dv.getInt16(0, true);
    if (magic !== 42) throw new Error(`bad dawg magic ${magic}`);
    const usize = dv.getInt32(2, true);
    const n = dv.getInt32(6, true);
    if (!(n > 0) || u8.byteLength < 10 + 8 * n) throw new Error("bad dawg size");
    this.type = type;
    this.perm = perm;
    this.unicharsetSize = usize;
    const fsb = Math.ceil(Math.log(usize + 1.0) / Math.log(2.0));   // Dawg::init
    const nnsb = fsb + 3;
    if (nnsb >= 32) throw new Error("unicharset too large for this reader");
    const lmask = 2 ** fsb;
    const emptyLo = ((0xffffffff << nnsb) >>> 0);
    this.letter = new Int32Array(n);
    this.last = new Uint8Array(n);
    this.backward = new Uint8Array(n);
    this.eow = new Uint8Array(n);
    this.occupied = new Uint8Array(n);
    this.next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const lo = dv.getUint32(10 + 8 * i, true);
      const hi = dv.getUint32(14 + 8 * i, true);
      this.letter[i] = lo % lmask;
      this.last[i] = (lo >>> fsb) & 1;
      this.backward[i] = (lo >>> (fsb + 1)) & 1;
      this.eow[i] = (lo >>> (fsb + 2)) & 1;
      this.occupied[i] = (lo === emptyLo && hi === 0xffffffff) ? 0 : 1;
      this.next[i] = hi * 2 ** (32 - nnsb) + (lo >>> nnsb);
    }
    this.numEdges = n;
    this.keyScale = 2 * lmask;
    this.n0 = this.numForwardEdges(0);
    this.cache = new Map();
  }

  numForwardEdges(node) {
    let edge = node, num = 0;
    if (this.occupied[edge] && !this.backward[edge]) {
      for (;;) {
        num++;
        if (this.last[edge]) break;
        edge++;
      }
    }
    return num;
  }

  edgeCharOf(node, uid, wordEnd) {
    const key = (node + 1) * this.keyScale + uid * 2 + (wordEnd ? 1 : 0);
    let r = this.cache.get(key);
    if (r === undefined) {
      r = this.edgeCharOfUncached(node, uid, wordEnd);
      this.cache.set(key, r);
    }
    return r;
  }

  edgeCharOfUncached(node, uid, wordEnd) {
    if (node === 0) {
      let start = 0, end = this.n0 - 1;
      while (start <= end) {
        const edge = (start + end) >> 1;
        const cur = this.letter[edge];
        let c;
        if (uid === cur && (!wordEnd || this.eow[edge])) c = 0;
        else if (uid > cur) c = 1;
        else c = -1;
        if (c === 0) return edge;
        if (c === 1) start = edge + 1;
        else end = edge - 1;
      }
    } else {
      let edge = node;
      if (edge !== NO_EDGE && this.occupied[edge]) {
        for (;;) {
          if (this.letter[edge] === uid && (!wordEnd || this.eow[edge])) return edge;
          if (this.last[edge]) break;
          edge++;
        }
      }
    }
    return NO_EDGE;
  }

  nextNode(edge) {
    if (edge < 0) throw new Error("next_node(NO_EDGE)");
    return this.next[edge];
  }

  endOfWord(edge) {
    if (edge < 0) throw new Error("end_of_word(NO_EDGE)");
    return this.eow[edge] === 1;
  }
}

function startingNode(dawg, edgeRef) {
  if (edgeRef === NO_EDGE) return 0;
  const node = dawg.nextNode(edgeRef);
  return node === 0 ? NO_EDGE : node;
}

// ---------------------------------------------------------------- unicharset properties

function parseUnicharset(text) {
  const lines = text.split("\n");
  const n = parseInt(lines[0].trim().split(/\s+/)[0], 10);
  const scripts = ["NULL"];
  const isdigit = [];
  const scriptId = [];
  for (const line of lines.slice(1, 1 + n)) {
    const tok = line.split(/\s+/).filter((s) => s.length);
    const props = tok.length > 1 ? parseInt(tok[1], 16) : 0;
    let script = "NULL";
    if (tok.length > 2) {
      script = tok[2].includes(",") && tok.length > 3 ? tok[3] : tok[2];
      if (script.includes(",")) script = "NULL";
    }
    if (!scripts.includes(script)) scripts.push(script);
    isdigit.push((props & 0x8) !== 0);
    scriptId.push(scripts.indexOf(script));
  }
  if (isdigit.length !== n) throw new Error(`unicharset declares ${n}, has ${isdigit.length}`);
  const sid = (name) => (scripts.includes(name) ? scripts.indexOf(name) : 0);
  return { isdigit, scriptId, han: sid("Han"), hiragana: sid("Hiragana"),
    katakana: sid("Katakana"), thai: sid("Thai"), hangul: sid("Hangul") };
}

// ---------------------------------------------------------------- Dict

function samePos(a, b) {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3] && a[4] === b[4];
}

function addUnique(list, p) {
  for (const q of list) if (samePos(q, p)) return;
  list.push(p);
}

class LSTMDict {
  constructor(comps) {
    const uni = parseUnicharset(new TextDecoder("utf-8").decode(comps.get(21)));
    this.isdigit = uni.isdigit;
    const nonSd = new Set([uni.han, uni.thai, uni.hangul, uni.hiragana, uni.katakana]);
    this.spaceDelimited = uni.scriptId.map((s) => !nonSd.has(s));
    this.spaceDelimitedLang = !(uni.han > 0 || uni.katakana > 0 || uni.thai > 0);
    this.dawgs = [];
    this.punc = null;
    for (const [idx, type, perm] of LSTM_DAWGS) {
      const bytes = comps.get(idx);
      if (bytes && bytes.byteLength) {
        const d = new SquishedDawg(bytes, type, perm);
        if (d.unicharsetSize !== this.isdigit.length) throw new Error(`dawg ${idx}: unicharset size`);
        this.dawgs.push(d);
        if (type === DAWG_PUNC) this.punc = d;
      }
    }
    this.successors = this.dawgs.map((d) =>
      this.dawgs.map((o, j) => (DAWG_SUCCESSORS[d.type][o.type] ? j : -1)).filter((j) => j >= 0));
    const puncOk = this.punc !== null &&
      this.punc.edgeCharOf(0, PATTERN_UNICHAR_ID, true) !== NO_EDGE;
    this.defaults = [];
    this.dawgs.forEach((d, i) => {
      if (d.type === DAWG_PUNC) this.defaults.push([-1, NO_EDGE, i, NO_EDGE, false]);
      else if (!puncOk || !DAWG_SUCCESSORS[DAWG_PUNC][d.type]) {
        this.defaults.push([i, NO_EDGE, -1, NO_EDGE, false]);
      }
    });
  }

  isSpaceDelimited(uid) {
    return uid === INVALID ? true : this.spaceDelimited[uid];
  }

  charForDawg(uid, dawg) {
    return dawg.type === DAWG_NUMBER && this.isdigit[uid] ? PATTERN_UNICHAR_ID : uid;
  }

  // Dict::def_letter_is_okay(args, unicharset, uid, word_end=false), permuter starting at
  // NO_PERM. Returns [permuter, updated positions, valid_end].
  letterIsOkay(active, uid) {
    if (uid === PATTERN_UNICHAR_ID || uid === INVALID) return [NO_PERM, [], false];
    const wordEnd = false;
    let curr = NO_PERM;
    const upd = [];
    let validEnd = false;
    const dawgs = this.dawgs;
    for (const pos of active) {
      const [dawgIndex, dawgRef, puncIndex, puncRef, back] = pos;
      const punc = puncIndex >= 0 ? dawgs[puncIndex] : null;
      const dawg = dawgIndex >= 0 ? dawgs[dawgIndex] : null;
      if (dawg === null && punc === null) continue;
      if (dawg === null) {
        const puncNode = startingNode(punc, puncRef);
        const pte = punc.edgeCharOf(puncNode, PATTERN_UNICHAR_ID, wordEnd);
        if (pte !== NO_EDGE) {
          for (const s of this.successors[puncIndex]) {
            const sd = dawgs[s];
            const de = sd.edgeCharOf(0, this.charForDawg(uid, sd), wordEnd);
            if (de !== NO_EDGE) {
              addUnique(upd, [s, de, puncIndex, pte, false]);
              if (sd.perm > curr) curr = sd.perm;
              if (sd.endOfWord(de) && punc.endOfWord(pte)) validEnd = true;
            }
          }
        }
        const pe = punc.edgeCharOf(puncNode, uid, wordEnd);
        if (pe !== NO_EDGE) {
          addUnique(upd, [-1, NO_EDGE, puncIndex, pe, false]);
          if (PUNC_PERM > curr) curr = PUNC_PERM;
          if (punc.endOfWord(pe)) validEnd = true;
        }
        continue;
      }
      if (punc !== null && dawg.endOfWord(dawgRef)) {
        const puncNode = startingNode(punc, puncRef);
        const pe = puncNode === NO_EDGE ? NO_EDGE : punc.edgeCharOf(puncNode, uid, wordEnd);
        if (pe !== NO_EDGE) {
          addUnique(upd, [dawgIndex, dawgRef, puncIndex, pe, true]);
          if (dawg.perm > curr) curr = dawg.perm;
          if (punc.endOfWord(pe)) validEnd = true;
        }
      }
      if (back) continue;
      if (dawg.type === DAWG_PATTERN) throw new Error("pattern dawgs are not loaded by LoadLSTM");
      const node = startingNode(dawg, dawgRef);
      const edge = node === NO_EDGE ? NO_EDGE
        : dawg.edgeCharOf(node, this.charForDawg(uid, dawg), wordEnd);
      if (edge !== NO_EDGE) {
        if (dawg.perm > curr) curr = dawg.perm;
        if (dawg.endOfWord(edge) && (punc === null || punc.endOfWord(puncRef))) validEnd = true;
        addUnique(upd, [dawgIndex, edge, puncIndex, puncRef, false]);
      }
    }
    return [curr, upd, validEnd];
  }
}

export function loadDict(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const comps = readComponents(u8);
  if (!LSTM_DAWGS.some(([idx]) => comps.has(idx) && comps.get(idx).byteLength)) return null;
  return new LSTMDict(comps);
}

// ---------------------------------------------------------------- recoder and heap

const recoderCache = new WeakMap();

function recoderFor(codes) {
  let rc = recoderCache.get(codes);
  if (rc !== undefined) return rc;
  let range = 0;
  for (const code of codes) for (const c of code) if (c + 1 > range) range = c + 1;
  const decoder = new Map(), finals = new Map(), nexts = new Map();
  codes.forEach((code, uid) => {
    decoder.set(code.join(","), uid);
    let ln = code.length - 1;
    const prefix = code.slice(0, ln).join(",");
    if (!finals.has(prefix)) {
      finals.set(prefix, [code[ln]]);
      for (ln -= 1; ln >= 0; ln--) {
        const p = code.slice(0, ln).join(",");
        if (!nexts.has(p)) nexts.set(p, [code[ln]]);
        else {
          if (!nexts.get(p).includes(code[ln])) nexts.get(p).push(code[ln]);
          break;
        }
      }
    } else if (!finals.get(prefix).includes(code[ln])) {
      finals.get(prefix).push(code[ln]);
    }
  });
  rc = { range, decoder, finals, nexts };
  recoderCache.set(codes, rc);
  return rc;
}

// GenericHeap<KDPairInc> over [key, data] pairs; the top is the smallest key.
function siftUp(h, hole, key) {
  while (hole > 0) {
    const parent = ((hole + 1) >> 1) - 1;
    if (key < h[parent][0]) {
      h[hole] = h[parent];
      hole = parent;
    } else break;
  }
  return hole;
}

function siftDown(h, hole, key) {
  const n = h.length;
  for (;;) {
    let child = 2 * hole + 1;
    if (child >= n) break;
    if (child + 1 < n && h[child + 1][0] < h[child][0]) child++;
    if (h[child][0] < key) {
      h[hole] = h[child];
      hole = child;
    } else break;
  }
  return hole;
}

function heapPush(h, pair) {
  h.push(pair);
  h[siftUp(h, h.length - 1, pair[0])] = pair;
}

function heapPop(h) {
  const top = h[0];
  const last = h.pop();
  if (h.length) h[siftDown(h, 0, last[0])] = last;
  return top;
}

function heapReshuffle(h, index) {
  const pair = h[index];
  let hole = siftDown(h, index, pair[0]);
  hole = siftUp(h, hole, pair[0]);
  h[hole] = pair;
}

// 64-bit code hash as two uint32 halves (web/beam.js): h = h*nc + floor(hi*nc/2^32) + code.
function hashNext(lo, hi, nc, code) {
  const carry = Math.floor((hi * nc) / TWO32);
  const pLo = lo * nc;
  const c1 = Math.floor(pLo / TWO32);
  let s32 = pLo - c1 * TWO32 + carry;
  const c2 = s32 >= TWO32 ? 1 : 0;
  lo = s32 - c2 * TWO32;
  hi = (hi * nc + c1 + c2) >>> 0;
  s32 = lo + code;
  const c3 = s32 >= TWO32 ? 1 : 0;
  lo = s32 - c3 * TWO32;
  hi = (hi + c3) >>> 0;
  return [lo, hi];
}

// ---------------------------------------------------------------- beam search

class Search {
  constructor(rc, nul, simple, dict) {
    this.rc = rc;
    this.nul = nul;
    this.simple = simple;
    this.dict = dict;
    this.spaceDelimited = dict === null || dict.spaceDelimitedLang;
    this.nbeams = dict !== null ? NUM_BEAMS : NUM_BEAMS / 2;
    this.beams = [];
    this.bestInit = null;
  }

  computeTopN(outs, topN) {
    const n = outs.length;
    const flags = new Array(n).fill(TN_ALSO_RAN);
    this.topCode = -1;
    this.secondCode = -1;
    const h = [];
    for (let i = 0; i < n; i++) {
      if (h.length < topN || outs[i] > h[0][0]) {
        heapPush(h, [outs[i], i]);
        if (h.length > topN) heapPop(h);
      }
    }
    while (h.length) {
      const e = heapPop(h);
      if (h.length > 1) flags[e[1]] = TN_TOPN;
      else {
        flags[e[1]] = TN_TOP2;
        if (!h.length) this.topCode = e[1];
        else this.secondCode = e[1];
      }
    }
    flags[this.nul] = TN_TOP2;
    this.flags = flags;
    this.outs = outs;
  }

  node(code, uid, perm, sod, sow, eow, dup, score, prev, dawgs) {
    let lo = 0, hi = 0;
    if (prev !== null) { lo = prev.lo; hi = prev.hi; }
    if (!dup && code !== this.nul) [lo, hi] = hashNext(lo, hi, this.rc.range, code);
    return { code, uid, perm, sod, sow, eow, dup, score, prev, dawgs, lo, hi };
  }

  static updateIfMatched(node, h) {
    for (let idx = 0; idx < h.length; idx++) {
      const old = h[idx][1];
      if (old.code === node.code && old.lo === node.lo && old.hi === node.hi &&
          old.perm === node.perm && old.sod === node.sod) {
        if (node.score > old.score) {
          h[idx] = [node.score, node];
          heapReshuffle(h, idx);
        }
        return true;
      }
    }
    return false;
  }

  pushHeap(maxSize, code, uid, perm, sod, sow, eow, dup, cert, prev, dawgs, h) {
    const score = prev === null ? cert : f32(cert + prev.score);
    if (h.length < maxSize || score > h[0][0]) {
      const node = this.node(code, uid, perm, sod, sow, eow, dup, score, prev, dawgs);
      if (Search.updateIfMatched(node, h)) return;
      heapPush(h, [score, node]);
      if (h.length > maxSize) heapPop(h);
    }
  }

  pushNode(maxSize, node, h) {
    if (h.length < maxSize || node.score > h[0][0]) {
      if (Search.updateIfMatched(node, h)) return;
      heapPush(h, [node.score, node]);
      if (h.length > maxSize) heapPop(h);
    }
  }

  pushInitialDawg(code, uid, perm, start, end, cert, cont, prev) {
    const score = prev === null ? cert : f32(cert + prev.score);
    const best = this.bestInit[cont];
    if (best === null || score > best.score) {
      this.bestInit[cont] = this.node(code, uid, perm, true, start, end, false, score, prev,
        this.dict.defaults);
    }
  }

  pushDup(length, dup, code, uid, cert, useDawgs, cont, prev, step) {
    if (useDawgs) {
      if (cert > WORST_DICT_CERT) {
        this.pushHeap(BEAM_WIDTHS[length], code, uid, prev !== null ? prev.perm : NO_PERM,
          false, false, false, dup, cert, prev, null, step[(3 + cont) * NUM_LENGTHS + length]);
      }
    } else {
      cert = f32(cert * DICT_RATIO);
      if (cert >= MIN_CERT || code === this.nul) {
        this.pushHeap(BEAM_WIDTHS[length], code, uid, prev !== null ? prev.perm : TOP_CHOICE_PERM,
          false, false, false, dup, cert, prev, null, step[cont * NUM_LENGTHS + length]);
      }
    }
  }

  continueUnichar(code, uid, cert, useDawgs, cont, prev, step) {
    if (useDawgs) {
      if (cert > WORST_DICT_CERT) this.continueDawg(code, uid, cert, cont, prev, step);
      return;
    }
    this.pushHeap(BW0, code, uid, TOP_CHOICE_PERM, false, false, false, false,
      f32(cert * DICT_RATIO), prev, null, step[cont * NUM_LENGTHS]);
    const d = this.dict;
    if (d !== null && ((uid === UNICHAR_SPACE && cert > WORST_DICT_CERT) ||
                       !d.isSpaceDelimited(uid))) {
      if (uid === UNICHAR_SPACE) this.pushInitialDawg(code, uid, NO_PERM, false, false, cert, cont, prev);
      else {
        this.pushInitialDawg(code, uid, TOP_CHOICE_PERM, false, false, f32(cert * DICT_RATIO),
          cont, prev);
      }
    }
  }

  continueDawg(code, uid, cert, cont, prev, step) {
    const dh = step[(3 + cont) * NUM_LENGTHS];
    const nh = step[cont * NUM_LENGTHS];
    if (uid === INVALID) {
      this.pushHeap(BW0, code, uid, NO_PERM, false, false, false, false, cert, prev, null, dh);
      return;
    }
    const score = prev === null ? cert : f32(cert + prev.score);
    if (dh.length >= BW0 && score <= dh[0][1].score && nh.length >= BW0 && score <= nh[0][1].score) {
      return;
    }
    let up = prev;
    while (up !== null && (up.uid === INVALID || up.dup)) up = up.prev;
    const d = this.dict;
    if (uid === UNICHAR_SPACE) {
      if (up !== null && up.eow) {
        this.pushInitialDawg(code, uid, up.perm, false, false, cert, cont, prev);
        this.pushHeap(BW0, code, uid, up.perm, false, false, false, false, cert, prev, null, nh);
      }
      return;
    } else if (up !== null && up.sod && up.uid !== UNICHAR_SPACE &&
               d.isSpaceDelimited(up.uid) && d.isSpaceDelimited(uid)) {
      return;
    }
    let active, wordStart;
    if (up === null) {
      active = d.defaults;
      wordStart = true;
    } else if (up.dawgs !== null) {
      active = up.dawgs;
      wordStart = up.sod;
    } else {
      return;
    }
    const [perm, updated, validEnd] = d.letterIsOkay(active, uid);
    if (perm !== NO_PERM) {
      this.pushHeap(BW0, code, uid, perm, false, wordStart, validEnd, false, cert, prev, updated, dh);
      if (validEnd && !this.spaceDelimited) {
        this.pushInitialDawg(code, uid, perm, wordStart, true, cert, cont, prev);
        this.pushHeap(BW0, code, uid, perm, false, wordStart, true, false, cert, prev, null, nh);
      }
    }
  }

  continueContext(prev, index, flag, step) {
    const nul = this.nul, outs = this.outs, flags = this.flags;
    const length = index % NUM_LENGTHS;
    const cont = Math.floor(index / NUM_LENGTHS) % 3;
    const useDawgs = index >= 3 * NUM_LENGTHS;
    const prefix = new Array(length).fill(nul);
    let p = prev;
    for (let i = length - 1; i >= 0; i--) {
      while (p !== null && (p.dup || p.code === nul)) p = p.prev;
      if (p !== null) prefix[i] = p.code;
      p = p === null ? null : p.prev;
    }
    const pkey = prefix.join(",");

    if (prev !== null && !this.simple) {
      if (flags[prev.code] === flag) {
        if (cont !== NC_NO_DUP) {
          this.pushDup(length, true, prev.code, prev.uid, certOf(outs[prev.code]), useDawgs,
            NC_ANY, prev, step);
        }
        if (cont === NC_ANY && flag === TN_TOP2 && prev.code !== nul) {
          this.pushDup(length, true, prev.code, prev.uid, certOf(f32(outs[prev.code] + outs[nul])),
            useDawgs, NC_NO_DUP, prev, step);
        }
      }
      if (cont === NC_ONLY_DUP) return;
      if (prev.code !== nul && length > 0 && flags[nul] === flag) {
        this.pushDup(length, false, nul, INVALID, certOf(outs[nul]), useDawgs, NC_ANY, prev, step);
      }
    }

    const tieBoost = (code, prob) => {
      if (prev !== null && cont === NC_ANY && prev.code !== nul &&
          ((prev.code === this.topCode && code === this.secondCode) ||
           (code === this.topCode && prev.code === this.secondCode))) {
        return f32(prob + outs[prev.code]);
      }
      return prob;
    };

    const finals = this.rc.finals.get(pkey);
    if (finals !== undefined) {
      for (const code of finals) {
        if (flags[code] !== flag) continue;
        if (prev !== null && prev.code === code && !this.simple) continue;
        const cert = certOf(outs[code]);
        if (cert < MIN_CERT && code !== nul) continue;
        let uid;
        if (length + 1 > MAX_CODE_LEN) uid = INVALID;
        else {
          const u = this.rc.decoder.get(length ? `${pkey},${code}` : `${code}`);
          uid = u === undefined ? INVALID : u;
        }
        if (length === 0 && code === nul) uid = INVALID;
        this.continueUnichar(code, uid, cert, useDawgs, NC_ANY, prev, step);
        if (flag === TN_TOP2 && code !== nul) {
          const prob = tieBoost(code, f32(outs[code] + outs[nul]));
          this.continueUnichar(code, uid, certOf(prob), useDawgs, NC_ONLY_DUP, prev, step);
        }
      }
    }
    const nexts = this.rc.nexts.get(pkey);
    if (nexts !== undefined) {
      for (const code of nexts) {
        if (flags[code] !== flag) continue;
        if (prev !== null && prev.code === code && !this.simple) continue;
        this.pushDup(length + 1, false, code, INVALID, certOf(outs[code]), useDawgs, NC_ANY, prev, step);
        if (flag === TN_TOP2 && code !== nul) {
          const prob = tieBoost(code, f32(outs[code] + outs[nul]));
          this.pushDup(length + 1, false, code, INVALID, certOf(prob), useDawgs, NC_ONLY_DUP, prev, step);
        }
      }
    }
  }

  decodeStep(t) {
    const step = [];
    for (let i = 0; i < this.nbeams; i++) step.push([]);
    this.beams.push(step);
    this.bestInit = [null, null, null];
    if (t === 0) {
      this.continueContext(null, 0, TN_TOP2, step);
      if (this.dict !== null) this.continueContext(null, 3 * NUM_LENGTHS, TN_TOP2, step);
      return;
    }
    const prevStep = this.beams[t - 1];
    let total = 0;
    for (let tn = 0; tn < 3 && total === 0; tn++) {
      for (let index = 0; index < this.nbeams; index++) {
        const h = prevStep[index];
        for (let i = h.length - 1; i >= 0; i--) this.continueContext(h[i][1], index, tn, step);
      }
      total = 0;
      for (let i = 0; i < NUM_LENGTHS; i++) total += step[i].length;
      if (this.dict !== null) for (let i = 30; i < 40; i++) total += step[i].length;
    }
    if (this.dict !== null) {
      for (let c = 0; c < 3; c++) {
        const node = this.bestInit[c];
        if (node !== null) this.pushNode(BW0, node, step[(3 + c) * NUM_LENGTHS]);
      }
    }
  }

  bestNode() {
    const last = this.beams[this.beams.length - 1];
    let best = null, second = null;
    const halves = this.dict !== null ? [0, 1] : [0];
    for (const cont of [NC_ANY, NC_NO_DUP]) {
      for (const isDawg of halves) {
        for (const pair of last[(isDawg * 3 + cont) * NUM_LENGTHS]) {
          const node = pair[1];
          if (isDawg) {
            let dn = node;
            while (dn !== null && (dn.uid === INVALID || dn.dup)) dn = dn.prev;
            if (dn === null || (!dn.eow && dn.uid !== UNICHAR_SPACE)) continue;
          }
          if (best === null || node.score > best.score) {
            second = best;
            best = node;
          } else if (second === null || node.score > second.score) {
            second = node;
          }
        }
      }
    }
    return best;
  }
}

function run(probs, T, C, model, dict) {
  if (!(T > 0)) return null;
  const s = new Search(recoderFor(model.codes), model.null_char, !!model.simple_text, dict);
  for (let t = 0; t < T; t++) {
    // Float32Array values are exact float32; plain arrays keep them as doubles of floats
    s.computeTopN(Array.from(probs.subarray(t * C, (t + 1) * C)), BW0);
    s.decodeStep(t);
  }
  return s.bestNode();
}

function pathOf(node) {
  const path = [];
  for (let n = node; n !== null; n = n.prev) path.push(n);
  return path.reverse();
}

function idsX(path) {
  const ids = [], xs = [];
  const width = path.length;
  let t = 0;
  while (t < width) {
    while (t < width && path[t].uid === INVALID) t++;
    if (t < width) {
      ids.push(path[t].uid);
      xs.push(t);
      t++;
      while (t < width && path[t].dup) t++;
    }
  }
  return { ids, xs };
}

export function decodeDict(probs, T, C, model, dict) {
  const best = run(probs, T, C, model, dict);
  return best === null ? [] : idsX(pathOf(best)).ids;
}

// decodeDict plus xcoords, per-id permuter and the ExtractBestPathAsWords word split:
// words = [[start, end, permuter]] over ids.
export function decodeDictFull(probs, T, C, model, dict) {
  const best = run(probs, T, C, model, dict);
  if (best === null) return { ids: [], xcoords: [], perm: [], words: [] };
  const path = pathOf(best);
  const { ids, xs } = idsX(path);
  const sd = (u) => (dict === null ? true : dict.isSpaceDelimited(u));
  const words = [];
  let start = 0;
  while (start < ids.length) {
    let end = start + 1;
    for (; end < ids.length; end++) {
      if (ids[end] === UNICHAR_SPACE) break;
      const node = path[xs[end]];
      if (node.sow) break;
      if (node.perm === TOP_CHOICE_PERM && (!sd(ids[end]) || !sd(ids[end - 1]))) break;
    }
    words.push([start, end, path[xs[end - 1]].perm]);
    if (end < ids.length && ids[end] === UNICHAR_SPACE) end++;
    start = end;
  }
  return { ids, xcoords: xs, perm: xs.map((x) => path[x].perm), words };
}
