// Port of Tesseract 5.3.4 RecodeBeamSearch, no-dictionary path (see tessgpu/beam.py, which this
// mirrors step for step). decode(probs, T, C, model) -> unichar ids, the same ids as
// ExtractBestPathAsUnicharIds.
//
//   probs : Float32Array of length T*C, row-major over timesteps (Tesseract's softmax output)
//   model : { null_char, codes, unichars, simple_text }
//           codes[uid] is the array of recoder codes for unichar id uid (the web pack manifest's
//           `codes`); unichars is only used by callers. simple_text is false for these models.
//
// Float32 points follow the C++ types: Math.fround wherever a C++ float is stored or produced.
// ProbToCertainty is fround(log(p)) and the `+ cert_offset` step is a double sum rounded back
// to float. The 64-bit code hash is BigInt, masked to 64 bits.

const INVALID = -1;
const MIN_CERT = -20;                       // RecodeBeamSearch::kMinCertainty (float)
const KMIN_PROB = Math.fround(Math.exp(-20));
const DICT_RATIO = 2.25;
const CERT_OFFSET = -0.085;
const BEAM_WIDTHS = [5, 10, 16, 16, 16, 16, 16, 16, 16, 16];
const NUM_LENGTHS = 10;
const MAX_CODE_LEN = 9;
const NC_ANY = 0, NC_ONLY_DUP = 1, NC_NO_DUP = 2;
const TN_TOP2 = 0, TN_TOPN = 1, TN_ALSO_RAN = 2;
const NUM_BEAMS = 3 * NUM_LENGTHS;
const MASK64 = (1n << 64n) - 1n;

const f32 = Math.fround;

function probToCertainty(p) {
  return p > KMIN_PROB ? f32(Math.log(p)) : MIN_CERT;
}

function certOf(p) {
  return f32(probToCertainty(p) + CERT_OFFSET);
}

// ---------------------------------------------------------------- GenericHeap (min-heap)
// Pairs are [key, data]; the top (index 0) is the smallest key, i.e. the worst entry.

function siftUp(h, hole, pair) {
  const key = pair[0];
  while (hole > 0) {
    const parent = ((hole + 1) >> 1) - 1;
    if (key < h[parent][0]) {
      h[hole] = h[parent];
      hole = parent;
    } else {
      break;
    }
  }
  return hole;
}

function siftDown(h, hole, pair) {
  const n = h.length;
  const key = pair[0];
  for (;;) {
    let child = 2 * hole + 1;
    if (child >= n) break;
    if (child + 1 < n && h[child + 1][0] < h[child][0]) child++;
    if (h[child][0] < key) {
      h[hole] = h[child];
      hole = child;
    } else {
      break;
    }
  }
  return hole;
}

class Heap {
  constructor() {
    this.h = [];
  }
  push(pair) {
    const h = this.h;
    h.push(pair);
    const hole = siftUp(h, h.length - 1, pair);
    h[hole] = pair;
  }
  pop() {
    const h = this.h;
    const top = h[0];
    const last = h.pop();
    if (h.length > 0) {
      const hole = siftDown(h, 0, last);
      h[hole] = last;
    }
    return top;
  }
  reshuffle(index) {
    const h = this.h;
    const pair = h[index];
    let hole = siftDown(h, index, pair);
    hole = siftUp(h, hole, pair);
    h[hole] = pair;
  }
}

// ---------------------------------------------------------------- recoder (SetupDecoder)

function keyOf(arr) {
  return arr.join(",");
}

function buildRecoder(codes) {
  let codeRange = 0;
  for (const code of codes) for (const c of code) if (c + 1 > codeRange) codeRange = c + 1;
  const decoder = new Map();
  const final = new Map();
  const next = new Map();
  codes.forEach((code, uid) => {
    decoder.set(keyOf(code), uid);
    let ln = code.length - 1;
    const prefix = keyOf(code.slice(0, ln));
    if (!final.has(prefix)) {
      final.set(prefix, [code[ln]]);
      ln -= 1;
      while (ln >= 0) {
        const p = keyOf(code.slice(0, ln));
        if (!next.has(p)) {
          next.set(p, [code[ln]]);
        } else {
          const list = next.get(p);
          if (!list.includes(code[ln])) list.push(code[ln]);
          break;
        }
        ln -= 1;
      }
    } else {
      const list = final.get(prefix);
      if (!list.includes(code[ln])) list.push(code[ln]);
    }
  });
  return { codeRange, decoder, final, next };
}

// ---------------------------------------------------------------- beam search

class Node {
  constructor(code, uid, dup, score, prev, hash) {
    this.code = code;
    this.uid = uid;
    this.dup = dup;
    this.score = score;
    this.prev = prev;
    this.hash = hash;
  }
}

class Search {
  constructor(rc, nullChar, simple) {
    this.rc = rc;
    this.null = nullChar;
    this.simple = simple;
    this.beams = [];
    this.outs = null;
    this.flags = null;
    this.topCode = -1;
    this.secondCode = -1;
  }

  computeTopN(outs, topN) {
    const n = outs.length;
    const flags = new Array(n).fill(TN_ALSO_RAN);
    this.topCode = -1;
    this.secondCode = -1;
    const heap = new Heap();
    for (let i = 0; i < n; i++) {
      if (heap.h.length < topN || outs[i] > heap.h[0][0]) {
        heap.push([outs[i], i]);
        if (heap.h.length > topN) heap.pop();
      }
    }
    while (heap.h.length > 0) {
      const e = heap.pop();
      if (heap.h.length > 1) {
        flags[e[1]] = TN_TOPN;
      } else {
        flags[e[1]] = TN_TOP2;
        if (heap.h.length === 0) this.topCode = e[1];
        else this.secondCode = e[1];
      }
    }
    flags[this.null] = TN_TOP2;
    this.flags = flags;
    this.outs = outs;
  }

  codeHash(code, dup, prev) {
    let h = prev === null ? 0n : prev.hash;
    if (!dup && code !== this.null) {
      const nc = BigInt(this.rc.codeRange);
      const carry = (((h >> 32n) * nc) >> 32n) & MASK64;
      h = (h * nc) & MASK64;
      h = (h + carry) & MASK64;
      h = (h + BigInt(code)) & MASK64;
    }
    return h;
  }

  static updateIfMatched(node, heap) {
    const h = heap.h;
    for (let idx = 0; idx < h.length; idx++) {
      const pair = h[idx];
      const old = pair[1];
      if (old.code === node.code && old.hash === node.hash) {
        if (node.score > old.score) {
          pair[1] = node;
          pair[0] = node.score;
          heap.reshuffle(idx);
        }
        return true;
      }
    }
    return false;
  }

  pushHeap(maxSize, code, uid, dup, cert, prev, heap) {
    const score = prev === null ? cert : f32(cert + prev.score);
    const h = heap.h;
    if (h.length < maxSize || score > h[0][0]) {
      const node = new Node(code, uid, dup, score, prev, this.codeHash(code, dup, prev));
      if (Search.updateIfMatched(node, heap)) return;
      heap.push([score, node]);
      if (h.length > maxSize) heap.pop();
    }
  }

  pushDup(length, dup, code, uid, cert, cont, prev, step) {
    cert = f32(cert * DICT_RATIO);
    if (cert >= MIN_CERT || code === this.null) {
      this.pushHeap(BEAM_WIDTHS[length], code, uid, dup, cert, prev, step[cont * NUM_LENGTHS + length]);
    }
  }

  continueUnichar(code, uid, cert, cont, prev, step) {
    cert = f32(cert * DICT_RATIO);
    this.pushHeap(BEAM_WIDTHS[0], code, uid, false, cert, prev, step[cont * NUM_LENGTHS]);
  }

  // Shared "prev code + null" / "prev code + second" boost used by both final and next codes.
  boosted(prob, code, prev, cont) {
    if (prev !== null && cont === NC_ANY && prev.code !== this.null &&
        ((prev.code === this.topCode && code === this.secondCode) ||
         (code === this.topCode && prev.code === this.secondCode))) {
      return f32(prob + this.outs[prev.code]);
    }
    return prob;
  }

  continueContext(prev, index, flag, step) {
    const nullCode = this.null;
    const outs = this.outs;
    const flags = this.flags;
    const length = index % NUM_LENGTHS;
    const cont = Math.floor(index / NUM_LENGTHS);
    const prefix = new Array(length).fill(nullCode);
    let p = prev;
    for (let i = length - 1; i >= 0; i--) {
      while (p !== null && (p.dup || p.code === nullCode)) p = p.prev;
      if (p !== null) prefix[i] = p.code;
      p = p === null ? null : p.prev;
    }
    const pkey = keyOf(prefix);

    if (prev !== null && !this.simple) {
      if (flags[prev.code] === flag) {
        if (cont !== NC_NO_DUP) {
          this.pushDup(length, true, prev.code, prev.uid, certOf(outs[prev.code]), NC_ANY, prev, step);
        }
        if (cont === NC_ANY && flag === TN_TOP2 && prev.code !== nullCode) {
          const cert = certOf(f32(outs[prev.code] + outs[nullCode]));
          this.pushDup(length, true, prev.code, prev.uid, cert, NC_NO_DUP, prev, step);
        }
      }
      if (cont === NC_ONLY_DUP) return;
      if (prev.code !== nullCode && length > 0 && flags[nullCode] === flag) {
        this.pushDup(length, false, nullCode, INVALID, certOf(outs[nullCode]), NC_ANY, prev, step);
      }
    }

    const finalCodes = this.rc.final.get(pkey);
    if (finalCodes) {
      for (const code of finalCodes) {
        if (flags[code] !== flag) continue;
        if (prev !== null && prev.code === code && !this.simple) continue;
        const cert = certOf(outs[code]);
        if (cert < MIN_CERT && code !== nullCode) continue;
        let uid;
        if (length + 1 > MAX_CODE_LEN) {
          uid = INVALID;
        } else {
          const full = pkey === "" ? String(code) : pkey + "," + code;
          uid = this.rc.decoder.has(full) ? this.rc.decoder.get(full) : INVALID;
        }
        if (length === 0 && code === nullCode) uid = INVALID;
        this.continueUnichar(code, uid, cert, NC_ANY, prev, step);
        if (flag === TN_TOP2 && code !== nullCode) {
          const prob = this.boosted(f32(outs[code] + outs[nullCode]), code, prev, cont);
          this.continueUnichar(code, uid, certOf(prob), NC_ONLY_DUP, prev, step);
        }
      }
    }

    const nextCodes = this.rc.next.get(pkey);
    if (nextCodes) {
      for (const code of nextCodes) {
        if (flags[code] !== flag) continue;
        if (prev !== null && prev.code === code && !this.simple) continue;
        const cert = certOf(outs[code]);
        this.pushDup(length + 1, false, code, INVALID, cert, NC_ANY, prev, step);
        if (flag === TN_TOP2 && code !== nullCode) {
          const prob = this.boosted(f32(outs[code] + outs[nullCode]), code, prev, cont);
          this.pushDup(length + 1, false, code, INVALID, certOf(prob), NC_ONLY_DUP, prev, step);
        }
      }
    }
  }

  decodeStep(t) {
    const step = [];
    for (let i = 0; i < NUM_BEAMS; i++) step.push(new Heap());
    this.beams.push(step);
    if (t === 0) {
      this.continueContext(null, 0, TN_TOP2, step);
      return;
    }
    const prevStep = this.beams[t - 1];
    let total = 0;
    for (let tn = 0; tn < 3 && total === 0; tn++) {
      for (let index = 0; index < NUM_BEAMS; index++) {
        const h = prevStep[index].h;
        for (let i = h.length - 1; i >= 0; i--) {
          this.continueContext(h[i][1], index, tn, step);
        }
      }
      total = 0;
      for (let i = 0; i < NUM_LENGTHS; i++) total += step[i].h.length;
    }
  }

  bestNode() {
    const last = this.beams[this.beams.length - 1];
    let best = null;
    let second = null;
    for (const cont of [NC_ANY, NC_NO_DUP]) {
      for (const pair of last[cont * NUM_LENGTHS].h) {
        const node = pair[1];
        if (best === null || node.score > best.score) {
          second = best;
          best = node;
        } else if (second === null || node.score > second.score) {
          second = node;
        }
      }
    }
    return best;
  }
}

function pathIds(node) {
  const path = [];
  for (let n = node; n !== null; n = n.prev) path.push(n);
  path.reverse();
  const ids = [];
  const width = path.length;
  let t = 0;
  while (t < width) {
    while (t < width && path[t].uid === INVALID) t++;
    if (t < width) {
      ids.push(path[t].uid);
      t++;
      while (t < width && path[t].dup) t++;
    }
  }
  return ids;
}

export function decode(probs, T, C, model) {
  if (!(T > 0)) return [];
  const rc = buildRecoder(model.codes);
  const s = new Search(rc, model.null_char, !!model.simple_text);
  for (let t = 0; t < T; t++) {
    const outs = probs.subarray(t * C, (t + 1) * C);
    s.computeTopN(outs, BEAM_WIDTHS[0]);
    s.decodeStep(t);
  }
  const best = s.bestNode();
  if (best === null) return [];
  return pathIds(best);
}
