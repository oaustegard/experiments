// Port of Tesseract 5.3.4 RecodeBeamSearch, no-dictionary path (see tessgpu/beam.py, which this
// mirrors step for step). decode(probs, T, C, model) -> unichar ids, the same ids as
// ExtractBestPathAsUnicharIds.
//
//   probs : Float32Array of length T*C, row-major over timesteps (Tesseract's softmax output)
//   model : { null_char, codes, simple_text }
//           codes[uid] is the array of recoder codes for unichar id uid (the web pack manifest's
//           `codes`). simple_text is false for these models.
//
// Float32 points follow the C++ types: Math.fround wherever a C++ float is stored or produced.
// ProbToCertainty is fround(log(p)) and the `+ cert_offset` step is a double sum rounded back
// to float.
//
// Speed notes (same output as the straightforward port, checked id-for-id by run/bench_beam.mjs):
//  - The recoder (SetupDecoder) is built once per codes array and kept as a trie over code
//    prefixes, so a prefix is a node index, not a string key.
//  - The 64-bit code hash is two uint32 halves, computed without BigInt. It is the same
//    function as before: h = h*nc + floor(hi*nc / 2^32) + code, mod 2^64.
//  - Search nodes live in typed arrays and are referred to by integer id; a heap is a fixed
//    Int32Array slot holding node ids (with copies of their code and hash for the duplicate
//    scan), keyed by score. Two sets of heaps (this step, previous step) are reused, so a
//    decode allocates no per-step objects. One Search is kept per recoder and shape.
//  - Certainties are cached per output index and timestep, so each Math.log runs once per class.
//  - The top two codes come from a threshold scan of the row (or a streaming scan), which is
//    exact whenever the top three values are distinct. The rest of the top five (only needed by
//    the second pass, which the first pass nearly always makes unnecessary) is computed by the
//    heap algorithm on demand. Ties or NaN fall back to the heap algorithm at once.

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
const TWO32 = 4294967296;
const GUESS = 0.1;                          // next row's threshold guess, relative to its 6th value

const f32 = Math.fround;

function certOf(p) {
  const pc = p > KMIN_PROB ? f32(Math.log(p)) : MIN_CERT;
  return f32(pc + CERT_OFFSET);
}

// ---------------------------------------------------------------- recoder (SetupDecoder)
// A trie over code prefixes. Node 0 is the empty prefix. child[n * W + c] is the child of node n
// on code c (-1 if none); finals[n] / nexts[n] are the SetupDecoder lists for prefix n (or null);
// uidAt[n] is the unichar id whose full code ends at n (or INVALID).

const recoderCache = new WeakMap();

function recoderFor(codes) {
  let rc = recoderCache.get(codes);
  if (rc === undefined) {
    rc = buildRecoder(codes);
    recoderCache.set(codes, rc);
  }
  return rc;
}

function buildRecoder(codes) {
  let W = 0;
  for (const code of codes) for (const c of code) if (c + 1 > W) W = c + 1;
  const kids = [new Map()];
  const finals = [null];
  const nexts = [null];
  const uidAt = [INVALID];
  const walk = (seq) => {
    let n = 0;
    for (const c of seq) {
      let m = kids[n].get(c);
      if (m === undefined) {
        m = kids.length;
        kids.push(new Map());
        finals.push(null);
        nexts.push(null);
        uidAt.push(INVALID);
        kids[n].set(c, m);
      }
      n = m;
    }
    return n;
  };
  codes.forEach((code, uid) => {
    uidAt[walk(code)] = uid;                  // later uids overwrite, as decoder.set did
    const last = code.length - 1;
    const pn = walk(code.slice(0, last));
    if (finals[pn] === null) {
      finals[pn] = [code[last]];
      for (let ln = last - 1; ln >= 0; ln--) {
        const p = walk(code.slice(0, ln));
        if (nexts[p] === null) {
          nexts[p] = [code[ln]];
        } else {
          if (!nexts[p].includes(code[ln])) nexts[p].push(code[ln]);
          break;
        }
      }
    } else if (!finals[pn].includes(code[last])) {
      finals[pn].push(code[last]);
    }
  });
  const N = kids.length;
  const child = new Int32Array(N * W).fill(-1);
  for (let n = 0; n < N; n++) for (const [c, k] of kids[n]) child[n * W + c] = k;
  // finalPos / nextPos: index of code c in finals[n] / nexts[n], or -1 (orders the flag subsets).
  const finalPos = new Int32Array(N * W).fill(-1);
  const nextPos = new Int32Array(N * W).fill(-1);
  for (let n = 0; n < N; n++) {
    if (finals[n] !== null) finals[n].forEach((c, i) => { finalPos[n * W + c] = i; });
    if (nexts[n] !== null) nexts[n].forEach((c, i) => { nextPos[n * W + c] = i; });
  }
  return { W, child, finals, nexts, uidAt, finalPos, nextPos };
}

function trieChild(rc, n, c) {
  return n < 0 || c < 0 || c >= rc.W ? -1 : rc.child[n * rc.W + c];
}

// ---------------------------------------------------------------- code hash
// ---------------------------------------------------------------- beam search
// Top-N selection over (value, index) pairs held in two typed arrays, with the same heap
// operations as the Heap above (tie order included).

let PI = 0;

function pairPush(hv, hi, n, v, idx) {
  let hole = n;
  while (hole > 0) {
    const parent = ((hole + 1) >> 1) - 1;
    if (v < hv[parent]) {
      hv[hole] = hv[parent];
      hi[hole] = hi[parent];
      hole = parent;
    } else {
      break;
    }
  }
  hv[hole] = v;
  hi[hole] = idx;
  return n + 1;
}

// Removes the minimum of the first n pairs into PI and returns the new size.
function pairPopMin(hv, hi, n) {
  PI = hi[0];
  n -= 1;
  if (n > 0) {
    const key = hv[n];
    const idx = hi[n];
    let hole = 0;
    for (;;) {
      let child = 2 * hole + 1;
      if (child >= n) break;
      if (child + 1 < n && hv[child + 1] < hv[child]) child++;
      if (hv[child] < key) {
        hv[hole] = hv[child];
        hi[hole] = hi[child];
        hole = child;
      } else {
        break;
      }
    }
    hv[hole] = key;
    hi[hole] = idx;
  }
  return n;
}

// Heap slot hx (0 .. 2*NUM_BEAMS-1) lives at H[hx*HCAP ..]; HL[hx] is its size. Slot hx is
// buffer * NUM_BEAMS + index, where index = cont * NUM_LENGTHS + length as in the Python twin.
// A heap holds at most maxSize + 1 ids (one more during a push), so HCAP = 17 suffices.

const HCAP = BEAM_WIDTHS[1] + 1;

class Search {
  constructor(rc, nul, simple, C) {
    this.rc = rc;
    this.W = rc.W;
    this.nul = nul;
    this.simple = simple;
    this.probs = null;
    this.off = 0;
    this.topCode = -1;
    this.secondCode = -1;
    this.lastCur = 0;
    this.C = C;
    this.lazy = false;          // the row's TOPN subset is not computed yet (see resolveTopN)
    const size = Math.max(C, nul + 1);
    this.flags = new Uint8Array(size).fill(TN_ALSO_RAN);
    this.certA = new Float64Array(size);
    this.stampA = new Int32Array(size).fill(-1);
    this.certB = new Float64Array(size);
    this.stampB = new Int32Array(size).fill(-1);
    this.tick = 0;
    this.pre = new Int32Array(NUM_LENGTHS);
    this.hv = new Float64Array(BEAM_WIDTHS[0] + 2);
    this.hi = new Int32Array(BEAM_WIDTHS[0] + 2);
    this.s2 = new Int32Array(size);
    this.sN = new Int32Array(size);
    this.n2 = 0;
    this.nN = 0;
    this.ord = new Int32Array(size);
    this.wv = new Float64Array(6);
    this.wi = new Int32Array(6);
    this.cand = new Int32Array(C + 1);
    this.guess = 0;
    this.ordp = new Int32Array(size);
    this.H = new Int32Array(2 * NUM_BEAMS * HCAP);
    this.HK = new Int32Array(2 * NUM_BEAMS * HCAP);
    this.HS = new Uint32Array(2 * NUM_BEAMS * HCAP);
    this.HT = new Uint32Array(2 * NUM_BEAMS * HCAP);
    this.HL = new Int32Array(2 * NUM_BEAMS);
    this.n2 = 0;
    this.nN = 0;
    this.cap = 4096;
    this.nn = 0;
    this.nCode = new Int32Array(this.cap);
    this.nUid = new Int32Array(this.cap);
    this.nDup = new Uint8Array(this.cap);
    this.nScore = new Float32Array(this.cap);
    this.nPrev = new Int32Array(this.cap);
    this.nLo = new Uint32Array(this.cap);
    this.nHi = new Uint32Array(this.cap);
  }

  // Forget the previous line; the ids and heap slots are all rewritten before they are read.
  reset() {
    this.nn = 0;
    this.HL.fill(0);
    this.flags.fill(TN_ALSO_RAN);
    this.n2 = 0;
    this.nN = 0;
  }

  grow() {
    const cap = this.cap * 2;
    const copy = (A, T) => {
      const B = new T(cap);
      B.set(A);
      return B;
    };
    this.nCode = copy(this.nCode, Int32Array);
    this.nUid = copy(this.nUid, Int32Array);
    this.nDup = copy(this.nDup, Uint8Array);
    this.nScore = copy(this.nScore, Float32Array);
    this.nPrev = copy(this.nPrev, Int32Array);
    this.nLo = copy(this.nLo, Uint32Array);
    this.nHi = copy(this.nHi, Uint32Array);
    this.cap = cap;
  }

  alloc(code, uid, dup, score, prev, lo, hi) {
    if (this.nn === this.cap) this.grow();
    const id = this.nn++;
    this.nCode[id] = code;
    this.nUid[id] = uid;
    this.nDup[id] = dup ? 1 : 0;
    this.nScore[id] = score;
    this.nPrev[id] = prev;
    this.nLo[id] = lo;
    this.nHi[id] = hi;
    return id;
  }

  // ---- GenericHeap over node ids, keyed by nScore (the top is the smallest key). A slot holds
  // the node id and a copy of its code and hash (HK, HS, HT), so the duplicate scan reads slots.
  moveSlot(dst, src) {
    const H = this.H, HK = this.HK, HS = this.HS, HT = this.HT;
    H[dst] = H[src];
    HK[dst] = HK[src];
    HS[dst] = HS[src];
    HT[dst] = HT[src];
  }

  placeSlot(dst, id) {
    this.H[dst] = id;
    this.HK[dst] = this.nCode[id];
    this.HS[dst] = this.nLo[id];
    this.HT[dst] = this.nHi[id];
  }

  siftUp(base, hole, id) {
    const H = this.H, sc = this.nScore;
    const key = sc[id];
    while (hole > 0) {
      const parent = ((hole + 1) >> 1) - 1;
      if (key < sc[H[base + parent]]) {
        this.moveSlot(base + hole, base + parent);
        hole = parent;
      } else {
        break;
      }
    }
    return hole;
  }

  siftDown(base, n, hole, id) {
    const H = this.H, sc = this.nScore;
    const key = sc[id];
    for (;;) {
      let child = 2 * hole + 1;
      if (child >= n) break;
      if (child + 1 < n && sc[H[base + child + 1]] < sc[H[base + child]]) child++;
      if (sc[H[base + child]] < key) {
        this.moveSlot(base + hole, base + child);
        hole = child;
      } else {
        break;
      }
    }
    return hole;
  }

  heapPush(hx, id) {
    const base = hx * HCAP;
    const n = this.HL[hx];
    this.HL[hx] = n + 1;
    this.placeSlot(base + this.siftUp(base, n, id), id);
  }

  heapPop(hx) {
    const base = hx * HCAP;
    const top = this.H[base];
    const m = this.HL[hx] - 1;
    this.HL[hx] = m;
    if (m > 0) {
      const last = this.H[base + m];
      this.placeSlot(base + this.siftDown(base, m, 0, last), last);
    }
    return top;
  }

  heapReshuffle(hx, index) {
    const base = hx * HCAP;
    const id = this.H[base + index];
    let hole = this.siftDown(base, this.HL[hx], index, id);
    hole = this.siftUp(base, hole, id);
    this.placeSlot(base + hole, id);
  }

  // ComputeTopN for the row probs[o .. o + C). Flags TOP2 for the two best codes (topCode,
  // secondCode) and the null code; the rest of the top five (TOPN) are only needed by a second
  // pass, so they are computed later (resolveTopN) unless the top three are not distinct.
  computeTopN(probs, o, C, topN) {
    const flags = this.flags;
    for (let j = 0; j < this.n2; j++) flags[this.s2[j]] = TN_ALSO_RAN;
    for (let j = 0; j < this.nN; j++) flags[this.sN[j]] = TN_ALSO_RAN;
    this.topCode = -1;
    this.secondCode = -1;
    this.tick++;
    this.probs = probs;
    this.off = o;
    this.lazy = false;
    let a2 = 0, aN = 0;
    if (topN === 5 && C >= 5 && this.topK(3)) {
      const wi = this.wi, s2 = this.s2;
      flags[wi[0]] = TN_TOP2;
      flags[wi[1]] = TN_TOP2;
      this.topCode = wi[0];
      this.secondCode = wi[1];
      s2[0] = wi[0];
      s2[1] = wi[1];
      a2 = 2;
      this.lazy = true;
    } else {
      aN = this.exactTopN(topN);
      a2 = this.n2;
    }
    this.finishSubsets(a2, aN);
  }

  // The null code is always TOP2: move it from the TOPN subset if it was there. Sets the subset
  // sizes.
  finishSubsets(a2, aN) {
    const flags = this.flags, s2 = this.s2, sN = this.sN, nul = this.nul;
    if (flags[nul] === TN_TOPN) {
      for (let j = 0; j < aN; j++) {
        if (sN[j] === nul) {
          sN[j] = sN[--aN];
          break;
        }
      }
      flags[nul] = TN_TOP2;
      s2[a2++] = nul;
    } else if (flags[nul] === TN_ALSO_RAN) {
      flags[nul] = TN_TOP2;
      s2[a2++] = nul;
    }
    this.n2 = a2;
    this.nN = aN;
  }

  // The exact TOPN subset of the current row, for the second pass (which only runs when the
  // first pass produced nothing). The TOP2 decision already made is the same one the heap gives.
  resolveTopN() {
    const flags = this.flags;
    for (let j = 0; j < this.n2; j++) flags[this.s2[j]] = TN_ALSO_RAN;
    for (let j = 0; j < this.nN; j++) flags[this.sN[j]] = TN_ALSO_RAN;
    const aN = this.exactTopN(BEAM_WIDTHS[0]);
    this.lazy = false;
    this.finishSubsets(this.n2, aN);
  }

  // The heap algorithm of ComputeTopN: flags and the TOP2 / TOPN subsets; returns the TOPN count.
  // Leaves the TOP2 count in this.n2.
  exactTopN(topN) {
    const flags = this.flags;
    const hv = this.hv, hi = this.hi;
    const s2 = this.s2, sN = this.sN;
    const probs = this.probs, off = this.off, C = this.C;
    let hn = 0;
    for (let i = 0; i < C; i++) {
      const v = probs[off + i];
      if (hn < topN || v > hv[0]) {
        hn = pairPush(hv, hi, hn, v, i);
        if (hn > topN) hn = pairPopMin(hv, hi, hn);
      }
    }
    let a2 = 0, aN = 0;
    while (hn > 0) {
      hn = pairPopMin(hv, hi, hn);
      const e = PI;
      if (hn > 1) {
        flags[e] = TN_TOPN;
        sN[aN++] = e;
      } else {
        flags[e] = TN_TOP2;
        s2[a2++] = e;
        if (hn === 0) this.topCode = e;
        else this.secondCode = e;
      }
    }
    this.n2 = a2;
    return aN;
  }

  // The K largest values of the row (wv / wi, strictly descending), when they are distinct and no
  // NaN sits among the first five values, where the heap would take them in unconditionally.
  // Otherwise false, and the exact heap decides. Threshold scan first, full scan as fallback.
  topK(K) {
    const head = this.C < 5 ? this.C : 5;
    const probs = this.probs, off = this.off, C = this.C;
    for (let i = 0; i < head; i++) if (probs[off + i] !== probs[off + i]) return false;
    const cand = this.cand, wv = this.wv, wi = this.wi;
    let thr = this.guess;
    let k = 0;
    for (let tries = 0; ; tries++) {
      k = 0;
      for (let i = 0; i < C; i++) {
        cand[k] = i;
        k += probs[off + i] > thr ? 1 : 0;
      }
      if (k >= K || tries >= 3 || !(thr > 0)) break;
      thr *= 0.25;
    }
    let cnt = 0;
    if (k >= K) {
      // Every one of the K largest is a candidate, so the candidates decide.
      for (let q = 0; q < k; q++) {
        const i = cand[q];
        const v = probs[off + i];
        if (cnt === K && !(v > wv[K - 1])) continue;
        let j = cnt < K ? cnt++ : K - 1;
        while (j > 0 && wv[j - 1] < v) {
          wv[j] = wv[j - 1];
          wi[j] = wi[j - 1];
          j--;
        }
        wv[j] = v;
        wi[j] = i;
      }
    } else {
      for (let i = 0; i < C; i++) {
        const v = probs[off + i];
        if (cnt === K && !(v > wv[K - 1])) continue;
        let j = cnt < K ? cnt++ : K - 1;
        while (j > 0 && wv[j - 1] < v) {
          wv[j] = wv[j - 1];
          wi[j] = wi[j - 1];
          j--;
        }
        wv[j] = v;
        wi[j] = i;
      }
    }
    if (cnt !== K) return false;
    for (let j = 0; j < K - 1; j++) if (!(wv[j] > wv[j + 1])) return false;
    this.guess = wv[K - 1] * GUESS;
    return true;
  }

  // Codes of the subset for flag (TOP2 or TOPN, not ALSO_RAN) that appear in a list whose
  // positions are pos[row + c], in list order. Writes them to this.ord; returns the count.
  gather(flag, pos, row, W) {
    const sel = flag === TN_TOP2 ? this.s2 : this.sN;
    const cnt = flag === TN_TOP2 ? this.n2 : this.nN;
    const ord = this.ord, ordp = this.ordp;
    let k = 0;
    for (let j = 0; j < cnt; j++) {
      const c = sel[j];
      if (c >= W) continue;
      const p = pos[row + c];
      if (p < 0) continue;
      let m = k++;
      while (m > 0 && ordp[m - 1] > p) {
        ord[m] = ord[m - 1];
        ordp[m] = ordp[m - 1];
        m--;
      }
      ord[m] = c;
      ordp[m] = p;
    }
    return k;
  }

  // certOf(outs[c]), cached for the current timestep.
  cOf(c) {
    if (this.stampA[c] !== this.tick) {
      this.certA[c] = certOf(this.probs[this.off + c]);
      this.stampA[c] = this.tick;
    }
    return this.certA[c];
  }

  // certOf(f32(outs[c] + outs[null])), cached for the current timestep.
  cNul(c) {
    if (this.stampB[c] !== this.tick) {
      this.certB[c] = certOf(f32(this.probs[this.off + c] + this.probs[this.off + this.nul]));
      this.stampB[c] = this.tick;
    }
    return this.certB[c];
  }

  // Certainty of the (code, null) pair, with the prev-code boost used by final and next codes.
  boostedCert(code, prev, cont) {
    const nul = this.nul;
    if (prev >= 0 && cont === NC_ANY) {
      const pc = this.nCode[prev];
      if (pc !== nul &&
          ((pc === this.topCode && code === this.secondCode) ||
           (code === this.topCode && pc === this.secondCode))) {
        const o = this.off;
        return certOf(f32(f32(this.probs[o + code] + this.probs[o + nul]) + this.probs[o + pc]));
      }
    }
    return this.cNul(code);
  }

  // PushHeapIfBetter, with UpdateHeapIfMatched folded in (no node is allocated for a match
  // that does not improve). hx is the heap slot.
  pushHeap(maxSize, code, uid, dup, cert, prev, hx) {
    const nScore = this.nScore;
    const H = this.H;
    const base = hx * HCAP;
    const n = this.HL[hx];
    const score = prev < 0 ? cert : f32(cert + nScore[prev]);
    if (n < maxSize || score > nScore[H[base]]) {
      let lo = 0, hi = 0;
      if (prev >= 0) {
        lo = this.nLo[prev];
        hi = this.nHi[prev];
      }
      if (!dup && code !== this.nul) {
        // The 64-bit code hash, as hashInto: h = h*nc + floor(hi*nc / 2^32) + code, mod 2^64.
        const nc = this.W;
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
      }
      const HK = this.HK, HS = this.HS, HT = this.HT;
      for (let idx = 0; idx < n; idx++) {
        const slot = base + idx;
        if (HK[slot] === code && HS[slot] === lo && HT[slot] === hi) {
          if (score > nScore[H[slot]]) {
            this.placeSlot(slot, this.alloc(code, uid, dup, score, prev, lo, hi));
            this.heapReshuffle(hx, idx);
          }
          return;
        }
      }
      this.heapPush(hx, this.alloc(code, uid, dup, score, prev, lo, hi));
      if (this.HL[hx] > maxSize) this.heapPop(hx);
    }
  }

  pushDup(length, dup, code, uid, cert, cont, prev, cur) {
    cert = f32(cert * DICT_RATIO);
    if (cert >= MIN_CERT || code === this.nul) {
      this.pushHeap(BEAM_WIDTHS[length], code, uid, dup, cert, prev,
                    cur * NUM_BEAMS + cont * NUM_LENGTHS + length);
    }
  }

  continueUnichar(code, uid, cert, cont, prev, cur) {
    cert = f32(cert * DICT_RATIO);
    this.pushHeap(BEAM_WIDTHS[0], code, uid, false, cert, prev, cur * NUM_BEAMS + cont * NUM_LENGTHS);
  }

  // ContinueContext, no-dawg branch. prev is a node id, or -1 for the empty path.
  continueContext(prev, index, flag, cur) {
    const nul = this.nul;
    const flags = this.flags;
    const simple = this.simple;
    const length = index % NUM_LENGTHS;
    const cont = (index - length) / NUM_LENGTHS;
    const pre = this.pre;
    for (let i = 0; i < length; i++) pre[i] = nul;
    let p = prev;
    for (let i = length - 1; i >= 0; i--) {
      while (p >= 0 && (this.nDup[p] === 1 || this.nCode[p] === nul)) p = this.nPrev[p];
      if (p >= 0) pre[i] = this.nCode[p];
      p = p < 0 ? -1 : this.nPrev[p];
    }
    const rc = this.rc;
    let pn = 0;
    for (let i = 0; i < length && pn >= 0; i++) pn = trieChild(rc, pn, pre[i]);

    if (prev >= 0 && !simple) {
      const pc = this.nCode[prev];
      const pu = this.nUid[prev];
      if (flags[pc] === flag) {
        if (cont !== NC_NO_DUP) {
          this.pushDup(length, true, pc, pu, this.cOf(pc), NC_ANY, prev, cur);
        }
        if (cont === NC_ANY && flag === TN_TOP2 && pc !== nul) {
          this.pushDup(length, true, pc, pu, this.cNul(pc), NC_NO_DUP, prev, cur);
        }
      }
      if (cont === NC_ONLY_DUP) return;
      if (pc !== nul && length > 0 && flags[nul] === flag) {
        this.pushDup(length, false, nul, INVALID, this.cOf(nul), NC_ANY, prev, cur);
      }
    }

    const W = rc.W;
    const finals = pn >= 0 ? rc.finals[pn] : null;
    if (finals !== null) {
      const row = pn * W;
      let k;
      if (flag === TN_ALSO_RAN) {
        k = -1;
      } else {
        k = this.gather(flag, rc.finalPos, row, W);
      }
      const cands = k < 0 ? finals : this.ord;
      const nc = k < 0 ? finals.length : k;
      for (let q = 0; q < nc; q++) {
        const code = cands[q];
        if (k < 0 && flags[code] !== flag) continue;
        if (prev >= 0 && this.nCode[prev] === code && !simple) continue;
        const cert = this.cOf(code);
        if (cert < MIN_CERT && code !== nul) continue;
        let uid = INVALID;
        if (length + 1 <= MAX_CODE_LEN) {
          const cn = trieChild(rc, pn, code);
          if (cn >= 0) uid = rc.uidAt[cn];
        }
        if (length === 0 && code === nul) uid = INVALID;
        this.continueUnichar(code, uid, cert, NC_ANY, prev, cur);
        if (flag === TN_TOP2 && code !== nul) {
          this.continueUnichar(code, uid, this.boostedCert(code, prev, cont), NC_ONLY_DUP, prev, cur);
        }
      }
    }

    const nexts = pn >= 0 ? rc.nexts[pn] : null;
    if (nexts !== null) {
      const row = pn * W;
      let k;
      if (flag === TN_ALSO_RAN) {
        k = -1;
      } else {
        k = this.gather(flag, rc.nextPos, row, W);
      }
      const cands = k < 0 ? nexts : this.ord;
      const nc = k < 0 ? nexts.length : k;
      for (let q = 0; q < nc; q++) {
        const code = cands[q];
        if (k < 0 && flags[code] !== flag) continue;
        if (prev >= 0 && this.nCode[prev] === code && !simple) continue;
        this.pushDup(length + 1, false, code, INVALID, this.cOf(code), NC_ANY, prev, cur);
        if (flag === TN_TOP2 && code !== nul) {
          this.pushDup(length + 1, false, code, INVALID, this.boostedCert(code, prev, cont),
                       NC_ONLY_DUP, prev, cur);
        }
      }
    }
  }

  // DecodeStep (no-dawg). Step t uses slots of parity t & 1; the previous step's slots are read.
  decodeStep(t) {
    const cur = t & 1;
    const prevB = cur ^ 1;
    this.lastCur = cur;
    for (let i = 0; i < NUM_BEAMS; i++) this.HL[cur * NUM_BEAMS + i] = 0;
    if (t === 0) {
      this.continueContext(-1, 0, TN_TOP2, cur);
      return;
    }
    let total = 0;
    for (let tn = 0; tn < 3 && total === 0; tn++) {
      if (tn === 1 && this.lazy) this.resolveTopN();
      for (let index = 0; index < NUM_BEAMS; index++) {
        const hx = prevB * NUM_BEAMS + index;
        const base = hx * HCAP;
        for (let i = this.HL[hx] - 1; i >= 0; i--) {
          this.continueContext(this.H[base + i], index, tn, cur);
        }
      }
      total = 0;
      for (let i = 0; i < NUM_LENGTHS; i++) total += this.HL[cur * NUM_BEAMS + i];
    }
  }

  // ExtractBestPaths: best node among the length-0 heaps of the ANY and NO_DUP continuations.
  bestNode() {
    const sc = this.nScore;
    let best = -1;
    let second = -1;
    for (const cont of [NC_ANY, NC_NO_DUP]) {
      const hx = this.lastCur * NUM_BEAMS + cont * NUM_LENGTHS;
      const base = hx * HCAP;
      for (let i = 0; i < this.HL[hx]; i++) {
        const id = this.H[base + i];
        if (best < 0 || sc[id] > sc[best]) {
          second = best;
          best = id;
        } else if (second < 0 || sc[id] > sc[second]) {
          second = id;
        }
      }
    }
    return best;
  }

  // ExtractPathAsUnicharIds, the unichar-id part only (certainties do not affect ids).
  pathIds(node) {
    const path = [];
    for (let n = node; n >= 0; n = this.nPrev[n]) path.push(n);
    path.reverse();
    const ids = [];
    const width = path.length;
    let t = 0;
    while (t < width) {
      while (t < width && this.nUid[path[t]] === INVALID) t++;
      if (t < width) {
        ids.push(this.nUid[path[t]]);
        t++;
        while (t < width && this.nDup[path[t]] === 1) t++;
      }
    }
    return ids;
  }
}

// One Search per recoder and shape, reused across lines: its typed arrays are the expensive part.
const searchCache = new WeakMap();

function searchFor(rc, nul, simple, C) {
  let byShape = searchCache.get(rc);
  if (byShape === undefined) {
    byShape = new Map();
    searchCache.set(rc, byShape);
  }
  const key = `${nul}|${simple}|${C}`;
  let s = byShape.get(key);
  if (s === undefined) {
    s = new Search(rc, nul, simple, C);
    byShape.set(key, s);
  } else {
    s.reset();
  }
  return s;
}

export function decode(probs, T, C, model) {
  if (!(T > 0)) return [];
  const rc = recoderFor(model.codes);
  const s = searchFor(rc, model.null_char, !!model.simple_text, C);
  for (let t = 0; t < T; t++) {
    s.computeTopN(probs, t * C, C, BEAM_WIDTHS[0]);
    s.decodeStep(t);
  }
  const best = s.bestNode();
  if (best < 0) return [];
  return s.pathIds(best);
}
