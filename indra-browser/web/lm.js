// Indra's token n-gram language model and CTC prefix beam search (indra-ocr lm.py), reading the
// packed tables run/lmpack.py writes: per history length n, sorted keys of (history, token)
// pairs with log-probabilities, and sorted history keys with backoff weights. A key packs tokens
// `bits` bits each (0 = line start, t + 1 = token t); keys reach 61 bits, so they are kept as
// (hi, lo) 32-bit halves.

const LOG = Math.log;

function decodeKeys(buf, off, len, n) {
  const bytes = new Uint8Array(buf, off, len);
  const hi = new Uint32Array(n), lo = new Uint32Array(n);
  let p = 0, h = 0, l = 0;
  for (let i = 0; i < n; i++) {
    // one LEB128 gap, added to (h, l)
    let dl = 0, dh = 0, shift = 0, b;
    do {
      b = bytes[p++];
      const v = b & 0x7f;
      if (shift < 32) {
        dl = (dl | (v << shift)) >>> 0;
        if (shift > 25) dh = (dh | (v >>> (32 - shift))) >>> 0;
      } else dh = (dh | (v << (shift - 32))) >>> 0;
      shift += 7;
    } while (b & 0x80);
    const s = l + dl;
    l = s >>> 0;
    h = (h + dh + (s > 0xffffffff ? 1 : 0)) >>> 0;
    hi[i] = h;
    lo[i] = l;
  }
  return { hi, lo };
}

export class NgramLM {
  constructor(buf) {
    const hl = new DataView(buf).getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    const base = 4 + hl;
    this.order = head.order;
    this.vocabSize = head.vocab_size;
    this.bits = head.bits;
    this.t = {};
    let maxP = -Infinity;
    const hbMax = [];
    for (const t of head.tables) {
      const keys = decodeKeys(buf, base + t.keys, t.keys_len, t.n);
      let vals;
      if (t.f32 !== undefined) vals = new Float32Array(buf.slice(base + t.f32, base + t.f32 + 4 * t.n));
      else {
        const lv = new Float32Array(buf.slice(base + t.levels, base + t.levels + 4 * t.nlevels));
        const codes = new Uint8Array(buf, base + t.codes, t.n);
        vals = new Float32Array(t.n);
        for (let i = 0; i < t.n; i++) vals[i] = lv[codes[i]];
      }
      this.t[t.name] = { ...keys, vals, n: t.n };
      let mx = -Infinity;
      for (const v of vals) if (v > mx) mx = v;
      if (t.name.startsWith("hk")) hbMax.push(Math.max(0, t.n ? mx : 0));
      else maxP = Math.max(maxP, mx);
    }
    this.maxLogp = Math.max(maxP, -LOG(this.vocabSize)) + hbMax.reduce((a, b) => a + b, 0);
    this.cache = new Map();
  }

  find(name, h, l) {
    const t = this.t[name];
    let lo = 0, hi = t.n;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (t.hi[m] < h || (t.hi[m] === h && t.lo[m] < l)) lo = m + 1;
      else hi = m;
    }
    return lo < t.n && t.hi[lo] === h && t.lo[lo] === l ? lo : -1;
  }

  // log P(token | history): `hist` the packed last order - 1 history tokens (a BigInt), `c` the token + 1
  logp(hist, c) {
    const key = `${hist}:${c}`;
    let v = this.cache.get(key);
    if (v !== undefined) return v;
    const B = BigInt(this.bits), N = this.order;
    const hb = [], pv = [], hok = [], pok = [];
    for (let n = 0; n < N; n++) {
      const hq = hist & ((1n << (B * BigInt(n))) - 1n);
      const pq = (hq << B) | BigInt(c);
      const hh = Number(hq >> 32n), hl = Number(hq & 0xffffffffn);
      const ph = Number(pq >> 32n), pl = Number(pq & 0xffffffffn);
      const i = this.find(`hk${n}`, hh, hl);
      const j = this.find(`k${n}`, ph, pl);
      hok.push(n === 0 ? true : i >= 0);
      hb.push(i >= 0 ? this.t[`hk${n}`].vals[i] : 0);
      pok.push(j >= 0);
      pv.push(j >= 0 ? this.t[`k${n}`].vals[j] : 0);
    }
    let m = 0;
    while (m + 1 < N && hok[m + 1]) m++;
    let chosen = -1;
    for (let n = m; n >= 0; n--) if (pok[n]) { chosen = n; break; }
    let acc = 0;
    if (chosen >= 0) {
      for (let n = chosen + 1; n <= m; n++) acc += hb[n];
      v = acc + pv[chosen];
    } else {
      for (let n = 0; n <= m; n++) acc += hb[n];
      v = acc - LOG(this.vocabSize);
    }
    if (this.cache.size < 1_000_000) this.cache.set(key, v);
    return v;
  }
}

function lse(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + LOG(Math.exp(a - m) + Math.exp(b - m));
}

// CTC prefix beam search over one line: blank [T], topv / topi [T * 8] (log-probabilities and
// classes, best first). Returns [token ids, the step where each was first emitted].
export function beamSearch(blank, topv, topi, T, lm, alpha = 0.5, beta = 1.5, beam = 8, blankSkip = 0.999, candFloor = 1e-3) {
  const bits = BigInt(lm.bits);
  const hmask = (1n << (bits * BigInt(lm.order - 1))) - 1n;
  // prefix (comma-joined ids) -> [b, nb, lmScore, steps[], hist(BigInt), ids[]]
  let beams = new Map([["", [0.0, -Infinity, 0.0, [], 0n, []]]]);
  const logSkip = LOG(blankSkip), logFloor = LOG(candFloor);
  const reach = beta + alpha * lm.maxLogp;
  for (let t = 0; t < T; t++) {
    const b0 = blank[t];
    if (b0 > logSkip) {
      const nb = new Map();
      for (const [p, e] of beams) nb.set(p, [lse(e[0], e[1]) + b0, -Infinity, e[2], e[3], e[4], e[5]]);
      beams = nb;
      continue;
    }
    const nxt = new Map();
    const add = (key, ids, b, nbv, s, st, hk) => {
      const old = nxt.get(key);
      if (old === undefined) nxt.set(key, [b, nbv, s, st, hk, ids]);
      else nxt.set(key, [lse(old[0], b), lse(old[1], nbv), old[2], old[3], old[4], old[5]]);
    };
    const cands = [];
    for (let q = 0; q < 8; q++) {
      const c = topi[t * 8 + q], v = topv[t * 8 + q];
      if (c !== 0 && v >= logFloor) cands.push([c, v]);
    }
    for (const [key, [b, nb, s, st, hk, ids]] of beams) {
      add(key, ids, lse(b, nb) + b0, -Infinity, s, st, hk);
      const last = ids.length ? ids[ids.length - 1] : null;
      for (const [c, lp] of cands) if (c === last) add(key, ids, -Infinity, nb + lp, s, st, hk);
    }
    let floor = -Infinity;
    if (beams.size >= beam) {
      floor = Infinity;
      for (const e of nxt.values()) floor = Math.min(floor, lse(e[0], e[1]) + e[2]);
    }
    const ext = [];
    for (const [key, [b, nb, s, st, hk, ids]] of beams) {
      const tot = lse(b, nb);
      const last = ids.length ? ids[ids.length - 1] : null;
      for (const [c, lp] of cands) {
        const from = c === last ? b + lp : tot + lp;
        if (from === -Infinity) continue;
        const qk = key === "" ? `${c}` : `${key},${c}`;
        if (nxt.has(qk)) add(qk, null, -Infinity, from, s, st, hk);
        else if (from + s + reach >= floor - 1e-9) ext.push([key, ids, c, from, s, st, hk]);
      }
    }
    for (const [key, ids, c, from, s, st, hk] of ext) {
      const lmp = lm.logp(hk, c + 1);
      const qk = key === "" ? `${c}` : `${key},${c}`;
      add(qk, [...ids, c], -Infinity, from, s + alpha * lmp + beta, [...st, t], ((hk << bits) | BigInt(c + 1)) & hmask);
    }
    const sorted = [...nxt.entries()].map((kv, i) => [kv, i]).sort((x, y) => {
      const sx = lse(x[0][1][0], x[0][1][1]) + x[0][1][2], sy = lse(y[0][1][0], y[0][1][1]) + y[0][1][2];
      return sy - sx || x[1] - y[1];
    }).slice(0, beam);
    beams = new Map(sorted.map(([kv]) => kv));
  }
  let best = null, bs = -Infinity;
  for (const e of beams.values()) {
    const sc = lse(e[0], e[1]) + e[2];
    if (best === null || sc > bs) { best = e; bs = sc; }
  }
  return [best[5], best[3]];
}

// The reader's hook: lazily load lm.bin / lm_<group>.bin of a pack, with the group's parameters.
export function lmFactory(defaults = { alpha: 0.5, beta: 1.5, gate: 0.9, beam: 8, candFloor: 3e-3 }) {
  const loaded = new Map();
  return (ocr, name) => {
    const file = ocr.man.lm?.[name];
    if (!file) return null;
    if (!loaded.has(name)) {
      loaded.set(name, (async () => {
        const t0 = performance.now();
        const lm = new NgramLM(await ocr.fetchBytes(new URL(file, ocr.base).href));
        const p = { ...defaults, ...(ocr.man.lm_params?.[name] ?? {}) };
        ocr.log(`loaded the ${name} n-gram table (order ${lm.order}) in ${(performance.now() - t0).toFixed(0)} ms`);
        return {
          gate: p.gate,
          search: (blank, topv, topi, T) => beamSearch(blank, topv, topi, T, lm, p.alpha, p.beta, p.beam, 0.999, p.candFloor),
        };
      })());
    }
    return loaded.get(name);
  };
}
