// Decision creator core: decision definitions, prompts, splits, metrics, spike packing, the training loop.
// No DOM here; index.html drives it and run/check.mjs tests it in headless Chromium.
//
// A decision is { id, description, questions: { name: { type, instructions, criteria } } } in the
// Typed Decisions shape (type choice | noul | score). A case is { id, state, gold: { name: label | {option: p} },
// split? }. The prompt layout is adapter-library's system.js build(), so a spike this page exports runs
// unchanged in adapter-library's SpikeEngine and in any system.json set.

export const TOKENIZERS = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
export const MAX_LEN = 1024;
const NOUL_DEFAULT = { true: "The statement is true for this case.", false: "The statement is not true for this case." };

export function optionsOf(q) {
  const c = q.criteria;
  if (q.type === "noul") return c && Object.keys(c).length ? [["true", c.true], ["false", c.false]] : Object.entries(NOUL_DEFAULT);
  if (Array.isArray(c)) return c.map((d, i) => [String(i), d]);
  return Object.entries(c ?? {});
}

// Problems with a decision or its cases, as strings; empty when it can be trained.
export function validate(decision, cases = []) {
  const errs = [];
  const qs = decision?.questions ?? {};
  if (!Object.keys(qs).length) errs.push("the decision has no questions");
  for (const [n, q] of Object.entries(qs)) {
    if (!["choice", "noul", "score"].includes(q.type)) errs.push(`${n}: type must be choice, noul or score`);
    if (!q.instructions) errs.push(`${n}: no instructions`);
    if (q.type !== "noul" && optionsOf(q).length < 2) errs.push(`${n}: needs at least two options`);
  }
  cases.forEach((c, i) => {
    if (typeof c.state !== "string" || !c.state) errs.push(`case ${c.id ?? i}: state must be a non-empty string`);
    for (const [n, g] of Object.entries(c.gold ?? {})) {
      if (!qs[n]) { errs.push(`case ${c.id ?? i}: gold for unknown question ${n}`); continue; }
      const names = optionsOf(qs[n]).map(([o]) => o);
      if (typeof g === "string" ? !names.includes(g) : Object.keys(g).some((o) => !names.includes(o)))
        errs.push(`case ${c.id ?? i}: ${n} gold names an option the question does not have`);
    }
  });
  return errs;
}

// Gold for one question as a distribution over its options, or null when the case has no label for it.
export function goldDist(q, g) {
  if (g === undefined || g === null) return null;
  const names = optionsOf(q).map(([o]) => o);
  const p = new Float32Array(names.length);
  if (typeof g === "string") { const i = names.indexOf(g); if (i < 0) return null; p[i] = 1; return p; }
  names.forEach((o, i) => { p[i] = Number(g[o] ?? 0); });
  const s = p.reduce((a, b) => a + b, 0);
  if (!(s > 0)) return null;
  for (let i = 0; i < p.length; i++) p[i] /= s;
  return p;
}

export async function loadTokenizer(url, mod = null) {
  const tj = await fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });
  const { Tokenizer } = mod ?? await import(TOKENIZERS);
  const tok = new Tokenizer(tj, {});
  const sp = (t) => tok.encode(t, { add_special_tokens: false }).ids;
  const [CLS, SEP, MASK] = ["[CLS]", "[SEP]", "[MASK]"].map((t) => sp(t)[0]);
  // adapter-library/web/system.js build(), verbatim layout
  const build = (type, instructions, options, stateStr) => {
    const ids = [CLS, ...sp(`${type}: ${instructions}`), SEP];
    const pos = [];
    for (const [name, desc] of options) { pos.push(ids.length); ids.push(MASK, ...sp(` ${name}: ${desc}`)); }
    ids.push(SEP);
    const budget = MAX_LEN - ids.length - 1;
    ids.push(...sp(stateStr).slice(0, budget));
    ids.push(SEP);
    return { ids, pos };
  };
  return { build, sp };
}

// One sequence per (case, question) that has a label; key identifies it for the frozen-row cache.
export function examples(decision, cases, build) {
  const out = [];
  for (const c of cases) {
    for (const [n, q] of Object.entries(decision.questions)) {
      const target = goldDist(q, c.gold?.[n]);
      if (!target) continue;
      const opts = optionsOf(q);
      out.push({ key: `${c.id}\u0000${n}`, caseId: c.id, q: n, type: q.type, options: opts.map(([o]) => o), target, ...build(q.type, q.instructions, opts, c.state) });
    }
  }
  return out;
}

export function rng(seed) {
  let a = (seed >>> 0) || 1;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function shuffle(arr, r) { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

// Cases marked split: "test" stay test. The rest are shuffled by seed (sorted by id first, so the split
// does not depend on the order cases were added); if no case is marked test, testFrac of them test.
// maxTrain caps the labelled cases used for training and validation together (a label budget, as in
// adapter-library's "n revealed cases"); valFrac of those validate.
export function splitCases(cases, { valFrac = 0.15, testFrac = 0.2, seed = 0, maxTrain = 0, maxTest = 0 } = {}) {
  const marked = cases.filter((c) => c.split === "test");
  const rest = shuffle(cases.filter((c) => c.split !== "test").sort((a, b) => String(a.id).localeCompare(String(b.id))), rng(seed + 1));
  const nTest = marked.length ? 0 : Math.round(rest.length * testFrac);
  const test = marked.length ? marked : rest.slice(0, nTest);
  let pool = rest.slice(nTest);
  if (maxTrain > 0) pool = pool.slice(0, maxTrain);
  const nVal = Math.max(pool.length > 4 && valFrac > 0 ? 1 : 0, Math.round(pool.length * valFrac));
  const train = pool.slice(nVal);
  // maxTest caps the test cases (a quick check on a slow GPU); the cap takes a seeded sample, not the first ones
  const testUsed = maxTest > 0 && test.length > maxTest ? shuffle(test, rng(seed + 2)).slice(0, maxTest) : test;
  return { train, val: pool.slice(0, nVal), test: testUsed };
}

export const softmax = (z, T = 1) => {
  let mx = -Infinity;
  for (const x of z) mx = Math.max(mx, x / T);
  const e = Array.from(z, (x) => Math.exp(x / T - mx)), s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};

// The dataset's scorer (browser-decider/run/common.py score_one): argmax match against gold's argmax, KL(gold || pred).
export function scoreOne(p, g) {
  let ap = 0, ag = 0, kl = 0;
  for (let i = 0; i < p.length; i++) { if (p[i] > p[ap]) ap = i; if (g[i] > g[ag]) ag = i; if (g[i] > 0) kl += g[i] * Math.log(g[i] / Math.max(p[i], 1e-9)); }
  return { ok: ap === ag, kl };
}

export function metrics(exs, probs) {
  const by = {};
  let ok = 0, kl = 0;
  exs.forEach((e, i) => {
    const r = scoreOne(probs[i], e.target);
    ok += r.ok; kl += r.kl;
    const b = (by[e.q] ??= { n: 0, ok: 0, kl: 0 });
    b.n++; b.ok += r.ok; b.kl += r.kl;
  });
  const per = Object.fromEntries(Object.entries(by).map(([q, b]) => [q, { n: b.n, accuracy: b.ok / b.n, kl: b.kl / b.n }]));
  return { n: exs.length, accuracy: exs.length ? ok / exs.length : null, kl: exs.length ? kl / exs.length : null, questions: per };
}

const TGRID = Array.from({ length: 49 }, (_, i) => +(0.25 * 1.07 ** i).toFixed(4));   // 0.25 .. ~6.4
// Per-type temperature minimising KL on (val) logits; temperature 1 for a type with no examples.
export function fitTemps(exs, logits) {
  const temps = {};
  for (const type of new Set(exs.map((e) => e.type))) {
    let best = 1, bestL = Infinity;
    for (const T of TGRID) {
      let L = 0;
      exs.forEach((e, i) => { if (e.type === type) L += scoreOne(softmax(logits[i], T), e.target).kl; });
      if (L < bestL) { bestL = L; best = T; }
    }
    temps[type] = best;
  }
  return temps;
}
export const probsOf = (exs, logits, temps = {}) => exs.map((e, i) => softmax(logits[i], temps[e.type] ?? 1));

// The prior: each question's mean train gold distribution, predicted for every case.
export function priorProbs(trainExs, exs) {
  const mean = {};
  for (const e of trainExs) { const m = (mean[e.q] ??= new Float64Array(e.target.length)); e.target.forEach((p, i) => { m[i] += p; }); }
  return exs.map((e) => { const m = mean[e.q]; if (!m) return e.target.map(() => 1 / e.target.length); const s = m.reduce((a, b) => a + b, 0); return Array.from(m, (x) => x / s); });
}

// ---- fp16 and spike packing (adapter-library/run/pack_system.py pack_spike) ----
const f32b = new Float32Array(1), u32b = new Uint32Array(f32b.buffer);
export function toF16(x) {
  f32b[0] = x; const b = u32b[0], s = (b >>> 16) & 0x8000, e = (b >>> 23) & 0xff, m = b & 0x7fffff;
  if (e === 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
  const E = e - 127 + 15;
  if (E >= 0x1f) return s | 0x7c00;
  if (E <= 0) { if (E < -10) return s; const mm = (m | 0x800000) >> (1 - E); return s | ((mm + 0x1000 + ((mm >> 13) & 1) - 1) >> 13); }
  const h = s | (E << 10) | (m >> 13), rem = m & 0x1fff;
  return rem > 0x1000 || (rem === 0x1000 && (h & 1)) ? h + 1 : h;
}
export function fromF16(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

export const MODS = ["Wqkv", "Wo", "Wi", "Wo_mlp"];

// Spike (manifest part + .bin) -> fp32 arrays: { layers: [{ mod: { A [r*K], B [N*r], r, K, N } }], head: {...} }
export function unpackSpike(sm, bytes) {
  bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const t = (x) => fromF16(new Uint16Array(bytes.buffer.slice(bytes.byteOffset + x.offset, bytes.byteOffset + x.offset + x.length)));
  const layers = sm.layers.map((L) => Object.fromEntries(MODS.map((k) => [k, { A: t(L[k].A), B: t(L[k].B), r: L[k].r, K: L[k].K, N: L[k].N }])));
  const h = sm.head;
  return { layers, head: { w0: t(h.w0), b0: t(h.b0), lnw: t(h.ln_w), lnb: t(h.ln_b), w1: t(h.w1), b1: h.b1 } };
}

// fp32 arrays -> { sm, bytes }; meta carries id, description, questions, temps.
export function packSpike({ layers, head }, meta = {}) {
  const parts = [];
  let off = 0;
  const add = (arr, dims) => {
    const u = new Uint16Array(arr.length);
    for (let i = 0; i < arr.length; i++) u[i] = toF16(arr[i]);
    const pad = (4 - (off % 4)) % 4;
    if (pad) { parts.push(new Uint8Array(pad)); off += pad; }
    const rec = { offset: off, length: u.byteLength, dims, dtype: "float16" };
    parts.push(new Uint8Array(u.buffer)); off += u.byteLength;
    return rec;
  };
  const L = layers.map((l) => Object.fromEntries(MODS.map((k) => {
    const m = l[k];
    return [k, { A: add(m.A, [m.r, m.K]), B: add(m.B, [m.N, m.r]), r: m.r, K: m.K, N: m.N }];
  })));
  const H = head.b0.length;
  const hd = { w0: add(head.w0, [H, H]), b0: add(head.b0, [H]), ln_w: add(head.lnw, [H]), ln_b: add(head.lnb, [H]), w1: add(head.w1, [H]), b1: head.b1 };
  const bytes = new Uint8Array(off);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  return { sm: { layers: L, head: hd, ...meta }, bytes };
}

// The library start: the mean of several spikes' weight deltas, kept exact by stacking their ranks
// (sum_i B_i A_i / n = [B_1 .. B_n] / n . [A_1; ..; A_n]), with their heads averaged. Three rank-8
// spikes give rank 24, inside the engine's padded rank of 32. adapter-library's G arm SVD-cut this
// mean to rank 8 before training it; a head-only run keeps it whole.
export function mergeSpikes(unpacked) {
  const n = unpacked.length;
  const layers = unpacked[0].layers.map((_, li) => Object.fromEntries(MODS.map((k) => {
    const ms = unpacked.map((u) => u.layers[li][k]), K = ms[0].K, N = ms[0].N, r = ms.reduce((a, m) => a + m.r, 0);
    const A = new Float32Array(r * K), B = new Float32Array(N * r);
    let j0 = 0;
    for (const m of ms) {
      A.set(m.A, j0 * K);
      for (let nn = 0; nn < N; nn++) for (let j = 0; j < m.r; j++) B[nn * r + j0 + j] = m.B[nn * m.r + j] / n;
      j0 += m.r;
    }
    return [k, { A, B, r, K, N }];
  })));
  const avg = (key) => { const out = new Float32Array(unpacked[0].head[key].length); for (const u of unpacked) u.head[key].forEach((x, i) => { out[i] += x / n; }); return out; };
  return { layers, head: { w0: avg("w0"), b0: avg("b0"), lnw: avg("lnw"), lnb: avg("lnb"), w1: avg("w1"), b1: unpacked.reduce((a, u) => a + u.head.b1, 0) / n } };
}

// An empty adapter (rank 0 is not packable, so rank 1 with A = B = 0) around a given head.
export function zeroLayers(manifest) {
  return manifest.layers.map((L) => Object.fromEntries(MODS.map((k) => [k, { A: new Float32Array(L[k].K), B: new Float32Array(L[k].N), r: 1, K: L[k].K, N: L[k].N }])));
}

// ---- the training loop, over any backend with step / score / exportSpike / loadSpike ----
//
// Each epoch: shuffled minibatches of `bs` sequences through backend.step; then val logits, per-type
// temperatures fitted on val, val KL. The best epoch by val KL is restored at the end (epoch 0, the
// untrained start, is a candidate, so training can only be kept when it helps on val). Reports go to
// onEvent({ kind: "step" | "epoch" | "done", ... }); shouldStop() ends the run after the current step.
export async function train(backend, data, opts = {}, onEvent = () => {}, shouldStop = () => false) {
  const { epochs = 10, lr = 8e-4, weightDecay = 0.01, bs = 16, seed = 0 } = opts;
  const { train: tr, val, test } = data;
  if (!tr.length) throw new Error("no labelled training sequences");
  const r = rng(seed + 7);
  const evalSet = async (exs) => (exs.length ? backend.score(exs) : []);
  const t0 = performance.now();
  let vz = await evalSet(val);
  let temps = val.length ? fitTemps(val, vz) : {};
  let vm = val.length ? metrics(val, probsOf(val, vz, temps)) : null;
  // exportSpike() may be async (a GPU trainer reads its buffers back); temps ride along in the manifest part
  const snapshot = async (T) => { const s = await backend.exportSpike(); return { sm: { ...s.sm, temps: T }, bytes: s.bytes }; };
  const start = { epoch: 0, val: vm, temps };
  let best = { ...start, spike: await snapshot(temps) };
  onEvent({ kind: "epoch", ...start, trainLoss: null, ms: performance.now() - t0 });
  // adapter-library run/lora.py's schedule: linear warmup over 6% of steps, then linear decay to 0
  const steps = Math.ceil(tr.length / bs), total = steps * epochs, warm = Math.max(1, Math.floor(0.06 * total));
  const lrAt = (s) => lr * Math.min(1, (s + 1) / warm) * Math.max(0, (total - s) / Math.max(1, total - warm));
  let step = 0;
  for (let ep = 1; ep <= epochs; ep++) {
    const te = performance.now(), order = shuffle(tr, r);
    let sum = 0, cnt = 0;
    for (let b = 0; b < steps; b++) {
      const batch = order.slice(b * bs, (b + 1) * bs);
      const { loss } = await backend.step(batch, batch.map((e) => e.target), { lr: lrAt(step), weightDecay, epoch: ep, step });
      sum += loss * batch.length; cnt += batch.length; step++;
      onEvent({ kind: "step", epoch: ep, step, of: steps * epochs, loss });
      if (shouldStop()) break;
    }
    vz = await evalSet(val);
    temps = val.length ? fitTemps(val, vz) : {};
    vm = val.length ? metrics(val, probsOf(val, vz, temps)) : null;
    const rec = { epoch: ep, val: vm, temps, trainLoss: sum / Math.max(cnt, 1), ms: performance.now() - te };
    if (!val.length || vm.kl < best.val.kl) best = { ...rec, spike: await snapshot(temps) };
    onEvent({ kind: "epoch", ...rec, best: best.epoch });
    if (shouldStop()) break;
  }
  await backend.loadSpike(best.spike.sm, best.spike.bytes);
  const tz = await evalSet(test);
  const result = {
    bestEpoch: best.epoch, temps: best.temps,
    val: best.val, test: test.length ? metrics(test, probsOf(test, tz, best.temps)) : null,
    prior: test.length ? metrics(test, priorProbs(tr, test)) : null,
    trainMs: performance.now() - t0, spike: best.spike,
  };
  onEvent({ kind: "done", ...result });
  return result;
}

// Score a set before training, with temperatures fitted on val (the zero-shot reading of the start spike).
export async function evaluateStart(backend, data) {
  const { val, test } = data;
  const vz = val.length ? await backend.score(val) : [];
  const temps = val.length ? fitTemps(val, vz) : {};
  const tz = test.length ? await backend.score(test) : [];
  return { temps, val: val.length ? metrics(val, probsOf(val, vz, temps)) : null, test: test.length ? metrics(test, probsOf(test, tz, temps)) : null };
}
