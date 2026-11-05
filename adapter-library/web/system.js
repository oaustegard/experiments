// Spiky-ball decider v0.1.0: one frozen Ettin-17m, a router spike, four workflow spikes and a kNN
// memory that grows from the corrections a user makes in the page.
//
//   const sys = await loadSystem("models/");
//   const r = await sys.route(stateStr);            // { probs: {spike: p, none: p}, spike, gap, pNone }
//   const a = await sys.answer(spikeId, stateStr);  // { questions: {name: {spike, knn, blend, lambda, ...}}, key }
//   sys.teach(spikeId, a, { action: "escalate_to_human" });   // adds a labelled case to the memory
//
// The prompt layout and the kNN rule mirror run/shared.py, run/knn.py and browser-decider's decider.js.
// The blend weight per question is chosen on the memory itself: the spike is frozen and never saw those
// cases, so its stored predictions are already out of sample, and kNN is scored leave-one-out. With
// fewer than MIN_LABELS labels for a question the spike answers alone.

import { SpikeEngine } from "./engine.js";

const TOKENIZERS = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
const MAX_LEN = 1024, K = 8, TAU = 0.05, EPS = 0.01, MIN_LABELS = 4;
// v0.2.0: the fitted lambda is pulled toward the spike while labels are few: (n * fit + PRIOR * 1) / (n + PRIOR),
// so 4 labels can move it at most halfway (v0.1.2 let 8 lucky labels hand a question wholly to the memory)
const PRIOR = 4;
const GRID = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
const NOUL_DEFAULT = { true: "The statement is true for this case.", false: "The statement is not true for this case." };
const STORE = "adapter-library/memory/";

export function optionsOf(q) {
  const c = q.criteria;
  if (q.type === "noul") return c && Object.keys(c).length ? [["true", c.true], ["false", c.false]] : Object.entries(NOUL_DEFAULT);
  if (Array.isArray(c)) return c.map((d, i) => [String(i), d]);
  return Object.entries(c);
}

const softmax = (z, T = 1) => {
  const v = Array.from(z, (x) => x / T), mx = Math.max(...v), e = v.map((x) => Math.exp(x - mx)), s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

export async function loadSystem(base, opts = {}) {
  base = new URL(base.endsWith("/") ? base : base + "/", location.href);
  const t0 = performance.now();
  const J = (p) => fetch(new URL(p, base)).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });
  const B = (p) => fetch(new URL(p, base)).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.arrayBuffer(); });
  const system = await J("system.json");
  const files = [system.router.file, ...system.spikes.map((s) => s.file)];
  const bb = system.base ?? "base/";
  const [manifest, tj, data, ...parts] = await Promise.all([J(bb + "manifest.json"), J(bb + "tokenizer.json"), B(bb + "weights.bin"),
    ...files.flatMap((f) => [J(`${f}.json`), B(`${f}.bin`)])]);
  const spikes = {};
  files.forEach((f, i) => { spikes[i === 0 ? "router" : system.spikes[i - 1].id] = { m: parts[2 * i], bytes: new Uint8Array(parts[2 * i + 1]) }; });
  const bytes = data.byteLength + parts.filter((x) => x instanceof ArrayBuffer).reduce((a, x) => a + x.byteLength, 0);
  const fetchMs = performance.now() - t0;

  const { Tokenizer } = opts.tokenizers ?? await import(TOKENIZERS);
  const tok = new Tokenizer(tj, {});
  const sp = (t) => tok.encode(t, { add_special_tokens: false }).ids;
  const [CLS, SEP, MASK] = ["[CLS]", "[SEP]", "[MASK]"].map((t) => tok.encode(t, { add_special_tokens: false }).ids[0]);
  const build = (type, instructions, options, stateStr, pool = false) => {
    const ids = [CLS, ...sp(`${type}: ${instructions}`), SEP];
    const pos = [];
    for (const [name, desc] of options) { pos.push(ids.length); ids.push(MASK, ...sp(` ${name}: ${desc}`)); }
    ids.push(SEP);
    const start = ids.length, budget = MAX_LEN - ids.length - 1;
    ids.push(...sp(stateStr).slice(0, budget));
    const end = ids.length;
    ids.push(SEP);
    return pool ? { ids, pos, pool: [start, end] } : { ids, pos };
  };

  const adapter = opts.device ? null : await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!opts.device && !adapter) throw new Error("WebGPU is not available in this browser");
  const device = opts.device ?? await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  const t1 = performance.now();
  const engine = await SpikeEngine.create(device, manifest, data, { maxTokens: 4096, maxSeqs: 16, maxPos: 2048 });
  const initMs = performance.now() - t1;
  let active = null;
  const use = (id) => { if (active !== id) { engine.setSpike(spikes[id].m, spikes[id].bytes); active = id; } };

  // ---- memory: per spike, labelled cases { id, key: Float32Array, labels: {q: option}, spikeP: {q: [p]} }
  let memory = {};
  const storeKey = STORE + (system.set ?? "workflows");
  try { memory = JSON.parse(localStorage.getItem(storeKey) || "{}"); } catch { memory = {}; }
  for (const k of Object.keys(memory)) memory[k].forEach((it) => { it.key = Float32Array.from(it.key); });
  const save = () => {
    try { localStorage.setItem(storeKey, JSON.stringify(Object.fromEntries(Object.entries(memory).map(([k, v]) => [k, v.map((it) => ({ ...it, key: Array.from(it.key) }))])))); } catch { /* storage unavailable: memory lasts for this page */ }
  };

  function knn(items, q, opts, key, skip = null) {
    const pool = items.filter((it) => it !== skip && it.labels[q] !== undefined);
    if (!pool.length) return null;
    const s = pool.map((it) => dot(it.key, key)), order = s.map((_, i) => i).sort((a, b) => s[b] - s[a]).slice(0, Math.min(K, pool.length));
    const mx = s[order[0]], p = new Array(opts.length).fill(0);
    let wsum = 0;
    for (const i of order) { const w = Math.exp((s[i] - mx) / TAU); wsum += w; p[opts.indexOf(pool[i].labels[q])] += w; }
    return p.map((x) => (1 - EPS) * x / wsum + EPS / opts.length);
  }

  // λ for question q of a spike: minimise the blend's cross-entropy on the memory (spike out of sample, kNN leave-one-out)
  function lambdaFor(id, q, opts) {
    const items = (memory[id] ?? []).filter((it) => it.labels[q] !== undefined);
    if (items.length < MIN_LABELS) return { lambda: 1, n: items.length };
    const pairs = items.map((it) => ({ y: opts.indexOf(it.labels[q]), ps: it.spikeP[q], pk: knn(items, q, opts, it.key, it) }));
    let best = 1, bestLoss = Infinity;
    for (const l of GRID) {
      const loss = pairs.reduce((a, r) => a - Math.log(l * r.ps[r.y] + (1 - l) * r.pk[r.y] + 1e-9), 0);
      if (loss < bestLoss - 1e-9) { bestLoss = loss; best = l; }
    }
    return { lambda: (items.length * best + PRIOR) / (items.length + PRIOR), fit: best, n: items.length };
  }

  async function route(stateStr) {
    use("router");
    const opts = [...system.spikes.map((s) => [s.id, s.description]), ["none", system.router.none]];
    const t = performance.now();
    const [z] = await engine.score([build("choice", system.router.question.replace(/^choice: /, ""), opts, stateStr)]);
    const p = softmax(z), probs = Object.fromEntries(opts.map(([o], i) => [o, p[i]]));
    const pNone = probs.none, ranked = system.spikes.map((s) => s.id).sort((a, b) => probs[b] - probs[a]);
    return { probs, pNone, gap: pNone >= system.router.threshold, spike: ranked[0], ranked, ms: performance.now() - t };
  }

  async function answer(id, stateStr) {
    use(id);
    const sm = spikes[id].m, names = Object.keys(sm.questions);
    const seqs = names.map((n, i) => { const q = sm.questions[n]; return { n, q, opts: optionsOf(q), ...build(q.type, q.instructions, optionsOf(q), stateStr, i === 0) }; });
    const t = performance.now();
    const z = await engine.score(seqs);
    const key = z.keys[0], out = {};
    seqs.forEach((s, i) => {
      const o = s.opts.map(([x]) => x), ps = softmax(z[i], sm.temps[s.q.type] ?? 1);
      const pk = knn(memory[id] ?? [], s.n, o, key);
      const { lambda, n } = lambdaFor(id, s.n, o);
      const blend = pk ? ps.map((x, j) => lambda * x + (1 - lambda) * pk[j]) : ps;
      out[s.n] = { type: s.q.type, instructions: s.q.instructions, options: s.opts, spike: ps, knn: pk, blend, lambda: pk ? lambda : 1, labels: n,
                   label: o[blend.indexOf(Math.max(...blend))] };
    });
    return { spike: id, questions: out, key, ms: performance.now() - t };
  }

  function teach(id, ans, labels, caseId = null) {
    memory[id] ??= [];
    const spikeP = Object.fromEntries(Object.entries(ans.questions).map(([q, v]) => [q, v.spike]));
    const prev = caseId && memory[id].find((it) => it.id === caseId);
    if (prev) Object.assign(prev.labels, labels);
    else memory[id].push({ id: caseId, key: ans.key, labels: { ...labels }, spikeP });
    save();
  }

  async function seed(id, items) {
    for (const it of items) teach(id, await answer(id, it.state), it.gold, it.id);
    return (memory[id] ?? []).length;
  }

  const clear = () => { memory = {}; save(); };
  const memCount = () => Object.fromEntries(system.spikes.map((s) => [s.id, (memory[s.id] ?? []).length]));
  return { system, route, answer, teach, seed, clear, memCount, spikes, engine, build,
           info: { bytes, fetchMs, initMs, version: system.version } };
}
