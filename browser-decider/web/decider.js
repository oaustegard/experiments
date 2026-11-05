// Browser decider: typed decisions (choice / score / noul) from a 10 MB Ettin-17m, on WebGPU.
//
//   const d = await loadDecider("models/spec-e17c-t4-w8L63/");
//   const answers = await d.decide(state, questions);
//   // { action: { type: "choice", label, confidence, probabilities: { answer_directly: 0.71, ... } }, ... }
//
// `state` is a string (sent as is) or an object (serialized like Python's json.dumps, the form the
// model was trained on). `questions` is the body of a POST /v1/systemone request:
// { name: { type, instructions, criteria } }, criteria a { option: description } map for choice,
// a list of level descriptions for score, and optional { true, false } for noul.
// The layout mirrors run/common.py and run/train_enc.py (Enc.build); change them together.

import { DeciderEngine } from "./engine.js";

const TOKENIZERS = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
const MAX_LEN = 1024;
export const NOUL_DEFAULT = { true: "The statement is true for this case.", false: "The statement is not true for this case." };

export function optionsOf(q) {
  const c = q.criteria;
  if (q.type === "noul") return c && Object.keys(c).length ? [["true", c.true], ["false", c.false]] : Object.entries(NOUL_DEFAULT);
  if (Array.isArray(c)) return c.map((d, i) => [String(i), d]);
  return Object.entries(c);
}

// Python json.dumps(obj, ensure_ascii=False) for the shapes states take. Floats that happen to be
// whole numbers print as 300020 here and 300020.0 in Python; pass the original string to avoid that.
export function pyJson(v) {
  if (v === null) return "null";
  if (Array.isArray(v)) return "[" + v.map(pyJson).join(", ") + "]";
  if (typeof v === "object") return "{" + Object.entries(v).map(([k, x]) => JSON.stringify(k) + ": " + pyJson(x)).join(", ") + "}";
  if (typeof v === "boolean") return v ? "true" : "false";
  return JSON.stringify(v);
}

export async function loadDecider(base, opts = {}) {
  base = new URL(base.endsWith("/") ? base : base + "/", location.href);
  const t0 = performance.now();
  const [manifest, tj, data] = await Promise.all([
    fetch(new URL("manifest.json", base)).then((r) => r.json()),
    fetch(new URL("tokenizer.json", base)).then((r) => r.json()),
    fetch(new URL("weights.bin", base)).then((r) => r.arrayBuffer())]);
  const d = await deciderFrom({ manifest, tj, data }, opts);
  d.info.fetchMs = performance.now() - t0 - d.info.initMs - d.info.tokMs;
  return d;
}

// Same, from parts already in memory (a GPU Bridge job passes them as inputs).
export async function deciderFrom({ manifest, tj, data }, opts = {}) {
  const t0 = performance.now();
  const { Tokenizer } = opts.tokenizers ?? await import(TOKENIZERS);
  const tok = new Tokenizer(tj, {});
  const sp = (t) => tok.encode(t, { add_special_tokens: false }).ids;
  const [CLS, SEP, MASK] = ["[CLS]", "[SEP]", "[MASK]"].map((t) => tok.encode(t, { add_special_tokens: false }).ids[0]);

  const build = (type, instructions, options, stateStr) => {
    const ids = [CLS, ...sp(`${type}: ${instructions}`), SEP];
    const pos = [];
    for (const [name, desc] of options) { pos.push(ids.length); ids.push(MASK, ...sp(` ${name}: ${desc}`)); }
    ids.push(SEP);
    const budget = MAX_LEN - ids.length - 1;
    ids.push(...sp(stateStr).slice(0, budget), SEP);
    return { ids, pos };
  };

  const adapter = opts.device ? null : await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!opts.device && !adapter) throw new Error("WebGPU is not available in this browser");
  const device = opts.device ?? await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  const t1 = performance.now();
  const engine = await DeciderEngine.create(device, manifest, data, { maxTokens: opts.maxTokens ?? 4096, maxSeqs: opts.maxSeqs ?? 32, maxPos: 256 });
  const initMs = performance.now() - t1;

  // Raw logits for prepared sequences (for parity checks against run/pack.py's reference).
  const logitsFor = (seqs) => engine.score(seqs);

  function answer(s, z) {
    const T = manifest.temps[s.type] ?? 1;
    const v = Array.from(z, (x) => x / T), mx = Math.max(...v);
    const e = v.map((x) => Math.exp(x - mx)), sum = e.reduce((a, b) => a + b, 0);
    const p = e.map((x) => x / sum);
    const probabilities = Object.fromEntries(s.options.map(([o], k) => [o, p[k]]));
    const k = p.indexOf(Math.max(...p)), K = p.length;
    const ans = { type: s.type, label: s.options[k][0], confidence: (K * p[k] - 1) / (K - 1), probabilities };
    if (s.type === "score") ans.score = p.reduce((a, x, j) => a + j * x, 0);
    if (s.type === "noul") ans.probability_true = probabilities.true;
    return ans;
  }

  async function decide(state, questions) {
    const stateStr = typeof state === "string" ? state : pyJson(state);
    const names = Object.keys(questions);
    const seqs = names.map((n) => {
      const q = questions[n], options = optionsOf(q);
      return { name: n, type: q.type, options, ...build(q.type, q.instructions, options, stateStr) };
    });
    const out = {};
    // one engine pass for all questions whose tokens fit; split if a huge state overflows maxTokens
    const groups = [];
    let cur = [], tokens = 0;
    for (const s of seqs) {
      if (cur.length && tokens + s.ids.length > engine.maxTokens) { groups.push(cur); cur = []; tokens = 0; }
      cur.push(s); tokens += s.ids.length;
    }
    if (cur.length) groups.push(cur);
    for (const g of groups) {
      const z = await engine.score(g);
      g.forEach((s, i) => { out[s.name] = answer(s, z[i]); });
    }
    return out;
  }

  // Several (state, questions) pairs in as few engine passes as fit: e.g. one account question set plus
  // a noul per post. Returns one answers object per pair, in order.
  async function decideMany(pairs) {
    const seqs = [];
    pairs.forEach(({ state, questions }, pi) => {
      const stateStr = typeof state === "string" ? state : pyJson(state);
      for (const [n, q] of Object.entries(questions)) {
        const options = optionsOf(q);
        seqs.push({ pi, name: n, type: q.type, options, ...build(q.type, q.instructions, options, stateStr) });
      }
    });
    const outs = pairs.map(() => ({}));
    let i = 0;
    while (i < seqs.length) {
      const g = [];
      let tokens = 0, pos = 0;
      while (i < seqs.length && (!g.length || (tokens + seqs[i].ids.length <= engine.maxTokens && g.length < engine.maxSeqs
             && pos + seqs[i].pos.length <= engine.maxPos))) {
        g.push(seqs[i]); tokens += seqs[i].ids.length; pos += seqs[i].pos.length; i++;
      }
      const z = await engine.score(g);
      g.forEach((s, k) => { outs[s.pi][s.name] = answer(s, z[k]); });
    }
    return outs;
  }

  const countTokens = (t) => sp(t).length;
  return { decide, decideMany, countTokens, logitsFor, build, engine, manifest, info: { initMs, tokMs: t1 - t0, weightBytes: data.byteLength } };
}
