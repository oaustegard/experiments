// monad-weight-edit bridge job: Monad (original or edited) on WebGPU through onnxruntime-web.
// The body of async function (ctx). The loop is ort_gen.py's, line for line: right-padded batches,
// GroupQueryAttention reads each row's length from the mask, greedy, stop on EOS or "<|im_end|>".
//
// inputs: graph (u8 model.onnx), p0, p1, ... (u8 parts of model.packed), meta (u8 JSON from
// export.py), pieces (u8 JSON, the decoded text of every token id), cfg (json):
//   {mode: "check" | "bench" | "gen", ort: base URL, budgetMs, ...mode fields}
//   check: {prompts: [[ids]], steps}          -> logits of every step, tokens
//   bench: {prompt: [ids], steps, runs, batches: [1, 8]}  -> decode ms per step, prefill ms
//   gen:   {batches: [{uids, prompts, maxNew}]} -> tokens, lengths, stopped; per-batch timings
const cfg = ctx.inputs.cfg;
let ORT = cfg.ort || "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
if (ORT.startsWith("/")) ORT = new URL(ORT, new URL(self.location.href).origin).href;   // local SwiftShader runs
const t00 = performance.now();
const td = new TextDecoder();
const meta = JSON.parse(td.decode(ctx.inputs.meta));
const PAD = 3, EOS = 2, END = "<|im_end|>";

let ort = ctx.cache.get("ort:" + ORT);
if (!ort) {
  ort = await ctx.importModule(ORT + "ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  ctx.cache.set("ort:" + ORT, ort);
}

const key = "monad:" + meta.graph_sha256 + ":" + meta.data_sha256 + ":" + (cfg.ep || "webgpu") + ":" + (cfg.opt || "all") + ":" + (cfg.cacheTag || "");
let M = ctx.cache.get(key);
const load = {};
if (!M) {
  if (!ctx.inputs.graph) throw new Error("NO_CACHE: this tab has no session for " + key + "; resubmit with weights");
  const tl = performance.now();
  const parts = Object.keys(ctx.inputs).filter((k) => /^p\d+$/.test(k)).sort((a, b) => +a.slice(1) - +b.slice(1)).map((k) => ctx.inputs[k]);
  const packed = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  { let o = 0; for (const p of parts) { packed.set(p, o); o += p.length; } }
  if (packed.length !== meta.packed_bytes) throw new Error(`packed ${packed.length} bytes, meta says ${meta.packed_bytes}`);
  const data = new Uint8Array(meta.data_bytes), d32 = new Uint32Array(data.buffer);
  for (const [off, ln, kind, pos] of meta.regions) {
    if (kind === "f32") { data.set(packed.subarray(pos, pos + ln), off); continue; }
    if (off % 4 || pos % 2) throw new Error("misaligned region " + off);
    const src = new Uint16Array(packed.buffer, pos, ln / 4), base = off / 4;
    for (let i = 0; i < src.length; i++) d32[base + i] = src[i] << 16;
  }
  const sha = [...new Uint8Array(await crypto.subtle.digest("SHA-256", data))].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (sha !== meta.data_sha256) throw new Error(`rebuilt data sha ${sha} != ${meta.data_sha256}`);
  load.unpackMs = performance.now() - tl;
  const names = [];
  for (let l = 0; l < meta.layers; l++) names.push(`present.${l}.key`, `present.${l}.value`);
  const pol = Object.fromEntries(names.map((n) => [n, (cfg.ep || "webgpu") === "webgpu" ? "gpu-buffer" : "cpu"]));
  pol.logits = "cpu";
  const ts = performance.now();
  const sess = await ort.InferenceSession.create(ctx.inputs.graph, {
    executionProviders: [cfg.ep || "webgpu"], externalData: [{ path: "model.onnx.data", data }],
    preferredOutputLocation: pol, graphOptimizationLevel: cfg.opt || "all",
  });
  load.sessionMs = performance.now() - ts;
  M = { sess, layers: meta.layers };
  ctx.cache.set(key, M);
  load.cached = false;
} else load.cached = true;

const pieces = ctx.cache.get("pieces") || JSON.parse(td.decode(ctx.inputs.pieces));
ctx.cache.set("pieces", pieces);

// Optional sampling (cfg.sample = {temperature, topP, penalty, window, seed}); greedy without it.
// Same arithmetic as the chat page's makeSampler (austegard.com/ai-tools/monad-chat/monad.js), with a
// seeded RNG (mulberry32) so a run can be repeated.
const SAMPLE = (() => {
  const c = cfg.sample;
  if (!c) return null;
  let st = (c.seed >>> 0) || 1;
  const rand = () => { st |= 0; st = (st + 0x6D2B79F5) | 0; let t = Math.imul(st ^ (st >>> 15), 1 | st);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return (logits, gen) => {
    const V = logits.length, l = Float32Array.from(logits);
    if (c.penalty !== 1 && c.window > 0 && gen.length) {
      for (const id of new Set(gen.slice(-c.window))) l[id] = l[id] > 0 ? l[id] / c.penalty : l[id] * c.penalty;
    }
    let best = 0;
    for (let i = 1; i < V; i++) if (l[i] > l[best]) best = i;
    if (!(c.temperature > 0)) return best;
    const mx = l[best], inv = 1 / c.temperature, p = new Float32Array(V);
    let z = 0;
    for (let i = 0; i < V; i++) { p[i] = Math.exp((l[i] - mx) * inv); z += p[i]; }
    // Sort only the tokens within a factor 1e-3 of the top one; widen to everything when they
    // hold less than top-p of the mass (a flat distribution). Same result, much less sorting.
    let cand = [], cm = 0;
    for (let i = 0; i < V; i++) if (p[i] > 1e-3) { cand.push(i); cm += p[i]; }
    if (cm / z < c.topP) { cand = []; for (let i = 0; i < V; i++) if (p[i] > 1e-9) cand.push(i); }
    cand.sort((a, b) => p[b] - p[a]);
    let keep = 0, mass = 0;
    while (keep < cand.length) { mass += p[cand[keep]] / z; keep++; if (mass >= c.topP) break; }
    let r = rand() * mass * z;
    for (let k = 0; k < keep; k++) { r -= p[cand[k]]; if (r <= 0) return cand[k]; }
    return cand[keep - 1];
  };
})();

const i64 = (arr, dims) => new ort.Tensor("int64", BigInt64Array.from(arr, (x) => BigInt(x)), dims);

// One batched greedy generation. opts: {maxNew, stop (bool), keepLogits (steps to keep)}
async function generate(prompts, { maxNew, stop = true, keepLogits = 0, onStep } = {}) {
  const B = prompts.length, L = M.layers;
  const lens = prompts.map((p) => p.length), T = Math.max(...lens);
  const ids = new Array(B * T).fill(PAD), mask = new Array(B * T).fill(0);
  prompts.forEach((p, b) => p.forEach((t, j) => { ids[b * T + j] = t; mask[b * T + j] = 1; }));
  let feeds = { input_ids: i64(ids, [B, T]), attention_mask: i64(mask, [B, T]), logits_index: i64(lens.map((n, b) => b * T + n - 1), [B]) };
  for (let l = 0; l < L; l++) {
    feeds[`past_key_values.${l}.key`] = new ort.Tensor("float32", new Float32Array(0), [B, meta.kv_heads || 4, 0, meta.head_dim || 64]);
    feeds[`past_key_values.${l}.value`] = new ort.Tensor("float32", new Float32Array(0), [B, meta.kv_heads || 4, 0, meta.head_dim || 64]);
  }
  const out = prompts.map(() => []), tails = prompts.map(() => ""), done = new Array(B).fill(false), stopped = new Array(B).fill(false);
  const cur = lens.slice(); let total = T;
  const stepMs = [], kept = [];
  let prev = null;
  for (let step = 0; step <= maxNew; step++) {
    const t0 = performance.now();
    const res = await M.sess.run(feeds);
    stepMs.push(performance.now() - t0);
    if (prev) for (const t of prev) if (t.location === "gpu-buffer") t.dispose();
    const logits = res.logits.data;   // [B, 1, V]
    const V = logits.length / B;
    if (step < keepLogits) kept.push(logits.slice());
    const nxt = new Array(B);
    for (let b = 0; b < B; b++) {
      if (SAMPLE) { nxt[b] = SAMPLE(logits.subarray(b * V, (b + 1) * V), out[b]); continue; }
      let bi = 0, bv = -Infinity;
      for (let v = 0, o = b * V; v < V; v++) if (logits[o + v] > bv) { bv = logits[o + v]; bi = v; }
      nxt[b] = bi;
    }
    prev = [];
    for (let l = 0; l < L; l++) prev.push(res[`present.${l}.key`], res[`present.${l}.value`]);
    if (step === maxNew) break;
    for (let b = 0; b < B; b++) {
      if (done[b]) { nxt[b] = PAD; continue; }
      const t = nxt[b];
      if (stop && t === EOS) { done[b] = stopped[b] = true; continue; }
      out[b].push(t);
      if (stop) { tails[b] = (tails[b] + pieces[t]).slice(-32); if (tails[b].includes(END)) done[b] = stopped[b] = true; }
    }
    if (done.every(Boolean)) break;
    total += 1;
    for (let b = 0; b < B; b++) cur[b] += 1;
    const m = new Array(B * total);
    for (let b = 0; b < B; b++) for (let j = 0; j < total; j++) m[b * total + j] = j < cur[b] ? 1 : 0;
    feeds = { input_ids: i64(nxt, [B, 1]), attention_mask: i64(m, [B, total]), logits_index: i64([...Array(B).keys()], [B]) };
    for (let l = 0; l < L; l++) { feeds[`past_key_values.${l}.key`] = prev[2 * l]; feeds[`past_key_values.${l}.value`] = prev[2 * l + 1]; }
    if (onStep) onStep(step);
  }
  if (prev) for (const t of prev) if (t.location === "gpu-buffer") t.dispose();
  return { tokens: out, stopped, stepMs, kept };
}

const med = (a) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const pct = (a, q) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : null; };
const result = { mode: cfg.mode, load, layers: M.layers, ort: ORT };

if (cfg.mode === "check") {
  const r = await generate(cfg.prompts, { maxNew: cfg.steps, stop: false, keepLogits: cfg.steps + 1 });
  const V = r.kept[0].length / cfg.prompts.length;
  const flat = new Float32Array(r.kept.length * r.kept[0].length);
  r.kept.forEach((x, i) => flat.set(x, i * x.length));
  result.logits = flat; result.shape = [r.kept.length, cfg.prompts.length, V];
  result.tokens = r.tokens; result.stepMs = r.stepMs;
} else if (cfg.mode === "bench") {
  result.bench = [];
  for (const B of cfg.batches || [1]) {
    const runs = [];
    for (let k = 0; k < (cfg.runs || 3); k++) {
      const r = await generate(Array(B).fill(cfg.prompt), { maxNew: cfg.steps, stop: false });
      const dec = r.stepMs.slice(1);
      runs.push({ prefillMs: r.stepMs[0], decodeMedMs: med(dec), decodeP90Ms: pct(dec, 0.9), decodeMeanMs: dec.reduce((s, x) => s + x, 0) / dec.length, steps: dec.length });
      ctx.progress(null, `bench B=${B} run ${k + 1}`);
    }
    result.bench.push({ B, promptLen: cfg.prompt.length, runs });
  }
} else if (cfg.mode === "gen") {
  const budget = cfg.budgetMs || 480000;
  const toks = [], lens = [], stopped = [], uids = [], batches = [];
  let done = 0;
  const totalPrompts = cfg.batches.reduce((s, b) => s + b.prompts.length, 0);
  for (const [bi, bt] of cfg.batches.entries()) {
    const el = performance.now() - t00;
    const est = batches.length ? Math.max(...batches.map((x) => x.ms)) : 0;
    if (bi > 0 && el + est > budget) break;
    const t0 = performance.now();
    const r = await generate(bt.prompts, { maxNew: bt.maxNew, onStep: (s) => ctx.progress((done + s / bt.maxNew * bt.prompts.length) / totalPrompts, `batch ${bi + 1}/${cfg.batches.length} step ${s}`) });
    const ms = performance.now() - t0;
    const nTok = r.tokens.reduce((s, x) => s + x.length, 0);
    batches.push({ i: bi, ms, steps: r.stepMs.length - 1, prefillMs: r.stepMs[0], stepMedMs: med(r.stepMs.slice(1)), stepP90Ms: pct(r.stepMs.slice(1), 0.9), tokens: nTok, B: bt.prompts.length });
    r.tokens.forEach((t, b) => { toks.push(...t); lens.push(t.length); stopped.push(r.stopped[b] ? 1 : 0); uids.push(bt.uids[b]); });
    done += bt.prompts.length;
  }
  result.tokens = Int32Array.from(toks); result.lengths = Int32Array.from(lens); result.stopped = Uint8Array.from(stopped);
  result.uids = uids; result.batches = batches; result.batchesDone = batches.length; result.batchesTotal = cfg.batches.length;
} else if (cfg.mode === "sharedkv") {
  // Greedy decode of one prompt two ways: the growing cache (generate above) and a fixed-size cache bound as
  // both past input and present output (the chat page's generateShared). Same tokens expected.
  const T = cfg.prompt.length, N = cfg.steps, L = M.layers, KVH = meta.kv_heads || 4, HD = meta.head_dim || 64;
  const g = await generate([cfg.prompt], { maxNew: N, stop: false });
  const MAX = Math.ceil((T + N + 1) / 128) * 128, dev = ort.env.webgpu.device, bufs = [], tens = [];
  for (let i = 0; i < 2 * L; i++) {
    const b = dev.createBuffer({ size: 4 * KVH * MAX * HD, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    bufs.push(b); tens.push(ort.Tensor.fromGpuBuffer(b, { dataType: "float32", dims: [1, KVH, MAX, HD] }));
  }
  const kvF = {}, kvO = {};
  for (let l = 0; l < L; l++) { kvF[`past_key_values.${l}.key`] = tens[2 * l]; kvF[`past_key_values.${l}.value`] = tens[2 * l + 1];
    kvO[`present.${l}.key`] = tens[2 * l]; kvO[`present.${l}.value`] = tens[2 * l + 1]; }
  const out = [], ms = [];
  let ids = cfg.prompt, total = T, err = null;
  try {
    for (let step = 0; step <= N; step++) {
      const t0 = performance.now();
      const res = await M.sess.run({ input_ids: i64(ids, [1, ids.length]), attention_mask: i64(new Array(total).fill(1), [1, total]),
        logits_index: i64([ids.length - 1], [1]), ...kvF }, { logits: null, ...kvO });
      ms.push(performance.now() - t0);
      const lg = res.logits.data; let bi = 0; for (let v = 1; v < lg.length; v++) if (lg[v] > lg[bi]) bi = v;
      if (step === N) break;
      out.push(bi); ids = [bi]; total += 1;
      if (step % 100 === 0) ctx.progress(step / N, `shared step ${step}`);
    }
  } catch (e) { err = String(e && e.message || e); }
  for (const b of bufs) b.destroy();
  let same = 0; while (same < Math.min(out.length, g.tokens[0].length) && out[same] === g.tokens[0][same]) same++;
  Object.assign(result, { growTokens: g.tokens[0], sharedTokens: out, sameTokens: same, growStepMedMs: med(g.stepMs.slice(1)),
    sharedStepMedMs: med(ms.slice(1)), growPrefillMs: g.stepMs[0], sharedPrefillMs: ms[0], kvMax: MAX, sharedError: err });
}

result.totalMs = performance.now() - t00;
return result;
