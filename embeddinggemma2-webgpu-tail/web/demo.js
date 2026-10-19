// EmbeddingGemma 2 -> onnxruntime-web (WebGPU, gpu-buffer outputs) -> RemexTail (2-bit codes, scan).
// Model-specific code is limited to prepareBatch / buildFeeds / the session setup. Everything after
// session.run (pooling fallback, colsum, mu, encode, gate, search, bench) is shared by both paths.
import { RemexTail } from './tail.js';
import { encodeRef, searchRef } from './tail_ref.js';
import { CORPUS, EXAMPLE_QUERIES, GATE_QUERY } from './corpus.js';

const D = 768;
const BATCH = 16;             // corpus embedding batch size
const GATE_DOCS = 32;         // startup gate: first 32 embeddings
const BENCH_DOCS = 32;        // bench batch size
const SCAN_N = 100000;        // synthetic scan corpus
const SCAN_CPU_N = 10000;     // rows the CPU reference scans (scaled to SCAN_N)
const SCAN_NQ = 20;
const FILL_CHUNK = 5000;
const TOPK = 10;
const RTOL = 1e-4;            // SPEC: 1e-4 relative, 5e-6 absolute floor
const ATOL = 5e-6;
const CODE_MISMATCH_MAX = 0.0005;
const SPREAD_MAX = 0.99;

const Q = new URLSearchParams(location.search);
const LOCAL = Q.get('local') === '1';
const MODEL = Q.get('model') || 'real';
const FORCE_POOL = Q.get('pool') === '1';
const BENCH_REPS = Number(Q.get('bench_reps') || 5);
const SCAN_REPS = Number(Q.get('scan_reps') || 3);

if (MODEL !== 'real' && MODEL !== 'dummy') throw new Error(`unknown model=${MODEL} (use real or dummy)`);
if (MODEL === 'dummy' && !LOCAL) throw new Error('model=dummy needs ?local=1 (it loads interop/dummy_eg2.onnx)');

// Local mode reaches the sibling interop/ install; hosted mode uses CDNs and Hugging Face only.
const CFG = LOCAL
  ? {
      ortModule: '../interop/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs',
      ortDist: '../interop/node_modules/onnxruntime-web/dist/',
      tokModule: '../interop/node_modules/@huggingface/tokenizers/dist/tokenizers.mjs',
      dummyModel: '../interop/dummy_eg2.onnx',
    }
  : {
      ortModule: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
      ortDist: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
      tokModule: 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs',
      dummyModel: null,
    };
// ?hf=<base> points the model files at a mirror with the same layout (run/run_demo.py --real serves one).
const HF = Q.get('hf') ? new URL(Q.get('hf'), location.href).href
  : 'https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/';
const REAL = {
  model: HF + 'onnx/model_q4.onnx',
  data: HF + 'onnx/model_q4.onnx_data',
  tokenizer: HF + 'tokenizer.json',
  tokenizerConfig: HF + 'tokenizer_config.json',
};

const abs = (p) => new URL(p, import.meta.url).href;
const $ = (id) => document.getElementById(id);
const logEl = $('log');
const log = (msg) => { logEl.textContent += '\n' + msg; };
const fail = (msg) => { throw new Error(msg); };

const state = window.demoState = {
  phase: 'loading',
  error: null,
  mode: { local: LOCAL, model: MODEL, forcePool: FORCE_POOL },
  inputs: null,
  outputs: null,
  pathUsed: null,
  gate: null,
  searches: [],
  bench: null,
};

const docPrompt = (t) => 'title: none | text: ' + t;
const queryPrompt = (t) => 'task: search result | query: ' + t;

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) fail(`fetch ${url}: HTTP ${r.status}`);
  return r.arrayBuffer();
}
async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) fail(`fetch ${url}: HTTP ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------- dummy stand-in tokenizer

function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const wordVecs = new Map();
function wordVec(w) {
  let v = wordVecs.get(w);
  if (!v) {
    const g = mulberry32(fnv1a(w));
    v = new Float32Array(D);
    for (let k = 0; k < D; k++) v[k] = g() * 2 - 1;
    wordVecs.set(w, v);
  }
  return v;
}
function hashWords(text) {
  const words = text.toLowerCase().match(/[a-z0-9]+/g);
  return words && words.length ? words : ['<empty>'];
}

// ---------------------------------------------------------------- model-specific: inputs

let ort, session, device, tail, params, R, tok = null, padId = 0;
let cpuParams;

function findPadId(tj, tc) {
  const want = typeof tc.pad_token === 'string' ? tc.pad_token : tc.pad_token?.content;
  const hit = (tj.added_tokens ?? []).find((t) => t.content === want);
  if (hit) return hit.id;
  return tj.padding?.pad_id ?? 0;    // GUESS: pad id 0 if the tokenizer names none
}

// texts -> one padded batch. Dummy: [B,S,768] hash-to-vector stand-in. Real: token ids.
function prepareBatch(texts) {
  const B = texts.length;
  if (MODEL === 'dummy') {
    const words = texts.map(hashWords);
    const S = Math.max(...words.map((w) => w.length));
    const x = new Float32Array(B * S * D);
    const maskF = new Float32Array(B * S);
    const maskU = new Uint32Array(B * S);
    for (let b = 0; b < B; b++) {
      words[b].forEach((w, s) => {
        x.set(wordVec(w), (b * S + s) * D);
        maskF[b * S + s] = 1;
        maskU[b * S + s] = 1;
      });
    }
    return { B, S, x, maskF, maskU };
  }
  const enc = texts.map((t) => tok.encode(t).ids);
  const S = Math.max(...enc.map((ids) => ids.length));
  const ids = new BigInt64Array(B * S).fill(BigInt(padId));
  const attn = new BigInt64Array(B * S);
  const maskU = new Uint32Array(B * S);
  enc.forEach((row, b) => row.forEach((id, s) => {
    ids[b * S + s] = BigInt(id);
    attn[b * S + s] = 1n;
    maskU[b * S + s] = 1;
  }));
  return { B, S, ids, attn, maskU };
}

function buildFeeds(names, batch) {
  const feeds = {};
  const { B, S } = batch;
  for (const name of names) {
    if (MODEL === 'dummy') {
      if (name === 'x') feeds.x = new ort.Tensor('float32', batch.x, [B, S, D]);
      else if (name === 'mask') feeds.mask = new ort.Tensor('float32', batch.maskF, [B, S]);
      else fail(`dummy model input "${name}" has no feeder`);
    } else if (name === 'input_ids') {
      feeds.input_ids = new ort.Tensor('int64', batch.ids, [B, S]);
    } else if (name === 'attention_mask') {
      feeds.attention_mask = new ort.Tensor('int64', batch.attn, [B, S]);
    } else if (/^(image|video|audio)_features$/.test(name)) {
      // GUESS: the text-only path takes empty multimodal features of shape [0, 512]
      feeds[name] = new ort.Tensor('float32', new Float32Array(0), [0, 512]);
    } else {
      fail(`model input "${name}" has no feeder in this demo; add one before running this model`);
    }
  }
  return feeds;
}

// ---------------------------------------------------------------- shared: after the encoder

function gpuBuffer(typed) {
  const buf = device.createBuffer({
    size: Math.max(16, typed.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buf, 0, typed);
  return buf;
}

// Runs the session on one batch. Returns an embedding GPUBuffer [B,768] that stays valid until release().
async function embedBatch(texts) {
  const batch = prepareBatch(texts);
  const res = await session.run(buildFeeds(session.inputNames, batch));
  const sentence = res.sentence_embedding;
  const usePool = FORCE_POOL || !sentence;
  const e = { B: batch.B, tensors: Object.values(res), sentence, ownBuf: null, pooled: usePool };
  if (!usePool) {
    e.emb = sentence.gpuBuffer;
  } else {
    if (!res.last_hidden_state) fail('model has neither sentence_embedding nor last_hidden_state');
    const maskBuf = gpuBuffer(batch.maskU);
    e.emb = tail.pool(res.last_hidden_state.gpuBuffer, maskBuf, batch.B, batch.S);
    e.ownBuf = e.emb;
    maskBuf.destroy();      // deferred until the submitted pool work finishes
  }
  return e;
}

async function release(e) {
  await device.queue.onSubmittedWorkDone();
  for (const t of e.tensors) t.dispose();
  if (e.ownBuf) e.ownBuf.destroy();
}

const close = (got, want) => Math.abs(got - want) <= ATOL + RTOL * Math.abs(want);

function dot(a, ao, b, bo) {
  let s = 0;
  for (let k = 0; k < D; k++) s += a[ao + k] * b[bo + k];
  return s;
}

async function runGate(n, corpus, RAW, EMB, gateEmb) {
  const reasons = [];
  const G = Math.min(GATE_DOCS, n);
  const enc = encodeRef(RAW.subarray(0, G * D), G, cpuParams, { normalize: true });
  const gc = await tail.readCorpus(corpus, G);

  let mism = 0;
  for (let i = 0; i < G; i++) {
    for (let j = 0; j < D; j++) {
      const byte = gc.packed[i * 192 + (j >> 2)];
      const code = (byte >> (6 - 2 * (j & 3))) & 3;
      if (code !== enc.codes[i * D + j]) mism++;
    }
  }
  const allowed = Math.floor(CODE_MISMATCH_MAX * G * D);
  if (mism > allowed) reasons.push(`codes: ${mism} of ${G * D} coordinates differ from encodeRef (limit ${allowed})`);

  let sBad = 0, bBad = 0, sMax = 0, bMax = 0;
  for (let i = 0; i < G; i++) {
    const ds = Math.abs(gc.scale[i] - enc.scale[i]);
    const db = Math.abs(gc.bias[i] - enc.bias[i]);
    sMax = Math.max(sMax, ds); bMax = Math.max(bMax, db);
    if (!close(gc.scale[i], enc.scale[i])) sBad++;
    if (!close(gc.bias[i], enc.bias[i])) bBad++;
  }
  if (sBad) reasons.push(`scale: ${sBad} docs outside 1e-4 relative of encodeRef`);
  if (bBad) reasons.push(`bias: ${bBad} docs outside 1e-4 relative of encodeRef`);

  const sb = tail.search(gateEmb.emb, 1, corpus, { n: G, normalize: true });
  const gs = await tail.readScores(sb, 1, corpus, G);
  sb.destroy();
  const qRaw = new Float32Array(await tail.readBuffer(gateEmb.emb, D * 4));
  const cs = searchRef(qRaw, 1, enc, cpuParams, { normalize: true });
  let scBad = 0, scMax = 0;
  for (let i = 0; i < G; i++) {
    scMax = Math.max(scMax, Math.abs(gs[i] - cs[i]));
    if (!close(gs[i], cs[i])) scBad++;
  }
  if (scBad) reasons.push(`scores: ${scBad} of ${G} GPU scores outside 1e-4 of searchRef`);

  let sum = 0, cnt = 0;
  for (let i = 0; i < G; i++) for (let j = i + 1; j < G; j++) {
    sum += dot(EMB, i * D, EMB, j * D); cnt++;
  }
  const spread = cnt ? sum / cnt : 0;
  if (!(spread < SPREAD_MAX)) reasons.push(`spread: mean off-diagonal cosine ${spread.toFixed(4)} >= ${SPREAD_MAX} (backend suspect)`);

  return {
    passed: reasons.length === 0,
    reasons,
    docs: G,
    codes: { mismatch: mism, allowed, total: G * D },
    scale_max_abs_err: sMax,
    bias_max_abs_err: bMax,
    score: { query: GATE_QUERY, n: G, bad: scBad, max_abs_err: scMax },
    spread_mean_offdiag_cos: spread,
  };
}

// ---------------------------------------------------------------- search

async function search(text) {
  const q = await embedBatch([queryPrompt(text)]);
  const top = await tail.topk(q.emb, 1, state.corpus, state.n, TOPK, { normalize: true });
  const qRaw = new Float32Array(await tail.readBuffer(q.emb, D * 4));
  await release(q);

  const qn = normalizeRow(qRaw, 0);
  const cos = new Float32Array(state.n);
  for (let i = 0; i < state.n; i++) cos[i] = dot(state.EMB, i * D, qn, 0);
  const exact = Array.from({ length: state.n }, (_, i) => i)
    .sort((a, b) => (cos[b] - cos[a]) || (a - b))
    .slice(0, TOPK);
  const gpuIdx = Array.from(top[0].idx);
  const overlap = gpuIdx.filter((i) => exact.includes(i)).length;
  const result = {
    query: text,
    recall_at_10: overlap / TOPK,
    hits: gpuIdx.map((i, r) => ({ rank: r + 1, doc: CORPUS[i], index: i, gpu: top[0].score[r], exact: cos[i] })),
  };
  state.searches.push(result);
  return result;
}

function normalizeRow(src, off) {
  let ss = 0;
  for (let k = 0; k < D; k++) ss += src[off + k] * src[off + k];
  const inv = 1 / Math.max(Math.sqrt(ss), 1e-8);
  const out = new Float32Array(D);
  for (let k = 0; k < D; k++) out[k] = src[off + k] * inv;
  return out;
}

// ---------------------------------------------------------------- bench

const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

async function scanBench(bench) {
  const head = new Float32Array(SCAN_CPU_N * D);
  const syn = tail.allocCorpus(SCAN_N);
  const rng = mulberry32(12345);
  const tFill = performance.now();
  for (let s = 0; s < SCAN_N; s += FILL_CHUNK) {
    const chunk = new Float32Array(FILL_CHUNK * D);
    for (let i = 0; i < chunk.length; i++) chunk[i] = rng() * 2 - 1;
    if (s < SCAN_CPU_N) head.set(chunk.subarray(0, Math.min(FILL_CHUNK, SCAN_CPU_N - s) * D), s * D);
    const buf = gpuBuffer(chunk);
    tail.encode(buf, FILL_CHUNK, syn, s, { normalize: true });
    await device.queue.onSubmittedWorkDone();
    buf.destroy();
    log(`scan corpus: ${s + FILL_CHUNK} / ${SCAN_N} rows encoded`);
  }
  bench.scan_fill_ms = +(performance.now() - tFill).toFixed(1);

  // queries: rows 0, 97, 194, ... of the synthetic corpus (all inside the CPU-checked head)
  const qHost = new Float32Array(SCAN_NQ * D);
  for (let q = 0; q < SCAN_NQ; q++) qHost.set(head.subarray((q * 97) * D, (q * 97 + 1) * D), q * D);
  const qbuf = gpuBuffer(qHost);
  const cpuEnc = encodeRef(head, SCAN_CPU_N, cpuParams, { normalize: true });

  // One-time agreement check on the first SCAN_CPU_N rows. The scan is checked against searchRef on
  // the GPU's own codes, scale and bias (as check_tail.py does), so encode differences cannot leak
  // into it; the float32-GPU vs float64-CPU encode differences are reported separately.
  const sCheck = tail.search(qbuf, SCAN_NQ, syn, { n: SCAN_CPU_N, normalize: true });
  const gChk = await tail.readScores(sCheck, SCAN_NQ, syn, SCAN_CPU_N);
  sCheck.destroy();
  const gEnc = await tail.readCorpus(syn, SCAN_CPU_N);
  const gCodes = new Uint8Array(SCAN_CPU_N * D);
  for (let i = 0; i < SCAN_CPU_N * D; i++) gCodes[i] = (gEnc.packed[i >> 2] >> (6 - 2 * (i & 3))) & 3;
  const flipRow = new Uint8Array(SCAN_CPU_N);
  let flips = 0;
  for (let i = 0; i < SCAN_CPU_N * D; i++) {
    if (gCodes[i] !== cpuEnc.codes[i]) { flips++; flipRow[(i / D) | 0] = 1; }
  }
  const cOwn = searchRef(qHost, SCAN_NQ, { codes: gCodes, scale: gEnc.scale, bias: gEnc.bias }, cpuParams, { normalize: true });
  const cCpu = searchRef(qHost, SCAN_NQ, cpuEnc, cpuParams, { normalize: true });
  let maxErr = 0, maxFlip = 0, maxClean = 0;
  for (let k = 0; k < gChk.length; k++) {
    maxErr = Math.max(maxErr, Math.abs(gChk[k] - cOwn[k]));
    const e = Math.abs(gChk[k] - cCpu[k]);
    if (flipRow[k % SCAN_CPU_N]) maxFlip = Math.max(maxFlip, e); else maxClean = Math.max(maxClean, e);
  }
  bench.scan_agreement_max_abs_err = maxErr;
  bench.scan_encode_vs_cpu = {
    code_flips: flips, codes: SCAN_CPU_N * D,
    rows_with_flip: flipRow.reduce((a, b) => a + b, 0), rows: SCAN_CPU_N,
    score_max_abs_diff_rows_with_flip: maxFlip,
    score_max_abs_diff_rows_without_flip: maxClean,
  };

  const gpuT = [];
  for (let r = 0; r <= SCAN_REPS; r++) {          // first pass is a warm-up
    const t0 = performance.now();
    const sb = tail.search(qbuf, SCAN_NQ, syn, { normalize: true });
    await device.queue.onSubmittedWorkDone();
    const dt = performance.now() - t0;
    sb.destroy();
    if (r > 0) gpuT.push(dt);
    log(`scan gpu rep ${r}: ${dt.toFixed(1)} ms`);
  }
  const cpuT = [];
  for (let r = 0; r <= SCAN_REPS; r++) {
    const t0 = performance.now();
    searchRef(qHost, SCAN_NQ, cpuEnc, cpuParams, { normalize: true });
    const dt = performance.now() - t0;
    if (r > 0) cpuT.push(dt);
    log(`scan cpu rep ${r}: ${dt.toFixed(1)} ms (${SCAN_CPU_N} rows)`);
  }
  const gMed = median(gpuT), cMed = median(cpuT);
  bench.scan = {
    n: SCAN_N, nq: SCAN_NQ, reps: SCAN_REPS,
    gpu_ms_median: +gMed.toFixed(2),
    cpu_ms_median_10k: +cMed.toFixed(2),
    cpu_ms_scaled_100k: +(cMed * SCAN_N / SCAN_CPU_N).toFixed(2),
    gpu_over_cpu_scaled_speedup: +((cMed * SCAN_N / SCAN_CPU_N) / gMed).toFixed(2),
    gpu_ms_all: gpuT.map((x) => +x.toFixed(2)),
    cpu_ms_all_10k: cpuT.map((x) => +x.toFixed(2)),
  };
  qbuf.destroy();
  syn.words.destroy();
  syn.aux.destroy();
}

async function runBench(corpus) {
  const bench = {
    docs_per_batch: BENCH_DOCS, reps: BENCH_REPS,
    path: state.pathUsed,
    embed_only_ms: null, embed_readback_encodeRef_ms: null, embed_gpu_encode_ms: null,
  };
  const texts = CORPUS.slice(0, BENCH_DOCS).map(docPrompt);
  const scratch = tail.allocCorpus(BENCH_DOCS);
  const ta = [], tb = [], tc = [];
  for (let r = 0; r <= BENCH_REPS; r++) {          // rep 0 is a warm-up
    let t0 = performance.now();
    let e = await embedBatch(texts);
    await device.queue.onSubmittedWorkDone();
    const a = performance.now() - t0;
    await release(e);

    t0 = performance.now();
    e = await embedBatch(texts);
    const data = await e.sentence.getData(false);
    encodeRef(data, BENCH_DOCS, cpuParams, { normalize: true });
    const b = performance.now() - t0;
    await release(e);

    t0 = performance.now();
    e = await embedBatch(texts);
    tail.encode(e.emb, BENCH_DOCS, scratch, 0, { normalize: true });
    await device.queue.onSubmittedWorkDone();
    const c = performance.now() - t0;
    await release(e);

    if (r > 0) { ta.push(a); tb.push(b); tc.push(c); }
    log(`bench rep ${r}: embed ${a.toFixed(1)} | +readback+CPU ${b.toFixed(1)} | +GPU encode ${c.toFixed(1)} ms`);
  }
  bench.embed_only_ms = { median: +median(ta).toFixed(2), runs: ta.map((x) => +x.toFixed(2)) };
  bench.embed_readback_encodeRef_ms = { median: +median(tb).toFixed(2), runs: tb.map((x) => +x.toFixed(2)) };
  bench.embed_gpu_encode_ms = { median: +median(tc).toFixed(2), runs: tc.map((x) => +x.toFixed(2)) };
  scratch.words.destroy();
  scratch.aux.destroy();

  await scanBench(bench);
  return bench;
}

// ---------------------------------------------------------------- startup

async function startup() {
  log(`mode: ${LOCAL ? 'local' : 'hosted'}, model: ${MODEL}${FORCE_POOL ? ', pool path forced' : ''}`);
  params = await fetchJson(abs('assets/params.json'));
  const rb = await fetchBytes(abs('assets/R.f32.txt'));
  R = new Float32Array(rb);
  if (R.length !== D * D) fail(`R.f32.txt has ${R.length} floats, expected ${D * D}`);
  log(`params: d=${params.d} bits=${params.bits} seed=${params.seed} rotation=${params.rotation}`);

  ort = await import(abs(CFG.ortModule));
  ort.env.wasm.wasmPaths = abs(CFG.ortDist);
  ort.env.wasm.numThreads = 1;
  log(`onnxruntime-web ${ort.env?.versions?.web ?? '1.30.0'} loaded`);

  if (MODEL === 'real') {
    const { Tokenizer } = await import(abs(CFG.tokModule));
    const tj = await fetchJson(REAL.tokenizer);
    const tc = await fetchJson(REAL.tokenizerConfig);
    tok = new Tokenizer(tj, tc);
    padId = findPadId(tj, tc);
    log(`tokenizer ready, pad id ${padId}`);
  } else {
    log('tokenizer: hash-to-vector stand-in (dummy model, no real tokenizer)');
  }

  const modelUrl = MODEL === 'real' ? REAL.model : abs(CFG.dummyModel);
  const opts = {
    executionProviders: ['webgpu'],
    preferredOutputLocation: { sentence_embedding: 'gpu-buffer', last_hidden_state: 'gpu-buffer' },
    // The real graph logs node-placement warnings through console.error. They come from the native
    // logger, which ort.env.logLevel does not reach; 3 = errors only.
    logSeverityLevel: 3,
  };
  if (MODEL === 'real') opts.externalData = [{ path: 'model_q4.onnx_data', data: REAL.data }];
  log(`loading session: ${modelUrl}`);
  session = await ort.InferenceSession.create(modelUrl, opts);
  state.inputs = session.inputNames;
  state.outputs = session.outputNames;
  log(`session inputs: ${session.inputNames.join(', ')}`);
  log(`session outputs: ${session.outputNames.join(', ')}`);
  for (const name of session.inputNames) {
    const known = MODEL === 'dummy'
      ? ['x', 'mask'].includes(name)
      : ['input_ids', 'attention_mask'].includes(name) || /^(image|video|audio)_features$/.test(name);
    if (!known) fail(`model input "${name}" has no feeder in this demo`);
  }

  device = await ort.env.webgpu.device;
  tail = await RemexTail.create(device, {
    R, mu: new Float32Array(D), boundaries: params.boundaries, centroids: params.centroids, kappa: params.kappa,
  });
  log('RemexTail created on onnxruntime-web\'s GPUDevice');

  // ---- embed the corpus; keep every batch's GPU embedding alive
  const docTexts = CORPUS.map(docPrompt);
  const n = docTexts.length;
  const batches = [];
  for (let i = 0; i < n; i += BATCH) {
    batches.push(await embedBatch(docTexts.slice(i, i + BATCH)));
  }
  state.pathUsed = batches[0].pooled ? 'last_hidden_state + tail.pool' : 'sentence_embedding';
  log(`embedded ${n} documents in ${batches.length} batches via ${state.pathUsed}`);

  // ---- mu: colsum every batch, read back only the 768-float sum
  const acc = device.createBuffer({
    size: D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(acc, 0, new Float32Array(D));
  for (const b of batches) tail.colsum(b.emb, b.B, acc);
  const sum = new Float32Array(await tail.readBuffer(acc, D * 4));
  acc.destroy();
  const mu = sum.map((v) => v / n);
  tail.setMean(mu);
  cpuParams = {
    R, Rmu: tail.rmu_host, mu: tail.mu_host,
    boundaries: params.boundaries, centroids: params.centroids, kappa: params.kappa,
  };
  log(`mu set from ${n} documents (colsum readback: 768 floats)`);

  // ---- encode every batch into the corpus
  const corpus = tail.allocCorpus(n);
  let start = 0;
  for (const b of batches) {
    tail.encode(b.emb, b.B, corpus, start, { normalize: true });
    start += b.B;
  }
  await device.queue.onSubmittedWorkDone();
  log(`encoded ${n} documents into the corpus on the GPU`);

  // ---- read back embeddings (after encoding): for the gate and exact cosine on this small corpus
  const RAW = new Float32Array(n * D);
  let off = 0;
  for (const b of batches) {
    RAW.set(new Float32Array(await tail.readBuffer(b.emb, b.B * D * 4)), off * D);
    off += b.B;
  }
  const EMB = new Float32Array(n * D);
  for (let i = 0; i < n; i++) EMB.set(normalizeRow(RAW, i * D), i * D);

  const gateEmb = await embedBatch([queryPrompt(GATE_QUERY)]);
  const gate = await runGate(n, corpus, RAW, EMB, gateEmb);
  await release(gateEmb);
  state.gate = gate;
  log(`startup gate: ${gate.passed ? 'PASSED' : 'FAILED'}`);
  log(`  codes: ${gate.codes.mismatch} / ${gate.codes.total} differ (limit ${gate.codes.allowed})`);
  log(`  scale max abs err ${gate.scale_max_abs_err.toExponential(2)}, bias max abs err ${gate.bias_max_abs_err.toExponential(2)}`);
  log(`  scores vs searchRef: ${gate.score.bad} of ${gate.score.n} outside tolerance, max abs err ${gate.score.max_abs_err.toExponential(2)}`);
  log(`  spread: mean off-diagonal cosine ${gate.spread_mean_offdiag_cos.toFixed(4)} (must be < ${SPREAD_MAX})`);
  for (const r of gate.reasons) log(`  FAIL: ${r}`);

  state.corpus = corpus;
  state.n = n;
  state.EMB = EMB;
  state.batchesOwned = batches;
  state.gateOk = gate.passed;
  return { corpus, n, RAW, EMB, batches };
}

function renderHits(res) {
  const box = document.createElement('div');
  box.className = 'hits';
  const h = document.createElement('p');
  h.textContent = `"${res.query}": recall@10 of GPU ranking vs fp32 cosine = ${(res.recall_at_10 * 100).toFixed(0)}%`;
  box.appendChild(h);
  const table = document.createElement('table');
  table.innerHTML = '<thead><tr><th>#</th><th>document</th><th>GPU score</th><th>fp32 cosine</th></tr></thead>';
  const tb = document.createElement('tbody');
  for (const hit of res.hits) {
    const tr = document.createElement('tr');
    const cells = [String(hit.rank), hit.doc, hit.gpu.toFixed(4), hit.exact.toFixed(4)];
    for (const c of cells) {
      const td = document.createElement('td');
      td.textContent = c;
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
  table.appendChild(tb);
  box.appendChild(table);
  return box;
}

function renderBench(b) {
  const box = $('bench-out');
  box.textContent = '';
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(b, null, 2);
  box.appendChild(pre);
}

function setBusy(on) {
  $('search-btn').disabled = on || !state.gateOk;
  $('bench-btn').disabled = on || !state.gateOk;
}

async function main() {
  try {
    const { corpus, n } = await startup();
    const gate = state.gate;
    const gatePanel = $('gate');
    gatePanel.className = gate.passed ? 'gate ok' : 'gate bad';
    gatePanel.textContent = gate.passed
      ? `Startup gate passed (${gate.docs} docs, ${gate.codes.mismatch}/${gate.codes.total} code mismatches, ${gate.score.bad} score mismatches).`
      : `Startup gate FAILED, so results are hidden: ${gate.reasons.join('; ')}`;

    if (gate.passed) {
      const ex = $('examples');
      for (const q of EXAMPLE_QUERIES) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'chip';
        btn.textContent = q;
        btn.addEventListener('click', () => { $('q').value = q; runSearch(q); });
        ex.appendChild(btn);
      }
      $('search-btn').disabled = false;
      $('bench-btn').disabled = false;
    }
    state.phase = 'ready';
    log(`ready: ${n} documents indexed`);
    void corpus;
  } catch (err) {
    state.phase = 'error';
    state.error = String(err?.stack ?? err);
    log(`ERROR: ${err?.message ?? err}`);
    console.error(err);
  }
}

async function runSearch(text) {
  const q = text.trim();
  if (!q || !state.gateOk) return;
  setBusy(true);
  try {
    const res = await search(q);
    const out = $('results');
    out.prepend(renderHits(res));
  } catch (err) {
    log(`search ERROR: ${err?.message ?? err}`);
    state.error = String(err?.stack ?? err);
  } finally {
    setBusy(false);
  }
}

$('search-form').addEventListener('submit', (ev) => {
  ev.preventDefault();
  runSearch($('q').value);
});

$('bench-btn').addEventListener('click', async () => {
  setBusy(true);
  $('bench-out').textContent = 'running...';
  try {
    state.bench = await runBench(state.corpus);
    renderBench(state.bench);
  } catch (err) {
    state.error = String(err?.stack ?? err);
    $('bench-out').textContent = `bench ERROR: ${err?.message ?? err}`;
  } finally {
    setBusy(false);
  }
});

// Test hook for run/run_demo.py: texts -> unit rows (plain arrays), through the same session path as the corpus.
window.demoEmbed = async (texts) => {
  const e = await embedBatch(texts);
  const raw = new Float32Array(await tail.readBuffer(e.emb, e.B * D * 4));
  await release(e);
  return Array.from({ length: e.B }, (_, i) => Array.from(normalizeRow(raw, i * D)));
};

window.addEventListener('error', (e) => { state.error = String(e.message); });
window.addEventListener('unhandledrejection', (e) => { state.error = String(e.reason?.message ?? e.reason); });

main();
