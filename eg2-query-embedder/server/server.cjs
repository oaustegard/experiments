// EmbeddingGemma 2 query embedder: one HTTP endpoint around model_q4.onnx on onnxruntime's CPU
// provider. Same graph, tokenizer and prompt strings as the browser path in
// embeddinggemma2-webgpu-tail/web/search.js, so a vector from here lands in the same space as
// corpus vectors the browser computed.
//
//   GET  /health -> {ready, timings_ms}
//   POST /embed  {text, prompt?: "query", dims?: 768|512|256|128} -> {embedding: number[], dims, tokens, ms}
//
// Rate limiting, CORS and caching belong to the Worker in front of this; the server only bounds
// its own input (MAX_CHARS, MAX_TOKENS) so a direct caller cannot make one request expensive.
const { createServer } = require('node:http');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const ort = require('onnxruntime-node');
const { Tokenizer } = require('@huggingface/tokenizers');

const T0 = performance.now();
const MODEL_DIR = process.env.MODEL_DIR || join(__dirname, 'model');
const PORT = Number(process.env.PORT || 8080);
const MAX_CHARS = Number(process.env.MAX_CHARS || 1024);
const MAX_TOKENS = Number(process.env.MAX_TOKENS || 256);
const THREADS = Number(process.env.THREADS || 0);           // 0 = onnxruntime's default
const DIMS = new Set([768, 512, 256, 128]);                  // EG2's Matryoshka sizes
const PROMPTS = JSON.parse(readFileSync(join(MODEL_DIR, 'config_sentence_transformers.json'), 'utf8')).prompts;

const timings = {};
const since = (t) => Math.round(performance.now() - t);
let tok, session;

async function load() {
  // Start the session first: onnxruntime builds it off the main thread while the tokenizer loads.
  let t = performance.now();
  const sessionP = ort.InferenceSession.create(join(MODEL_DIR, 'model_q4.onnx'), {
    executionProviders: ['cpu'], logSeverityLevel: 3, ...(THREADS ? { intraOpNumThreads: THREADS } : {}),
  }).then((s) => { timings.session_create = since(t); return s; });

  // Building the tokenizer from tokenizer.json (32 MB, 262k vocab, 515k merges) is the slowest step
  // of a cold start, about 1.7 s; see RESULTS.md for what did and did not shorten it.
  const t2 = performance.now();
  tok = new Tokenizer(
    JSON.parse(readFileSync(join(MODEL_DIR, 'tokenizer.json'), 'utf8')),
    JSON.parse(readFileSync(join(MODEL_DIR, 'tokenizer_config.json'), 'utf8')),
  );
  timings.tokenizer_load = since(t2);

  session = await sessionP;
  t = performance.now();
  await embed('warm up');
  timings.first_inference = since(t);
  timings.ready = since(T0);                                 // since this module started, excludes node boot
}

async function embed(text, prompt = 'query', dims = 768) {
  const ids = tok.encode(PROMPTS[prompt] + text).ids.slice(0, MAX_TOKENS);
  const S = ids.length;
  const feeds = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, S]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(S).fill(1n), [1, S]),
  };
  for (const name of session.inputNames) {
    if (/^(image|video|audio)_features$/.test(name)) feeds[name] = new ort.Tensor('float32', new Float32Array(0), [0, 512]);
  }
  const out = await session.run(feeds, ['sentence_embedding']);
  const v = out.sentence_embedding.data.slice(0, dims);      // unit length at 768
  if (dims !== 768) {                                        // a Matryoshka prefix needs renormalizing
    let n = 0;
    for (let i = 0; i < dims; i++) n += v[i] * v[i];
    n = Math.sqrt(n);
    for (let i = 0; i < dims; i++) v[i] /= n;
  }
  return { embedding: v, tokens: S };
}

const json = (res, code, body) => {
  res.writeHead(code, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
};

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) { reject(Object.assign(new Error('body too large'), { code: 413 })); req.destroy(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

const server = createServer(async (req, res) => {
  try {
    if (req.method === 'GET' && req.url === '/health') return json(res, 200, { ready: true, timings_ms: timings });
    if (req.method !== 'POST' || req.url !== '/embed') return json(res, 404, { error: 'not found' });
    let body;
    try { body = JSON.parse(await readBody(req, 4 * MAX_CHARS + 1024)); }
    catch (e) { return json(res, e.code === 413 ? 413 : 400, { error: e.message }); }
    const { text, prompt = 'query', dims = 768 } = body ?? {};
    if (typeof text !== 'string' || !text.trim()) return json(res, 400, { error: 'text must be a non-empty string' });
    if (text.length > MAX_CHARS) return json(res, 413, { error: `text over ${MAX_CHARS} characters` });
    if (!Object.hasOwn(PROMPTS, prompt)) return json(res, 400, { error: `unknown prompt; one of ${Object.keys(PROMPTS).join(', ')}` });
    if (!DIMS.has(dims)) return json(res, 400, { error: 'dims must be 768, 512, 256 or 128' });
    const t = performance.now();
    const { embedding, tokens } = await embed(text, prompt, dims);
    return json(res, 200, { embedding: Array.from(embedding), dims, tokens, prompt, ms: Math.round((performance.now() - t) * 10) / 10 });
  } catch (e) {
    return json(res, 500, { error: String(e?.message ?? e) });
  }
});

// The port opens only once the model is loaded, so the container's readiness check is the port.
const ready = load().then(() => {
  if (process.env.NO_LISTEN !== '1') {
    server.listen(PORT, () => console.log(JSON.stringify({ listening: PORT, timings_ms: timings })));
  }
});
ready.catch((e) => { console.error(e); process.exit(1); });

module.exports = { embed, ready };
