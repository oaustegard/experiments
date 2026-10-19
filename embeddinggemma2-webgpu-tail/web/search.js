// Semantic search over a Bluesky list's posts, all in the browser: EmbeddingGemma 2 (onnxruntime-web,
// WebGPU) embeds the query, RemexTail scans the precomputed 2-bit remex codes on the same GPUDevice,
// and the matching posts are fetched from Bluesky's public API for display. The index (built by
// run/build_bsky_index.py) holds codes and post ids only, never post text.
import { RemexTail } from './tail.js';

const D = 768;
const K = 10;
const Q = new URLSearchParams(location.search);
const LOCAL = Q.get('local') === '1';
const HF = Q.get('hf') ? new URL(Q.get('hf'), location.href).href
  : 'https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/';
const CFG = LOCAL
  ? { ortModule: '../interop/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs',
      ortDist: '../interop/node_modules/onnxruntime-web/dist/',
      tokModule: '../interop/node_modules/@huggingface/tokenizers/dist/tokenizers.mjs' }
  : { ortModule: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs',
      ortDist: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
      tokModule: 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs' };
const BSKY = 'https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts';

const $ = (id) => document.getElementById(id);
const abs = (p) => new URL(p, import.meta.url).href;
const logLines = [];
const log = (m) => { logLines.push(m); $('log').textContent = logLines.join('\n'); };
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
const now = () => performance.now();

const state = window.searchState = {
  phase: 'loading', error: null, adapter: null, user_agent: navigator.userAgent,
  index: null, load_ms: {}, parity: null, queries: [],
};

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
  return r;
}

let ort, session, tok, tail, corpus, index, posts, device;

async function embedQuery(text) {
  const ids = tok.encode(index.query_prompt + text).ids.slice(0, index.max_tokens);
  const S = ids.length;
  const feeds = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, S]),
    attention_mask: new ort.Tensor('int64', new BigInt64Array(S).fill(1n), [1, S]),
  };
  for (const name of session.inputNames) {
    if (/^(image|video|audio)_features$/.test(name)) feeds[name] = new ort.Tensor('float32', new Float32Array(0), [0, 512]);
  }
  const res = await session.run(feeds);
  return res;      // caller disposes
}

async function search(text) {
  const t0 = now();
  const res = await embedQuery(text);
  await device.queue.onSubmittedWorkDone();
  const t1 = now();
  const [top] = await tail.topk(res.sentence_embedding.gpuBuffer, 1, corpus, index.n, K);
  const t2 = now();
  for (const t of Object.values(res)) t.dispose();
  return { idx: Array.from(top.idx), score: Array.from(top.score), embed_ms: t1 - t0, search_ms: t2 - t1 };
}

async function startup() {
  let t = now();
  index = await (await fetchOk(abs('bsky/index.json'))).json();
  const [codes, aux, postList, parity, params] = await Promise.all([
    fetchOk(abs('bsky/' + index.files.codes)).then((r) => r.arrayBuffer()),
    fetchOk(abs('bsky/' + index.files.aux)).then((r) => r.arrayBuffer()),
    fetchOk(abs('bsky/' + index.files.posts)).then((r) => r.json()),
    fetchOk(abs('bsky/' + index.files.parity)).then((r) => r.json()),
    fetchOk(abs('assets/params.json')).then((r) => r.json()),
  ]);
  posts = postList;
  state.load_ms.index_fetch = Math.round(now() - t);
  state.index = { n: index.n, authors: index.authors.length, built: index.built, source: index.source };
  log(`index: ${index.n} posts by ${index.authors.length} authors (${(codes.byteLength / 1e6).toFixed(1)} MB of codes), ${state.load_ms.index_fetch} ms`);
  $('lede').append(` ${index.n.toLocaleString()} posts by ${index.authors.length} people, ${index.source.first_post?.slice(0, 10)} to ${index.source.last_post?.slice(0, 10)}.`);

  t = now();
  ort = await import(abs(CFG.ortModule));
  ort.env.wasm.wasmPaths = abs(CFG.ortDist);
  ort.env.wasm.numThreads = 1;
  const { Tokenizer } = await import(abs(CFG.tokModule));
  const [tj, tc] = await Promise.all([
    fetchOk(HF + 'tokenizer.json').then((r) => r.json()),
    fetchOk(HF + 'tokenizer_config.json').then((r) => r.json()),
  ]);
  tok = new Tokenizer(tj, tc);
  $('status').textContent = 'Loading EmbeddingGemma 2 (175 MB the first time)…';
  session = await ort.InferenceSession.create(HF + 'onnx/model_q4.onnx', {
    executionProviders: ['webgpu'],
    preferredOutputLocation: { sentence_embedding: 'gpu-buffer', last_hidden_state: 'gpu-buffer' },
    externalData: [{ path: 'model_q4.onnx_data', data: HF + 'onnx/model_q4.onnx_data' }],
    logSeverityLevel: 3,
  });
  state.load_ms.model = Math.round(now() - t);
  log(`model and tokenizer: ${state.load_ms.model} ms`);

  t = now();
  device = await ort.env.webgpu.device;
  const info = device.adapterInfo ?? {};
  state.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
  tail = await RemexTail.create(device, {
    rht: params.rht, mu: new Float32Array(index.mu), boundaries: params.boundaries,
    centroids: params.centroids, kappa: params.kappa,
  });
  corpus = tail.allocCorpus(index.n);
  tail.loadCodes(corpus, new Uint8Array(codes), new Float32Array(aux), 0);
  await device.queue.onSubmittedWorkDone();
  state.load_ms.codes_to_gpu = Math.round(now() - t);
  log(`codes on the GPU: ${state.load_ms.codes_to_gpu} ms (adapter ${state.adapter.vendor} ${state.adapter.architecture})`);

  // Parity: each test query, embedded here, must get remex's own top-10 (built natively from the
  // CPU embedding of the same query), allowing a swap where remex's scores nearly tie.
  t = now();
  const rows = [];
  for (const p of parity) {
    const r = await search(p.query);
    const shared = r.idx.filter((i) => p.remex_top10.includes(i)).length;
    const exact = r.idx.filter((i) => p.exact_top10.includes(i)).length;
    let maxScoreErr = 0;
    r.idx.forEach((i, j) => { const k = p.remex_top10.indexOf(i); if (k >= 0) maxScoreErr = Math.max(maxScoreErr, Math.abs(r.score[j] - p.remex_scores[k])); });
    rows.push({ query: p.query, overlap_with_remex: shared / K, recall_vs_exact: exact / K, top1_match: r.idx[0] === p.remex_top10[0],
                max_score_err: maxScoreErr, embed_ms: +r.embed_ms.toFixed(1), search_ms: +r.search_ms.toFixed(1) });
  }
  const overlap = rows.reduce((a, r) => a + r.overlap_with_remex, 0) / rows.length;
  const passed = rows.every((r) => r.overlap_with_remex >= 0.8) && overlap >= 0.9 && rows.every((r) => r.max_score_err < 0.01);
  state.parity = { passed, mean_overlap_with_remex: overlap,
                   mean_recall_vs_exact: rows.reduce((a, r) => a + r.recall_vs_exact, 0) / rows.length, rows,
                   ms: Math.round(now() - t) };
  log(`parity with remex: mean top-10 overlap ${overlap.toFixed(3)}, ${passed ? 'PASSED' : 'FAILED'}`);
  for (const r of rows) log(`  ${r.overlap_with_remex.toFixed(1)} remex, ${r.recall_vs_exact.toFixed(1)} exact, score err ${r.max_score_err.toExponential(1)}  "${r.query}"`);
  return passed;
}

async function fetchPosts(idxs) {
  const uris = idxs.map((i) => {
    const [a, rkey] = posts[i];
    return `at://${index.authors[a].did}/app.bsky.feed.post/${rkey}`;
  });
  const qs = uris.map((u) => 'uris=' + encodeURIComponent(u)).join('&');
  try {
    const r = await fetch(`${BSKY}?${qs}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    const byUri = new Map(d.posts.map((p) => [p.uri, p]));
    return uris.map((u) => byUri.get(u) ?? null);
  } catch (e) {
    log(`Bluesky fetch failed: ${e.message}`);
    return uris.map(() => null);
  }
}

function render(res, fetched) {
  const ol = $('hits');
  ol.textContent = '';
  res.idx.forEach((i, j) => {
    const [a, rkey] = posts[i];
    const author = index.authors[a];
    const p = fetched[j];
    const li = document.createElement('li');
    const meta = document.createElement('div');
    meta.className = 'meta';
    const who = document.createElement('a');
    who.href = `https://bsky.app/profile/${author.handle}`;
    who.textContent = p?.author?.displayName || '@' + author.handle;
    who.target = '_blank';
    const when = document.createElement('a');
    when.href = `https://bsky.app/profile/${author.handle}/post/${rkey}`;
    when.target = '_blank';
    when.textContent = p?.record?.createdAt ? new Date(p.record.createdAt).toLocaleDateString() : 'open post';
    const sc = document.createElement('span');
    sc.textContent = `score ${res.score[j].toFixed(3)}`;
    meta.append(who, when, sc);
    const text = document.createElement('div');
    text.className = 'text';
    text.textContent = p ? p.record.text : '(post no longer available, or Bluesky did not answer)';
    li.append(meta, text);
    ol.append(li);
  });
}

let pending = null, running = false;
async function runQuery(text) {
  if (running) { pending = text; return; }
  running = true;
  try {
    const res = await search(text);
    const t = now();
    const fetched = await fetchPosts(res.idx);
    const fetchMs = now() - t;
    render(res, fetched);
    state.queries.push({ embed_ms: +res.embed_ms.toFixed(1), search_ms: +res.search_ms.toFixed(1), fetch_ms: +fetchMs.toFixed(1) });
    const e = state.queries.map((q) => q.embed_ms), s = state.queries.map((q) => q.search_ms);
    $('timing').textContent = `embed ${res.embed_ms.toFixed(0)} ms · scan + top-10 of ${index.n.toLocaleString()} ${res.search_ms.toFixed(1)} ms · Bluesky ${fetchMs.toFixed(0)} ms` +
      (state.queries.length > 1 ? `  (medians over ${state.queries.length}: ${median(e).toFixed(0)} / ${median(s).toFixed(1)} ms)` : '');
  } catch (err) {
    state.error = String(err?.stack ?? err);
    log(`search ERROR: ${err?.message ?? err}`);
  } finally {
    running = false;
    if (pending !== null) { const p = pending; pending = null; runQuery(p); }
  }
}

$('copy').addEventListener('click', () => {
  const e = state.queries.map((q) => q.embed_ms), s = state.queries.map((q) => q.search_ms);
  const out = { ...state, query_medians: { embed_ms: median(e), search_ms: median(s), n: state.queries.length }, when: new Date().toISOString() };
  navigator.clipboard.writeText(JSON.stringify(out, null, 1));
});

let timer = null;
$('q').addEventListener('input', (ev) => {
  clearTimeout(timer);
  const v = ev.target.value.trim();
  if (v.length < 3) return;
  timer = setTimeout(() => runQuery(v), 200);
});

(async () => {
  try {
    if (!navigator.gpu) throw new Error('this browser has no WebGPU');
    const ok = await startup();
    const st = $('status');
    st.className = ok ? 'status ok' : 'status bad';
    st.textContent = ok
      ? `Ready. Parity with remex: ${(state.parity.mean_overlap_with_remex * 100).toFixed(0)}% top-10 overlap on ${state.parity.rows.length} test queries.`
      : 'Parity check FAILED: GPU results disagree with remex, so search is disabled. See the log below.';
    state.phase = ok ? 'ready' : 'parity-failed';
    if (ok) {
      const q = $('q');
      q.disabled = false;
      q.placeholder = 'Search posts, e.g. “do LLMs reason or pattern-match?”';
      q.focus();
    }
  } catch (err) {
    state.phase = 'error';
    state.error = String(err?.stack ?? err);
    $('status').className = 'status bad';
    $('status').textContent = `Error: ${err?.message ?? err}`;
    log(`ERROR: ${err?.stack ?? err}`);
  }
})();
