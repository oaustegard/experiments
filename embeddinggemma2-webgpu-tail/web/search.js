// Semantic search over a Bluesky list's posts on this GPU. Two modes:
//
// default    Loads the prebuilt index in bsky/ (remex 2-bit codes, built by run/build_bsky_index.py)
//            onto a WebGPU device of its own, and embeds each query with the hosted EG2 query
//            embedder (same q4 graph and prompt), so no model is downloaded. The scan and top-10 run
//            on the GPU; the matching posts' text comes from Bluesky. If the embedder refuses
//            (rate limit, origin), the page loads the model and embeds locally from then on.
// ?build=1   Builds the index here: fetches the list's members and posts from Bluesky, embeds
//            every post with EmbeddingGemma 2 (onnxruntime-web, WebGPU), keeps the embeddings on
//            the GPU, encodes them with RemexTail on the same GPUDevice (no readback), and saves
//            the index in this browser (IndexedDB).
//
// URL options: ?parity=1 checks the prebuilt index against remex's own top-10 for its test
// queries; with ?build=1: ?per=200 posts per member, ?members=N first N members (tests),
// ?rebuild=1 ignore the saved index, ?compare=1 cross-check against bsky/. ?hf=<base> and
// ?local=1 as in demo.html.
import { RemexTail } from './tail.js';

const D = 768;
const K = 10;
const BATCH = 32;
const MAX_TOKENS = 128;
const CHUNK_ROWS = 16384;             // embedding store chunk (48 MB), under the default binding limit
const REENCODE_EVERY = 64;            // batches between re-encodes with the running mean
const DOC = 'title: none | text: ';
const QUERY = 'task: search result | query: ';
const LIST = 'at://did:plc:r2whjvupgfw55mllpksnombn/app.bsky.graph.list/3lankcdrlip2f';
const API = 'https://public.api.bsky.app/xrpc/';
const DB_KEY = 'ai-list-v1';

const EMBEDDER = 'https://eg2-query-embedder.austegard.workers.dev/';
const Q = new URLSearchParams(location.search);
const BUILD = Q.get('build') === '1';
const PARITY = Q.get('parity') === '1';
const PER = Number(Q.get('per') || 200);
const MAX_MEMBERS = Q.get('members') ? Number(Q.get('members')) : Infinity;
const REBUILD = Q.get('rebuild') === '1';
const COMPARE = Q.get('compare') === '1';
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

const $ = (id) => document.getElementById(id);
const abs = (p) => new URL(p, import.meta.url).href;
const now = () => performance.now();
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; };
const logLines = [];
const log = (m) => { logLines.push(m); $('log').textContent = logLines.join('\n'); };
const status = (text, cls = '') => { const s = $('status'); s.textContent = text; s.className = 'status ' + cls; };

const state = window.searchState = {
  phase: 'loading', error: null, adapter: null, user_agent: navigator.userAgent,
  mode: BUILD ? 'build' : 'prebuilt',
  options: { per: PER, members: Number.isFinite(MAX_MEMBERS) ? MAX_MEMBERS : null, compare: COMPARE },
  source: null, load_ms: {}, build: null, compare: null, parity: null, queries: [],
};

let ort, session, tok, specials, tail, device;
let posts = [];            // { a, rkey, text, created_at } in corpus order
let authors = [];          // { did, handle, name }
let corpus = null, filled = 0;

// ---------------------------------------------------------------- session access, one run at a time

let sessionQueue = Promise.resolve();
function withSession(fn) {
  const run = sessionQueue.then(fn, fn);
  sessionQueue = run.catch(() => {});
  return run;
}

function clean(text) {
  for (const sp of specials) if (text.includes(sp)) text = text.split(sp).join(' ');
  return text;
}

async function embedTexts(texts) {
  const enc = texts.map((t) => tok.encode(clean(t)).ids.slice(0, MAX_TOKENS));
  const B = enc.length, S = Math.max(...enc.map((e) => e.length));
  const ids = new BigInt64Array(B * S), attn = new BigInt64Array(B * S);
  enc.forEach((row, b) => row.forEach((id, s) => { ids[b * S + s] = BigInt(id); attn[b * S + s] = 1n; }));
  const feeds = {
    input_ids: new ort.Tensor('int64', ids, [B, S]),
    attention_mask: new ort.Tensor('int64', attn, [B, S]),
  };
  for (const name of session.inputNames) {
    if (/^(image|video|audio)_features$/.test(name)) feeds[name] = new ort.Tensor('float32', new Float32Array(0), [0, 512]);
  }
  return session.run(feeds);          // caller disposes every output tensor
}

// ---------------------------------------------------------------- Bluesky

async function bsky(method, params) {
  const url = API + method + '?' + new URLSearchParams(params);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) return r.json();
    if (r.status === 400 || attempt >= 4) return null;
    await new Promise((res) => setTimeout(res, 500 * 2 ** attempt));
  }
}

function postText(p) {
  const rec = p.record;
  const parts = [rec.text || ''];
  const emb = rec.embed || {};
  const imgs = emb.images || emb.media?.images || [];
  for (const i of imgs) if (i.alt) parts.push(i.alt);
  return parts.filter(Boolean).join('\n').trim();
}

async function harvest() {
  const members = [];
  let cursor, listName = null;
  do {
    const d = await bsky('app.bsky.graph.getList', { list: LIST, limit: 100, ...(cursor ? { cursor } : {}) });
    listName = d.list.name;
    for (const it of d.items) members.push({ did: it.subject.did, handle: it.subject.handle, name: it.subject.displayName || '' });
    cursor = d.cursor;
  } while (cursor && members.length < MAX_MEMBERS);
  const ms = members.slice(0, MAX_MEMBERS);
  log(`list "${listName}": ${ms.length} members`);
  const out = new Array(ms.length);
  let next = 0, done = 0, count = 0;
  async function worker() {
    while (next < ms.length) {
      const i = next++;
      const m = ms[i];
      const kept = [];
      let c, pages = 0;
      while (kept.length < PER && pages < 20) {
        const d = await bsky('app.bsky.feed.getAuthorFeed', { actor: m.did, limit: 100, filter: 'posts_and_author_threads', ...(c ? { cursor: c } : {}) });
        pages++;
        if (!d?.feed?.length) break;
        for (const it of d.feed) {
          const p = it.post;
          if (it.reason || p.author.did !== m.did) continue;
          const text = postText(p);
          if (text.length < 20) continue;
          kept.push({ rkey: p.uri.split('/').pop(), text, created_at: p.record.createdAt });
          if (kept.length >= PER) break;
        }
        c = d.cursor;
        if (!c) break;
      }
      out[i] = kept;
      done++;
      count += kept.length;
      status(`Fetching posts from Bluesky: ${done}/${ms.length} members, ${count.toLocaleString()} posts`);
    }
  }
  await Promise.all(Array.from({ length: 6 }, worker));
  authors = [];
  posts = [];
  ms.forEach((m, i) => {
    if (!out[i].length) return;
    const a = authors.length;
    authors.push(m);
    for (const p of out[i]) posts.push({ a, ...p });
  });
  return listName;
}

// ---------------------------------------------------------------- build on the GPU

async function build() {
  const t0 = now();
  const listName = await harvest();
  const tFetch = now() - t0;
  // corpus order = batch order: sorted by length so each batch pads little
  posts.sort((x, y) => x.text.length - y.text.length);
  const n = posts.length;
  state.source = { list: LIST, list_name: listName, members: authors.length, posts: n, per_member: PER };
  log(`fetched ${n.toLocaleString()} posts by ${authors.length} people in ${(tFetch / 1000).toFixed(1)} s`);

  corpus = tail.allocCorpus(n);
  const chunks = [];
  for (let r = 0; r < n; r += CHUNK_ROWS) {
    chunks.push(device.createBuffer({ size: Math.min(CHUNK_ROWS, n - r) * D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }));
  }
  const acc = device.createBuffer({ size: D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(acc, 0, new Float32Array(D));

  let embedded = 0, encodeMs = [], batchMs = [];
  const tEmbed = now();
  async function reencode() {
    const te = now();
    const sum = new Float32Array(await tail.readBuffer(acc, D * 4));
    tail.setMean(sum.map((v) => v / embedded));
    for (let c = 0; c * CHUNK_ROWS < embedded; c++) {
      tail.encode(chunks[c], Math.min(CHUNK_ROWS, embedded - c * CHUNK_ROWS), corpus, c * CHUNK_ROWS, { normalize: true });
    }
    await device.queue.onSubmittedWorkDone();
    filled = embedded;
    encodeMs.push(now() - te);
    enableSearch();
  }

  for (let b = 0, batchNo = 0; b < n; b += BATCH, batchNo++) {
    const rows = posts.slice(b, Math.min(n, b + BATCH));
    const tb = now();
    await withSession(async () => {
      const res = await embedTexts(rows.map((p) => DOC + p.text));
      const emb = res.sentence_embedding.gpuBuffer;
      // copy each row into its chunk (a batch never straddles a chunk: CHUNK_ROWS % BATCH == 0)
      const c = Math.floor(b / CHUNK_ROWS), off = b - c * CHUNK_ROWS;
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(emb, 0, chunks[c], off * D * 4, rows.length * D * 4);
      device.queue.submit([enc.finish()]);
      tail.colsum(emb, rows.length, acc);
      await device.queue.onSubmittedWorkDone();
      for (const t of Object.values(res)) t.dispose();
    });
    batchMs.push(now() - tb);
    embedded += rows.length;
    if ((batchNo + 1) % REENCODE_EVERY === 0 || embedded === n) await withSession(reencode);   // not mid-search
    const rate = embedded / ((now() - tEmbed) / 1000);
    status(`Embedding on this GPU: ${embedded.toLocaleString()}/${n.toLocaleString()} posts, ${rate.toFixed(0)} posts/s` +
      (filled ? ` · searchable: ${filled.toLocaleString()}` : ''));
  }
  const tEmbedTotal = now() - tEmbed;
  acc.destroy();
  state.build = {
    posts: n, fetch_ms: Math.round(tFetch), embed_ms: Math.round(tEmbedTotal),
    embed_posts_per_s: +(n / (tEmbedTotal / 1000)).toFixed(1),
    batch_ms_median: +median(batchMs).toFixed(1), batches: batchMs.length,
    encode_all_ms_last: +encodeMs[encodeMs.length - 1].toFixed(1), reencodes: encodeMs.length,
  };
  log(`embedded ${n.toLocaleString()} posts in ${(tEmbedTotal / 1000).toFixed(1)} s (${state.build.embed_posts_per_s} posts/s, median batch ${state.build.batch_ms_median} ms)`);
  log(`last encode of all ${n.toLocaleString()} on the GPU: ${state.build.encode_all_ms_last} ms`);
  return chunks;
}

// ---------------------------------------------------------------- persistence (IndexedDB)

function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open('remex-tail-search', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('index');
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function idbGet(key) {
  const db = await idb();
  return new Promise((res, rej) => {
    const r = db.transaction('index').objectStore('index').get(key);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function idbPut(key, value) {
  const db = await idb();
  return new Promise((res, rej) => {
    const tx = db.transaction('index', 'readwrite');
    tx.objectStore('index').put(value, key);
    tx.oncomplete = () => res();
    tx.onerror = () => rej(tx.error);
  });
}

async function save() {
  const c = await tail.readCorpus(corpus, filled);
  const sb = new Float32Array(filled * 2);
  for (let i = 0; i < filled; i++) { sb[i * 2] = c.scale[i]; sb[i * 2 + 1] = c.bias[i]; }
  await idbPut(DB_KEY, { built: new Date().toISOString(), source: state.source, authors, posts,
                         mu: tail.mu_host, packed: c.packed, scaleBias: sb, options: state.options });
}

async function restore(saved) {
  authors = saved.authors;
  posts = saved.posts;
  corpus = tail.allocCorpus(posts.length);
  tail.setMean(saved.mu);
  tail.loadCodes(corpus, saved.packed, saved.scaleBias, 0);
  await device.queue.onSubmittedWorkDone();
  filled = posts.length;
  state.source = saved.source;
}

// ---------------------------------------------------------------- search

async function searchLocal(text) {
  const t0 = now();
  return withSession(async () => {
    const res = await embedTexts([QUERY + text]);
    let qb = res.sentence_embedding.gpuBuffer;
    if (!BUILD) {
      // fallback session in prebuilt mode: it may sit on onnxruntime's own device, so hand the
      // 768 floats across instead of the buffer
      qbuf ??= device.createBuffer({ size: D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(qbuf, 0, await res.sentence_embedding.getData());
      qb = qbuf;
    } else {
      await device.queue.onSubmittedWorkDone();
    }
    const t1 = now();
    const [top] = await tail.topk(qb, 1, corpus, filled, K);
    const t2 = now();
    for (const t of Object.values(res)) t.dispose();
    return { idx: Array.from(top.idx), score: Array.from(top.score), embed_ms: t1 - t0, search_ms: t2 - t1, path: 'local' };
  });
}

// ---------------------------------------------------------------- hosted query embedder

const vecCache = new Map();
let lastWarm = -Infinity, embedderDown = null, localLoading = null;
function warm() {
  if (BUILD || embedderDown || now() - lastWarm < 9 * 60_000) return;
  lastWarm = now();
  fetch(EMBEDDER + 'warm', { method: 'POST' }).catch(() => {});
}

async function serverEmbed(text) {
  const key = text.slice(0, 512);
  if (vecCache.has(key)) return { vec: vecCache.get(key), path: 'client-cache', server_ms: 0 };
  const r = await fetch(EMBEDDER + 'embed', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: key, dims: D }),
  });
  if (!r.ok) throw Object.assign(new Error(`query embedder HTTP ${r.status}`), { http: r.status });
  const d = await r.json();
  const vec = new Float32Array(d.embedding);
  vecCache.set(key, vec);
  lastWarm = now();
  return { vec, path: d.cached ? 'server-cached' : 'server', server_ms: d.ms };
}

let qbuf = null;
async function searchServer(text) {
  const t0 = now();
  const e = await serverEmbed(text);
  const t1 = now();
  qbuf ??= device.createBuffer({ size: D * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(qbuf, 0, e.vec);
  const [top] = await tail.topk(qbuf, 1, corpus, filled, K);
  const t2 = now();
  // a cold container answers in seconds, a warm one in about half a second
  const path = e.path === 'server' && t1 - t0 > 2000 ? 'server-cold' : e.path;
  return { idx: Array.from(top.idx), score: Array.from(top.score), embed_ms: t1 - t0, search_ms: t2 - t1,
           path, server_ms: e.server_ms };
}

async function search(text) {
  if (BUILD || embedderDown) {
    if (!session) {
      localLoading ??= loadModel().then(() => log('local model ready; queries now embed in this browser'));
      await localLoading;
    }
    return searchLocal(text);
  }
  try {
    return await searchServer(text);
  } catch (err) {
    embedderDown = err.message;
    log(`${err.message}: loading the model to embed queries here instead`);
    status(`The query embedder said no (${err.message}); loading the model locally, this takes a while the first time…`);
    return search(text);
  }
}

async function fetchTexts(idxs) {
  const need = idxs.filter((i) => posts[i].text == null);
  if (!need.length) return;
  const uris = need.map((i) => `at://${authors[posts[i].a].did}/app.bsky.feed.post/${posts[i].rkey}`);
  try {
    const r = await fetch(API + 'app.bsky.feed.getPosts?' + uris.map((u) => 'uris=' + encodeURIComponent(u)).join('&'));
    const d = r.ok ? await r.json() : { posts: [] };
    const byUri = new Map(d.posts.map((p) => [p.uri, p]));
    need.forEach((i, j) => {
      const p = byUri.get(uris[j]);
      posts[i].text = p ? p.record.text : '(post no longer available)';
      posts[i].created_at = p?.record?.createdAt;
      if (p?.author?.displayName) authors[posts[i].a].name = p.author.displayName;
    });
  } catch (e) {
    log(`Bluesky fetch failed: ${e.message}`);
  }
}

function render(res) {
  const ol = $('hits');
  ol.textContent = '';
  res.idx.forEach((i, j) => {
    const p = posts[i];
    const author = authors[p.a];
    const li = document.createElement('li');
    const meta = document.createElement('div');
    meta.className = 'meta';
    const who = document.createElement('a');
    who.href = `https://bsky.app/profile/${author.handle}`;
    who.target = '_blank';
    who.textContent = author.name || '@' + author.handle;
    const when = document.createElement('a');
    when.href = `https://bsky.app/profile/${author.handle}/post/${p.rkey}`;
    when.target = '_blank';
    when.textContent = p.created_at ? new Date(p.created_at).toLocaleDateString() : 'open post';
    const sc = document.createElement('span');
    sc.textContent = `score ${res.score[j].toFixed(3)}`;
    meta.append(who, when, sc);
    const t = document.createElement('div');
    t.className = 'text';
    t.textContent = p.text ?? '(Bluesky did not answer)';
    li.append(meta, t);
    ol.append(li);
  });
}

let pending = null, running = false;
async function runQuery(text) {
  if (running) { pending = text; return; }
  running = true;
  try {
    const res = await search(text);
    const tf = now();
    await fetchTexts(res.idx);
    const fetchMs = now() - tf;
    render(res);
    state.queries.push({ path: res.path, embed_ms: +res.embed_ms.toFixed(1), server_ms: res.server_ms ?? null,
                         search_ms: +res.search_ms.toFixed(1), fetch_ms: +fetchMs.toFixed(1), n: filled });
    const e = state.queries.map((q) => q.embed_ms), s = state.queries.map((q) => q.search_ms);
    $('timing').textContent = `embed ${res.embed_ms.toFixed(0)} ms (${res.path}) · scan + top-10 of ${filled.toLocaleString()} ${res.search_ms.toFixed(1)} ms` +
      (fetchMs > 1 ? ` · Bluesky ${fetchMs.toFixed(0)} ms` : '') +
      (state.queries.length > 1 ? `  (medians over ${state.queries.length}: ${median(e).toFixed(0)} / ${median(s).toFixed(1)} ms)` : '');
  } catch (err) {
    state.error = String(err?.stack ?? err);
    log(`search ERROR: ${err?.message ?? err}`);
  } finally {
    running = false;
    if (pending !== null) { const p = pending; pending = null; runQuery(p); }
  }
}

let searchEnabled = false;
function enableSearch() {
  if (searchEnabled) return;
  searchEnabled = true;
  const q = $('q');
  q.disabled = false;
  q.placeholder = 'Search posts, e.g. “do LLMs reason or pattern-match?”';
}

// ---------------------------------------------------------------- cross-check against the CPU-built reference

async function compare(chunks) {
  const t0 = now();
  const ref = await (await fetch(abs('bsky/index.json'))).json();
  const [codes, aux, refPosts, parity] = await Promise.all([
    fetch(abs('bsky/' + ref.files.codes)).then((r) => r.arrayBuffer()),
    fetch(abs('bsky/' + ref.files.aux)).then((r) => r.arrayBuffer()),
    fetch(abs('bsky/' + ref.files.posts)).then((r) => r.json()),
    fetch(abs('bsky/' + ref.files.parity)).then((r) => r.json()),
  ]);
  const refIdx = new Map(refPosts.map(([a, rkey], i) => [ref.authors[a].did + '/' + rkey, i]));
  const shared = [];       // [browser row, reference row]
  posts.forEach((p, i) => { const j = refIdx.get(authors[p.a].did + '/' + p.rkey); if (j !== undefined) shared.push([i, j]); });
  // encode this GPU's embeddings with the reference mean, so the codes differ only by the embeddings
  const keepMu = tail.mu_host;
  tail.setMean(new Float32Array(ref.mu));
  const scratch = tail.allocCorpus(posts.length);
  for (let c = 0; c < chunks.length; c++) {
    tail.encode(chunks[c], Math.min(CHUNK_ROWS, posts.length - c * CHUNK_ROWS), scratch, c * CHUNK_ROWS, { normalize: true });
  }
  const mine = await tail.readCorpus(scratch, posts.length);
  scratch.words.destroy(); scratch.aux.destroy();
  tail.setMean(keepMu);
  const refBytes = new Uint8Array(codes);
  let diff = 0;
  for (const [i, j] of shared) {
    for (let k = 0; k < 192; k++) {
      let x = mine.packed[i * 192 + k] ^ refBytes[j * 192 + k];
      while (x) { if (x & 3) diff++; x >>= 2; }
    }
  }
  // the reference's own parity queries: top-10 over the reference corpus, searched with codes
  // loaded from the reference files, queries embedded here
  const refCorpus = tail.allocCorpus(ref.n);
  const keep2 = tail.mu_host;
  tail.setMean(new Float32Array(ref.mu));
  tail.loadCodes(refCorpus, refBytes, new Float32Array(aux), 0);
  let overlap = 0;
  for (const p of parity) {
    const r = await withSession(async () => {
      const res = await embedTexts([QUERY + p.query]);
      const [top] = await tail.topk(res.sentence_embedding.gpuBuffer, 1, refCorpus, ref.n, K);
      for (const t of Object.values(res)) t.dispose();
      return Array.from(top.idx);
    });
    overlap += r.filter((i) => p.remex_top10.includes(i)).length / K;
  }
  refCorpus.words.destroy(); refCorpus.aux.destroy();
  tail.setMean(keep2);
  state.compare = {
    reference_posts: ref.n, shared_posts: shared.length,
    code_flips: diff, codes_compared: shared.length * D, code_agreement: 1 - diff / Math.max(1, shared.length * D),
    parity_overlap_with_remex: overlap / parity.length, ms: Math.round(now() - t0),
  };
  log(`compare: ${shared.length.toLocaleString()} posts shared with the CPU-built index; ` +
      `codes from this GPU's embeddings agree on ${(state.compare.code_agreement * 100).toFixed(3)}% of coordinates; ` +
      `remex parity on its own corpus: ${(state.compare.parity_overlap_with_remex * 100).toFixed(0)}% top-10 overlap`);
}

// ---------------------------------------------------------------- startup

// Loads onnxruntime-web, the tokenizer and EG2. In build mode the tail then runs on onnxruntime's
// device; in prebuilt mode (embedder fallback) the tail keeps its own device.
async function loadModel() {
  const t = now();
  ort = await import(abs(CFG.ortModule));
  ort.env.wasm.wasmPaths = abs(CFG.ortDist);
  ort.env.wasm.numThreads = 1;
  const { Tokenizer } = await import(abs(CFG.tokModule));
  const [tj, tc] = await Promise.all([
    fetch(HF + 'tokenizer.json').then((r) => r.json()),
    fetch(HF + 'tokenizer_config.json').then((r) => r.json()),
  ]);
  tok = new Tokenizer(tj, tc);
  specials = tj.added_tokens.filter((a) => a.special).map((a) => a.content).sort((x, y) => y.length - x.length);
  session = await ort.InferenceSession.create(HF + 'onnx/model_q4.onnx', {
    executionProviders: ['webgpu'],
    preferredOutputLocation: { sentence_embedding: 'gpu-buffer', last_hidden_state: 'gpu-buffer' },
    externalData: [{ path: 'model_q4.onnx_data', data: HF + 'onnx/model_q4.onnx_data' }],
    logSeverityLevel: 3,
  });
  device ??= await ort.env.webgpu.device;
  state.load_ms.model = Math.round(now() - t);
  log(`model loaded in ${(state.load_ms.model / 1000).toFixed(1)} s`);
}

function noteAdapter(info = {}) {
  state.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
}

async function makeTail(params, mu) {
  return RemexTail.create(device, {
    rht: params.rht, mu, boundaries: params.boundaries, centroids: params.centroids, kappa: params.kappa,
  });
}

async function startupPrebuilt() {
  let t = now();
  const index = await (await fetch(abs('bsky/index.json'))).json();
  const [codes, aux, postList, params] = await Promise.all([
    fetch(abs('bsky/' + index.files.codes)).then((r) => r.arrayBuffer()),
    fetch(abs('bsky/' + index.files.aux)).then((r) => r.arrayBuffer()),
    fetch(abs('bsky/' + index.files.posts)).then((r) => r.json()),
    fetch(abs('assets/params.json')).then((r) => r.json()),
  ]);
  state.load_ms.index_fetch = Math.round(now() - t);
  t = now();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize,
  } });
  noteAdapter(adapter.info ?? device.adapterInfo);
  tail = await makeTail(params, new Float32Array(index.mu));
  authors = index.authors.map((a) => ({ ...a, name: '' }));
  posts = postList.map(([a, rkey]) => ({ a, rkey, text: null, created_at: null }));
  corpus = tail.allocCorpus(index.n);
  tail.loadCodes(corpus, new Uint8Array(codes), new Float32Array(aux), 0);
  await device.queue.onSubmittedWorkDone();
  filled = index.n;
  state.load_ms.codes_to_gpu = Math.round(now() - t);
  state.source = { ...index.source, posts: index.n, members: index.authors.length, built: index.built };
  log(`prebuilt index: ${index.n.toLocaleString()} posts by ${index.authors.length} people (${(codes.byteLength / 1e6).toFixed(1)} MB of codes), ` +
      `fetched in ${state.load_ms.index_fetch} ms, on the GPU in ${state.load_ms.codes_to_gpu} ms (${state.adapter.vendor} ${state.adapter.architecture})`);
  if (PARITY) {
    const parity = await (await fetch(abs('bsky/' + index.files.parity))).json();
    let overlap = 0;
    for (const p of parity) {
      const r = await searchServer(p.query);
      overlap += r.idx.filter((i) => p.remex_top10.includes(i)).length / K;
    }
    state.parity = { queries: parity.length, overlap_with_remex: overlap / parity.length };
    log(`parity with remex over ${parity.length} test queries: ${(state.parity.overlap_with_remex * 100).toFixed(0)}% top-10 overlap`);
  }
  enableSearch();
  warm();
  const span = `${index.source.first_post?.slice(0, 10)} to ${index.source.last_post?.slice(0, 10)}`;
  return `Ready: ${index.n.toLocaleString()} posts by ${index.authors.length} people (${span}). ` +
         'Queries are embedded by the hosted EG2 embedder; the scan runs on your GPU.';
}

async function startup() {
  if (!BUILD) return startupPrebuilt();
  status('Loading EmbeddingGemma 2 (175 MB the first time, then cached by the browser)…');
  await loadModel();
  noteAdapter(device.adapterInfo);
  const params = await (await fetch(abs('assets/params.json'))).json();
  tail = await makeTail(params, new Float32Array(D));

  const saved = REBUILD || COMPARE ? null : await idbGet(DB_KEY).catch(() => null);
  if (saved && saved.options?.per === PER && (saved.options?.members ?? null) === state.options.members) {
    t = now();
    await restore(saved);
    log(`loaded the index this browser built on ${saved.built.slice(0, 10)}: ${posts.length.toLocaleString()} posts, ${Math.round(now() - t)} ms`);
    enableSearch();
    return `Ready: ${posts.length.toLocaleString()} posts, indexed on this GPU on ${saved.built.slice(0, 10)}. ` +
           'Add ?rebuild=1 to the URL to fetch and index again.';
  }
  const chunks = await build();
  if (COMPARE) await compare(chunks);
  for (const c of chunks) c.destroy();
  try { await save(); log('index saved in this browser'); } catch (e) { log(`could not save the index: ${e.message}`); }
  return `Ready: ${posts.length.toLocaleString()} posts by ${authors.length} people, embedded and encoded on this GPU ` +
         `at ${state.build.embed_posts_per_s} posts/s.`;
}

$('q').addEventListener('input', (() => {
  let timer = null;
  return (ev) => {
    clearTimeout(timer);
    const v = ev.target.value.trim();
    if (v.length < 3) return;
    timer = setTimeout(() => runQuery(v), BUILD || embedderDown ? 200 : 450);   // the embedder allows ~20/min
  };
})());

$('q').addEventListener('focus', warm);

$('copy').addEventListener('click', () => {
  const e = state.queries.map((q) => q.embed_ms), s = state.queries.map((q) => q.search_ms);
  const out = { ...state, query_medians: { embed_ms: median(e), search_ms: median(s), n: state.queries.length }, when: new Date().toISOString() };
  navigator.clipboard.writeText(JSON.stringify(out, null, 1));
});

(async () => {
  try {
    if (!navigator.gpu) throw new Error('this browser has no WebGPU');
    const msg = await startup();
    status(msg, 'ok');
    state.phase = 'ready';
    $('q').focus();
  } catch (err) {
    state.phase = 'error';
    state.error = String(err?.stack ?? err);
    status(`Error: ${err?.message ?? err}`, 'bad');
    log(`ERROR: ${err?.stack ?? err}`);
  }
})();
