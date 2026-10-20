// Computes EmbeddingGemma 2 query-prompt vectors for web/items.json on the browser's GPU and
// saves them as int8 shards (one file per 30,000 items, under GitHub's 25 MB upload cap).
//
// items.json: a string is a query; an integer is a post index into the published
// embeddinggemma2-webgpu-tail index, whose snippet is built here from text fetched from
// Bluesky (same rule as run/build_web.py snippet()). parity.json: queries with CPU vectors,
// embedded first; the page reports their cosine to the CPU reference.
//
// Shard file layout (little-endian): "EGQT", uint32 header length H, H bytes of JSON header
// (padded to 4), then int8[n*768] (x / scale, rounded), float32[n] scale (max|x| / 127),
// uint8[n] status (1 = embedded, 0 = post text unavailable), padding to 2, uint16[n] snippet
// length in code points (0 for queries). run/read_shards.py reads it.
//
// URL options: ?shard=k starts at shard k; ?limit=n keeps only the first n items (smoke
// tests); ?parity_n=n checks only n parity queries; ?items=<file> reads another items file; ?hf=<base> loads the model from a mirror.

const Q = new URLSearchParams(location.search);
const HF = Q.get('hf') ? new URL(Q.get('hf'), location.href).href
  : 'https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/main/';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
const TOK = 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs';
const BSKY = 'https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts?';
const TAIL = new URL('../../embeddinggemma2-webgpu-tail/web/bsky/', location.href).href;
const QUERY = 'task: search result | query: ';
const MAX_TOKENS = 128, D = 768, BATCH = 64;

const $ = (id) => document.getElementById(id);
const now = () => performance.now();
const logLines = [];
const log = (m) => { logLines.push(m); $('log').textContent = logLines.join('\n'); $('log').scrollTop = 1e9; };
const status = (t, cls = '') => { $('status').textContent = t; $('status').className = 'status ' + cls; };

const state = window.egState = {
  phase: 'loading', error: null, user_agent: navigator.userAgent, adapter: null,
  load_ms: null, parity: null, shards: [], options: Object.fromEntries(Q),
};

let ort, session, tok, specials;

function clean(text) {
  for (const sp of specials) if (text.includes(sp)) text = text.split(sp).join(' ');
  return text;
}

function snippet(text) {
  const t = Array.from(text.split(/\s+/).filter(Boolean).join(' '));
  if (t.length <= 200) return t.join('');
  const cut = t.slice(0, 200);
  const sp = cut.lastIndexOf(' ');
  return (sp >= 120 ? cut.slice(0, sp) : cut).join('');
}

function postText(p) {
  const rec = p.record;
  const parts = [rec.text || ''];
  const emb = rec.embed || {};
  const imgs = emb.images || emb.media?.images || [];
  for (const i of imgs) if (i.alt) parts.push(i.alt);
  return parts.filter(Boolean).join('\n').trim();
}

async function getPosts(uris) {
  const url = BSKY + uris.map((u) => 'uris=' + encodeURIComponent(u)).join('&');
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(url);
      if (r.ok) return (await r.json()).posts;
      if (r.status === 400 || attempt >= 5) return [];
    } catch (e) {
      if (attempt >= 5) return [];
    }
    await new Promise((res) => setTimeout(res, 500 * 2 ** attempt));
  }
}

// Post index -> snippet text (null when Bluesky no longer serves the post).
async function fetchSnippets(ids, uriOf) {
  const out = new Map();
  const batches = [];
  for (let i = 0; i < ids.length; i += 25) batches.push(ids.slice(i, i + 25));
  let next = 0;
  async function worker() {
    while (next < batches.length) {
      const b = batches[next++];
      const got = new Map((await getPosts(b.map(uriOf))).map((p) => [p.uri, postText(p)]));
      for (const i of b) { const t = got.get(uriOf(i)); out.set(i, t ? snippet(t) : null); }
    }
  }
  await Promise.all(Array.from({ length: 6 }, worker));
  return out;
}

// Unit vectors for texts, in input order. Batches go in token-length order to cut padding.
async function embed(texts, onBatch) {
  const enc = texts.map((t) => tok.encode(QUERY + clean(t)).ids.slice(0, MAX_TOKENS));
  const order = enc.map((e, i) => i).sort((a, b) => enc[a].length - enc[b].length);
  const out = new Float32Array(texts.length * D);
  for (let s = 0; s < order.length; s += BATCH) {
    const rows = order.slice(s, s + BATCH);
    const B = rows.length, S = Math.max(...rows.map((r) => enc[r].length));
    const ids = new BigInt64Array(B * S), attn = new BigInt64Array(B * S);
    rows.forEach((r, b) => enc[r].forEach((id, j) => { ids[b * S + j] = BigInt(id); attn[b * S + j] = 1n; }));
    const feeds = { input_ids: new ort.Tensor('int64', ids, [B, S]), attention_mask: new ort.Tensor('int64', attn, [B, S]) };
    for (const name of session.inputNames) {
      if (/^(image|video|audio)_features$/.test(name)) feeds[name] = new ort.Tensor('float32', new Float32Array(0), [0, 512]);
    }
    const res = await session.run(feeds, ['sentence_embedding']);
    const e = res.sentence_embedding.data;
    rows.forEach((r, b) => {
      let n = 0;
      for (let k = 0; k < D; k++) n += e[b * D + k] ** 2;
      n = Math.sqrt(n);
      for (let k = 0; k < D; k++) out[r * D + k] = e[b * D + k] / n;
    });
    res.sentence_embedding.dispose?.();
    onBatch?.(B);
  }
  return out;
}

function packShard(header, vecs, statusArr, lens) {
  const n = statusArr.length;
  const hjson = new TextEncoder().encode(JSON.stringify(header));
  const H = Math.ceil(hjson.length / 4) * 4;
  const base = 8 + H;
  const offScale = base + n * D, offStatus = offScale + 4 * n;
  const offLens = offStatus + n + ((offStatus + n) % 2);
  const buf = new ArrayBuffer(offLens + 2 * n);
  const dv = new DataView(buf);
  new Uint8Array(buf, 0, 4).set([69, 71, 81, 84]);   // "EGQT"
  dv.setUint32(4, H, true);
  new Uint8Array(buf, 8, hjson.length).set(hjson);
  for (let i = 8 + hjson.length; i < 8 + H; i++) dv.setUint8(i, 32);   // pad the JSON with spaces
  const codes = new Int8Array(buf, base, n * D);
  for (let i = 0; i < n; i++) {
    let m = 0;
    for (let k = 0; k < D; k++) m = Math.max(m, Math.abs(vecs[i * D + k]));
    const sc = m / 127 || 1;
    dv.setFloat32(offScale + 4 * i, sc, true);
    for (let k = 0; k < D; k++) codes[i * D + k] = Math.round(vecs[i * D + k] / sc);
    dv.setUint8(offStatus + i, statusArr[i]);
    dv.setUint16(offLens + 2 * i, lens[i], true);
  }
  return new Blob([buf], { type: 'application/octet-stream' });
}

function offer(blob, name) {
  const url = URL.createObjectURL(blob);
  const li = document.createElement('li');
  const a = document.createElement('a');
  a.href = url; a.download = name; a.textContent = `${name} (${(blob.size / 1e6).toFixed(1)} MB)`;
  li.append(a);
  $('files').append(li);
  a.click();
}

async function loadModel() {
  const t = now();
  ort = await import(ORT + 'ort.webgpu.min.mjs');
  ort.env.wasm.wasmPaths = ORT;
  ort.env.wasm.numThreads = 1;
  const { Tokenizer } = await import(TOK);
  const [tj, tc] = await Promise.all([
    fetch(HF + 'tokenizer.json').then((r) => r.json()),
    fetch(HF + 'tokenizer_config.json').then((r) => r.json()),
  ]);
  tok = new Tokenizer(tj, tc);
  specials = tj.added_tokens.filter((a) => a.special).map((a) => a.content).sort((x, y) => y.length - x.length);
  session = await ort.InferenceSession.create(HF + 'onnx/model_q4.onnx', {
    executionProviders: ['webgpu'],
    externalData: [{ path: 'model_q4.onnx_data', data: HF + 'onnx/model_q4.onnx_data' }],
    logSeverityLevel: 3,
  });
  const adapter = await navigator.gpu?.requestAdapter();
  const info = adapter?.info || {};
  state.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
  state.load_ms = Math.round(now() - t);
  log(`model loaded in ${(state.load_ms / 1000).toFixed(1)} s on ${info.vendor || '?'} ${info.architecture || ''}`);
}

async function parity() {
  const p = await fetch('parity.json').then((r) => r.json());
  if (Q.get('parity_n')) { p.queries = p.queries.slice(0, +Q.get('parity_n')); p.vectors = p.vectors.slice(0, +Q.get('parity_n')); }
  const t = now();
  const v = await embed(p.queries);
  const cos = p.vectors.map((ref, i) => ref.reduce((s, x, k) => s + x * v[i * D + k], 0));
  state.parity = { n: cos.length, min_cos: Math.min(...cos), mean_cos: cos.reduce((a, b) => a + b, 0) / cos.length,
                   ms: Math.round(now() - t) };
  log(`parity: ${cos.length} queries, cosine to CPU min ${state.parity.min_cos.toFixed(6)} mean ${state.parity.mean_cos.toFixed(6)}`);
  return state.parity.min_cos >= 0.995;
}

async function run() {
  $('run').disabled = true;
  state.phase = 'running';
  try {
    status('Checking parity with the CPU reference…');
    if (!(await parity())) {
      status(`Parity failed (min cosine ${state.parity.min_cos.toFixed(4)}). Stopping; please copy the JSON into issue #102.`, 'bad');
      state.phase = 'parity_failed';
      return;
    }
    const { shard: SHARD, items: all } = await fetch(Q.get('items') || 'items.json').then((r) => r.json());
    const items = Q.get('limit') ? all.slice(0, Number(Q.get('limit'))) : all;
    const meta = await fetch(TAIL + 'index.json').then((r) => r.json());
    const posts = await fetch(TAIL + 'posts.json').then((r) => r.json());
    const uriOf = (i) => `at://${meta.authors[posts[i][0]].did}/app.bsky.feed.post/${posts[i][1]}`;
    const nShards = Math.ceil(items.length / SHARD);
    let done = 0;
    $('bar').max = items.length;
    for (let s = Number(Q.get('shard') || 0); s < nShards; s++) {
      const t0 = now();
      const part = items.slice(s * SHARD, (s + 1) * SHARD);
      const postIds = part.filter((x) => typeof x === 'number');
      status(`Shard ${s + 1}/${nShards}: fetching ${postIds.length} posts from Bluesky…`);
      const snips = await fetchSnippets(postIds, uriOf);
      const fetchMs = Math.round(now() - t0);
      const texts = part.map((x) => (typeof x === 'number' ? snips.get(x) : x));
      const keep = texts.map((t, i) => i).filter((i) => texts[i]);
      status(`Shard ${s + 1}/${nShards}: embedding ${keep.length} texts…`);
      const t1 = now();
      const v = await embed(keep.map((i) => texts[i]), (b) => { done += b; $('bar').value = done; });
      const vecs = new Float32Array(part.length * D);
      keep.forEach((i, j) => vecs.set(v.subarray(j * D, (j + 1) * D), i * D));
      const statusArr = new Uint8Array(part.length);
      keep.forEach((i) => { statusArr[i] = 1; });
      const lens = Uint16Array.from(part, (x, i) => (typeof x === 'number' && texts[i] ? Array.from(texts[i]).length : 0));
      const rec = { shard: s, start: s * SHARD, n: part.length, embedded: keep.length,
                    missing_posts: part.length - keep.length, fetch_ms: fetchMs, embed_ms: Math.round(now() - t1) };
      const header = { ...rec, d: D, prompt: QUERY, max_tokens: MAX_TOKENS, items_total: items.length,
                       limit: Q.get('limit'), adapter: state.adapter, parity: state.parity };
      offer(packShard(header, vecs, statusArr, lens), `eg2_targets_${String(s).padStart(2, '0')}.bin`);
      state.shards.push(rec);
      log(`shard ${s}: ${keep.length}/${part.length} embedded, fetch ${(fetchMs / 1000).toFixed(0)} s, embed ${(rec.embed_ms / 1000).toFixed(0)} s`);
    }
    state.phase = 'done';
    status(`Done: ${state.shards.length} shard file(s) saved to your Downloads folder.`, 'ok');
  } catch (e) {
    state.phase = 'error';
    state.error = String(e?.stack || e);
    status('Error: ' + e, 'bad');
    log(state.error);
  } finally {
    $('copy').disabled = false;
  }
}

$('copy').onclick = () => navigator.clipboard.writeText(JSON.stringify(state, null, 1));
$('run').onclick = run;
loadModel().then(() => {
  state.phase = 'ready';
  $('run').disabled = false;
  $('copy').disabled = false;
  status('Ready. Press Run.');
}).catch((e) => {
  state.phase = 'error';
  state.error = String(e?.stack || e);
  status('Could not load the model: ' + e, 'bad');
});
