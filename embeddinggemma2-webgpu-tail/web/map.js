// 2D / 3D map of the AI list's posts, with a time window. Layout, topics and labels come from
// run/map_bsky.py (UMAP of the EmbeddingGemma 2 embeddings, k-means topics, c-TF-IDF words);
// rendering is deck.gl on WebGL, with the time window applied on the GPU (DataFilterExtension).
// Post text is fetched from Bluesky when a point is clicked; the map files hold none.
const { Deck, OrthographicView, OrbitView, ScatterplotLayer, TextLayer, DataFilterExtension } = deck;
import { RemexTail } from './tail.js';

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const DAY = 86400;
const API = 'https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts';
const SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];   // blue 100..700

const st = { view: '2d', colour: 'topics', sel: [], author: null, tEnd: null, width: 0, playing: false, hover: null };
let M, n, xy, xyz, tsec, topic, posts, authors, deckgl;
let pos2, pos3, filt, ageDays, sortedT, nd, colours, radii;   // filt = a post's rank in time order

async function load() {
  const [map, bin, postList, index] = await Promise.all([
    fetch('bsky/map.json').then((r) => r.json()),
    fetch('bsky/map.u8.txt').then((r) => { if (!r.ok) throw new Error(`bsky/map.u8.txt: HTTP ${r.status}`); return r.arrayBuffer(); }),
    fetch('bsky/posts.json').then((r) => r.json()),
    fetch('bsky/index.json').then((r) => r.json()),
  ]);
  M = map; n = map.n; posts = postList; authors = index.authors;
  if (bin.byteLength !== n * 25) throw new Error(`map data is ${bin.byteLength} bytes, expected ${n * 25} for ${n} posts`);
  if (posts.length !== n) throw new Error(`posts.json has ${posts.length} posts, map.json ${n}`);
  let o = 0;
  xy = new Float32Array(bin, o, n * 2); o += n * 8;
  xyz = new Float32Array(bin, o, n * 3); o += n * 12;
  tsec = new Uint32Array(bin, o, n); o += n * 4;
  topic = new Uint8Array(bin, o, n);
  // robust scaling: each axis's 1st..99th percentile maps to [-S, S]; a few far-out posts would
  // otherwise shrink the map to a dot. Outliers are pinned 10% past the edge.
  pos2 = new Float32Array(n * 3);
  pos3 = new Float32Array(n * 3);
  filt = new Float32Array(n);
  scaleAxes(xy, 2, pos2, [1, -1]);
  scaleAxes(xyz, 3, pos3, [1, 1, 1]);
  // The time window counts posts, not days: posts are rank-ordered by time and a window is N
  // consecutive ranks, so every frame holds the same number of posts however sparse the month.
  ageDays = new Float32Array(n);
  const dated = [];
  for (let i = 0; i < n; i++) {
    ageDays[i] = tsec[i] ? (tsec[i] - M.time_min) / DAY : -1;
    if (tsec[i]) dated.push(i);
  }
  dated.sort((x, y) => tsec[x] - tsec[y]);
  nd = dated.length;
  sortedT = new Uint32Array(nd);
  filt.fill(-1);
  dated.forEach((i, r) => { filt[i] = r; sortedT[r] = tsec[i]; });
  scaleCentroids();
  colours = new Uint8Array(n * 4);
  radii = new Float32Array(n).fill(1.7);
  st.tEnd = nd - 1;
  st.width = 0;
}

const S = 600;             // world units for the scaled layout
const fits = {};
function scaleAxes(src, dims, dst, flip) {
  const f = [];
  for (let d = 0; d < dims; d++) {
    const v = Array.from({ length: n }, (_, i) => src[i * dims + d]).sort((a, b) => a - b);
    const lo = v[Math.floor(n * 0.01)], hi = v[Math.floor(n * 0.99)];
    const mid = (lo + hi) / 2, half = (hi - lo) / 2;
    f.push([mid, half]);
    for (let i = 0; i < n; i++) {
      const u = Math.max(-1.1, Math.min(1.1, (src[i * dims + d] - mid) / half));
      dst[i * 3 + d] = u * S * flip[d];
    }
  }
  fits[dims] = { f, flip };
}
function scaleCentroids() {
  for (const t of M.topics) {
    t.p2 = [0, 1].map((d) => ((t.c2[d] - fits[2].f[d][0]) / fits[2].f[d][1]) * S * fits[2].flip[d]).concat(0);
    t.p3 = [0, 1, 2].map((d) => ((t.c3[d] - fits[3].f[d][0]) / fits[3].f[d][1]) * S * fits[3].flip[d]);
  }
}

// ---------------------------------------------------------------- colour

// OKLCH -> sRGB bytes (Björn Ottosson's OKLab)
function oklch(L, C, h) {
  const a = C * Math.cos(h), b = C * Math.sin(h);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  const lin = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
               -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
               -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
  return lin.map((v) => {
    const c = Math.max(0, Math.min(1, v));
    return Math.round(255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055));
  });
}

// Each topic's hue is the angle of its centre around the middle of the 2D map, so neighbouring
// regions get neighbouring hues and the colour repeats the position. Identity is carried by the
// labels, not by telling hues apart.
function topicHues(darkSurface) {
  const L = darkSurface ? 0.72 : 0.62, C = 0.12;
  return M.topics.map((t) => oklch(L, C, Math.atan2(t.p2[1], t.p2[0])));
}

function recolour() {
  const surf0 = hex(css('--surface-1'));
  const hues = topicHues(surf0[0] + surf0[1] + surf0[2] < 384);
  const series = ['--series-1', '--series-2', '--series-3'].map((v) => [...hex(css(v)), 235]);
  // newest = most contrast against the surface: light->dark on a light surface, reversed on dark
  const surf = hex(css('--surface-1'));
  const darkSurface = surf[0] + surf[1] + surf[2] < 384;
  const seq = (darkSurface ? [...SEQ].reverse() : SEQ).map(hex);
  const span = (M.time_max - M.time_min) / DAY;
  const dim = st.sel.length ? 38 : 120;        // alpha of unselected topics
  radii.fill(1.7);
  if (st.colour === 'search' && Q.rank) {
    // relevance on one sequential ramp: the strongest matches carry the most contrast and size;
    // posts outside the top HITS fade back in their topic hue
    for (let i = 0; i < n; i++) {
      const r = Q.rank[i];
      let c;
      if (r < HITS) {
        const f = 1 - r / HITS;                         // 1 = best
        const s = seq[Math.min(seq.length - 1, Math.floor((0.35 + 0.65 * f) * seq.length))];
        c = [...s, 160 + Math.round(95 * f)];
        radii[i] = r < 10 ? 4.5 : 1.9 + 1.6 * f;
      } else {
        c = [...hues[topic[i]], 16];
      }
      colours.set(c, i * 4);
    }
    return;
  }
  for (let i = 0; i < n; i++) {
    let c = [...hues[topic[i]], dim];
    if (st.colour === 'topics') {
      const k = st.sel.indexOf(topic[i]);
      if (k >= 0) c = series[k];
    } else if (st.colour === 'age') {
      const f = ageDays[i] < 0 ? 0 : ageDays[i] / span;
      const s = seq[Math.min(seq.length - 1, Math.floor(f * seq.length))];
      c = [...s, 200];
    } else if (st.colour === 'author') {
      c = st.author !== null && posts[i][0] === st.author ? series[0] : [...hues[topic[i]], 30];
    }
    colours.set(c, i * 4);
  }
}

// ---------------------------------------------------------------- layers

function windowRange() {
  const end = st.tEnd;
  const start = st.width ? end - st.width : -2;      // -2 also admits undated posts (rank -1)
  return [start, end];
}

function layers() {
  const is3 = st.view === '3d';
  const [a, b] = windowRange();
  const soft = st.width ? Math.max(1, st.width * 0.15) : 0;   // the newest 15% of the window fades in
  const pts = new ScatterplotLayer({
    id: 'posts',
    data: { length: n, attributes: {
      getPosition: { value: is3 ? pos3 : pos2, size: 3 },
      getFillColor: { value: colours, size: 4, normalized: true },
      getRadius: { value: radii, size: 1 },
      getFilterValue: { value: filt, size: 1 },
    } },
    radiusUnits: 'pixels', radiusMinPixels: 1, stroked: false, billboard: true,
    pickable: true, autoHighlight: true, highlightColor: [...hex(css('--text-primary')), 255],
    extensions: [new DataFilterExtension({ filterSize: 1 })],
    filterRange: [a, b], filterSoftRange: soft ? [a + soft, b] : [a, b], filterTransformColor: true, filterTransformSize: true,
    updateTriggers: { getFillColor: [st.colour, st.sel.join(), st.author, Q.version], getRadius: [Q.version, st.colour] },
    onHover: (info) => tip(info), onClick: (info) => showPost(info),
  });
  const labels = new TextLayer({
    id: 'labels',
    data: M.topics.filter((t) => st.colour === 'search' ? (Q.topicHits?.[t.id] ?? 0) >= 0.05 * HITS
      : st.colour !== 'topics' || !st.sel.length || st.sel.includes(t.id)),
    getPosition: (t) => (is3 ? t.p3 : t.p2),
    getText: (t) => t.words.slice(0, 2).join(' · '),
    getSize: 11.5, sizeUnits: 'pixels', billboard: true,
    getColor: [...hex(css('--text-primary')), 220],
    outlineWidth: 3, outlineColor: [...hex(css('--surface-1')), 230], fontSettings: { sdf: true },
    fontFamily: 'system-ui, sans-serif', fontWeight: 600, characterSet: 'auto',   // '·' is outside the default ASCII atlas
    getTextAnchor: 'middle', getAlignmentBaseline: 'center',
  });
  return [pts, labels];
}

function render() {
  const is3 = st.view === '3d';
  const view = is3 ? new OrbitView({ id: '3d', orbitAxis: 'Y' }) : new OrthographicView({ id: '2d' });
  const props = { views: view, layers: layers() };
  if (!deckgl) {
    deckgl = new Deck({ parent: $('deck'), ...props, controller: true,
      onViewStateChange: ({ viewState }) => { st.vs = viewState; },
      initialViewState: initialState(is3), getCursor: ({ isHovering }) => (isHovering ? 'pointer' : 'grab') });
  } else {
    deckgl.setProps(props);
  }
  $('hud').textContent = `${countVisible().toLocaleString()} posts in view` + (is3 ? ' · drag to orbit, scroll to zoom' : ' · drag to pan, scroll to zoom');
}

function initialState(is3) {
  const fit = Math.log2(Math.min($('deck').clientWidth, $('deck').clientHeight) / (2.4 * S));
  return is3 ? { target: [0, 0, 0], rotationX: 15, rotationOrbit: 30, zoom: fit - 0.6, minZoom: -4, maxZoom: 6 }
             : { target: [0, 0, 0], zoom: fit, minZoom: -4, maxZoom: 8 };
}

function switchView(v) {
  st.view = v;
  deckgl.setProps({ initialViewState: { ...initialState(v === '3d'), transitionDuration: 0 } });
  render();
}

function countVisible() {
  const [a, b] = windowRange();
  let c = 0;
  for (let i = 0; i < n; i++) if (filt[i] >= a && filt[i] <= b) c++;
  return c;
}

// ---------------------------------------------------------------- hover, click

function fmtDate(sec) { return new Date(sec * 1000).toISOString().slice(0, 10); }

function tip(info) {
  const el = $('tip');
  if (info.index < 0 || info.layer?.id !== 'posts') { el.style.display = 'none'; return; }
  const i = info.index;
  const a = authors[posts[i][0]];
  el.textContent = '';
  const l1 = document.createElement('div'); l1.textContent = `@${a.handle} · ${tsec[i] ? fmtDate(tsec[i]) : ''}`;
  const l2 = document.createElement('div'); l2.className = 'muted'; l2.textContent = M.topics[topic[i]].label;
  el.append(l1, l2);
  el.style.left = `${info.x + 12}px`; el.style.top = `${info.y + 12}px`; el.style.display = 'block';
}

async function showPost(info) {
  if (info.index < 0 || info.layer?.id !== 'posts') return;
  const i = info.index;
  const [ai, rkey] = posts[i];
  const a = authors[ai];
  const box = $('post');
  box.style.display = 'block';
  box.textContent = 'Loading post…';
  const uri = `at://${a.did}/app.bsky.feed.post/${rkey}`;
  let text = '(post no longer available)', name = '';
  try {
    const d = await fetch(`${API}?uris=${encodeURIComponent(uri)}`).then((r) => r.json());
    if (d.posts?.[0]) { text = d.posts[0].record.text; name = d.posts[0].author.displayName || ''; }
  } catch { text = '(Bluesky did not answer)'; }
  box.textContent = '';
  const head = document.createElement('div');
  const link = document.createElement('a');
  link.href = `https://bsky.app/profile/${a.handle}/post/${rkey}`; link.target = '_blank';
  link.textContent = `${name || '@' + a.handle} · ${tsec[i] ? fmtDate(tsec[i]) : ''}`;
  const t = document.createElement('div'); t.className = 'text'; t.textContent = text;
  const tp = document.createElement('div'); tp.className = 'muted'; tp.textContent = `topic: ${M.topics[topic[i]].label}`;
  const btn = document.createElement('button'); btn.textContent = `Colour @${a.handle}'s posts`;
  btn.style.cssText = 'margin-top:8px;border:1px solid var(--line);background:transparent;color:var(--accent);border-radius:6px;padding:4px 8px;cursor:pointer';
  btn.onclick = () => { st.colour = 'author'; st.author = ai; syncControls(); recolour(); render(); };
  head.append(link);
  box.append(head, t, tp, btn);
}

// ---------------------------------------------------------------- topics list and share chart

let HUES = [];
function topicList() {
  const surf0 = hex(css('--surface-1'));
  HUES = topicHues(surf0[0] + surf0[1] + surf0[2] < 384);
  const ul = $('topics');
  ul.textContent = '';
  const series = ['--series-1', '--series-2', '--series-3'];
  for (const t of [...M.topics].sort((x, y) => y.size - x.size)) {
    const li = document.createElement('li');
    const k = st.sel.indexOf(t.id);
    li.setAttribute('aria-selected', String(k >= 0));
    const sw = document.createElement('span'); sw.className = 'sw';
    sw.style.background = k >= 0 ? `var(${series[k]})` : `rgb(${HUES[t.id].join(',')})`;
    const name = document.createElement('span'); name.textContent = t.label;
    const cnt = document.createElement('span'); cnt.className = 'n'; cnt.textContent = t.size.toLocaleString();
    li.append(sw, name, cnt);
    li.onclick = () => {
      if (k >= 0) st.sel.splice(k, 1);
      else if (st.sel.length < 3) st.sel.push(t.id);
      else { st.sel.shift(); st.sel.push(t.id); }
      st.colour = 'topics'; syncControls(); recolour(); topicList(); shareChart(); render();
    };
    ul.append(li);
  }
}

// share of each selected topic among all posts that month; one axis (percent), 2px lines
function shareChart() {
  const svg = $('share');
  const W = svg.clientWidth || 270, H = 120, L = 30, R = 6, T = 8, B = 18;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.textContent = '';
  const ns = 'http://www.w3.org/2000/svg';
  const mk = (tag, attrs) => { const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.append(e); return e; };
  const months = M.months, tot = M.month_totals;
  // months with too few posts to give a share are skipped
  const ok = months.map((_, j) => tot[j] >= 200);
  const series = ['--series-1', '--series-2', '--series-3'];
  const searching = st.colour === 'search' && Q.rank;
  let qMonthly = null;
  if (searching) {
    const mi = new Map(months.map((m, j) => [m, j]));
    qMonthly = new Array(months.length).fill(0);
    for (let k = 0; k < HITS; k++) { const i = Q.order[k]; if (tsec[i]) qMonthly[mi.get(fmtDate(tsec[i]).slice(0, 7))]++; }
  }
  const lines = searching ? [qMonthly.map((c, j) => (ok[j] ? c / tot[j] : null))]
    : st.sel.map((id) => M.topics[id].monthly.map((c, j) => (ok[j] ? c / tot[j] : null)));
  const ymax = Math.max(0.05, ...lines.flat().filter((v) => v !== null)) * 1.1;
  // from the month after which every month has enough posts
  let j0 = months.length - 1;
  while (j0 > 0 && ok[j0 - 1]) j0--;
  const x = (j) => L + ((j - j0) / Math.max(1, months.length - 1 - j0)) * (W - L - R);
  const y = (v) => T + (1 - v / ymax) * (H - T - B);
  for (const v of [0, ymax / 2]) {
    mk('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: 'var(--line)', 'stroke-width': 1 });
    const tx = mk('text', { x: L - 4, y: y(v) + 3, 'text-anchor': 'end' }); tx.textContent = `${Math.round(v * 100)}%`;
  }
  const first = j0;
  for (const j of [first, months.length - 1]) {
    if (j < 0) continue;
    const tx = mk('text', { x: x(j), y: H - 4, 'text-anchor': j === first ? 'start' : 'end' }); tx.textContent = months[j];
  }
  lines.forEach((ln, k) => {
    let d = '', pen = false;
    ln.forEach((v, j) => { if (v === null || j < j0) { pen = false; return; } d += `${pen ? 'L' : 'M'}${x(j).toFixed(1)},${y(v).toFixed(1)}`; pen = true; });
    mk('path', { d, fill: 'none', stroke: `var(${series[k]})`, 'stroke-width': 2, 'stroke-linejoin': 'round' });
  });
  // the window's end, so the map and the chart read together
  const endMonth = new Date(rankTime(st.tEnd) * 1000).toISOString().slice(0, 7);
  const je = months.indexOf(endMonth);
  if (je >= 0) mk('line', { x1: x(je), x2: x(je), y1: T, y2: H - B, stroke: 'var(--text-secondary)', 'stroke-width': 1, 'stroke-dasharray': '2 3' });
  if (!searching && !st.sel.length) { const tx = mk('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }); tx.textContent = 'click topics below to compare'; }
  if (searching) { const tx = mk('text', { x: L + 4, y: T + 9 }); tx.textContent = `top ${HITS} matches, share of each month's posts`; }
  // hover: month, and each selected topic's share
  const hit = mk('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' });
  const tipEl = $('tip');
  hit.addEventListener('mousemove', (ev) => {
    const r = svg.getBoundingClientRect();
    const j = j0 + Math.round(((ev.clientX - r.left) * (W / r.width) - L) / (W - L - R) * (months.length - 1 - j0));
    if (j < j0 || j >= months.length) return;
    tipEl.textContent = `${months[j]}: ` + (ok[j] && searching ? `${qMonthly[j]} of the top ${HITS} matches, ${((lines[0][j] ?? 0) * 100).toFixed(2)}% of ${tot[j]} posts`
      : ok[j] ? st.sel.map((id, k) => `${M.topics[id].words[0]} ${((lines[k][j] ?? 0) * 100).toFixed(1)}%`).join(' · ') || `${tot[j]} posts` : `${tot[j]} posts, too few for a share`);
    tipEl.style.display = 'block'; tipEl.style.left = `${ev.clientX - $('deck').getBoundingClientRect().left + 10}px`; tipEl.style.top = `${ev.clientY - 30}px`;
  });
  hit.addEventListener('mouseleave', () => { tipEl.style.display = 'none'; });
}


// ---------------------------------------------------------------- search

// Queries are embedded by the hosted EG2 embedder (the same q4 graph and prompt that embedded the
// posts) and scored against all posts' 2-bit codes with RemexTail on WebGPU. Every post's score is
// read back, so the whole map can light up, not just the top ten.
const EMBEDDER = 'https://eg2-query-embedder.austegard.workers.dev/';
const HITS = 500;
const Q = { version: 0, rank: null, scores: null, topicHits: null, text: '' };
let tail = null, corpus = null, gdevice = null, qbuf = null, searchReady = null, warmed = false;

function initSearch() {
  if (searchReady) return searchReady;
  if (!warmed) { warmed = true; fetch(EMBEDDER + 'warm', { method: 'POST' }).catch(() => {}); }
  searchReady = (async () => {
    if (!navigator.gpu) throw new Error('this browser has no WebGPU, so the scan cannot run');
    $('qstatus').textContent = 'Loading the 2-bit index onto the GPU…';
    const index = await fetch('bsky/index.json').then((r) => r.json());
    const [codes, aux, params] = await Promise.all([
      fetch('bsky/' + index.files.codes).then((r) => r.arrayBuffer()),
      fetch('bsky/' + index.files.aux).then((r) => r.arrayBuffer()),
      fetch('assets/params.json').then((r) => r.json()),
    ]);
    if (index.n !== n) throw new Error(`index has ${index.n} posts, the map ${n}`);
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    gdevice = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    tail = await RemexTail.create(gdevice, { rht: params.rht, mu: new Float32Array(index.mu),
      boundaries: params.boundaries, centroids: params.centroids, kappa: params.kappa });
    corpus = tail.allocCorpus(n);
    tail.loadCodes(corpus, new Uint8Array(codes), new Float32Array(aux), 0);
    qbuf = gdevice.createBuffer({ size: 768 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    await gdevice.queue.onSubmittedWorkDone();
    $('qstatus').textContent = 'Ready.';
  })();
  return searchReady;
}

async function runSearch(text) {
  const t0 = performance.now();
  await initSearch();
  $('qstatus').textContent = 'Embedding the query…';
  const r = await fetch(EMBEDDER + 'embed', { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text: text.slice(0, 512), dims: 768 }) });
  if (!r.ok) throw new Error(r.status === 429 ? 'the query embedder is rate-limited (about 20 a minute); try again shortly'
                                              : `the query embedder answered HTTP ${r.status}`);
  const d = await r.json();
  const t1 = performance.now();
  gdevice.queue.writeBuffer(qbuf, 0, new Float32Array(d.embedding));
  const sb = tail.search(qbuf, 1, corpus, { n });
  const scores = await tail.readScores(sb, 1, corpus, n);
  sb.destroy();
  const t2 = performance.now();
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const rank = new Uint32Array(n);
  order.forEach((i, k) => { rank[i] = k; });
  const topicHits = new Array(M.topics.length).fill(0);
  for (let k = 0; k < HITS; k++) topicHits[topic[order[k]]]++;
  Object.assign(Q, { version: Q.version + 1, rank, scores, order, topicHits, text });
  st.colour = 'search';
  syncControls(); recolour(); render(); shareChart();
  queryTopics(); flyToHits(order);
  $('qstatus').textContent = `embed ${(t1 - t0).toFixed(0)} ms · scan of ${n.toLocaleString()} on the GPU ${(t2 - t1).toFixed(1)} ms` +
    (d.cached ? ' (cached query)' : '');
  await queryHits(order.slice(0, 10));
}

function queryTopics() {
  const box = $('qtopics');
  box.textContent = '';
  const ranked = M.topics.map((t) => [t, Q.topicHits[t.id]]).filter(([, c]) => c > 0).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const head = document.createElement('div'); head.className = 'muted'; head.style.cssText = 'font-size:12px;margin-top:8px';
  head.textContent = `Topics of the top ${HITS} matches`;
  box.append(head);
  for (const [t, c] of ranked) {
    const row = document.createElement('div'); row.className = 'qt';
    const name = document.createElement('span'); name.textContent = t.label;
    const barWrap = document.createElement('span');
    const bar = document.createElement('div'); bar.className = 'bar'; bar.style.width = `${(c / ranked[0][1]) * 100}%`;
    barWrap.append(bar);
    const pct = document.createElement('span'); pct.className = 'pct'; pct.textContent = `${Math.round((c / HITS) * 100)}%`;
    row.append(name, barWrap, pct);
    box.append(row);
  }
}

async function queryHits(idx) {
  const ol = $('qhits');
  ol.textContent = '';
  const uris = idx.map((i) => `at://${authors[posts[i][0]].did}/app.bsky.feed.post/${posts[i][1]}`);
  let byUri = new Map();
  try {
    const d = await fetch(API + '?' + uris.map((u) => 'uris=' + encodeURIComponent(u)).join('&')).then((r) => r.json());
    byUri = new Map(d.posts.map((p) => [p.uri, p]));
  } catch { /* text stays blank */ }
  idx.forEach((i, k) => {
    const p = byUri.get(uris[k]);
    const li = document.createElement('li');
    const who = document.createElement('div'); who.className = 'who';
    who.textContent = `${p?.author?.displayName || '@' + authors[posts[i][0]].handle} · ${tsec[i] ? fmtDate(tsec[i]) : ''} · ${Q.scores[i].toFixed(3)}`;
    const t = document.createElement('div'); t.className = 't'; t.textContent = p ? p.record.text : '(post no longer available)';
    li.append(who, t);
    li.onclick = () => flyTo(i);
    ol.append(li);
  });
}

function centreOf(ids) {
  const pos = st.view === '3d' ? pos3 : pos2;
  const c = [0, 0, 0];
  for (const i of ids) for (let d = 0; d < 3; d++) c[d] += pos[i * 3 + d] / ids.length;
  return c;
}
function flyToHits(order) { flyTarget(centreOf(order.slice(0, 50)), 0.6); }
function flyTo(i) { flyTarget(centreOf([i]), 2); }
function flyTarget(target, dz) {
  const base = initialState(st.view === '3d');
  const cur = st.vs || base;
  deckgl.setProps({ initialViewState: { ...cur, target, zoom: base.zoom + dz, transitionDuration: 900 } });
}

function clearSearch() {
  Object.assign(Q, { version: Q.version + 1, rank: null, scores: null, order: null, topicHits: null, text: '' });
  $('qtopics').textContent = ''; $('qhits').textContent = ''; $('qstatus').textContent = '';
  st.colour = 'topics';
  syncControls(); recolour(); render(); shareChart();
}

// ---------------------------------------------------------------- controls

function syncControls() {
  for (const b of $('view').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.v === st.view));
  for (const b of $('colour').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.c === st.colour));
  $('author').hidden = st.colour !== 'author';
  $('agekey').hidden = st.colour !== 'age';
  if (st.colour === 'age') {
    const surf = hex(css('--surface-1'));
    const ramp = surf[0] + surf[1] + surf[2] < 384 ? [...SEQ].reverse() : SEQ;
    $('agebar').style.background = `linear-gradient(90deg, ${ramp.join(', ')})`;
    $('age0').textContent = `older (${fmtDate(M.time_min)})`;
    $('age1').textContent = `newer (${fmtDate(M.time_max)})`;
  }
  if (st.author !== null) $('author').value = String(st.author);
  const [a, b] = windowRange();
  $('when').textContent = st.width ? `${fmtDate(rankTime(a))} – ${fmtDate(rankTime(b))}` : `all, to ${fmtDate(rankTime(b))}`;
}

function rankTime(r) { return sortedT[Math.max(0, Math.min(nd - 1, Math.round(r)))]; }

function setEnd(rank) {
  st.tEnd = Math.max(st.width ? Math.min(st.width, nd - 1) : 0, Math.min(nd - 1, rank));
  $('tend').value = String(Math.round((st.tEnd / (nd - 1)) * 1000));
  syncControls(); render(); shareChart();
}

let raf = null;
function play() {
  st.playing = !st.playing;
  $('play').setAttribute('aria-pressed', String(st.playing));
  $('play').textContent = st.playing ? '❚❚ Pause' : '▶ Play';
  if (!st.playing) { cancelAnimationFrame(raf); return; }
  if (!st.width) { st.width = 5000; $('twidth').value = '5000'; }
  const span = nd - 1;
  if (st.tEnd >= span - 0.5) st.tEnd = st.width;
  let last = performance.now();
  const step = (t) => {
    const dt = (t - last) / 1000; last = t;
    const next = st.tEnd + dt * span / 60;     // the whole corpus in about a minute
    if (next >= span) { setEnd(span); play(); return; }
    setEnd(next);
    raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
}

function wire() {
  $('view').addEventListener('click', (e) => { const v = e.target.dataset?.v; if (v && v !== st.view) { switchView(v); syncControls(); } });
  $('colour').addEventListener('click', (e) => {
    const c = e.target.dataset?.c; if (!c) return;
    st.colour = c;
    if (c === 'author' && st.author === null) st.author = Number($('author').value);
    syncControls(); recolour(); render();
  });
  const sel = $('author');
  const counts = new Map();
  for (const [a] of posts) counts.set(a, (counts.get(a) || 0) + 1);
  for (const [a, c] of [...counts].sort((x, y) => y[1] - x[1])) {
    const o = document.createElement('option'); o.value = String(a); o.textContent = `@${authors[a].handle} (${c})`; sel.append(o);
  }
  sel.addEventListener('change', () => { st.author = Number(sel.value); recolour(); render(); });
  $('tend').addEventListener('input', (e) => setEnd((Number(e.target.value) / 1000) * (nd - 1)));
  $('twidth').addEventListener('change', (e) => { st.width = Number(e.target.value); setEnd(st.tEnd); });
  $('play').addEventListener('click', play);
  $('q').addEventListener('focus', () => { initSearch().catch((e) => { $('qstatus').textContent = e.message; }); });
  $('q').addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const v = e.target.value.trim();
    if (!v) { clearSearch(); return; }
    runSearch(v).catch((err) => { $('qstatus').textContent = `Search failed: ${err.message}`; });
  });
  $('q').addEventListener('search', (e) => { if (!e.target.value.trim()) clearSearch(); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { recolour(); topicList(); render(); });
}

(async () => {
  try {
    await load();
    $('summary').textContent = `${n.toLocaleString()} posts by ${authors.length} members of the Artificial (and real) Intelligence list, ` +
      `${fmtDate(M.time_min)} to ${fmtDate(M.time_max)}. Nearby points are posts EmbeddingGemma 2 places near each other; ` +
      `the layout is UMAP, the topics k-means with their most distinctive words.`;
    $('bias').textContent = `The corpus is each member's newest 200 posts, so it thins out going back: ` +
      `a third of it is from the last two months, and prolific posters only reach back weeks. ` +
      `The chart shows each topic's share of a month's posts, not counts, but who was posting still shapes it.`;
    recolour();
    wire();
    topicList();
    syncControls();
    render();
    shareChart();
  } catch (err) {
    $('summary').textContent = `Error: ${err.message}`;
    console.error(err);
  }
})();
