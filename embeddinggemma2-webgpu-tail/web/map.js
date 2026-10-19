// 2D / 3D map of the AI list's posts, with a time window. Layout, topics and labels come from
// run/map_bsky.py (UMAP of the EmbeddingGemma 2 embeddings, k-means topics, c-TF-IDF words);
// rendering is deck.gl on WebGL, with the time window applied on the GPU (DataFilterExtension).
// Post text is fetched from Bluesky when a point is clicked; the map files hold none.
const { Deck, OrthographicView, OrbitView, ScatterplotLayer, TextLayer, DataFilterExtension } = deck;

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const DAY = 86400;
const API = 'https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts';
const SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];   // blue 100..700

const st = { view: '2d', colour: 'topics', sel: [], author: null, tEnd: null, width: 0, playing: false, hover: null };
let M, n, xy, xyz, tsec, topic, posts, authors, deckgl;
let pos2, pos3, filt, colours;

async function load() {
  const [map, bin, postList, index] = await Promise.all([
    fetch('bsky/map.json').then((r) => r.json()),
    fetch('bsky/map.bin').then((r) => r.arrayBuffer()),
    fetch('bsky/posts.json').then((r) => r.json()),
    fetch('bsky/index.json').then((r) => r.json()),
  ]);
  M = map; n = map.n; posts = postList; authors = index.authors;
  let o = 0;
  xy = new Float32Array(bin, o, n * 2); o += n * 8;
  xyz = new Float32Array(bin, o, n * 3); o += n * 12;
  tsec = new Uint32Array(bin, o, n); o += n * 4;
  topic = new Uint8Array(bin, o, n);
  const S = 600;           // world units for the [-1, 1] layout
  pos2 = new Float32Array(n * 3);
  pos3 = new Float32Array(n * 3);
  filt = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    pos2[i * 3] = xy[i * 2] * S; pos2[i * 3 + 1] = -xy[i * 2 + 1] * S;
    pos3[i * 3] = xyz[i * 3] * S; pos3[i * 3 + 1] = xyz[i * 3 + 1] * S; pos3[i * 3 + 2] = xyz[i * 3 + 2] * S;
    filt[i] = tsec[i] ? (tsec[i] - M.time_min) / DAY : -1;
  }
  colours = new Uint8Array(n * 4);
  st.tEnd = (M.time_max - M.time_min) / DAY;
}

// ---------------------------------------------------------------- colour

function recolour() {
  const point = [...hex(css('--point')), 110];
  const series = ['--series-1', '--series-2', '--series-3'].map((v) => [...hex(css(v)), 230]);
  // newest = most contrast against the surface: light->dark on a light surface, reversed on dark
  const surf = hex(css('--surface-1'));
  const darkSurface = surf[0] + surf[1] + surf[2] < 384;
  const seq = (darkSurface ? [...SEQ].reverse() : SEQ).map(hex);
  const span = (M.time_max - M.time_min) / DAY;
  for (let i = 0; i < n; i++) {
    let c = point;
    if (st.colour === 'topics') {
      const k = st.sel.indexOf(topic[i]);
      if (k >= 0) c = series[k];
    } else if (st.colour === 'age') {
      const f = filt[i] < 0 ? 0 : filt[i] / span;
      const s = seq[Math.min(seq.length - 1, Math.floor(f * seq.length))];
      c = [...s, 200];
    } else if (st.colour === 'author') {
      if (st.author !== null && posts[i][0] === st.author) c = series[0];
    }
    colours.set(c, i * 4);
  }
}

// ---------------------------------------------------------------- layers

function windowRange() {
  const end = st.tEnd;
  const start = st.width ? end - st.width : -2;
  return [start, end];
}

function layers() {
  const is3 = st.view === '3d';
  const [a, b] = windowRange();
  const soft = st.width ? Math.max(1, st.width * 0.3) : 0;
  const pts = new ScatterplotLayer({
    id: 'posts',
    data: { length: n, attributes: {
      getPosition: { value: is3 ? pos3 : pos2, size: 3 },
      getFillColor: { value: colours, size: 4, normalized: true },
      getFilterValue: { value: filt, size: 1 },
    } },
    radiusUnits: 'pixels', getRadius: 1.7, radiusMinPixels: 1, stroked: false, billboard: true,
    pickable: true, autoHighlight: true, highlightColor: [...hex(css('--text-primary')), 255],
    extensions: [new DataFilterExtension({ filterSize: 1 })],
    filterRange: [a, b], filterSoftRange: soft ? [a + soft, b] : [a, b], filterTransformColor: true, filterTransformSize: true,
    updateTriggers: { getFillColor: [st.colour, st.sel.join(), st.author] },
    onHover: (info) => tip(info), onClick: (info) => showPost(info),
  });
  const labels = new TextLayer({
    id: 'labels',
    data: M.topics.filter((t) => st.colour !== 'topics' || !st.sel.length || st.sel.includes(t.id)),
    getPosition: (t) => is3 ? t.c3.map((v) => v * 600) : [t.c2[0] * 600, -t.c2[1] * 600, 0],
    getText: (t) => t.words.slice(0, 2).join(' · '),
    getSize: 12, sizeUnits: 'pixels', billboard: true,
    getColor: [...hex(css('--text-primary')), 220],
    outlineWidth: 3, outlineColor: [...hex(css('--surface-1')), 230], fontSettings: { sdf: true },
    fontFamily: 'system-ui, sans-serif', getTextAnchor: 'middle', getAlignmentBaseline: 'center',
  });
  return [pts, labels];
}

function render() {
  const is3 = st.view === '3d';
  const view = is3 ? new OrbitView({ id: '3d', orbitAxis: 'Y' }) : new OrthographicView({ id: '2d' });
  const props = { views: view, layers: layers() };
  if (!deckgl) {
    deckgl = new Deck({ parent: $('deck'), ...props, controller: true,
      initialViewState: initialState(is3), getCursor: ({ isHovering }) => (isHovering ? 'pointer' : 'grab') });
  } else {
    deckgl.setProps(props);
  }
  $('hud').textContent = `${countVisible().toLocaleString()} posts in view` + (is3 ? ' · drag to orbit, scroll to zoom' : ' · drag to pan, scroll to zoom');
}

function initialState(is3) {
  return is3 ? { target: [0, 0, 0], rotationX: 15, rotationOrbit: 30, zoom: -0.6, minZoom: -3, maxZoom: 6 }
             : { target: [0, 0, 0], zoom: -0.5, minZoom: -3, maxZoom: 8 };
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

function topicList() {
  const ul = $('topics');
  ul.textContent = '';
  const series = ['--series-1', '--series-2', '--series-3'];
  for (const t of [...M.topics].sort((x, y) => y.size - x.size)) {
    const li = document.createElement('li');
    const k = st.sel.indexOf(t.id);
    li.setAttribute('aria-selected', String(k >= 0));
    const sw = document.createElement('span'); sw.className = 'sw';
    if (k >= 0) sw.style.background = `var(${series[k]})`;
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
  const lines = st.sel.map((id) => M.topics[id].monthly.map((c, j) => (ok[j] ? c / tot[j] : null)));
  const ymax = Math.max(0.05, ...lines.flat().filter((v) => v !== null)) * 1.1;
  const x = (j) => L + (j / Math.max(1, months.length - 1)) * (W - L - R);
  const y = (v) => T + (1 - v / ymax) * (H - T - B);
  for (const v of [0, ymax / 2]) {
    mk('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: 'var(--line)', 'stroke-width': 1 });
    const tx = mk('text', { x: L - 4, y: y(v) + 3, 'text-anchor': 'end' }); tx.textContent = `${Math.round(v * 100)}%`;
  }
  const first = months.findIndex((_, j) => ok[j]);
  for (const j of [first, months.length - 1]) {
    if (j < 0) continue;
    const tx = mk('text', { x: x(j), y: H - 4, 'text-anchor': j === first ? 'start' : 'end' }); tx.textContent = months[j];
  }
  lines.forEach((ln, k) => {
    let d = '', pen = false;
    ln.forEach((v, j) => { if (v === null) { pen = false; return; } d += `${pen ? 'L' : 'M'}${x(j).toFixed(1)},${y(v).toFixed(1)}`; pen = true; });
    mk('path', { d, fill: 'none', stroke: `var(${series[k]})`, 'stroke-width': 2, 'stroke-linejoin': 'round' });
  });
  // the window's end, so the map and the chart read together
  const endMonth = new Date((M.time_min + st.tEnd * DAY) * 1000).toISOString().slice(0, 7);
  const je = months.indexOf(endMonth);
  if (je >= 0) mk('line', { x1: x(je), x2: x(je), y1: T, y2: H - B, stroke: 'var(--text-secondary)', 'stroke-width': 1, 'stroke-dasharray': '2 3' });
  if (!st.sel.length) { const tx = mk('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }); tx.textContent = 'click topics below to compare'; }
  // hover: month, and each selected topic's share
  const hit = mk('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' });
  const tipEl = $('tip');
  hit.addEventListener('mousemove', (ev) => {
    const r = svg.getBoundingClientRect();
    const j = Math.round(((ev.clientX - r.left) * (W / r.width) - L) / (W - L - R) * (months.length - 1));
    if (j < 0 || j >= months.length) return;
    tipEl.textContent = `${months[j]}: ` + (ok[j] ? st.sel.map((id, k) => `${M.topics[id].words[0]} ${((lines[k][j] ?? 0) * 100).toFixed(1)}%`).join(' · ') || `${tot[j]} posts` : `${tot[j]} posts, too few for a share`);
    tipEl.style.display = 'block'; tipEl.style.left = `${ev.clientX - $('deck').getBoundingClientRect().left + 10}px`; tipEl.style.top = `${ev.clientY - 30}px`;
  });
  hit.addEventListener('mouseleave', () => { tipEl.style.display = 'none'; });
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
  $('when').textContent = st.width ? `${fmtDate(M.time_min + Math.max(0, a) * DAY)} – ${fmtDate(M.time_min + b * DAY)}` : `all, to ${fmtDate(M.time_min + b * DAY)}`;
}

function setEnd(days) {
  st.tEnd = days;
  $('tend').value = String(Math.round((days / ((M.time_max - M.time_min) / DAY)) * 1000));
  syncControls(); render(); shareChart();
}

let raf = null;
function play() {
  st.playing = !st.playing;
  $('play').setAttribute('aria-pressed', String(st.playing));
  $('play').textContent = st.playing ? '❚❚ Pause' : '▶ Play';
  if (!st.playing) { cancelAnimationFrame(raf); return; }
  if (!st.width) { st.width = 30; $('twidth').value = '30'; }
  const span = (M.time_max - M.time_min) / DAY;
  // start where the corpus gets dense enough to watch (the first month with 200+ posts)
  const first = M.months.findIndex((_, j) => M.month_totals[j] >= 200);
  const startDay = first > 0 ? (Date.parse(M.months[first] + '-01T00:00:00Z') / 1000 - M.time_min) / DAY : 0;
  if (st.tEnd >= span - 0.5) st.tEnd = startDay + st.width;
  let last = performance.now();
  const step = (t) => {
    const dt = (t - last) / 1000; last = t;
    const next = st.tEnd + dt * 10;            // 10 days per second
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
  $('tend').addEventListener('input', (e) => setEnd((Number(e.target.value) / 1000) * ((M.time_max - M.time_min) / DAY)));
  $('twidth').addEventListener('change', (e) => { st.width = Number(e.target.value); syncControls(); render(); });
  $('play').addEventListener('click', play);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { recolour(); render(); });
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
