// Demo page for the browser NER runtime (ner.js): text mode, Bluesky mode and firehose mode, plus link
// parameters (see the "Link to this" section of index.html). window.app exposes the state for
// run/page_smoke.mjs and run/page_smoke2.mjs.
import { NER } from "./ner.js";
import { resolveSource, fetchPosts, EntityGroups, normEntity } from "./bsky.js";
import { Jetstream, makeFilter, makeSampler } from "./firehose.js";

const MODEL = "models/gliner-bi-edge-v2.0-t4-w8L63-h8/";
const DEFAULT_LABELS = ["person", "company", "location", "product", "date"];
const DEFAULT_TEXT = "Satya Nadella flew to Lisbon on 4 November to open Microsoft's new AI hub, two weeks after Google " +
  "showed its Pixel 10 phones in New York. Portugal's prime minister, Luís Montenegro, called it the biggest tech investment " +
  "the country has seen since Web Summit moved there in 2016.";
const NCOL = 8;
const FH = { batch: 64, waitMs: 250, maxQueue: 2000, renderMs: 500, hiddenStopMs: 60_000, detailCap: 50 };
const FH_DEFAULTS = { sample: "1", lang: "en", replies: false, minlen: 20, include: "", exclude: "", window: "1000" };

const $ = (id) => document.getElementById(id);
const app = window.app = { state: "starting", ner: null, labels: [...DEFAULT_LABELS], threshold: 0.5, last: null, bsky: null, fh: null, error: null,
  mode: "text", pending: null, linkText: null, copied: null };
const fmtMs = (x) => (x < 10 ? x.toFixed(1) : Math.round(x).toLocaleString("en-US")) + " ms";
const fmtMB = (b) => (b / 1e6).toFixed(1) + " MB";
const fmtN = (x) => x.toLocaleString("en-US");
const fmtRate = (x) => (x < 10 ? x.toFixed(1) : fmtN(Math.round(x)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- status line ----
function status(text, kind = "", progress = null) {
  const s = $("status");
  s.className = "status " + kind;
  $("status-text").textContent = text;
  const bar = $("status-bar");
  bar.hidden = progress === null;
  if (progress !== null) bar.value = progress;
}

// ---- one model call at a time (the engine has one batch in flight) ----
let chain = Promise.resolve();
const locked = (fn) => { const p = chain.then(fn, fn); chain = p.catch(() => {}); return p; };

// ---- labels ----
const colourOf = (label) => { const i = app.labels.indexOf(label); return i < 0 ? 0 : i % NCOL; };
const colourStyle = (el, label) => { const c = colourOf(label); el.style.setProperty("--cc", `var(--c${c})`); el.style.setProperty("--ce", `var(--e${c})`); };
const cleanLabels = (list) => [...new Set(list.map((s) => s.trim().replace(/\s+/g, " ").slice(0, 60)).filter(Boolean))];

function renderChips() {
  const box = $("chips");
  box.replaceChildren(...app.labels.map((l) => {
    const chip = document.createElement("span");
    chip.className = "chip"; colourStyle(chip, l);
    const t = document.createElement("span"); t.textContent = l;
    const x = document.createElement("button");
    x.type = "button"; x.textContent = "×"; x.setAttribute("aria-label", `remove ${l}`);
    x.onclick = () => { app.labels = app.labels.filter((y) => y !== l); labelsChanged(); };
    chip.append(t, x);
    return chip;
  }));
  if (!app.labels.length) box.innerHTML = '<span class="empty">no entity types: add one below</span>';
  const legend = $("legend");
  legend.replaceChildren(...app.labels.map((l) => {
    const s = document.createElement("span"), i = document.createElement("i");
    colourStyle(i, l); s.append(i, document.createTextNode(l));
    return s;
  }));
}

function addLabels(text) {
  const fresh = cleanLabels(text.split(",")).filter((s) => !app.labels.includes(s));
  if (!fresh.length) return;
  app.labels.push(...fresh.slice(0, 64 - app.labels.length));
  labelsChanged();
}

function labelsChanged() {
  renderChips(); schedule(); syncUrl();
  if (app.fh && app.fh.groups.size) fhClearWindow("window cleared: the entity types changed");
}

// ---- highlighting ----
// text + entities ({ u16Start, u16End, label }) -> nodes; tag: show the label after each entity;
// only: a predicate picking which entities to mark
function highlight(text, ents, { tag = true, only = null } = {}) {
  const out = [];
  let at = 0;
  for (const e of [...ents].sort((a, b) => a.u16Start - b.u16Start)) {
    if (only && !only(e)) continue;
    if (e.u16Start < at) continue;
    if (e.u16Start > at) out.push(document.createTextNode(text.slice(at, e.u16Start)));
    const m = document.createElement("mark");
    m.className = "ent"; m.dataset.label = e.label; colourStyle(m, e.label);
    m.title = `${e.label} (${e.score.toFixed(2)})`;
    m.append(document.createTextNode(text.slice(e.u16Start, e.u16End)));
    if (tag) { const t = document.createElement("span"); t.className = "tag"; t.textContent = e.label; m.append(t); }
    out.push(m);
    at = e.u16End;
  }
  if (at < text.length) out.push(document.createTextNode(text.slice(at)));
  return out;
}

// ---- text mode ----
let timer = null;
function schedule() {
  clearTimeout(timer);
  timer = setTimeout(() => {
    if (app.state !== "ready") return;
    if (app.mode === "text") runText();
    else if (app.mode === "bsky" && app.bsky?.posts?.length && !app.bsky.running) reanalyse();
  }, 250);
}

async function runText() {
  const text = $("text").value;
  if (!app.labels.length) { $("out").replaceChildren(text); $("facts").textContent = "Add an entity type to search for."; return; }
  try {
    const { ents, tm, lt } = await locked(async () => {
      const t0 = performance.now();
      const lt = await app.ner.setLabels(app.labels);
      const ents = await app.ner.extract(text, { threshold: app.threshold, truncate: true });
      return { ents, tm: { ...app.ner.lastTimings, wallMs: performance.now() - t0 }, lt };
    });
    app.last = { text, ents, timings: tm, labelTimings: lt };
    $("out").replaceChildren(...highlight(text, ents));
    if (!ents.length) $("out").append(Object.assign(document.createElement("div"), { className: "empty", textContent: "No entities above the threshold." }));
    const enc = lt.encoded ? `entity types ${fmtMs(lt.totalMs)} (${lt.encoded} new) · ` : "";
    $("facts").textContent = `${app.modelFacts} · last run, ${tm.words} words × ${tm.labels} types: ${enc}tokenise ${fmtMs(tm.tokenizeMs)} · ` +
      `GPU (encoder + span head) ${fmtMs(tm.textEncoderGpuMs + tm.readbackMs)} · JavaScript head ${fmtMs(tm.spanHeadMs + tm.decodeMs)} · total ${fmtMs(tm.wallMs)}`;
  } catch (e) { fail(e); }
}

// ---- grouped view, shared by Bluesky mode and firehose mode ----
// ui: { st() -> { groups, expanded, selected, view }, groupsEl, detailEl, postsOf(st, entity) -> [post], meta(post) -> [node],
//       unit, emptyText }
function renderGroups(ui, final = false) {
  const b = ui.st();
  const view = b.groups.view(app.labels);
  b.view = view;
  const box = $(ui.groupsEl);
  if (!view.length) {
    if (final) box.replaceChildren(Object.assign(document.createElement("p"), { className: "empty", textContent: ui.emptyText }));
    else box.replaceChildren();
    return;
  }
  box.replaceChildren(...view.map((g) => {
    const sec = document.createElement("section");
    sec.className = "group"; colourStyle(sec, g.label); sec.dataset.label = g.label;
    const h = document.createElement("h3");
    h.append(document.createElement("span"), document.createElement("small"));
    h.firstChild.textContent = g.label; h.lastChild.textContent = `${fmtN(g.entities.length)} distinct`;
    const ol = document.createElement("ol");
    const limit = b.expanded.has(g.label) ? Infinity : 15;
    for (const e of g.entities.slice(0, limit)) {
      const li = document.createElement("li"), btn = document.createElement("button");
      btn.type = "button"; btn.dataset.key = e.key;
      btn.setAttribute("aria-pressed", String(b.selected?.label === g.label && b.selected?.key === e.key));
      btn.append(document.createElement("span"), document.createElement("span"));
      btn.firstChild.className = "name"; btn.lastChild.className = "n";
      btn.firstChild.textContent = e.name; btn.lastChild.textContent = fmtN(e.count);
      btn.title = `${e.name}: ${e.count} ${ui.unit}${e.count === 1 ? "" : "s"}`;
      btn.onclick = () => { b.selected = { label: g.label, key: e.key }; renderGroups(ui, true); renderDetail(ui); };
      li.append(btn); ol.append(li);
    }
    sec.append(h, ol);
    if (g.entities.length > 15) {
      const more = document.createElement("button");
      more.type = "button"; more.className = "linkish more";
      more.textContent = b.expanded.has(g.label) ? "show fewer" : `show all ${fmtN(g.entities.length)}`;
      more.onclick = () => { b.expanded.has(g.label) ? b.expanded.delete(g.label) : b.expanded.add(g.label); renderGroups(ui, true); };
      sec.append(more);
    }
    return sec;
  }));
}

function renderDetail(ui) {
  const b = ui.st(), box = $(ui.detailEl);
  const g = b?.view?.find((x) => x.label === b.selected?.label), e = g?.entities.find((x) => x.key === b.selected.key);
  if (!e) { box.hidden = true; box.replaceChildren(); return; }
  box.hidden = false; colourStyle(box, g.label);
  const h = document.createElement("h3"), title = document.createElement("span"), close = document.createElement("button");
  close.type = "button"; close.className = "linkish"; close.textContent = "close";
  title.textContent = `${e.name} · ${g.label} · ${fmtN(e.count)} ${ui.unit}${e.count === 1 ? "" : "s"}`;
  close.onclick = () => { b.selected = null; box.hidden = true; renderGroups(ui, true); };
  h.append(title, close);
  const ul = document.createElement("ul");
  for (const p of ui.postsOf(b, e)) {
    const li = document.createElement("li"), t = document.createElement("div");
    t.className = "post";
    t.append(...highlight(p.text, p.ents, { tag: false, only: (x) => x.label === g.label && normEntity(x.text) === e.key }));
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.append(...ui.meta(p));
    li.append(t, meta); ul.append(li);
  }
  box.replaceChildren(h, ul);
}

function postLink(href) {
  const a = document.createElement("a");
  a.href = href; a.target = "_blank"; a.rel = "noopener"; a.textContent = "open on Bluesky";
  return a;
}

// ---- Bluesky mode ----
const bskyUI = {
  st: () => app.bsky, groupsEl: "groups", detailEl: "bsky-detail", unit: "post", emptyText: "No entities found in these posts.",
  postsOf: (b, e) => e.posts.slice(0, 100).map((pi) => b.groups.post(pi)),
  meta: (p) => [document.createTextNode(`@${p.author}${p.createdAt ? " · " + p.createdAt.slice(0, 10) : ""}${p.repost ? " · repost" : ""} · `),
    ...(p.url ? [postLink(p.url)] : [])],
};

function progressLine(final) {
  const b = app.bsky, secs = (performance.now() - b.t0) / 1000;
  const rate = b.analysed / Math.max(secs, 1e-3), nerRate = b.analysed / Math.max(b.nerMs / 1000, 1e-3);
  const where = b.src ? `${b.src.label}: ` : "";
  $("bsky-progress").textContent = `${where}fetched ${fmtN(b.fetched)} · analysed ${fmtN(b.analysed)} posts · ` +
    `${rate.toFixed(1)} posts/s overall, ${nerRate.toFixed(1)} posts/s in the model · ${fmtN(b.entities)} entities` +
    (final ? (b.stopped ? " · stopped" : " · done") : (b.done ? "" : " · fetching…"));
}

async function analyse(b, batch) {
  const t0 = performance.now();
  const ents = await locked(async () => {
    await app.ner.setLabels(b.labels);
    return app.ner.extract(batch.map((p) => p.text), { threshold: app.threshold, truncate: true });
  });
  b.nerMs += performance.now() - t0;
  batch.forEach((p, i) => { b.groups.add(p, ents[i]); b.entities += ents[i].length; });
  b.analysed += batch.length;
}

async function runBsky(ev) {
  ev?.preventDefault();
  if (app.state !== "ready" || app.bsky?.running) return;
  syncUrl();
  if (!app.labels.length) { $("bsky-progress").textContent = "Add an entity type to search for."; return; }
  const ctl = new AbortController();
  const b = app.bsky = { running: true, done: false, stopped: false, ctl, posts: [], queue: [], fetched: 0, analysed: 0, entities: 0, nerMs: 0,
    groups: new EntityGroups(), expanded: new Set(), selected: null, t0: performance.now(), src: null, error: null, labels: [...app.labels] };
  $("bsky-run").disabled = true; $("bsky-stop").hidden = false; $("bsky-detail").hidden = true; $("groups").replaceChildren();
  $("bsky-progress").textContent = "resolving…";
  let wake = null;
  const poke = () => { const w = wake; wake = null; w?.(); };
  try {
    b.src = await resolveSource($("bsky-input").value, ctl.signal);
    const producer = (async () => {
      try {
        for await (const page of fetchPosts(b.src, { max: +$("bsky-count").value, reposts: $("bsky-reposts").checked, signal: ctl.signal })) {
          b.posts.push(...page); b.queue.push(...page); b.fetched += page.length; poke();
        }
      } finally { b.done = true; poke(); }
    })();
    producer.catch(() => {});   // awaited below, after the queue drains
    let lastPaint = 0;
    while (!b.stopped && (!b.done || b.queue.length)) {
      if (!b.queue.length) { await new Promise((r) => { wake = r; }); continue; }
      await analyse(b, b.queue.splice(0, 50));
      if (performance.now() - lastPaint > 400) { renderGroups(bskyUI); progressLine(false); lastPaint = performance.now(); }
    }
    await producer;
  } catch (e) {
    if (!b.stopped) b.error = e;
  }
  b.running = false;
  $("bsky-run").disabled = false; $("bsky-stop").hidden = true;
  if (b.error) { $("bsky-progress").textContent = `Could not fetch posts: ${b.error.message}`; }
  else { progressLine(true); }
  renderGroups(bskyUI, true);
}

// labels or threshold changed after a run: analyse the fetched posts again, no refetch
async function reanalyse() {
  const b = app.bsky;
  if (!b?.posts.length || b.running || !app.labels.length) return;
  Object.assign(b, { labels: [...app.labels], running: true, groups: new EntityGroups(), analysed: 0, entities: 0, nerMs: 0, t0: performance.now(), selected: null, done: true, stopped: false });
  $("bsky-detail").hidden = true;
  try { for (let i = 0; i < b.posts.length; i += 100) await analyse(b, b.posts.slice(i, i + 100)); progressLine(true); }
  catch (e) { fail(e); }
  b.running = false;
  renderGroups(bskyUI, true);
}

// ---- firehose mode ----
// Socket handler -> filters -> sampler -> queue (cap FH.maxQueue, oldest dropped) -> batches of up to FH.batch, or
// whatever arrived within FH.waitMs -> ner.extract -> the rolling window of analysed posts (EntityGroups with
// removal). The view is re-rendered at most every FH.renderMs.
const DID_RE = /^did:[a-z]+:[A-Za-z0-9._:%-]+$/, RKEY_RE = /^[A-Za-z0-9._:~-]{1,512}$/;
const shortDid = (d) => (d.length > 20 ? d.slice(0, 16) + "…" + d.slice(-4) : d);
const fhUI = {
  st: () => app.fh, groupsEl: "fh-groups", detailEl: "fh-detail", unit: "post", emptyText: "No entities in the window yet.",
  postsOf: (f, e) => e.posts.slice(-FH.detailCap).reverse().map((pi) => f.groups.post(pi)),
  meta: (p) => {
    const who = document.createElement("span");
    who.textContent = shortDid(p.did); who.title = p.did;
    const when = p.createdAt ? new Date(p.createdAt) : null;
    const nodes = [who, document.createTextNode(`${when && !isNaN(when) ? " · " + when.toISOString().slice(11, 19) + " UTC" : ""} · `)];
    if (DID_RE.test(p.did) && RKEY_RE.test(p.rkey)) nodes.push(postLink(`https://bsky.app/profile/${p.did}/post/${p.rkey}`));
    return nodes;
  },
};

function fhSettings() {
  return { sample: $("fh-sample").value, lang: $("fh-lang").value, replies: !$("fh-skip-replies").checked,
    minlen: Math.max(0, Math.min(10_000, Math.floor(+$("fh-minlen").value) || 0)), include: $("fh-include").value, exclude: $("fh-exclude").value,
    window: $("fh-window").value };
}

function fhApplySettings() {
  const f = app.fh;
  syncUrl();
  if (!f) return;
  const s = fhSettings();
  f.filter = makeFilter(s);
  if (+s.sample !== f.sampleEvery) { f.sampleEvery = +s.sample; f.sampler = makeSampler(f.sampleEvery); }
  f.window = +s.window;
  fhTrim(f);
  fhDirty();
}

function fhNew() {
  const s = fhSettings();
  return { running: false, js: null, queue: [], fifo: [], groups: new EntityGroups(), expanded: new Set(), selected: null, view: null,
    filter: makeFilter(s), sampleEvery: +s.sample, sampler: makeSampler(+s.sample), window: +s.window,
    counts: { events: 0, posts: 0, passed: 0, sampled: 0, analysed: 0, dropped: 0, entities: 0 },
    conn: { state: "idle", endpoint: null, note: "" }, hist: [], renders: 0, lastRender: 0, renderTimer: null, tick: null,
    wake: null, full: null, nerMs: 0, t0: performance.now(), note: "" };
}

function fhStart() {
  if (app.state !== "ready" || app.fh?.running) return;
  if (!app.labels.length) { $("fh-stats").textContent = "Add an entity type to search for."; return; }
  const f = app.fh = fhNew();
  f.running = true;
  $("fh-start").hidden = true; $("fh-stop").hidden = false; $("fh-detail").hidden = true; $("fh-groups").replaceChildren();
  f.js = new Jetstream({
    onEvent: () => { f.counts.events++; },
    onPost: (p) => fhOnPost(f, p),
    onState: (state, endpoint, note) => { f.conn = { state, endpoint, note }; fhDirty(); },
  });
  f.js.start();
  f.tick = setInterval(fhDirty, 1000);
  fhPump(f);
  fhDirty();
}

function fhStop(note = "stopped") {
  const f = app.fh;
  if (!f?.running) return;
  f.running = false;
  f.js.stop(note);
  clearInterval(f.tick);
  f.queue.length = 0;
  f.wake?.(); f.full?.();
  $("fh-start").hidden = false; $("fh-stop").hidden = true;
  fhRender();
}

// The socket handler: counting, filtering, sampling and queueing only. Never waits on the model.
function fhOnPost(f, p) {
  if (!f.running) return;
  f.counts.posts++;
  if (!f.filter(p)) return;
  f.counts.passed++;
  if (!f.sampler()) return;
  f.counts.sampled++;
  f.queue.push(p);
  if (f.queue.length > FH.maxQueue) { const n = f.queue.length - FH.maxQueue; f.queue.splice(0, n); f.counts.dropped += n; }
  if (f.wake) { const w = f.wake; f.wake = null; w(); }
  if (f.full && f.queue.length >= FH.batch) { const w = f.full; f.full = null; w(); }
}

async function fhPump(f) {
  while (f.running) {
    if (!f.queue.length) { await new Promise((r) => { f.wake = r; }); continue; }
    if (f.queue.length < FH.batch) { await Promise.race([sleep(FH.waitMs), new Promise((r) => { f.full = r; })]); f.full = null; }
    if (!f.running) break;
    const batch = f.queue.splice(0, FH.batch);
    if (!batch.length) continue;
    let ents;
    const t0 = performance.now();
    try {
      ents = await locked(async () => {
        await app.ner.setLabels(app.labels);
        return app.ner.extract(batch.map((p) => p.text), { threshold: app.threshold, truncate: true });
      });
    } catch (e) { fhStop("stopped: the model failed"); fail(e); return; }
    f.nerMs += performance.now() - t0;
    if (!f.running || app.fh !== f) break;
    batch.forEach((p, i) => { f.fifo.push(f.groups.add(p, ents[i])); f.counts.entities += ents[i].length; });
    f.counts.analysed += batch.length;
    fhTrim(f);
    fhDirty();
  }
}

function fhTrim(f) {
  const n = f.fifo.length - f.window;
  if (n <= 0) return;
  for (const id of f.fifo.splice(0, n)) f.groups.remove(id);
}

function fhClearWindow(note) {
  const f = app.fh;
  for (const id of f.fifo) f.groups.remove(id);
  f.fifo.length = 0; f.selected = null; f.note = note;
  fhDirty();
}

function fhDirty() {
  const f = app.fh;
  if (!f || f.renderTimer) return;
  f.renderTimer = setTimeout(fhRender, Math.max(0, FH.renderMs - (performance.now() - f.lastRender)));
}

function fhRender() {
  const f = app.fh;
  if (!f) return;
  clearTimeout(f.renderTimer); f.renderTimer = null;
  f.lastRender = performance.now(); f.renders++;
  const c = f.counts, now = performance.now();
  f.hist.push({ t: now, ...c });
  while (f.hist.length > 2 && now - f.hist[1].t > 5000) f.hist.shift();
  const h0 = f.hist[0], dt = Math.max((now - h0.t) / 1000, 1);
  const rate = (k) => fmtRate((c[k] - h0[k]) / dt);
  const conn = f.conn.state === "open" ? `connected to ${f.conn.endpoint}` : f.conn.state === "stopped" ? (f.conn.note || "stopped")
    : `${f.conn.state}${f.conn.endpoint ? " to " + f.conn.endpoint : ""}${f.conn.note ? ` (${f.conn.note})` : ""}`;
  $("fh-stats").textContent = f.running ? `${conn} · received ${rate("events")}/s · passed filters ${rate("passed")}/s · analysed ${rate("analysed")}/s · ` +
    `queue ${fmtN(f.queue.length)} · dropped ${fmtN(c.dropped)}` : `${conn} · dropped ${fmtN(c.dropped)}`;
  $("fh-totals").textContent = `Totals: ${fmtN(c.events)} events, ${fmtN(c.posts)} new posts, ${fmtN(c.passed)} passed the filters, ` +
    `${fmtN(c.sampled)} sampled, ${fmtN(c.analysed)} analysed, ${fmtN(c.entities)} entities · window: last ${fmtN(f.groups.size)} of ${fmtN(f.window)} analysed posts` +
    (f.note ? ` · ${f.note}` : "");
  if (app.mode === "firehose") { renderGroups(fhUI, !f.running || c.analysed > 0); renderDetail(fhUI); }
}

// stop the socket after the page has been hidden for FH.hiddenStopMs
let hiddenTimer = null;
document.addEventListener("visibilitychange", () => {
  clearTimeout(hiddenTimer);
  if (document.hidden && app.fh?.running) {
    hiddenTimer = setTimeout(() => {
      fhStop("stopped: the page was hidden for more than a minute");
      status("The firehose was stopped because the page was hidden for more than a minute. Press Start to resume.", "warn");
    }, FH.hiddenStopMs);
  }
});

// ---- tabs ----
const MODES = { text: "text", bluesky: "bsky", bsky: "bsky", firehose: "firehose" };
function tab(which) {
  if (app.mode === "firehose" && which !== "firehose" && app.fh?.running) {
    fhStop("stopped: you switched away from the Firehose tab");
    status("The firehose was stopped because you switched tabs. Press Start on the Firehose tab to resume.", "warn");
  }
  app.mode = which;
  for (const t of ["text", "bsky", "firehose"]) {
    $(`tab-${t}`).setAttribute("aria-selected", String(t === which));
    $(`panel-${t}`).hidden = t !== which;
  }
  if (which === "text") schedule();
  else if (which === "bsky" && app.bsky?.posts.length) renderGroups(bskyUI, true);
  else if (which === "firehose" && app.fh) fhRender();
  syncUrl();
}

// ---- link parameters ----
// Read once on load; anything unknown or malformed keeps the default. Values only ever reach the DOM as
// .value / .textContent.
function intParam(q, k, lo, hi) {
  const v = q.get(k);
  if (v === null || !/^\d{1,7}$/.test(v.trim())) return null;
  const n = +v;
  return n >= lo && n <= hi ? n : null;
}
const flagParam = (q, k) => { const v = q.get(k); return v === "1" ? true : v === "0" ? false : null; };
const strParam = (q, k, max) => { const v = q.get(k); return v === null ? null : v.slice(0, max); };

function ensureOption(sel, value, text) {
  if (![...sel.options].some((o) => o.value === value)) {
    const o = document.createElement("option");
    o.value = value; o.textContent = text;
    sel.append(o);
  }
  sel.value = value;
}

function readParams() {
  let q;
  try { q = new URLSearchParams(location.search); } catch { return; }
  const labels = q.get("labels");
  if (labels !== null) { const l = cleanLabels(labels.slice(0, 2000).split(",")).slice(0, 64); if (l.length) app.labels = l; }
  const th = q.get("threshold");
  if (th !== null && /^\s*0?\.\d+\s*$|^\s*0\s*$|^\s*1\s*$/.test(th)) {
    const v = Math.round(Math.min(0.95, Math.max(0.05, +th)) * 20) / 20;
    app.threshold = v; $("threshold").value = String(v); $("threshold-val").textContent = v.toFixed(2);
  }
  const text = strParam(q, "text", 5000);
  if (text !== null && text.trim()) { $("text").value = text; app.linkText = text; }
  const source = strParam(q, "source", 500);
  if (source !== null && source.trim()) $("bsky-input").value = source.trim();
  const n = intParam(q, "n", 1, 1000);
  if (n !== null) ensureOption($("bsky-count"), String(n), `${fmtN(n)} posts`);
  const reposts = flagParam(q, "reposts");
  if (reposts !== null) $("bsky-reposts").checked = reposts;
  // firehose settings
  const sample = q.get("sample");
  if (["1", "2", "5", "10", "50", "100"].includes(sample)) $("fh-sample").value = sample;
  const lang = q.get("lang");
  if (lang && /^(any|[a-z]{2,3}(-[a-z0-9]{2,8})?)$/i.test(lang)) ensureOption($("fh-lang"), lang.toLowerCase(), lang.toLowerCase());
  const replies = flagParam(q, "replies");
  if (replies !== null) $("fh-skip-replies").checked = !replies;
  const minlen = intParam(q, "minlen", 0, 10_000);
  if (minlen !== null) $("fh-minlen").value = String(minlen);
  for (const k of ["include", "exclude"]) { const v = strParam(q, k, 500); if (v !== null) $(`fh-${k}`).value = v; }
  const win = intParam(q, "window", 10, 100_000);
  if (win !== null) ensureOption($("fh-window"), String(win), `last ${fmtN(win)} analysed posts`);
  // mode: explicit, else implied by source / text
  let mode = MODES[q.get("mode")] || null;
  if (!mode) mode = source?.trim() ? "bsky" : text?.trim() ? "text" : null;
  if (mode) tab(mode);
  if (mode === "bsky" && $("bsky-input").value) app.pending = "bsky";
  if (mode === "firehose" && q.get("start") === "1") app.pending = "firehose";
}

// The address bar follows the current settings (replaceState, debounced). The free text only while it is
// still the text the link brought, and only if short.
let urlTimer = null;
function syncUrl(now = false) {
  clearTimeout(urlTimer);
  if (!now) { urlTimer = setTimeout(() => syncUrl(true), 150); return; }
  const q = new URLSearchParams();
  const mode = app.mode;
  if (mode !== "text") q.set("mode", mode === "bsky" ? "bluesky" : mode);
  if (mode === "text" && app.linkText !== null && $("text").value === app.linkText && app.linkText.length <= 300) q.set("text", app.linkText);
  if (mode === "bsky") {
    const src = $("bsky-input").value.trim();
    if (src) q.set("source", src.slice(0, 500));
    q.set("n", $("bsky-count").value);
    if ($("bsky-reposts").checked) q.set("reposts", "1");
  }
  if (mode === "firehose") {
    const s = fhSettings();
    if (s.sample !== FH_DEFAULTS.sample) q.set("sample", s.sample);
    if (s.lang !== FH_DEFAULTS.lang) q.set("lang", s.lang);
    if (s.replies !== FH_DEFAULTS.replies) q.set("replies", s.replies ? "1" : "0");
    if (s.minlen !== FH_DEFAULTS.minlen) q.set("minlen", String(s.minlen));
    if (s.include.trim()) q.set("include", s.include.trim().slice(0, 500));
    if (s.exclude.trim()) q.set("exclude", s.exclude.trim().slice(0, 500));
    if (s.window !== FH_DEFAULTS.window) q.set("window", s.window);
  }
  if (app.labels.join(",") !== DEFAULT_LABELS.join(",")) q.set("labels", app.labels.join(","));
  if (app.threshold !== 0.5) q.set("threshold", app.threshold.toFixed(2));
  const qs = q.toString().replace(/%2C/g, ",");
  const href = location.pathname + (qs ? "?" + qs : "") + location.hash;
  if (href !== location.pathname + location.search + location.hash) {
    try { history.replaceState(history.state, "", href); } catch { /* sandboxed or rate-limited: keep the old URL */ }
  }
}

async function copyLink() {
  syncUrl(true);
  const url = location.href, btn = $("copy-link");
  let ok = false;
  try { await navigator.clipboard.writeText(url); ok = true; } catch {
    const ta = Object.assign(document.createElement("textarea"), { value: url });
    ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.append(ta); ta.select();
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    ta.remove();
  }
  app.copied = ok ? url : null;
  btn.textContent = ok ? "Link copied" : "Copy failed: use the address bar";
  clearTimeout(copyLink.t);
  copyLink.t = setTimeout(() => { btn.textContent = "Copy link"; }, 2000);
}

// waiting for the model before a run the link asked for
function showPending() {
  if (app.pending === "bsky") $("bsky-progress").textContent = `Waiting for the model to load; then this fetches and analyses ${$("bsky-input").value}.`;
  if (app.pending === "firehose") $("fh-stats").textContent = "Waiting for the model to load; then the firehose starts.";
  app.waitedForModel = !!app.pending;
}

function runPending() {
  const p = app.pending;
  app.pending = null;
  if (p === "bsky" && app.mode === "bsky") runBsky();
  if (p === "firehose" && app.mode === "firehose") fhStart();
}

// ---- loading ----
async function fetchProgress(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split("/").pop()}: HTTP ${r.status}`);
  if (!r.body) { const b = new Uint8Array(await r.arrayBuffer()); onBytes(b.length); return b; }
  const reader = r.body.getReader(), parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); n += value.length; onBytes(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function disableRuns(on) { $("bsky-run").disabled = on; $("fh-start").disabled = on; }

function fail(e) {
  console.error(e);
  app.state = "error"; app.error = String(e?.message || e);
  status(`Something went wrong: ${app.error}. Reloading the page may help.`, "error");
  disableRuns(true);
}

async function load() {
  disableRuns(true);
  if (!navigator.gpu) { noGpu("this browser does not expose WebGPU"); return; }
  let adapter = null;
  try { adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }); } catch { /* treated as none */ }
  if (!adapter) { noGpu("WebGPU is present but no GPU adapter is available"); return; }
  app.state = "loading";
  const t0 = performance.now();
  try {
    const base = new URL(MODEL, location.href);
    const manifest = await fetch(new URL("manifest.json", base)).then((r) => { if (!r.ok) throw new Error(`manifest.json: HTTP ${r.status}`); return r.json(); });
    const g = manifest.gliner, total = g.parts_end.total;
    let got = 0, side = 0;
    const show = () => status(`Downloading the model: ${fmtMB(got)} of ${fmtMB(total)}`, "", Math.min(got / total, 1));
    show();
    const [tj, ltj, data] = await Promise.all([
      fetchProgress(new URL(g.tokenizer, base), (n) => { side += n; }),
      fetchProgress(new URL(g.label_tokenizer, base), (n) => { side += n; }),
      fetchProgress(new URL("weights.bin", base), (n) => { got += n; show(); }),
    ]);
    const dl = performance.now() - t0;
    status("Preparing the model on the GPU…", "", 1);
    const dec = new TextDecoder();
    const device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    device.lost.then((info) => { if (info.reason !== "destroyed") fail(new Error(`the GPU device was lost (${info.message || info.reason})`)); });
    const ner = await NER.from({ manifest, tj: JSON.parse(dec.decode(tj)), ltj: JSON.parse(dec.decode(ltj)), data }, { device });
    app.ner = ner;
    app.load = { bytes: data.length + side, downloadMs: dl, totalMs: performance.now() - t0, initMs: ner.info.initMs };
    app.modelFacts = `model ${fmtMB(app.load.bytes)} downloaded, ready in ${(app.load.totalMs / 1000).toFixed(1)} s`;
    app.state = "ready"; app.readyAt = performance.now();
    status(`Ready: ${app.modelFacts}. Edit the text or the entity types; results update as you type.`, "ready");
    disableRuns(false);
    if (app.mode === "text") runText();
    runPending();
  } catch (e) { fail(e); }
}

function noGpu(why) {
  app.state = "nogpu";
  status(`This page needs WebGPU to run the model, and ${why}. It works in recent Chrome and Edge (desktop and Android), Safari 26, ` +
    "and Firefox 141 or later on Windows. Everything else on the page still works, but no entities can be found here.", "warn");
  $("out").innerHTML = '<span class="empty">The model cannot run in this browser.</span>';
  if (app.pending === "bsky") $("bsky-progress").textContent = "The link asked to analyse these posts, but the model cannot run in this browser.";
  if (app.pending === "firehose") $("fh-stats").textContent = "The link asked to start the firehose, but the model cannot run in this browser.";
  app.pending = null;
  disableRuns(true);
}

// ---- wiring ----
$("text").value = DEFAULT_TEXT;
$("text").addEventListener("input", () => { schedule(); syncUrl(); });
$("label-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addLabels(e.target.value); e.target.value = ""; }
});
$("label-add").onclick = () => { addLabels($("label-input").value); $("label-input").value = ""; $("label-input").focus(); };
for (const p of document.querySelectorAll(".preset")) {
  p.onclick = () => {
    if (p.hasAttribute("data-reset")) { app.labels = [...DEFAULT_LABELS]; labelsChanged(); }
    else addLabels(p.dataset.labels);
  };
}
$("threshold").addEventListener("input", (e) => {
  app.threshold = +e.target.value;
  $("threshold-val").textContent = app.threshold.toFixed(2);
  schedule(); syncUrl();
});
$("tab-text").onclick = () => tab("text");
$("tab-bsky").onclick = () => tab("bsky");
$("tab-firehose").onclick = () => tab("firehose");
$("copy-link").onclick = copyLink;
$("bsky-form").addEventListener("submit", runBsky);
$("bsky-stop").onclick = () => { if (app.bsky) { app.bsky.stopped = true; app.bsky.ctl.abort(); } };
for (const id of ["bsky-input", "bsky-count", "bsky-reposts"]) $(id).addEventListener("change", () => syncUrl());
for (const x of document.querySelectorAll("[data-example]")) x.onclick = () => { $("bsky-input").value = x.dataset.example; $("bsky-input").focus(); syncUrl(); };
$("fh-form").addEventListener("submit", (e) => e.preventDefault());
$("fh-start").onclick = fhStart;
$("fh-stop").onclick = () => fhStop("stopped");
for (const id of ["fh-sample", "fh-lang", "fh-skip-replies", "fh-minlen", "fh-include", "fh-exclude", "fh-window"]) $(id).addEventListener("change", fhApplySettings);
readParams();
renderChips();
$("out").innerHTML = '<span class="empty">Results appear here once the model has loaded.</span>';
showPending();
syncUrl();
load();
