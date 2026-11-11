// Demo page for the browser NER runtime (ner.js): text mode and Bluesky mode. window.app exposes the
// state for run/page_smoke.mjs.
import { NER } from "./ner.js";
import { resolveSource, fetchPosts, EntityGroups, normEntity } from "./bsky.js";

const MODEL = "models/gliner-bi-edge-v2.0-t4-w8L63-h8/";
const DEFAULT_LABELS = ["person", "company", "location", "product", "date"];
const DEFAULT_TEXT = "Satya Nadella flew to Lisbon on 4 November to open Microsoft's new AI hub, two weeks after Google " +
  "showed its Pixel 10 phones in New York. Portugal's prime minister, Luís Montenegro, called it the biggest tech investment " +
  "the country has seen since Web Summit moved there in 2016.";
const NCOL = 8;

const $ = (id) => document.getElementById(id);
const app = window.app = { state: "starting", ner: null, labels: [...DEFAULT_LABELS], threshold: 0.5, last: null, bsky: null, error: null };
const fmtMs = (x) => (x < 10 ? x.toFixed(1) : Math.round(x).toLocaleString("en-US")) + " ms";
const fmtMB = (b) => (b / 1e6).toFixed(1) + " MB";

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
  const fresh = text.split(",").map((s) => s.trim().replace(/\s+/g, " ")).filter((s) => s && !app.labels.includes(s));
  if (!fresh.length) return;
  app.labels.push(...fresh.slice(0, 64 - app.labels.length));
  labelsChanged();
}

function labelsChanged() { renderChips(); schedule(); }

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
    if (!$("panel-text").hidden) runText();
    else if (app.bsky?.posts?.length && !app.bsky.running) reanalyse();
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

// ---- Bluesky mode ----
function renderGroups(final = false) {
  const b = app.bsky;
  const view = b.groups.view(app.labels);
  b.view = view;
  const box = $("groups");
  if (!view.length) { box.innerHTML = final ? '<p class="empty">No entities found in these posts.</p>' : ""; return; }
  box.replaceChildren(...view.map((g) => {
    const sec = document.createElement("section");
    sec.className = "group"; colourStyle(sec, g.label); sec.dataset.label = g.label;
    const h = document.createElement("h3");
    h.innerHTML = `<span></span><small></small>`;
    h.firstChild.textContent = g.label; h.lastChild.textContent = `${g.entities.length} distinct`;
    const ol = document.createElement("ol");
    const limit = b.expanded.has(g.label) ? Infinity : 15;
    for (const e of g.entities.slice(0, limit)) {
      const li = document.createElement("li"), btn = document.createElement("button");
      btn.type = "button"; btn.dataset.key = e.key;
      btn.setAttribute("aria-pressed", String(b.selected?.label === g.label && b.selected?.key === e.key));
      btn.innerHTML = `<span class="name"></span><span class="n"></span>`;
      btn.firstChild.textContent = e.name; btn.lastChild.textContent = e.count;
      btn.title = `${e.name}: ${e.count} post${e.count === 1 ? "" : "s"}`;
      btn.onclick = () => { b.selected = { label: g.label, key: e.key }; renderGroups(true); renderDetail(); };
      li.append(btn); ol.append(li);
    }
    sec.append(h, ol);
    if (g.entities.length > 15) {
      const more = document.createElement("button");
      more.type = "button"; more.className = "linkish more";
      more.textContent = b.expanded.has(g.label) ? "show fewer" : `show all ${g.entities.length}`;
      more.onclick = () => { b.expanded.has(g.label) ? b.expanded.delete(g.label) : b.expanded.add(g.label); renderGroups(true); };
      sec.append(more);
    }
    return sec;
  }));
}

function renderDetail() {
  const b = app.bsky, box = $("bsky-detail");
  const g = b?.view?.find((x) => x.label === b.selected?.label), e = g?.entities.find((x) => x.key === b.selected.key);
  if (!e) { box.hidden = true; return; }
  box.hidden = false; colourStyle(box, g.label);
  const h = document.createElement("h3");
  h.innerHTML = `<span></span><button type="button" class="linkish">close</button>`;
  h.firstChild.textContent = `${e.name} · ${g.label} · ${e.count} post${e.count === 1 ? "" : "s"}`;
  h.lastChild.onclick = () => { b.selected = null; box.hidden = true; renderGroups(true); };
  const ul = document.createElement("ul");
  for (const pi of e.posts.slice(0, 100)) {
    const p = b.groups.posts[pi], li = document.createElement("li");
    const t = document.createElement("div");
    t.className = "post";
    t.append(...highlight(p.text, p.ents, { tag: false, only: (x) => x.label === g.label && normEntity(x.text) === e.key }));
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.append(document.createTextNode(`@${p.author}${p.createdAt ? " · " + p.createdAt.slice(0, 10) : ""}${p.repost ? " · repost" : ""} · `));
    if (p.url) { const a = document.createElement("a"); a.href = p.url; a.target = "_blank"; a.rel = "noopener"; a.textContent = "open on Bluesky"; meta.append(a); }
    li.append(t, meta); ul.append(li);
  }
  box.replaceChildren(h, ul);
}

function progressLine(final) {
  const b = app.bsky, secs = (performance.now() - b.t0) / 1000;
  const rate = b.analysed / Math.max(secs, 1e-3), nerRate = b.analysed / Math.max(b.nerMs / 1000, 1e-3);
  const where = b.src ? `${b.src.label}: ` : "";
  $("bsky-progress").textContent = `${where}fetched ${b.fetched.toLocaleString("en-US")} · analysed ${b.analysed.toLocaleString("en-US")} posts · ` +
    `${rate.toFixed(1)} posts/s overall, ${nerRate.toFixed(1)} posts/s in the model · ${b.entities.toLocaleString("en-US")} entities` +
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
      if (performance.now() - lastPaint > 400) { renderGroups(); progressLine(false); lastPaint = performance.now(); }
    }
    await producer;
  } catch (e) {
    if (!b.stopped) b.error = e;
  }
  b.running = false;
  $("bsky-run").disabled = false; $("bsky-stop").hidden = true;
  if (b.error) { $("bsky-progress").textContent = `Could not fetch posts: ${b.error.message}`; }
  else { progressLine(true); }
  renderGroups(true);
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
  renderGroups(true);
}

// ---- tabs ----
function tab(which) {
  for (const t of ["text", "bsky"]) {
    $(`tab-${t}`).setAttribute("aria-selected", String(t === which));
    $(`panel-${t}`).hidden = t !== which;
  }
  if (which === "text") schedule();
  else if (app.bsky?.posts.length) renderGroups(true);
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

function fail(e) {
  console.error(e);
  app.state = "error"; app.error = String(e?.message || e);
  status(`Something went wrong: ${app.error}. Reloading the page may help.`, "error");
  $("bsky-run").disabled = true;
}

async function load() {
  $("bsky-run").disabled = true;
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
    app.state = "ready";
    status(`Ready: ${app.modelFacts}. Edit the text or the entity types; results update as you type.`, "ready");
    $("bsky-run").disabled = false;
    runText();
  } catch (e) { fail(e); }
}

function noGpu(why) {
  app.state = "nogpu";
  status(`This page needs WebGPU to run the model, and ${why}. It works in recent Chrome and Edge (desktop and Android), Safari 26, ` +
    "and Firefox 141 or later on Windows. Everything else on the page still works, but no entities can be found here.", "warn");
  $("out").innerHTML = '<span class="empty">The model cannot run in this browser.</span>';
  $("bsky-run").disabled = true;
}

// ---- wiring ----
$("text").value = DEFAULT_TEXT;
$("text").addEventListener("input", schedule);
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
  schedule();
});
$("tab-text").onclick = () => tab("text");
$("tab-bsky").onclick = () => tab("bsky");
$("bsky-form").addEventListener("submit", runBsky);
$("bsky-stop").onclick = () => { if (app.bsky) { app.bsky.stopped = true; app.bsky.ctl.abort(); } };
for (const x of document.querySelectorAll("[data-example]")) x.onclick = () => { $("bsky-input").value = x.dataset.example; $("bsky-input").focus(); };
renderChips();
$("out").innerHTML = '<span class="empty">Results appear here once the model has loaded.</span>';
load();
