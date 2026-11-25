// Strudel FM with taste. Every phrase is picked from eight candidates the station's own generator wrote.
// A logistic model over 31 counted features per candidate (hits per bar, section, mode, ...), trained in
// this tab from likes, skips and phrases left playing, scores the candidates, and the next phrase is
// sampled from softmax(score / TAU).
//
// ?brain=ettin chooses with a model of the same form over Ettin-17m vectors instead: the browser-decider
// engine reads a description of each candidate on the GPU and the hidden state at its [MASK] is the
// vector. In run/sim.py's simulated listeners it learned far slower than the counted features (RESULTS.md),
// so it is opt-in. ?text=code feeds it the first bar's Strudel code. ?harvest exposes window.harvest(n) for
// run/harvest.mjs.

import { loadDecider } from "../../browser-decider/web/decider.js";

const fm = window.__fm, T = fm.taste, $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const MODEL = Q.get("model") || "../../browser-decider/web/models/decider/";
const TEXT = Q.get("text") === "code" ? "code" : "desc";
const BRAIN = Q.get("brain") === "ettin" ? "ettin" : "feats";
const INSTR = "Which phrase should this radio station play next for this listener?";
const STORE = "strudel-taste-v1";
// Constants shared with run/sim.py (keep in step): passive weight, softmax temperature, exploration floor, ridge.
// TAU and LAMBDA are the values run/sim.py chose on its dev listeners for both models.
export const W_PASSIVE = 0.3, TAU = 0.25, EPS = 0.05, LAMBDA = 0.1;
const MAX_EX = 600;

// ------------------------------------------------------------------ the GPU encoder
let dec = null, loading = null, gpuQ = Promise.resolve();
const exclusive = (fn) => { const r = gpuQ.then(fn, fn); gpuQ = r.catch(() => {}); return r; };

function ensureModel() {
  if (dec || loading) return loading ?? Promise.resolve(dec);
  if (!navigator.gpu) { status("No WebGPU in this browser: choosing with the 32-feature model only."); return Promise.resolve(null); }
  status("Loading the 10 MB decision model…");
  loading = loadDecider(MODEL).then((d) => {
    const eng = d.engine, orig = eng.headLogits.bind(eng);
    eng.headLogits = (rows, P) => { eng._rows = rows.slice(0, P * eng.dim); return orig(rows, P); };
    dec = d; status(""); paint(); return d;
  }, (e) => { console.warn(e); status("The decision model did not load (" + e.message + "): choosing with the 32-feature model."); return null; });
  return loading;
}

// One vector per candidate: the encoder's final hidden state at the candidate's [MASK].
function embedTexts(texts, state) {
  return exclusive(async () => {
    const eng = dec.engine, H = eng.dim, out = [];
    const seqs = texts.map((t) => dec.build("choice", INSTR, [["candidate", t]], state));
    for (let i = 0; i < seqs.length;) {
      const g = []; let tok = 0;
      while (i < seqs.length && (!g.length || tok + seqs[i].ids.length <= eng.maxTokens)) { tok += seqs[i].ids.length; g.push(seqs[i++]); }
      await eng.score(g);
      for (let k = 0; k < g.length; k++) out.push(Float32Array.from(eng._rows.subarray(k * H, (k + 1) * H)));
    }
    return out;
  });
}
const textOf = (inf, mode) => (mode === "code" ? inf.code : inf.text);
async function embed(P) { return dec && BRAIN === "ettin" ? embedTexts(P.info.map((i) => textOf(i, TEXT)), P.state) : null; }

// ------------------------------------------------------------------ the two taste models
class Stats { // running mean and variance per dimension, over every candidate seen (labelled or not)
  constructor(d) { this.n = 0; this.m = new Float64Array(d); this.s = new Float64Array(d); }
  add(x) { this.n++; for (let i = 0; i < x.length; i++) { const d = x[i] - this.m[i]; this.m[i] += d / this.n; this.s[i] += d * (x[i] - this.m[i]); } }
  z(x) { const o = new Float64Array(x.length); for (let i = 0; i < x.length; i++) { const sd = this.n > 1 ? Math.sqrt(this.s[i] / (this.n - 1)) : 1; o[i] = (x[i] - this.m[i]) / (sd > 1e-6 ? sd : 1); } return o; }
}
class Logit { // weighted logistic regression with an L2 penalty, full-batch gradient descent, warm-started
  constructor(d) { this.d = d; this.w = new Float64Array(d); this.b = 0; }
  f(z) { let s = this.b; for (let i = 0; i < this.d; i++) s += this.w[i] * z[i]; return s; }
  fit(Z, y, wt, iters = 150) {
    const n = Z.length, W = wt.reduce((a, b) => a + b, 0) || 1;
    // step 1/L: the loss is (sum w_i CE_i + LAMBDA/2 |w|^2) / W, whose curvature is at most max|z|^2/4 + LAMBDA/W
    let zz = 0; for (const z of Z) { let q = 0; for (let i = 0; i < this.d; i++) q += z[i] * z[i]; zz = Math.max(zz, q); }
    const lr = 1 / (0.25 * zz + LAMBDA / W + 0.25);
    for (let it = 0; it < iters; it++) {
      const g = new Float64Array(this.d); let gb = 0;
      for (let j = 0; j < n; j++) { const p = 1 / (1 + Math.exp(-this.f(Z[j]))), e = (p - y[j]) * wt[j]; gb += e; for (let i = 0; i < this.d; i++) g[i] += e * Z[j][i]; }
      for (let i = 0; i < this.d; i++) this.w[i] -= lr * (g[i] + LAMBDA * this.w[i]) / W;
      this.b -= lr * gb / W;
    }
  }
}
const D_FEAT = fm.FEAT_NAMES.length;
const models = { feats: { stats: new Stats(D_FEAT), m: new Logit(D_FEAT) }, ettin: { stats: null, m: null } };
let examples = []; // { f: number[], r: Float32Array|null, y, w, kind, t }

function ettinModel(d) { if (!models.ettin.m) { models.ettin.stats = new Stats(d); models.ettin.m = new Logit(d); } return models.ettin; }
function refit() {
  const F = models.feats;
  F.m.fit(examples.map((e) => F.stats.z(e.f)), examples.map((e) => e.y), examples.map((e) => e.w));
  const withRows = examples.filter((e) => e.r);
  if (withRows.length) {
    const E = ettinModel(withRows[0].r.length);
    E.m.fit(withRows.map((e) => E.stats.z(e.r)), withRows.map((e) => e.y), withRows.map((e) => e.w));
  }
}

// ------------------------------------------------------------------ choosing
const recent = []; // [feats, score] of scored candidates, for the "leans" line
function choose(P) {
  const F = models.feats; P.info.forEach((i) => F.stats.add(i.feats));
  let X, M;
  if (P.rows) { M = ettinModel(P.rows[0].length); P.rows.forEach((r) => M.stats.add(r)); X = P.rows; } else { M = F; X = P.info.map((i) => i.feats); }
  const s = X.map((x) => M.m.f(M.stats.z(x)));
  const mx = Math.max(...s), e = s.map((v) => Math.exp((v - mx) / TAU)), sum = e.reduce((a, b) => a + b, 0);
  const p = e.map((v) => (v / sum) * (1 - EPS) + EPS / s.length);
  let u = Math.random(), k = 0; while (k < p.length - 1 && (u -= p[k]) > 0) k++;
  P.scores = s; P.probs = p; P.by = P.rows ? "ettin" : "feats";
  P.info.forEach((inf, i) => { recent.push([inf.feats, s[i]]); });
  while (recent.length > 240) recent.shift();
  return k;
}

// ------------------------------------------------------------------ reactions
function label(ph, y, w, kind) {
  const P = ph?.taste; if (!P || ph.tasteReacted) return false;
  ph.tasteReacted = kind;
  const k = P.chosen;
  examples.push({ f: P.info[k].feats, r: P.rows ? P.rows[k] : null, y, w, kind, t: Date.now() });
  if (examples.length > MAX_EX) examples = examples.slice(-MAX_EX);
  refit(); save(); paint();
  return true;
}
function like() {
  const ph = T.playing;
  if (!ph?.taste) return status("This phrase was not picked from candidates (the first ones after POWER are written plain). Wait a phrase.");
  if (ph.tasteReacted) return status("Already counted: " + ph.tasteReacted + ".");
  label(ph, 1, 1, "like"); status("Liked. More phrases like this one will come up.");
}
async function skip() {
  const ph = T.playing;
  if (ph?.taste && !ph.tasteReacted) label(ph, 0, 1, "skip");
  const ok = await fm.skipNow();
  status(ok ? "Skipped. The replacement starts at the next bar." : "Skip works while the station is on air.");
}
function onPhrase(ph, prev) { // a phrase left to finish counts as a mild yes
  if (prev?.taste && !prev.tasteReacted) label(prev, 1, W_PASSIVE, "pass");
  paint();
}

// ------------------------------------------------------------------ storage (this browser only)
const b64 = (f32) => { const u = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength); let s = ""; for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]); return btoa(s); };
const unb64 = (s) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); };
function save() {
  try { localStorage.setItem(STORE, JSON.stringify(examples.map((e) => ({ ...e, r: e.r ? b64(e.r) : null })))); } catch (e) { /* storage full or blocked: taste lasts this visit */ }
}
function load() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE) || "[]");
    examples = raw.filter((e) => Array.isArray(e.f) && e.f.length === D_FEAT).map((e) => ({ ...e, r: e.r ? unb64(e.r) : null }));
    for (const e of examples) { models.feats.stats.add(e.f); if (e.r) ettinModel(e.r.length).stats.add(e.r); }
    if (examples.length) refit();
  } catch (e) { examples = []; }
}

// ------------------------------------------------------------------ the panel
function status(t) { const el = $("t-status"); if (el) el.textContent = t; }
const pct = (x) => Math.round(x * 100) + "%";
const sig = (x) => 1 / (1 + Math.exp(-x));
let shown = null;
function paint(P = shown) {
  shown = P;
  const nl = examples.filter((e) => e.kind === "like").length, ns = examples.filter((e) => e.kind === "skip").length, np = examples.filter((e) => e.kind === "pass").length;
  $("t-count").textContent = `${nl} liked · ${ns} skipped · ${np} let play`;
  $("t-brain").textContent = BRAIN === "feats" ? "31 counted features per phrase" : dec ? `Ettin-17m on your GPU, reading ${TEXT === "code" ? "each candidate's code" : "a description of each candidate"}` : loading ? "loading the decision model…" : "31 counted features (the GPU model has not loaded)";
  const box = $("t-cands"); box.innerHTML = "";
  if (P && P.probs) {
    const order = P.probs.map((p, i) => i);
    for (const i of order) {
      const m = P.info[i].meta, row = document.createElement("div"); row.className = "tc" + (i === P.chosen ? " on" : "");
      const tag = [m.section, "e" + m.e, m.key, m.tilt > 0 ? "+energy" : m.tilt < 0 ? "−energy" : "", m.cue || "", i === 0 ? "(plain FM)" : ""].filter(Boolean).join(" · ");
      row.innerHTML = `<span class="tl"></span><span class="tb"><i style="width:${(P.probs[i] * 100).toFixed(1)}%"></i></span><span class="tp">${pct(P.probs[i])}</span>`;
      row.querySelector(".tl").textContent = (i === P.chosen ? "▶ " : "") + tag;
      row.title = P.info[i].text;
      box.append(row);
    }
    $("t-head").textContent = `Next phrase, picked from ${P.probs.length} by the ${P.by === "ettin" ? "Ettin" : "feature"} model (share of picks):`;
  } else $("t-head").textContent = "Candidates appear once the station is on air.";
  $("t-lean").textContent = leans();
  $("t-on").setAttribute("aria-pressed", String(T.on)); $("t-on").textContent = T.on ? "Taste on" : "Taste off";
}
function leans() { // which counted properties the current scores follow, by correlation over recent candidates
  if (examples.length < 4 || recent.length < 40) return "";
  const names = fm.FEAT_NAMES, n = recent.length, s = recent.map((r) => r[1]), ms = s.reduce((a, b) => a + b, 0) / n;
  const out = [];
  for (let j = 0; j < names.length; j++) {
    const x = recent.map((r) => r[0][j]), mx = x.reduce((a, b) => a + b, 0) / n;
    let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (s[i] - ms); sxx += (x[i] - mx) ** 2; syy += (s[i] - ms) ** 2; }
    if (sxx > 1e-9 && syy > 1e-9) out.push([names[j], sxy / Math.sqrt(sxx * syy)]);
  }
  const up = out.filter((o) => o[1] > 0.25).sort((a, b) => b[1] - a[1]).slice(0, 3).map((o) => o[0]);
  const dn = out.filter((o) => o[1] < -0.25).sort((a, b) => a[1] - b[1]).slice(0, 3).map((o) => o[0]);
  if (!up.length && !dn.length) return "No clear lean yet.";
  return "Leaning toward " + (up.join(", ") || "nothing in particular") + (dn.length ? "; away from " + dn.join(", ") : "") + ".";
}

// ------------------------------------------------------------------ wiring
T.on = true;
T.hooks = { embed, choose, committed: (P) => paint(P), phrase: onPhrase };
load();
$("t-like").onclick = like;
$("t-skip").onclick = skip;
$("t-on").onclick = () => { T.on = !T.on; if (!T.on) T.prepared = null; paint(); };
$("t-reset").onclick = () => {
  examples = []; models.feats.m = new Logit(D_FEAT); models.ettin.m = models.ettin.m ? new Logit(models.ettin.m.d) : null;
  try { localStorage.removeItem(STORE); } catch (e) {}
  status("Taste forgotten. The station picks evenly again."); paint();
};
if (BRAIN === "ettin") $("power").addEventListener("click", () => { ensureModel(); });
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "l" || e.key === "L" || e.key === "ArrowUp") { e.preventDefault(); like(); }
  if (e.key === "s" || e.key === "S" || e.key === "ArrowRight") { e.preventDefault(); skip(); }
});
paint();

// ------------------------------------------------------------------ harvest (run/harvest.mjs drives this)
if (Q.has("harvest")) {
  window.harvest = async (n = 10, opts = {}) => {
    if (opts.embed !== false) { await ensureModel(); if (!dec) throw new Error("decision model not loaded"); }
    T.on = true; T.hooks = { embed: null, choose: () => Math.floor(Math.random() * T.N), committed: null, phrase: null };
    const st = fm.station, styles = fm.STYLE_IDS.filter((s) => s !== "sleep"), out = [];
    let start = fm.lastPhrase ? fm.lastPhrase.start + 4 : 0, idx = opts.offset ?? 0;
    for (let i = 0; i < n; i++, idx++) {
      if (idx % 10 === 0) { // a new station every ten decision points
        const id = styles[Math.floor(idx / 10) % styles.length];
        st.style = id; st.bpm = fm.STYLES[id].bpm; st.mood = ["dark", "deep", "bright"][Math.floor(Math.random() * 3)];
        st.energy = [-1, 0, 1][Math.floor(Math.random() * 3)]; st.key = null;
      }
      const t0 = performance.now();
      const P = fm.prepare(start);
      // run/embed.py computes the same vectors on CPU (checked to 4e-7), so a harvest can skip the GPU
      const desc = opts.embed === false ? [] : await embedTexts(P.info.map((x) => x.text), P.state);
      const code = opts.embed === false ? [] : await embedTexts(P.info.map((x) => x.code), P.state);
      out.push({ i: idx, style: st.style, state: P.state, ms: performance.now() - t0,
        info: P.info.map(({ text, code, feats, meta }) => ({ text, code, feats, meta })), desc: desc.map(b64), code: code.map(b64) });
      fm.nextPhrase(start); start += 4; fm.bars.clear();
    }
    return out;
  };
}
