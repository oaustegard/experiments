// Edgewalk: keep the model stuck on "is it a sandwich?" while you change the food as far as you can.
// The model is a small network distilled from Jev's verdicts (run/train.py); it runs here in plain JS.

const TURNS = 12, PROBES = 3;
const EDGE = 0.05, FENCE = 0.15, FALL = 0.35;   // |p - 0.5| thresholds
const relay = window.relay || { log() {} };
const log = (m, a) => { try { relay.log(m, a); } catch (e) {} };
const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
};

// ---------- utilities ----------
function rngFrom(seed) {   // mulberry32 over a string hash
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) { h = Math.imul(h ^ seed.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); }
  let a = h >>> 0;
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- the model ----------
let M = null;
function predict(food) {
  let x = [];
  for (const pc of M.pieces) { const v = new Array(pc.values.length).fill(0); v[pc.values.indexOf(food[pc.key])] = 1; x = x.concat(v); }
  M.layers.forEach((L, li) => {
    const y = L.b.slice();
    for (let i = 0; i < y.length; i++) { const w = L.w[i]; let s = y[i]; for (let j = 0; j < x.length; j++) s += w[j] * x[j]; y[i] = s; }
    x = li < M.layers.length - 1 ? y.map(Math.tanh) : y;
  });
  return 1 / (1 + Math.exp(-x[0]));
}

// ---------- rules ----------
function zoneOf(p) {
  const d = Math.abs(p - 0.5);
  return d <= EDGE ? "edge" : d <= FENCE ? "fence" : d <= FALL ? "slip" : "fall";
}
const ZONE_TEXT = { edge: "on the edge ×2", fence: "on the fence", slip: "slipping", fall: "decided" };
const ZONE_EMOJI = { edge: "🟩", fence: "🟨", slip: "🟧", fall: "🟥" };
function pieceDist(pc, a, b) {
  if (a === b) return 0;
  if (!pc.ordered) return 1;
  return Math.abs(pc.values.indexOf(a) - pc.values.indexOf(b)) / (pc.values.length - 1);
}
const drift = (start, cur) => M.pieces.reduce((s, pc) => s + pieceDist(pc, start[pc.key], cur[pc.key]), 0) / M.pieces.length;

// ---------- describing a food ----------
const EMOJI = {
  "two slices of sandwich bread": "🍞", "a sub roll": "🥖", "a hot dog bun": "🌭", "a hamburger bun": "🍔", "a bagel": "🥯",
  "a croissant": "🥐", "a pita pocket": "🫓", "a flour tortilla": "🌯", "a crispy taco shell": "🌮", "two waffles": "🧇",
  "two chocolate chip cookies": "🍪", "two lettuce leaves": "🥬", "a sheet of nori": "🍙", "two rice cakes": "🍘",
  "two slices of pizza": "🍕", "two pancakes": "🥞",
};
function describe(f) {
  const sauce = f.sauce === "none" ? "" : f.sauce === "gravy poured over the top" ? ", with gravy poured over the top" : `, with ${f.sauce}`;
  const name = f.menu_name.startsWith("nothing") ? "The menu doesn't name it." : `The menu calls it ${f.menu_name}.`;
  return `${cap(f.outside)}, ${f.assembly}, filled with ${f.filling}${sauce}. Served ${f.temperature}, eaten with ${f.utensils}, ${f.size}, from ${f.venue}. ${name}`;
}
const cap = (s) => s[0].toUpperCase() + s.slice(1);

// ---------- game state ----------
const G = { mode: "daily", start: null, cur: null, hist: [], score: 0, probes: PROBES, locked: -1, slipping: false,
            over: false, open: -1, probing: false, probed: new Map(), ghost: null, seed: "" };

function newGame(mode) {
  G.mode = mode;
  G.seed = mode === "daily" ? `${today()}:sandwich` : `${Date.now()}:${Math.random()}`;
  const rnd = rngFrom(G.seed);
  const pick = M.starts[Math.floor(rnd() * M.starts.length)], start = pick.food;
  const an = /^[aeiou]/i.test(pick.from) ? "an" : "a";
  G.origin = pick.changes === 0 ? `${an} ${pick.from}, unchanged` : `${an} ${pick.from}, ${pick.changes} piece${pick.changes > 1 ? "s" : ""} changed`;
  Object.assign(G, { start: { ...start }, cur: { ...start }, hist: [], score: 0, probes: PROBES, locked: -1, slipping: false,
                     over: false, open: -1, probing: false, probed: new Map(), ghost: null });
  const p = predict(start);
  G.hist.push({ p, drift: 0, zone: zoneOf(p), pts: 0, move: null });
  if ($("end").open) $("end").close();
  log("new game", { mode, p: +p.toFixed(4), start });
  render();
}

function choose(i, v) {
  if (G.over || i === G.locked) return;
  const pc = M.pieces[i];
  if (v === G.cur[pc.key]) return;
  const next = { ...G.cur, [pc.key]: v };
  const p = predict(next);
  const key = `${i}:${v}`;
  if (G.probing) {
    if (G.probes <= 0 || G.probed.has(key)) return;
    G.probes--; G.probing = false;
    G.probed.set(key, p);
    G.ghost = { p, drift: drift(G.start, next) };
    log("probe", { piece: pc.key, value: v, p: +p.toFixed(4) });
    render();
    return;
  }
  G.cur = next; G.ghost = null; G.probed = new Map(); G.locked = i; G.open = -1;
  const d = drift(G.start, G.cur);
  let z = zoneOf(p);
  if (z === "slip" && G.slipping) z = "fall";
  G.slipping = z === "slip";
  const pts = z === "edge" ? Math.round(200 * d) : z === "fence" ? Math.round(100 * d) : 0;
  G.score += pts;
  G.hist.push({ p, drift: d, zone: z, pts, move: `${pc.label}: ${v}` });
  log("move", { turn: G.hist.length - 1, piece: pc.key, value: v, p: +p.toFixed(4), zone: z, drift: +d.toFixed(3), pts, score: G.score });
  if (z === "fall" || G.hist.length - 1 >= TURNS) G.over = true;
  render();
  if (G.over) setTimeout(showEnd, z === "fall" ? 900 : 500);
}

// ---------- rendering ----------
function render() {
  const h = G.hist[G.hist.length - 1], p = h.p;
  $("pct").textContent = `${Math.round(p * 100)}%`;
  const z = $("zone"); z.className = `zone z-${h.zone}`; z.textContent = ZONE_TEXT[h.zone];
  $("turn").textContent = G.hist.length - 1;
  $("drift").textContent = `${Math.round(h.drift * 100)}%`;
  $("score").textContent = G.score;
  $("probes").textContent = G.probes;
  $("best").textContent = store.get("edgewalk:best:sandwich") ?? "–";
  const pb = $("probeBtn"); pb.disabled = G.over || G.probes <= 0; pb.classList.toggle("on", G.probing);
  pb.textContent = G.probing ? "Probe: tap a value" : "Probe";
  const lean = p >= 0.5 ? "a sandwich" : "not a sandwich";
  const msg = $("msg"); msg.className = "msg";
  if (G.over) msg.textContent = h.zone === "fall" ? `The model made up its mind: ${lean}. Game over.` : "Twelve moves and it still can't decide. Game over.";
  else if (h.zone === "slip") { msg.className = "msg warn"; msg.textContent = `It's leaning “${lean}”. Your next move must pull it back to 35–65%.`; }
  else if (G.hist.length === 1) msg.textContent = "The model can't decide. Change anything to start walking.";
  else msg.textContent = `Last move: ${h.move} (+${h.pts})`;
  $("food").innerHTML = `<span class="emo">${EMOJI[G.cur.outside] || "🍽️"}</span><span>${esc(describe(G.cur))}</span>`;
  $("caseId").textContent = `${G.mode === "daily" ? `daily ${today()}` : "random"} · started as ${G.origin}`;
  const mini = $("mini"); mini.className = `zone z-${h.zone}`; mini.textContent = `${Math.round(p * 100)}% · ${ZONE_TEXT[h.zone]}`;
  renderPieces();
  renderPath();
}

function renderPieces() {
  const root = $("pieces"); root.innerHTML = "";
  M.pieces.forEach((pc, i) => {
    const v0 = G.start[pc.key], v = G.cur[pc.key], changed = v0 !== v, locked = i === G.locked && !G.over;
    const el = document.createElement("div");
    el.className = "piece" + (locked ? " locked" : "") + (G.open === i ? " open" : "");
    el.innerHTML = `<div class="ph"><span class="lbl">${esc(pc.label)}</span><span class="val${changed ? " changed" : ""}">${esc(v)}</span>${changed ? `<span class="was">${esc(v0)}</span>` : ""}${locked ? '<span class="lock">locked</span>' : ""}</div>`;
    el.querySelector(".ph").addEventListener("click", () => { if (locked || G.over) return; G.open = G.open === i ? -1 : i; renderPieces(); });
    if (G.open === i) {
      const opts = document.createElement("div"); opts.className = "opts";
      for (const o of pc.values) {
        const btn = document.createElement("button");
        btn.className = o === v ? "cur" : ""; btn.disabled = o === v;
        btn.textContent = o;
        const pr = G.probed.get(`${i}:${o}`);
        if (pr !== undefined) { const s = document.createElement("span"); s.className = "probe"; s.textContent = `→ ${Math.round(pr * 100)}%`; s.style.color = `var(--${zoneOf(pr)})`; btn.appendChild(s); }
        btn.addEventListener("click", (e) => { e.stopPropagation(); choose(i, o); });
        opts.appendChild(btn);
      }
      el.appendChild(opts);
    }
    root.appendChild(el);
  });
}

function renderPath() {
  const svg = $("canyon"), W = 400, H = 300, L = 14, R = 14, T = 16, B = 34;
  const maxD = Math.max(0.3, ...G.hist.map((h) => h.drift), G.ghost?.drift ?? 0);
  const top = Math.min(1, Math.ceil((maxD + 0.08) * 10) / 10);
  const X = (p) => L + p * (W - L - R), Y = (d) => H - B - (d / top) * (H - T - B);
  const band = (a, b, fill) => `<rect x="${X(a)}" y="${T}" width="${X(b) - X(a)}" height="${H - T - B}" fill="${fill}"/>`;
  let s = "";
  s += band(0, 0.15, "var(--abyss)") + band(0.85, 1, "var(--abyss)");
  s += band(0.15, 0.35, "var(--slope)") + band(0.65, 0.85, "var(--slope)");
  s += band(0.35, 0.65, "var(--ground)") + band(0.45, 0.55, "var(--ridge)");
  for (const x of [0.15, 0.35, 0.65, 0.85]) s += `<line x1="${X(x)}" x2="${X(x)}" y1="${T}" y2="${H - B}" stroke="var(--line)" stroke-dasharray="3 3"/>`;
  s += `<line x1="${X(0.5)}" x2="${X(0.5)}" y1="${T}" y2="${H - B}" stroke="var(--edge)" stroke-opacity=".5"/>`;
  for (let d = 0.1; d < top + 1e-9; d += 0.1) {
    const y = Y(d);
    s += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="var(--line)" stroke-opacity=".6"/>`;
    s += `<text x="${L + 3}" y="${y - 3}" font-size="9" fill="var(--muted)">${Math.round(d * 100)}%</text>`;
  }
  s += `<text x="${X(0.5)}" y="${T - 4}" font-size="10" fill="var(--muted)" text-anchor="middle">how far the food has changed ↑</text>`;
  s += `<text x="${L}" y="${H - B + 14}" font-size="11" fill="var(--muted)">not a sandwich</text>`;
  s += `<text x="${W - R}" y="${H - B + 14}" font-size="11" fill="var(--muted)" text-anchor="end">sandwich</text>`;
  s += `<text x="${X(0.5)}" y="${H - B + 14}" font-size="11" fill="var(--muted)" text-anchor="middle">50%</text>`;
  s += `<text x="${X(0.5)}" y="${H - B + 28}" font-size="10" fill="var(--muted)" text-anchor="middle">the model's verdict →</text>`;
  const pts = G.hist.map((h) => [X(h.p), Y(h.drift)]);
  if (pts.length > 1) s += `<polyline points="${pts.map((q) => q.join(",")).join(" ")}" fill="none" stroke="var(--ink)" stroke-opacity=".55" stroke-width="1.6" stroke-linejoin="round"/>`;
  G.hist.slice(0, -1).forEach((h, i) => { s += `<circle cx="${pts[i][0]}" cy="${pts[i][1]}" r="3.2" fill="var(--${h.zone})"/>`; });
  if (G.ghost) {
    const [x0, y0] = pts[pts.length - 1], gx = X(G.ghost.p), gy = Y(G.ghost.drift);
    s += `<line x1="${x0}" y1="${y0}" x2="${gx}" y2="${gy}" stroke="var(--accent)" stroke-dasharray="4 3"/>`;
    s += `<circle cx="${gx}" cy="${gy}" r="6" fill="none" stroke="var(--accent)" stroke-width="2" stroke-dasharray="3 2"/>`;
  }
  const h = G.hist[G.hist.length - 1], [cx, cy] = pts[pts.length - 1];
  if (h.zone === "fall") {
    const ex = h.p >= 0.5 ? W - R - 4 : L + 4;
    s += `<circle cx="${cx}" cy="${cy}" r="8" fill="var(--fall)"><animate attributeName="cx" from="${cx}" to="${ex}" dur="0.8s" fill="freeze"/><animate attributeName="cy" from="${cy}" to="${H - B - 6}" dur="0.8s" fill="freeze" calcMode="spline" keySplines="0.5 0 1 1" keyTimes="0;1"/></circle>`;
  } else {
    s += `<circle cx="${cx}" cy="${cy}" r="8" fill="var(--${h.zone})" stroke="var(--panel)" stroke-width="2"/>`;
    if (!G.over) s += `<circle cx="${cx}" cy="${cy}" r="8" fill="none" stroke="var(--${h.zone})" stroke-width="2"><animate attributeName="r" values="8;16" dur="1.6s" repeatCount="indefinite"/><animate attributeName="stroke-opacity" values=".8;0" dur="1.6s" repeatCount="indefinite"/></circle>`;
  }
  svg.innerHTML = s;
}

function resultText() {
  const h = G.hist[G.hist.length - 1], maxD = Math.max(...G.hist.map((x) => x.drift));
  return `Edgewalk: is it a sandwich? · ${G.mode === "daily" ? today() : "random"}\n${G.score} pts · ${Math.round(maxD * 100)}% changed · ${G.hist.length - 1} moves\n${G.hist.slice(1).map((x) => ZONE_EMOJI[x.zone]).join("")}${h.zone === "fall" ? "" : " 🥪"}\n${EMOJI[G.start.outside] || "🍽️"} → ${EMOJI[G.cur.outside] || "🍽️"} ${describe(G.cur)}\n${location.origin}${location.pathname}`;
}

function showEnd() {
  const h = G.hist[G.hist.length - 1], fell = h.zone === "fall";
  const prev = +(store.get("edgewalk:best:sandwich") ?? -1);
  if (G.score > prev) store.set("edgewalk:best:sandwich", String(G.score));
  $("endTitle").textContent = fell ? (h.p >= 0.5 ? "Verdict: sandwich" : "Verdict: not a sandwich") : "Still undecided";
  $("endSub").textContent = fell ? `Move ${G.hist.length - 1} settled it at ${Math.round(h.p * 100)}%.` : `Twelve moves, and the model still says ${Math.round(h.p * 100)}%.`;
  $("endEmoji").textContent = G.hist.slice(1).map((x) => ZONE_EMOJI[x.zone]).join("");
  const maxD = Math.max(...G.hist.map((x) => x.drift));
  $("endStats").innerHTML = `<b>${G.score}</b> points · up to ${Math.round(maxD * 100)}% changed${G.score > prev && prev >= 0 ? " · new best" : ""}`;
  $("endFrom").textContent = describe(G.start);
  $("endTo").textContent = describe(G.cur);
  log("game over", { mode: G.mode, score: G.score, fell, moves: G.hist.length - 1, maxDrift: +maxD.toFixed(3), end: G.cur });
  $("end").showModal();
}

// ---------- wiring ----------
function modeButtons() { for (const b of $("modeSeg").children) b.setAttribute("aria-pressed", String(b.dataset.mode === G.mode)); }

async function main() {
  if (!store.get("edgewalk:seen")) { $("rules").open = true; store.set("edgewalk:seen", "1"); }
  try { M = await fetch("sandwich.json").then((r) => r.json()); }
  catch (e) { $("msg").textContent = `Could not load the model: ${e.message || e}`; log("load failed", { error: String(e) }); return; }
  G.mode = new URLSearchParams(location.search).get("mode") === "random" ? "random" : "daily";
  modeButtons();
  for (const b of $("modeSeg").children) b.addEventListener("click", () => { G.mode = b.dataset.mode; modeButtons(); newGame(G.mode); });
  $("probeBtn").addEventListener("click", () => { if (G.over || G.probes <= 0) return; G.probing = !G.probing; render(); });
  $("againBtn").addEventListener("click", () => { G.mode = "random"; modeButtons(); newGame("random"); });
  $("closeBtn").addEventListener("click", () => $("end").close());
  $("copyBtn").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(resultText()); $("copyBtn").textContent = "Copied"; } catch (e) { $("copyBtn").textContent = "Copy failed"; }
    setTimeout(() => ($("copyBtn").textContent = "Copy result"), 1500);
  });
  newGame(G.mode);
}

window.__edgewalk = { G, choose, predict, zoneOf, drift, get M() { return M; } };   // test hook
main();
