// DOOM Arena: the TRM bot plays gPu(doom) on this page's GPU, level after level, and ships a log of
// every attempt to the GPU-bridge repo (doom-logs/<session>/), commenting on the wake PR so the Claude
// session picks it up. The game runs in lockstep with the bot: each tic waits for the bot's input, so
// a slow GPU slows the game instead of leaving the bot behind it.
//
// ?doom=<url of the gPu(doom) directory> (default /fun-and-games/doom/), ?weights=<url without .bin/.json>,
// ?map= ?skill= ?stuck= ?cap= ?tries= ?fast preset the controls, ?auto starts at once, ?planner=oracle swaps
// TRM for exact BFS (a debug stand-in, never for reported runs).

import { Bot } from "./bot.js?v=20261030-122108";
import { TRMPlanner } from "./trm.js?v=20261030-122108";
import { OraclePlanner } from "./oracle.js?v=20261030-122108";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const DOOM = new URL(params.get("doom") || "/fun-and-games/doom/", location.href).href;
const WEIGHTS = params.get("weights") || DOOM + "bot/trm";
const API = "https://api.github.com", BRANCH = "main";
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
const status = (t, err) => { $("#status").textContent = t; $("#status").className = err ? "err" : ""; };
const logLine = (t) => { const el = $("#log"); el.textContent = `${new Date().toLocaleTimeString()} ${t}\n` + el.textContent; };

// ---------------------------------------------------------------- GitHub (the bridge's token and repo)
const store = {
  get(k) { try { return localStorage.getItem("gpubridge:" + k) ?? sessionStorage.getItem("gpubridge:" + k); } catch { return null; } },
  set(k, v, persist) { try { (persist ? localStorage : sessionStorage).setItem("gpubridge:" + k, v); } catch {} },
};
let cfg = { token: store.get("token") || "", repo: store.get("repo") || "oaustegard/gpu-bridge-comms" }, wakePR;
async function gh(path, { method = "GET", body, retry = true } = {}) {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(API + path, { method, cache: "no-store", body: body ? JSON.stringify(body) : undefined,
        headers: { Authorization: "Bearer " + cfg.token, Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", ...(body ? { "Content-Type": "application/json" } : {}) } });
      if (!res.ok) { const e = new Error(`${method} ${path}: HTTP ${res.status}`); e.status = res.status; throw e; }
      return res.status === 204 ? null : await res.json();
    } catch (e) {
      if (!retry || attempt >= 3 || (e.status && e.status !== 429 && e.status < 500)) throw e;
      await delay(1000 * 3 ** attempt);
    }
  }
}
const R = () => "/repos/" + cfg.repo;
function b64(bytes) { let s = ""; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000)); return btoa(s); }
// {path: {text} | {bytes}} in one commit on main, rebuilt when the head moved meanwhile
async function commitFiles(files, message) {
  const tree = [];
  for (const [path, f] of Object.entries(files)) {
    const body = f.bytes ? { content: b64(f.bytes), encoding: "base64" } : { content: f.text, encoding: "utf-8" };
    tree.push({ path, mode: "100644", type: "blob", sha: (await gh(`${R()}/git/blobs`, { method: "POST", body })).sha });
  }
  for (let attempt = 0; attempt < 6; attempt++) {
    const head = (await gh(`${R()}/git/ref/heads/${BRANCH}`)).object.sha;
    const base = (await gh(`${R()}/git/commits/${head}`)).tree.sha;
    const t = (await gh(`${R()}/git/trees`, { method: "POST", body: { base_tree: base, tree } })).sha;
    const c = (await gh(`${R()}/git/commits`, { method: "POST", body: { message, tree: t, parents: [head] } })).sha;
    try { await gh(`${R()}/git/refs/heads/${BRANCH}`, { method: "PATCH", body: { sha: c, force: false } }); return c; }
    catch (e) { if (e.status !== 422 && e.status !== 409) throw e; await delay(400 * (attempt + 1)); }
  }
  throw new Error("the branch kept moving");
}
async function wake(text) {
  if (!$("#wake").checked) return;
  const pr = await findWakePR();
  if (pr) try { await gh(`${R()}/issues/${pr}/comments`, { method: "POST", body: { body: text }, retry: false }); } catch (e) { console.warn(e); }
}
// a failed upload is kept and retried with the next one, so a dropped connection doesn't lose an attempt
const outbox = [];
let shipped = 0, lastSha = "";
async function ship(files, message, wakeText) {
  if (files) { outbox.push({ files, message, wakeText }); showUploads(); }
  if (!cfg.token || ship.busy) return;
  ship.busy = true;
  try { await flush(); } finally { ship.busy = false; }
}
async function flush() {
  while (outbox.length) {
    const job = outbox[0];
    try {
      const sha = await commitFiles(job.files, job.message); outbox.shift(); shipped++; lastSha = sha.slice(0, 7); showUploads();
      if (job.wakeText) await wake(`${job.wakeText} at ${sha.slice(0, 7)}`);
    }
    catch (e) { logLine("upload failed, will retry: " + e.message); showUploads(); if (e.status === 401 || e.status === 403) verify(); return; }
  }
}

// ---------------------------------------------------------------- version: which build is this, and is there a newer one?
let VERSION = "unknown", VNOTE = "";
async function deployed() { try { return await (await fetch(new URL("version.json", import.meta.url), { cache: "no-store" })).json(); } catch { return null; } }
async function checkVersion() {
  const v = await deployed();
  if (!v || v.version === VERSION || VERSION === "unknown") return;
  $("#newvertext").textContent = `Version ${v.version} is deployed${v.note ? ` (${v.note})` : ""}; this page runs ${VERSION}${running ? ". It reloads when you press Reload; the current attempt is lost" : ""}`;
  $("#newver").hidden = false;
}
$("#reload").addEventListener("click", () => location.reload());

// ---------------------------------------------------------------- game
let device, context, format, eng, doom, weights, man, planner, globals = new Map(), fast = false;
const SESSION = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19) + "-" + Math.random().toString(36).slice(2, 6);
const deviceLabel = () => /iPhone/.test(navigator.userAgent) ? "iPhone" : /iPad/.test(navigator.userAgent) ? "iPad" : /Android/.test(navigator.userAgent) ? "Android" : /Mac/.test(navigator.userAgent) ? "Mac" : /Windows/.test(navigator.userAgent) ? "Windows" : "Linux";

async function init() {
  const v = await deployed();
  // the version this page loaded is the one stamped into its own URL, not whatever is deployed now
  VERSION = new URL(import.meta.url).searchParams.get("v") || v?.version || "unknown"; VNOTE = v?.version === VERSION ? v.note || "" : "";
  $("#ver").textContent = `version ${VERSION}${VNOTE ? " · " + VNOTE : ""}`;
  setInterval(checkVersion, 60000);
  if (!navigator.gpu) throw new Error("this browser has no WebGPU");
  const ad = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  device = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
  device.lost.then((i) => status("GPU device lost: " + i.message, true));
  context = $("#screen").getContext("webgpu"); format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });
  status("Loading the engine…");
  doom = { engine: await import(DOOM + "engine.js"), info: await import(DOOM + "info.js"), specials: await import(DOOM + "specials.js") };
  const wad = await doom.engine.unpackWad(await (await fetch(DOOM + "e1.wad.gz")).arrayBuffer());
  eng = await doom.engine.Engine.create(device, wad, { width: 640, height: 400, format, maxWidth: 640, maxHeight: 400 });
  for (const m of eng.maps()) $("#map").add(new Option(m, m));
  status("Loading TRM…");
  [weights, man] = await Promise.all([fetch(WEIGHTS + ".bin").then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)), fetch(WEIGHTS + ".json").then((r) => r.json())]);
  planner = await TRMPlanner.create(device, weights, man);
  const t0 = performance.now();
  await planner.plan(new Uint8Array(1024).fill(2), { sup: 1 }); await planner.plan(new Uint8Array(1024).fill(2), { sup: 1 });
  fast = (performance.now() - t0) / 2 < 120; // whole-level plans only where the GPU can afford them
  status(`Ready on ${deviceLabel()}: ${fast ? "window and whole-level plans" : "window plans only (slow GPU)"}.`);
}

async function readState() {
  const enc = device.createCommandEncoder(), out = {};
  const parts = { world: [eng.worldBuf, eng.worldBuf.size], sec: [eng.secBuf, eng.secBuf.size], mobj: [eng.mobjBuf, eng.mobjBuf.size], ls: [eng.lsBuf, eng.map.lines.length * 4] };
  const rbs = Object.entries(parts).map(([k, [b, size]]) => {
    const rb = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(b, 0, rb, 0, rb.size); return [k, rb];
  });
  device.queue.submit([enc.finish()]);
  for (const [k, rb] of rbs) { await rb.mapAsync(GPUMapMode.READ); out[k] = rb.getMappedRange().slice(0); rb.unmap(); rb.destroy(); }
  return out;
}

// a small PNG of the current view (tics = 0: draws without advancing the game)
async function thumb() {
  const s = await eng.snapshot(0, { frac: 1 });
  const big = new OffscreenCanvas(eng.width, eng.height); big.getContext("2d").putImageData(new ImageData(s.rgba, eng.width, eng.height), 0, 0);
  const small = new OffscreenCanvas(240, 150); const c = small.getContext("2d"); c.imageSmoothingEnabled = false; c.drawImage(big, 0, 0, 240, 150);
  return new Uint8Array(await (await small.convertToBlob({ type: "image/png" })).arrayBuffer());
}

let running = false, attemptNo = 0;
const usedSeeds = {};
async function playLevel(map, carry, opts) {
  const { W, freshPlayer } = doom.engine;
  // a retry gets a fresh random seed (the monsters' choices), or every try of a level replays the last one
  // DOOM's generator has 256 starting points, so a retry also skips seeds this level already had (E1M4 drew 79 twice)
  let seed = null;
  if (!carry) { const used = (usedSeeds[map] ||= new Set()); do seed = crypto.getRandomValues(new Uint8Array(1))[0]; while (used.has(seed) && used.size < 256); used.add(seed); }
  await eng.loadMap(map, carry || { ...freshPlayer(), rng: seed });
  const plan = { plans: 0, ms: 0 };
  const oracle = params.get("planner") === "oracle" ? new OraclePlanner() : null;
  const local = { async plan(g) { const r = oracle ? await oracle.plan(g) : await planner.plan(g, { sup: 4, minSup: 2 }); plan.plans++; plan.ms += r.ms; return r; } };
  const bot = new Bot({ wad: eng.wad, map: eng.map, MOBJINFO: doom.info.MOBJINFO, MF: doom.info.MF, LINE_SPECIALS: doom.specials.LINE_SPECIALS, TR: doom.specials.TR, W }, local);
  if (oracle) bot.globalPlanner = async () => oracle;
  else if (fast) bot.globalPlanner = async (n) => { if (!globals.has(n)) globals.set(n, await TRMPlanner.create(device, weights, man, { w: n, h: n })); return globals.get(n); };
  const t0 = performance.now(), thumbs = [], trace = [];
  let result = "running", world = null, tic = 0, bestScore = -1, bestTic = 0, last = performance.now();
  const score = () => { const p = bot.s.p; let v = 0; for (const x of bot.visited) v += x; return p.kills * 3 + p.items * 2 + v + Object.values(bot.stats.targets).reduce((a, b) => a + b, 0) * 2; };
  while (running) {
    const st = await readState(); world = st.world; bot.readState(st);
    const p = bot.s.p;
    if (p.exit) { result = p.exit === 2 ? "secret-exit" : "exit"; break; }
    if (p.dead) { result = "dead"; break; }
    const sc = score();
    if (sc > bestScore) { bestScore = sc; bestTic = tic; }
    if (tic - bestTic > opts.stuck * 35) { result = "stuck"; break; } // the game-side time out: no progress for a while
    if (tic > opts.cap * 60 * 35) { result = "time-limit"; break; }
    if (tic % 35 === 0) {
      trace.push([tic, Math.round(p.x), Math.round(p.y), p.health]);
      const d = bot.debug();
      status(`${map} try ${opts.tryNo} · ${Math.floor(tic / 35)} s · hp ${p.health} · kills ${p.kills} · ${d.target ? d.target.kind : "looking"} · ${plan.plans} plans, ${(plan.ms / Math.max(1, plan.plans)).toFixed(0)} ms`);
    }
    if (tic % (35 * 20) === 0 && thumbs.length < 30) thumbs.push([tic, await thumb()]);
    const input = await bot.step();
    const enc = device.createCommandEncoder();
    eng.encodeFrame(enc, 1, { ...input, frac: 1 });
    eng.present(enc, context.getCurrentTexture().createView());
    device.queue.submit([enc.finish()]);
    tic++;
    if (!$("#fast").checked) { const wait = 1000 / 35 - (performance.now() - last); if (wait > 1) await delay(wait); }
    else if (tic % 4 === 0) await new Promise((r) => requestAnimationFrame(r)); // let the page draw
    last = performance.now();
  }
  await device.queue.onSubmittedWorkDone();
  if (running) thumbs.push([tic, await thumb()]);
  const wi = new Int32Array(world);
  return {
    map, seed, result: running ? result : "stopped", tic, seconds: +(tic / 35).toFixed(1), wallS: Math.round((performance.now() - t0) / 1000),
    kills: `${wi[W.killcount]}/${wi[W.totalkills]}`, items: `${wi[W.itemcount]}/${wi[W.totalitems]}`, secrets: `${wi[W.secretcount]}/${wi[W.totalsecrets]}`, health: wi[W.health],
    plans: plan.plans, planMs: Math.round(plan.ms / Math.max(1, plan.plans)), stats: bot.stats, events: bot.events.slice(-500), trace,
    carry: result === "exit" || result === "secret-exit" ? eng.playerFrom(world) : null, thumbs,
  };
}

const NEXT = { E1M1: "E1M2", E1M2: "E1M3", E1M3: "E1M4", E1M4: "E1M5", E1M5: "E1M6", E1M6: "E1M7", E1M7: "E1M8", E1M9: "E1M4" };
const SECRET = { E1M3: "E1M9" };

async function campaign() {
  running = true; $("#play").disabled = true; $("#stop").disabled = false;
  const opts = { stuck: +$("#stuck").value, cap: +$("#cap").value, tries: +$("#tries").value };
  eng.skill = +$("#skill").value;
  let map = $("#map").value, carry = null, tryNo = 1;
  const meta = { session: SESSION, version: VERSION, note: VNOTE, device: deviceLabel(), ua: navigator.userAgent, skill: eng.skill, opts, weights: WEIGHTS, global: fast, startedAt: new Date().toISOString() };
  ship({ [`doom-logs/${SESSION}/session.json`]: { text: JSON.stringify(meta, null, 1) } }, `doom-arena: ${SESSION} started on ${meta.device}`);
  logLine(`session ${SESSION}`);
  while (running && map) {
    attemptNo++;
    const r = await playLevel(map, carry, { ...opts, tryNo });
    if (r.result === "stopped") break;
    const n = String(attemptNo).padStart(3, "0"), base = `doom-logs/${SESSION}/${n}-${map}-${r.result}`;
    const files = { [base + ".json"]: { text: JSON.stringify({ ...r, thumbs: r.thumbs.map(([t]) => t), tryNo, skill: eng.skill, version: VERSION }, null, 0) } };
    for (const [t, png] of r.thumbs) files[`${base}-t${String(t).padStart(6, "0")}.png`] = { bytes: png };
    const line = `${map} try ${tryNo}: ${r.result} at ${r.seconds} s, kills ${r.kills}, hp ${r.health}`;
    logLine(line);
    ship(files, `doom-arena: ${line}`, `doom-arena \`${SESSION}\` (v ${VERSION}): ${line} — \`${base}.json\``);
    if (r.result === "exit" || r.result === "secret-exit") { map = (r.result === "secret-exit" ? SECRET[map] : null) || NEXT[map]; carry = r.carry; tryNo = 1; }
    else if (tryNo < opts.tries) { tryNo++; carry = null; }             // pistol start on a retry
    else { logLine(`${map}: giving up after ${tryNo} tries`); map = NEXT[map]; carry = null; tryNo = 1; }
  }
  running = false; $("#play").disabled = false; $("#stop").disabled = true;
  status(map ? "Stopped." : "Episode over.");
  ship({ [`doom-logs/${SESSION}/end.json`]: { text: JSON.stringify({ endedAt: new Date().toISOString(), attempts: attemptNo, lastMap: map }) } }, `doom-arena: ${SESSION} ended`, `doom-arena \`${SESSION}\` ended after ${attemptNo} attempts`);
}

// ---------------------------------------------------------------- page
$("#play").addEventListener("click", () => campaign().catch((e) => { running = false; status(e.message, true); console.error(e); $("#play").disabled = false; }));
$("#stop").addEventListener("click", () => { running = false; });
let lock = null;
$("#lock").addEventListener("click", async (e) => {
  try { if (lock) { await lock.release(); lock = null; } else { lock = await navigator.wakeLock.request("screen"); lock.addEventListener("release", () => { lock = null; e.target.textContent = "Keep screen awake"; }); } } catch {}
  e.target.textContent = lock ? "Screen stays awake" : "Keep screen awake";
});
// connection state, checked against GitHub rather than assumed from a stored token
const conn = (cls, text) => { const el = $("#connstate"); el.className = "pill " + cls; el.textContent = text; };
async function verify() {
  if (!cfg.token) { conn("", "Not connected: logs stay on this page"); $("#change").textContent = "Connect to GitHub"; return false; }
  conn("wait", `Connecting to ${cfg.repo}…`);
  try {
    const repo = await gh(R(), { retry: false });
    if (repo.permissions && !repo.permissions.push) throw Object.assign(new Error(`the token can read ${cfg.repo} but not write to it`), { status: 403 });
    let login = ""; try { login = (await gh("/user", { retry: false })).login; } catch {}
    wakePR = undefined;
    const pr = await findWakePR();
    conn("ok", `✓ Connected to ${cfg.repo}${login ? " as " + login : ""}, can write${pr ? `, wakes the session on PR #${pr}` : ", but no wake PR is open (logs still upload)"}`);
    $("#change").textContent = "Change token";
    ship(null);
    return true;
  } catch (err) {
    conn("bad", "✗ " + (err.status === 401 ? "GitHub refused the token (401): paste it again" : err.status === 404 ? `${cfg.repo} not found, or the token can't see it (404)` : err.status === 403 ? err.message : "Couldn't reach GitHub: " + err.message));
    $("#conn").hidden = false; $("#change").textContent = "Change token";
    return false;
  }
}
async function findWakePR() {
  if (wakePR === undefined) {
    try { wakePR = (await gh(`${R()}/pulls?state=open&per_page=50`)).find((pr) => /^wake(\/|$)/.test(pr.head?.ref || ""))?.number ?? null; }
    catch { wakePR = null; }
  }
  return wakePR;
}
function showUploads() {
  const q = outbox.length;
  $("#uploads").textContent = (shipped ? `${shipped} log${shipped === 1 ? "" : "s"} uploaded, last at ${lastSha}` : "") + (q ? `${shipped ? " · " : ""}${q} waiting to upload` : "");
}
$("#change").addEventListener("click", () => { $("#conn").hidden = !$("#conn").hidden; if (!$("#conn").hidden) $("#token").focus(); });
$("#conn").addEventListener("submit", async (e) => {
  e.preventDefault();
  const token = $("#token").value.trim();
  if (!token) { conn("bad", "✗ Paste the token first"); return; }
  cfg = { token, repo: $("#repo").value.trim() || cfg.repo };
  $("#connect").disabled = true;
  if (await verify()) {
    store.set("token", cfg.token, $("#remember").checked); store.set("repo", cfg.repo, true);
    $("#conn").hidden = true; $("#token").value = "";
  }
  $("#connect").disabled = false;
});
$("#repo").value = cfg.repo;
verify();
setInterval(() => ship(null), 60000); // retry uploads that failed
init().then(() => {
  for (const k of ["map", "skill", "stuck", "cap", "tries"]) if (params.has(k)) $("#" + k).value = params.get(k);
  if (params.has("fast")) $("#fast").checked = true;
  $("#play").disabled = false;
  if (params.has("auto")) $("#play").click();
}).catch((e) => { status(e.message, true); console.error(e); });
window.ARENA = { get running() { return running; }, get attempts() { return attemptNo; }, outbox, session: SESSION, get log() { return $("#log").textContent; } };
