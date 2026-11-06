// Run a trained decision spike on any web page.
//
//   import { loadDecider } from "https://austegard.com/experiments/decider-lora-creator/web/embed.js";
//   const decider = await loadDecider({ spike: "./my_decision.json" });   // my_decision.bin next to it
//   const answer = await decider.decide("the case, as text or JSON");
//   // { questionName: { label: "refund", confidence: 0.81, probabilities: { refund: 0.81, other: 0.19 } }, ... }
//
// The base model (Ettin-17m, 11 MB, fetched once and then cached by the browser), the WebGPU engine and the
// tokenizer come from this site unless `base` names another copy. A spike runs on the base it was trained
// on: its .json names it ("base": "base31" for spikes made since 2026-11-06); a spike without the field was
// trained on the original ±63 base, models/base/, which stays published for it. The spike is the pair the creator page
// exports: a .json manifest (questions, calibration, tensor offsets) and a .bin of weights. Pass it as a URL
// to the .json (the .bin is found beside it), or as { sm, bytes } you already hold. Needs WebGPU (Chrome or
// Edge 113+, Safari 26); loadDecider throws a plain error without it.

import { SpikeEngine } from "../../adapter-library/web/engine.js";
import { loadTokenizer, optionsOf, softmax } from "./creator.js";

const MODELS = new URL("../../adapter-library/web/models/", import.meta.url).href;

async function fetchOk(url, kind) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return kind === "json" ? r.json() : new Uint8Array(await r.arrayBuffer());
}

export async function loadDecider({ spike, base = null, device = null, onProgress = () => {} } = {}) {
  if (!spike) throw new Error("loadDecider needs a spike: the URL of its .json, or { sm, bytes }");
  if (!device && !navigator.gpu) throw new Error("This browser has no WebGPU, which the decision model needs (Chrome or Edge 113+, Safari 26).");
  onProgress("downloading");
  const sp = typeof spike === "string" || spike instanceof URL
    ? await (async () => { const j = new URL(spike, location.href); return { sm: await fetchOk(j, "json"), bytes: await fetchOk(new URL(j.href.replace(/\.json(\?.*)?$/, ".bin$1")), "bin") }; })()
    : { sm: spike.sm, bytes: spike.bytes instanceof Uint8Array ? spike.bytes : new Uint8Array(spike.bytes) };
  if (!sp.sm?.questions || !sp.sm?.layers) throw new Error("that is not a decision spike: its .json has no questions or layers");
  base = base ?? MODELS + (sp.sm.base ?? "base") + "/";
  base = base.endsWith("/") ? base : base + "/";
  const [manifest, weights, tok] = await Promise.all([fetchOk(base + "manifest.json", "json"), fetchOk(base + "weights.bin", "bin"), loadTokenizer(base + "tokenizer.json")]);
  if (!device) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter is available on this device.");
    device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
  }
  onProgress("starting");
  const engine = await SpikeEngine.create(device, manifest, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength), { maxTokens: 4096, maxSeqs: 16, maxPos: 2048 });
  engine.setSpike(sp.sm, sp.bytes);
  const questions = Object.entries(sp.sm.questions), temps = sp.sm.temps ?? {};

  // one case -> one sequence per question; several cases are packed into batches under the engine's limits
  async function decideMany(states) {
    const seqs = [];
    states.forEach((state, ci) => {
      const text = typeof state === "string" ? state : JSON.stringify(state);
      for (const [name, q] of questions) seqs.push({ ci, name, q, opts: optionsOf(q), ...tok.build(q.type, q.instructions, optionsOf(q), text) });
    });
    const out = states.map(() => ({})), batch = [];
    let tokens = 0, pos = 0;
    const flush = async () => {
      if (!batch.length) return;
      const z = await engine.score(batch);
      batch.forEach((s, i) => {
        const p = softmax(z[i], temps[s.q.type] ?? 1), names = s.opts.map(([o]) => o);
        const top = p.indexOf(Math.max(...p));
        out[s.ci][s.name] = { label: names[top], confidence: p[top], probabilities: Object.fromEntries(names.map((o, j) => [o, p[j]])) };
      });
      batch.length = 0; tokens = 0; pos = 0;
    };
    for (const s of seqs) {
      if (batch.length && (tokens + s.ids.length > 4096 || batch.length >= 16 || pos + s.pos.length > 2048)) await flush();
      batch.push(s); tokens += s.ids.length; pos += s.pos.length;
    }
    await flush();
    return out;
  }

  onProgress("ready");
  return {
    id: sp.sm.id, description: sp.sm.description, questions: sp.sm.questions,
    decide: async (state) => (await decideMany([state]))[0],
    decideMany,
    destroy: () => engine.destroy?.(),
  };
}

// A complete page around one spike, with the spike inlined so it is a single file to host or open:
// a text box, an Answer button, and a bar per option. embedUrl is where this module is served from.
export function standalonePage(sm, bytes, embedUrl) {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const title = (sm.id || "decision").replace(/_/g, " ");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<style>
:root { --bg: #fbfaf7; --panel: #fff; --fg: #1d1c19; --muted: #6e6c64; --line: #e4e2db; --accent: #2f6db3; --track: #f3f1ec; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #171715; --panel: #201f1c; --fg: #efede7; --muted: #a6a39a; --line: #34322d; --accent: #6aa5e8; --track: #2a2925; } }
:root[data-theme="dark"] { --bg: #171715; --panel: #201f1c; --fg: #efede7; --muted: #a6a39a; --line: #34322d; --accent: #6aa5e8; --track: #2a2925; }
* { box-sizing: border-box; } body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 24px 16px 48px; } h1 { font-size: 22px; margin: 0 0 4px; } .muted { color: var(--muted); font-size: 13px; }
textarea { width: 100%; font: 13px/1.45 ui-monospace, monospace; padding: 8px; color: var(--fg); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }
button { font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid var(--accent); background: var(--accent); color: #fff; cursor: pointer; } button:disabled { opacity: .5; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 10px 0; } .card h3 { margin: 0 0 6px; font-size: 14px; }
.opt { display: grid; grid-template-columns: minmax(90px, 32%) 1fr 44px; gap: 8px; align-items: center; font-size: 13px; margin: 3px 0; }
.track { height: 8px; background: var(--track); border-radius: 3px; overflow: hidden; } .track div { height: 100%; background: var(--accent); } .pct { text-align: right; }
#err { color: #b3452f; white-space: pre-wrap; }
</style></head>
<body><main>
<h1>${esc(title)}</h1>
<p class="muted">${esc(sm.description || "")} Runs on your GPU in this tab; nothing is sent anywhere.</p>
<textarea id="state" rows="8" placeholder="Paste a case"></textarea>
<p><button id="go" disabled>Loading the model…</button> <span id="ms" class="muted"></span></p>
<div id="err"></div><div id="out"></div>
<script type="module">
import { loadDecider } from ${JSON.stringify(embedUrl)};
const SPIKE = ${JSON.stringify(sm).replace(/</g, "\\u003c")};
const BIN = Uint8Array.from(atob("${btoa(bin)}"), (c) => c.charCodeAt(0));
const $ = (id) => document.getElementById(id), esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
try {
  const d = await loadDecider({ spike: { sm: SPIKE, bytes: BIN } });
  $("go").disabled = false; $("go").textContent = "Answer";
  $("go").onclick = async () => {
    const t = performance.now(), a = await d.decide($("state").value);
    $("ms").textContent = Math.round(performance.now() - t) + " ms";
    $("out").innerHTML = Object.entries(a).map(([q, r]) => '<div class="card"><h3>' + esc(q) + ': ' + esc(r.label) + '</h3><div class="muted">' + esc(d.questions[q].instructions) + '</div>' +
      Object.entries(r.probabilities).map(([o, p]) => '<div class="opt"><span>' + esc(o) + '</span><span class="track"><div style="width:' + (100 * p).toFixed(1) + '%"></div></span><span class="pct">' + Math.round(100 * p) + '%</span></div>').join("") + '</div>').join("");
    window.__answer = a;
  };
  window.__ready = true;
} catch (e) { $("err").textContent = String(e.message || e); $("go").textContent = "Unavailable"; window.__ready = "error"; }
</script>
</main></body></html>
`;
}
