// Bench and parity check for the WGSL engine against onnxruntime-web (WebGPU and wasm) on the same
// model file. Query token ids and their onnxruntime CPU vectors come from queries.json (run/ref.py),
// so tokenization is out of the measurement. Params: ?model=<dir with model_t3w.onnx[_data]>,
// &reps=N, &ort=0 to skip onnxruntime-web, &auto=1 to run on load (the headless check uses it).
import { Eg2Engine } from "./engine.js";

const P = new URLSearchParams(location.search);
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const MODEL = new URL(P.get("model") || (location.hostname.endsWith("austegard.com") ? "/bsky/image-search/model/" : "../out/model/"), location.href);
const REPS = +(P.get("reps") || 20);
const $ = (id) => document.getElementById(id);
const status = (t, bad = false) => { $("status").textContent = t; $("status").className = bad ? "bad" : ""; };
window.result = { phase: "idle" };

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };
const cos = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / Math.sqrt(na * nb); };
const round = (x, n = 3) => +x.toFixed(n);

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

// Time fn() REPS times after 3 warm-up calls; returns the median and the spread.
async function time(fn) {
  for (let i = 0; i < 3; i++) await fn();
  const ms = [];
  for (let i = 0; i < REPS; i++) { const t0 = performance.now(); await fn(); ms.push(performance.now() - t0); }
  return { median: round(median(ms)), min: round(Math.min(...ms)), max: round(Math.max(...ms)) };
}

async function run() {
  $("run").disabled = true;
  const res = window.result = { phase: "loading", ua: navigator.userAgent, reps: REPS, model: MODEL.href };
  try {
    status("loading model, manifest and queries…");
    const [manifest, queries] = await Promise.all([
      fetch("./manifest.json").then((r) => r.json()), fetch("./queries.json").then((r) => r.json())]);
    const [graph, data] = await Promise.all([fetchBytes(new URL(manifest.source, MODEL)), fetchBytes(new URL(manifest.data, MODEL))]);
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const info = adapter.info || {};
    res.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
    const device = await adapter.requestDevice();
    device.lost.then((l) => { res.device_lost = l.message; });
    let t0 = performance.now();
    const eng = await Eg2Engine.create(device, manifest, data, { maxTokens: 512, maxSeqs: 16 });
    res.engine_init_ms = round(performance.now() - t0, 1);
    const ids = queries.map((q) => q.ids);
    const ref = queries.map((q) => Float32Array.from(q.ort));

    status("WGSL engine: parity…");
    const single = [];
    for (const x of ids) single.push(await eng.embed([x]));
    const batch = await eng.embed(ids);
    res.wgsl = {
      cos_single: single.map((v, i) => round(cos(v, ref[i]), 7)),
      cos_batch_min: round(Math.min(...ids.map((_, i) => cos(batch.subarray(i * 768, (i + 1) * 768), ref[i]))), 7),
    };
    res.wgsl.cos_single_min = Math.min(...res.wgsl.cos_single);
    status("WGSL engine: timing…");
    res.wgsl.ms_per_query = [];
    for (const x of ids) res.wgsl.ms_per_query.push({ tokens: x.length, ...(await time(() => eng.embed([x]))) });
    res.wgsl.ms_batch12 = await time(() => eng.embed(ids));

    if (P.get("ort") !== "0") {
      status("onnxruntime-web: loading…");
      const ort = await import(ORT + "ort.webgpu.min.mjs");
      ort.env.wasm.wasmPaths = ORT;
      for (const ep of ["webgpu", "wasm"]) {
        status(`onnxruntime-web ${ep}: session…`);
        t0 = performance.now();
        const session = await ort.InferenceSession.create(graph, {
          executionProviders: [ep], logSeverityLevel: 3, externalData: [{ path: manifest.data, data }] });
        const r = res["ort_" + ep] = { init_ms: round(performance.now() - t0, 1) };
        const feed = (x) => ({
          input_ids: new ort.Tensor("int64", BigInt64Array.from(x, BigInt), [1, x.length]),
          attention_mask: new ort.Tensor("int64", new BigInt64Array(x.length).fill(1n), [1, x.length]) });
        const vec = async (x) => new Float32Array((await session.run(feed(x))).embedding.data);
        const vs = [];
        for (const x of ids) vs.push(await vec(x));
        r.cos_single_min = round(Math.min(...vs.map((v, i) => cos(v, ref[i]))), 7);
        status(`onnxruntime-web ${ep}: timing…`);
        r.ms_per_query = [];
        for (const x of ids) r.ms_per_query.push({ tokens: x.length, ...(await time(() => vec(x))) });
        await session.release();
      }
    }
    const typical = (k) => res[k] && round(median(res[k].ms_per_query.filter((q) => q.tokens <= 20).map((q) => q.median)));
    res.summary = { wgsl_ms: typical("wgsl"), ort_webgpu_ms: typical("ort_webgpu"), ort_wasm_ms: typical("ort_wasm"),
                    note: "median over the 10 queries of <= 20 tokens of each query's median ms, readback included" };
    res.phase = "done";
    status(`done: WGSL ${res.summary.wgsl_ms} ms, ORT-web WebGPU ${res.summary.ort_webgpu_ms ?? "–"} ms, wasm ${res.summary.ort_wasm_ms ?? "–"} ms per short query`);
    render(res);
  } catch (e) {
    res.phase = "error";
    res.error = String(e?.stack || e);
    status(String(e?.message || e), true);
  }
  $("json").textContent = JSON.stringify(res, null, 1);
  $("copy").disabled = false;
  $("run").disabled = false;
}

function render(res) {
  const rows = [["query tokens", "WGSL ms", "ORT WebGPU ms", "ORT wasm ms"]];
  res.wgsl.ms_per_query.forEach((q, i) => rows.push([q.tokens, q.median, res.ort_webgpu?.ms_per_query[i].median ?? "–", res.ort_wasm?.ms_per_query[i].median ?? "–"]));
  rows.push(["cos to CPU (min)", res.wgsl.cos_single_min, res.ort_webgpu?.cos_single_min ?? "–", res.ort_wasm?.cos_single_min ?? "–"]);
  $("table").replaceChildren(...rows.map((r, i) => {
    const tr = document.createElement("tr");
    tr.append(...r.map((c) => Object.assign(document.createElement(i ? "td" : "th"), { textContent: c })));
    return tr;
  }));
}

$("run").onclick = run;
$("copy").onclick = () => navigator.clipboard.writeText($("json").textContent);
if (P.get("auto") === "1") run();
