// A/B an engine at a git ref against the working tree, on the same weights and inputs, in headless
// Chromium on SwiftShader. Use it when moving an engine onto shared kernels: a move that changes no
// shader should give identical outputs, and any other move shows its max |diff|.
//
//   node _lib/wgsl/tests/engine_ab.mjs [--ref HEAD] [case ...]
//
// The ref's tree is exported (git archive) to a temp dir and served under /__old/; the working tree
// under /. Each case below loads its engine from both, runs the same batches, and reports max |diff|.
// Exits 1 when a case's max |diff| exceeds its tolerance (0 by default: bit-identical).
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const args = process.argv.slice(2);
const ri = args.indexOf("--ref");
const ref = ri < 0 ? "HEAD" : args.splice(ri, 2)[1];
const only = args;

// A case runs in the page: ({ device, load(modulePath, which) -> module, bytes(path), json(path) }) -> list of
// Float32Arrays per engine. Inputs must be deterministic. Weights not in git are listed in `fetch`: the
// script downloads each URL once into a cache (EXPERIMENTS_AB_CACHE, default /tmp/engine-ab-cache) with
// curl, which goes through the container's proxy, and the case reads it as path "__cache/<name>".
const CASES = {
  "browser-decider": { tol: 0, dirs: ["browser-decider/web", "_lib/wgsl"], src: `async ({ device, load, bytes, json }, which) => {
    const { DeciderEngine } = await load("browser-decider/web/engine.js", which);
    const base = "browser-decider/web/models/decider/";
    const m = await json(base + "manifest.json"), w = await bytes(base + "weights.bin");
    const e = await DeciderEngine.create(device, m, w, { maxTokens: 4096, maxSeqs: 16, maxPos: 256 });
    let s = 7; const rnd = (n) => { s = (s * 1103515245 + 12345) % 2147483648; return s % n; };
    const seq = (len, npos) => ({ ids: Array.from({ length: len }, (_, i) => i === 0 ? 50281 : 1000 + rnd(40000)), pos: Array.from({ length: npos }, (_, i) => 1 + i * 3) });
    const out = [];
    for (const b of [[seq(40, 3)], [seq(700, 5), seq(129, 2), seq(1, 1)], [seq(1024, 4)]]) out.push(...await e.score(b));
    e.destroy();
    return out;
  }` },
  "adapter-library": { tol: 0, dirs: ["adapter-library/web", "_lib/wgsl"], src: `async ({ device, load, bytes, json }, which) => {
    const { SpikeEngine } = await load("adapter-library/web/engine.js", which);
    const base = "adapter-library/web/models/";
    const sys = await json(base + "workflows/system.json");
    const bb = base + "workflows/" + (sys.base ?? "base/");
    const m = await json(bb + "manifest.json"), w = await bytes(bb + "weights.bin");
    const e = await SpikeEngine.create(device, m, w, { maxTokens: 4096, maxSeqs: 16, maxPos: 2048 });
    let s = 11; const rnd = (n) => { s = (s * 1103515245 + 12345) % 2147483648; return s % n; };
    const seq = (len, npos) => ({ ids: Array.from({ length: len }, (_, i) => i === 0 ? 50281 : 1000 + rnd(40000)), pos: Array.from({ length: npos }, (_, i) => 1 + i * 3) });
    const out = [];
    for (const f of [sys.router.file, sys.spikes[0].file]) {
      e.setSpike(await json(base + "workflows/" + f + ".json"), new Uint8Array(await bytes(base + "workflows/" + f + ".bin")));
      for (const b of [[seq(60, 3)], [seq(700, 4), seq(129, 2)]]) out.push(...await e.score(b));
    }
    e.destroy();
    return out;
  }` },
  "ettin-native": { tol: 0, dirs: ["ettin-native/web", "_lib/wgsl"],
    fetch: { "ettin-manifest.json": "https://austegard.com/bsky/embedders/models/ettin/manifest.json",
             "ettin-model.onnx": "https://austegard.com/bsky/embedders/models/ettin/model.onnx" },
    src: `async ({ device, load, bytes, json }, which) => {
    const { SmallEngine } = await load("ettin-native/web/engine.js", which);
    const m = await json("__cache/ettin-manifest.json"), w = await bytes("__cache/ettin-model.onnx");
    const fx = await json("ettin-native/web/models/e17c-t4-w8L63/fixtures.json");
    const e = await SmallEngine.create(device, m, w, { maxTokens: 512, maxSeqs: 16 });
    const ids = fx.map((x) => x.ids), out = [];
    for (const q of ids.slice(0, 6)) out.push(await e.embed([q]));
    out.push(await e.embed(ids.slice(0, 8)));
    e.destroy();
    return out;
  }` },
  "eg2-cut": { tol: 0, dirs: ["eg2-wgsl-engine/web", "_lib/wgsl"],
    fetch: { "eg2-cut.onnx_data": "https://austegard.com/bsky/image-search/model/ort/model_t3w.onnx_data" },
    src: `async ({ device, load, bytes, json }, which) => {
    const { Eg2Engine } = await load("eg2-wgsl-engine/web/engine.js", which);
    const m = await json("eg2-wgsl-engine/web/manifest.json"), w = await bytes("__cache/eg2-cut.onnx_data");
    const ids = (await json("eg2-wgsl-engine/web/queries.json")).map((x) => x.ids);
    const e = await Eg2Engine.create(device, m, w, { maxTokens: 512, maxSeqs: 16 });
    const out = [];
    for (const q of ids.slice(0, 6)) out.push(await e.embed([q]));
    out.push(await e.embed(ids.slice(0, 8)));
    e.destroy();
    return out;
  }` },
  "moedern-1.5M": { tol: 0, dirs: ["moederncolbert-bench/web", "_lib/wgsl"],
    fetch: { "mcb15-config.json": "https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/resolve/main/config.json",
             "mcb15-model.safetensors": "https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/resolve/main/model.safetensors",
             "mcb15-dense.safetensors": "https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/resolve/main/1_Dense/model.safetensors" },
    src: `async ({ device, load, bytes, json }, which) => {
    const { MoEdernEngine } = await load("moederncolbert-bench/web/engine.js", which);
    const config = await json("__cache/mcb15-config.json"), model = await bytes("__cache/mcb15-model.safetensors"), dense = await bytes("__cache/mcb15-dense.safetensors");
    const fx = await json("moederncolbert-bench/web/fixtures.json");
    const out = [];
    for (const kernel of ["tiled", "template"]) {
      const e = await MoEdernEngine.create(device, { config, dense }, model, { maxTokens: 4096, maxSeqs: 64, kernel });
      for (const q of fx.queries.slice(0, 3)) out.push(await e.embed([q]));
      out.push(await e.embed(fx.docs.slice(0, 4)));
      e.destroy();
    }
    return out;
  }` },
};

const CACHE = process.env.EXPERIMENTS_AB_CACHE ?? path.join(os.tmpdir(), "engine-ab-cache");
fs.mkdirSync(CACHE, { recursive: true });

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "engine-ab-"));
const names = Object.keys(CASES).filter((n) => !only.length || only.includes(n));
const dirs = [...new Set(names.flatMap((n) => CASES[n].dirs))];
execSync(`git archive ${ref} ${dirs.join(" ")} | tar -x -C ${tmp}`, { cwd: REPO, shell: "/bin/bash" });
for (const n of names) for (const [f, url] of Object.entries(CASES[n].fetch ?? {})) {
  const dst = path.join(CACHE, f);
  if (!fs.existsSync(dst)) { console.error(`fetching ${url}`); execSync(`curl -sSfL --retry 3 -o ${dst}.part '${url}' && mv ${dst}.part ${dst}`, { shell: "/bin/bash" }); }
}

const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>ab</title>"); }
  const old = u.startsWith("/__old/"), cache = u.startsWith("/__cache/");
  const root = old ? tmp : cache ? CACHE : REPO, f = path.join(root, old ? u.slice(6) : cache ? u.slice(8) : u);
  if (!f.startsWith(root + path.sep)) { res.writeHead(403); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": f.endsWith(".js") ? "text/javascript" : f.endsWith(".json") ? "application/json" : "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`[page] ${m.text()}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
let bad = 0;
for (const n of names) {
  const r = await page.evaluate(async ([src]) => {
    const adapter = await navigator.gpu.requestAdapter();
    const L = adapter.limits;
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
    const errs = [];
    device.addEventListener("uncapturederror", (e) => errs.push(e.error.message));
    const ctx = {
      device,
      load: (p, which) => import((which === "old" ? "/__old/" : "/") + p),
      bytes: async (p) => (await fetch("/" + p)).arrayBuffer(),
      json: async (p) => (await fetch("/" + p)).json(),
    };
    const run = (0, eval)(src);
    try {
      const a = await run(ctx, "old"), b = await run(ctx, "new");
      let maxDiff = 0, maxAbs = 0, n = 0;
      if (a.length !== b.length) return { error: `old gave ${a.length} outputs, new ${b.length}` };
      a.forEach((x, i) => { for (let j = 0; j < x.length; j++) { maxDiff = Math.max(maxDiff, Math.abs(x[j] - b[i][j])); maxAbs = Math.max(maxAbs, Math.abs(x[j])); n++; } });
      return { outputs: a.length, values: n, maxDiff, maxAbs, gpuErrors: errs };
    } catch (e) { return { error: String(e.stack || e), gpuErrors: errs }; }
  }, [CASES[n].src]);
  const ok = !r.error && !r.gpuErrors.length && r.values > 0 && r.maxDiff <= CASES[n].tol;
  bad += !ok;
  console.log(`${ok ? "ok  " : "FAIL"} ${n} vs ${ref}: ${r.error ?? `${r.outputs} outputs, ${r.values} values, max |diff| ${r.maxDiff} (max |value| ${r.maxAbs.toFixed(3)})`}${r.gpuErrors?.length ? " GPU: " + r.gpuErrors.join("; ") : ""}`);
}
await browser.close();
server.close();
fs.rmSync(tmp, { recursive: true, force: true });
process.exit(bad ? 1 : 0);
