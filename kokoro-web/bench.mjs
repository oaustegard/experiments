// Run web/ in headless Chromium and record what it costs: download, start-up, speaking time, memory, and the audio
// and word times (written in compare.py's format so they can be held against the PyTorch reference).
//
//   NODE_PATH=$(npm root -g) node bench.mjs NAME "model=models/kokoro-mixed.onnx&ep=wasm&threads=4"
//
// Serves web/ with COOP/COEP (so wasm threads and performance.measureUserAgentSpecificMemory work),
// /models/ from $MODELS and /ort/ from $ORT_DIST (onnxruntime-web's dist folder).
import { createServer } from "node:http";
import { execSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync, statSync, existsSync } from "node:fs";
import { join, extname, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

// ESM ignores NODE_PATH; resolve the globally installed playwright through require.
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");

const HERE = dirname(fileURLToPath(import.meta.url));
const [name, params = ""] = process.argv.slice(2);
const MODELS = process.env.MODELS || "/opt/kokoro-web-models";
const ORT_DIST = process.env.ORT_DIST;
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
  ".wasm": "application/wasm" };

const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const file = path.startsWith("/models/") ? join(MODELS, path.slice(8))
    : path.startsWith("/ort/") ? join(ORT_DIST, path.slice(5))
    : join(HERE, "web", path === "/" ? "index.html" : path);
  if (!existsSync(file)) { res.writeHead(404); res.end(); return; }
  // NOCOI=1 serves like GitHub Pages (no isolation headers), leaving isolation to web/coi.js's service worker.
  const coi = process.env.NOCOI ? {} : { "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin" };
  res.writeHead(200, { "Content-Type": TYPES[extname(file)] || "application/octet-stream",
    "Content-Length": statSync(file).size, ...coi });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, r));
const url = `http://localhost:${server.address().port}/?ort=/ort/&${params}`;

const gpu = params.includes("ep=webgpu");
const browser = await chromium.launch({ args: ["--js-flags=--expose-gc", ...(gpu ? ["--enable-unsafe-webgpu", "--use-angle=swiftshader",
  "--use-webgpu-adapter=swiftshader", "--use-vulkan=swiftshader", "--enable-features=Vulkan"] : [])] });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

// Peak resident memory of Chromium's renderer and GPU processes while the page works.
function chromiumRss() {
  const out = execSync("ps -eo rss,args").toString().split("\n");
  let renderer = 0, gpuProc = 0;
  for (const l of out) {
    const rss = +l.trim().split(/\s+/)[0] * 1024;
    if (l.includes("--type=renderer")) renderer = Math.max(renderer, rss);
    if (l.includes("--type=gpu-process")) gpuProc = Math.max(gpuProc, rss);
  }
  return { renderer, gpu: gpuProc };
}
const peak = { renderer: 0, gpu: 0 };
const timer = setInterval(() => { const r = chromiumRss(); peak.renderer = Math.max(peak.renderer, r.renderer); peak.gpu = Math.max(peak.gpu, r.gpu); }, 100);

const t0 = Date.now();
await page.goto(url);
await page.waitForFunction(() => window.ready === true || document.getElementById("status")?.textContent.startsWith("Error"),
  null, { timeout: 600000 });
const loadS = (Date.now() - t0) / 1000;
const peakAfterLoad = { ...peak };
// What the page holds once loaded: collect garbage (the model bytes, if the page let them go) and read RSS now.
await page.evaluate(() => { window.gc?.(); window.gc?.(); });
await new Promise((r) => setTimeout(r, 1500));
const settled = chromiumRss();
const result = await page.evaluate((n) => window.bench(n || undefined), +(process.env.BENCH_LINES || 0));
clearInterval(timer);
await browser.close();
server.close();

const outDir = join(HERE, "out", name);
mkdirSync(outDir, { recursive: true });
const lines = result.lines.map((l) => {
  const pcm = Buffer.alloc(44 + l.audio.length * 2);
  pcm.write("RIFF", 0); pcm.writeUInt32LE(36 + l.audio.length * 2, 4); pcm.write("WAVEfmt ", 8);
  pcm.writeUInt32LE(16, 16); pcm.writeUInt16LE(1, 20); pcm.writeUInt16LE(1, 22); pcm.writeUInt32LE(24000, 24);
  pcm.writeUInt32LE(48000, 28); pcm.writeUInt16LE(2, 32); pcm.writeUInt16LE(16, 34); pcm.write("data", 36);
  pcm.writeUInt32LE(l.audio.length * 2, 40);
  l.audio.forEach((v, k) => pcm.writeInt16LE(Math.round(Math.max(-1, Math.min(1, v)) * 32767), 44 + 2 * k));
  writeFileSync(join(outDir, `${l.id}.wav`), pcm);
  return { id: l.id, words: l.words, phones: [], seconds: l.audio.length / 24000, synth_s: l.ms / 1000 };
});
const audioS = lines.reduce((a, l) => a + l.seconds, 0), synthS = lines.reduce((a, l) => a + l.synth_s, 0);
const summary = { arm: name, engine: "onnxruntime-web", python: "browser", version: `onnxruntime-web 1.31.1, ${params}`,
  load_s: +(result.stats.init ?? 0).toFixed(3), page_load_s: loadS, synth_s: +synthS.toFixed(3), audio_s: +audioS.toFixed(3),
  peak_rss_mb: Math.round(peak.renderer / 1e6), peak_rss_after_load_mb: Math.round(peakAfterLoad.renderer / 1e6),
  peak_gpu_process_mb: Math.round(peak.gpu / 1e6), rss_loaded_after_gc_mb: Math.round(settled.renderer / 1e6), page_memory_mb: result.memoryMB && Math.round(result.memoryMB),
  stats: result.stats, errors, lines };
writeFileSync(join(outDir, "result.json"), JSON.stringify(summary, null, 1));
const { lines: _, ...brief } = summary;
console.log(JSON.stringify(brief));
