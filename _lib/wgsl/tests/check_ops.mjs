// Per-op parity of the shared WGSL kernels (ops.js) and the training kernels (train_ops.js), in headless Chromium on SwiftShader WebGPU.
//
//   node _lib/wgsl/tests/check_ops.mjs [filter] [--reps N] [--json out.json]
//
// Exits 1 when any case fails. SwiftShader is a CPU rasterizer: its correctness counts and its timings
// do not. Real-GPU timing goes through gpu-bridge/jobs/wgsl-ops.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); if (i < 0) return d; const v = args[i + 1]; args.splice(i, 2); return v; };
const reps = Number(opt("--reps", 0)), jsonOut = opt("--json", null);
const only = args[0] ?? "";

const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>ops</title>"); }
  const f = path.join(REPO, u);
  if (!f.startsWith(REPO + path.sep)) { res.writeHead(403); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": f.endsWith(".js") || f.endsWith(".mjs") ? "text/javascript" : "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`${m.text()}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
const out = await page.evaluate(async ([only, reps]) => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { error: "no WebGPU adapter" };
  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (e) => console.log("GPU error: " + e.error.message));
  const { runOps } = await import("/_lib/wgsl/tests/ops.js");
  const { runTrainOps } = await import("/_lib/wgsl/tests/train_ops.js");
  const info = adapter.info ?? {}, log = (m) => console.log(m);
  const results = [...await runOps(device, { only, reps, log }), ...await runTrainOps(device, { only, log })];
  return { device: `${info.vendor} ${info.architecture} ${info.description}`.trim(), results };
}, [only, reps]);
await browser.close();
server.close();
if (out.error) { console.error(out.error); process.exit(2); }
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(out, null, 1));
const bad = out.results.filter((r) => !r.pass);
console.log(`${out.results.length - bad.length}/${out.results.length} ops pass on ${out.device}`);
process.exit(bad.length || !out.results.length ? 1 : 0);
