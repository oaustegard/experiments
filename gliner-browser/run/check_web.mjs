// Run web/check.html in headless Chromium (SwiftShader WebGPU): the browser NER runtime against the
// NumPy reference's fixtures for each packed model, then the JavaScript stage timings.
//
//   node run/check_web.mjs TOKENIZERS_MJS [--pkgs a,b] [--nbench N]
//
// TOKENIZERS_MJS: a local copy of @huggingface/tokenizers@0.2.0/dist/tokenizers.mjs (npm pack it
// anywhere). The page imports that file from jsDelivr; the container's Chromium has no proxied egress,
// so the request is answered from the local copy (as browser-decider/run/check_engine.mjs does). The
// package is only ever executed inside the page. Serves the experiments repo on 127.0.0.1 (WebGPU in
// headless Chromium needs an http://127.0.0.1 page), writes results/web_parity.json, exits 0 on PASS.
// Timing input: out/web_bench_texts.json from run/bench_texts.py (skipped if absent).
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const { chromium } = createRequire(import.meta.url)("playwright");
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(ROOT, "..");
const args = process.argv.slice(2);
const tokLib = args[0];
const opt = (k, d) => { const i = args.indexOf(k); return i > 0 ? args[i + 1] : d; };
if (!tokLib || !existsSync(tokLib)) { console.error("usage: node run/check_web.mjs TOKENIZERS_MJS [--pkgs a,b] [--nbench N]"); process.exit(2); }
const pkgs = opt("--pkgs", "gliner-bi-edge-v2.0-fp32,gliner-bi-edge-v2.0-t4-w8L63-h8");
const nbench = opt("--nbench", "100");
const bench = existsSync(resolve(ROOT, "out", "web_bench_texts.json")) ? "../out/web_bench_texts.json" : "";
const port = 8700 + Math.floor(Math.random() * 200);
const server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: REPO, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
const errors = [];
let res, browser;
try {
  browser = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--disable-vulkan-surface"] });
  const page = await browser.newPage();
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(m.type() + ": " + m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.route("https://cdn.jsdelivr.net/**", (route) => route.fulfill({ status: 200, contentType: "text/javascript", body: readFileSync(tokLib) }));
  const t0 = Date.now();
  await page.goto(`http://127.0.0.1:${port}/${ROOT.split("/").pop()}/web/check.html?pkgs=${pkgs}&bench=${bench}&nbench=${nbench}`);
  await page.waitForFunction("window.check && window.check.done", null, { timeout: 3_600_000 });
  res = await page.evaluate("window.check");
  res.wallS = (Date.now() - t0) / 1000;
} finally {
  await browser?.close();
  server.kill();
}
res.console_errors = errors;
res.tolerance = { ids: "exact", entities: "same spans, labels, offsets and text; |score diff| <= 1e-3", logits: "max |diff| < 2e-3 (reported)", batching: "max |diff| < 1e-5" };
const checks = [];
for (const [pkg, r] of Object.entries(res.packs || {})) {
  const ok = !r.gpuErrors.length && r.tokenIdsMatch === r.nCases && r.wordsMatch === r.nCases && r.wordsMaskMatch === r.nCases
    && r.caseLabelIdsMatch === r.nCases && r.labelNormMatch === r.nCases && r.labelIdsMatch === r.labels
    && r.entityMatches === r.entityChecks && r.logitMaxAbsDiff < 2e-3
    && r.batching.oneBatch.maxAbsDiff < 1e-5 && r.batching.split.maxAbsDiff < 1e-5 && r.batching.splitNoPipeline.maxAbsDiff < 1e-5;
  checks.push([pkg, ok]);
}
res.pass = !res.error && checks.length > 0 && checks.every(([, ok]) => ok);
mkdirSync(resolve(ROOT, "results"), { recursive: true });
writeFileSync(resolve(ROOT, "results", "web_parity.json"), JSON.stringify(res, null, 1));
const brief = Object.fromEntries(Object.entries(res.packs || {}).map(([k, r]) => { const { cases, ...rest } = r; return [k, rest]; }));
console.log(JSON.stringify({ error: res.error, packs: brief, cpu: res.cpu, wallS: res.wallS, console_errors: errors.slice(0, 10) }, null, 1));
console.log(res.pass ? "PASS" : "FAIL", checks.map(([p, ok]) => `${p}:${ok ? "ok" : "FAIL"}`).join(" "));
process.exit(res.pass ? 0 : 1);
