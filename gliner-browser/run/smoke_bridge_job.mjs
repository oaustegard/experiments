// Run web/bridge_job.js once in headless Chromium (SwiftShader) with a stand-in for the GPU Bridge's ctx,
// to catch errors before a real submit. Timings from this run mean nothing.
//
//   node run/smoke_bridge_job.mjs TOKENIZERS_MJS [PKG] [--n 40]
//
// The job imports ner.js from jsDelivr at cfg.sha; here sha is "local" and those requests are answered
// from the working tree, as is the tokenizer library (a local copy of @huggingface/tokenizers@0.2.0's
// dist/tokenizers.mjs). Writes out/bridge_smoke.json and exits 0 when the job returns with parity passing.
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const { chromium } = createRequire(import.meta.url)("playwright");
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(ROOT, "..");
const args = process.argv.slice(2);
const tokLib = args[0];
const pkg = args[1] && !args[1].startsWith("--") ? args[1] : "gliner-bi-edge-v2.0-t4-w8L63-h8";
const n = +(args.includes("--n") ? args[args.indexOf("--n") + 1] : 40);
const benchPath = resolve(ROOT, "out", "web_bench_texts.json");
if (!tokLib || !existsSync(tokLib) || !existsSync(benchPath)) { console.error("usage: node run/smoke_bridge_job.mjs TOKENIZERS_MJS [PKG] [--n N] (needs out/web_bench_texts.json)"); process.exit(2); }
const name = ROOT.split("/").pop();
const port = 8700 + Math.floor(Math.random() * 200);
const server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: REPO, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
let res, browser;
const errors = [];
try {
  browser = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--disable-vulkan-surface"] });
  const page = await browser.newPage();
  page.on("pageerror", (e) => errors.push(String(e)));
  // jsDelivr: the experiments repo at sha "local" from the working tree, anything else (the tokenizer library) from tokLib
  await page.route("https://cdn.jsdelivr.net/**", (route) => {
    const u = route.request().url(), pre = "https://cdn.jsdelivr.net/gh/oaustegard/experiments@local/";
    const body = u.startsWith(pre) ? readFileSync(resolve(REPO, u.slice(pre.length))) : readFileSync(tokLib);
    return route.fulfill({ status: 200, contentType: "text/javascript", body });
  });
  await page.goto(`http://127.0.0.1:${port}/${name}/web/check.html?pkgs=`);   // an http://127.0.0.1 origin for WebGPU; runs no checks with no pkgs
  res = await page.evaluate(async ({ name, pkg, n }) => {
    const M = `/${name}/web/models/${pkg}/`;
    const bytes = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); }).then((b) => new Uint8Array(b));
    const inputs = { weights: await bytes(M + "weights.bin"), manifest: await bytes(M + "manifest.json"), tokenizer: await bytes(M + "tokenizer.json"),
      labels_tokenizer: await bytes(M + "labels_tokenizer.json"), fixtures: await bytes(M + "fixtures.json"),
      bench: await bytes(`/${name}/out/web_bench_texts.json`), cfg: { sha: "local", reps: 1, n } };
    const logs = [];
    const ctx = {
      inputs, log: (...a) => logs.push(a.join(" ")), progress: (f, l) => logs.push(`progress ${f} ${l}`),
      importModule: (u) => import(u), fetchBytes: bytes,
      gpu: async () => { const a = await navigator.gpu.requestAdapter(); return a.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxBufferSize: a.limits.maxBufferSize } }); },
      describe: async () => ({ ua: navigator.userAgent }),
    };
    const src = await fetch(`/${name}/web/bridge_job.js`).then((r) => r.text());
    const job = new (Object.getPrototypeOf(async function () {}).constructor)("ctx", src);
    try { return { result: await job(ctx), logs }; } catch (e) { return { error: String(e.stack || e), logs }; }
  }, { name, pkg, n });
} finally {
  await browser?.close();
  server.kill();
}
res.page_errors = errors;
mkdirSync(resolve(ROOT, "out"), { recursive: true });
writeFileSync(resolve(ROOT, "out", "bridge_smoke.json"), JSON.stringify(res, null, 1));
const p = res.result?.parity;
const brief = p && { ...p, heads: Object.fromEntries(Object.entries(p.heads ?? {}).map(([h, x]) => [h, { ...x, mismatches: x.mismatches.length }])) };
console.log(res.error ?? JSON.stringify({ parity: brief, keys: Object.keys(res.result?.timing ?? {}) }));
const ok = !res.error && p?.pass;
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
