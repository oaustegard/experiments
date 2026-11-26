// Headless parity run for the WGSL engine. Serves web/gpu/ at /gpu/ and web/gpu/test/data/ (make_ref.py)
// at /fix/, runs test/check.js in Chromium on SwiftShader WebGPU (--gpu: the machine's own adapter),
// writes results/gpu_parity.json and prints a summary. Exit 0 when every model passes:
//   det     prob max abs diff < 1e-3 against onnxruntime on the unpacked-equivalent weights
//   router  group / flip logits max abs diff < 1e-3, argmax identical on every line
//   rec_*   ids identical on >= 99.9% of timesteps (rec_latin: >= 99% against onnxruntime's QDQ run)
//
//   python3 -I web/gpu/test/make_ref.py && node web/gpu/test/run_parity.mjs [--gpu] [model ...]
//
// SwiftShader is a CPU emulation of a GPU: its timings say nothing about a real GPU.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
for (const p of ["playwright", "/opt/node-tools/node_modules/playwright", path.join(process.execPath, "../../lib/node_modules/playwright")]) {
  try { ({ chromium } = require(p)); break; } catch { /* next */ }
}
if (!chromium) { console.error("playwright not found"); process.exit(2); }

const HERE = path.dirname(fileURLToPath(import.meta.url));
const GPU = path.resolve(HERE, "..");
const FIX = path.join(HERE, "data");
const RESULTS = path.resolve(GPU, "../../results/gpu_parity.json");
const args = process.argv.slice(2);
const useGpu = args.includes("--gpu");
const only = args.filter((a) => !a.startsWith("--"));
if (!fs.existsSync(path.join(FIX, "manifest.json"))) { console.error("no fixtures: run python3 -I web/gpu/test/make_ref.py"); process.exit(2); }

const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>ocria gpu parity</title>"); }
  let f = null;
  if (u.startsWith("/gpu/")) f = path.join(GPU, u.slice(5));
  else if (u.startsWith("/fix/")) f = path.join(FIX, u.slice(5));
  if (!f || !(f.startsWith(GPU) || f.startsWith(FIX))) { res.writeHead(404); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const flags = useGpu ? ["--no-sandbox", "--enable-unsafe-webgpu"]
  : ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader",
     "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"];
const exe = process.env.CHROMIUM || (fs.existsSync("/opt/pw-browsers/chromium") ? "/opt/pw-browsers/chromium" : undefined);
const browser = await chromium.launch({ args: flags, executablePath: exe });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`${m.text()}\n`));
page.on("pageerror", (e) => process.stderr.write(`pageerror: ${e.message}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
const wall0 = Date.now();
let out;
try {
  out = await page.evaluate(async ({ only }) => (await import("/gpu/test/check.js")).default({ only }), { only });
} catch (e) {
  out = { ok: false, error: String(e && e.message || e) };
}
await browser.close();
server.close();
out.wall_s = Math.round((Date.now() - wall0) / 100) / 10;
out.backend = useGpu ? "machine GPU" : "SwiftShader (CPU emulation; timings are not GPU timings)";
if (out.models) {
  fs.mkdirSync(path.dirname(RESULTS), { recursive: true });
  fs.writeFileSync(RESULTS, JSON.stringify(out, null, 1) + "\n");
}

// summary
const f = (v, d = 2) => (typeof v === "number" ? (Math.abs(v) < 1e-2 && v !== 0 ? v.toExponential(d) : v.toFixed(d + 2)) : String(v));
if (!out.models) { console.log(JSON.stringify(out, null, 1)); process.exit(1); }
console.log(`backend: ${out.backend}; adapter ${JSON.stringify(out.adapter)}; onnxruntime ${out.onnxruntime}; load ${out.load_ms} ms; wall ${out.wall_s} s`);
for (const [m, s] of Object.entries(out.models)) {
  let line;
  if (m === "det") {
    line = `max|d| ${f(s.max_abs)} (vs fp32 ${f(s.max_abs_vs_fp32)}), mask mismatch ${f(s.mask_mismatch)}; pages ${s.pages.map((p) => `${p.shape[2]}x${p.shape[3]} ${p.ms} ms`).join(", ")}`;
  } else if (m === "router") {
    line = `group max|d| ${f(s.group.max_abs)} argmax ${s.group.argmax_agree}, flip max|d| ${f(s.flip.max_abs)} argmax ${s.flip.argmax_agree}` +
      ` (vs fp32 argmax ${s.group.argmax_agree_vs_fp32.toFixed(4)} / ${s.flip.argmax_agree_vs_fp32.toFixed(4)}); ${s.items} lines ${s.total_ms} ms`;
  } else {
    const r = s.vs_ref, altK = Object.keys(s).find((k) => k.startsWith("vs_") && k !== "vs_ref"), a = s[altK];
    line = `ids ${(100 * r.ids_agree_all).toFixed(3)}% (valid ${(100 * r.ids_agree_valid).toFixed(3)}%), conf max|d| ${f(r.conf_max_abs)} mean ${f(r.conf_mean_abs)}, text ${(100 * r.text_identical).toFixed(1)}%` +
      ` | ${altK.slice(3)}: ids ${(100 * a.ids_agree_all).toFixed(3)}% text ${(100 * a.text_identical).toFixed(1)}% conf max|d| ${f(a.conf_max_abs)}; ${s.items} lines ${s.total_ms} ms (${s.lines_per_s} lines/s)`;
  }
  console.log(`${s.pass ? "PASS" : "FAIL"} ${m.padEnd(9)} ${line}`);
}
console.log(out.ok ? "ALL PASS" : "SOME FAILED");
process.exit(out.ok ? 0 : 1);
