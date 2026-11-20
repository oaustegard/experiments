// Headless parity run for the detector. Serves web/ at /web/ and data/fixtures/ at /fix/, runs
// web/det/test/check_det.js in Chromium (SwiftShader WebGPU unless --gpu), merges the WGSL numbers
// into results/det_parity.json, and exits 0 when every model and page is within tolerance.
//
//   python3 run/ref_det.py && node web/det/test/run_det.mjs [--gpu] [tiny] [small]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.execPath, "../../lib/node_modules/playwright"))); }

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(HERE, "../..");
const ROOT = path.resolve(WEB, "..");
const FIX = path.join(ROOT, "data/fixtures");
const RESULTS = path.join(ROOT, "results/det_parity.json");
const args = process.argv.slice(2);
const useGpu = args.includes("--gpu");
let models = args.filter((a) => ["tiny", "small"].includes(a));
if (!models.length) models = ["tiny", "small"];
const pages = fs.readdirSync(FIX).filter((d) => d.startsWith("det_")).map((d) => d.slice(4)).sort();
if (!pages.length) { console.error("no det_* fixtures; run python3 run/ref_det.py first"); process.exit(2); }

const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".wgsl": "text/plain", ".bin": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>det test</title>"); }
  let f = null;
  if (u.startsWith("/web/")) f = path.join(WEB, u.slice(5));
  else if (u.startsWith("/fix/")) f = path.join(FIX, u.slice(5));
  if (!f || !f.startsWith(WEB) && !f.startsWith(FIX)) { res.writeHead(404); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const flags = useGpu ? ["--enable-unsafe-webgpu"]
  : ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"];
const browser = await chromium.launch({ args: flags });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`${m.text()}\n`));
page.on("pageerror", (e) => process.stderr.write(`pageerror: ${e.message}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
let out;
try {
  out = await page.evaluate(async ({ models, pages }) => {
    const m = await import("/web/det/test/check_det.js");
    return await m.default({ models, pages });
  }, { models, pages });
} catch (e) {
  out = { ok: false, error: String(e && e.message || e) };
}
await browser.close();
server.close();

if (out.models) {
  const res = fs.existsSync(RESULTS) ? JSON.parse(fs.readFileSync(RESULTS, "utf8")) : { models: {} };
  res.adapter = out.adapter;
  for (const [m, pages_] of Object.entries(out.models)) {
    res.models[m] = res.models[m] || { pages: {} };
    for (const [p, r] of Object.entries(pages_)) {
      res.models[m].pages[p] = { ...(res.models[m].pages[p] || {}), wgsl_max_abs: r.wgsl_max_abs,
        wgsl_mismatch_share: r.wgsl_mismatch_share, swiftshader_ms: r.swiftshader_ms, wgsl_H: r.H, wgsl_W: r.W, wgsl_pass: r.pass };
    }
  }
  res.wgsl_pass = out.ok;
  res.pass = Boolean(res.numpy_pass && out.ok);
  fs.writeFileSync(RESULTS, JSON.stringify(res, null, 1) + "\n");
}
console.log(JSON.stringify(out.models ? { ok: out.ok, adapter: out.adapter, models: out.models } : out, null, 1));
process.exit(out && out.ok ? 0 : 1);
