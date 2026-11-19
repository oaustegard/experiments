// Runs one browser-side test module in headless Chromium (SwiftShader WebGPU) and prints its
// JSON result. Exit 0 when the result has ok: true.
//
//   node web/test/run.mjs test/check_conv.js best_eng fast_eng best_grc
//
// The page serves web/ at /web/ and data/fixtures/<case>/ at /fix/<case>/. The module's
// default export is called as run({ cases: ["/fix/best_eng/", ...] }) and must return
// { ok: boolean, ... }. Pass --gpu to use the platform GPU instead of SwiftShader.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.execPath, "../../lib/node_modules/playwright"))); }

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FIX = path.resolve(WEB, "../data/fixtures");
const args = process.argv.slice(2);
const useGpu = args.includes("--gpu");
const [mod, ...cases] = args.filter((a) => a !== "--gpu");
if (!mod) { console.error("usage: node web/test/run.mjs <module under web/> <case>..."); process.exit(2); }

const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".wgsl": "text/plain" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>test</title>"); }
  let f = null;
  if (u.startsWith("/web/")) f = path.join(WEB, u.slice(5));
  else if (u.startsWith("/fix/")) f = path.join(FIX, u.slice(5));
  if (!f) { res.writeHead(404); return res.end(); }
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
  out = await page.evaluate(async ({ mod, cases }) => {
    const m = await import("/web/" + mod);
    return await m.default({ cases: cases.map((c) => `/fix/${c}/`) });
  }, { mod, cases });
} catch (e) {
  out = { ok: false, error: String(e && e.message || e) };
}
console.log(JSON.stringify(out, null, 1));
await browser.close();
server.close();
process.exit(out && out.ok ? 0 : 1);
