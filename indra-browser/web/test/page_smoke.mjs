// The page end to end in headless Chromium (SwiftShader WebGPU): load, read one image, report.
//   node web/test/page_smoke.mjs IMAGE [--shot out.png]
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const args = process.argv.slice(2);
const shot = args.includes("--shot") ? args[args.indexOf("--shot") + 1] : null;
const image = path.resolve(args[0]);
const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".css": "text/css", ".png": "image/png" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const f = path.join(ROOT, u);
  if (!f.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] ?? "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/indra-browser/web/index.html`);
await page.waitForFunction(() => /Ready|Could not/.test(document.getElementById("status-text").textContent), null, { timeout: 120000 });
console.log("status:", await page.textContent("#status-text"));
await page.setInputFiles("#file", image);
await page.waitForFunction(() => /lines in|failed|Could not/.test(document.getElementById("status-text").textContent), null, { timeout: 900000 });
console.log("status:", await page.textContent("#status-text"));
const md = await page.textContent("#md");
console.log("markdown:", md.length, "chars\n" + md.slice(0, 600));
if (shot) await page.screenshot({ path: shot, fullPage: true });
console.log("console errors:", errors.length ? errors : "none");
await browser.close();
server.close();
process.exit(errors.length ? 1 : 0);
