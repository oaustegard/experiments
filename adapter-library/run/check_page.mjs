// Serve adapter-library/web and run check.html (or index.html smoke) in headless Chromium on SwiftShader.
//   node run/check_page.mjs [check|index]
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "web");
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".bin": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const f = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
  fs.readFile(f, (err, data) => { if (err) { res.writeHead(404); res.end(); return; } res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" }); res.end(data); });
}).listen(0);
const port = server.address().port, which = process.argv[2] || "check";
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
await page.goto(`http://localhost:${port}/${which}.html`);
if (which === "check") {
  await page.waitForFunction(() => window.__result, null, { timeout: 900000 });
  console.log(JSON.stringify(await page.evaluate(() => window.__result), null, 1));
} else {
  await page.waitForFunction(() => document.querySelector("#ans .card") || document.querySelector("#route .gap") || document.getElementById("err").textContent, null, { timeout: 900000 });
  console.log(JSON.stringify(await page.evaluate(() => ({ err: document.getElementById("err").textContent, route: document.getElementById("route").innerText.slice(0, 400),
    ans: document.getElementById("ans").innerText.slice(0, 600), stats: document.getElementById("stats").innerText })), null, 1));
  await page.screenshot({ path: path.join(root, "..", "out", "page.png"), fullPage: true });
}
await browser.close(); server.close();
