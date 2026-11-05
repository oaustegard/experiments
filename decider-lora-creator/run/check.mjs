// Serve the experiments repo root and run decider-lora-creator's pages in headless Chromium on SwiftShader.
//   node run/check.mjs check [query]     -> web/check.html, prints window.__result, writes results/check_page.json
//   node run/check.mjs flow              -> index.html?demo: a short head-only run through the UI, screenshot to out/
import { createRequire } from "node:module";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const { chromium } = createRequire(import.meta.url)("playwright");
const exp = path.resolve(path.dirname(new URL(import.meta.url).pathname), ".."), root = path.resolve(exp, "..");
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".bin": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const f = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!f.startsWith(root)) { res.writeHead(403); res.end(); return; }
  fs.readFile(f, (err, data) => { if (err) { res.writeHead(404); res.end(); return; } res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" }); res.end(data); });
}).listen(0);
const port = server.address().port, which = process.argv[2] || "check";
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 900 }, deviceScaleFactor: which === "flow" ? 2 : 1 });
page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
page.on("pageerror", (e) => console.log("pageerror:", e.message));
const base = `http://localhost:${port}/decider-lora-creator/web/`;
fs.mkdirSync(path.join(exp, "out"), { recursive: true });
if (which === "check") {
  await page.goto(base + "check.html" + (process.argv[3] ? "?" + process.argv[3] : ""));
  await page.waitForFunction(() => window.__result, null, { timeout: 1800000 });
  const r = await page.evaluate(() => window.__result);
  console.log(JSON.stringify(r, null, 1));
  fs.writeFileSync(path.join(exp, "results", "check_page.json"), JSON.stringify(r, null, 1));
  process.exitCode = r.ok ? 0 : 1;
} else {
  await page.goto(base + "index.html?demo");
  await page.waitForFunction(() => window.__creator?.state.cases.length > 0, null, { timeout: 60000 });
  await page.fill("#maxtr", process.argv[3] || "16"); await page.fill("#maxte", process.argv[4] || "6"); await page.fill("#epochs", process.argv[5] || "3");
  await page.click("#train");
  await page.waitForFunction(() => window.__lastRun, null, { timeout: 1800000 });
  const r = await page.evaluate(() => window.__lastRun);
  await page.fill("#tstate", await page.evaluate(() => window.__creator.state.cases.find((c) => c.split === "test").state));
  await page.click("#try");
  await page.waitForFunction(() => document.querySelector("#ans .card") || document.getElementById("err").textContent, null, { timeout: 600000 });
  const err = await page.evaluate(() => document.getElementById("err").textContent);
  await page.screenshot({ path: path.join(exp, "out", "page.png"), fullPage: true });
  console.log(JSON.stringify({ err, run: r, answer: await page.evaluate(() => document.getElementById("ans").innerText.slice(0, 400)) }, null, 1));
  process.exitCode = err || r.error ? 1 : 0;
}
await browser.close(); server.close();
