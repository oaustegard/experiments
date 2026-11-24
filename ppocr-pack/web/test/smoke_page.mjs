// Headless smoke test of web/index.html on SwiftShader WebGPU: the page loads, Run completes, text appears, no
// console errors. Correctness only; SwiftShader timings mean nothing.
//
//   node web/test/smoke_page.mjs [tiny|small] [samples/letter.png] [long=640]     # exits 0 on success
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.execPath, "../../lib/node_modules/playwright"))); }
const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const size = args.find((a) => a === "tiny" || a === "small") || "tiny";
const sample = args.find((a) => a.startsWith("samples/")) || "samples/letter.png";
const long = (args.find((a) => a.startsWith("long=")) || "long=640").slice(5);
const types = { ".js": "text/javascript", ".json": "application/json", ".html": "text/html", ".png": "image/png", ".txt": "text/plain; charset=utf-8" };
const server = http.createServer((req, res) => {
  let u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") u = "/index.html";
  const f = u.startsWith("/gpu-bridge/") ? path.join(WEB, "../..", u) : path.join(WEB, u);
  if (!f.startsWith(WEB) && !u.startsWith("/gpu-bridge/")) { res.writeHead(404); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const flags = ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"];
const exe = fs.existsSync("/opt/pw-browsers/chromium") ? "/opt/pw-browsers/chromium" : undefined;
const browser = await chromium.launch({ executablePath: exe, args: flags });
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/?long=${long}`);
await page.waitForFunction(() => !document.getElementById("run").disabled, null, { timeout: 60000 });
await page.selectOption("#size", size);
await page.selectOption("#sample", sample);
await page.waitForTimeout(500);
const t0 = Date.now();
await page.click("#run");
await page.waitForFunction(() => /lines in/.test(document.getElementById("status").textContent) || document.getElementById("status").className === "err", null, { timeout: 1500000, polling: 2000 });
const out = await page.evaluate(() => ({ status: document.getElementById("status").textContent, err: document.getElementById("status").className === "err",
  text: document.getElementById("text").textContent, stats: [...document.querySelectorAll("#stats tr")].map((r) => r.textContent) }));
await browser.close();
server.close();
console.log(JSON.stringify({ size, sample, long, seconds: Math.round((Date.now() - t0) / 1000), ...out, errors }, null, 1));
process.exit(out.err || errors.length || !out.text.trim() || /Run to read/.test(out.text) ? 1 : 0);
