// The iPhone bug: play a line, leave the page, come back, play another. Checks that the second play gets a fresh,
// running AudioContext and starts sound in it. Serves web/ locally like click_check.mjs.
//   NODE_PATH=$(npm root -g) ORT_DIST=... node revisit_check.mjs
import { createServer } from "node:http";
import { readFileSync, statSync, existsSync } from "node:fs";
import { join, extname, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const HERE = dirname(fileURLToPath(import.meta.url));
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm" };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const file = path.startsWith("/models/") ? join(process.env.MODELS || "/opt/kokoro-web-models", path.slice(8))
    : path.startsWith("/ort/") ? join(process.env.ORT_DIST, path.slice(5)) : join(HERE, "web", path === "/" ? "index.html" : path);
  if (!existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "Content-Type": TYPES[extname(file)] || "application/octet-stream", "Content-Length": statSync(file).size });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, r));
const browser = await chromium.launch({ args: ["--autoplay-policy=user-gesture-required"] });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.addInitScript(() => {
  window.__ctx = []; window.__starts = [];
  const Orig = AudioContext;
  window.AudioContext = class extends Orig { constructor(...a) { super(...a); window.__ctx.push(this); } };
  const start = AudioBufferSourceNode.prototype.start;
  AudioBufferSourceNode.prototype.start = function (...a) {
    window.__starts.push({ ctx: window.__ctx.indexOf(this.context), samples: this.buffer?.length ?? 0, state: this.context.state });
    return start.apply(this, a);
  };
});
await page.goto(`http://localhost:${server.address().port}/?ort=/ort/&model=models/kokoro-web.onnx`);
for (let i = 0; ; i++) { try { await page.waitForFunction(() => window.ready === true, null, { timeout: 300000 }); break; } catch (e) { if (i > 3) throw e; } }
const waitDone = (k) => page.waitForFunction((k) => /real time|Error/.test(document.querySelectorAll(".line .meta")[k].textContent), k, { timeout: 300000 });
await page.locator(".line button").nth(4).click(); // a short line
await waitDone(4);
// leave and come back: the page is hidden, then visible again
await page.evaluate(() => {
  Object.defineProperty(document, "hidden", { configurable: true, get: () => true });
  document.dispatchEvent(new Event("visibilitychange"));
  Object.defineProperty(document, "hidden", { configurable: true, get: () => false });
  document.dispatchEvent(new Event("visibilitychange"));
});
await page.locator(".line button").nth(9).click();
await waitDone(9);
const r = await page.evaluate(() => ({ contexts: window.__ctx.map((c) => c.state), starts: window.__starts }));
console.log(JSON.stringify({ ...r, errors }));
await browser.close(); server.close();
