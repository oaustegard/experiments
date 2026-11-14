// Click Play on line 1 as a visitor would and check what they would see and hear: the Generating clock ticks
// while the line is made (the page is not frozen), the AudioContext runs, a sound source starts, and the line's
// timing message appears. Serves web/ like GitHub Pages (no isolation headers), with the model from $MODELS.
//   NODE_PATH=$(npm root -g) ORT_DIST=... node click_check.mjs [extra query]
import { createServer } from "node:http";
import { readFileSync, statSync, existsSync } from "node:fs";
import { join, extname, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");

const HERE = dirname(fileURLToPath(import.meta.url));
const MODELS = process.env.MODELS || "/opt/kokoro-web-models", ORT_DIST = process.env.ORT_DIST;
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm" };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const file = path.startsWith("/models/") ? join(MODELS, path.slice(8)) : path.startsWith("/ort/") ? join(ORT_DIST, path.slice(5))
    : join(HERE, "web", path === "/" ? "index.html" : path);
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
  window.__starts = [];
  const orig = AudioBufferSourceNode.prototype.start;
  AudioBufferSourceNode.prototype.start = function (...a) {
    window.__starts.push({ at: performance.now(), samples: this.buffer?.length ?? 0, state: this.context.state });
    return orig.apply(this, a);
  };
});
await page.goto(`http://localhost:${server.address().port}/?ort=/ort/&model=models/kokoro-web.onnx&${process.argv[2] || ""}`);
for (let i = 0; ; i++) {
  try { await page.waitForFunction(() => window.ready === true || document.getElementById("status")?.textContent.startsWith("Error"), null, { timeout: 300000 }); break; }
  catch (e) { if (i > 3) throw e; }
}
const backend = await page.evaluate(() => document.getElementById("s-backend")?.textContent);
const t0 = Date.now();
const clickAt = await page.evaluate(() => performance.now());
await page.locator(".line button").first().click();
const seen = new Set();
let done = false;
while (!done && Date.now() - t0 < 300000) {
  const m = await page.locator(".line .meta").first().textContent();
  seen.add(m.replace(/\d+ s$/, "N s"));
  if (m.startsWith("Generating")) seen.add(m);
  done = /real time|Error/.test(m) && !/^Playing/.test(m);
  await new Promise((r) => setTimeout(r, 400));
}
const state = await page.evaluate((c) => ({ starts: window.__starts.map((x) => ({ after_s: +((x.at - c) / 1000).toFixed(2), samples: x.samples, ctx: x.state })), meta: document.querySelector(".line .meta").textContent,
  highlighted: document.querySelectorAll(".words span").length }), clickAt);
const ticks = [...seen].filter((s) => /^Generating… \d+ s$/.test(s)).length;
console.log(JSON.stringify({ backend, seconds: (Date.now() - t0) / 1000, clockTicksSeen: ticks, ...state, errors }));
await browser.close(); server.close();
