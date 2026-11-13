// Open the published page as a visitor would (through the session's proxy), let coi.js reload it once, download
// the model from R2, and speak one line.
//   NODE_PATH=$(npm root -g) node live_check.mjs [URL]
import { createRequire } from "node:module";
import { join } from "node:path";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");

const url = process.argv[2] || "https://austegard.com/experiments/kokoro-web/web/";
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY } : undefined;
const browser = await chromium.launch({ proxy });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
const t0 = Date.now();
await page.goto(url);
for (let tries = 0; ; tries++) { // the service worker reloads the page once; the wait survives that
  try {
    await page.waitForFunction(() => window.ready === true || document.getElementById("status")?.textContent.startsWith("Error"),
      null, { timeout: 600000 });
    break;
  } catch (e) { if (tries > 3) throw e; }
}
const loaded = (Date.now() - t0) / 1000;
const info = await page.evaluate(async () => {
  if (!window.bench) return { isolated: crossOriginIsolated, status: document.getElementById("status").textContent };
  const r = await window.bench(1);
  return { isolated: crossOriginIsolated, status: document.getElementById("status").textContent, stats: r.stats,
           line: { words: r.lines[0].words.length, ms: Math.round(r.lines[0].ms), seconds: r.lines[0].audio.length / 24000 } };
});
console.log(JSON.stringify({ url, loaded_s: loaded, ...info, errors }));
await browser.close();
