// First visit: bytes transferred, by origin. Second visit: the network is switched off, and the page must still
// load and speak from its caches. Also checks that a second TinyVoca.load() on one page returns the first instance.
//   NODE_PATH=$(npm root -g) node run/cache_check.mjs OUT.json [BASE_URL]
import { createRequire } from "node:module";
import { join } from "node:path";
import { writeFileSync } from "node:fs";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const out = process.argv[2];
const base = process.argv[3] || "http://localhost:8765/";
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } : undefined;
const browser = await chromium.launch({ proxy });
const context = await browser.newContext();
const rows = [];
context.on("requestfinished", async (req) => {
  try {
    const sizes = await req.sizes();
    rows.push({ url: req.url().split("?")[0], bytes: sizes.responseBodySize + sizes.responseHeadersSize });
  } catch (e) { /* request gone */ }
});
const errors = [];
const TEXT = "Turn left at the second traffic light.";

async function visit(label) {
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(`${label}: ${e}`));
  const t0 = Date.now();
  await page.goto(base);
  await page.waitForFunction(() => ready === true || document.getElementById("status").classList.contains("error"), null, { timeout: 600000 });
  const loaded = (Date.now() - t0) / 1000;
  const spoke = await page.evaluate(async (text) => {
    if (!ready) return { error: document.getElementById("status").textContent };
    document.getElementById("text").value = text;
    const before = request;
    speak();
    const t = performance.now();
    while (!(lastClip && lastClip.id === before + 1)) {
      await new Promise((r) => setTimeout(r, 50));
      if (document.getElementById("status").classList.contains("error")) return { error: document.getElementById("status").textContent };
      if (performance.now() - t > 30000) return { error: "timed out" };
    }
    return { samples: lastClip.audio.length };
  }, TEXT);
  await page.close();
  return { loaded_s: loaded, ...spoke };
}

const first = await visit("first");
const firstBytes = rows.reduce((s, r) => s + r.bytes, 0);
const byHost = {};
for (const r of rows) { const h = new URL(r.url).host; byHost[h] = (byHost[h] || 0) + r.bytes; }
const biggest = [...rows].sort((a, b) => b.bytes - a.bytes).slice(0, 8).map((r) => [r.url, r.bytes]);
rows.length = 0;
await context.setOffline(true);
const second = await visit("second, offline");
await context.setOffline(false);

// Memoized load: two calls on one page give the same instance, the second without loading anything.
const page = await context.newPage();
await page.goto(new URL("example.html", base).href);
await page.waitForFunction(() => document.getElementById("status").textContent === "Ready.", null, { timeout: 600000 });
const memo = await page.evaluate(async () => {
  const a = await TinyVoca.load();
  const t = performance.now();
  const b = await TinyVoca.load();
  return { same_instance: a === b, second_load_ms: Math.round(performance.now() - t) };
});
const report = { base, first_visit: { ...first, transferred_mb: +(firstBytes / 1e6).toFixed(2), requests: biggest.length && undefined,
  by_host_mb: Object.fromEntries(Object.entries(byHost).map(([h, b]) => [h, +(b / 1e6).toFixed(2)])), biggest },
  second_visit_offline: second, memoized_load: memo, errors };
writeFileSync(out, JSON.stringify(report, null, 1));
console.log(JSON.stringify(report, null, 1));
await browser.close();
