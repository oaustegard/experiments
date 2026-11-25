// Page smoke test: POWER on, a candidate set appears, ♥ and Skip each record a label, skip cuts to a
// chosen replacement, nothing throws. SwiftShader is too slow for the GPU model to keep up with
// phrases, so the feature model does the choosing here; the GPU path is covered by harvest parity.
//   node run/smoke.mjs [?brain=ettin]   (with the experiments root served on 127.0.0.1:8765)
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");

const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required", "--enable-unsafe-webgpu",
  "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.goto("http://127.0.0.1:8765/strudel-taste/web/index.html" + (process.argv[2] || ""), { waitUntil: "load" });
await page.evaluate(() => localStorage.clear());
await page.click("#power");
await page.waitForFunction(() => window.__fm.running, null, { timeout: 60000 });
await page.waitForFunction(() => document.querySelectorAll("#t-cands .tc").length === 8, null, { timeout: 90000 });
const cands = await page.$$eval("#t-cands .tc", (els) => els.map((e) => e.textContent));
// wait until a phrase picked from candidates is the one playing, then like it
await page.waitForFunction(() => !!window.__fm.taste.playing?.taste, null, { timeout: 90000 });
await page.keyboard.press("l");
const afterLike = await page.textContent("#t-count");
await page.waitForFunction(() => { const p = window.__fm.taste.playing; return p?.taste && !p.tasteReacted; }, null, { timeout: 90000 });
const before = await page.evaluate(() => window.__fm.taste.playing.start);
await page.click("#t-skip");
await page.waitForFunction((b) => { const p = window.__fm.taste.playing; return p && p.start !== b; }, before, { timeout: 30000 });
const after = await page.evaluate(() => ({ start: window.__fm.taste.playing.start, picked: !!window.__fm.taste.playing.taste, c: Math.floor(window.__fm.nowCycle()) }));
const count = await page.textContent("#t-count");
const brain = await page.textContent("#t-brain");
const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("strudel-taste-v1") || "[]").map((e) => e.kind));
console.log(JSON.stringify({ cands: cands.slice(0, 3), afterLike, skippedFrom: before, after, count, brain, stored, errors }, null, 1));
await browser.close();
process.exit(errors.length || !stored.includes("like") || !stored.includes("skip") ? 1 : 0);
