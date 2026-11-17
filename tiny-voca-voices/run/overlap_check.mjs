// After a second performance starts, is anything from the first still playing? Wraps AudioBufferSourceNode
// start/stop, tags each source with the request it belongs to, sings twice in a row (as a slider release with
// "Play again" does) and lists the requests whose sources are still sounding.
//   NODE_PATH=$(npm root -g) node run/overlap_check.mjs [BASE_URL] [ENGINE]
import { createRequire } from "node:module";
import { join } from "node:path";
const pw = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const base = process.argv[2] || "http://localhost:8765/";
const engine = process.argv[3] || "chromium";
const browser = await pw[engine].launch({ proxy: { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } });
const ctx = await browser.newContext(engine === "webkit" ? pw.devices["iPhone 13"] : {});
await ctx.addInitScript(() => {
  window.__sounding = new Map();
  const start = AudioBufferSourceNode.prototype.start, stop = AudioBufferSourceNode.prototype.stop;
  AudioBufferSourceNode.prototype.start = function (...a) {
    const tag = typeof request !== "undefined" ? request : -1, end = this.context.currentTime + (a[0] || 0) + this.buffer.duration;
    window.__sounding.set(this, { tag, end });
    return start.apply(this, a);
  };
  AudioBufferSourceNode.prototype.stop = function (...a) { window.__sounding.delete(this); return stop.apply(this, a); };
});
const page = await ctx.newPage();
await page.goto(base + "?mode=sing&seed=12345");
await page.waitForFunction(() => ready === true, null, { timeout: 120000 });
for (const voices of [1, 4]) {
  const r = await page.evaluate(async (voices) => {
    const c = SONG.find((x) => x.id === "voices"); c.input.value = voices; c.show();
    const wait = async (id) => { while (!(lastClip && lastClip.id === id)) await new Promise((r) => setTimeout(r, 50)); };
    const first = request + 1; perform(); await wait(first);
    await new Promise((r) => setTimeout(r, 300));
    const second = request + 1; perform(); await wait(second);
    await new Promise((r) => setTimeout(r, 300));
    const now = [...window.__sounding.values()].filter((s) => s.end > (window.__ctxTime = document.querySelector("body") && (typeof context !== "undefined" ? context.currentTime : 0)));
    const tags = [...new Set(now.map((s) => s.tag))];
    return { first, second, stillSounding: tags, sources: now.length };
  }, voices);
  console.log(`${engine} | ${voices === 1 ? "solo" : voices + "-voice chorus"}: requests ${r.first} then ${r.second}; still sounding from requests ${JSON.stringify(r.stillSounding)} (${r.sources} sources)`);
}
await browser.close();
