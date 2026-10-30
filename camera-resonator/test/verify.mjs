// Headless checks on SwiftShader WebGPU (correctness, not speed):
//   node test/verify.mjs selftest   GPU block stream against the float64 reference (plate, random strikes)
//   node test/verify.mjs live       the page with Chromium's fake camera: strikes fire, blocks stream, no WebGPU errors
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");   // the container has it global; require() honours NODE_PATH
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const WEB = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "web");
const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
const srv = http.createServer((q, r) => {
  const p = path.join(WEB, decodeURIComponent(new URL(q.url, "http://x").pathname).replace(/\/$/, "/index.html"));
  if (!p.startsWith(WEB) || !fs.existsSync(p)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "content-type": MIME[path.extname(p)] || "application/octet-stream" }); r.end(fs.readFileSync(p));
});
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${srv.address().port}/`;
// hand tracking fetches MediaPipe from jsDelivr and its model from storage.googleapis.com; in a proxied container
// Chromium needs the proxy passed explicitly
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "127.0.0.1,localhost" } : undefined;
const b = await chromium.launch({ proxy, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader",
  "--use-webgpu-adapter=swiftshader", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"] });
const page = await b.newPage(); const errors = [];
page.on("pageerror", (e) => errors.push(e.message)); page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
const what = process.argv[2] || "selftest";
let res;
if (what === "selftest") {
  await page.goto(base + "index.html?nostart=1");
  res = await page.evaluate(async () => {
    const E = await import("./engine.js");
    const ad = await navigator.gpu.requestAdapter(), dev = await ad.requestDevice();
    const out = [];
    for (const block of [512, 1024]) out.push(await E.selfTest(dev, { modes: 1200, blocks: 16, block }));
    const S = E.bells(48000), e = await E.ModalEngine.create(dev, S, { block: 512 });
    const blocks = [[{ x: 0.1, y: 0.3, vel: 1, tau: 4e-4, js: 100 }, { x: 0.9, y: 0.7, vel: 0.8, tau: 3e-4, js: 400 }], [], [], []];
    const gpu = []; for (const s of blocks) gpu.push(...await e.render(s));
    const ref = E.reference(S, blocks, 512); let pk = 0, se = 0;
    ref.forEach((v, i) => { pk = Math.max(pk, Math.abs(v)); se += (gpu[i] - v) ** 2; });
    out.push({ preset: S.label, peak: pk, rmsDb: 20 * Math.log10(Math.sqrt(se / ref.length) / pk) });
    return out;
  });
} else if (what === "synthetic") {
  // the synthetic scene: impacts should land where the disc stopped, twice per 4 s cycle, and nowhere else
  const out = {};
  for (const input of ["impact", "any"]) {
    await page.goto(base + `index.html?autostart=1&synthetic=1&input=${input}&modes=300`);
    await page.waitForTimeout(+(process.argv[3] || 12500));
    const s = await page.evaluate(() => window.__stats());
    const near = (k) => s.expected.map((e) => Math.hypot(k.x - e[0], k.y - e[1])).some((d) => d < 0.08);
    out[input] = { frames: s.frames, sceneSeconds: +(s.frames / 30).toFixed(1), strikes: s.log.length, atStops: s.log.filter(near).length, elsewhere: s.log.filter((k) => !near(k)).map((k) => [k.t, k.x, k.y]) };
  }
  res = out;
} else if (what === "hands") {
  // the hand model loads and runs on the fake camera (which shows no hand): no errors, frames advance, no strikes
  await page.goto(base + `index.html?autostart=1&input=hands&modes=300`);
  await page.waitForTimeout(+(process.argv[3] || 45000));
  const s = await page.evaluate(() => window.__stats());
  res = { input: s.input, frames: s.frames, strikes: s.strikes, note: await page.evaluate(() => document.getElementById("status").textContent.split(" · ").pop()) };
  if (s.input !== "hands") errors.push("fell back from hand tracking: " + res.note);
} else if (what === "switch") {
  // queue depth must stay bounded after the resonator is rebuilt (a reset counter once let it grow without limit)
  await page.goto(base + `index.html?autostart=1&input=impact&modes=${process.argv[3] || 400}`);
  await page.waitForTimeout(4000);
  const before = await page.evaluate(() => window.__stats());
  await page.selectOption("#preset", "gong");
  await page.waitForTimeout(+(process.argv[4] || 8000));
  const after = await page.evaluate(() => window.__stats());
  res = { before: { queue: before.queue, sent: before.sent, label: before.label }, after: { queue: after.queue, sent: after.sent, done: after.done, label: after.label, latencyMs: after.latencyMs } };
  if (!(after.queue >= 0 && after.queue <= 8)) errors.push(`queue after rebuild is ${after.queue} blocks`);
} else {
  await page.goto(base + `index.html?autostart=1&input=${process.argv[5] || "impact"}&modes=${process.argv[3] || 400}`);
  await page.waitForTimeout(+(process.argv[4] || 12000));
  res = await page.evaluate(() => window.__stats && window.__stats());
}
console.log(JSON.stringify({ what, res, errors }, null, 1));
await b.close(); srv.close();
if (errors.length) process.exit(1);
