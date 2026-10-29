// The whole page in headless Chromium (SwiftShader WebGPU; correctness, not speed), with a fake camera and a fake
// microphone made by make-fixtures.py: a drawing of a circle and a square, then a child's hand with the index finger
// inside the circle, and seven taps in the audio. MediaPipe and its model come from their CDNs through the proxy.
//   python3 test/make-fixtures.py && node test/e2e.mjs      writes ../results/e2e.json
// Passes when the hand tracker finds the fingertip inside the circle, the walls find both drums, and at least four
// taps are heard and struck on the circle, with no errors. SwiftShader runs the hand tracker at ~3 frames a second
// here, so the page is told to accept a hand seen up to 1.5 s from a tap (handwin); in use the window is 250 ms.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url)), WEB = path.join(HERE, "..", "web"), FX = path.join(HERE, "fixtures");
const MIME = { ".html": "text/html", ".js": "text/javascript" };
const srv = http.createServer((q, r) => {
  const p = path.join(WEB, decodeURIComponent(new URL(q.url, "http://x").pathname).replace(/\/$/, "/index.html"));
  if (!p.startsWith(WEB) || !fs.existsSync(p)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "content-type": MIME[path.extname(p)] || "application/octet-stream" }); r.end(fs.readFileSync(p));
});
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${srv.address().port}/`;
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "127.0.0.1,localhost" } : undefined;
const b = await chromium.launch({ proxy, args: ["--ignore-certificate-errors", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader",
  "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
  `--use-file-for-fake-video-capture=${path.join(FX, "fake.y4m")}`, `--use-file-for-fake-audio-capture=${path.join(FX, "fake.wav")}`,
  "--autoplay-policy=no-user-gesture-required"] });
const page = await b.newPage(); const errors = [];
page.on("pageerror", (e) => errors.push(e.message)); page.on("console", (m) => m.type() === "error" && !m.text().startsWith("INFO:") && errors.push(m.text()));   // MediaPipe logs its INFO lines as errors
await page.goto(base + "index.html?autostart=1&grid=128x96&handwin=1500&handdelegate=" + (process.argv[2] || "CPU"));
const samples = []; let s = null;
for (let k = 0; k < 40; k++) {
  await page.waitForTimeout(1000);
  s = await page.evaluate(() => window.__stats && window.__stats());
  if (s) samples.push({ t: k + 1, taps: s.taps, hits: s.hits, hands: s.hands, tips: s.tips, drums: s.drums, under: s.under, frames: s.frames, handFrames: s.handFrames });
  if (s && s.hits >= 5) break;
}
await page.screenshot({ path: path.join(HERE, "..", "results", "e2e.png") });
await b.close(); srv.close();
const tipInCircle = samples.some((x) => x.tips.some(([tx, ty]) => Math.hypot((tx - 0.36) / 0.18, (ty - 0.48) / 0.24) < 1));
const res = { final: s, samples, errors, tipInCircle,
  pass: !!s && tipInCircle && s.drums.length >= 2 && s.hits >= 4 && !errors.length && s.log.filter((e) => e.result === "hit").every((e) => Math.hypot((e.cell[0] / 128 - 0.36) / 0.18, (e.cell[1] / 96 - 0.48) / 0.24) < 1) };
fs.writeFileSync(path.join(HERE, "..", "results", "e2e.json"), JSON.stringify(res, null, 1));
console.log(JSON.stringify({ pass: res.pass, tipInCircle, errors, final: s && { taps: s.taps, hits: s.hits, miss: s.miss, drums: s.drums, under: s.under, handMs: s.handMs, handDelegate: s.handDelegate, handLagMs: s.handLagMs, handFrames: s.handFrames, frames: s.frames, wallMs: s.wallMs, log: s.log.slice(-8), mic: s.mic } }, null, 1));
process.exit(res.pass ? 0 : 1);
