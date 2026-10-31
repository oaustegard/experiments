// Headless checks on SwiftShader WebGPU (correctness, not speed):
//   node test/verify.mjs math     the orientation -> wheel/pedal mapping on known poses
//   node test/verify.mjs demo     autopilot race: frames render, no WebGPU errors, cars lap; writes results/demo.json + shots
//   node test/verify.mjs tilt     synthetic deviceorientation events drive the player car: right turn turns right, tilt-away accelerates
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url)), WEB = path.join(HERE, "..", "web"), OUT = path.join(HERE, "..", "results");
const srv = http.createServer((q, r) => {
  const p = path.join(WEB, decodeURIComponent(new URL(q.url, "http://x").pathname).replace(/\/$/, "/index.html"));
  if (!p.startsWith(WEB) || !fs.existsSync(p)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "content-type": p.endsWith(".html") ? "text/html" : "text/javascript" }); r.end(fs.readFileSync(p));
});
await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${srv.address().port}/`;
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"] });
const what = process.argv[2] || "math";
let failed = false;
const check = (name, ok, detail = "") => { console.log((ok ? "ok   " : "FAIL ") + name + (detail ? "  " + detail : "")); if (!ok) failed = true; };
async function open(url, viewport = { width: 844, height: 390 }) {
  const page = await b.newPage({ viewport }), errors = [];
  page.on("pageerror", (e) => errors.push(e.message)); page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(base + url); return { page, errors };
}
const waitFor = (page, fn, ms = 60000) => page.waitForFunction(fn, null, { timeout: ms, polling: 250 });
fs.mkdirSync(OUT, { recursive: true });

if (what === "math") {
  const { page } = await open("index.html?nostart");
  await waitFor(page, () => window.__tilt);
  const r = await page.evaluate(() => {
    const { upFromOrientation: up, wheelFrom: w } = window.__tilt, deg = (x) => x * 180 / Math.PI;
    const N = up(0, -75);                                   // landscape, screen tilted back 15 degrees
    return {
      flat: up(0, 0), portrait: up(90, 0), landscape: up(0, -90),
      right20: deg(w(up(20, -75), N).wheel),                // the wheel turned clockwise: beta rises in this pose
      left20: deg(w(up(-20, -75), N).wheel),
      away15: deg(w(up(0, -60), N).pitch),                  // screen faces more of the sky
      toward15: deg(w(up(0, -89), N).pitch),
      seam: deg(w(up(180, 89), up(0, -89)).pitch),          // past vertical: beta flips to 180, gamma to +89; up.z goes 0.017 -> -0.017
      otherLandscape: deg(w(up(-20, 75), up(0, 75)).wheel), // the phone the other way round: clockwise lowers beta here
    };
  });
  const near = (a, b, e = 1.0) => Math.abs(a - b) < e;
  check("flat on a table: up = +z", near(r.flat[2], 1, 1e-6), JSON.stringify(r.flat));
  check("upright portrait: up = +y", near(r.portrait[1], 1, 1e-6));
  check("landscape upright: up = +x", near(r.landscape[0], 1, 1e-6));
  check("clockwise 20 degrees steers right +20", near(r.right20, 20), r.right20.toFixed(2));
  check("counter-clockwise 20 degrees steers left -20", near(r.left20, -20), r.left20.toFixed(2));
  check("top tilted away 15 degrees is +15 pedal", near(r.away15, 15), r.away15.toFixed(2));
  check("tilted toward you 14 degrees is -14 pedal", near(r.toward15, -14), r.toward15.toFixed(2));
  check("continuous through the beta/gamma seam", near(r.seam, -2, 0.2), r.seam.toFixed(2));
  check("other landscape: clockwise is still right", near(r.otherLandscape, 20), r.otherLandscape.toFixed(2));
}

if (what === "demo") {
  const { page, errors } = await open("index.html?demo&seed=" + (process.argv[3] || 7));
  await waitFor(page, () => window.__game?.()?.state === "race", 120000);
  const t0 = Date.now();
  await page.screenshot({ path: path.join(OUT, "demo-start.png") });
  await waitFor(page, () => window.__game().raceT > 25, 600000);
  await page.screenshot({ path: path.join(OUT, "demo-25s.png") });
  const s = await page.evaluate(() => window.__game());
  s.wallSeconds = (Date.now() - t0) / 1000;
  fs.writeFileSync(path.join(OUT, "demo.json"), JSON.stringify(s, null, 1));
  console.log(JSON.stringify(s.stats), "frames", s.frames, "wall", s.wallSeconds.toFixed(1));
  for (const c of s.cars) console.log("  ", c.name, "lap", c.lap, "idx", c.idx, "v", (c.speed * 3.6).toFixed(0), "km/h lat", c.lat.toFixed(1), "surf", c.surf);
  check("no page or WebGPU errors", errors.length === 0, errors.slice(0, 3).join(" | "));
  check("every car moving and on the road", s.cars.every((c) => c.speed > 8 && Math.abs(c.lat) < 9));
  check("track has no corner tighter than 25 m", s.stats.minR > 25, s.stats.minR.toFixed(1) + " m");
}

if (what === "tilt") {
  const { page, errors } = await open("index.html");
  await waitFor(page, () => window.__game?.() && !document.getElementById("start").disabled && document.getElementById("status").textContent === "Ready.", 120000);
  const send = (beta, gamma) => page.evaluate(([b, g]) => window.dispatchEvent(Object.assign(new Event("deviceorientation"), { alpha: 0, beta: b, gamma: g })), [beta, gamma]);
  await page.click("#start");
  for (let k = 0; k < 20; k++) { await send(0, -75); await page.waitForTimeout(100); }
  await waitFor(page, () => window.__game().state === "race", 60000);
  // pedal: tilt away 15 degrees for a while, then check speed
  const t1 = await page.evaluate(() => window.__game().raceT);
  while ((await page.evaluate(() => window.__game().raceT)) < t1 + 3) { await send(0, -60); await page.waitForTimeout(50); }
  const s1 = await page.evaluate(() => window.__game().cars.find((c) => c.name === "You"));
  check("tilting the top away accelerates", s1.speed > 10, (s1.speed * 3.6).toFixed(0) + " km/h");
  // steering: turn 30 degrees clockwise and watch the lateral offset move right (lat > 0 is right of the centreline)
  const lat0 = s1.lat, t2 = await page.evaluate(() => window.__game().raceT);
  while ((await page.evaluate(() => window.__game().raceT)) < t2 + 1.2) { await send(30, -60); await page.waitForTimeout(50); }
  const s2 = await page.evaluate(() => window.__game().cars.find((c) => c.name === "You"));
  check("turning the phone clockwise moves the car right", s2.lat > lat0 + 1, lat0.toFixed(1) + " -> " + s2.lat.toFixed(1));
  const t3 = await page.evaluate(() => window.__game().raceT);
  while ((await page.evaluate(() => window.__game().raceT)) < t3 + 1.5) { await send(0, -89); await page.waitForTimeout(50); }
  const s3 = await page.evaluate(() => window.__game().cars.find((c) => c.name === "You"));
  check("tilting toward you brakes", s3.speed < s2.speed - 5, (s2.speed * 3.6).toFixed(0) + " -> " + (s3.speed * 3.6).toFixed(0) + " km/h");
  await page.screenshot({ path: path.join(OUT, "tilt.png") });
  check("no page or WebGPU errors", errors.length === 0, errors.slice(0, 3).join(" | "));
}

await b.close(); srv.close();
process.exit(failed ? 1 : 0);
