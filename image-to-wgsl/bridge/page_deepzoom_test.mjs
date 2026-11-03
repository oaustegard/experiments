// node page_deepzoom_test.mjs <image> <outdir>: headless functional test of the page's deep zoom on SwiftShader
// (it checks that tiles train and draw, not speed). Encodes the image with short training, zooms in, waits for the
// view's detail tiles, and screenshots the view with detail on and off.
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const [image, outdir] = process.argv.slice(2);
fs.mkdirSync(outdir, { recursive: true });
const HERE = path.dirname(new URL(import.meta.url).pathname), WEB = path.join(HERE, "../web");
const types = { ".js": "text/javascript", ".html": "text/html", ".jpg": "image/jpeg", ".avif": "image/avif", ".webp": "image/webp", ".png": "image/png" };
const srv = http.createServer((q, r) => { const u = q.url.split("?")[0], f = path.join(WEB, u === "/" ? "index.html" : u);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "Content-Type": types[path.extname(f)] || "application/octet-stream" }); r.end(fs.readFileSync(f)); });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const p = await b.newPage({ viewport: { width: 900, height: 1100 } });
let errors = 0;
p.on("console", (m) => { const t = m.text(); if (/error|failed/i.test(t)) errors++; console.log("[page]", t.slice(0, 300)); });
p.on("pageerror", (e) => { errors++; console.log("[err]", e.message); });
await p.goto(`http://127.0.0.1:${srv.address().port}/`);
await p.waitForFunction(() => window.gpuDone, null, { timeout: 60000 });
await p.evaluate(() => { window.__encodeOpts = { stepsStage: 10, stepsFinal: 30, stepsQat: 10 };
  window.__dzOpts = { nTile: 32, steps: { stepsStage: 10, stepsFinal: 30, stepsQat: 10 } }; });
await p.evaluate(() => document.querySelector("button.own").click());
await p.setInputFiles("#file", image);
await p.waitForFunction(() => !document.getElementById("go").disabled, null, { timeout: 600000 });
await p.selectOption("#nsel", "64");
await p.evaluate(() => document.getElementById("go").click());
await p.waitForFunction(() => window.__encodeResult || /failed/.test(document.getElementById("encstat").textContent), null, { timeout: 1800000 });
console.log("after:", await p.textContent("#encstat"));
await p.waitForTimeout(7000);   // the paint-in
console.log("dz row hidden:", await p.$eval("#dzrow", (e) => e.hidden), "|", await p.textContent("#dzstat"));
const box = await p.locator("#stage").boundingBox();
await p.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.45);
for (let i = 0; i < 4; i++) await p.mouse.wheel(0, -350);
await p.waitForTimeout(500);
console.log("zoom:", await p.textContent("#zlvl"), "|", await p.textContent("#dzstat"));
const t0 = Date.now();
await p.waitForFunction(() => /This view shows level/.test(document.getElementById("dzstat").textContent), null, { timeout: 1800000 });
console.log(`detail done in ${((Date.now() - t0) / 1000).toFixed(0)} s:`, await p.textContent("#dzstat"));
await p.evaluate(() => window.gpuDone());
console.log("readout:", await p.textContent("#readout"), "| tag:", await p.textContent("#tag"));
await p.locator("#stage").screenshot({ path: path.join(outdir, "dz_on.png") });
await p.click("#dzon"); await p.waitForTimeout(400); await p.evaluate(() => window.gpuDone());
console.log("off readout:", await p.textContent("#readout"), "| tag:", await p.textContent("#tag"));
await p.locator("#stage").screenshot({ path: path.join(outdir, "dz_off.png") });
await p.click("#dzon"); await p.waitForTimeout(400);
await p.click("#views button[data-view=orig]"); await p.waitForTimeout(300);
await p.locator("#stage").screenshot({ path: path.join(outdir, "dz_orig.png") });
console.log("errors:", errors);
await b.close(); srv.close();
process.exit(errors ? 1 : 0);
