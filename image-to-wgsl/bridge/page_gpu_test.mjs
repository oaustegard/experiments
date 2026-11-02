// node page_gpu_test.mjs <image> <outdir> [n]: headless functional test of the page's WebGPU encoder and its
// HTML download. SwiftShader checks that it runs and what it shows, not speed.
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const [image, outdir, nArg = "64"] = process.argv.slice(2);
const HERE = path.dirname(new URL(import.meta.url).pathname), WEB = path.join(HERE, "../web");
const types = { ".js": "text/javascript", ".html": "text/html", ".jpg": "image/jpeg", ".avif": "image/avif", ".webp": "image/webp", ".png": "image/png" };
const srv = http.createServer((q, r) => { const u = q.url.split("?")[0], f = path.join(WEB, u === "/" ? "index.html" : u);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "Content-Type": types[path.extname(f)] || "application/octet-stream" }); r.end(fs.readFileSync(f)); });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const p = await b.newPage({ viewport: { width: 900, height: 1100 } });
p.on("console", (m) => console.log("[page]", m.text().slice(0, 300)));
p.on("pageerror", (e) => console.log("[err]", e.message));
await p.goto(`http://127.0.0.1:${srv.address().port}/`);
await p.waitForFunction(() => window.gpuDone, null, { timeout: 60000 });
await p.evaluate(() => { window.__encodeOpts = { stepsStage: 10, stepsFinal: 20, stepsQat: 10 }; });
await p.evaluate(() => document.querySelector("button.own").click());
await p.setInputFiles("#file", image);
await p.waitForFunction(() => !document.getElementById("go").disabled, null, { timeout: 600000 });
console.log("menu:", await p.$$eval("#nsel option", (o) => o.map((x) => x.textContent + (x.selected ? " *" : ""))));
await p.selectOption("#nsel", nArg);
await p.evaluate(() => document.getElementById("go").click());
await p.waitForFunction(() => window.__encodeResult || /failed/.test(document.getElementById("encstat").textContent), null, { timeout: 1800000 });
console.log("after:", await p.textContent("#encstat"));
const res = await p.evaluate(() => window.__encodeResult);
if (!res) { await b.close(); srv.close(); process.exit(1); }
console.log("points", JSON.stringify(res.points));
await p.waitForTimeout(7000);
await p.evaluate(() => window.gpuDone());
await p.screenshot({ path: path.join(outdir, "page_after.png") });
// zoom the gallery view with the mouse wheel, then compare the shader with the original at that zoom
const box = await p.locator("#stage").boundingBox();
await p.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.4);
for (let i = 0; i < 4; i++) await p.mouse.wheel(0, -350);
await p.waitForTimeout(800); await p.evaluate(() => window.gpuDone());
console.log("gallery zoom:", await p.textContent("#zlvl"));
await p.locator("#stage").screenshot({ path: path.join(outdir, "zoom_shader.png") });
await p.click("#views button[data-view=orig]"); await p.waitForTimeout(300);
await p.locator("#stage").screenshot({ path: path.join(outdir, "zoom_orig.png") });
await p.click("#views button[data-view=shader]");
await p.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5); await p.mouse.down(); await p.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.5, { steps: 5 }); await p.mouse.up();
await p.waitForTimeout(300); await p.evaluate(() => window.gpuDone());
await p.locator("#stage").screenshot({ path: path.join(outdir, "zoom_shader_panned.png") });
await p.click("#zfit"); await p.waitForTimeout(300);
console.log("after Fit:", JSON.stringify(await p.textContent("#zlvl")));
// the HTML download, opened as its own page
const html = await p.evaluate(async () => (await fetch(document.getElementById("dlh").href)).text());
fs.writeFileSync(path.join(outdir, "export.html"), html);
const q = await b.newPage({ viewport: { width: 700, height: 800 } });
q.on("pageerror", (e) => console.log("[export err]", e.message));
const EXP = `http://127.0.0.1:${srv.address().port}/__export.html`;   // a secure origin, so WebGPU is on
await q.route(EXP, (r) => r.fulfill({ status: 200, contentType: "text/html", body: html }));
await q.goto(EXP);
await q.waitForFunction(() => document.getElementById("ro").textContent.length > 0, null, { timeout: 60000 });
const ro = () => q.textContent("#ro"), kv = () => q.$eval("#k", (k) => [k.value, k.max]);
console.log("export opens:", await ro(), await kv());
await q.waitForTimeout(1500);
await q.screenshot({ path: path.join(outdir, "export_open.png") });
await q.$eval("#k", (k) => { k.value = "1"; k.dispatchEvent(new Event("input")); });
console.log("slider at 1:", await ro(), await kv());
await q.screenshot({ path: path.join(outdir, "export_k1.png") });
await q.click("#go");
await q.waitForTimeout(2500);
console.log("painting:", await ro(), await kv());
await q.waitForTimeout(5000);
console.log("after paint:", await ro(), await kv());
await q.screenshot({ path: path.join(outdir, "export_after_paint.png") });
const cb = await q.locator("canvas").boundingBox();
await q.mouse.move(cb.x + cb.width * 0.5, cb.y + cb.height * 0.45);
for (let i = 0; i < 3; i++) await q.mouse.wheel(0, -350);
await q.waitForTimeout(800);
console.log("export zoom:", await q.textContent("#zl"));
await q.screenshot({ path: path.join(outdir, "export_zoom.png") });
await b.close(); srv.close();
