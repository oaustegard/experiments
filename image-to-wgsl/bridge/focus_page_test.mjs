// node focus_page_test.mjs <image> <outdir> [mode]: headless test of the page's focus control (face, point or
// depth), the focused WebGPU encode, and the in-focus PSNR it reports. SwiftShader: function only, not speed.
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const [image, outdir, mode = "face"] = process.argv.slice(2);
const HERE = path.dirname(new URL(import.meta.url).pathname), WEB = path.join(HERE, "../web");
const types = { ".js": "text/javascript", ".html": "text/html", ".jpg": "image/jpeg", ".avif": "image/avif", ".webp": "image/webp", ".png": "image/png" };
const srv = http.createServer((q, r) => { const u = q.url.split("?")[0], f = path.join(WEB, u === "/" ? "index.html" : u);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "Content-Type": types[path.extname(f)] || "application/octet-stream" }); r.end(fs.readFileSync(f)); });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const p = await b.newPage({ viewport: { width: 900, height: 1200 } });
p.on("console", (m) => { const t = m.text(); if (/focus|error|Error|encode/.test(t)) console.log("[page]", t.slice(0, 300)); });
p.on("pageerror", (e) => console.log("[err]", e.message));
await p.goto(`http://127.0.0.1:${srv.address().port}/`);
await p.waitForFunction(() => window.gpuDone, null, { timeout: 60000 });
await p.evaluate(() => { window.__encodeOpts = { stepsStage: 10, stepsFinal: 20, stepsQat: 10 }; });
await p.evaluate(() => document.querySelector("button.own").click());
await p.selectOption("#fsel", mode);
await p.setInputFiles("#file", image);
await p.waitForFunction(() => /Found|No face|Tap|failed/.test(document.getElementById("fstat").textContent), null, { timeout: 300000 });
console.log("focus status:", await p.textContent("#fstat"), "| tag:", await p.textContent("#tag"));
if (mode !== "face") {      // tap the upper left of the picture and wait for the map to follow
  const box = await p.locator("#stage").boundingBox();
  await p.mouse.click(box.x + box.width * 0.42, box.y + box.height * 0.25);
  await p.waitForTimeout(1500);
  console.log("after tap:", await p.textContent("#fstat"));
}
await p.locator("#stage").screenshot({ path: path.join(outdir, `focus_${mode}_map.png`) });
await p.waitForFunction(() => !document.getElementById("go").disabled, null, { timeout: 900000 });
await p.selectOption("#nsel", "64");
await p.evaluate(() => document.getElementById("go").click());
await p.waitForFunction(() => window.__encodeResult || /failed/.test(document.getElementById("encstat").textContent), null, { timeout: 1800000 });
console.log("after:", await p.textContent("#encstat"));
console.log("note:", await p.textContent("#cmpnote"));
const res = await p.evaluate(() => window.__encodeResult);
if (res) console.log("points", JSON.stringify(res.points), "focus", res.focus);
await p.waitForTimeout(7000); await p.evaluate(() => window.gpuDone());
console.log("readout:", await p.textContent("#readout"));
await p.locator("#stage").screenshot({ path: path.join(outdir, `focus_${mode}_after.png`) });
await b.close(); srv.close();
