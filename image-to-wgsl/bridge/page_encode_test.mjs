// node page_encode_test.mjs <image> <outdir>: drive web/index.html's "Your image" encoder headless.
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const [image, outdir] = process.argv.slice(2);
const HERE = path.dirname(new URL(import.meta.url).pathname), WEB = path.join(HERE, "../web");
const TF = { "tf.min.js": path.join(HERE, "../scratch/tfjs/node_modules/@tensorflow/tfjs/dist/tf.min.js"),
             "tf-backend-webgpu.min.js": path.join(HERE, "../scratch/tfjs/node_modules/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.min.js") };
const types = { ".js": "text/javascript", ".html": "text/html", ".jpg": "image/jpeg", ".avif": "image/avif", ".webp": "image/webp", ".png": "image/png" };
const srv = http.createServer((q, r) => { const u = q.url.split("?")[0], f = path.join(WEB, u === "/" ? "index.html" : u);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "Content-Type": types[path.extname(f)] || "application/octet-stream" }); r.end(fs.readFileSync(f)); });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const p = await b.newPage({ viewport: { width: 900, height: 1100 } });
await p.route("https://cdn.jsdelivr.net/**", (route) => {
  const name = route.request().url().split("/").pop();
  route.fulfill({ status: 200, contentType: "text/javascript", body: fs.readFileSync(TF[name]) });
});
p.on("console", (m) => { const t = m.text(); if (!/High memory|performance of synchronously/.test(t)) console.log("[page]", t.slice(0, 300)); });
p.on("pageerror", (e) => console.log("[err]", e.message));
await p.goto(`http://127.0.0.1:${srv.address().port}/`);
await p.waitForFunction(() => window.gpuDone, null, { timeout: 60000 });
await p.evaluate(() => { window.__encodeOpts = { stepsStage: 20, stepsFinal: 40, stepsQat: 20, previewEvery: 10 }; });
await p.click("button.own");
await p.setInputFiles("#file", image);
await p.waitForFunction(() => !document.getElementById("go").disabled);
await p.selectOption("#nsel", "64");
await p.click("#go");
await p.waitForFunction(() => !document.getElementById("prev").hidden, null, { timeout: 600000 });
await p.waitForTimeout(4000);
await p.screenshot({ path: path.join(outdir, "enc_during.png") });
console.log("during:", await p.textContent("#encstat"));
await p.waitForFunction(() => window.__encodeResult || /failed/.test(document.getElementById("encstat").textContent), null, { timeout: 1800000 });
console.log("after:", await p.textContent("#encstat"));
await p.waitForTimeout(12000);
await p.evaluate(() => window.gpuDone && window.gpuDone());
await p.screenshot({ path: path.join(outdir, "enc_after.png") });
const res = await p.evaluate(() => window.__encodeResult);
if (res) { fs.writeFileSync(path.join(outdir, "enc_page_result.json"), JSON.stringify(res)); console.log("points", JSON.stringify(res.points), "cmp", JSON.stringify(res.cmp)); }
console.log("buttons:", await p.$$eval("#views button", (bs) => bs.map((x) => x.textContent)).catch(() => []));
console.log("downloads visible:", await p.isVisible("#dl"));
await b.close(); srv.close();
