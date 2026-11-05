// Runs a web/*.js check in headless Chromium on SwiftShader WebGPU and prints its JSON result.
//
//   node run/headless.mjs gradcheck [out.json]
//
// Serves the experiment directory over http://127.0.0.1 (WebGPU needs a real origin), with model/ mapped
// to bsky-account-decider/web/models/account/. SwiftShader is a CPU rasterizer: correct, not fast.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");

const HERE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = path.resolve(HERE, "../bsky-account-decider/web/models/account");
// which: "file" (calls export of the same name) or "file:export"; opts: JSON passed as the third argument
const [which = "gradcheck", out, opts = "{}"] = process.argv.slice(2);
const [file, fn = file] = which.split(":");
const TYPES = { ".js": "text/javascript", ".html": "text/html", ".json": "application/json" };

const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const f = u.startsWith("/model/") ? path.join(MODEL, u.slice(7)) : path.join(HERE, u);
  if (!f.startsWith(HERE) && !f.startsWith(MODEL)) { res.writeHead(403); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": TYPES[path.extname(f)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`[page] ${m.text()}\n`));
await page.goto(`http://127.0.0.1:${port}/web/blank.html`);
const result = await page.evaluate(async ([file, fn, opts]) => {
  const adapter = await navigator.gpu.requestAdapter();
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize,
    maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage } });
  device.addEventListener("uncapturederror", (e) => console.log("GPU error: " + e.error.message));
  const get = async (p) => { const r = await fetch("/" + p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
  const mod = await import(`/web/${file}.js`);
  const o = JSON.parse(opts);
  o.log = (m) => console.log(m);
  const plain = (v) => ArrayBuffer.isView(v) ? Array.from(v) : Array.isArray(v) ? v.map(plain) : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, plain(x)])) : v;
  try { return plain(await mod[fn](device, get, o)); } catch (e) { return { error: String(e.stack || e) }; }
}, [file, fn, opts]);
await browser.close();
server.close();
const s = JSON.stringify(result, null, 1);
if (out) fs.writeFileSync(out, s);
console.log(s);
