// node gpu_gradcheck.mjs in.json out.json: one forward+backward of web/gputrain.js in headless Chromium.
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright"); const http = require("http"), fs = require("fs"), path = require("path");
const [inp, outp] = process.argv.slice(2);
const WEB = path.join(path.dirname(new URL(import.meta.url).pathname), "../web");
const srv = http.createServer((q, r) => { const u = q.url.split("?")[0];
  if (u === "/") { r.writeHead(200, { "Content-Type": "text/html" }); return r.end("<!doctype html><body>"); }
  const f = path.join(WEB, u); if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "Content-Type": "text/javascript" }); r.end(fs.readFileSync(f)); });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const p = await b.newPage(); p.on("console", (m) => console.log("[page]", m.text())); p.on("pageerror", (e) => console.log("[err]", e.message));
await p.goto(`http://127.0.0.1:${srv.address().port}/`);
const spec = JSON.parse(fs.readFileSync(inp));
const res = await p.evaluate(async (s) => {
  const { lossAndGrad } = await import("/gputrain.js");
  const dev = await (await navigator.gpu.requestAdapter()).requestDevice();
  dev.addEventListener("uncapturederror", (e) => console.log("gpu error: " + e.error.message));
  const img = { width: s.W, height: s.H, rgba: Uint8Array.from(s.rgba) };
  return await lossAndGrad(dev, img, s.P, Int32Array.from(s.bg), { cut: 1e9, focus: s.focus ? Float32Array.from(s.focus) : null, bgImg: s.bgimg ? Float32Array.from(s.bgimg) : null });
}, spec);
fs.writeFileSync(outp, JSON.stringify(res));
await b.close(); srv.close();
