// node bridge/run_local.mjs JOB.js INPUTS.json OUT.json: run a GPU Bridge job in headless Chromium (SwiftShader),
// for correctness only (never for timing). INPUTS: {name: {"b64": "..."} (u8) | any JSON}. Typed arrays in the result
// are written as {"u8": [...]} and so on; ctx.log lines go to stdout.
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright"); const fs = require("fs"), http = require("http");
const [jobp, inp, outp] = process.argv.slice(2);
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const p = await b.newPage(); p.on("console", (m) => console.log(m.text())); p.on("pageerror", (e) => console.log("[err]", e.message));
const srv = http.createServer((q, r) => { r.writeHead(200, { "Content-Type": "text/html" }); r.end("<!doctype html><body>"); });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
await p.goto(`http://127.0.0.1:${srv.address().port}/`);   // WebGPU needs a secure context
const res = await p.evaluate(async ({ body, inputs }) => {
  const ins = {};
  for (const [k, v] of Object.entries(inputs)) ins[k] = v && v.b64 != null ? Uint8Array.from(atob(v.b64), (c) => c.charCodeAt(0)) : v;
  const ad = await navigator.gpu.requestAdapter();
  const dev = await ad.requestDevice({ requiredFeatures: ad.features.has("float32-blendable") ? ["float32-blendable"] : [],
    requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
  const ctx = { inputs: ins, gpu: async () => dev, log: (...a) => console.log("[job]", ...a), progress: () => {}, cache: new Map() };
  const r = await (new Function("ctx", `return (async () => {${body}\n})();`))(ctx);
  return JSON.parse(JSON.stringify(r, (k, v) => ArrayBuffer.isView(v) ? { [v.constructor.name]: Array.from(v) } : v));
}, { body: fs.readFileSync(jobp, "utf8"), inputs: JSON.parse(fs.readFileSync(inp, "utf8")) });
fs.writeFileSync(outp, JSON.stringify(res));
await b.close(); srv.close();
