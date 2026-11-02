// node instanced_test.mjs in.json out.json: render splat lists with web/instanced.js in headless Chromium,
// at each requested prefix, to rgba8unorm, and write the pixels (base64) for comparison with codec.py.
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
const jobs = JSON.parse(fs.readFileSync(inp));
const res = await p.evaluate(async (jobs) => {
  const { emitInstanced, createRenderer } = await import("/instanced.js");
  const ad = await navigator.gpu.requestAdapter();
  const dev = await ad.requestDevice({ requiredFeatures: ad.features.has("float32-blendable") ? ["float32-blendable"] : [] });
  dev.addEventListener("uncapturederror", (e) => console.log("gpu error: " + e.error.message));
  const out = {};
  for (const [name, j] of Object.entries(jobs)) {
    const r = { P: Int32Array.from(j.P), bg: Int32Array.from(j.bg), A: j.A, n: j.n };
    const code = emitInstanced(r);
    const mod = dev.createShaderModule({ code }), info = await mod.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error").map((m) => m.lineNum + ":" + m.linePos + " " + m.message);
    if (errs.length) { out[name] = { errors: errs }; continue; }
    const rd = createRenderer(dev, "rgba8unorm", code);
    out[name] = { codeLen: code.length, format: rd.format, imgs: {} };
    for (const n of j.prefixes) {
      const W = j.W, H = j.H, tex = dev.createTexture({ size: [W, H], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      rd.draw(tex.createView(), W, H, n);
      const bpr = Math.ceil(W * 4 / 256) * 256, buf = dev.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = dev.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [W, H]); dev.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const src = new Uint8Array(buf.getMappedRange()), px = new Uint8Array(W * H * 4);
      for (let y = 0; y < H; y++) px.set(src.subarray(y * bpr, y * bpr + W * 4), y * W * 4);
      let s = ""; for (let i = 0; i < px.length; i += 0x8000) s += String.fromCharCode.apply(null, px.subarray(i, i + 0x8000));
      out[name].imgs[n] = btoa(s); buf.unmap();
    }
  }
  return out;
}, Object.fromEntries(Object.entries(jobs).map(([k, v]) => [k, { ...v, W: v.W || 384, H: v.H || 256, prefixes: v.prefixes || [16, v.n] }])));
fs.writeFileSync(outp, JSON.stringify(res));
await b.close(); srv.close();
