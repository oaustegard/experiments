// node encode_test.mjs <backend webgpu|webgl> <W> <n> <stepsStage> <stepsFinal> <stepsQat> <out.json>
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const [backend, W, n, ss, sf, sq, out] = process.argv.slice(2);
const HERE = path.dirname(new URL(import.meta.url).pathname);
const files = {
  "/tf.min.js": path.join(HERE, "../scratch/tfjs/node_modules/@tensorflow/tfjs/dist/tf.min.js"),
  "/tf-backend-webgpu.min.js": path.join(HERE, "../scratch/tfjs/node_modules/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.min.js"),
  "/trainer.js": path.join(HERE, "../web/trainer.js"),
  "/img.png": path.join(HERE, "../data/kodak/kodim23.png"),
};
const PAGE = `<!doctype html><body><script src="/tf.min.js"></script><script src="/tf-backend-webgpu.min.js"></script>
<script type="module">
import { encode, emitWGSL, gzipSize } from "/trainer.js";
window.run = async (backend, W, n, ss, sf, sq) => {
  await tf.setBackend(backend); await tf.ready();
  const im = new Image(); im.src = "/img.png"; await im.decode();
  const H = Math.round(W * im.height / im.width);
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const x = c.getContext("2d"); x.drawImage(im, 0, 0, W, H);
  const rgba = x.getImageData(0, 0, W, H).data;
  const prog = [];
  const r = await encode(tf, { rgba, width: W, height: H }, { n, stepsStage: ss, stepsFinal: sf, stepsQat: sq,
    onProgress: (p) => prog.push(p) });
  const wgsl = emitWGSL(r);
  return { backend: tf.getBackend(), W, H, n, seconds: r.seconds, points: r.points, P: Array.from(r.P), bg: Array.from(r.bg),
           A: r.A, wgsl, gz: await gzipSize(wgsl), prog: prog.filter((_, i) => i % 5 === 0), rgba: Array.from(rgba) };
};
window.ready = true;
</script>`;
const srv = http.createServer((q, r) => { const u = q.url.split("?")[0];
  if (u === "/") { r.writeHead(200, { "Content-Type": "text/html" }); return r.end(PAGE); }
  const f = files[u]; if (!f) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "Content-Type": u.endsWith(".js") ? "text/javascript" : "image/png" }); r.end(fs.readFileSync(f)); });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const p = await b.newPage();
p.on("console", (m) => { const t = m.text(); if (!t.includes("High memory usage")) console.log("[page]", t.slice(0, 300)); });
p.on("pageerror", (e) => console.log("[err]", e.message));
await p.goto(`http://127.0.0.1:${srv.address().port}/`);
await p.waitForFunction(() => window.ready);
const t = Date.now();
const res = await p.evaluate(([a, b, c, d, e, f]) => window.run(a, +b, +c, +d, +e, +f), [backend, W, n, ss, sf, sq]);
console.log("wall", (Date.now() - t) / 1000, "s backend", res.backend, "points", JSON.stringify(res.points), "gz", res.gz);
fs.writeFileSync(out, JSON.stringify(res));
await b.close(); srv.close();
