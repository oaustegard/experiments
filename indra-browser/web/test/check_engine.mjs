// Parity of the browser engine (web/nets.js) with the NumPy reference (run/ref.py), in headless
// Chromium on SwiftShader WebGPU. Fixtures come from run/make_fixtures.py.
//
//   node web/test/check_engine.mjs PACK_DIR FIXTURE_DIR
//
// Exits 1 when a model is off by more than 1e-4 of its output range (or a CTC argmax differs).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [pack, fix] = process.argv.slice(2).map((p) => path.resolve(p));
const roots = { "/web/": WEB, "/pack/": pack, "/fix/": fix };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>engine</title>"); }
  const pre = Object.keys(roots).find((k) => u.startsWith(k));
  if (!pre) { res.writeHead(404); return res.end(); }
  const f = path.join(roots[pre], u.slice(pre.length));
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": f.endsWith(".js") ? "text/javascript" : f.endsWith(".json") ? "application/json" : "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`${m.text()}\n`));
page.on("pageerror", (e) => process.stderr.write(`pageerror: ${e.message}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
const out = await page.evaluate(async () => {
  const { Gpu } = await import("/web/gpu.js");
  const { ConvNet, Recognizer, OrderNet } = await import("/web/nets.js");
  const { loadModel } = await import("/web/pack.js");
  const gpu = await Gpu.create();
  gpu.device.addEventListener("uncapturederror", (e) => console.log("GPU error: " + e.error.message));
  const base = new URL("/pack/", location.href);
  const man = await (await fetch("/pack/manifest.json")).json();
  const cases = await (await fetch("/fix/cases.json")).json();
  const f32 = async (r) => new Float32Array(await (await fetch(`/fix/${r.file}`)).arrayBuffer());
  const diff = (a, b) => { let d = 0, m = 0; for (let i = 0; i < b.length; i++) { d = Math.max(d, Math.abs(a[i] - b[i])); m = Math.max(m, Math.abs(b[i])); } return { d, m }; };
  const res = [];
  const t0 = performance.now();
  // det
  {
    const c = cases.det, [H, W] = c.x.shape;
    const net = new ConvNet(gpu, await loadModel(base, man, "det"));
    const got = await net.det(await f32(c.x), H, W);
    res.push({ name: "det", ...diff(got.data, await f32(c.want)) });
  }
  {
    const c = cases.orient, [N, H, W] = c.x.shape;
    const net = new ConvNet(gpu, await loadModel(base, man, "orient"));
    const got = (await net.classify(await f32(c.x), N, H, W)).flat();
    res.push({ name: "orient", ...diff(got, await f32(c.want)) });
  }
  {
    const c = cases.script, [N, H, W] = c.x.shape;
    const net = new ConvNet(gpu, await loadModel(base, man, "script"));
    const got = (await net.classify(await f32(c.x), N, H, W, c.widths)).flat();
    res.push({ name: "script", ...diff(got, await f32(c.want)) });
  }
  for (const name of ["rec", "rec_world"]) {
    const c = cases[name], [B, H, W] = c.x.shape;
    const net = new Recognizer(gpu, await loadModel(base, man, name));
    const got = await net.run(await f32(c.x), B, H, W, c.widths);
    const want = await f32(c.want);
    const T = got.T;
    let d = 0, m = 0, idsEqual = true;
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < got.lengths[b]; t++) {
        const o = (b * T + t) * 19;
        if (got.out[o] !== want[o]) idsEqual = false;
        for (const j of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) { d = Math.max(d, Math.abs(got.out[o + j] - want[o + j])); m = Math.max(m, Math.abs(want[o + j])); }
      }
    }
    res.push({ name, d, m, idsEqual, lengths: got.lengths.join(",") === c.lengths.join(",") });
  }
  {
    const c = cases.order, [N] = c.x.shape;
    const net = new OrderNet(await loadModel(base, man, "order"));
    const got = net.run(await f32(c.x), N);
    for (const k of ["a", "cls", "same", "tab"]) res.push({ name: `order.${k}`, ...diff(got[k], await f32(c[k])) });
  }
  return { device: gpu.label, ms: performance.now() - t0, res };
});
await browser.close();
server.close();
let bad = 0;
for (const r of out.res) {
  const ok = r.d <= 1e-4 * Math.max(1, r.m) && r.idsEqual !== false && r.lengths !== false;
  bad += !ok;
  console.log(`${ok ? "ok  " : "FAIL"} ${r.name.padEnd(12)} max|diff| ${r.d.toExponential(2)} (max|ref| ${r.m.toExponential(2)})${r.idsEqual === undefined ? "" : ` ids ${r.idsEqual} lengths ${r.lengths}`}`);
}
console.log(`${out.res.length - bad}/${out.res.length} on ${out.device}, ${out.ms.toFixed(0)} ms`);
process.exit(bad ? 1 : 0);
