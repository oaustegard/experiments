// Parity of the browser's math reader (web/math.js, and mathcls through nets.js) with ONNX Runtime
// on real line crops, in headless Chromium on SwiftShader WebGPU. Fixtures come from
// run/make_math_fixtures.py (the same weights as the pack: run it on the pack's arm).
//
//   node web/test/check_math.mjs PACK_DIR FIXTURE_DIR
//
// Reports, per output, the largest absolute difference and the output's largest magnitude, and
// whether every greedy token of the fixture's 24 steps matches. Exits 1 on a token mismatch or a
// difference over 1e-3 of an output's range.
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
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>math</title>"); }
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
  const { ConvNet } = await import("/web/nets.js");
  const { MathNet } = await import("/web/math.js");
  const { loadModel } = await import("/web/pack.js");
  const gpu = await Gpu.create();
  gpu.device.addEventListener("uncapturederror", (e) => console.log("GPU error: " + e.error.message));
  const base = new URL("/pack/", location.href);
  const man = await (await fetch("/pack/manifest.json")).json();
  const mman = { models: man.math.models };
  const cases = await (await fetch("/fix/cases.json")).json();
  const f32 = async (r) => new Float32Array(await (await fetch(`/fix/${r.file}`)).arrayBuffer());
  const diff = (a, b) => { let d = 0, m = 0; for (let i = 0; i < b.length; i++) { d = Math.max(d, Math.abs(a[i] - b[i])); m = Math.max(m, Math.abs(b[i])); } return { d, m }; };
  const res = [];
  {
    const c = cases.cls, [N, H, W] = c.x.shape;
    const net = new ConvNet(gpu, await loadModel(base, mman, "mathcls"));
    const got = (await net.classify(await f32(c.x), N, H, W, c.widths)).flat();
    res.push({ name: "mathcls", ...diff(got, await f32(c.want)), got: got.map((v) => +v.toFixed(5)) });
  }
  const c = cases.math, [B, H, W] = c.x.shape;
  const t0 = performance.now();
  const net = new MathNet(gpu, await loadModel(base, mman, "math"));
  const load = performance.now() - t0;
  const r = await net.probe({ data: await f32(c.x), B, H, W, widths: c.widths }, 24, [0, 1, 23]);
  const T = r.T;
  // ORT cross_k / cross_v [3, B, 8, T, 24] -> ours [B, T, 384] (k | v, head h at h * 24)
  const ck = await f32(c.cross_k), cv = await f32(c.cross_v);
  for (let l = 0; l < 3; l++) {
    const want = new Float32Array(B * T * 384);
    for (let b = 0; b < B; b++)
      for (let h = 0; h < 8; h++)
        for (let t = 0; t < T; t++)
          for (let d = 0; d < 24; d++) {
            const src = ((((l * B + b) * 8 + h) * T) + t) * 24 + d;
            want[(b * T + t) * 384 + h * 24 + d] = ck[src];
            want[(b * T + t) * 384 + 192 + h * 24 + d] = cv[src];
          }
    res.push({ name: `cross_kv.${l}`, ...diff(r.kv[l], want) });
  }
  res.push({ name: "mask", ...diff(r.mask, await f32(c.mask)) });
  for (const [s, k] of [[0, "logits0"], [1, "logits1"], [23, "logits23"]]) res.push({ name: `logits.${s}`, ...diff(r.logits[s], await f32(c[k])) });
  const wantTop = await f32(c.top);
  res.push({ name: "top", ...diff(r.tops.flat(), wantTop) });
  const sameTok = JSON.stringify(r.toks) === JSON.stringify(c.tokens);
  let firstDiff = null;
  for (let s = 0; s < c.tokens.length && !firstDiff; s++) for (let b = 0; b < B; b++) if (r.toks[s][b] !== c.tokens[s][b]) { firstDiff = { step: s, line: b, got: r.toks[s][b], want: c.tokens[s][b] }; break; }
  return { res, sameTok, firstDiff, B, T, load_ms: load };
});
let bad = !out.sameTok;
for (const r of out.res) {
  const rel = r.d / Math.max(r.m, 1e-9);
  if (rel > 1e-3) bad = true;
  console.log(`${r.name.padEnd(12)} max|diff| ${r.d.toExponential(2)}  max|want| ${r.m.toFixed(3)}  rel ${rel.toExponential(2)}${r.got ? "  " + JSON.stringify(r.got) : ""}`);
}
console.log(`tokens (24 steps x ${out.B} lines, T ${out.T}): ${out.sameTok ? "all equal" : "DIFFER " + JSON.stringify(out.firstDiff)}; load ${out.load_ms.toFixed(0)} ms`);
await browser.close();
server.close();
process.exit(bad ? 1 : 0);
