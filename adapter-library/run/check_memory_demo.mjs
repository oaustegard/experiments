// The everyday memory demo in headless Chromium: teach four Amazon charges as "household", answer a fifth.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import http from "node:http"; import fs from "node:fs"; import path from "node:path";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "web");
const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" };
const server = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(new URL(q.url, "http://x").pathname));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); return; } r.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" }); r.end(d); }); }).listen(0);
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const p = await b.newPage();
await p.goto(`http://localhost:${server.address().port}/check.html?set=everyday`);
await p.waitForFunction(() => window.__result, null, { timeout: 900000 });
const out = await p.evaluate(async () => {
  const { loadSystem } = await import("./system.js");
  const sys = await loadSystem("models/everyday/");
  sys.clear();
  const s = await fetch("models/everyday/samples.json").then((r) => r.json());
  const amzn = s.cases.filter((c) => c.state.includes("AMZN MKTP"));
  const steps = [];
  for (let i = 0; i < amzn.length; i++) {
    const a = await sys.answer("transactions", amzn[i].state);
    const q = a.questions.category, o = q.options.map(([x]) => x);
    steps.push({ case: i, labelsBefore: q.labels, spike: o[q.spike.indexOf(Math.max(...q.spike))], blend: q.label,
                 pHouseholdSpike: +q.spike[o.indexOf("household")].toFixed(3), pHouseholdBlend: +q.blend[o.indexOf("household")].toFixed(3), lambda: +q.lambda.toFixed(2) });
    if (i < 5) sys.teach("transactions", a, { category: "household" }, amzn[i].id);
  }
  return steps;
});
console.log(JSON.stringify(out, null, 1));
await b.close(); server.close();
