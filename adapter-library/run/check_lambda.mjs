// Seed one spike's memory in headless Chromium and report each question's lambda on a sample case.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import http from "node:http"; import fs from "node:fs"; import path from "node:path";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "web");
const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json" };
const server = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(new URL(q.url, "http://x").pathname));
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); return; } r.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" }); r.end(d); }); }).listen(0);
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const p = await b.newPage();
await p.goto(`http://localhost:${server.address().port}/check.html`);
await p.waitForFunction(() => window.__result, null, { timeout: 900000 });
const out = await p.evaluate(async () => {
  const { loadSystem } = await import("./system.js");
  const sys = await loadSystem("models/");
  const s = await fetch("models/samples.json").then((r) => r.json());
  const id = "invoice_processing";
  await sys.seed(id, s.seed[id].slice(0, 8));
  const c = s.cases.find((x) => x.workflow === id);
  const a = await sys.answer(id, c.state);
  return Object.fromEntries(Object.entries(a.questions).map(([q, v]) => [q, { lambda: v.lambda, labels: v.labels }]));
});
console.log(JSON.stringify(out));
await b.close(); server.close();
