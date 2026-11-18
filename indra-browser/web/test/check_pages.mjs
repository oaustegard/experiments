// The browser pipeline (web/indra.js) on prerendered pages in headless Chromium (SwiftShader
// WebGPU); writes one JSON per page (lines, markdown, blocks, timings) for compare_pages.py.
//
//   node web/test/check_pages.mjs PACK_DIR GRAY_DIR PAGES.txt OUT_DIR
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [pack, gray, list, outDir] = process.argv.slice(2).map((p) => path.resolve(p));
const pages = fs.readFileSync(list, "utf8").split(/\s+/).filter(Boolean);
fs.mkdirSync(outDir, { recursive: true });
const roots = { "/web/": WEB, "/pack/": pack, "/gray/": gray };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>pages</title>"); }
  const pre = Object.keys(roots).find((k) => u.startsWith(k));
  if (!pre) { res.writeHead(404); return res.end(); }
  fs.readFile(path.join(roots[pre], u.slice(pre.length)), (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    const t = u.endsWith(".js") ? "text/javascript" : u.endsWith(".json") ? "application/json" : u.endsWith(".png") ? "image/png" : "application/octet-stream";
    res.writeHead(200, { "content-type": t });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`${m.text()}\n`));
page.on("pageerror", (e) => process.stderr.write(`pageerror: ${e.message}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
await page.evaluate(async () => {
  const { Indra } = await import("/web/indra.js");
  window.ocr = await Indra.load("/pack/", { log: (m) => console.log(m) });
  window.ocr.gpu.device.addEventListener("uncapturederror", (e) => console.log("GPU error: " + e.error.message));
});
let done = 0;
for (const id of pages) {
  const dst = path.join(outDir, `${id}.json`);
  if (fs.existsSync(dst)) { done++; continue; }
  const r = await page.evaluate(async (id) => {
    const bm = await createImageBitmap(await (await fetch(`/gray/${id}.png`)).blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    const c = new OffscreenCanvas(bm.width, bm.height);
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bm, 0, 0);
    const rgba = ctx.getImageData(0, 0, bm.width, bm.height).data;
    const g = new Uint8Array(bm.width * bm.height);
    for (let i = 0; i < g.length; i++) g[i] = rgba[4 * i];
    const t0 = performance.now();
    try {
      const p = await window.ocr.read({ w: bm.width, h: bm.height, data: g });
      return { ...p, ms: performance.now() - t0 };
    } catch (e) {
      return { error: String(e && e.stack || e) };
    }
  }, id);
  fs.writeFileSync(dst, JSON.stringify({ page_id: id, ...r }));
  done++;
  process.stderr.write(`${done}/${pages.length} ${id} ${r.error ? "ERROR " + r.error.slice(0, 300) : (r.ms | 0) + " ms"}\n`);
}
await browser.close();
server.close();
