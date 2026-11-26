// End to end in headless Chromium (SwiftShader WebGPU unless --gpu): gpu/ocria-gpu.js on whole pages, against
// page/ocr.js on onnxruntime-node for the same pages (page/test/out/pages/<id>.json, from page/test/run.mjs).
// Reports, per page, the share of onnxruntime line texts the WebGPU path reproduces verbatim, and the reverse.
//
//   node web/gpu/test/check_page.mjs ID [ID ...]    (pages from /home/user/opb/gray600)
import fs from "node:fs";
import http from "node:http";
import path from "node:path";

import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
for (const p of ["playwright", "/opt/node-tools/node_modules/playwright", path.join(process.execPath, "../../lib/node_modules/playwright")]) {
  try { ({ chromium } = require(p)); break; } catch { /* next */ }
}
const here = path.dirname(new URL(import.meta.url).pathname);
const web = path.resolve(here, "..", "..");
const gray = process.env.GRAY_DIR || "/home/user/opb/gray600";
const ids = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split("?")[0]);
  const p = u.startsWith("/gray/") ? path.join(gray, u.slice(6)) : path.join(web, u === "/" ? "gpu/test/blank.html" : u);
  if (!fs.existsSync(p)) { res.writeHead(404); res.end(); return; }
  const type = p.endsWith(".js") || p.endsWith(".mjs") ? "text/javascript" : p.endsWith(".json") ? "application/json"
    : p.endsWith(".html") ? "text/html" : p.endsWith(".png") ? "image/png" : "application/octet-stream";
  res.writeHead(200, { "content-type": type });
  res.end(fs.readFileSync(p));
});
fs.writeFileSync(path.join(here, "blank.html"), "<!doctype html><meta charset=utf-8><title>t</title>");
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const flags = process.argv.includes("--gpu") ? ["--no-sandbox", "--enable-unsafe-webgpu"]
  : ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader",
     "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"];
const browser = await chromium.launch({ args: flags });
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("pageerror:", e.message));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
const out = await page.evaluate(async (ids) => {
  const { loadOcriaGpu } = await import("/gpu/ocria-gpu.js");
  const ocr = await loadOcriaGpu({ scripts: ["world", "rtl", "cjk"] });
  const rows = [];
  for (const id of ids) {
    const img = new Image();
    img.src = `/gray/${id}.png`;
    await img.decode();
    const cv = document.createElement("canvas");
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    cv.getContext("2d").drawImage(img, 0, 0);
    const t = performance.now();
    const lines = await ocr.read(cv);
    rows.push({ id, ms: Math.round(performance.now() - t), texts: lines.map((l) => l.text) });
  }
  return { loadMs: Math.round(ocr.loadMs), rows };
}, ids);
await browser.close();
server.close();
let hitA = 0, nA = 0, hitB = 0, nB = 0;
for (const r of out.rows) {
  const ref = JSON.parse(fs.readFileSync(path.join(web, "page/test/out/pages", r.id + ".json"), "utf8")).results.map((x) => x.text);
  const mine = new Set(r.texts), theirs = new Set(ref);
  const a = ref.filter((t) => mine.has(t)).length, b = r.texts.filter((t) => theirs.has(t)).length;
  hitA += a; nA += ref.length; hitB += b; nB += r.texts.length;
  console.log(`${r.id}: ${r.ms} ms, ${r.texts.length} lines; onnxruntime lines reproduced ${a}/${ref.length}, ours found there ${b}/${r.texts.length}`);
}
console.log(`SUMMARY load ${out.loadMs} ms; onnxruntime lines reproduced ${(100 * hitA / nA).toFixed(2)}% (${hitA}/${nA}), ours found ${(100 * hitB / nB).toFixed(2)}%`);
