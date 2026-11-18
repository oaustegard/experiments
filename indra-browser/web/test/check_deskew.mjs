// GPU deskew (web/deskew.js) against the CPU port (imgproc.warpAffine) on prerendered pages:
// the angle each finds, every trial angle's row-sharpness score, and the rotated page's pixels.
//   node web/test/check_deskew.mjs GRAY_DIR PAGES.txt
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [gray, list] = process.argv.slice(2).map((p) => path.resolve(p));
const pages = fs.readFileSync(list, "utf8").split(/\s+/).filter(Boolean);
const roots = { "/web/": WEB, "/gray/": gray };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>d</title>"); }
  const pre = Object.keys(roots).find((k) => u.startsWith(k));
  if (!pre) { res.writeHead(404); return res.end(); }
  fs.readFile(path.join(roots[pre], u.slice(pre.length)), (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": u.endsWith(".js") ? "text/javascript" : u.endsWith(".png") ? "image/png" : "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`${m.text()}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
const out = await page.evaluate(async (pages) => {
  const { Gpu } = await import("/web/gpu.js");
  const { Deskew } = await import("/web/deskew.js");
  const { estimateSkew, preview, rotatePage } = await import("/web/indra.js");
  const gpu = await Gpu.create();
  const dk = new Deskew(gpu);
  const res = [];
  for (const id of pages) {
    const bm = await createImageBitmap(await (await fetch(`/gray/${id}.png`)).blob());
    const c = new OffscreenCanvas(bm.width, bm.height), ctx = c.getContext("2d");
    ctx.drawImage(bm, 0, 0);
    const rgba = ctx.getImageData(0, 0, bm.width, bm.height).data;
    const g = new Uint8Array(bm.width * bm.height);
    for (let i = 0; i < g.length; i++) g[i] = rgba[4 * i];
    const small = preview({ w: bm.width, h: bm.height, data: g });
    let cs = [], gs = [];
    const t0 = performance.now();
    const aCpu = await estimateSkew(small, 800, async (ink, angles) => { const v = angles.map((a) => { const r = []; return a; }); const { warpAffine, rotationMatrix } = await import("/web/imgproc.js"); const s = angles.map((a) => { const q = warpAffine(ink, rotationMatrix(ink.w / 2, ink.h / 2, a, 1), ink.w, ink.h, 0); let prev = null, t = 0; for (let y = 0; y < q.h; y++) { let p = 0; for (let x = 0; x < q.w; x++) p += q.data[y * q.w + x]; if (prev !== null) t += (p - prev) ** 2; prev = p; } return t; }); cs.push(...s); return s; });
    const t1 = performance.now();
    const aGpu = await estimateSkew(small, 800, async (ink, angles) => { const s = await dk.sharpness(ink, angles); gs.push(...s); return s; });
    const t2 = performance.now();
    let rel = 0;
    for (let i = 0; i < Math.min(cs.length, gs.length); i++) rel = Math.max(rel, Math.abs(cs[i] - gs[i]) / Math.max(1, cs[i]));
    const full = { w: bm.width, h: bm.height, data: g };
    const deg = aCpu || 0.7;
    const r1 = rotatePage(full, deg), r2 = await dk.rotate(full, deg);
    let n = 0, mx = 0;
    for (let i = 0; i < r1.data.length; i++) { const d = Math.abs(r1.data[i] - r2.data[i]); if (d) n++; mx = Math.max(mx, d); }
    res.push({ id, aCpu, aGpu, scores: cs.length === gs.length, maxRelScoreDiff: rel, rotateDeg: deg, pixelsDiffering: n, of: r1.data.length, maxPixelDiff: mx, cpuMs: Math.round(t1 - t0), gpuMs: Math.round(t2 - t1) });
  }
  return { device: gpu.label, res };
}, pages);
await browser.close(); server.close();
console.log(JSON.stringify(out, null, 1));
