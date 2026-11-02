// node focus_test.mjs <image> <outdir>: run web/focus.js (point, face, depth) on an image in headless Chromium
// and save each focus map as a greyscale PNG. Needs network for the face and depth models.
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const http = require("http"), fs = require("fs"), path = require("path");
const [image, outdir] = process.argv.slice(2);
const WEB = path.join(path.dirname(new URL(import.meta.url).pathname), "../web");
const srv = http.createServer((q, r) => { const u = q.url.split("?")[0];
  if (u === "/") { r.writeHead(200, { "Content-Type": "text/html" }); return r.end("<!doctype html><body>"); }
  if (u === "/img") { r.writeHead(200); return r.end(fs.readFileSync(image)); }
  const f = path.join(WEB, u); if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { "Content-Type": "text/javascript" }); r.end(fs.readFileSync(f)); });
await new Promise((r) => srv.listen(0, "127.0.0.1", r));
const b = await chromium.launch();
const p = await b.newPage(); p.on("console", (m) => console.log("[page]", m.text().slice(0, 300))); p.on("pageerror", (e) => console.log("[err]", e.message));
await p.goto(`http://127.0.0.1:${srv.address().port}/`);
const res = await p.evaluate(async () => {
  const f = await import("/focus.js");
  const bmp = await createImageBitmap(await (await fetch("/img")).blob());
  const s = Math.min(1, 512 / Math.max(bmp.width, bmp.height)), W = Math.round(bmp.width * s), H = Math.round(bmp.height * s);
  const c = new OffscreenCanvas(W, H), x = c.getContext("2d"); x.drawImage(bmp, 0, 0, W, H);
  const img = { rgba: x.getImageData(0, 0, W, H).data, width: W, height: H }, out = { W, H };
  const enc = (m) => { let s = ""; const u = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) u[i] = Math.round(255 * Math.min(1, m[i]));
    for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
  let t = performance.now();
  out.point = enc(await f.focusMap("point", img, { point: { x: 0.3, y: 0.3 }, radius: 0.15 }));
  try { out.faces = await f.detectFaces(img); out.faceMs = performance.now() - t; t = performance.now();
        const m = await f.focusMap("face", img); out.face = m ? enc(m) : null; } catch (e) { out.faceErr = String(e?.stack || e); }
  t = performance.now();
  try { const d = await f.depthMap(img); out.depthMs = performance.now() - t; out.depth = enc(d);
        out.depthNear = enc(await f.focusMap("depth", img, { depth: 1, width: 0.15 })); } catch (e) { out.depthErr = String(e?.stack || e); }
  return out;
});
for (const k of ["point", "face", "depth", "depthNear"]) if (res[k]) { fs.writeFileSync(path.join(outdir, `${k}.gray`), Buffer.from(res[k], "base64")); }
console.log(JSON.stringify({ W: res.W, H: res.H, faces: res.faces, faceMs: res.faceMs, depthMs: res.depthMs, faceErr: res.faceErr, depthErr: res.depthErr }, null, 1));
await b.close(); srv.close();
