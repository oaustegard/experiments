// Network trace of austegard.com's anything-to-text with Indra's math on Auto, in headless Chromium
// (SwiftShader WebGPU): which Indra files the page fetches for a page without math, then for a page
// with math. The page is served from a checkout of oaustegard.github.io; requests for the Indra
// module and pack on austegard.com are answered from this checkout's web/ and from PACK_DIR, so an
// unpublished build can be traced. Everything else (pdf.js, relay.js) loads from the network.
//
//   node web/test/a2t_math_trace.mjs SITE_DIR PACK_DIR TEXT_PAGE.png MATH_PAGE.png [OUT.json]
//
// Prints each page's Indra requests and the math lines read, and exits 1 unless the page without
// math fetched no math.bin and the page with math fetched it once.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [site, pack, textPng, mathPng, outJson] = process.argv.slice(2).map((p) => p && path.resolve(p));
const INDRA = "https://austegard.com/experiments/indra-browser/web/";
const type = (f) => (f.endsWith(".js") || f.endsWith(".mjs") ? "text/javascript" : f.endsWith(".json") ? "application/json"
  : f.endsWith(".html") ? "text/html" : f.endsWith(".css") ? "text/css" : f.endsWith(".png") ? "image/png" : "application/octet-stream");
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const f = path.join(site, u.endsWith("/") ? u + "index.html" : u);
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": type(f) });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, } : undefined;
const browser = await chromium.launch({ proxy, args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const page = await browser.newPage();
const log = [];
page.on("console", (m) => { const t = m.text(); log.push(t); if (/indra|status|math|error/i.test(t)) process.stderr.write(`${t.slice(0, 300)}\n`); });
page.on("pageerror", (e) => process.stderr.write(`pageerror: ${e.message}\n`));
// Indra's module and pack from this checkout
await page.route(INDRA + "**", (route) => {
  const rel = decodeURIComponent(new URL(route.request().url()).pathname).replace("/experiments/indra-browser/web/", "");
  const f = rel.startsWith("models/mix8/") ? path.join(pack, rel.slice("models/mix8/".length)) : path.join(WEB, rel);
  if (!fs.existsSync(f)) return route.fulfill({ status: 404, body: "" });
  return route.fulfill({ status: 200, contentType: type(f), body: fs.readFileSync(f), headers: { "access-control-allow-origin": "*" } });
});
let reqs = [];
page.on("request", (r) => { if (r.url().startsWith(INDRA)) reqs.push(r.url().slice(INDRA.length)); });
// the site from the checkout, at a made-up origin the proxy never sees
await page.route("https://a2t.local/**", (route) => {
  const u = decodeURIComponent(new URL(route.request().url()).pathname);
  const f = path.join(site, u);
  if (!fs.existsSync(f)) return route.fulfill({ status: 404, body: "" });
  return route.fulfill({ status: 200, contentType: type(f), body: fs.readFileSync(f) });
});
await page.goto("https://a2t.local/web-utilities/anything-to-text.html");
const mathMode = await page.$eval("#math", (e) => e.value);

async function readOne(png) {
  reqs = [];
  const before = log.length;
  await page.setInputFiles("#file", png);
  await page.waitForFunction(() => !document.querySelector("#run").disabled, null, { timeout: 120000 });
  await page.click("#run");
  await page.waitForFunction(() => /^Done\.|remaining/.test(document.querySelector("#status").textContent), null, { timeout: 3600000, polling: 2000 });
  const pageLog = log.slice(before).filter((t) => t.startsWith("[indra]"));
  const math = pageLog.map((t) => t.match(/math_lines: (\d+)[\s\S]*?math_used: (\d+)/) || t.match(/"math_lines":(\d+).*"math_used":(\d+)/)).find(Boolean);
  const md = await page.evaluate(() => document.querySelector("#pages")?.innerText || "");
  return { file: path.basename(png), indra_requests: reqs.slice(), math_bin: reqs.filter((u) => u.endsWith("math.bin")).length,
           mathcls_bin: reqs.filter((u) => u.endsWith("mathcls.bin")).length, latex_spans: (md.match(/\$[^$\n]+\$/g) || []).length,
           indra_log: pageLog.map((t) => t.slice(0, 400)) };
}

const text = await readOne(textPng);
await page.click("#clear");
const math = await readOne(mathPng);
const res = { math_mode: mathMode, text_page: text, math_page: math };
for (const r of [text, math]) {
  console.log(`${r.file}: Indra requests ${JSON.stringify(r.indra_requests)}`);
  console.log(`  math.bin fetched ${r.math_bin}x, mathcls.bin ${r.mathcls_bin}x, $...$ spans on the page ${r.latex_spans}`);
}
if (outJson) fs.writeFileSync(outJson, JSON.stringify(res, null, 1));
await browser.close();
server.close();
process.exit(text.math_bin === 0 && math.math_bin === 1 ? 0 : 1);
