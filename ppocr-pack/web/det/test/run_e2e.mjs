// End-to-end smoke test in headless Chromium (SwiftShader WebGPU): detector, pipeline.js and recognizer on a
// fixture page. Needs the det and rec packs (run/export_det.py, run/export_rec.py) and the fixtures
// (run/ref_det.py). Prints the lines and exits 0 when every expected string of the page is read.
//
//   taskset -c 0,1 node web/det/test/run_e2e.mjs [tiny|small] [page]      # default tiny heading_table
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.execPath, "../../lib/node_modules/playwright"))); }

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(HERE, "../..");
const ROOT = path.resolve(WEB, "..");
const FIX = path.join(HERE, "data/fixtures");
const args = process.argv.slice(2);
const model = args.find((a) => a === "tiny" || a === "small") || "tiny";
const pageName = args.find((a) => a !== "tiny" && a !== "small") || "heading_table";
// strings the synthetic pages contain (web/det/test/make_pages.py); a line may hold more than the string
const EXPECT = {
  heading_table: ["Quarterly results", "Summary of the regional figures reported in March.", "Region", "Units", "Revenue", "Change", "North", "Central"],
  dense_small: ["INVOICE 2026-0117", "page 1 of 1", "payment within 30 days"],
};

const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".txt": "text/plain; charset=utf-8" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>e2e</title>"); }
  const f = u.startsWith("/web/") ? path.join(WEB, u.slice(5)) : u.startsWith("/fix/") ? path.join(FIX, u.slice(5)) : null;
  if (!f || !(f.startsWith(WEB) || f.startsWith(FIX))) { res.writeHead(404); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const flags = ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader",
  "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"];
const exe = fs.existsSync("/opt/pw-browsers/chromium") ? "/opt/pw-browsers/chromium" : undefined;
const browser = await chromium.launch({ executablePath: exe, args: flags });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`${m.text()}\n`));
page.on("pageerror", (e) => process.stderr.write(`pageerror: ${e.message}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
let out;
try {
  out = await page.evaluate(async (a) => (await import("/web/det/test/check_e2e.js")).default(a), { model, page: pageName });
} catch (e) {
  out = { error: String((e && e.message) || e) };
}
await browser.close();
server.close();
if (out.error) { console.log(JSON.stringify(out)); process.exit(1); }
for (const l of out.lines) console.log(`${l.score.toFixed(2)} det ${l.detScore.toFixed(2)}  ${JSON.stringify(l.box[0])}  ${l.text}`);
const text = out.lines.map((l) => l.text).join("\n").toLowerCase();
const want = EXPECT[pageName] || [];
const missing = want.filter((s) => !text.includes(s.toLowerCase()));
console.log(JSON.stringify({ model, page: pageName, lines: out.lines.length, loadMs: out.loadMs, timing: out.timing, missing }));
const res = path.join(ROOT, "results/pipeline_e2e.json");
fs.mkdirSync(path.dirname(res), { recursive: true });
const all = fs.existsSync(res) ? JSON.parse(fs.readFileSync(res, "utf8")) : {};
all[`${model}/${pageName}`] = { adapter: out.adapter, lines: out.lines.length, timing_swiftshader_ms: out.timing, missing_expected: missing,
  text: out.lines.map((l) => l.text) };
fs.writeFileSync(res, JSON.stringify(all, null, 1) + "\n");
process.exit(missing.length ? 1 : 0);
