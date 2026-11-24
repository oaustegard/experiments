// Headless parity run for the recognizer. Serves ppocr-pack/web/ at /web/, runs web/rec/test/check_rec.js
// in Chromium (SwiftShader WebGPU unless --gpu), merges the WGSL numbers into results/rec_parity.json and
// exits 0 when every packed line is within 1e-3 max abs of the NumPy reference with the same greedy text
// (the batch too). The f32 variant against onnxruntime is reported, not gated.
//
//   python3 -I web/rec/test/make_lines.py && python3 -I run/export_rec.py && python3 -I run/export_rec.py --scheme f32
//   python3 -I run/ref_rec.py
//   taskset -c 0,1 node web/rec/test/run_rec.mjs [tiny] [small] [--no-f32] [--lines l00_en,l05_en] [--pack P] [--gpu]
//
// --pack P checks web/rec/models/<m>-P against the references ref_rec.py --pack P wrote.
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
const FIX = path.join(HERE, "fixtures");
const RESULTS = path.join(ROOT, "results/rec_parity.json");
const TOL = 1e-3;
const args = process.argv.slice(2);
const useGpu = args.includes("--gpu");
let models = args.filter((a) => ["tiny", "small"].includes(a));
if (!models.length) models = ["tiny", "small"];
const variants = args.includes("--no-f32") ? ["packed"] : ["packed", "f32"];
const li = args.indexOf("--lines");
const pi = args.indexOf("--pack");   // packed variant web/rec/models/<m>-<pack> (default <m>), as ref_rec.py --pack
const pack = pi >= 0 ? args[pi + 1] : "";
const index = JSON.parse(fs.readFileSync(path.join(FIX, "index.json"), "utf8"));
let lines = index.lines;
if (li >= 0) { const want = new Set(args[li + 1].split(",")); lines = lines.filter((e) => want.has(e.id)); }

const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".txt": "text/plain; charset=utf-8" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>rec test</title>"); }
  const f = u.startsWith("/web/") ? path.join(WEB, u.slice(5)) : null;
  if (!f || !f.startsWith(WEB)) { res.writeHead(404); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const flags = useGpu ? ["--enable-unsafe-webgpu"]
  : ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader"];
const exe = fs.existsSync("/opt/pw-browsers/chromium") ? "/opt/pw-browsers/chromium" : undefined;
const browser = await chromium.launch({ executablePath: exe, args: flags });
const page = await browser.newPage();
page.on("console", (m) => process.stderr.write(`${m.text()}\n`));
page.on("pageerror", (e) => process.stderr.write(`pageerror: ${e.message}\n`));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
let out;
try {
  out = await page.evaluate(async (a) => (await import("/web/rec/test/check_rec.js")).default(a),
    { models, lines: lines.map((e) => ({ id: e.id, W: e.W })), batch: li >= 0 ? null : index.batch, variants, pack });
} catch (e) {
  out = { error: String(e && e.message || e) };
}
await browser.close();
server.close();
if (out.error) { console.log(JSON.stringify(out)); process.exit(1); }

const res = fs.existsSync(RESULTS) ? JSON.parse(fs.readFileSync(RESULTS, "utf8")) : { models: {} };
res.models = res.models || {};
res.adapter = out.adapter;
let ok = true;
const summary = {};
for (const [m, vs] of Object.entries(out.models)) {
  const key = pack ? `${m}-${pack}` : m;   // the results key ref_rec.py used
  const rm = res.models[key] = res.models[key] || {};
  const s = summary[key] = {};
  for (const [v, r] of Object.entries(vs)) {
    const L = Object.values(r.lines);
    const py = rm.lines || {};
    const pyKey = v === "packed" ? "text_packed" : "text_ort";
    const entry = {
      lines: L.length,
      max_abs: Math.max(...L.map((x) => x.max_abs)),
      same_text: L.filter((x) => x.same_text).length,
      // greedy text of the WGSL output against the text Python decoded from its own reference
      same_text_as_python: Object.entries(r.lines).filter(([id, x]) => py[id] && py[id][pyKey] === x.text).length,
      swiftshader_ms_total: L.reduce((a, x) => a + x.swiftshader_ms, 0),
      per_line: Object.fromEntries(Object.entries(r.lines).map(([id, x]) => [id, { T: x.T, max_abs: x.max_abs, same_text: x.same_text, text: x.text }])),
    };
    if (r.batch) entry.batch = r.batch;
    if (v === "packed") {
      entry.pass = entry.max_abs <= TOL && entry.same_text === L.length && entry.same_text_as_python === L.length
        && (!r.batch || (r.batch.max_abs <= TOL && r.batch.same_text));
      ok = ok && entry.pass;
    }
    rm[`wgsl_${v}`] = entry;
    s[v] = { max_abs: entry.max_abs, same_text: `${entry.same_text}/${L.length}`, same_text_as_python: `${entry.same_text_as_python}/${L.length}`,
      batch_max_abs: r.batch ? r.batch.max_abs : undefined, batch_same_text: r.batch ? r.batch.same_text : undefined, pass: entry.pass };
  }
}
res.wgsl_pass = ok;
fs.mkdirSync(path.dirname(RESULTS), { recursive: true });
fs.writeFileSync(RESULTS, JSON.stringify(res, null, 1) + "\n");
console.log(JSON.stringify({ ok, adapter: out.adapter, models: summary }, null, 1));
process.exit(ok ? 0 : 1);
