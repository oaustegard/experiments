// Checks web/tesslayout.js against the system Tesseract CLI on five pages.
//
//   node web/test/check_tesslayout.mjs
//
// The yardstick is Tesseract's own layout-only result: tesserocr (PyTessBaseAPI, PSM.AUTO, AnalyseLayout, RIL.TEXTLINE
// boxes) run on the same pages. Every reference line is matched to the WASM line with the highest IoU; it is matched
// when that IoU is >= 0.8. Pass: at least 95% of reference lines matched on every page, plus order and block sanity
// checks on the WASM output. For context the same matching is also reported against the system tesseract CLI's
// recognised TSV lines (level 4), which include recognition-stage geometry and are not a pass criterion.
// Exit 0 on pass, 1 on fail, 2 when the setup is missing.
//
// Serves web/ at /web/ and data/ at /data/ on a loopback port. jsDelivr is loaded by the page at run time.
// Playwright is resolved from PLAYWRIGHT_MODULE (a path) or the bare name "playwright".

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const here = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(here, "..");
const DATA = path.resolve(WEB, "../data");
const PAGES = [
  "tesslayout/synth_two_column.png",
  "tesslayout/synth_heading_table.png",
  "tesslayout/synth_rotated_1deg.png",
  "tesslayout/funsd_form_0.png",
  "tesslayout/funsd_form_1.png",
];
const MATCH_IOU = 0.8;
const TESSDATA = path.join(DATA, "tesslayout/tessdata_fast");
const TESSDATA_URL = "https://cdn.jsdelivr.net/gh/tesseract-ocr/tessdata_fast@87416418657359cb625c412a48b6e1d6d41c29bd/eng.traineddata";

// tesserocr reference: Tesseract's layout-only lines on the same pages (see data/tesslayout/ref_layout.py).
function tesserocrRef() {
  const py = path.join(DATA, "tesslayout/ref_layout.py");
  const out = execFileSync("python3", ["-I", py, TESSDATA, ...PAGES.map((p) => path.join(DATA, p))], {
    env: { ...process.env, OMP_THREAD_LIMIT: "1" },
    encoding: "utf8",
    maxBuffer: 64 << 20,
    stdio: ["ignore", "pipe", "ignore"],
  });
  const j = JSON.parse(out);
  const byPage = {};
  for (const p of PAGES) byPage[p] = j.pages[path.join(DATA, p)].lines;
  return { version: j.tesseract, pages: byPage };
}

async function ensureTessdata() {
  if (fs.existsSync(path.join(TESSDATA, "eng.traineddata"))) return;
  const res = await fetch(TESSDATA_URL);
  if (!res.ok) throw new Error(`fetching tessdata_fast: ${res.status}`);
  fs.mkdirSync(TESSDATA, { recursive: true });
  fs.writeFileSync(path.join(TESSDATA, "eng.traineddata"), Buffer.from(await res.arrayBuffer()));
}
const PASS_RATE = 0.95;

async function loadPlaywright() {
  const candidates = [process.env.PLAYWRIGHT_MODULE, "playwright", "/opt/node-tools/node_modules/playwright"].filter(Boolean);
  for (const c of candidates) {
    try {
      const require = createRequire(import.meta.url);
      const resolved = c.startsWith("/") ? c : require.resolve(c);
      const mod = await import(pathToFileURL(resolved).href);
      return mod.chromium ?? mod.default.chromium;
    } catch (_) { /* try the next candidate */ }
  }
  console.error("check_tesslayout: cannot find Playwright; set PLAYWRIGHT_MODULE to its directory or entry file");
  process.exit(2);
}

const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".png": "image/png", ".html": "text/html" };
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (u === "/") { res.writeHead(200, { "content-type": "text/html" }); return res.end("<!doctype html><title>tesslayout</title>"); }
  let f = null;
  if (u.startsWith("/web/")) f = path.join(WEB, u.slice(5));
  else if (u.startsWith("/data/")) f = path.join(DATA, u.slice(6));
  if (!f || !f.startsWith(WEB + path.sep) && !f.startsWith(DATA + path.sep)) { res.writeHead(404); return res.end(); }
  fs.readFile(f, (err, buf) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
    res.end(buf);
  });
});

// CLI reference: system tesseract 5.3.4. OMP_THREAD_LIMIT=1 because OpenMP thread start-up made the CLI stall here.
// CLI reference: the system tesseract 5.3.4 (the same command as the spec). Its stderr chatter is discarded.
function cliLines(file) {
  const args = [file, "-", "--psm", "3", "--oem", "1", "-l", "eng", "tsv"];
  const out = execFileSync("tesseract", args, {
    env: { ...process.env, OMP_THREAD_LIMIT: "1" },
    encoding: "utf8",
    stdio: ["ignore", "pipe", "ignore"],
    maxBuffer: 64 << 20,
  });
  const lines = [];
  for (const row of out.split("\n").slice(1)) {
    const f = row.split("\t");
    if (f[0] !== "4" || f.length < 10) continue;
    const left = +f[6], top = +f[7], w = +f[8], h = +f[9];
    lines.push([left, top, left + w, top + h]);
  }
  return lines;
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

function compare(cli, wasm) {
  const best = cli.map((c) => wasm.reduce((m, w) => Math.max(m, iou(c, w.box)), 0));
  const matched = best.filter((v) => v >= MATCH_IOU).length;
  const sorted = [...best].sort((a, b) => a - b);
  return {
    cli: cli.length,
    wasm: wasm.length,
    matched,
    rate: cli.length ? matched / cli.length : 0,
    medianIoU: sorted.length ? sorted[sorted.length >> 1] : 0,
  };
}

function sanity(lines) {
  const problems = [];
  lines.forEach((l, i) => {
    if (l.order !== i) problems.push(`order ${l.order} at index ${i}`);
    if (l.box.some((v) => !Number.isFinite(v))) problems.push(`non-finite box at ${i}`);
    if (i > 0 && l.block < lines[i - 1].block) problems.push(`block number decreases at ${i}`);
  });
  return problems;
}

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;
const chromium = await loadPlaywright();
const browser = await chromium.launch();
let exitCode = 0;
try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => process.stderr.write(`pageerror: ${e.message}\n`));
  await page.goto(`${base}/`);

  await ensureTessdata();
  const ref = tesserocrRef();
  const wasm = await page.evaluate(async (pages) => {
    const { tessLayout } = await import("/web/tesslayout.js");
    const out = {};
    for (const p of pages) {
      const blob = await (await fetch("/data/" + p)).blob();
      const bmp = await createImageBitmap(blob);
      const canvas = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(bmp, 0, 0);
      const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      const r = await tessLayout(rgba, bmp.width, bmp.height, { psm: 3 });
      out[p] = { lines: r.lines, ms: r.ms, width: bmp.width, height: bmp.height };
    }
    const { tessLayoutInfo } = await import("/web/tesslayout.js");
    return { pages: out, info: tessLayoutInfo() };
  }, PAGES);

  // Download size: encoded bytes of the jsDelivr resources this page fetched (core glue, wasm, model).
  const transfers = await page.evaluate(() => performance.getEntriesByType("resource")
    .filter((e) => e.name.startsWith("https://cdn.jsdelivr.net/"))
    .map((e) => ({ name: e.name.split("/").slice(-1)[0], encoded: e.encodedBodySize, decoded: e.decodedBodySize })));

  let allPass = true;
  console.log(`reference: tesserocr, Tesseract ${ref.version} (layout-only, PSM.AUTO)`);
  console.log(`tesseract.js-core ${wasm.info.tesseractCoreVersion} (${wasm.info.coreFile}), Tesseract ${wasm.info.tesseractVersion}; system CLI 5.3.4`);
  console.log(`tessdata_fast ${wasm.info.tessdataCommit.slice(0, 12)}, model ${wasm.info.modelBytes} bytes, page load ${wasm.info.loadMs.toFixed(0)} ms`);
  for (const t of transfers) console.log(`  fetched ${t.name}: ${t.encoded} bytes on the wire, ${t.decoded} decoded`);

  for (const p of PAGES) {
    const file = path.join(DATA, p);
    const w = wasm.pages[p];
    const c = compare(ref.pages[p].map((l) => l.box), w.lines);
    const cli = cliLines(file).sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    const cc = compare(cli, w.lines);
    const problems = sanity(w.lines);
    const pass = c.rate >= PASS_RATE && problems.length === 0 && c.cli > 0;
    allPass = allPass && pass;
    console.log(`${pass ? "PASS" : "FAIL"} ${p}: vs tesserocr layout-only ${c.matched}/${c.cli} matched at IoU>=${MATCH_IOU} `
      + `(${(100 * c.rate).toFixed(1)}%), WASM ${c.wasm} lines, median best IoU ${c.medianIoU.toFixed(3)}, WASM ${w.ms.toFixed(1)} ms`);
    console.log(`     context vs system CLI recognised lines: ${cc.matched}/${cc.cli} (${(100 * cc.rate).toFixed(1)}%), CLI ${cc.cli} lines`);
    for (const pr of problems.slice(0, 5)) console.log(`     sanity: ${pr}`);
  }
  console.log(allPass ? "OK: all pages pass" : `FAILED: below ${100 * PASS_RATE}% vs tesserocr on at least one page`);
  exitCode = allPass ? 0 : 1;
} catch (e) {
  console.error(`check_tesslayout: ${e && e.stack || e}`);
  exitCode = 1;
} finally {
  await browser.close();
  server.close();
}
process.exit(exitCode);
