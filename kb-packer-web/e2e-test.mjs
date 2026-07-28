// E2E: PDF ingestion in kb-packer.html. Generates a real PDF (Chromium page.pdf),
// routes the cdnjs pdf.js to the local vendored copy (offline test), drives the
// packer, and verifies the downloaded .skill contains the extracted text and is
// queryable by the shipped search.py.
//
// Optional/manual: needs `npm i playwright-core`, $CHROMIUM (defaults to the CCotw
// preinstalled build), and vendor/pdfjs/{pdf.min.js,pdf.worker.min.js}
// (curl from cdnjs/3.11.174). Run from a dir where node resolves playwright-core.
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const DIR = path.dirname(fileURLToPath(import.meta.url));
const CHROME = process.env.CHROMIUM || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";

const srv = http.createServer((req, res) => {
  const fp = path.join(DIR, decodeURIComponent(req.url.split("?")[0]));
  if (!fp.startsWith(DIR) || !fs.existsSync(fp) || fs.statSync(fp).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "content-type": fp.endsWith(".html") ? "text/html" : "application/octet-stream" });
  fs.createReadStream(fp).pipe(res);
});
await new Promise((r) => srv.listen(0, r));
const port = srv.address().port;
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "pdf-"));

const b = await chromium.launch({ executablePath: CHROME, args: ["--no-sandbox"] });
const ctx = await b.newContext({ acceptDownloads: true });
await ctx.route(/cdnjs\.cloudflare\.com.*pdf\.(min|worker\.min)\.js/, (route) => {
  const name = route.request().url().includes("worker") ? "pdf.worker.min.js" : "pdf.min.js";
  route.fulfill({ path: path.join(DIR, "vendor/pdfjs", name), contentType: "text/javascript" });
});

const gen = await ctx.newPage();
await gen.setContent("<h1>Database Notes</h1><p>PostgreSQL is a relational database with strong ACID guarantees and SQL support.</p>");
const pdfPath = path.join(tmp, "notes.pdf");
await gen.pdf({ path: pdfPath });
await gen.close();

const page = await ctx.newPage();
const errs = [];
page.on("pageerror", (e) => errs.push(String(e)));
await page.goto(`http://localhost:${port}/kb-packer.html`);
await page.setInputFiles("#picker", [pdfPath]);
await page.fill("#name", "pdf-kb");
await page.click("#build");
await page.waitForSelector("a.dl", { timeout: 15000 });
const status = await page.textContent("#status");
const [dl] = await Promise.all([page.waitForEvent("download"), page.click("a.dl")]);
const out = path.join(tmp, "pdf-kb.skill");
await dl.saveAs(out);
await b.close();
srv.close();

const bundle = path.join(tmp, "u");
execFileSync("python3", ["-c", `import zipfile;zipfile.ZipFile('${out}').extractall('${bundle}')`]);
const chunks = fs.readFileSync(path.join(bundle, "pdf-kb", "chunks.jsonl"), "utf8");
const hasText = /relational database/i.test(chunks);
const hit = JSON.parse(execFileSync("python3", [path.join(bundle, "pdf-kb", "search.py"),
  "--index", path.join(bundle, "pdf-kb"), "--core", "database", "--expand", "relational", "--k", "1"],
  { encoding: "utf8" })).hits[0];

console.log("status:", status);
console.log("extracted text present:", hasText, "| query hit:", hit && hit.id);
const ok = !errs.length && /Built 1 chunks/.test(status) && hasText && hit && hit.id === "notes.pdf#chunk-0";
if (errs.length) console.log("ERRORS", errs);
console.log("\nPDF E2E:", ok ? "PASS" : "FAIL");
fs.rmSync(tmp, { recursive: true, force: true });
process.exit(ok ? 0 : 1);
