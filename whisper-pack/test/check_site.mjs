// End-to-end check of the packed engine inside anything-to-text, on SwiftShader, from local packs.
// usage: node test/check_site.mjs [path to the oaustegard.github.io checkout, sibling of experiments by default]
// Serves the directory that holds both checkouts; the page gets ?packbase= and ?packworker= pointing at this repo.
import { createRequire } from "node:module";
import { spawn, execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(execSync("npm root -g").toString().trim() + "/");
const { chromium } = require("playwright");
const here = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.resolve(here, "../..");
const site = path.relative(root, process.argv[2] || path.join(root, "oaustegard.github.io"));
const exp = path.relative(root, here);
const port = 8793;
const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
let code = 1;
try {
  const browser = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => { const t = m.text(); if (/\[load\]|\[worker|\[run\]|\[vad\]|pack|error/i.test(t)) { logs.push(t); console.log("  " + t.slice(0, 200)); } });
  page.on("pageerror", (e) => console.log("pageerror", e.message));
  const base = `http://127.0.0.1:${port}`;
  const q = `packbase=${encodeURIComponent(`${base}/${exp}/out/local/`)}&packworker=${encodeURIComponent(`${base}/${exp}/web/worker.js`)}`;
  await page.goto(`${base}/${site}/web-utilities/anything-to-text.html?${q}`);
  const results = [];
  for (const [lang, want] of [["en", "tiny.en-r63/"], ["auto", "tiny-r63/"]]) {
    await page.setInputFiles("#file", path.join(here, "test/fixtures/ts/clip1.wav"));
    await page.waitForSelector("#model", { state: "visible", timeout: 60_000 });
    await page.selectOption("#model", "pack:tiny");
    await page.selectOption("#lang", lang);
    await page.waitForFunction(() => !document.getElementById("tr-run").disabled, null, { timeout: 60_000 });
    logs.length = 0;
    await page.click("#tr-run");
    await page.waitForFunction(() => /Done in|failed|error/i.test(document.getElementById("tr-status").textContent), null, { timeout: 900_000 });
    const status = await page.textContent("#tr-status");
    const text = await page.evaluate(() => Array.from(document.querySelectorAll("#segments *")).map((e) => e.value || "").join(" ") + " " + document.getElementById("segments").textContent);
    const note = await page.evaluate(() => { const n = document.getElementById("tech-note") || document.querySelector(".tech-note"); return n ? n.textContent : ""; });
    const usedPack = logs.some((l) => l.includes("[load] pack=") && l.includes(want)) || logs.every((l) => !l.includes("[load]"));
    const ok = /Done in/.test(status) && /solution of their problem/i.test(text) && !logs.some((l) => /pack load failed/.test(l));
    results.push({ lang, want, status, ok, text: text.replace(/\s+/g, " ").trim().slice(0, 160), note: note.slice(0, 120) });
    console.log(JSON.stringify(results[results.length - 1]));
  }
  await browser.close();
  if (results.every((r) => r.ok)) code = 0;
  console.log(code === 0 ? "PASS" : "FAIL");
} finally { srv.kill(); }
process.exit(code);
