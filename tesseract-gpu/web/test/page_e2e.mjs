// End-to-end page check in headless Chromium (SwiftShader WebGPU): load the page, read each sample with clean-up on
// and off and with both reading orders, print the lines. Needs network for jsDelivr (layout engine and models).
//   node web/test/page_e2e.mjs            (serves the experiments repo root on :8765)
import { createRequire } from "node:module";
import path from "node:path";
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.execPath, "../../lib/node_modules/playwright"))); }
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "../../..");
const srv = spawn("python3", ["-m", "http.server", "8765", "--bind", "127.0.0.1"], { cwd: repo, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
let ok = true;
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("http://127.0.0.1:8765/tesseract-gpu/web/", { waitUntil: "load" });
  const ready = () => page.waitForFunction(() => /ready|error/.test(document.getElementById("status").className), null, { timeout: 180000 });
  await ready();
  console.log("status:", await page.textContent("#status-text"));
  let run = 0;
  // every action below triggers exactly one reading; wait for it, then report
  async function step(label, action) {
    await action();
    run += 1;
    await page.waitForFunction((n) => Number(document.getElementById("result").dataset.run || 0) >= n || document.getElementById("status").classList.contains("error"), run, { timeout: 180000 });
    const lines = await page.$$eval("#lines li .t", (els) => els.map((e) => e.textContent));
    const timings = await page.$$eval("#timings tr", (rs) => rs.map((r) => r.textContent));
    console.log(`\n== ${label}: ${lines.length} lines; ${await page.textContent("#status-text")}`);
    console.log(lines.slice(0, 6).map((l, i) => `  ${i + 1}. ${l}`).join("\n"));
    console.log("  " + timings.join(" | "));
    if (!lines.filter((l) => l !== "no text").length) ok = false;
  }
  await step("english, clean-up on, Tesseract order", () => page.setInputFiles("#file", join(here, "../samples/english.png")));
  await step("english, clean-up off", () => page.setChecked("#clean", false));
  await step("english, clean-up on", () => page.setChecked("#clean", true));
  await step("english, column bands", () => page.selectOption("#order", "columns"));
  await step("english, word lists off", () => page.setChecked("#dict", false));
  await step("english, word lists on", () => page.setChecked("#dict", true));
  await page.selectOption("#lang", "eng-ft");
  run += 1; // switching the model rereads the open page
  await page.waitForFunction((n) => Number(document.getElementById("result").dataset.run || 0) >= n, run, { timeout: 180000 });
  console.log(`\n== english, fine-tuned model: ${await page.textContent("#status-text")} | ${await page.textContent("#model-info")}`);
  console.log((await page.$$eval("#lines li .t", (els) => els.map((e) => e.textContent))).slice(0, 3).map((l, i) => `  ${i + 1}. ${l}`).join("\n"));
  await page.selectOption("#lang", "fra");
  run += 1; // switching the model rereads the open page
  await page.waitForFunction((n) => Number(document.getElementById("result").dataset.run || 0) >= n, run, { timeout: 180000 });
  await step("french sample, French model", () => page.setInputFiles("#file", join(here, "../samples/french.png")));
  await step("two-page PDF, page 1 (French model)", () => page.setInputFiles("#file", join(here, "../samples/two-pages.pdf")));
  if (await page.isHidden("#pdfnav")) { console.log("FAIL pdf navigation hidden"); ok = false; }
  await step("two-page PDF, page 2", () => page.click("#pdfnext"));
  if (errors.length) { ok = false; console.log("page errors:", errors); }
} finally {
  await browser.close();
  srv.kill();
}
process.exit(ok ? 0 : 1);
