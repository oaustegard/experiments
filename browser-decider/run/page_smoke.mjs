// Load web/index.html in headless Chromium (SwiftShader WebGPU), wait for the first case to render,
// save a screenshot. node run/page_smoke.mjs TOKENIZERS_MJS OUT_PNG
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const { chromium } = createRequire(import.meta.url)("playwright");
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const [tokLib, png] = process.argv.slice(2);
const port = 8900 + Math.floor(Math.random() * 90);
const server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: resolve(ROOT, ".."), stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
try {
  const b = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--disable-vulkan-surface"] });
  const page = await b.newPage({ viewport: { width: 900, height: 1400 }, deviceScaleFactor: 1 });
  const errs = [];
  page.on("pageerror", (e) => errs.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
  await page.route("https://cdn.jsdelivr.net/**", (r) => r.fulfill({ status: 200, contentType: "text/javascript", body: readFileSync(tokLib) }));
  await page.goto(`http://127.0.0.1:${port}/browser-decider/web/index.html`);
  await page.waitForFunction("document.querySelectorAll('.q').length >= 5 || document.getElementById('err').textContent", null, { timeout: 1_500_000 });
  console.log(JSON.stringify({ err: await page.textContent("#err"), questions: await page.locator(".q").count(),
    verdicts: await page.locator(".verdict").allTextContents(), stats: await page.locator(".stat").allTextContents(), errs }));
  await page.screenshot({ path: png, fullPage: false });
  await b.close();
} finally { server.kill(); }
