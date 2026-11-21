// Runs test/check.html in headless Chromium on SwiftShader and exits 1 unless every case matches
// the NumPy reference: identical tokens, encoder cosine >= 0.9999, first-step logit cosine >= 0.9999.
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { execSync } from "node:child_process";
const require = createRequire(execSync("npm root -g").toString().trim() + "/");
const { chromium } = require("playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const port = 8791;
const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
let code = 1;
try {
  const browser = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
  const page = await browser.newPage();
  page.on("console", (m) => console.log(m.text()));
  page.on("pageerror", (e) => console.log("pageerror", e.message));
  const q = process.argv[2] ? "?only=" + process.argv[2] : "";
  await page.goto(`http://127.0.0.1:${port}/test/check.html${q}`);
  await page.waitForFunction(() => window.result, null, { timeout: 3600_000 });
  const r = await page.evaluate(() => window.result);
  await browser.close();
  if (!r.error && r.rows.length && r.rows.every((x) => x.same && x.encCos >= 0.9999 && x.logitCos >= 0.9999)) code = 0;
  console.log(code === 0 ? `PASS ${r.rows.length} cases` : "FAIL");
} finally { srv.kill(); }
process.exit(code);
