// Runs test/check.html in headless Chromium on SwiftShader and exits 1 unless the engine matches the
// quantized PyTorch reference (run/ref.py fixtures) at the reduced canvas: preprocessing within 1e-5,
// encoder output cosine >= 0.9999, first-step argmax equal, the generated tokens identical.
//   node test/check.mjs [pack-name] [fix=<fixtures dir>]
import { createRequire } from "node:module";
import { spawn, execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(execSync("npm root -g").toString().trim() + "/");
const { chromium } = require("playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const port = 8793;
const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
let code = 1;
try {
  const browser = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
  const page = await browser.newPage();
  page.on("console", (m) => console.log(m.text()));
  page.on("pageerror", (e) => console.log("pageerror", e.message));
  await page.goto(`http://127.0.0.1:${port}/test/check.html?pack=${process.argv[2] || "default"}${process.argv[3] ? "&" + process.argv[3] : ""}`);
  await page.waitForFunction(() => window.result, null, { timeout: 3600_000 });
  const r = await page.evaluate(() => window.result);
  await browser.close();
  const ok = (x) => x.pixMax <= 1e-5 && x.encCos >= 0.9999 && x.argmax === x.argmaxRef && x.same && x.rawSame;
  if (!r.error && r.rows.length && r.rows.every(ok)) code = 0;
  console.log(code === 0 ? "PASS" : "FAIL");
} finally { srv.kill(); }
process.exit(code);
