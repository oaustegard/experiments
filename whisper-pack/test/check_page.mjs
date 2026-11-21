// Opens web/index.html in headless Chromium on SwiftShader with the local tiny pack and a fixture clip,
// waits for window.result, and exits 1 unless there is no error, a chunk, and the expected phrase.
import { createRequire } from "node:module";
import { spawn, execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(execSync("npm root -g").toString().trim() + "/");
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..", "..");   // gpu-bridge/web/relay.js must resolve
const port = 8792;
const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: repoRoot, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
let code = 1;
try {
  const browser = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
  const page = await browser.newPage();
  page.on("console", (m) => console.log(m.text()));
  page.on("pageerror", (e) => console.log("pageerror", e.message));
  await page.goto(`http://127.0.0.1:${port}/whisper-pack/web/index.html?pack=../out/packs/whisper-tiny-0/&clip=../test/fixtures/ts/clip1.f32`);
  await page.waitForFunction(() => window.result, null, { timeout: 600_000 });
  const r = await page.evaluate(() => window.result);
  await browser.close();
  console.log(JSON.stringify(r, null, 2));
  if (!r.error && r.chunks && r.chunks.length > 0 && /solution of their problem/i.test(r.text)) code = 0;
  console.log(code === 0 ? "PASS" : "FAIL");
} finally { srv.kill(); }
process.exit(code);
