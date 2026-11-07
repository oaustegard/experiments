// Run web/_test.html (or argv[2]); argv[3] = comma list of variants (f32,f32-nosplit,f16,f16-nosplit).
// Run web/test.html in headless Chromium (SwiftShader WebGPU) and print the comparison with torch.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const web = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "web");
const port = 8765 + Math.floor(Math.random() * 1000);
const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: web, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader",
  "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
try {
  const page = await browser.newPage();
  page.on("console", (m) => console.log("[page]", m.text()));
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto(`http://127.0.0.1:${port}/${process.argv[2] || "_test.html"}`);
  await page.waitForFunction(() => window.ready, null, { timeout: 60000 });
  const out = await page.evaluate((o) => window.runTests(o), process.argv[3] ? process.argv[3].split(",") : null);
  console.log(JSON.stringify(out, null, 1));
} finally { await browser.close(); srv.kill(); }
