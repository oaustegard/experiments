// Shared: serve the repo root and open a page in headless Chromium on SwiftShader WebGPU.
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const { chromium } = createRequire(import.meta.url)("playwright");
export const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
export async function open(rel, { ready = () => window.ready, viewport = null, dsf = 1 } = {}) {
  const port = 8765 + Math.floor(Math.random() * 1000);
  const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 800));
  const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader",
    "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
  const page = await browser.newPage({ viewport: viewport || { width: 1280, height: 800 }, deviceScaleFactor: dsf, acceptDownloads: true });
  page.on("console", (m) => console.log("[page]", m.text()));
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto(`http://127.0.0.1:${port}/${rel}`);
  await page.waitForFunction(ready, null, { timeout: 120000 });
  return { page, port, browser, close: async () => { await browser.close(); srv.kill(); } };
}
