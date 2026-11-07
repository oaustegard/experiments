// Load web/index.html in headless Chromium (SwiftShader), wait for one generation, screenshot.
// node run/shot.mjs OUT.png [query] [width] [height]
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const { chromium } = createRequire(import.meta.url)("playwright");
const web = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "web");
const [out = "/tmp/claude-0/shot.png", query = "", w = "1200", h = "800"] = process.argv.slice(2);
const port = 9765 + Math.floor(Math.random() * 1000);
const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: web, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader",
  "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1 });
  page.on("console", (m) => console.log("[page]", m.text()));
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto(`http://127.0.0.1:${port}/index.html${query}`);
  await page.waitForFunction(() => /steps in|Error|lost/.test(document.getElementById("status").textContent), null, { timeout: 600000 });
  console.log("status:", await page.textContent("#status"));
  await page.screenshot({ path: out });
} finally { await browser.close(); srv.kill(); }
