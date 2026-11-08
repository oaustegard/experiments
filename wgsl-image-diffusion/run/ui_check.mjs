// Drives the page's steering controls in headless Chromium (SwiftShader) and screenshots each stage:
// generate from noise -> use the result as the picture -> mark a region to redo -> transform; then a
// class blend from noise. node run/ui_check.mjs OUTDIR [query]
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const { chromium } = createRequire(import.meta.url)("playwright");
const web = process.env.WEB_DIR || path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "web");
const [out = "/tmp/claude-0/ui", query = "?steps=4&seed=5"] = process.argv.slice(2);
const port = 9765 + Math.floor(Math.random() * 1000);
const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: web, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader",
  "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const done = (page) => page.waitForFunction(() => / s · seed|Stopped|Error|lost/.test(document.getElementById("status").textContent),
  null, { timeout: 900000 });
try {
  const page = await browser.newPage({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  page.on("console", (m) => { if (m.type() === "error") console.log("[console]", m.text().slice(0, 300)); });
  await page.goto(`http://127.0.0.1:${port}/index.html${query}`);
  await done(page);
  console.log("1:", await page.textContent("#status"));
  await page.screenshot({ path: `${out}-1-noise.png` });
  await page.click('#mode button[data-v="picture"]');
  await page.click("#useRes");
  await page.click('#tool button[data-v="mask"]');
  const box = await page.locator("#ov").boundingBox();
  await page.mouse.move(box.x + box.width * 0.25, box.y + box.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.75, box.y + box.height * 0.3, { steps: 12 });
  await page.mouse.move(box.x + box.width * 0.75, box.y + box.height * 0.5, { steps: 6 });
  await page.mouse.up();
  await page.screenshot({ path: `${out}-2-marked.png` });
  await page.fill("#strength", "1");
  await page.click("#go");
  await done(page);
  console.log("2:", await page.textContent("#status"));
  await page.screenshot({ path: `${out}-3-redone.png` });
  await page.click('#mode button[data-v="noise"]');
  await page.selectOption("#blend", "1");
  await page.click("#go");
  await done(page);
  console.log("3:", await page.textContent("#status"));
  await page.screenshot({ path: `${out}-4-blend.png` });
} finally { await browser.close(); srv.kill(); }
