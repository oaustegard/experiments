// Records candidate sets from the page in harvest mode: per decision point, the eight candidates'
// descriptions, first-bar code and 32 features. Appends JSON lines; run/embed.py adds the Ettin vectors.
//   python3 -m http.server 8765 --bind 127.0.0.1 -d <experiments root> &
//   node run/harvest.mjs <n> <out.jsonl> [chunk]
import { createRequire } from "node:module";
import fs from "node:fs";
const { chromium } = createRequire(import.meta.url)("playwright"); // CJS resolution honours NODE_PATH=$(npm root -g)

const [n = "4", out = "results/harvest.jsonl", chunk = "10"] = process.argv.slice(2);
const N = +n, C = +chunk;
const done = fs.existsSync(out) ? fs.readFileSync(out, "utf8").split("\n").filter(Boolean).length : 0;
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader",
  "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
await page.goto("http://127.0.0.1:8765/strudel-taste/web/index.html?harvest", { waitUntil: "load" });
await page.waitForFunction(() => typeof window.harvest === "function", null, { timeout: 60000 });
for (let i = done; i < N; i += C) {
  const t0 = Date.now();
  const rows = await page.evaluate(([k, off]) => window.harvest(k, { offset: off, embed: false }), [Math.min(C, N - i), i]);
  fs.appendFileSync(out, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  console.log(`${i + rows.length}/${N} points, ${((Date.now() - t0) / rows.length / 1000).toFixed(1)} s per point`);
}
if (errors.length) console.log("console:", errors.slice(0, 8).join("\n"));
await browser.close();
