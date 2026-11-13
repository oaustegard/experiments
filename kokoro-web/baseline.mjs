// Resident memory of Chromium's renderer for an empty page, the floor under every measurement in bench.mjs.
import { createRequire } from "node:module";
import { join } from "node:path";
import { execSync } from "node:child_process";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const browser = await chromium.launch({ args: ["--js-flags=--expose-gc"] });
const page = await browser.newPage();
await page.setContent("<!doctype html><title>empty</title><p>empty");
await new Promise((r) => setTimeout(r, 1500));
const rss = Math.max(...execSync("ps -eo rss,args").toString().split("\n").filter((l) => l.includes("--type=renderer"))
  .map((l) => +l.trim().split(/\s+/)[0] * 1024));
console.log(JSON.stringify({ empty_page_renderer_rss_mb: Math.round(rss / 1e6) }));
await browser.close();
