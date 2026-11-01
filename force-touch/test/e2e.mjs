// The whole loop on one machine, no phone: a local room (rally-lobby under wrangler dev), observe.mjs, and the probe
// page pressed by a stand-in thumb through Chromium's touch emulation. The stand-in reports a contact radius per
// condition and a fixed force, so the run checks the plumbing and the analysis, not any real phone.
//
//   LOBBY_DIR=../oaustegard.github.io/workers/rally-lobby [WRANGLER=path/to/wrangler] node force-touch/test/e2e.mjs
//   python3 force-touch/analyze.py force-touch/results/raw-<ROOM>.jsonl
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { spawn } from "node:child_process"; import { fileURLToPath } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url)), EXP = path.resolve(HERE, "..", "..");
const LOBBY_DIR = process.env.LOBBY_DIR || path.resolve(EXP, "..", "oaustegard.github.io", "workers", "rally-lobby");
const [cmd, ...args] = process.env.WRANGLER ? [process.env.WRANGLER] : ["npx", "--yes", "wrangler@4.20.0"];
const dev = spawn(cmd, [...args, "dev", "--port", "8787", "--ip", "127.0.0.1"], { cwd: LOBBY_DIR, stdio: "ignore" });
try {
  for (let i = 0; i < 90; i++) { try { if ((await fetch("http://127.0.0.1:8787/")).ok) break; } catch (e) {} await new Promise((r) => setTimeout(r, 1000)); }
  const srv = http.createServer((q, r) => { const p = path.join(EXP, new URL(q.url, "http://x").pathname); if (!p.startsWith(EXP) || !fs.existsSync(p)) { r.writeHead(404); return r.end(); } r.writeHead(200, { "content-type": "text/html" }); r.end(fs.readFileSync(p)); });
  await new Promise((ok) => srv.listen(0, "127.0.0.1", ok));
  const ROOM = "FT" + Math.floor(Math.random() * 9000 + 1000);
  const obs = spawn("node", [path.join(EXP, "force-touch", "observe.mjs"), ROOM, "4", "ws://127.0.0.1:8787"], { stdio: "inherit" });
  await new Promise((r) => setTimeout(r, 4000));
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 390, height: 760 }, hasTouch: true, isMobile: true }), page = await ctx.newPage();
  page.on("pageerror", (e) => console.log("PAGE ERROR", e.message));
  await page.goto(`http://127.0.0.1:${srv.address().port}/force-touch/web/index.html?room=${ROOM}&lobby=ws://127.0.0.1:8787`);
  // the observer joined first, so the phone must learn of it from the room's hello
  await page.waitForFunction(() => /Claude is watching/.test(document.getElementById("status").textContent), null, { timeout: 15000 })
    .then(() => console.log("ok   the phone sees the observer that was there first"), () => { console.log("FAIL the phone never saw the observer"); process.exitCode = 1; });
  await page.click("#start");
  const cdp = await ctx.newCDPSession(page), box = await page.locator("#pad").boundingBox(), x = box.x + box.width / 2, y = box.y + box.height / 2;
  // the stand-in thumb: firmer = bigger contact, except the posture pair, where a flat light thumb is the biggest
  const size = { light: 8, medium: 12, firm: 18, "flat-light": 20, "tip-firm": 7 };
  for (const cond of await page.evaluate(() => window.__ft.PLAN)) {
    await page.waitForFunction(() => document.getElementById("pad").classList.contains("armed"), null, { timeout: 15000 });
    const pt = (dr) => [{ x: x + dr, y, radiusX: size[cond] + dr, radiusY: size[cond] + dr, force: 1 }];
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: pt(0) });
    for (let k = 0; k < 22; k++) { await page.waitForTimeout(100); await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: pt((k % 3) * 0.3) }); }
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  }
  await new Promise((r) => obs.on("exit", r));
  await b.close(); srv.close();
  console.log(`room ${ROOM}: now run python3 force-touch/analyze.py force-touch/results/raw-${ROOM}.jsonl`);
} finally { dev.kill(); }
