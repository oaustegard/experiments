// The splat page on SwiftShader: first collage, controls, a region's menu, zoom and pan, the three exports
// (and the exported HTML rendering on its own), a fresh texture (6 steps), and a recut mid-generation.
//   node run/check_page.mjs -> results/page_check.json, results/page.png, results/page_zoom.png, results/page_mobile.png
import fs from "node:fs";
import path from "node:path";
import { open } from "../../run/browser.mjs";
const out = { errors: [] };
const ok = (name, cond, extra = {}) => { out[name] = { pass: !!cond, ...extra }; console.log(cond ? "ok  " : "FAIL", name, JSON.stringify(extra)); };
const t0 = Date.now();
const { page, browser, port, close } = await open("texture-collage/splat/index.html", { ready: () => window.app && window.app.ready && window.app.renders >= 1, dsf: 2 });
page.on("pageerror", (e) => out.errors.push(e.message));
const renders = () => page.evaluate(() => window.app.renders);
const after = async (n) => page.waitForFunction((n) => window.app.renders > n, n, { timeout: 120000 });
ok("first_collage", true, { ms: Date.now() - t0, timing: await page.evaluate(() => window.app.timing) });
await page.waitForTimeout(1500);
await page.screenshot({ path: "results/page.png" });
for (const [id, v] of [["mode", "nearest"], ["lw", "96"], ["tint", "none"], ["pick", "colour"], ["mode", "spline"], ["mode", "distance"], ["lw", "192"], ["tint", "region"], ["pick", "structure"], ["view", "regions"], ["view", "collage"]]) {
  const n = await renders();
  await page.click(`#${id} [data-v="${v}"]`);
  await after(n);
}
ok("controls_rerender", true, { renders: await renders() });
const regs = await page.evaluate(() => window.app.regions());
const k0 = regs.map((r, k) => [r.frac, k]).sort((a, b) => b[0] - a[0])[0][1];
await page.selectOption(`#reg${k0} select`, { label: "chequered" });
await page.waitForFunction((k) => window.app.regions()[k].name === "chequered", k0);
ok("menu_override", true, { region: k0 });
// zoom 8x into the centre with the wheel, then pan
const box = await page.locator("#gl").boundingBox();
let n = await renders();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
for (let i = 0; i < 8; i++) { await page.mouse.wheel(0, -180); await page.waitForTimeout(50); }
await after(n);
n = await renders();
await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 120, box.y + box.height / 2 + 40, { steps: 5 }); await page.mouse.up();
await after(n);
await page.waitForTimeout(1500);
await page.screenshot({ path: "results/page_zoom.png" });
ok("zoom_pan", true);
const save = async (id) => { const dl = page.waitForEvent("download", { timeout: 300000 }); await page.click(`#${id}`); const d = await dl; return { name: d.suggestedFilename(), path: await d.path() }; };
const w = await save("wgsl"), h = await save("html");
await page.click('#pngw [data-v="2"]');
const p = await save("png");
const size = (f) => fs.statSync(f.path).size;
ok("exports", size(w) > 50000 && size(h) > size(w) && size(p) > 100000, { wgsl: size(w), html: size(h), png: size(p) });
// the exported HTML must render on its own
const root = path.join(path.dirname(new URL(import.meta.url).pathname), "..", "..", "..");
fs.copyFileSync(h.path, path.join(root, "texture-collage/splat/out/_exported.html"));
const v = await browser.newPage({ viewport: { width: 900, height: 600 } });
const verrs = [];
v.on("pageerror", (e) => verrs.push(e.message));
v.on("console", (m) => { if (m.type() === "error") verrs.push(m.text()); });
const tv = Date.now();
await v.goto(`http://127.0.0.1:${port}/texture-collage/splat/out/_exported.html`);
await v.waitForTimeout(90000);
await v.screenshot({ path: "results/exported_html.png" });
ok("exported_html_runs", verrs.length === 0, { errors: verrs, waitedMs: Date.now() - tv });
await v.close();
// one fresh texture, then a recut while a second batch is generating
await page.waitForFunction(() => window.app.gpu !== undefined, null, { timeout: 300000 });
if (await page.evaluate(() => window.app.gpu)) {
  const g0 = Date.now();
  await page.evaluate((k) => window.app.generate({ steps: 6, only: [k] }), k0);
  const r = await page.evaluate(() => window.app.regions());
  ok("fresh_texture", r[k0].seed > 0, { ms: Date.now() - g0 });
  await page.evaluate(() => { window.app.generate({ steps: 6 }); });
  await page.waitForFunction(() => document.getElementById("go").textContent === "Stop", null, { timeout: 60000 });
  await page.waitForTimeout(3000);
  n = await renders();
  await page.fill("#K", "7"); await page.dispatchEvent("#K", "change");
  await page.waitForFunction((n) => window.app.renders > n && window.app.regions().length <= 7, n, { timeout: 300000 });
  if (await page.evaluate(() => document.getElementById("go").textContent === "Stop")) await page.click("#go");
  await page.waitForFunction(() => document.getElementById("go").textContent !== "Stop", null, { timeout: 600000 });
  n = await renders();
  await page.click('#tint [data-v="none"]');
  await after(n);
  ok("recut_during_generation", true);
} else ok("model_loaded", false);
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(1500);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
ok("mobile_no_hscroll", overflow <= 0, { overflow });
await page.screenshot({ path: "results/page_mobile.png" });
ok("no_page_errors", out.errors.length === 0, { errors: out.errors });
await close();
fs.writeFileSync("results/page_check.json", JSON.stringify(out, null, 1));
process.exit(Object.values(out).every((x) => !x || x.pass !== false) ? 0 : 1);
