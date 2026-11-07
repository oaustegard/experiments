// The page on SwiftShader: loads, builds a collage from the stored tiles, reacts to controls,
// generates one fresh texture through wgsl-diffusion (64 px, 2 steps: SwiftShader is slow), and exports.
//   node run/check_page.mjs  ->  results/page_check.json, results/page.png, results/page_mobile.png
import fs from "node:fs";
import { open } from "./browser.mjs";
const out = { errors: [] };
const ok = (name, cond, extra = {}) => { out[name] = { pass: !!cond, ...extra }; console.log(cond ? "ok  " : "FAIL", name, JSON.stringify(extra)); };

const t0 = Date.now();
const { page, close } = await open("texture-collage/web/index.html", { ready: () => window.app && window.app.ready && window.app.renders >= 1, dsf: 2 });
page.on("pageerror", (e) => out.errors.push(e.message));
ok("first_collage", true, { ms: Date.now() - t0, timing: await page.evaluate(() => window.app.timing) });
await page.screenshot({ path: "results/page.png" });

for (const [id, v] of [["mode", "colour"], ["tint", "none"], ["scale", "2"], ["mode", "structure"], ["tint", "region"], ["scale", "auto"]]) {
  const n = await page.evaluate(() => window.app.renders);
  await page.click(`#${id} [data-v="${v}"]`);
  await page.waitForFunction((n) => window.app.renders > n, n, { timeout: 60000 });
}
ok("controls_rerender", true, { renders: await page.evaluate(() => window.app.renders) });

// a region's menu overrides its class
const before = await page.evaluate(() => window.app.regions());
const k0 = before.map((r, k) => [r.frac, k]).sort((a, b) => b[0] - a[0])[0][1];
await page.selectOption(`#reg${k0} select`, { label: "chequered" });
await page.waitForFunction((k) => window.app.regions()[k].name === "chequered", k0, { timeout: 60000 });
ok("menu_override", true, { region: k0 });

await page.waitForFunction(() => window.app.gpu !== undefined, null, { timeout: 120000 });
const gpu = await page.evaluate(() => window.app.gpu);
ok("model_loaded", gpu, { modelLoadMs: await page.evaluate(() => window.app.timing.modelLoad) });
if (gpu) {
  const n = await page.evaluate(() => window.app.renders), g0 = Date.now();
  await page.evaluate((k) => window.app.generate({ size: 64, steps: 2, only: [k] }), k0);
  const after = await page.evaluate(() => window.app.regions());
  ok("fresh_texture", after[k0].seed > 0 && (await page.evaluate(() => window.app.renders)) > n, { ms: Date.now() - g0, S: after[k0].S });
}
const dl = page.waitForEvent("download", { timeout: 30000 });
await page.click("#save");
const d = await dl;
const p = await d.path();
ok("export_png", fs.statSync(p).size > 10000, { bytes: fs.statSync(p).size });
await page.screenshot({ path: "results/page_after.png" });
await page.setViewportSize({ width: 390, height: 844 });
await page.waitForTimeout(500);
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
ok("mobile_no_hscroll", overflow <= 0, { overflow });
await page.screenshot({ path: "results/page_mobile.png", fullPage: false });
ok("no_page_errors", out.errors.length === 0, { errors: out.errors });
await close();
fs.writeFileSync("results/page_check.json", JSON.stringify(out, null, 1));
process.exit(Object.values(out).every((v) => !v || v.pass !== false) ? 0 : 1);
