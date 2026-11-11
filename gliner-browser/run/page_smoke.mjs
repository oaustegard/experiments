// Headless smoke test of the demo page web/index.html (SwiftShader WebGPU in Chromium).
//
//   NODE_PATH=$(npm root -g) node run/page_smoke.mjs TOKENIZERS_MJS
//
// TOKENIZERS_MJS: a local copy of @huggingface/tokenizers@0.2.0/dist/tokenizers.mjs, served in place of
// the jsDelivr request (as run/check_web.mjs does). Serves the experiments repo on 127.0.0.1, then:
//   1. no WebGPU: navigator.gpu removed before the page runs; the warning shows, nothing throws.
//   2. text mode: waits for the model, checks the default example gets highlighted entities, among them
//      "Satya Nadella" as person and Microsoft as company; screenshot out/page_text.png.
//   3. Bluesky mode against a stubbed public AppView (run/fixtures/bsky_author_feed.json: 30 posts in two
//      pages plus two reposts); checks both pages were fetched, the reposts skipped, all 30 posts analysed,
//      Microsoft grouped across "Microsoft", "Microsoft's" and "@microsoft" with the post count, and that
//      clicking it lists its posts with the entity highlighted and a bsky.app link; screenshot out/page_bsky.png.
//   Plus out/page_phone.png (390 px wide) for a layout look. Any request off 127.0.0.1 other than the two
//   stubs fails the test. Exits 0 on PASS; details in out/page_smoke.json.
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const { chromium } = createRequire(import.meta.url)("playwright");
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(ROOT, "..");
const tokLib = process.argv[2];
if (!tokLib || !existsSync(tokLib)) { console.error("usage: node run/page_smoke.mjs TOKENIZERS_MJS"); process.exit(2); }
const fx = JSON.parse(readFileSync(resolve(ROOT, "run", "fixtures", "bsky_author_feed.json"), "utf8"));
const OUT = resolve(ROOT, "out");
mkdirSync(OUT, { recursive: true });
const port = 8700 + Math.floor(Math.random() * 200);
const server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: REPO, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
const url = `http://127.0.0.1:${port}/${ROOT.split("/").pop()}/web/index.html`;
const res = { checks: {}, errors: [], offsite: [], appview: [] };
const check = (name, ok, detail) => { res.checks[name] = { ok: !!ok, detail }; console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail !== undefined ? " " + JSON.stringify(detail) : ""}`); };

async function newPage(browser, opts) {
  const ctx = await browser.newContext(opts);
  const page = await ctx.newPage();
  page.on("console", (m) => { if (m.type() === "error") res.errors.push("console: " + m.text()); });
  page.on("pageerror", (e) => res.errors.push("pageerror: " + String(e)));
  await page.route("**/*", (route) => {
    const u = new URL(route.request().url());
    if (u.hostname === "127.0.0.1") return route.continue();
    if (u.hostname === "cdn.jsdelivr.net") return route.fulfill({ status: 200, contentType: "text/javascript", body: readFileSync(tokLib) });
    if (u.hostname === "public.api.bsky.app") {
      res.appview.push(u.pathname.split("/").pop() + u.search);
      const m = u.pathname.split("/").pop(), q = u.searchParams;
      let body;
      if (m === "com.atproto.identity.resolveHandle" && q.get("handle") === fx.handle) body = { did: fx.did };
      else if (m === "app.bsky.feed.getAuthorFeed" && q.get("actor") === fx.did) body = fx.pages[q.get("cursor") ?? ""];
      if (!body) return route.fulfill({ status: 400, contentType: "application/json", body: JSON.stringify({ error: "InvalidRequest", message: "not in fixture" }) });
      return route.fulfill({ status: 200, contentType: "application/json", headers: { "access-control-allow-origin": "*" }, body: JSON.stringify(body) });
    }
    res.offsite.push(u.href);
    return route.abort();
  });
  return { ctx, page };
}

let browser;
try {
  browser = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--disable-vulkan-surface"] });

  // 1. no WebGPU
  {
    const { ctx, page } = await newPage(browser, { viewport: { width: 900, height: 700 } });
    await page.addInitScript(() => { delete Navigator.prototype.gpu; });
    const n0 = res.errors.length;
    await page.goto(url);
    await page.waitForFunction("window.app && window.app.state === 'nogpu'", null, { timeout: 30_000 });
    const s = await page.evaluate(() => ({ cls: document.getElementById("status").className, text: document.getElementById("status-text").textContent,
      run: document.getElementById("bsky-run").disabled, chips: document.querySelectorAll(".chip").length }));
    check("no WebGPU: warning shown, controls render, no errors", s.cls.includes("warn") && /WebGPU/.test(s.text) && s.run && s.chips === 5 && res.errors.length === n0, s);
    await ctx.close();
  }

  // 2. text mode
  const { page } = await newPage(browser, { viewport: { width: 1100, height: 900 }, deviceScaleFactor: 2 });
  const t0 = Date.now();
  await page.goto(url);
  const sawProgress = await page.waitForFunction(() => /Downloading the model: [\d.]+ MB of [\d.]+ MB/.test(document.getElementById("status-text").textContent), null, { timeout: 60_000 }).then(() => true, () => false);
  check("download progress in MB shown", sawProgress);
  await page.waitForFunction("window.app && (window.app.state === 'ready' || window.app.state === 'error')", null, { timeout: 900_000 });
  const st = await page.evaluate(() => ({ state: app.state, error: app.error, load: app.load }));
  check("model loaded", st.state === "ready", { ...st, wallS: (Date.now() - t0) / 1000 });
  if (st.state !== "ready") throw new Error("model did not load: " + st.error);
  await page.waitForFunction("window.app.last && document.querySelector('#out mark.ent')", null, { timeout: 300_000 });
  const marks = await page.$$eval("#out mark.ent", (ms) => ms.map((m) => ({ text: m.firstChild.textContent, label: m.dataset.label })));
  res.textEntities = marks;
  check("default example: highlighted entities", marks.length >= 3, marks);
  check("Satya Nadella highlighted as person", marks.some((m) => m.text === "Satya Nadella" && m.label === "person"));
  check("Microsoft highlighted as company", marks.some((m) => /^Microsoft/.test(m.text) && m.label === "company"));
  const facts = await page.textContent("#facts");
  check("facts line: size, load time, stage times", /MB downloaded, ready in [\d.]+ s/.test(facts) && /tokenise/.test(facts) && /GPU/.test(facts), facts);
  // threshold and label edits re-run: drop "person", Satya Nadella must lose its highlight
  await page.click('.chip button[aria-label="remove person"]');
  await page.waitForFunction(() => app.last && !app.last.ents.some((e) => e.label === "person"), null, { timeout: 120_000 });
  check("removing a label re-runs", !(await page.$$eval("#out mark.ent", (ms) => ms.some((m) => m.dataset.label === "person"))));
  await page.click('.preset[data-reset]');
  await page.waitForFunction(() => app.last && app.last.ents.some((e) => e.label === "person"), null, { timeout: 120_000 });
  res.textTimings = await page.evaluate(() => app.last.timings);
  await page.screenshot({ path: resolve(OUT, "page_text.png"), fullPage: true });

  // 3. Bluesky mode
  await page.click("#tab-bsky");
  await page.fill("#bsky-input", fx.handle);
  await page.selectOption("#bsky-count", "200");
  await page.click("#bsky-run");
  await page.waitForFunction("window.app.bsky && !window.app.bsky.running && window.app.bsky.done", null, { timeout: 600_000 });
  const b = await page.evaluate(() => ({ fetched: app.bsky.fetched, analysed: app.bsky.analysed, error: app.bsky.error && String(app.bsky.error),
    reposts: app.bsky.posts.filter((p) => p.repost).length, view: app.bsky.view.map((g) => ({ label: g.label, entities: g.entities.map((e) => [e.name, e.count, e.key]) })),
    progress: document.getElementById("bsky-progress").textContent }));
  res.bsky = b;
  check("AppView: resolveHandle then both feed pages", res.appview.some((x) => x.startsWith("com.atproto.identity.resolveHandle")) && res.appview.filter((x) => x.startsWith("app.bsky.feed.getAuthorFeed")).length === 2, res.appview);
  check("30 posts fetched and analysed, reposts skipped", !b.error && b.fetched === 30 && b.analysed === 30 && b.reposts === 0, { fetched: b.fetched, analysed: b.analysed, error: b.error });
  check("progress line: posts per second", /[\d.]+ posts\/s overall/.test(b.progress), b.progress);
  const company = b.view.find((g) => g.label === "company");
  const ms = company?.entities.find((e) => e[2] === "microsoft");
  // Microsoft appears as "Microsoft" (posts 0, 1), "Microsoft's" (2) and "@microsoft" (22)
  check("grouped view: company group with Microsoft merged across surface forms, >= 3 posts", ms && ms[0] === "Microsoft" && ms[1] >= 3, ms);
  check("grouped view: sorted by post count", b.view.every((g) => g.entities.every((e, i) => i === 0 || g.entities[i - 1][1] >= e[1])));
  const domGroups = await page.$$eval("#groups .group", (gs) => gs.map((g) => g.dataset.label));
  check("grouped view rendered per label", domGroups.length >= 3 && domGroups.includes("company") && domGroups.includes("location"), domGroups);
  await page.click('#groups .group[data-label="company"] button[data-key="microsoft"]');
  const detail = await page.evaluate(() => {
    const d = document.getElementById("bsky-detail");
    return { hidden: d.hidden, title: d.querySelector("h3 span")?.textContent, posts: d.querySelectorAll("li").length,
      marks: [...d.querySelectorAll("mark.ent")].map((m) => m.textContent), links: [...d.querySelectorAll("a")].map((a) => a.href) };
  });
  check("clicking an entity lists its posts, highlighted, with links", !detail.hidden && detail.posts === ms?.[1] && detail.marks.length >= detail.posts
    && detail.links.length === detail.posts && detail.links.every((h) => h.startsWith("https://bsky.app/profile/fixture.example.com/post/")), detail);
  await page.screenshot({ path: resolve(OUT, "page_bsky.png"), fullPage: true });

  // phone width, text mode, for a layout look
  await page.setViewportSize({ width: 390, height: 844 });
  await page.click("#tab-text");
  await page.waitForTimeout(800);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check("no horizontal scroll at 390 px", overflow <= 0, overflow);
  await page.screenshot({ path: resolve(OUT, "page_phone.png"), fullPage: true });
  await page.click("#tab-bsky");
  await page.waitForTimeout(300);
  await page.screenshot({ path: resolve(OUT, "page_phone_bsky.png"), fullPage: true });
} catch (e) {
  res.fatal = String(e.stack || e);
  console.log("FATAL " + res.fatal);
} finally {
  await browser?.close();
  server.kill();
}
check("no page errors", res.errors.length === 0, res.errors.slice(0, 5));
check("no requests off 127.0.0.1 except the stubs", res.offsite.length === 0, res.offsite);
res.pass = !res.fatal && Object.values(res.checks).every((c) => c.ok);
writeFileSync(resolve(OUT, "page_smoke.json"), JSON.stringify(res, null, 1));
console.log(res.pass ? "PASS" : "FAIL");
process.exit(res.pass ? 0 : 1);
