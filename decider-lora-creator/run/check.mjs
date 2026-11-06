// Serve the experiments repo root and run decider-lora-creator's pages in headless Chromium on SwiftShader.
//   node run/check.mjs check [query]     -> web/check.html, prints window.__result, writes results/check_page.json
//   node run/check.mjs flow              -> index.html?demo: a short head-only run through the UI, screenshot to out/
import { createRequire } from "node:module";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const { chromium } = createRequire(import.meta.url)("playwright");
const exp = path.resolve(path.dirname(new URL(import.meta.url).pathname), ".."), root = path.resolve(exp, "..");
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".bin": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const f = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!f.startsWith(root)) { res.writeHead(403); res.end(); return; }
  fs.readFile(f, (err, data) => { if (err) { res.writeHead(404); res.end(); return; } res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" }); res.end(data); });
}).listen(0);
const port = server.address().port, which = process.argv[2] || "check";
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1000, height: 900 }, deviceScaleFactor: which === "flow" ? 2 : 1 });
page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
page.on("pageerror", (e) => console.log("pageerror:", e.message));
const base = `http://localhost:${port}/decider-lora-creator/web/`;
fs.mkdirSync(path.join(exp, "out"), { recursive: true });
if (which === "check") {
  await page.goto(base + "check.html" + (process.argv[3] ? "?" + process.argv[3] : ""));
  await page.waitForFunction(() => window.__result, null, { timeout: 1800000 });
  const r = await page.evaluate(() => window.__result);
  console.log(JSON.stringify(r, null, 1));
  fs.writeFileSync(path.join(exp, "results", "check_page.json"), JSON.stringify(r, null, 1));
  process.exitCode = r.ok ? 0 : 1;
} else if (which === "import") {
  // the dataset panel: the prompt fills in (with the drafted decision), and the prompt's own example, pasted inside
  // a code fence as agents return it, imports as a decision with its cases and a start spike
  page.on("dialog", (d) => d.accept());
  await page.goto(base + "index.html?demo");
  await page.waitForFunction(() => window.__ready, null, { timeout: 120000 });
  await page.click("#prep summary");
  await page.waitForFunction(() => document.getElementById("promptText").value.length > 0, null, { timeout: 10000 });
  const prompt = await page.inputValue("#promptText");
  const a = prompt.indexOf("nothing else:\n") + "nothing else:\n".length, b = prompt.indexOf("\nLeave \"split\"");
  await page.fill("#pasteData", "```json\n" + prompt.slice(a, b).replace(/"severity": "1"/, '"severity": 1').replace(/"urgent": "true"/, '"urgent": true') + "\n```");
  await page.click("#pasteImport");
  const st = await page.evaluate(() => ({ id: window.__creator.state.decision.id, questions: Object.keys(window.__creator.state.decision.questions),
    cases: window.__creator.state.cases.length, golds: window.__creator.state.cases.map((c) => c.gold), init: window.__creator.state.init.length,
    err: document.getElementById("err").textContent, csum: document.getElementById("csum").textContent }));
  const out = { promptChars: prompt.length, promptHasDemoDecision: prompt.includes('"discrepancy_severity"'), imported: st,
    ok: prompt.includes('"discrepancy_severity"') && st.id === "snake_case_name" && st.cases === 2 && !st.err && st.init === 3 && st.golds[0].severity === "1" && st.golds[1].urgent === "true" };
  console.log(JSON.stringify(out, null, 1));
  fs.writeFileSync(path.join(exp, "results", "check_import_page.json"), JSON.stringify(out, null, 1));
  process.exitCode = out.ok ? 0 : 1;
} else if (which === "resume") {
  // Stop / Resume and reload / resume through the page's own buttons and IndexedDB, on the fake backend (no GPU
  // training): each interrupted run must end exactly where the uninterrupted one does.
  const url = base + "index.html?demo&backend=fake&fakeDelay=40";
  const setup = async (p) => {
    p.on("dialog", (d) => d.accept());
    await p.goto(url);
    await p.waitForFunction(() => window.__ready && window.__creator?.state.cases.length > 0, null, { timeout: 120000 });
    for (const [id, v] of [["maxtr", "24"], ["maxte", "6"], ["epochs", "4"], ["bs", "8"], ["valp", "20"]]) await p.fill("#" + id, v);
  };
  const stepOf = (p) => p.evaluate(() => +((document.getElementById("tstat").textContent.match(/step (\d+)\//) || [0, 0])[1]));
  const result = (p) => p.evaluate(() => { const r = window.__lastRun; return r.error ? r : { test: r.result.test.accuracy, testKL: r.result.test.kl, best: r.result.bestEpoch, valKL: r.result.val?.kl, epochsRun: r.result.epochsRun }; });
  const fresh = async () => { const ctx = await browser.newContext(); return [ctx, await ctx.newPage()]; };
  const out = {};
  // A: uninterrupted
  let [cA, pA] = await fresh(); await setup(pA); await pA.click("#train");
  await pA.waitForFunction(() => window.__lastRun, null, { timeout: 600000 });
  out.uninterrupted = await result(pA); out.wakeLock = await pA.evaluate(() => ("wakeLock" in navigator));
  await cA.close();
  // B: Stop mid-epoch, then Resume in the same page
  let [cB, pB] = await fresh(); await setup(pB); await pB.click("#train");
  await pB.waitForFunction(() => +((document.getElementById("tstat").textContent.match(/step (\d+)\//) || [0, 0])[1]) >= 7, null, { timeout: 600000 });
  await pB.click("#stop");
  await pB.waitForFunction(() => window.__paused, null, { timeout: 60000 });
  out.pausedAt = await pB.evaluate(() => window.__paused);
  out.resumeVisible = await pB.isVisible("#resume");
  await pB.click("#resume");
  await pB.waitForFunction(() => window.__lastRun, null, { timeout: 600000 });
  out.resumedInPage = await result(pB);
  await cB.close();
  // C: Stop, reload the page (a crash or a closed tab), resume from IndexedDB
  let [cC, pC] = await fresh(); await setup(pC); await pC.click("#train");
  await pC.waitForFunction(() => +((document.getElementById("tstat").textContent.match(/step (\d+)\//) || [0, 0])[1]) >= 18, null, { timeout: 600000 });
  await pC.click("#stop");
  await pC.waitForFunction(() => window.__paused, null, { timeout: 60000 });
  out.pausedBeforeReload = await pC.evaluate(() => window.__paused);
  await pC.waitForTimeout(800);   // the checkpoint's IndexedDB write
  await pC.goto(url);
  await pC.waitForFunction(() => window.__ready, null, { timeout: 120000 });
  out.offerAfterReload = (await pC.textContent("#saved")).replace(/\s+/g, " ").trim().slice(0, 160);
  await pC.click("#savedGo");
  await pC.waitForFunction(() => window.__lastRun, null, { timeout: 600000 });
  out.resumedAfterReload = await result(pC);
  out.savedClearedAfterDone = await pC.evaluate(() => new Promise((res) => { const q = indexedDB.open("decider-lora-creator", 1); q.onsuccess = () => { const g = q.result.transaction("runs").objectStore("runs").get("current"); g.onsuccess = () => res(g.result === undefined); }; }));
  await pC.screenshot({ path: path.join(exp, "out", "resume.png"), fullPage: true });
  await cC.close();
  const same = (x) => JSON.stringify(x) === JSON.stringify(out.uninterrupted);
  out.ok = !out.uninterrupted.error && same(out.resumedInPage) && same(out.resumedAfterReload) && out.resumeVisible && out.savedClearedAfterDone;
  console.log(JSON.stringify(out, null, 1));
  fs.writeFileSync(path.join(exp, "results", "check_resume_page.json"), JSON.stringify(out, null, 1));
  process.exitCode = out.ok ? 0 : 1;
} else {
  await page.goto(base + "index.html?demo");
  await page.waitForFunction(() => window.__creator?.state.cases.length > 0, null, { timeout: 60000 });
  await page.fill("#maxtr", process.argv[3] || "16"); await page.fill("#maxte", process.argv[4] || "6"); await page.fill("#epochs", process.argv[5] || "3");
  await page.click("#train");
  await page.waitForFunction(() => window.__lastRun, null, { timeout: 1800000 });
  const r = await page.evaluate(() => window.__lastRun);
  await page.fill("#tstate", await page.evaluate(() => window.__creator.state.cases.find((c) => c.split === "test").state));
  await page.click("#try");
  await page.waitForFunction(() => document.querySelector("#ans .card") || document.getElementById("err").textContent, null, { timeout: 600000 });
  const err = await page.evaluate(() => document.getElementById("err").textContent);
  await page.screenshot({ path: path.join(exp, "out", "page.png"), fullPage: true });
  console.log(JSON.stringify({ err, run: r, answer: await page.evaluate(() => document.getElementById("ans").innerText.slice(0, 400)) }, null, 1));
  process.exitCode = err || r.error ? 1 : 0;
}
await browser.close(); server.close();
