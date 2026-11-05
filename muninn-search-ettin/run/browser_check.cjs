// The site's in-browser search (components/local-search.js) against the live Worker, in headless Chromium on
// SwiftShader WebGPU, then the search page's swap from server to browser. Timings here are an emulated GPU.
//   node run/browser_check.cjs <site checkout>      -> results/browser.json
const { chromium } = require("playwright");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const SITE = process.argv[2], PORT = 8765;
const OUT = path.join(__dirname, "..", "results", "browser.json");
const CHROME = fs.readdirSync("/opt/pw-browsers").filter((d) => d.startsWith("chromium-")).sort().reverse()
  .map((d) => ["chrome-linux", "chrome-linux64"].map((s) => `/opt/pw-browsers/${d}/${s}/chrome`)).flat().find((p) => fs.existsSync(p));
const ARGS = ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"];
const QS = ["a decision model in a browser tab", "centered SimHash", "teaching a raven to sleep", "launching subagents from a script",
  "Can a chatbot spawn simultaneous helper workers without credentials?", "WebGPU kernel for attention"];
(async () => {
  const srv = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: SITE, stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1000));
  const b = await chromium.launch({ executablePath: CHROME, args: ARGS });
  try {
    const page = await b.newPage(); const errs = [];
    page.on("console", (m) => ["error", "warning"].includes(m.type()) && errs.push(m.text()));
    page.on("pageerror", (e) => errs.push(String(e)));
    await page.goto(`http://127.0.0.1:${PORT}/search.html?q=centered%20SimHash`);
    await page.waitForFunction(() => document.querySelector(".muninn-search-status")?.textContent.includes("ms"), null, { timeout: 60000 });
    const first = await page.textContent(".muninn-search-status");
    console.log("first search status:", first);
    const r = await page.evaluate(async (qs) => {
      const W = "https://muninn-search.austegard.workers.dev";
      const { loadLocalSearch } = await import("/components/local-search.js");
      const local = await loadLocalSearch(W);
      const out = { info: local.info, rows: [] };
      for (const q of qs) {
        const t = performance.now(); const lh = await local.search(q, { k: 8 }); const ms = performance.now() - t;
        const wh = (await (await fetch(W + "/api/search", { method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ query: q, k: 8 }) })).json()).hits;
        out.rows.push({ q, ms, same: JSON.stringify(lh.map((h) => h.chunk_id)) === JSON.stringify(wh.map((h) => h.chunk_id)),
          verified: lh.every((h) => h.verified), local: lh.slice(0, 3).map((h) => h.chunk_id), worker: wh.slice(0, 3).map((h) => h.chunk_id) });
      }
      local.destroy();
      return out;
    }, QS);
    console.log("info:", JSON.stringify(r.info));
    for (const row of r.rows) {
      console.log(row.same ? "SAME" : "DIFF", `${row.ms.toFixed(0)} ms`, row.verified ? "verified" : "UNVERIFIED", row.q.slice(0, 50));
      if (!row.same) console.log("   local ", row.local, "\n   worker", row.worker);
    }
    let st = "";
    await page.fill("muninn-search input", "persistent memory for an AI");
    for (let i = 0; i < 60; i++) {
      await page.click("muninn-search button[type=submit]");
      await page.waitForFunction(() => /ms/.test(document.querySelector(".muninn-search-status")?.textContent || "") && !document.querySelector("muninn-search [data-busy]"), null, { timeout: 60000 });
      st = await page.textContent(".muninn-search-status");
      if (st.includes("browser")) break;
      await new Promise((r) => setTimeout(r, 2000));
    }
    console.log("swapped status:", st);
    console.log("console:", errs.slice(0, 6));
    fs.writeFileSync(OUT, JSON.stringify({ gpu: "SwiftShader (emulated)", queries: r.rows.length,
      same_as_worker: r.rows.filter((x) => x.same).length, all_verified: r.rows.every((x) => x.verified),
      download_bytes: r.info.bytes, download_mb: +(r.info.bytes / 1e6).toFixed(1), ready_ms: Math.round(r.info.readyMs), first_status: first, swapped_status: st,
      console_errors: errs.length, rows: r.rows }, null, 1));
  } finally { await b.close(); srv.kill(); }
})().catch((e) => { console.error(e); process.exit(1); });
