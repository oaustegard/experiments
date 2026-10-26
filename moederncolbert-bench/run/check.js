// Run web/check.html in headless Chromium (SwiftShader WebGPU): engine.js vs run/mcb.py, per token.
// Usage: node run/check.js [1.5M 27M]   (needs .work/models and run/fixtures.py's output)
// Passes when every token vector, one sequence per call and packed, is at cosine >= 0.9999.
// SwiftShader is for correctness only; its timings mean nothing about a real GPU.
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const path = require("path"), fs = require("fs");
const ROOT = path.resolve(__dirname, ".."), REPO = path.dirname(ROOT);
const models = process.argv.slice(2).length ? process.argv.slice(2) : ["1.5M", "27M"];
const K = process.env.K || "tiled";
const port = 8000 + Math.floor(Math.random() * 1000);
(async () => {
  const srv = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: REPO, stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1500));
  const b = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader",
    "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
  const out = {}; let ok = true;
  try {
    for (const m of models) {
      const page = await b.newPage();
      const errors = [];
      page.on("console", (x) => x.type() === "error" && errors.push(x.text()));
      await page.goto(`http://127.0.0.1:${port}/${path.basename(ROOT)}/web/check.html?m=${m}&k=${K}`);
      await page.waitForFunction("['done', 'error'].includes(window.result.phase)", null, { timeout: 3600000 });
      const r = await page.evaluate("window.result");
      r.console_errors = errors;
      out[m] = r;
      const good = r.phase === "done" && r.cos_single_min >= 0.9999 && r.cos_batch_min >= 0.9999;
      ok = ok && good;
      console.log(m, JSON.stringify(r), good ? "PASS" : "FAIL");
    }
  } finally { await b.close(); srv.kill(); }
  fs.writeFileSync(path.join(ROOT, "results", `check_swiftshader_${K}.json`), JSON.stringify(out, null, 1));
  process.exit(ok ? 0 : 1);
})();
