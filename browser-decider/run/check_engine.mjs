// Run web/check.html in headless Chromium (SwiftShader WebGPU) against a packed model.
//   node run/check_engine.mjs PKG [TOKENIZERS_MJS]     (PKG = directory under out/web/)
// Serves the experiments repo locally; the jsDelivr tokenizer import is answered from a local copy of
// @huggingface/tokenizers@0.2.0 (npm i it anywhere and pass dist/tokenizers.mjs), since the
// container's Chromium has no proxied egress. Passes when the engine's logits match run/pack.py's
// NumPy reference (max |diff| < 2e-3, argmax agreement on every question) and the JS tokenizer +
// layout reproduce the Python ids. SwiftShader timings say nothing about a real GPU.
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = resolve(ROOT, "..");
const [pkg, tokLib] = process.argv.slice(2);
const port = 8700 + Math.floor(Math.random() * 200);
const server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: REPO, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 1500));
const errors = [];
let res;
try {
  const b = await chromium.launch({ args: ["--no-sandbox", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-features=Vulkan", "--use-vulkan=swiftshader", "--disable-vulkan-surface"] });
  const page = await b.newPage();
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(m.type() + ": " + m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.route("https://cdn.jsdelivr.net/**", (route) => route.fulfill({ status: 200, contentType: "text/javascript", body: readFileSync(tokLib) }));
  await page.goto(`http://127.0.0.1:${port}/${ROOT.split("/").pop()}/web/check.html?pkg=${pkg}&reps=1`);
  await page.waitForFunction("window.check && window.check.done", null, { timeout: 3_600_000 });
  res = await page.evaluate("window.check");
  await b.close();
} finally { server.kill(); }
res.console_errors = errors; res.pkg = pkg;
mkdirSync(resolve(ROOT, "results"), { recursive: true });
writeFileSync(resolve(ROOT, "results", `check_engine_${pkg}.json`), JSON.stringify(res, null, 1));
const ok = !res.error && res.maxAbsLogitDiff < 2e-3 && res.argmaxAgree === res.nQuestions && res.tokenizerMatches === res.nQuestions;
const { sample, ...rest } = res;
console.log(JSON.stringify(rest, null, 1));
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
