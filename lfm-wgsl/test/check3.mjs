// node lfm-wgsl/test/check.mjs  — serves the repo root, runs test/check.html in headless Chromium
// on SwiftShader WebGPU, prints parity and peak PSS of all Chromium processes.
import { createRequire } from "module";
const { chromium } = createRequire(import.meta.url)("playwright");   // global install; require() honours NODE_PATH
import { execSync, spawn } from "child_process";
import { fileURLToPath } from "url";
import path from "path";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const port = 8793;
const srv = spawn("python3", ["-m", "http.server", String(port)], { cwd: root, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const pss = () => Number(execSync("for p in $(pgrep -f '[c]hrom|[h]eadless_shell'); do awk '/^Pss:/ {print $2}' /proc/$p/smaps_rollup 2>/dev/null; done | awk '{s+=$1} END {print s+0}'").toString()) / 1024;
const b = await chromium.launch({ args: ["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"] });
const p = await b.newPage();
p.on("console", (m) => process.stdout.write("  | " + m.text() + "\n"));
const base = pss(); let peak = 0;
const iv = setInterval(() => { peak = Math.max(peak, pss()); }, 100);
await p.goto(`http://localhost:${port}/lfm-wgsl/test/check3.html`);
await p.waitForFunction(() => window.result, null, { timeout: 1800000 });
clearInterval(iv);
const r = await p.evaluate(() => window.result);
console.log(`PSS: base ${base.toFixed(0)} MB, peak ${peak.toFixed(0)} MB, after ${pss().toFixed(0)} MB`);
await b.close(); srv.kill();
process.exit(r.error || r.cosMin < 0.999 ? 1 : 0);
