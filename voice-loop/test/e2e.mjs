// node voice-loop/test/e2e.mjs — loads the page from a local server in headless Chromium
// (SwiftShader WebGPU), loads the model, types a question, and waits for the answer and for
// Supertonic to synthesize at least one sentence. Prints the answer, timings and peak PSS.
import { createRequire } from "module";
import { execSync, spawn } from "child_process";
import { fileURLToPath } from "url";
import path from "path";
const { chromium } = createRequire(import.meta.url)("playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const port = 8791;
const srv = spawn("python3", ["-m", "http.server", String(port)], { cwd: root, stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const pss = () => Number(execSync("for p in $(pgrep -f '[c]hrom|[h]eadless_shell'); do awk '/^Pss:/ {print $2}' /proc/$p/smaps_rollup 2>/dev/null; done | awk '{s+=$1} END {print s+0}'").toString()) / 1024;
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } : undefined;
const b = await chromium.launch({ proxy, args: ["--ignore-certificate-errors", "--enable-unsafe-webgpu", "--use-angle=swiftshader", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan", "--autoplay-policy=no-user-gesture-required"] });
const p = await b.newPage();
const errs = [], failures = [];
p.on("console", (m) => { if (m.type() === "error") errs.push(m.text()); });
p.on("pageerror", (e) => errs.push("pageerror " + e.message));
let peak = 0; const iv = setInterval(() => { peak = Math.max(peak, pss()); }, 150);
try {
  await p.goto(`http://localhost:${port}/voice-loop/web/`);
  await p.waitForTimeout(1000);
  console.log("chips:", await p.$$eval(".chip", (cs) => cs.map((c) => c.textContent).join(" | ")));
  await p.click("#load");
  await p.waitForFunction(() => /Ready|Failed/.test(document.getElementById("loadmeta").textContent), null, { timeout: 900000 });
  console.log("load:", await p.textContent("#loadmeta"), "| PSS peak so far", peak.toFixed(0), "MB");
  await p.fill("#typed", "Why is the sky blue?");
  await p.press("#typed", "Enter");
  await p.waitForFunction(() => { const b = [...document.querySelectorAll(".msg.bot")].pop(); return b && b.querySelector(".t"); }, null, { timeout: 1800000 });
  const answer = await p.$eval(".msg.bot:last-of-type", (e) => e.textContent);
  console.log("answer:", answer);
  const pt = Number((answer.match(/(\d+) prompt tokens/) || [])[1] || 0);
  if (pt < 10) failures.push(`prompt had ${pt} tokens`);
  if (!/scatter|light|wavelength|atmosphere/i.test(answer)) failures.push("answer does not mention light or scattering");
  await p.waitForFunction(() => /Last sentence|error/i.test(document.getElementById("ttsmeta").textContent), null, { timeout: 900000 });
  console.log("tts:", await p.textContent("#ttsmeta"));
} catch (e) { failures.push(e.message); }
clearInterval(iv);
console.log(`PSS peak ${peak.toFixed(0)} MB, now ${pss().toFixed(0)} MB`);
console.log("errors:", errs.slice(0, 6));
await b.close(); srv.kill();
console.log(failures.length ? "FAIL: " + failures.join("; ") : "PASS");
process.exit(failures.length || errs.length ? 1 : 0);
