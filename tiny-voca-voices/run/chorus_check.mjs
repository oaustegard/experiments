// Sing the default score (Amazing Grace, public domain) as a chorus in the tuner and save the stereo mix and a
// mono fold-down for scoring with run/evaluate.py-style tools.
//   NODE_PATH=$(npm root -g) node run/chorus_check.mjs OUT_DIR ENGINE [BASE_URL]
import { createRequire } from "node:module";
import { join } from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
const pw = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const [out, engine, base = "http://localhost:8765/"] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const browser = await pw[engine].launch({ proxy: { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } });
const page = await browser.newPage(engine === "webkit" ? pw.devices["iPhone 13"] : {});
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(base + "?mode=sing&seed=12345");
await page.waitForFunction(() => ready === true, null, { timeout: 120000 });
function wavFile(channels) {
  const n = channels[0].length, k = channels.length, buf = Buffer.alloc(44 + n * 2 * k);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + n * 2 * k, 4); buf.write("WAVEfmt ", 8); buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); buf.writeUInt16LE(k, 22); buf.writeUInt32LE(24000, 24); buf.writeUInt32LE(48000 * k, 28);
  buf.writeUInt16LE(2 * k, 32); buf.writeUInt16LE(16, 34); buf.write("data", 36); buf.writeUInt32LE(n * 2 * k, 40);
  for (let i = 0, o = 44; i < n; i++) for (let c = 0; c < k; c++, o += 2) buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, channels[c][i])) * 32767), o);
  return buf;
}
const report = { engine, errors, runs: {} };
for (const count of [4, 8]) {
  const r = await page.evaluate(async (count) => {
    const c = SONG.find((x) => x.id === "voices");
    c.input.value = count; c.show();
    const parts = chorusParts(count).map((p) => ({ mix: p.settings.mix, transpose: +p.transpose.toFixed(3), delayMs: Math.round(p.delayMs), pan: +p.pan.toFixed(2) }));
    const before = request;
    perform();
    while (!(lastClip && lastClip.id === before + 1)) {
      await new Promise((r) => setTimeout(r, 50));
      if (document.getElementById("status").classList.contains("error")) return { error: document.getElementById("status").textContent };
    }
    return { status: document.getElementById("status").textContent, parts, left: Array.from(lastClip.left), right: Array.from(lastClip.right) };
  }, count);
  if (r.error) { report.runs[count] = r; console.log(engine, count, r.error); continue; }
  const left = Float32Array.from(r.left), right = Float32Array.from(r.right);
  const mono = left.map((x, i) => (x + right[i]) / 2);
  writeFileSync(join(out, `chorus${count}-stereo.wav`), wavFile([left, right]));
  writeFileSync(join(out, `chorus${count}-1.wav`), wavFile([mono]));
  let diff = 0; for (let i = 0; i < left.length; i++) diff = Math.max(diff, Math.abs(left[i] - right[i]));
  report.runs[count] = { status: r.status, parts: r.parts, seconds: left.length / 24000, maxLeftRightDiff: +diff.toFixed(3) };
  console.log(engine, count, r.status, "| L/R differ by up to", diff.toFixed(3), "\n   parts", JSON.stringify(r.parts));
}
writeFileSync(join(out, "chorus.json"), JSON.stringify(report, null, 1));
console.log(JSON.stringify({ errors }));
await browser.close();
