// Sing the default score (Amazing Grace, public domain) in several voices on sing.html and save the audio.
//   NODE_PATH=$(npm root -g) node run/sing_check.mjs OUT_DIR [BASE_URL]
import { createRequire } from "node:module";
import { join } from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const out = process.argv[2];
const base = process.argv[3] || "http://localhost:8765/";
mkdirSync(out, { recursive: true });
const CASES = [["F2", 0], ["F2-adult", 0], ["M2-baritone", -12], ["M2", -12]];
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } : undefined;
const browser = await chromium.launch({ proxy });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.goto(new URL("sing.html", base).href);
await page.waitForFunction(() => window.ready === true, null, { timeout: 600000 });
const report = { cases: {}, errors };
for (const [voice, transpose] of CASES) {
  const r = await page.evaluate(async ([voice, transpose]) => {
    document.getElementById("voice").value = `voices/${voice}.voice.json`;
    document.getElementById("transpose").value = transpose;
    document.getElementById("transpose").dispatchEvent(new Event("input"));
    window.lastSong = null;
    await window.singNow();
    if (window.lastSong.error) return { error: window.lastSong.error };
    return { status: document.getElementById("status").textContent, audio: Array.from(window.lastSong.audio) };
  }, [voice, transpose]);
  if (r.error) { report.cases[voice] = r; continue; }
  const s = Float32Array.from(r.audio);
  const buf = Buffer.alloc(44 + s.length * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + s.length * 2, 4); buf.write("WAVEfmt ", 8); buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(24000, 24); buf.writeUInt32LE(48000, 28);
  buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write("data", 36); buf.writeUInt32LE(s.length * 2, 40);
  s.forEach((x, i) => buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, x)) * 32767), 44 + 2 * i));
  writeFileSync(join(out, `${voice}.wav`), buf);
  report.cases[voice] = { transpose, status: r.status, seconds: s.length / 24000 };
  console.log(voice, r.status);
}
report.score = await page.evaluate(() => ({ lyrics: document.getElementById("lyrics").value, notes: document.getElementById("notes").value,
  bpm: Number(document.getElementById("bpm").value) }));
writeFileSync(join(out, "sing.json"), JSON.stringify(report, null, 1));
console.log(JSON.stringify({ errors }));
await browser.close();
