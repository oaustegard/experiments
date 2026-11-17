// Load the tuner page in headless Chromium, speak each preset and a set of extreme settings, and save the audio
// for scoring (run/score_browser.py).
//   python3 -m http.server 8765 -d web &   then
//   NODE_PATH=$(npm root -g) node run/browser_check.mjs OUT_DIR [URL]
import { createRequire } from "node:module";
import { join } from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");

const out = process.argv[2];
const url = process.argv[3] || "http://localhost:8765/";
mkdirSync(out, { recursive: true });
const TEXT = "The meeting moved to three thirty on Thursday, so please bring the quarterly numbers and a working laptop.";
const EXTREMES = {
  "mix-0.5": { mix: -0.5 }, "mix1.5": { mix: 1.5 },
  "a1+1.5": { a1: 1.5 }, "a1-1.5": { a1: -1.5 }, "a2+1.5": { a2: 1.5 }, "a2-1.5": { a2: -1.5 },
  "a3+1.5": { a3: 1.5 }, "a3-1.5": { a3: -1.5 }, "a4+1.5": { a4: 1.5 }, "a4-1.5": { a4: -1.5 },
  "formant0.8": { formant: 0.8 }, "formant1.25": { formant: 1.25 },
  "pitch-12": { pitch: -12 }, "pitch+12": { pitch: 12 }, "speed2": { speed: 2 }, "speed0.5": { speed: 0.5 },
  "tilt-4": { tilt: -4 }, "tilt+4": { tilt: 4 }, "breath+12": { breath: 12 },
};
// Round 2 (CASES=axes): the axes at +-1 and +-0.75 in both directions, on F2 and M2, and 12 "Surprise me" voices.
if (process.env.CASES === "axes") {
  for (const k of Object.keys(EXTREMES)) delete EXTREMES[k];
  for (const a of ["a1", "a2", "a3", "a4"]) for (const v of [1, -1, 0.75, -0.75]) {
    EXTREMES[`${a}${v > 0 ? "+" : ""}${v}`] = { [a]: v };
    EXTREMES[`M2-${a}${v > 0 ? "+" : ""}${v}`] = { mix: 1, [a]: v };
  }
}
// Round 3 (CASES=random): 16 "Surprise me" voices from the bounded sampler, nothing else.
if (process.env.CASES === "random") for (const k of Object.keys(EXTREMES)) delete EXTREMES[k];
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } : undefined;
const browser = await chromium.launch({ proxy });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
const t0 = Date.now();
await page.goto(url);
await page.waitForFunction(() => ready === true || document.getElementById("status").classList.contains("error"), null, { timeout: 600000 });
const loadSeconds = (Date.now() - t0) / 1000;
const cases = await page.evaluate(() => Object.keys(PRESETS));
const results = {};
async function run(name, settings) {
  const info = await page.evaluate(async ([settings, text]) => {
    apply(settings);
    document.getElementById("text").value = text;
    const before = request;
    speak();
    while (!(lastClip && lastClip.id === before + 1)) {
      await new Promise((r) => setTimeout(r, 50));
      if (document.getElementById("status").classList.contains("error")) return { error: document.getElementById("status").textContent };
    }
    return { status: document.getElementById("status").textContent, audio: Array.from(lastClip.audio) };
  }, [settings, TEXT]);
  if (info.error) { results[name] = { error: info.error }; return; }
  const samples = Float32Array.from(info.audio);
  const buf = Buffer.alloc(44 + samples.length * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + samples.length * 2, 4); buf.write("WAVEfmt ", 8);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(24000, 24);
  buf.writeUInt32LE(48000, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write("data", 36);
  buf.writeUInt32LE(samples.length * 2, 40);
  samples.forEach((x, i) => buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, x)) * 32767), 44 + 2 * i));
  writeFileSync(join(out, `${name}-1.wav`), buf);
  results[name] = { status: info.status, seconds: samples.length / 24000 };
  console.log(name, info.status);
}
if (!process.env.CASES) for (const name of cases) await run(name, await page.evaluate((n) => PRESETS[n], name));
if (process.env.CASES) for (let i = 0; i < (process.env.CASES === "random" ? 16 : 12); i++) {
  const settings = await page.evaluate(() => { document.getElementById("random").click(); return values(); });
  await page.evaluate(() => new Promise((r) => setTimeout(r, 200)));
  await run(`random${i}`, settings);
  results[`random${i}`].settings = settings;
}
for (const [name, s] of Object.entries(EXTREMES)) await run(name, s);
writeFileSync(join(out, "browser.json"), JSON.stringify({ url, load_seconds: loadSeconds, errors, results }, null, 1));
console.log(JSON.stringify({ load_seconds: loadSeconds, errors }));
await browser.close();
