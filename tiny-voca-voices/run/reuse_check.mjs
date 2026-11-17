// The reuse path end to end: design a voice in the tuner, save it with the page's own button, then speak the
// same text with it (1) in the tuner, (2) through tinyvoca.js on example.html, via its file input and Speak
// button and via a direct library call. Writes the voice file and the audio for run/reuse_score.py.
//   NODE_PATH=$(npm root -g) node run/reuse_check.mjs OUT_DIR [BASE_URL]
import { createRequire } from "node:module";
import { join } from "node:path";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");

const out = process.argv[2];
const base = process.argv[3] || "http://localhost:8765/";
mkdirSync(out, { recursive: true });
const TEXT = "The meeting moved to three thirty on Thursday, so please bring the quarterly numbers and a working laptop.";
const SETTINGS = { mix: 0.2, a1: 0.3, a2: -0.4, a3: 0, a4: 0.5, pitch: -2.5, range: 0.9, speed: 0.95, formant: 0.95, tilt: -0.5, breath: 2 };
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } : undefined;
const browser = await chromium.launch({ proxy });
const context = await browser.newContext({ acceptDownloads: true });
const errors = [];
const watch = (page) => { page.on("pageerror", (e) => errors.push(String(e))); page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); }); };

function writeWav(path, samples) {
  const buf = Buffer.alloc(44 + samples.length * 2);
  buf.write("RIFF", 0); buf.writeUInt32LE(36 + samples.length * 2, 4); buf.write("WAVEfmt ", 8);
  buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22); buf.writeUInt32LE(24000, 24);
  buf.writeUInt32LE(48000, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write("data", 36);
  buf.writeUInt32LE(samples.length * 2, 40);
  samples.forEach((x, i) => buf.writeInt16LE(Math.round(Math.max(-1, Math.min(1, x)) * 32767), 44 + 2 * i));
  writeFileSync(path, buf);
}

// 1. Tuner: set the voice, speak, save the voice file with the button.
const tuner = await context.newPage(); watch(tuner);
await tuner.goto(base);
await tuner.waitForFunction(() => ready === true, null, { timeout: 600000 });
await tuner.fill("#voice-name", "reuse check");
const tunerAudio = await tuner.evaluate(async ([settings, text]) => {
  apply(settings);
  document.getElementById("text").value = text;
  const before = request;
  speak();
  while (!(lastClip && lastClip.id === before + 1)) await new Promise((r) => setTimeout(r, 50));
  return Array.from(lastClip.audio);
}, [SETTINGS, TEXT]);
writeWav(join(out, "tuner-1.wav"), Float32Array.from(tunerAudio));
const [download] = await Promise.all([tuner.waitForEvent("download"), tuner.click("#export")]);
const voicePath = join(out, download.suggestedFilename());
await download.saveAs(voicePath);
const voice = JSON.parse(readFileSync(voicePath, "utf8"));

// 1b. Load the file back into a fresh tuner: the sliders must come back.
const again = await context.newPage(); watch(again);
await again.goto(base);
await again.waitForFunction(() => ready === true, null, { timeout: 600000 });
await again.setInputFiles("#import", voicePath);
// Loading speaks at once, which replaces the "Loaded" status; the restored name marks the load instead.
await again.waitForFunction(() => document.getElementById("voice-name").value === "reuse check" ||
  document.getElementById("status").classList.contains("error"));
const reloaded = await again.evaluate(() => values());

// 2. example.html: the file input and Speak button, then a direct library call for the samples.
const example = await context.newPage(); watch(example);
await example.goto(new URL("example.html", base).href);
await example.waitForFunction(() => document.getElementById("status").textContent === "Ready.", null, { timeout: 600000 });
await example.setInputFiles("#file", voicePath);
await example.waitForFunction(() => document.getElementById("name").textContent === "reuse check");
await example.fill("#text", TEXT);
await example.click("#speak");
await example.waitForFunction(() => / s$/.test(document.getElementById("status").textContent), null, { timeout: 60000 });
const exampleStatus = await example.textContent("#status");
const libAudio = await example.evaluate(async ([voice, text]) => {
  const tts = await TinyVoca.load();
  return Array.from(await tts.speak(text, voice));
}, [voice, TEXT]);
writeWav(join(out, "library-1.wav"), Float32Array.from(libAudio));

const report = {
  voice_file: download.suggestedFilename(), voice_bytes: readFileSync(voicePath).length,
  voice_keys: Object.keys(voice), settings_roundtrip: Object.keys(SETTINGS).every((k) => Math.abs(reloaded[k] - SETTINGS[k]) < 1e-9),
  example_status: exampleStatus, tuner_samples: tunerAudio.length, library_samples: libAudio.length, errors,
};
writeFileSync(join(out, "reuse.json"), JSON.stringify(report, null, 1));
console.log(JSON.stringify(report));
await browser.close();
